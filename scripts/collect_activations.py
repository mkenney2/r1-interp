#!/usr/bin/env python3
"""Phase 3 — Collect top activating examples for each transcoder feature.

Runs the model on a diverse corpus, hooks transcoders into MLP layers,
and records top-N activating examples per feature with context.

Output is saved as JSON per layer for downstream autointerp labeling.

Usage:
    python scripts/collect_activations.py \
        --checkpoint-base checkpoints \
        --layers 8 9 10 11 12 13 14 15 16 17 \
        --output-dir results/features/activations \
        [--n-examples 20] \
        [--corpus-tokens 5000000]
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import MODEL_ID
from r1_interp.utils import ensure_dir


# ---------------------------------------------------------------------------
# Transcoder (same as eval_sweep.py)
# ---------------------------------------------------------------------------
class Transcoder(torch.nn.Module):
    """Load a sparsify transcoder checkpoint and run forward pass."""

    def __init__(self, sae_dir: Path, device: str = "cuda"):
        super().__init__()
        cfg_path = sae_dir / "cfg.json"
        weights_path = sae_dir / "sae.safetensors"

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        state = load_file(str(weights_path), device=device)
        state = {k: v.bfloat16() for k, v in state.items()}

        self.W_enc = state.get("W_enc", state.get("encoder.weight"))
        self.b_enc = state.get("b_enc", state.get("encoder.bias"))
        self.W_dec = state["W_dec"]
        self.b_dec = state.get("b_dec", torch.zeros(self.W_dec.shape[-1], device=device))
        self.W_skip = state.get("W_skip", None)

        self.k = self.cfg.get("k", 64)
        self.n_features = self.W_enc.shape[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse activations."""
        pre_acts = x @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        if self.k > 0 and self.k < acts.shape[-1]:
            topk_vals, topk_idx = acts.topk(self.k, dim=-1)
            sparse_acts = torch.zeros_like(acts)
            sparse_acts.scatter_(-1, topk_idx, topk_vals)
            return sparse_acts
        return acts


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def find_layer_checkpoint(base_dir: Path, layer: int) -> Path | None:
    """Find the sae_dir for a specific layer."""
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        m = re.search(rf"_L{layer}$", entry.name)
        if m:
            sae_dir = entry / f"layers.{layer}"
            if (sae_dir / "sae.safetensors").exists():
                return sae_dir
    return None


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def load_corpus_tokens(
    tokenizer, n_tokens: int, device: str
) -> torch.Tensor:
    """Load corpus tokens from openwebtext via streaming."""
    from datasets import load_dataset

    print(f"Loading {n_tokens:,} corpus tokens from openwebtext...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    all_tokens: list[int] = []
    for example in ds:
        text = example.get("text", "")
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
        if len(all_tokens) >= n_tokens:
            break

    tokens = torch.tensor(all_tokens[:n_tokens], dtype=torch.long, device=device)
    print(f"  Got {len(tokens):,} tokens")
    return tokens


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------
def collect_layer_activations(
    model,
    tokenizer,
    transcoder: Transcoder,
    layer: int,
    corpus_tokens: torch.Tensor,
    n_examples: int = 20,
    batch_size: int = 8,
    seq_len: int = 128,
) -> dict:
    """Collect top-activating examples for every feature in one layer."""

    # Min-heap per feature: (activation_value, token_offset, context_snippet)
    top_examples: dict[int, list] = defaultdict(list)
    activation_count: dict[int, int] = defaultdict(int)
    max_activation: dict[int, float] = defaultdict(float)
    total_tokens_seen = 0

    n_seqs = len(corpus_tokens) // seq_len
    tokens_2d = corpus_tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)

    # Hook to capture activations (not replace output — just observe)
    captured_acts = {}

    def hook_fn(module, input, output):
        x = input[0]
        with torch.no_grad():
            sparse_acts = transcoder.encode(x)
            captured_acts["acts"] = sparse_acts
        # Return original output — don't modify model behavior
        return output

    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_hook(hook_fn)

    for i in range(0, n_seqs, batch_size):
        batch = tokens_2d[i : i + batch_size]
        if len(batch) == 0:
            break

        with torch.no_grad():
            model(batch)

        if "acts" not in captured_acts:
            continue

        layer_acts = captured_acts["acts"]  # (batch, seq_len, n_features)

        # Process each position
        for b_idx in range(layer_acts.shape[0]):
            for pos in range(layer_acts.shape[1]):
                active_mask = layer_acts[b_idx, pos] > 0
                active_indices = active_mask.nonzero(as_tuple=True)[0]

                for feat_idx_t in active_indices:
                    feat_idx = feat_idx_t.item()
                    val = layer_acts[b_idx, pos, feat_idx].item()

                    activation_count[feat_idx] += 1
                    max_activation[feat_idx] = max(
                        max_activation[feat_idx], val
                    )

                    # Context: decode surrounding tokens
                    global_offset = (i + b_idx) * seq_len + pos
                    ctx_start = max(0, pos - 10)
                    ctx_end = min(seq_len, pos + 15)
                    ctx_tokens = batch[b_idx][ctx_start:ctx_end].tolist()
                    context = tokenizer.decode(ctx_tokens)

                    entry = (val, global_offset, context)
                    if len(top_examples[feat_idx]) < n_examples:
                        heapq.heappush(top_examples[feat_idx], entry)
                    elif val > top_examples[feat_idx][0][0]:
                        heapq.heapreplace(top_examples[feat_idx], entry)

        total_tokens_seen += batch.numel()
        if total_tokens_seen % 500_000 < batch_size * seq_len:
            print(f"    {total_tokens_seen:,} tokens processed...")

    handle.remove()

    # Build output
    layer_data = {
        "layer": layer,
        "total_tokens": total_tokens_seen,
        "n_features": transcoder.n_features,
        "n_active_features": len(activation_count),
        "features": {},
    }

    for feat_idx in sorted(activation_count.keys()):
        freq = activation_count[feat_idx] / max(total_tokens_seen, 1)
        examples = sorted(top_examples[feat_idx], key=lambda x: -x[0])
        layer_data["features"][str(feat_idx)] = {
            "activation_frequency": freq,
            "max_activation": max_activation[feat_idx],
            "activation_count": activation_count[feat_idx],
            "top_examples": [
                {"activation": e[0], "token_offset": e[1], "context": e[2]}
                for e in examples
            ],
        }

    return layer_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect top activating examples per feature"
    )
    parser.add_argument("--checkpoint-base", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(8, 18)),
        help="Layers to collect activations for (default: 8-17)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/features/activations"),
    )
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--corpus-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.output_dir)

    # Load model
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load corpus
    corpus_tokens = load_corpus_tokens(tokenizer, args.corpus_tokens, device)

    # Process each layer
    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        sae_dir = find_layer_checkpoint(args.checkpoint_base, layer)
        if sae_dir is None:
            print(f"  No checkpoint found for layer {layer}, skipping")
            continue

        transcoder = Transcoder(sae_dir, device=device)
        print(f"  Loaded transcoder: {transcoder.n_features} features, k={transcoder.k}")

        layer_data = collect_layer_activations(
            model,
            tokenizer,
            transcoder,
            layer,
            corpus_tokens,
            n_examples=args.n_examples,
            batch_size=args.batch_size,
        )

        out_path = args.output_dir / f"activations_layer{layer}.json"
        with open(out_path, "w") as f:
            json.dump(layer_data, f, indent=2)

        n_active = layer_data["n_active_features"]
        n_total = layer_data["n_features"]
        print(f"  Active features: {n_active}/{n_total} ({n_active/n_total:.1%})")
        print(f"  Saved -> {out_path}")

        # Free GPU memory
        del transcoder
        torch.cuda.empty_cache()

    print("\nDone! Run autointerp.py next to label features.")


if __name__ == "__main__":
    main()
