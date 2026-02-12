#!/usr/bin/env python3
"""Phase 3 — Collect top activating examples for each transcoder feature.

Runs the model on a diverse corpus, hooks transcoders into MLP layers,
and records top-N activating examples per feature with context.

OPTIMIZED: hooks ALL requested layers simultaneously so the corpus is
processed in a single pass (10 layers → 10x fewer forward passes).
Uses vectorized counting and heap-min filtering to minimize Python overhead.

Output is saved as JSON per layer for downstream autointerp labeling.

Usage:
    python scripts/collect_activations.py \
        --checkpoint-base checkpoints \
        --layers 8 9 10 11 12 13 14 15 16 17 \
        --output-dir results/features/activations \
        [--n-examples 20] \
        [--corpus-tokens 5000000] \
        [--batch-size 32] \
        [--seq-len 256]
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
import time
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

    def encode_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to sparse topk activations. Returns (topk_vals, topk_idx).

        This avoids materializing the full (batch, seq, n_features) dense tensor,
        saving significant GPU memory.
        """
        pre_acts = x @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        topk_vals, topk_idx = acts.topk(self.k, dim=-1)
        return topk_vals, topk_idx


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
# Optimized multi-layer activation collection
# ---------------------------------------------------------------------------
def collect_all_layers(
    model,
    tokenizer,
    transcoders: dict[int, Transcoder],
    corpus_tokens: torch.Tensor,
    n_examples: int = 20,
    batch_size: int = 32,
    seq_len: int = 256,
) -> dict[int, dict]:
    """Collect top-activating examples for ALL layers in a single corpus pass.

    Key optimizations vs the per-layer version:
    - Hooks all layers simultaneously → 1 pass instead of N
    - Uses topk representation → no dense (batch, seq, 49152) tensor
    - Vectorized counting via torch.bincount
    - Heap-min filtering: only enters Python loop for features that can
      beat the current top-N threshold
    - Defers tokenizer.decode() to save time — stores raw token tuples
    """
    device = corpus_tokens.device

    # Per-layer tracking state
    layer_state = {}
    for layer, tc in transcoders.items():
        layer_state[layer] = {
            "activation_count": torch.zeros(tc.n_features, dtype=torch.long),
            "max_activation": torch.zeros(tc.n_features, dtype=torch.float32),
            # Track heap minimum per feature for fast filtering
            "heap_mins": torch.zeros(tc.n_features, dtype=torch.float32),
            "heap_full": torch.zeros(tc.n_features, dtype=torch.bool),
            "heaps": defaultdict(list),  # feat_idx -> min-heap of (val, offset, token_tuple)
        }

    n_seqs = len(corpus_tokens) // seq_len
    tokens_2d = corpus_tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)

    # Register hooks for ALL layers at once
    captured_acts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    for layer, tc in transcoders.items():
        def make_hook(l, t):
            def hook_fn(module, input, output):
                x = input[0]
                with torch.no_grad():
                    topk_vals, topk_idx = t.encode_topk(x)
                    captured_acts[l] = (topk_vals, topk_idx)
                return output  # Don't modify model behavior
            return hook_fn

        mlp = model.model.layers[layer].mlp
        handle = mlp.register_forward_hook(make_hook(layer, tc))
        handles.append(handle)

    total_tokens = 0
    t_start = time.time()

    for batch_i in range(0, n_seqs, batch_size):
        batch = tokens_2d[batch_i : batch_i + batch_size]
        if len(batch) == 0:
            break

        B = batch.shape[0]
        captured_acts.clear()

        with torch.no_grad():
            model(batch)

        # Process each layer's captured activations
        for layer, tc in transcoders.items():
            if layer not in captured_acts:
                continue

            topk_vals, topk_idx = captured_acts[layer]
            # topk_vals: (B, seq_len, k), topk_idx: (B, seq_len, k)
            st = layer_state[layer]

            # --- Vectorized frequency counting ---
            flat_idx = topk_idx.reshape(-1)       # (B * seq_len * k,)
            flat_vals = topk_vals.reshape(-1)      # (B * seq_len * k,)
            active_mask = flat_vals > 0
            active_idx = flat_idx[active_mask]

            if len(active_idx) == 0:
                continue

            # Count per feature
            counts = torch.bincount(active_idx, minlength=tc.n_features)
            st["activation_count"] += counts.cpu().long()

            # Max per feature via scatter_reduce
            batch_max = torch.zeros(tc.n_features, device=device, dtype=torch.float32)
            batch_max.scatter_reduce_(
                0, active_idx.long(),
                flat_vals[active_mask].float(),
                reduce="amax",
            )
            st["max_activation"] = torch.maximum(
                st["max_activation"], batch_max.cpu()
            )

            # --- Smart heap updates ---
            # Only process features where batch_max > current heap minimum
            # (or heap isn't full yet)
            heap_mins_dev = st["heap_mins"].to(device)
            needs_update = batch_max.to(device) > heap_mins_dev
            needs_update &= (batch_max.to(device) > 0)

            # Also update features whose heaps aren't full yet
            not_full_dev = ~st["heap_full"].to(device)
            has_activations = counts.to(device) > 0
            needs_update |= (not_full_dev & has_activations)

            update_features = needs_update.nonzero(as_tuple=True)[0]

            for f_idx_t in update_features:
                f_idx = f_idx_t.item()
                heap = st["heaps"][f_idx]
                min_val = st["heap_mins"][f_idx].item()

                # Find positions in this batch where feature f_idx is in the topk
                feat_match = (topk_idx == f_idx)  # (B, S, k)
                feat_active = feat_match & (topk_vals > max(min_val, 0))

                if not feat_active.any():
                    continue

                # Get (b, s, k_slot) positions
                positions = feat_active.nonzero(as_tuple=False)  # (N, 3)

                for pos_entry in positions:
                    b_idx = pos_entry[0].item()
                    pos = pos_entry[1].item()
                    k_slot = pos_entry[2].item()
                    val = topk_vals[b_idx, pos, k_slot].item()

                    if val <= 0:
                        continue

                    # Store raw token tuple (decode later at save time)
                    global_offset = (batch_i + b_idx) * seq_len + pos
                    ctx_start = max(0, pos - 10)
                    ctx_end = min(seq_len, pos + 15)
                    ctx_tokens = tuple(batch[b_idx][ctx_start:ctx_end].tolist())

                    entry = (val, global_offset, ctx_tokens)
                    if len(heap) < n_examples:
                        heapq.heappush(heap, entry)
                        if len(heap) == n_examples:
                            st["heap_full"][f_idx] = True
                            st["heap_mins"][f_idx] = heap[0][0]
                    elif val > heap[0][0]:
                        heapq.heapreplace(heap, entry)
                        st["heap_mins"][f_idx] = heap[0][0]

        total_tokens += batch.numel()
        elapsed = time.time() - t_start
        tps = total_tokens / elapsed if elapsed > 0 else 0
        if total_tokens % 500_000 < batch_size * seq_len:
            print(
                f"  {total_tokens:,} / {len(corpus_tokens):,} tokens "
                f"({total_tokens/len(corpus_tokens):.0%}) — "
                f"{tps:,.0f} tok/s — {elapsed:.0f}s elapsed"
            )

    # Remove all hooks
    for h in handles:
        h.remove()

    elapsed = time.time() - t_start
    print(f"\nCorpus pass complete: {total_tokens:,} tokens in {elapsed:.1f}s")

    # Build output per layer, decoding contexts now
    results = {}
    for layer, tc in transcoders.items():
        st = layer_state[layer]
        n_active = (st["activation_count"] > 0).sum().item()

        layer_data = {
            "layer": layer,
            "total_tokens": total_tokens,
            "n_features": tc.n_features,
            "n_active_features": n_active,
            "features": {},
        }

        for f_idx in range(tc.n_features):
            count = st["activation_count"][f_idx].item()
            if count == 0:
                continue

            freq = count / max(total_tokens, 1)
            examples = sorted(st["heaps"][f_idx], key=lambda x: -x[0])

            layer_data["features"][str(f_idx)] = {
                "activation_frequency": freq,
                "max_activation": st["max_activation"][f_idx].item(),
                "activation_count": count,
                "top_examples": [
                    {
                        "activation": e[0],
                        "token_offset": e[1],
                        "context": tokenizer.decode(list(e[2])),
                    }
                    for e in examples
                ],
            }

        results[layer] = layer_data

    return results


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
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

    # Load ALL transcoders at once (~300MB each, ~3GB total for 10 layers)
    print(f"\nLoading transcoders for layers {args.layers}...")
    transcoders: dict[int, Transcoder] = {}
    for layer in args.layers:
        sae_dir = find_layer_checkpoint(args.checkpoint_base, layer)
        if sae_dir is None:
            print(f"  No checkpoint found for layer {layer}, skipping")
            continue
        tc = Transcoder(sae_dir, device=device)
        transcoders[layer] = tc
        print(f"  Layer {layer}: {tc.n_features} features, k={tc.k}")

    if not transcoders:
        print("No transcoders loaded, exiting")
        sys.exit(1)

    print(f"\nLoaded {len(transcoders)} transcoders — hooking all layers simultaneously")

    # Load corpus
    corpus_tokens = load_corpus_tokens(tokenizer, args.corpus_tokens, device)

    # Single pass through corpus for ALL layers
    print(
        f"\nProcessing {len(corpus_tokens):,} tokens "
        f"(batch_size={args.batch_size}, seq_len={args.seq_len}, "
        f"{len(transcoders)} layers hooked)..."
    )
    results = collect_all_layers(
        model,
        tokenizer,
        transcoders,
        corpus_tokens,
        n_examples=args.n_examples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Save results per layer
    print()
    for layer, layer_data in sorted(results.items()):
        out_path = args.output_dir / f"activations_layer{layer}.json"
        with open(out_path, "w") as f:
            json.dump(layer_data, f, indent=2)

        n_active = layer_data["n_active_features"]
        n_total = layer_data["n_features"]
        print(f"  Layer {layer}: {n_active}/{n_total} active ({n_active/n_total:.1%}) -> {out_path}")

    # Free GPU memory
    del transcoders
    torch.cuda.empty_cache()

    print("\nDone! Run autointerp.py next to label features.")


if __name__ == "__main__":
    main()
