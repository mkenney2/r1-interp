#!/usr/bin/env python3
"""Evaluate sweep checkpoints — standalone script using safetensors directly.

Loads each trained transcoder, hooks it into the model's MLP layer,
and computes CE loss increase, dead feature %, and L0.

Usage:
    python scripts/eval_sweep.py --checkpoint-base checkpoints/sweep
    python scripts/eval_sweep.py --checkpoint-base checkpoints/sweep --output-csv results/metrics/sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import MODEL_ID


# ---------------------------------------------------------------------------
# Minimal transcoder implementation (no sparsify dependency)
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
        # Cast to bf16 to match model dtype
        state = {k: v.bfloat16() for k, v in state.items()}

        # Print shapes on first load for debugging
        print(f"    Transcoder weights: {', '.join(f'{k}: {v.shape}' for k, v in state.items())}")

        # Sparsify uses encoder.weight/encoder.bias instead of W_enc/b_enc
        self.W_enc = state.get("W_enc", state.get("encoder.weight"))
        self.b_enc = state.get("b_enc", state.get("encoder.bias"))
        self.W_dec = state["W_dec"]
        self.b_dec = state.get("b_dec", torch.zeros(self.W_dec.shape[-1], device=device))
        self.W_skip = state.get("W_skip", None)

        self.k = self.cfg.get("k", 64)
        self.d_in = self.W_enc.shape[-1]
        self.n_features = self.W_enc.shape[0]
        self.d_out = self.W_dec.shape[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run transcoder: returns (output, sparse_activations)."""
        # Encode: (batch, seq, d_in) -> (batch, seq, n_features)
        pre_acts = x @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)

        # TopK sparsification
        if self.k > 0 and self.k < acts.shape[-1]:
            topk_vals, topk_idx = acts.topk(self.k, dim=-1)
            sparse_acts = torch.zeros_like(acts)
            sparse_acts.scatter_(-1, topk_idx, topk_vals)
        else:
            sparse_acts = acts

        # Decode: (batch, seq, n_features) -> (batch, seq, d_out)
        output = sparse_acts @ self.W_dec + self.b_dec

        # Skip connection
        if self.W_skip is not None:
            output = output + x @ self.W_skip.T

        return output, sparse_acts


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def find_checkpoints(base_dir: Path) -> list[dict]:
    """Find all valid checkpoints (sweep or full training layout).

    Supports two layouts:
      Sweep:  base/sweep_exp32_k32_.../sweep_exp32_k32_..._L2/layers.2/sae.safetensors
      Full:   base/train_full_exp32_k64_..._L0/layers.0/sae.safetensors
    """
    import re

    checkpoints = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue

        m = re.search(r"exp(\d+)_k(\d+)", entry.name)
        if not m:
            continue
        exp = int(m.group(1))
        k = int(m.group(2))

        # Flat layout: dir itself has _L{N} and contains layers.{N}/
        layer_m = re.search(r"_L(\d+)$", entry.name)
        if layer_m:
            layer = int(layer_m.group(1))
            sae_dir = entry / f"layers.{layer}"
            if (sae_dir / "sae.safetensors").exists():
                checkpoints.append({
                    "expansion_factor": exp,
                    "top_k": k,
                    "layer": layer,
                    "sae_dir": sae_dir,
                    "run_name": entry.name,
                })
            continue

        # Nested layout: dir contains subdirs with _L{N}
        for layer_dir in sorted(entry.iterdir()):
            if not layer_dir.is_dir() or "_L" not in layer_dir.name:
                continue
            layer = int(layer_dir.name.split("_L")[-1])
            sae_dir = layer_dir / f"layers.{layer}"
            if not (sae_dir / "sae.safetensors").exists():
                continue
            checkpoints.append({
                "expansion_factor": exp,
                "top_k": k,
                "layer": layer,
                "sae_dir": sae_dir,
                "run_name": entry.name,
            })
    return checkpoints


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def get_eval_tokens(tokenizer, n_tokens: int = 500_000, device: str = "cuda") -> torch.Tensor:
    """Get eval tokens from openwebtext (streaming, small sample)."""
    from datasets import load_dataset

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    all_tokens: list[int] = []
    for example in ds:
        text = example.get("text", "")
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
        if len(all_tokens) >= n_tokens:
            break

    return torch.tensor(all_tokens[:n_tokens], dtype=torch.long, device=device)


def compute_ce_loss(
    model,
    input_ids: torch.Tensor,
    batch_size: int = 8,
    seq_len: int = 128,
) -> float:
    """Compute average CE loss over eval tokens."""
    n_seqs = len(input_ids) // seq_len
    tokens = input_ids[: n_seqs * seq_len].reshape(n_seqs, seq_len)

    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_seqs, batch_size):
        batch = tokens[i : i + batch_size]
        if len(batch) == 0:
            break
        with torch.no_grad():
            logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_transcoder(
    model,
    transcoder: Transcoder,
    layer: int,
    eval_tokens: torch.Tensor,
    batch_size: int = 8,
    seq_len: int = 128,
) -> dict:
    """Evaluate a single transcoder: CE loss increase, dead features, L0."""
    n_seqs = len(eval_tokens) // seq_len
    tokens = eval_tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)

    total_loss = 0.0
    n_batches = 0
    total_active = 0
    total_positions = 0
    feature_ever_active = torch.zeros(transcoder.n_features, dtype=torch.bool)

    # Hook: replace MLP output with transcoder output
    def hook_fn(module, input, output):
        nonlocal total_active, total_positions, feature_ever_active
        x = input[0]
        with torch.no_grad():
            tc_output, sparse_acts = transcoder(x)

            # Track feature stats
            active_mask = sparse_acts.abs() > 0
            n_active_per_pos = active_mask.sum(dim=-1).float()
            total_active += n_active_per_pos.sum().item()
            total_positions += x.shape[0] * x.shape[1]

            ever_active = active_mask.any(dim=0).any(dim=0)
            feature_ever_active |= ever_active.cpu()

        return tc_output

    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_hook(hook_fn)

    for i in range(0, n_seqs, batch_size):
        batch = tokens[i : i + batch_size]
        if len(batch) == 0:
            break
        with torch.no_grad():
            logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item()
            n_batches += 1

    handle.remove()

    replacement_loss = total_loss / max(n_batches, 1)
    dead_pct = 1.0 - feature_ever_active.float().mean().item()
    l0 = total_active / max(total_positions, 1)

    return {
        "replacement_loss": replacement_loss,
        "dead_feature_pct": dead_pct,
        "l0": l0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sweep checkpoints")
    parser.add_argument(
        "--checkpoint-base",
        type=Path,
        default=Path("checkpoints/sweep"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/metrics/sweep_results.csv"),
    )
    parser.add_argument("--eval-tokens", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoint_base)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_base}")
        sys.exit(1)
    print(f"Found {len(checkpoints)} checkpoints")

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

    # Get eval tokens
    print(f"Loading eval tokens ({args.eval_tokens:,})...")
    eval_tokens = get_eval_tokens(tokenizer, args.eval_tokens, device)
    print(f"  Got {len(eval_tokens):,} tokens")

    # Compute baseline CE loss (no transcoders)
    print("Computing baseline CE loss...")
    baseline_loss = compute_ce_loss(model, eval_tokens, args.batch_size)
    print(f"  Baseline CE loss: {baseline_loss:.4f}")

    # Evaluate each checkpoint
    results = []
    for i, ckpt in enumerate(checkpoints, 1):
        exp = ckpt["expansion_factor"]
        k = ckpt["top_k"]
        layer = ckpt["layer"]
        print(f"\n[{i}/{len(checkpoints)}] exp={exp}, k={k}, layer={layer}")

        try:
            transcoder = Transcoder(ckpt["sae_dir"], device=device)
            metrics = evaluate_transcoder(
                model, transcoder, layer, eval_tokens, args.batch_size
            )

            ce_increase = metrics["replacement_loss"] - baseline_loss
            result = {
                "run_name": ckpt["run_name"],
                "expansion_factor": exp,
                "top_k": k,
                "layer": layer,
                "baseline_loss": baseline_loss,
                "replacement_loss": metrics["replacement_loss"],
                "ce_loss_increase": ce_increase,
                "dead_feature_pct": metrics["dead_feature_pct"],
                "l0": metrics["l0"],
            }
            results.append(result)

            print(f"    CE increase: {ce_increase:.4f} nats")
            print(f"    Dead features: {metrics['dead_feature_pct']:.2%}")
            print(f"    L0: {metrics['l0']:.1f}")

            # Clean up GPU memory
            del transcoder
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Save CSV
    if results:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_name", "expansion_factor", "top_k", "layer",
            "baseline_loss", "replacement_loss", "ce_loss_increase",
            "dead_feature_pct", "l0",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output_csv}")

    # Summary table
    if results:
        print(f"\n{'='*80}")
        print("SWEEP RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Config':<30} {'Layer':>5} {'CE incr':>10} {'Dead %':>10} {'L0':>8}")
        print(f"{'-'*80}")
        for r in sorted(results, key=lambda x: (x["expansion_factor"], x["top_k"], x["layer"])):
            name = f"exp={r['expansion_factor']}, k={r['top_k']}"
            print(
                f"{name:<30} {r['layer']:>5} "
                f"{r['ce_loss_increase']:>10.4f} "
                f"{r['dead_feature_pct']:>9.2%} "
                f"{r['l0']:>8.1f}"
            )

        # Aggregate by config (average across layers)
        from collections import defaultdict
        groups: dict[tuple, list] = defaultdict(list)
        for r in results:
            groups[(r["expansion_factor"], r["top_k"])].append(r)

        print(f"\n{'='*80}")
        print("AVERAGED ACROSS LAYERS")
        print(f"{'='*80}")
        print(f"{'Config':<30} {'CE incr':>10} {'Dead %':>10} {'L0':>8} {'Status':>10}")
        print(f"{'-'*80}")

        for (exp, k), group in sorted(groups.items()):
            avg_ce = sum(r["ce_loss_increase"] for r in group) / len(group)
            avg_dead = sum(r["dead_feature_pct"] for r in group) / len(group)
            avg_l0 = sum(r["l0"] for r in group) / len(group)
            status = "PASS" if avg_ce < 0.1 and avg_dead < 0.10 else "FAIL"
            name = f"exp={exp}, k={k}"
            print(
                f"{name:<30} {avg_ce:>10.4f} "
                f"{avg_dead:>9.2%} {avg_l0:>8.1f} {status:>10}"
            )

        # Best config
        best = min(
            groups.items(),
            key=lambda x: sum(r["ce_loss_increase"] for r in x[1]) / len(x[1]),
        )
        print(f"\nBest config: expansion={best[0][0]}, top_k={best[0][1]}")


if __name__ == "__main__":
    main()
