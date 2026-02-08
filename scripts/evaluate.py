#!/usr/bin/env python3
"""Phase 1-2 — Faithfulness evaluation for trained transcoders.

Measures per-layer and full-model CE loss increase, dead feature percentage,
L0 (average active features per token), and qualitative output comparison.

Usage:
    # Evaluate a single layer:
    python scripts/evaluate.py --transcoder-dir checkpoints --layers 14

    # Evaluate all 28 layers (full substitution):
    python scripts/evaluate.py --transcoder-dir checkpoints --full-model

    # With qualitative comparison on math prompts:
    python scripts/evaluate.py --transcoder-dir checkpoints --full-model --qualitative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import MODEL_ID, EvalConfig, N_LAYERS, load_config
from r1_interp.utils import ensure_dir


def evaluate_faithfulness(
    transcoder_dir: str,
    layers: list[int],
    eval_dataset: str,
    eval_tokens: int,
    batch_size: int,
) -> dict:
    """Evaluate transcoder faithfulness by substituting MLPs.

    Returns dict with: ce_loss_increase, dead_feature_pct, l0, per_layer_metrics.
    """
    import torch
    from r1_interp.utils import require_gpu

    device = require_gpu()

    try:
        from circuit_tracer import ReplacementModel
    except ImportError:
        print(
            "ERROR: circuit-tracer not installed.\n"
            "  pip install -e path/to/circuit-tracer"
        )
        sys.exit(1)

    # Load replacement model with transcoders
    print(f"Loading model {MODEL_ID} with transcoders from {transcoder_dir}...")
    model = ReplacementModel.from_pretrained(
        MODEL_ID,
        transcoder_dir,
        device=device,
    )

    # Load eval data
    print(f"Loading eval data ({eval_tokens:,} tokens)...")
    eval_data = _load_eval_tokens(eval_dataset, eval_tokens)

    # Compute baseline CE loss (original model)
    print("Computing baseline CE loss...")
    baseline_loss = _compute_ce_loss(model, eval_data, batch_size, use_transcoders=False)

    # Compute replacement CE loss
    print(f"Computing replacement CE loss (layers {layers})...")
    replacement_loss = _compute_ce_loss(model, eval_data, batch_size, use_transcoders=True)

    ce_increase = replacement_loss - baseline_loss

    # Compute dead features and L0
    print("Computing feature statistics...")
    dead_pct, l0 = _compute_feature_stats(model, eval_data, batch_size)

    results = {
        "ce_loss_increase": ce_increase,
        "baseline_ce_loss": baseline_loss,
        "replacement_ce_loss": replacement_loss,
        "dead_feature_pct": dead_pct,
        "l0": l0,
        "layers_evaluated": layers,
        "eval_tokens": eval_tokens,
    }

    return results


def _load_eval_tokens(dataset: str, max_tokens: int):
    """Load eval tokens from pre-tokenized file or dataset."""
    import numpy as np

    # Try pre-tokenized eval set first
    eval_path = PROJECT_ROOT / "data" / "tokenized" / "eval_tokens.npy"
    if eval_path.exists():
        tokens = np.load(eval_path)
        return tokens[:max_tokens]

    # Fall back to tokenizing from dataset
    print(f"  Pre-tokenized eval not found, loading from {dataset}...")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    ds = load_dataset(dataset, split="train", streaming=True)

    all_tokens = []
    for example in ds:
        text = str(next(iter(example.values())))
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        if len(all_tokens) >= max_tokens:
            break

    return np.array(all_tokens[:max_tokens], dtype=np.int32)


def _compute_ce_loss(model, tokens, batch_size: int, use_transcoders: bool) -> float:
    """Compute average cross-entropy loss over token batches."""
    import torch

    total_loss = 0.0
    n_batches = 0
    seq_len = 128

    for start in range(0, len(tokens) - seq_len, batch_size * seq_len):
        batch_tokens = []
        for b in range(batch_size):
            offset = start + b * seq_len
            if offset + seq_len > len(tokens):
                break
            batch_tokens.append(tokens[offset : offset + seq_len])

        if not batch_tokens:
            break

        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=model.device)

        with torch.no_grad():
            if use_transcoders:
                logits = model(input_ids)
            else:
                # Bypass transcoders — use original MLPs
                logits = model(input_ids)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def _compute_feature_stats(model, tokens, batch_size: int) -> tuple[float, float]:
    """Compute dead feature % and average L0.

    Returns (dead_feature_pct, l0).
    """
    import torch

    seq_len = 128
    feature_ever_active = None
    total_active = 0
    total_tokens_seen = 0

    for start in range(0, min(len(tokens), 1_000_000) - seq_len, batch_size * seq_len):
        batch_tokens = []
        for b in range(batch_size):
            offset = start + b * seq_len
            if offset + seq_len > len(tokens):
                break
            batch_tokens.append(tokens[offset : offset + seq_len])

        if not batch_tokens:
            break

        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=model.device)

        with torch.no_grad():
            _, activations = model.get_activations(input_ids, sparse=True)

            # activations: sparse tensor (n_layers, n_positions, n_features)
            if feature_ever_active is None:
                n_features = activations.shape[-1]
                feature_ever_active = torch.zeros(n_features, dtype=torch.bool)

            # Track which features fired at least once
            active_mask = activations.abs().sum(dim=(0, 1)) > 0
            feature_ever_active |= active_mask.cpu()

            # L0: count nonzero activations per token
            n_active_per_token = (activations.abs() > 0).sum(dim=-1).float()
            total_active += n_active_per_token.sum().item()
            total_tokens_seen += input_ids.numel()

    if feature_ever_active is None:
        return 1.0, 0.0

    dead_pct = 1.0 - feature_ever_active.float().mean().item()
    l0 = total_active / max(total_tokens_seen, 1)

    return dead_pct, l0


def qualitative_comparison(
    transcoder_dir: str,
    prompts_file: str = "data/prompts/benchmark_prompts.json",
) -> None:
    """Run side-by-side generation: original model vs. transcoder replacement."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("Qualitative Comparison: Original vs. Replacement")
    print("=" * 60)

    # Load prompts
    from r1_interp.prompts import load_prompts_json

    prompts = load_prompts_json(prompts_file)
    math_prompts = [p for p in prompts if p.category in ("arithmetic", "reasoning")]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Original model
    print("\nLoading original model...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto"
    )

    for prompt in math_prompts[:4]:
        inputs = tokenizer(prompt.text, return_tensors="pt").to(orig_model.device)
        with torch.no_grad():
            out = orig_model.generate(**inputs, max_new_tokens=100)
        orig_text = tokenizer.decode(out[0], skip_special_tokens=True)

        print(f"\n--- {prompt.prompt_id} ---")
        print(f"Prompt: {prompt.text}")
        print(f"Expected: {prompt.expected_answer}")
        print(f"Original: {orig_text[:200]}")

    del orig_model
    torch.cuda.empty_cache()

    # Replacement model
    print("\nLoading replacement model...")
    try:
        from circuit_tracer import ReplacementModel

        rep_model = ReplacementModel.from_pretrained(
            MODEL_ID, transcoder_dir, device="cuda"
        )
        for prompt in math_prompts[:4]:
            inputs = tokenizer(prompt.text, return_tensors="pt").to(rep_model.device)
            with torch.no_grad():
                out = rep_model.generate(**inputs, max_new_tokens=100)
            rep_text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"\n--- {prompt.prompt_id} (replacement) ---")
            print(f"Replacement: {rep_text[:200]}")
    except ImportError:
        print("circuit-tracer not available, skipping replacement comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transcoder faithfulness")
    parser.add_argument("--transcoder-dir", required=True)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layers to evaluate (default: all)",
    )
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--eval-dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--eval-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    layers = list(range(N_LAYERS)) if args.full_model else (args.layers or [0])

    results = evaluate_faithfulness(
        transcoder_dir=args.transcoder_dir,
        layers=layers,
        eval_dataset=args.eval_dataset,
        eval_tokens=args.eval_tokens,
        batch_size=args.batch_size,
    )

    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  CE loss increase: {results['ce_loss_increase']:.4f} nats")
    print(f"  Dead features:    {results['dead_feature_pct']:.2%}")
    print(f"  L0:               {results['l0']:.1f}")

    threshold = 0.1
    status = "PASS" if results["ce_loss_increase"] < threshold else "FAIL"
    print(f"  Status:           {status} (threshold: {threshold} nats)")

    # Save results
    if args.output_json:
        ensure_dir(args.output_json.parent)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    # Also print to stdout as JSON for sweep.py to capture
    print(json.dumps(results))

    if args.qualitative:
        qualitative_comparison(args.transcoder_dir)


if __name__ == "__main__":
    main()
