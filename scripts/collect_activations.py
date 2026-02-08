#!/usr/bin/env python3
"""Phase 3 — Collect top activating examples for each transcoder feature.

Runs the replacement model on a diverse corpus and records:
- Top-N activating examples per feature (context + activation value)
- Activation frequency per feature
- Max activation value per feature

Output is saved as JSON per layer for downstream autointerp labeling.

Usage:
    python scripts/collect_activations.py --transcoder-dir checkpoints
                                          --output-dir results/features/activations
                                          [--n-examples 20]
                                          [--corpus-tokens 50000000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import MODEL_ID, N_LAYERS
from r1_interp.utils import ensure_dir, require_gpu


def collect_top_activations(
    transcoder_dir: str,
    output_dir: Path,
    n_examples: int = 20,
    corpus_tokens: int = 50_000_000,
    batch_size: int = 32,
    seq_len: int = 128,
    min_activation_freq: float = 0.001,
) -> None:
    """Collect top-activating examples for every feature."""
    import heapq

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    device = require_gpu()
    ensure_dir(output_dir)

    try:
        from circuit_tracer import ReplacementModel
    except ImportError:
        print("ERROR: circuit-tracer not installed.")
        sys.exit(1)

    print(f"Loading model and transcoders from {transcoder_dir}...")
    model = ReplacementModel.from_pretrained(MODEL_ID, transcoder_dir, device=device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load tokenized corpus
    eval_path = PROJECT_ROOT / "data" / "tokenized" / "train_tokens.npy"
    if eval_path.exists():
        tokens = np.load(eval_path)[:corpus_tokens]
    else:
        print("Pre-tokenized data not found. Run prepare_data.py first.")
        sys.exit(1)

    # Per-feature tracking: heap of (activation_value, context_text)
    # Using min-heaps of size n_examples to keep top-N
    n_features = model.transcoders[0].W_dec.shape[0] if hasattr(model, "transcoders") else 0

    # Track per layer
    for layer_idx in range(N_LAYERS):
        print(f"\nProcessing layer {layer_idx}/{N_LAYERS - 1}...")

        # Min-heap per feature: (activation_value, token_offset, context_snippet)
        top_examples: dict[int, list] = defaultdict(list)
        activation_count: dict[int, int] = defaultdict(int)
        max_activation: dict[int, float] = defaultdict(float)
        total_tokens_seen = 0

        for start in range(0, len(tokens) - seq_len, batch_size * seq_len):
            batch = []
            offsets = []
            for b in range(batch_size):
                offset = start + b * seq_len
                if offset + seq_len > len(tokens):
                    break
                batch.append(tokens[offset : offset + seq_len])
                offsets.append(offset)

            if not batch:
                break

            input_ids = torch.tensor(batch, dtype=torch.long, device=device)

            with torch.no_grad():
                _, activations = model.get_activations(input_ids, sparse=True)

                # activations shape: (n_layers, batch, seq_len, n_features)
                # Extract this layer
                layer_acts = activations[layer_idx]  # (batch, seq_len, n_features)

                for b_idx in range(layer_acts.shape[0]):
                    for pos in range(layer_acts.shape[1]):
                        active = layer_acts[b_idx, pos].nonzero(as_tuple=True)[0]
                        for feat_idx in active:
                            feat_idx = feat_idx.item()
                            val = layer_acts[b_idx, pos, feat_idx].item()

                            activation_count[feat_idx] += 1
                            max_activation[feat_idx] = max(max_activation[feat_idx], val)

                            # Context: decode surrounding tokens
                            ctx_start = max(0, pos - 5)
                            ctx_end = min(seq_len, pos + 10)
                            ctx_tokens = batch[b_idx][ctx_start:ctx_end].tolist()
                            context = tokenizer.decode(ctx_tokens)

                            entry = (val, offsets[b_idx] + pos, context)
                            if len(top_examples[feat_idx]) < n_examples:
                                heapq.heappush(top_examples[feat_idx], entry)
                            elif val > top_examples[feat_idx][0][0]:
                                heapq.heapreplace(top_examples[feat_idx], entry)

            total_tokens_seen += input_ids.numel()
            if total_tokens_seen % 1_000_000 < batch_size * seq_len:
                print(f"  {total_tokens_seen:,} tokens processed...")

        # Save results for this layer
        layer_data = {
            "layer": layer_idx,
            "total_tokens": total_tokens_seen,
            "n_active_features": len(activation_count),
            "features": {},
        }

        for feat_idx in sorted(activation_count.keys()):
            freq = activation_count[feat_idx] / max(total_tokens_seen, 1)
            if freq < min_activation_freq:
                continue

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

        out_path = output_dir / f"activations_layer{layer_idx}.json"
        with open(out_path, "w") as f:
            json.dump(layer_data, f, indent=2)
        print(f"  Saved {len(layer_data['features'])} features -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect top activating examples per feature"
    )
    parser.add_argument("--transcoder-dir", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/features/activations"))
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--corpus-tokens", type=int, default=50_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    collect_top_activations(
        transcoder_dir=args.transcoder_dir,
        output_dir=args.output_dir,
        n_examples=args.n_examples,
        corpus_tokens=args.corpus_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
