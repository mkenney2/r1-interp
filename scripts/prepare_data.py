#!/usr/bin/env python3
"""Phase 0 — Download and pre-tokenize training corpus.

Downloads OpenR1-Math (or a specified dataset), tokenizes with the Qwen
tokenizer, and saves as a memory-mapped NumPy array for fast data loading
during transcoder training.

Usage:
    python scripts/prepare_data.py [--dataset open-r1/OpenR1-Math-220k]
                                   [--max-tokens 500000000]
                                   [--eval-fraction 0.02]
                                   [--output-dir data/tokenized]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_DATASET = "open-r1/OpenR1-Math-220k"
DEFAULT_OUTPUT_DIR = Path("data/tokenized")
DEFAULT_MAX_TOKENS = 500_000_000  # 500M
DEFAULT_EVAL_FRACTION = 0.02


def load_dataset_streaming(dataset_name: str, split: str = "train"):
    """Load a HuggingFace dataset in streaming mode."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    return ds


def get_text_field(example: dict) -> str:
    """Extract text from a dataset example, trying common field names."""
    for field in ("text", "content", "problem", "question", "instruction"):
        if field in example and example[field]:
            return str(example[field])
    # For OpenR1-Math: concatenate problem + solution if both exist
    parts = []
    if "problem" in example:
        parts.append(str(example["problem"]))
    if "solution" in example:
        parts.append(str(example["solution"]))
    if "generated_solution" in example:
        parts.append(str(example["generated_solution"]))
    if parts:
        return "\n".join(parts)
    # Last resort: join all string values
    return "\n".join(str(v) for v in example.values() if isinstance(v, str))


def tokenize_and_save(
    dataset_name: str,
    output_dir: Path,
    max_tokens: int,
    eval_fraction: float,
) -> None:
    """Download, tokenize, and save as memory-mapped arrays."""
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Stream and tokenize
    print(f"Streaming {dataset_name}...")
    ds = load_dataset_streaming(dataset_name)

    all_tokens: list[int] = []
    n_examples = 0

    pbar = tqdm(total=max_tokens, unit="tok", desc="Tokenizing")
    for example in ds:
        text = get_text_field(example)
        if not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(token_ids)
        n_examples += 1
        pbar.update(len(token_ids))
        if len(all_tokens) >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            break
    pbar.close()

    print(f"Tokenized {n_examples} examples -> {len(all_tokens):,} tokens")

    # Convert to numpy
    token_array = np.array(all_tokens, dtype=np.int32)

    # Split eval set
    n_eval = int(len(token_array) * eval_fraction)
    n_train = len(token_array) - n_eval

    train_tokens = token_array[:n_train]
    eval_tokens = token_array[n_train:]

    # Save as memory-mapped files
    train_path = output_dir / "train_tokens.npy"
    eval_path = output_dir / "eval_tokens.npy"

    np.save(train_path, train_tokens)
    np.save(eval_path, eval_tokens)

    # Save metadata
    meta = {
        "dataset": dataset_name,
        "model_id": MODEL_ID,
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": len(token_array),
        "train_tokens": n_train,
        "eval_tokens": n_eval,
        "n_examples": n_examples,
        "dtype": "int32",
    }
    import json

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({n_train:,} tokens)")
    print(f"  Eval:  {eval_path} ({n_eval:,} tokens)")
    print(f"  Meta:  {output_dir / 'metadata.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize training corpus")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to collect (default: {DEFAULT_MAX_TOKENS:,})",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=DEFAULT_EVAL_FRACTION,
        help=f"Fraction of tokens held out for eval (default: {DEFAULT_EVAL_FRACTION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    tokenize_and_save(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        eval_fraction=args.eval_fraction,
    )


if __name__ == "__main__":
    main()
