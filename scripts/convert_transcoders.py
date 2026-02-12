#!/usr/bin/env python3
"""Convert sparsify transcoder checkpoints to circuit-tracer format.

Sparsify saves:
  checkpoints/train_full_..._L{N}/layers.{N}/sae.safetensors
  with keys: encoder.weight, encoder.bias, W_dec, b_dec, W_skip

Circuit-tracer expects:
  transcoders/
    config.yaml
    layer_{N}.safetensors  (keys: W_enc, b_enc, W_dec, b_dec, W_skip)

Usage:
    python scripts/convert_transcoders.py \
        --checkpoint-base checkpoints \
        --layers 8 9 10 11 12 13 14 15 16 17 \
        --output-dir transcoders
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Key name mapping: sparsify -> circuit-tracer
KEY_MAP = {
    "encoder.weight": "W_enc",
    "encoder.bias": "b_enc",
    # These are the same in both formats:
    "W_enc": "W_enc",
    "b_enc": "b_enc",
    "W_dec": "W_dec",
    "b_dec": "b_dec",
    "W_skip": "W_skip",
}


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


def convert_layer(sae_dir: Path, output_path: Path) -> dict:
    """Convert a single layer's transcoder weights."""
    state = load_file(str(sae_dir / "sae.safetensors"))

    # Read config
    cfg_path = sae_dir / "cfg.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Rename keys
    new_state = {}
    for old_key, tensor in state.items():
        new_key = KEY_MAP.get(old_key, old_key)
        new_state[new_key] = tensor

    save_file(new_state, str(output_path))

    return {
        "n_features": new_state["W_enc"].shape[0],
        "d_model": new_state["W_enc"].shape[1],
        "k": cfg.get("k", 64),
        "has_skip": "W_skip" in new_state,
        "keys": list(new_state.keys()),
    }


def write_config(output_dir: Path, layers: list[int]) -> None:
    """Write circuit-tracer config.yaml."""
    config = {
        "model_kind": "transcoder_set",
        "feature_input_hook": "hook_mlp_in",
        "feature_output_hook": "hook_mlp_out",
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert sparsify transcoders to circuit-tracer format"
    )
    parser.add_argument("--checkpoint-base", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(8, 18)),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("transcoders"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting transcoders from {args.checkpoint_base} -> {args.output_dir}")

    converted = []
    for layer in args.layers:
        sae_dir = find_layer_checkpoint(args.checkpoint_base, layer)
        if sae_dir is None:
            print(f"  Layer {layer}: no checkpoint found, skipping")
            continue

        out_path = args.output_dir / f"layer_{layer}.safetensors"
        info = convert_layer(sae_dir, out_path)
        converted.append(layer)
        print(
            f"  Layer {layer}: {info['n_features']} features, "
            f"d_model={info['d_model']}, k={info['k']}, "
            f"skip={info['has_skip']} -> {out_path.name}"
        )

    # Write config.yaml
    write_config(args.output_dir, converted)
    print(f"\nConverted {len(converted)} layers. Config written to {args.output_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
