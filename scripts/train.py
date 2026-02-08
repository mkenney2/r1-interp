#!/usr/bin/env python3
"""Phase 1-2 — Transcoder training wrapper.

Thin wrapper that translates our YAML config into sparsify CLI args or
Python API calls.  Supports single-layer and multi-GPU (torchrun) modes.

Usage:
    # Single layer, single GPU:
    python scripts/train.py --config configs/train_full.yaml --layer 14

    # All layers, multi-GPU (via torchrun):
    torchrun --nproc_per_node=8 scripts/train.py --config configs/train_full.yaml

    # Dry-run (print commands without executing):
    python scripts/train.py --config configs/train_full.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import (
    MODEL_ID,
    TrainRunConfig,
    TranscoderHyperparams,
    load_config,
)
from r1_interp.utils import checkpoint_name, ensure_dir


def build_sparsify_command(
    hp: TranscoderHyperparams,
    layer: int,
    dataset: str,
    wandb_project: str,
    run_name: str,
    checkpoint_dir: str,
) -> list[str]:
    """Build CLI args for sparsify training."""
    cmd = [
        sys.executable,
        "-m",
        "sparsify",
        MODEL_ID,
        dataset,
        "--transcode",
        "--expansion_factor",
        str(hp.expansion_factor),
        "-k",
        str(hp.top_k),
        "--layers",
        str(layer),
        "--batch_size",
        str(hp.batch_size),
        "--lr_warmup_steps",
        str(hp.warmup_steps),
    ]
    if hp.learning_rate is not None:
        cmd.extend(["--lr", str(hp.learning_rate)])
    if hp.skip_connection:
        cmd.append("--skip_connection")
    cmd.extend(["--run_name", f"{run_name}_L{layer}"])
    cmd.extend(["--save_dir", checkpoint_dir])
    return cmd


def build_sparsify_env(wandb_project: str) -> dict[str, str]:
    """Build environment variables for sparsify (e.g. WANDB_PROJECT)."""
    env = dict(os.environ)
    env["WANDB_PROJECT"] = wandb_project
    return env


def train_layer(
    hp: TranscoderHyperparams,
    layer: int,
    dataset: str,
    wandb_project: str,
    run_name: str,
    checkpoint_dir: str,
    dry_run: bool = False,
) -> int:
    """Train a single transcoder for one layer. Returns exit code."""
    cmd = build_sparsify_command(
        hp, layer, dataset, wandb_project, run_name, checkpoint_dir
    )
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY RUN] Layer {layer}: {cmd_str}")
        return 0

    print(f"\n{'='*60}")
    print(f"Training layer {layer}")
    print(f"Command: {cmd_str}")
    print(f"{'='*60}\n")

    env = build_sparsify_env(wandb_project)
    result = subprocess.run(cmd, env=env)
    return result.returncode


def train_python_api(
    hp: TranscoderHyperparams,
    layers: list[int],
    dataset: str,
    wandb_project: str,
    run_name: str,
    checkpoint_dir: str,
    distribute_modules: bool = True,
) -> None:
    """Train via sparsify Python API (for multi-GPU with torchrun)."""
    try:
        from sparsify import Trainer, TrainerConfig
    except ImportError:
        print(
            "ERROR: sparsify not installed. Clone and install it:\n"
            "  git clone https://github.com/EleutherAI/sparsify.git\n"
            "  pip install -e sparsify"
        )
        sys.exit(1)

    config = TrainerConfig(
        model_name=MODEL_ID,
        dataset=dataset,
        transcode=True,
        expansion_factor=hp.expansion_factor,
        k=hp.top_k,
        skip_connection=hp.skip_connection,
        layers=layers,
        lr=hp.learning_rate,
        num_tokens=hp.training_tokens,
        batch_size=hp.batch_size,
        warmup_steps=hp.warmup_steps,
        wandb_project=wandb_project,
        run_name=run_name,
        save_dir=checkpoint_dir,
        distribute_modules=distribute_modules,
    )

    trainer = Trainer(config)
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcoder training wrapper")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Train a single layer (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use sparsify Python API instead of CLI subprocess",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not isinstance(cfg, TrainRunConfig):
        print(f"ERROR: Config type is {type(cfg).__name__}, expected TrainRunConfig")
        sys.exit(1)

    ensure_dir(cfg.checkpoint_dir)

    layers = [args.layer] if args.layer is not None else list(cfg.layers)

    if args.use_api:
        train_python_api(
            hp=cfg.hyperparams,
            layers=layers,
            dataset=cfg.dataset,
            wandb_project=cfg.wandb_project,
            run_name=cfg.run_name,
            checkpoint_dir=cfg.checkpoint_dir,
            distribute_modules=cfg.distribute_modules,
        )
    else:
        failed_layers = []
        for layer in layers:
            rc = train_layer(
                hp=cfg.hyperparams,
                layer=layer,
                dataset=cfg.dataset,
                wandb_project=cfg.wandb_project,
                run_name=cfg.run_name,
                checkpoint_dir=cfg.checkpoint_dir,
                dry_run=args.dry_run,
            )
            if rc != 0:
                failed_layers.append(layer)
                print(f"WARNING: Layer {layer} training failed (exit code {rc})")

        if failed_layers:
            print(f"\nFailed layers: {failed_layers}")
            sys.exit(1)
        else:
            print(f"\nAll {len(layers)} layers trained successfully.")


if __name__ == "__main__":
    main()
