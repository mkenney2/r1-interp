#!/usr/bin/env python3
"""Phase 1 — Hyperparameter sweep driver.

Iterates over all (expansion_factor, top_k) combinations from SweepConfig,
trains each on the pilot layers, evaluates faithfulness, and selects the
Pareto-optimal configuration.

Usage:
    python scripts/sweep.py --config configs/sweep_pilot.yaml
    python scripts/sweep.py --config configs/sweep_pilot.yaml --dry-run
    python scripts/sweep.py --results-only --results-csv sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import SweepConfig, TranscoderHyperparams, load_config
from r1_interp.utils import ensure_dir


@dataclass
class SweepResult:
    """Metrics from a single sweep configuration."""

    expansion_factor: int
    top_k: int
    layer: int
    ce_loss_increase: float  # nats
    dead_feature_pct: float
    l0: float  # avg active features per token
    run_name: str


def run_sweep(
    config: SweepConfig,
    checkpoint_base: str = "checkpoints/sweep",
    results_csv: Path = Path("results/metrics/sweep_results.csv"),
    dry_run: bool = False,
) -> list[SweepResult]:
    """Run all sweep configs, evaluate, and save results."""
    ensure_dir(checkpoint_base)
    ensure_dir(results_csv.parent)

    sweep_configs = config.generate_configs()
    print(f"Sweep: {len(sweep_configs)} configs x {len(config.layers)} layers")

    results: list[SweepResult] = []

    for i, cfg in enumerate(sweep_configs, 1):
        hp = TranscoderHyperparams(**cfg["hyperparams"])
        run_name = cfg["run_name"]
        layers = cfg["layers"]

        print(f"\n{'='*60}")
        print(f"[{i}/{len(sweep_configs)}] {run_name}")
        print(f"  expansion={hp.expansion_factor}, top_k={hp.top_k}")
        print(f"  layers={layers}")
        print(f"{'='*60}")

        for layer in layers:
            ckpt_dir = f"{checkpoint_base}/{run_name}"

            # Train
            train_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train.py"),
                "--config", "/dev/null",  # we pass args directly below
            ]
            # Build sparsify command directly
            from scripts.train import build_sparsify_command

            cmd = build_sparsify_command(
                hp=hp,
                layer=layer,
                dataset=cfg["dataset"],
                wandb_project=cfg["wandb_project"],
                run_name=run_name,
                checkpoint_dir=ckpt_dir,
            )

            if dry_run:
                print(f"  [DRY RUN] L{layer}: {' '.join(cmd)}")
                results.append(
                    SweepResult(
                        expansion_factor=hp.expansion_factor,
                        top_k=hp.top_k,
                        layer=layer,
                        ce_loss_increase=0.0,
                        dead_feature_pct=0.0,
                        l0=0.0,
                        run_name=run_name,
                    )
                )
                continue

            # Run training
            print(f"  Training layer {layer}...")
            train_result = subprocess.run(cmd)
            if train_result.returncode != 0:
                print(f"  WARNING: Training failed for L{layer}, skipping eval")
                continue

            # Evaluate
            print(f"  Evaluating layer {layer}...")
            eval_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                "--transcoder-dir", ckpt_dir,
                "--layers", str(layer),
                "--output-json", f"{ckpt_dir}/eval_L{layer}.json",
            ]
            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

            if eval_result.returncode == 0:
                import json

                metrics = json.loads(eval_result.stdout)
                results.append(
                    SweepResult(
                        expansion_factor=hp.expansion_factor,
                        top_k=hp.top_k,
                        layer=layer,
                        ce_loss_increase=metrics.get("ce_loss_increase", float("inf")),
                        dead_feature_pct=metrics.get("dead_feature_pct", 1.0),
                        l0=metrics.get("l0", 0.0),
                        run_name=run_name,
                    )
                )
            else:
                print(f"  WARNING: Eval failed for L{layer}")

    # Save results to CSV
    if results:
        save_results_csv(results, results_csv)
        print(f"\nResults saved to {results_csv}")

        # Select Pareto-optimal config
        best = select_pareto_optimal(results)
        if best:
            print(f"\nPareto-optimal config:")
            print(f"  expansion_factor = {best.expansion_factor}")
            print(f"  top_k = {best.top_k}")
            print(f"  Avg CE loss increase = {best.ce_loss_increase:.4f} nats")
            print(f"  Avg dead features = {best.dead_feature_pct:.2%}")

    return results


def save_results_csv(results: list[SweepResult], path: Path) -> None:
    """Save sweep results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name", "expansion_factor", "top_k", "layer",
                "ce_loss_increase", "dead_feature_pct", "l0",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "run_name": r.run_name,
                    "expansion_factor": r.expansion_factor,
                    "top_k": r.top_k,
                    "layer": r.layer,
                    "ce_loss_increase": r.ce_loss_increase,
                    "dead_feature_pct": r.dead_feature_pct,
                    "l0": r.l0,
                }
            )


def select_pareto_optimal(
    results: list[SweepResult],
    ce_threshold: float = 0.1,
    dead_threshold: float = 0.10,
) -> SweepResult | None:
    """Select the best config: lowest CE loss increase that meets thresholds.

    Aggregates across layers by averaging metrics per (expansion, top_k) pair.
    """
    from collections import defaultdict

    # Group by (expansion, top_k)
    groups: dict[tuple[int, int], list[SweepResult]] = defaultdict(list)
    for r in results:
        groups[(r.expansion_factor, r.top_k)].append(r)

    # Average metrics per config
    candidates: list[SweepResult] = []
    for (exp, k), group in groups.items():
        avg_ce = sum(r.ce_loss_increase for r in group) / len(group)
        avg_dead = sum(r.dead_feature_pct for r in group) / len(group)
        avg_l0 = sum(r.l0 for r in group) / len(group)
        candidates.append(
            SweepResult(
                expansion_factor=exp,
                top_k=k,
                layer=-1,  # aggregated
                ce_loss_increase=avg_ce,
                dead_feature_pct=avg_dead,
                l0=avg_l0,
                run_name=group[0].run_name,
            )
        )

    # Filter by thresholds
    valid = [
        c for c in candidates
        if c.ce_loss_increase <= ce_threshold and c.dead_feature_pct <= dead_threshold
    ]

    if not valid:
        print("WARNING: No config meets both thresholds. Relaxing to best CE loss.")
        valid = candidates

    # Sort by CE loss (primary), then dead features (secondary)
    valid.sort(key=lambda c: (c.ce_loss_increase, c.dead_feature_pct))
    return valid[0] if valid else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep driver")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-base", default="checkpoints/sweep")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/metrics/sweep_results.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not isinstance(cfg, SweepConfig):
        print(f"ERROR: Config type is {type(cfg).__name__}, expected SweepConfig")
        sys.exit(1)

    run_sweep(
        config=cfg,
        checkpoint_base=args.checkpoint_base,
        results_csv=args.results_csv,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
