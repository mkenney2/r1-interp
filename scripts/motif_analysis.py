#!/usr/bin/env python3
"""Phase 4 — Run circuit-motifs pipeline on R1 attribution graphs.

Wrapper around circuit-motifs ``run_pipeline()`` that points at the
R1 graph JSON directory.  Handles sys.path setup so circuit-motifs
can be imported from its installed location.

Usage:
    python scripts/motif_analysis.py --graph-dir results/graphs/json/by_category
                                     --results-dir results/motifs
                                     [--n-random 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# circuit-motifs is installed editable at C:\Users\mkenn\netowrk-motif-analysis
# Ensure it's importable
CIRCUIT_MOTIFS_DIR = Path(r"C:\Users\mkenn\netowrk-motif-analysis")
if CIRCUIT_MOTIFS_DIR.exists():
    sys.path.insert(0, str(CIRCUIT_MOTIFS_DIR))


def run_motif_pipeline(
    graph_dir: Path,
    results_dir: Path,
    n_random: int = 1000,
    motif_size: int = 3,
) -> dict:
    """Run the full circuit-motifs pipeline on R1 graphs.

    Expects graph_dir to contain category subdirectories:
        graph_dir/arithmetic/*.json
        graph_dir/reasoning/*.json
        ...
    """
    from r1_interp.utils import ensure_dir, timer

    ensure_dir(results_dir)

    try:
        from src.pipeline import run_pipeline
    except ImportError:
        print(
            "ERROR: circuit-motifs not importable.\n"
            "  Ensure it's installed: pip install -e path/to/netowrk-motif-analysis"
        )
        sys.exit(1)

    # Register R1 models in the registry
    from r1_interp.model_registry import register_r1_models

    register_r1_models()

    print(f"Running motif pipeline on {graph_dir}...")
    print(f"  n_random={n_random}, motif_size={motif_size}")

    with timer("motif pipeline"):
        results = run_pipeline(
            data_dir=str(graph_dir),
            results_dir=str(results_dir),
            n_random=n_random,
            motif_size=motif_size,
        )

    # Print summary
    _print_summary(results, results_dir)

    return results


def _print_summary(results: dict, results_dir: Path) -> None:
    """Print a human-readable summary of the motif analysis."""
    print("\n" + "=" * 60)
    print("Motif Analysis Summary")
    print("=" * 60)

    # Per-task profiles
    if "task_profiles" in results:
        for task_name, profile in results["task_profiles"].items():
            print(f"\n  {task_name}: {profile.n_graphs} graphs")
            if hasattr(profile, "mean_sp") and profile.mean_sp is not None:
                # Top enriched motifs
                sp = profile.mean_sp
                try:
                    from src.motif_census import TRIAD_LABELS, CONNECTED_TRIAD_INDICES

                    print("    Top motifs (by significance profile):")
                    sorted_motifs = sorted(
                        CONNECTED_TRIAD_INDICES,
                        key=lambda i: abs(sp[i]) if i < len(sp) else 0,
                        reverse=True,
                    )
                    for idx in sorted_motifs[:5]:
                        if idx < len(sp) and idx < len(TRIAD_LABELS):
                            print(f"      {TRIAD_LABELS[idx]}: {sp[idx]:.3f}")
                except ImportError:
                    pass

    # Pairwise comparisons
    if "comparisons" in results:
        print(f"\n  Pairwise comparisons: {len(results['comparisons'])}")
        for comp in results["comparisons"]:
            print(
                f"    {comp.task_a} vs {comp.task_b}: "
                f"cosine={comp.cosine_similarity:.3f}, "
                f"significant_motifs={comp.significant_motifs}"
            )

    # Kruskal-Wallis
    if "kruskal_wallis" in results:
        sig = [kw for kw in results["kruskal_wallis"] if kw.get("significant")]
        print(f"\n  Kruskal-Wallis: {len(sig)} motifs differ across tasks")

    print(f"\n  Full results saved to: {results_dir}")


def analyze_single_prompt(
    json_path: Path,
    n_random: int = 100,
) -> dict:
    """Quick single-graph analysis for debugging / exploration."""
    try:
        from src.pipeline import analyze_single_graph
    except ImportError:
        print("ERROR: circuit-motifs not importable.")
        sys.exit(1)

    result = analyze_single_graph(str(json_path), n_random=n_random)

    print(f"\nGraph: {json_path.name}")
    print(f"  Nodes: {result['summary']['n_nodes']}")
    print(f"  Edges: {result['summary']['n_edges']}")
    print(f"  Density: {result['summary']['density']:.4f}")

    census = result["census"]
    print(f"  Connected triads: {sum(census.connected_counts().values())}")

    null = result["null_result"]
    print(f"  Significant motifs (|Z| > 2):")
    for label, z in zip(null.labels, null.z_scores):
        if abs(z) > 2:
            print(f"    {label}: Z={z:.2f}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run circuit-motifs analysis")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("results/graphs/json/by_category"),
        help="Directory with category subdirectories of graph JSONs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/motifs"),
    )
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--motif-size", type=int, default=3)
    parser.add_argument(
        "--single",
        type=Path,
        default=None,
        help="Analyze a single graph JSON file (quick debug mode)",
    )
    args = parser.parse_args()

    if args.single:
        analyze_single_prompt(args.single, n_random=args.n_random)
    else:
        run_motif_pipeline(
            graph_dir=args.graph_dir,
            results_dir=args.results_dir,
            n_random=args.n_random,
            motif_size=args.motif_size,
        )


if __name__ == "__main__":
    main()
