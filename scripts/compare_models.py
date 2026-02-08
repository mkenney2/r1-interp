#!/usr/bin/env python3
"""Phase 4 — Compare R1-distill vs base Qwen circuit structure.

Runs motif analysis on graphs from both models and computes:
1. Pairwise cosine similarity between motif profiles
2. Kruskal-Wallis tests for per-motif differences
3. Hierarchical clustering
4. <think> vs answer phase comparison within R1 graphs

Usage:
    # Full comparison (requires graphs from both models):
    python scripts/compare_models.py --r1-graphs results/graphs/json/by_category
                                     --qwen-graphs results/qwen_graphs/json/by_category
                                     --output-dir results/comparison

    # Think vs answer comparison (R1 only):
    python scripts/compare_models.py --r1-graphs results/graphs/json/by_category
                                     --think-vs-answer
                                     --output-dir results/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CIRCUIT_MOTIFS_DIR = Path(r"C:\Users\mkenn\netowrk-motif-analysis")
if CIRCUIT_MOTIFS_DIR.exists():
    sys.path.insert(0, str(CIRCUIT_MOTIFS_DIR))

from r1_interp.utils import ensure_dir, timer


def compare_r1_vs_qwen(
    r1_graph_dir: Path,
    qwen_graph_dir: Path,
    output_dir: Path,
    n_random: int = 1000,
) -> dict:
    """Compare motif profiles between R1-distill and base Qwen."""
    try:
        from src.pipeline import run_pipeline
        from src.comparison import (
            all_pairwise_comparisons,
            cosine_similarity_matrix,
            build_task_profile,
            hierarchical_clustering,
        )
    except ImportError:
        print("ERROR: circuit-motifs not importable.")
        sys.exit(1)

    ensure_dir(output_dir)

    # Run pipeline on both models
    print("Analyzing R1-distill graphs...")
    with timer("R1 pipeline"):
        r1_results = run_pipeline(
            str(r1_graph_dir), str(output_dir / "r1"), n_random=n_random
        )

    print("\nAnalyzing base Qwen graphs...")
    with timer("Qwen pipeline"):
        qwen_results = run_pipeline(
            str(qwen_graph_dir), str(output_dir / "qwen"), n_random=n_random
        )

    # Build cross-model profiles
    # Prefix task names with model for comparison
    all_profiles = {}
    for task_name, profile in r1_results.get("task_profiles", {}).items():
        all_profiles[f"r1_{task_name}"] = profile
    for task_name, profile in qwen_results.get("task_profiles", {}).items():
        all_profiles[f"qwen_{task_name}"] = profile

    # Cross-model comparisons
    print("\nCross-model comparison...")
    comparisons = all_pairwise_comparisons(all_profiles)
    sim_matrix, task_names = cosine_similarity_matrix(all_profiles)

    # Focus on R1 vs Qwen for same task categories
    r1_tasks = {k for k in all_profiles if k.startswith("r1_")}
    qwen_tasks = {k for k in all_profiles if k.startswith("qwen_")}
    common_categories = {t.replace("r1_", "") for t in r1_tasks} & {
        t.replace("qwen_", "") for t in qwen_tasks
    }

    cross_model_results = []
    for cat in sorted(common_categories):
        r1_key = f"r1_{cat}"
        qwen_key = f"qwen_{cat}"
        for comp in comparisons:
            if (comp.task_a == r1_key and comp.task_b == qwen_key) or (
                comp.task_a == qwen_key and comp.task_b == r1_key
            ):
                cross_model_results.append(
                    {
                        "category": cat,
                        "cosine_similarity": comp.cosine_similarity,
                        "significant_motifs": comp.significant_motifs,
                    }
                )
                print(
                    f"  {cat}: cosine={comp.cosine_similarity:.3f}, "
                    f"sig_motifs={comp.significant_motifs}"
                )

    # Hierarchical clustering
    linkage, labels = hierarchical_clustering(all_profiles)

    # Save results
    summary = {
        "cross_model_comparisons": cross_model_results,
        "similarity_matrix": sim_matrix.tolist(),
        "task_names": task_names,
        "n_r1_tasks": len(r1_tasks),
        "n_qwen_tasks": len(qwen_tasks),
        "common_categories": sorted(common_categories),
    }
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return summary


def think_vs_answer_comparison(
    graph_dir: Path,
    output_dir: Path,
    n_random: int = 1000,
) -> dict:
    """Compare circuits during <think> reasoning vs. final answer.

    This requires generating separate graphs for:
    1. The <think>...</think> prefix (reasoning phase)
    2. The post-</think> answer generation phase

    Graphs should be pre-generated and organized as:
        graph_dir/think/{category}/*.json
        graph_dir/answer/{category}/*.json
    """
    try:
        from src.pipeline import run_pipeline
        from src.comparison import (
            all_pairwise_comparisons,
            cosine_similarity_matrix,
        )
    except ImportError:
        print("ERROR: circuit-motifs not importable.")
        sys.exit(1)

    ensure_dir(output_dir)

    think_dir = graph_dir / "think"
    answer_dir = graph_dir / "answer"

    if not think_dir.exists() or not answer_dir.exists():
        print(
            "ERROR: Expected subdirectories 'think/' and 'answer/' in graph_dir.\n"
            "Generate separate graphs for reasoning and answer phases first."
        )
        sys.exit(1)

    print("Analyzing <think> phase graphs...")
    with timer("think pipeline"):
        think_results = run_pipeline(
            str(think_dir), str(output_dir / "think"), n_random=n_random
        )

    print("\nAnalyzing answer phase graphs...")
    with timer("answer pipeline"):
        answer_results = run_pipeline(
            str(answer_dir), str(output_dir / "answer"), n_random=n_random
        )

    # Build combined profiles
    all_profiles = {}
    for task, profile in think_results.get("task_profiles", {}).items():
        all_profiles[f"think_{task}"] = profile
    for task, profile in answer_results.get("task_profiles", {}).items():
        all_profiles[f"answer_{task}"] = profile

    comparisons = all_pairwise_comparisons(all_profiles)
    sim_matrix, task_names = cosine_similarity_matrix(all_profiles)

    # Cross-phase comparisons for same categories
    print("\nThink vs Answer comparison:")
    phase_results = []
    think_tasks = {k.replace("think_", "") for k in all_profiles if k.startswith("think_")}
    answer_tasks = {k.replace("answer_", "") for k in all_profiles if k.startswith("answer_")}
    common = think_tasks & answer_tasks

    for cat in sorted(common):
        for comp in comparisons:
            if {comp.task_a, comp.task_b} == {f"think_{cat}", f"answer_{cat}"}:
                phase_results.append(
                    {
                        "category": cat,
                        "cosine_similarity": comp.cosine_similarity,
                        "significant_motifs": comp.significant_motifs,
                    }
                )
                print(
                    f"  {cat}: cosine={comp.cosine_similarity:.3f}, "
                    f"sig_motifs={comp.significant_motifs}"
                )

    summary = {
        "phase_comparisons": phase_results,
        "similarity_matrix": sim_matrix.tolist(),
        "task_names": task_names,
    }
    with open(output_dir / "think_vs_answer.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model circuit comparison")
    parser.add_argument("--r1-graphs", type=Path, required=True)
    parser.add_argument("--qwen-graphs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument(
        "--think-vs-answer",
        action="store_true",
        help="Compare <think> vs answer phase (R1 only)",
    )
    args = parser.parse_args()

    if args.think_vs_answer:
        think_vs_answer_comparison(
            graph_dir=args.r1_graphs,
            output_dir=args.output_dir,
            n_random=args.n_random,
        )
    elif args.qwen_graphs:
        compare_r1_vs_qwen(
            r1_graph_dir=args.r1_graphs,
            qwen_graph_dir=args.qwen_graphs,
            output_dir=args.output_dir,
            n_random=args.n_random,
        )
    else:
        print("ERROR: Provide --qwen-graphs for cross-model comparison, or --think-vs-answer")
        sys.exit(1)


if __name__ == "__main__":
    main()
