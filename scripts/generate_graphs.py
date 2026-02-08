#!/usr/bin/env python3
"""Phase 4 — Generate attribution graphs for benchmark prompts.

Uses circuit-tracer's ``attribute()`` to generate per-prompt attribution
graphs, saves them as .pt files, and exports to Neuronpedia JSON via
``create_graph_files()``.

Usage:
    python scripts/generate_graphs.py --config configs/graph_generation.yaml
    python scripts/generate_graphs.py --transcoder-dir checkpoints
                                      --prompts-file data/prompts/benchmark_prompts.json
                                      --output-dir results/graphs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import MODEL_ID, GraphConfig, load_config
from r1_interp.prompts import BenchmarkPrompt, load_prompts_json
from r1_interp.utils import ensure_dir, graph_filename, require_gpu


def generate_attribution_graph(
    model,
    prompt: BenchmarkPrompt,
    max_feature_nodes: int,
    max_n_logits: int,
    desired_logit_prob: float,
    batch_size: int,
):
    """Generate a single attribution graph for a prompt.

    Returns a circuit_tracer.Graph object.
    """
    from circuit_tracer import attribute

    print(f"  Attributing: {prompt.prompt_id} — {prompt.text[:60]}...")

    graph = attribute(
        prompt.text,
        model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        verbose=True,
    )

    return graph


def export_graph_json(
    graph,
    prompt: BenchmarkPrompt,
    output_dir: Path,
    node_threshold: float,
    edge_threshold: float,
    scan: str | None = None,
) -> Path:
    """Export a Graph to Neuronpedia JSON format.

    Returns the path to the written JSON file.
    """
    from circuit_tracer.utils import create_graph_files

    slug = f"{prompt.category}_{prompt.prompt_id}"
    json_dir = output_dir / "json"
    ensure_dir(json_dir)

    create_graph_files(
        graph,
        slug=slug,
        output_path=json_dir,
        scan=scan,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    json_path = json_dir / f"{slug}.json"
    return json_path


def generate_all_graphs(
    transcoder_dir: str,
    prompts_file: str,
    output_dir: Path,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    max_feature_nodes: int = 7500,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    categories: list[str] | None = None,
) -> None:
    """Generate attribution graphs for all benchmark prompts."""
    import torch

    device = require_gpu()

    try:
        from circuit_tracer import ReplacementModel
    except ImportError:
        print("ERROR: circuit-tracer not installed.")
        sys.exit(1)

    # Load model
    print(f"Loading model {MODEL_ID} with transcoders from {transcoder_dir}...")
    model = ReplacementModel.from_pretrained(MODEL_ID, transcoder_dir, device=device)

    # Load prompts
    prompts = load_prompts_json(prompts_file)
    if categories:
        prompts = [p for p in prompts if p.category in categories]

    print(f"Generating graphs for {len(prompts)} prompts...")

    # Organize output by category
    pt_dir = output_dir / "pt"
    json_dir = output_dir / "json"
    ensure_dir(pt_dir)
    ensure_dir(json_dir)

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt.prompt_id}")

        # Generate attribution graph
        graph = generate_attribution_graph(
            model=model,
            prompt=prompt,
            max_feature_nodes=max_feature_nodes,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
        )

        # Save as .pt
        pt_path = pt_dir / f"{prompt.category}_{prompt.prompt_id}.pt"
        graph.to_pt(str(pt_path))
        print(f"  Saved .pt: {pt_path}")

        # Export to Neuronpedia JSON
        json_path = export_graph_json(
            graph=graph,
            prompt=prompt,
            output_dir=output_dir,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        print(f"  Saved JSON: {json_path}")

        # Compute graph quality scores
        from circuit_tracer.graph import compute_graph_scores

        replacement_score, completeness_score = compute_graph_scores(graph)
        results.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "pt_path": str(pt_path),
                "json_path": str(json_path),
                "replacement_score": replacement_score,
                "completeness_score": completeness_score,
            }
        )
        print(f"  Replacement={replacement_score:.3f}, Completeness={completeness_score:.3f}")

    # Save summary
    summary_path = output_dir / "graph_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Organize JSONs into category subdirectories for circuit-motifs
    _organize_by_category(json_dir, prompts)


def _organize_by_category(json_dir: Path, prompts: list[BenchmarkPrompt]) -> None:
    """Copy JSON files into category subdirectories for circuit-motifs pipeline.

    circuit-motifs expects: data_dir/{category}/*.json
    """
    import shutil

    categories_dir = json_dir / "by_category"
    ensure_dir(categories_dir)

    for prompt in prompts:
        cat_dir = categories_dir / prompt.category
        ensure_dir(cat_dir)
        src = json_dir / f"{prompt.category}_{prompt.prompt_id}.json"
        if src.exists():
            dst = cat_dir / f"{prompt.prompt_id}.json"
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate attribution graphs for benchmark prompts"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--transcoder-dir", default="checkpoints")
    parser.add_argument("--prompts-file", default="data/prompts/benchmark_prompts.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/graphs"))
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--max-feature-nodes", type=int, default=7500)
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        if not isinstance(cfg, GraphConfig):
            print(f"ERROR: Expected GraphConfig, got {type(cfg).__name__}")
            sys.exit(1)
        generate_all_graphs(
            transcoder_dir=cfg.transcoder_dir,
            prompts_file=cfg.prompts_file,
            output_dir=Path(cfg.output_dir),
            node_threshold=cfg.node_threshold,
            edge_threshold=cfg.edge_threshold,
            max_feature_nodes=cfg.max_feature_nodes,
            max_n_logits=cfg.max_n_logits,
            desired_logit_prob=cfg.desired_logit_prob,
            batch_size=cfg.batch_size,
            categories=args.categories,
        )
    else:
        generate_all_graphs(
            transcoder_dir=args.transcoder_dir,
            prompts_file=args.prompts_file,
            output_dir=args.output_dir,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            max_feature_nodes=args.max_feature_nodes,
            categories=args.categories,
        )


if __name__ == "__main__":
    main()
