#!/usr/bin/env python3
"""Phase 3 — Automated interpretability: label features with Claude API.

Reads top-activating examples per feature (from collect_activations.py)
and sends them to the Claude API with a structured prompt asking for a
concise feature label.  Rate-limited with progress saving so it can be
resumed if interrupted.

Usage:
    python scripts/autointerp.py --config configs/autointerp.yaml
    python scripts/autointerp.py --activations-dir results/features/activations
                                 --output-dir results/features/labels
                                 [--layer 14]
                                 [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r1_interp.config import AutointerpConfig, load_config
from r1_interp.utils import ensure_dir

LABELING_PROMPT = """\
You are an expert at interpreting sparse autoencoder / transcoder features \
from large language models.

Below are the top-activating examples for a single feature from layer {layer} \
of DeepSeek-R1-Distill-Qwen-1.5B, a reasoning-distilled language model.

For each example, the feature's activation value is shown, along with the \
surrounding context.  The token that maximally activates the feature is \
typically near the middle of each context window.

{examples_text}

Based on these examples, provide:
1. A concise label (3-8 words) describing what this feature responds to.
2. A confidence score (0.0 - 1.0) for how interpretable this feature is.
3. A brief explanation (1-2 sentences) of the pattern.

Respond in JSON format:
{{"label": "...", "confidence": 0.0, "explanation": "..."}}
"""


def format_examples(examples: list[dict]) -> str:
    """Format top-activating examples for the prompt."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"Example {i} (activation={ex['activation']:.3f}):\n"
            f"  \"{ex['context']}\""
        )
    return "\n\n".join(lines)


async def label_feature(
    client,
    layer: int,
    feature_idx: str,
    examples: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Label a single feature using the Claude API."""
    async with semaphore:
        examples_text = format_examples(examples)
        prompt = LABELING_PROMPT.format(layer=layer, examples_text=examples_text)

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON from response
            # Handle case where Claude wraps in ```json
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            result["feature_idx"] = feature_idx
            result["layer"] = layer
            result["status"] = "success"
            return result

        except json.JSONDecodeError:
            return {
                "feature_idx": feature_idx,
                "layer": layer,
                "label": "parse_error",
                "confidence": 0.0,
                "explanation": f"Could not parse: {text[:200]}",
                "status": "parse_error",
            }
        except Exception as e:
            return {
                "feature_idx": feature_idx,
                "layer": layer,
                "label": "api_error",
                "confidence": 0.0,
                "explanation": str(e),
                "status": "error",
            }


async def label_layer(
    activations_path: Path,
    output_path: Path,
    model: str,
    max_concurrent: int,
    resume: bool = False,
    min_frequency: float = 0.0,
) -> dict:
    """Label all features for a single layer."""
    import anthropic

    with open(activations_path) as f:
        data = json.load(f)

    layer = data["layer"]
    features = data["features"]

    # Load existing results if resuming
    existing: dict[str, dict] = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
        existing = {r["feature_idx"]: r for r in existing_data.get("labels", [])}
        print(f"  Resuming: {len(existing)} features already labeled")

    # Filter to unlabeled features
    to_label = {
        idx: feat
        for idx, feat in features.items()
        if idx not in existing
        and feat.get("activation_frequency", 0) >= min_frequency
    }

    if not to_label:
        print(f"  Layer {layer}: all features already labeled")
        return {"layer": layer, "labels": list(existing.values())}

    print(f"  Layer {layer}: labeling {len(to_label)} features...")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for feat_idx, feat_data in to_label.items():
        examples = feat_data["top_examples"]
        tasks.append(
            label_feature(client, layer, feat_idx, examples, model, semaphore)
        )

    # Process in batches for progress reporting
    results = list(existing.values())
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

        # Progress save
        output_data = {"layer": layer, "labels": results}
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        n_success = sum(1 for r in results if r.get("status") == "success")
        print(
            f"  Progress: {len(results)}/{len(features)} features "
            f"({n_success} successful)"
        )

    return {"layer": layer, "labels": results}


def label_all_layers(
    activations_dir: Path,
    output_dir: Path,
    model: str = "claude-haiku-4-5-20251001",
    max_concurrent: int = 10,
    layer: int | None = None,
    resume: bool = False,
    min_frequency: float = 0.0,
) -> None:
    """Run autointerp labeling on all (or specified) layers."""
    ensure_dir(output_dir)

    # Find activation files
    if layer is not None:
        files = [activations_dir / f"activations_layer{layer}.json"]
    else:
        files = sorted(activations_dir.glob("activations_layer*.json"))

    if not files:
        print(f"No activation files found in {activations_dir}")
        sys.exit(1)

    print(f"Labeling features for {len(files)} layers...")

    for act_file in files:
        out_file = output_dir / act_file.name.replace("activations_", "labels_")
        asyncio.run(
            label_layer(
                act_file, out_file, model, max_concurrent, resume, min_frequency
            )
        )

    # Merge all labels into a single file
    all_labels = []
    for label_file in sorted(output_dir.glob("labels_layer*.json")):
        with open(label_file) as f:
            data = json.load(f)
        all_labels.extend(data.get("labels", []))

    merged_path = output_dir / "all_labels.json"
    with open(merged_path, "w") as f:
        json.dump({"total_features": len(all_labels), "labels": all_labels}, f, indent=2)
    print(f"\nMerged {len(all_labels)} labels -> {merged_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated feature labeling")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--activations-dir",
        type=Path,
        default=Path("results/features/activations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/features/labels"),
    )
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--min-frequency",
        type=float,
        default=0.0,
        help="Skip features with activation frequency below this threshold (e.g. 0.001 = 0.1%%)",
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        if not isinstance(cfg, AutointerpConfig):
            print(f"ERROR: Expected AutointerpConfig, got {type(cfg).__name__}")
            sys.exit(1)
        label_all_layers(
            activations_dir=Path(cfg.output_dir) / "activations",
            output_dir=Path(cfg.output_dir) / "labels",
            model=cfg.claude_model,
            max_concurrent=cfg.max_concurrent_requests,
            layer=args.layer,
            resume=args.resume,
            min_frequency=args.min_frequency,
        )
    else:
        label_all_layers(
            activations_dir=args.activations_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_concurrent=args.max_concurrent,
            layer=args.layer,
            resume=args.resume,
            min_frequency=args.min_frequency,
        )


if __name__ == "__main__":
    main()
