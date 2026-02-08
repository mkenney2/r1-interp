#!/usr/bin/env python3
"""Phase 0 — Verify environment, download model, clone sparsify.

Run locally before provisioning GPU compute to catch missing deps early.

Usage:
    python scripts/setup_env.py [--download-model] [--clone-sparsify]
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

# Packages to check: (import_name, pip_name, required)
REQUIRED_PACKAGES = [
    ("torch", "torch", True),
    ("transformers", "transformers", True),
    ("safetensors", "safetensors", True),
    ("yaml", "pyyaml", True),
    ("wandb", "wandb", True),
    ("datasets", "datasets", True),
    ("numpy", "numpy", True),
    ("tqdm", "tqdm", True),
]

OPTIONAL_PACKAGES = [
    ("circuit_tracer", "circuit-tracer", False),
    ("igraph", "python-igraph", False),
    ("anthropic", "anthropic", False),
]

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SPARSIFY_REPO = "https://github.com/EleutherAI/sparsify.git"
SPARSIFY_DIR = Path(__file__).resolve().parent.parent / "sparsify"


def check_packages() -> tuple[list[str], list[str]]:
    """Check for required and optional packages. Returns (ok, missing) lists."""
    ok, missing = [], []
    for import_name, pip_name, required in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            ok.append(f"  [OK]  {pip_name} ({version})")
        except ImportError:
            tag = "MISSING" if required else "optional"
            missing.append(f"  [{tag}] {pip_name}")
    return ok, missing


def check_gpu() -> str:
    """Return GPU info string."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            return f"  GPU: {name} ({mem:.1f} GB)"
        return "  GPU: None (CPU-only torch)"
    except Exception as e:
        return f"  GPU: Error checking — {e}"


def check_circuit_motifs() -> str:
    """Check if circuit-motifs is importable."""
    try:
        from src.models import ALL_MODELS

        return f"  [OK]  circuit-motifs ({len(ALL_MODELS)} models in registry)"
    except ImportError:
        return "  [skip] circuit-motifs not on sys.path"


def download_model() -> None:
    """Download model weights via huggingface_hub."""
    from huggingface_hub import snapshot_download

    print(f"\nDownloading {MODEL_ID}...")
    path = snapshot_download(MODEL_ID)
    print(f"  Downloaded to: {path}")


def sanity_check_model() -> None:
    """Quick inference check — load model and generate a few tokens."""
    print(f"\nSanity-checking {MODEL_ID}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto"
    )
    inputs = tokenizer("What is 2 + 3?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: 'What is 2 + 3?'")
    print(f"  Output: {text[:200]}")
    print("  [OK] Model loaded and generated output.")


def clone_sparsify() -> None:
    """Clone EleutherAI/sparsify if not already present."""
    if SPARSIFY_DIR.exists():
        print(f"\n  sparsify already cloned at {SPARSIFY_DIR}")
        return
    print(f"\nCloning {SPARSIFY_REPO} -> {SPARSIFY_DIR}...")
    subprocess.run(
        ["git", "clone", SPARSIFY_REPO, str(SPARSIFY_DIR)],
        check=True,
    )
    print(f"  [OK] Cloned to {SPARSIFY_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R1-Interp environment setup")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the R1-Distill model from HuggingFace",
    )
    parser.add_argument(
        "--clone-sparsify",
        action="store_true",
        help="Clone EleutherAI/sparsify repo",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a quick inference test on the model",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("R1-Interp Environment Check")
    print("=" * 60)

    # 1. Packages
    print("\nPackages:")
    ok, missing = check_packages()
    for line in ok:
        print(line)
    for line in missing:
        print(line)

    # 2. GPU
    print("\nCompute:")
    print(check_gpu())

    # 3. Circuit-motifs
    print("\nIntegrations:")
    print(check_circuit_motifs())

    # 4. Sparsify
    if SPARSIFY_DIR.exists():
        print(f"  [OK]  sparsify cloned at {SPARSIFY_DIR}")
    else:
        print(f"  [skip] sparsify not cloned (use --clone-sparsify)")

    # 5. Git
    git_path = shutil.which("git")
    print(f"  [{'OK' if git_path else 'MISSING'}]  git: {git_path or 'not found'}")

    # Optional actions
    if args.clone_sparsify:
        clone_sparsify()
    if args.download_model:
        download_model()
    if args.sanity_check:
        sanity_check_model()

    print("\n" + "=" * 60)
    required_missing = [m for m in missing if "MISSING" in m]
    if required_missing:
        print(f"RESULT: {len(required_missing)} required package(s) missing.")
        sys.exit(1)
    else:
        print("RESULT: All required packages present. Ready for training.")


if __name__ == "__main__":
    main()
