"""Shared utilities: device detection, path helpers, timing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import torch

from r1_interp.config import HIDDEN_DIM, N_LAYERS


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_gpu() -> torch.device:
    """Return a CUDA device or raise if unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required but not available. "
            "Run this script on a GPU instance (RunPod / Lambda)."
        )
    return torch.device("cuda")


@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager that prints elapsed wall-clock time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        tag = f" [{label}]" if label else ""
        print(f"Elapsed{tag}: {elapsed:.2f}s")


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoint_name(layer: int, expansion: int, top_k: int) -> str:
    """Standard checkpoint filename for a transcoder.

    Example: ``transcoder_layer14_exp64_k64.pt``
    """
    return f"transcoder_layer{layer}_exp{expansion}_k{top_k}.pt"


def sweep_run_name(
    layer: int, expansion: int, top_k: int, lr: float
) -> str:
    """Standard wandb run name for a sweep config.

    Example: ``sweep_L14_exp64_k64_lr3e-04``
    """
    return f"sweep_L{layer}_exp{expansion}_k{top_k}_lr{lr:.0e}"


def graph_filename(category: str, prompt_id: str) -> str:
    """Standard filename for an attribution graph JSON.

    Example: ``graph_arithmetic_arith_01.json``
    """
    return f"graph_{category}_{prompt_id}.json"
