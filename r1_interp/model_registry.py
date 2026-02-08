"""Register DeepSeek-R1-Distill-Qwen-1.5B and base Qwen in the circuit-motifs model registry.

Imports ``ModelSpec`` and ``TranscoderConfig`` from the circuit-motifs
``src.models`` module and defines specs for both the R1-distill and
base Qwen models.  ``register_r1_models()`` adds them to the global
``ALL_MODELS`` dict so that the full pipeline can discover them.
"""

from __future__ import annotations

from r1_interp.config import HIDDEN_DIM, MODEL_ID, BASE_QWEN_MODEL_ID, N_LAYERS

# Lazy import from circuit-motifs — may not be installed in all envs
_REGISTRY_AVAILABLE = False

try:
    from src.models import ModelSpec, TranscoderConfig, ALL_MODELS

    _REGISTRY_AVAILABLE = True
except ImportError:
    # Provide stub classes so the module is still importable without
    # circuit-motifs on sys.path.
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TranscoderConfig:  # type: ignore[no-redef]
        hf_repo: str
        transcoder_folder: str = "transcoder_all"
        width: int = 16384
        is_clt: bool = False

    @dataclass(frozen=True)
    class ModelSpec:  # type: ignore[no-redef]
        model_id: str
        family: str
        variant: str
        n_params: int
        n_layers: int
        hidden_dim: int
        transcoders: tuple = ()
        neuronpedia_id: str | None = None

    ALL_MODELS: dict[str, ModelSpec] = {}  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# R1-distill spec
# ---------------------------------------------------------------------------
R1_DISTILL_SPEC = ModelSpec(
    model_id="r1-distill-qwen-1.5b",
    family="deepseek-r1",
    variant="qwen-1.5b-distill",
    n_params=1500,
    n_layers=N_LAYERS,
    hidden_dim=HIDDEN_DIM,
    transcoders=(
        TranscoderConfig(
            # Placeholder — will be updated once transcoders are trained & uploaded
            hf_repo="r1-interp/r1-distill-qwen-1.5b-transcoders",
            transcoder_folder="transcoder_all",
            width=HIDDEN_DIM * 64,  # default expansion=64; updated after sweep
            is_clt=False,
        ),
    ),
    neuronpedia_id="r1-distill-qwen-1.5b",
)

# ---------------------------------------------------------------------------
# Base Qwen spec (for comparison)
# ---------------------------------------------------------------------------
BASE_QWEN_SPEC = ModelSpec(
    model_id="qwen2.5-1.5b",
    family="qwen-2.5",
    variant="base",
    n_params=1500,
    n_layers=N_LAYERS,
    hidden_dim=HIDDEN_DIM,
    transcoders=(),  # no transcoders trained yet for base
    neuronpedia_id="qwen2.5-1.5b",
)


def register_r1_models() -> None:
    """Add R1-distill and base Qwen specs to the circuit-motifs global registry.

    No-op if circuit-motifs is not importable.
    """
    if not _REGISTRY_AVAILABLE:
        return
    ALL_MODELS[R1_DISTILL_SPEC.model_id] = R1_DISTILL_SPEC
    ALL_MODELS[BASE_QWEN_SPEC.model_id] = BASE_QWEN_SPEC
