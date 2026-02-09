"""Configuration dataclasses for the R1-Interp training and analysis pipeline."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B"
N_LAYERS = 28
HIDDEN_DIM = 1536


# ---------------------------------------------------------------------------
# Transcoder hyperparameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TranscoderHyperparams:
    """Hyperparameters for a single transcoder training run."""

    expansion_factor: int = 64
    top_k: int = 64
    skip_connection: bool = True
    learning_rate: float = 3e-4
    training_tokens: int = 200_000_000  # 200M
    batch_size: int = 8  # micro-batch sequences (use grad_acc for effective batch)
    grad_acc_steps: int = 4  # effective batch = batch_size * grad_acc_steps
    warmup_steps: int = 1000
    weight_decay: float = 0.0

    @property
    def dict_size(self) -> int:
        return self.expansion_factor * HIDDEN_DIM

    @property
    def tag(self) -> str:
        return (
            f"exp{self.expansion_factor}_k{self.top_k}"
            f"_lr{self.learning_rate:.0e}"
        )


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepConfig:
    """Hyperparameter sweep grid for pilot layers."""

    layers: tuple[int, ...] = (2, 14, 26)
    expansion_factors: tuple[int, ...] = (32, 64, 128)
    top_k_values: tuple[int, ...] = (32, 64, 128)
    skip_connection: bool = True
    learning_rate: float = 3e-4
    training_tokens: int = 200_000_000
    batch_size: int = 8  # micro-batch sequences (use grad_acc for effective batch)
    grad_acc_steps: int = 4  # effective batch = batch_size * grad_acc_steps
    warmup_steps: int = 1000
    dataset: str = "Skylion007/openwebtext"
    wandb_project: str = "r1-interp-sweep"

    def generate_configs(self) -> list[dict[str, Any]]:
        """Generate all (expansion, topk) sweep combinations.

        Returns one config dict per (expansion_factor, top_k) pair.
        Each config is run on every layer in ``self.layers``.
        """
        configs: list[dict[str, Any]] = []
        for exp, k in itertools.product(self.expansion_factors, self.top_k_values):
            hp = TranscoderHyperparams(
                expansion_factor=exp,
                top_k=k,
                skip_connection=self.skip_connection,
                learning_rate=self.learning_rate,
                training_tokens=self.training_tokens,
                batch_size=self.batch_size,
                grad_acc_steps=self.grad_acc_steps,
                warmup_steps=self.warmup_steps,
            )
            configs.append(
                {
                    "layers": list(self.layers),
                    "hyperparams": asdict(hp),
                    "dataset": self.dataset,
                    "wandb_project": self.wandb_project,
                    "run_name": f"sweep_{hp.tag}",
                }
            )
        return configs


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainRunConfig:
    """Configuration for training transcoders across all 28 layers."""

    hyperparams: TranscoderHyperparams = field(
        default_factory=TranscoderHyperparams
    )
    layers: tuple[int, ...] = tuple(range(N_LAYERS))
    dataset: str = "Skylion007/openwebtext"
    wandb_project: str = "r1-interp-train"
    checkpoint_dir: str = "checkpoints"
    distribute_modules: bool = True
    num_gpus: int = 8

    @property
    def run_name(self) -> str:
        return f"train_full_{self.hyperparams.tag}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    """Configuration for faithfulness evaluation."""

    model_id: str = MODEL_ID
    transcoder_dir: str = "checkpoints"
    eval_dataset: str = "open-r1/OpenR1-Math-220k"
    eval_tokens: int = 10_000_000
    ce_threshold: float = 0.1  # nats
    dead_feature_threshold: float = 0.10  # 10%
    batch_size: int = 32
    math_prompts_file: str = "data/prompts/benchmark_prompts.json"


# ---------------------------------------------------------------------------
# Automated interpretability
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AutointerpConfig:
    """Configuration for Claude-based feature labeling."""

    claude_model: str = "claude-sonnet-4-5-20250929"
    examples_per_feature: int = 20
    min_activation_freq: float = 0.001  # 0.1%
    batch_size: int = 50
    max_concurrent_requests: int = 10
    output_dir: str = "results/features"
    transcoder_dir: str = "checkpoints"
    corpus_dataset: str = "open-r1/OpenR1-Math-220k"
    corpus_tokens: int = 50_000_000


# ---------------------------------------------------------------------------
# Attribution graph generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GraphConfig:
    """Configuration for circuit-tracer attribution graphs."""

    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    max_feature_nodes: int = 7500
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    batch_size: int = 512
    prompts_file: str = "data/prompts/benchmark_prompts.json"
    output_dir: str = "results/graphs"
    transcoder_dir: str = "checkpoints"


# ---------------------------------------------------------------------------
# YAML serialization helpers
# ---------------------------------------------------------------------------

# Map from config type name to class
_CONFIG_CLASSES: dict[str, type] = {
    "TranscoderHyperparams": TranscoderHyperparams,
    "SweepConfig": SweepConfig,
    "TrainRunConfig": TrainRunConfig,
    "EvalConfig": EvalConfig,
    "AutointerpConfig": AutointerpConfig,
    "GraphConfig": GraphConfig,
}


def _tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists for safe YAML serialization."""
    if isinstance(obj, (tuple, list)):
        return [_tuples_to_lists(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    return obj


def save_config(config: Any, path: str | Path) -> None:
    """Serialize a config dataclass to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _tuples_to_lists({"type": type(config).__name__, **asdict(config)})
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(path: str | Path) -> Any:
    """Deserialize a config dataclass from YAML."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    type_name = data.pop("type", None)
    if type_name is None:
        raise ValueError(f"YAML config at {path} missing 'type' field")
    cls = _CONFIG_CLASSES.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown config type '{type_name}'. "
            f"Known types: {list(_CONFIG_CLASSES)}"
        )
    # Convert lists back to tuples for frozen dataclass fields
    if cls is SweepConfig:
        for key in ("layers", "expansion_factors", "top_k_values"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
    if cls is TrainRunConfig:
        if "layers" in data and isinstance(data["layers"], list):
            data["layers"] = tuple(data["layers"])
        if "hyperparams" in data and isinstance(data["hyperparams"], dict):
            data["hyperparams"] = TranscoderHyperparams(**data["hyperparams"])
    return cls(**data)
