"""Shared fixtures for R1-Interp tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Also add circuit-motifs for graph_compat tests
CIRCUIT_MOTIFS_DIR = Path(r"C:\Users\mkenn\netowrk-motif-analysis")
if CIRCUIT_MOTIFS_DIR.exists():
    sys.path.insert(0, str(CIRCUIT_MOTIFS_DIR))


@pytest.fixture
def tmp_yaml(tmp_path):
    """Return a callable that writes YAML to a temp file and returns the path."""
    import yaml

    def _write(data: dict, name: str = "test.yaml") -> Path:
        p = tmp_path / name
        with open(p, "w") as f:
            yaml.dump(data, f)
        return p

    return _write


@pytest.fixture
def sample_graph_json(tmp_path) -> Path:
    """Create a minimal Neuronpedia-format attribution graph JSON.

    This fixture creates a graph with the exact schema that
    circuit-motifs ``load_attribution_graph()`` expects.
    """
    graph_data = {
        "metadata": {
            "slug": "test_graph",
            "scan": "r1-distill-qwen-1.5b/transcoder_all",
            "prompt": "What is 2 + 3?",
            "prompt_tokens": [1, 2, 3, 4, 5],
            "node_threshold": 0.8,
            "schema_version": 1,
        },
        "qParams": {
            "pinnedIds": "",
            "supernodes": "",
            "linkType": "",
            "clickedId": "",
            "sg_pos": "",
        },
        "nodes": [
            {
                "node_id": "embed_0",
                "feature_type": "embedding",
                "layer": "E",
                "ctx_idx": 0,
                "feature": 0,
                "clerp": "What",
                "activation": 1.0,
                "influence": 0.5,
                "is_target_logit": False,
                "token_prob": 0.0,
            },
            {
                "node_id": "feature_L2_F100",
                "feature_type": "cross layer transcoder",
                "layer": 2,
                "ctx_idx": 1,
                "feature": 100,
                "clerp": "",
                "activation": 3.5,
                "influence": 0.8,
                "is_target_logit": False,
                "token_prob": 0.0,
            },
            {
                "node_id": "feature_L5_F200",
                "feature_type": "cross layer transcoder",
                "layer": 5,
                "ctx_idx": 2,
                "feature": 200,
                "clerp": "",
                "activation": 2.1,
                "influence": 0.6,
                "is_target_logit": False,
                "token_prob": 0.0,
            },
            {
                "node_id": "feature_L10_F50",
                "feature_type": "cross layer transcoder",
                "layer": 10,
                "ctx_idx": 3,
                "feature": 50,
                "clerp": "",
                "activation": 1.8,
                "influence": 0.4,
                "is_target_logit": False,
                "token_prob": 0.0,
            },
            {
                "node_id": "error_L2",
                "feature_type": "mlp reconstruction error",
                "layer": 2,
                "ctx_idx": 1,
                "feature": None,
                "clerp": "",
                "activation": 0.1,
                "influence": 0.02,
                "is_target_logit": False,
                "token_prob": 0.0,
            },
            {
                "node_id": "logit_5",
                "feature_type": "logit",
                "layer": 28,
                "ctx_idx": -1,
                "feature": 5,
                "clerp": "5",
                "activation": 0.0,
                "influence": 1.0,
                "is_target_logit": True,
                "token_prob": 0.95,
            },
        ],
        "links": [
            {"source": "embed_0", "target": "feature_L2_F100", "weight": 0.6},
            {"source": "feature_L2_F100", "target": "feature_L5_F200", "weight": 0.4},
            {"source": "feature_L5_F200", "target": "logit_5", "weight": 0.8},
            {"source": "feature_L2_F100", "target": "logit_5", "weight": 0.3},
            {"source": "embed_0", "target": "feature_L5_F200", "weight": 0.2},
            {"source": "feature_L10_F50", "target": "logit_5", "weight": 0.5},
            {"source": "feature_L2_F100", "target": "feature_L10_F50", "weight": 0.15},
        ],
    }

    json_path = tmp_path / "test_graph.json"
    with open(json_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    return json_path
