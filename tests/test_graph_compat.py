"""Tests for graph JSON compatibility with circuit-motifs.

Verifies that our sample Neuronpedia-format JSON loads correctly in
circuit-motifs' ``load_attribution_graph()`` and can be processed by
``compute_motif_census()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Skip all tests if circuit-motifs not available
CIRCUIT_MOTIFS_DIR = Path(r"C:\Users\mkenn\netowrk-motif-analysis")

try:
    if CIRCUIT_MOTIFS_DIR.exists():
        sys.path.insert(0, str(CIRCUIT_MOTIFS_DIR))
    from src.graph_loader import load_attribution_graph, graph_summary
    from src.motif_census import compute_motif_census

    HAS_CIRCUIT_MOTIFS = True
except ImportError:
    HAS_CIRCUIT_MOTIFS = False

pytestmark = pytest.mark.skipif(
    not HAS_CIRCUIT_MOTIFS,
    reason="circuit-motifs not available",
)


class TestGraphLoading:
    def test_loads_without_error(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        assert g is not None

    def test_is_directed(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        assert g.is_directed()

    def test_has_nodes(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        # 6 nodes total, minus 1 error node (excluded by default) = 5
        assert g.vcount() == 5

    def test_has_edges(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        # 7 edges total, minus any touching the error node
        assert g.ecount() >= 5

    def test_error_nodes_excluded(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        feature_types = set(g.vs["feature_type"])
        assert "mlp reconstruction error" not in feature_types

    def test_has_required_node_attributes(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        required_attrs = {"layer", "feature_type", "influence", "activation"}
        actual_attrs = set(g.vs.attributes())
        assert required_attrs.issubset(actual_attrs)

    def test_has_required_edge_attributes(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        required_attrs = {"weight", "raw_weight", "sign"}
        actual_attrs = set(g.es.attributes())
        assert required_attrs.issubset(actual_attrs)

    def test_layer_parsing(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        layers = set(g.vs["layer"])
        # Embedding "E" -> -1, plus layers 2, 5, 10, 28
        assert -1 in layers  # embedding
        assert 2 in layers

    def test_metadata_stored(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json, include_metadata=True)
        assert g["prompt"] == "What is 2 + 3?"
        assert g["slug"] == "test_graph"

    def test_weight_threshold(self, sample_graph_json):
        g_all = load_attribution_graph(sample_graph_json, weight_threshold=0.0)
        g_high = load_attribution_graph(sample_graph_json, weight_threshold=0.5)
        assert g_high.ecount() <= g_all.ecount()


class TestGraphSummary:
    def test_summary_keys(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        s = graph_summary(g)
        assert "n_nodes" in s
        assert "n_edges" in s
        assert "density" in s
        assert "node_type_counts" in s

    def test_node_type_breakdown(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        s = graph_summary(g)
        types = s["node_type_counts"]
        assert "cross layer transcoder" in types
        assert "embedding" in types
        assert "logit" in types


class TestMotifCensus:
    def test_census_runs(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        result = compute_motif_census(g)
        assert result is not None

    def test_census_has_16_classes(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        result = compute_motif_census(g, size=3)
        assert len(result.raw_counts) == 16

    def test_census_labels(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        result = compute_motif_census(g, size=3)
        assert len(result.labels) == 16
        assert "003" in result.labels  # empty triad always present

    def test_connected_counts(self, sample_graph_json):
        g = load_attribution_graph(sample_graph_json)
        result = compute_motif_census(g, size=3)
        connected = result.connected_counts()
        # Connected triads should only include non-empty classes
        assert "003" not in connected
        # With 5 nodes and 5+ edges, we should have some connected triads
        assert sum(connected.values()) > 0
