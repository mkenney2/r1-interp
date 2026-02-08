"""Tests for r1_interp.model_registry — specs and registration."""

from __future__ import annotations

from r1_interp.config import HIDDEN_DIM, N_LAYERS
from r1_interp.model_registry import (
    BASE_QWEN_SPEC,
    R1_DISTILL_SPEC,
    register_r1_models,
    _REGISTRY_AVAILABLE,
    ALL_MODELS,
)


class TestModelSpecs:
    def test_r1_distill_layers(self):
        assert R1_DISTILL_SPEC.n_layers == N_LAYERS == 28

    def test_r1_distill_hidden_dim(self):
        assert R1_DISTILL_SPEC.hidden_dim == HIDDEN_DIM == 1536

    def test_r1_distill_family(self):
        assert R1_DISTILL_SPEC.family == "deepseek-r1"

    def test_base_qwen_layers(self):
        assert BASE_QWEN_SPEC.n_layers == N_LAYERS

    def test_base_qwen_hidden_dim(self):
        assert BASE_QWEN_SPEC.hidden_dim == HIDDEN_DIM

    def test_r1_has_transcoders(self):
        assert len(R1_DISTILL_SPEC.transcoders) >= 1

    def test_base_qwen_no_transcoders(self):
        assert len(BASE_QWEN_SPEC.transcoders) == 0

    def test_transcoder_width(self):
        tc = R1_DISTILL_SPEC.transcoders[0]
        assert tc.width == HIDDEN_DIM * 64

    def test_specs_are_frozen(self):
        try:
            R1_DISTILL_SPEC.n_layers = 999  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass


class TestRegistration:
    def test_register_adds_to_global(self):
        """register_r1_models() should add both specs to ALL_MODELS if circuit-motifs is available."""
        register_r1_models()
        if _REGISTRY_AVAILABLE:
            assert R1_DISTILL_SPEC.model_id in ALL_MODELS
            assert BASE_QWEN_SPEC.model_id in ALL_MODELS
        # If not available, register is a no-op — still passes

    def test_register_is_idempotent(self):
        """Calling register twice doesn't break anything."""
        register_r1_models()
        register_r1_models()
        # Just verify no error
