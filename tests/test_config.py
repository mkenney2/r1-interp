"""Tests for r1_interp.config — dataclass properties, sweep generation, YAML roundtrip."""

from __future__ import annotations

from r1_interp.config import (
    HIDDEN_DIM,
    SweepConfig,
    TrainRunConfig,
    TranscoderHyperparams,
    load_config,
    save_config,
)


class TestTranscoderHyperparams:
    def test_dict_size(self):
        hp = TranscoderHyperparams(expansion_factor=64)
        assert hp.dict_size == 64 * HIDDEN_DIM

    def test_tag_format(self):
        hp = TranscoderHyperparams(expansion_factor=32, top_k=128, learning_rate=3e-4)
        assert "exp32" in hp.tag
        assert "k128" in hp.tag

    def test_frozen(self):
        hp = TranscoderHyperparams()
        try:
            hp.expansion_factor = 999  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestSweepConfig:
    def test_default_generates_9_configs(self):
        """3 expansion factors x 3 top_k values = 9 configs."""
        sc = SweepConfig()
        configs = sc.generate_configs()
        assert len(configs) == 9

    def test_configs_have_required_keys(self):
        sc = SweepConfig()
        for cfg in sc.generate_configs():
            assert "layers" in cfg
            assert "hyperparams" in cfg
            assert "dataset" in cfg
            assert "wandb_project" in cfg
            assert "run_name" in cfg

    def test_all_combos_present(self):
        sc = SweepConfig(
            expansion_factors=(32, 64),
            top_k_values=(32, 64),
        )
        configs = sc.generate_configs()
        combos = {
            (c["hyperparams"]["expansion_factor"], c["hyperparams"]["top_k"])
            for c in configs
        }
        assert combos == {(32, 32), (32, 64), (64, 32), (64, 64)}

    def test_custom_layers(self):
        sc = SweepConfig(layers=(0, 5))
        for cfg in sc.generate_configs():
            assert cfg["layers"] == [0, 5]


class TestYAMLRoundtrip:
    def test_sweep_config_roundtrip(self, tmp_yaml):
        original = SweepConfig(
            layers=(2, 14, 26),
            expansion_factors=(32, 64),
            top_k_values=(64,),
        )
        path = tmp_yaml(
            {
                "type": "SweepConfig",
                "layers": [2, 14, 26],
                "expansion_factors": [32, 64],
                "top_k_values": [64],
                "skip_connection": True,
                "learning_rate": 3e-4,
                "training_tokens": 200_000_000,
                "batch_size": 4096,
                "warmup_steps": 1000,
                "dataset": "open-r1/OpenR1-Math-220k",
                "wandb_project": "r1-interp-sweep",
            }
        )
        loaded = load_config(path)
        assert isinstance(loaded, SweepConfig)
        assert loaded.layers == (2, 14, 26)
        assert loaded.expansion_factors == (32, 64)

    def test_save_and_load(self, tmp_path):
        original = SweepConfig()
        path = tmp_path / "test_sweep.yaml"
        save_config(original, path)
        loaded = load_config(path)
        assert isinstance(loaded, SweepConfig)
        assert loaded.layers == original.layers
        assert len(loaded.generate_configs()) == len(original.generate_configs())

    def test_train_run_config_roundtrip(self, tmp_path):
        hp = TranscoderHyperparams(expansion_factor=128, top_k=32)
        original = TrainRunConfig(hyperparams=hp, layers=(0, 1, 2))
        path = tmp_path / "train.yaml"
        save_config(original, path)
        loaded = load_config(path)
        assert isinstance(loaded, TrainRunConfig)
        assert loaded.hyperparams.expansion_factor == 128
        assert loaded.hyperparams.top_k == 32
        assert loaded.layers == (0, 1, 2)

    def test_unknown_type_raises(self, tmp_yaml):
        path = tmp_yaml({"type": "NonexistentConfig", "foo": "bar"})
        try:
            load_config(path)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Unknown config type" in str(e)

    def test_missing_type_raises(self, tmp_yaml):
        path = tmp_yaml({"foo": "bar"})
        try:
            load_config(path)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "missing 'type'" in str(e)
