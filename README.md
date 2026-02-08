# R1-Interp

**Interpretability toolkit for DeepSeek-R1-Distill-Qwen-1.5B** — the first complete circuit-analysis pipeline for a reasoning-distilled language model.

R1-Interp trains [skip transcoders](https://arxiv.org/abs/2406.11944) on all 28 MLP layers of R1-Distill-Qwen-1.5B, generates Anthropic-style [attribution graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), and runs [network motif analysis](https://en.wikipedia.org/wiki/Network_motif) to compare circuit structure between reasoning and non-reasoning models.

## Why This Project

Reasoning-distilled models like DeepSeek-R1 represent a new paradigm in LLM capabilities, but we don't yet understand *how* reasoning distillation changes a model's internal circuitry. R1-Interp aims to answer:

- Does reasoning distillation change circuit structure, or just feature usage?
- Are there distinct motif signatures in `<think>` reasoning traces vs. final answers?
- How do R1-distill circuits compare to base Qwen and Claude Haiku?

## Pipeline Overview

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `setup_env.py`, `prepare_data.py` | Environment setup, data download & tokenization |
| 1 | `sweep.py` | Hyperparameter sweep on 3 pilot layers (2, 14, 26) |
| 2 | `train.py` | Full transcoder training across all 28 layers |
| 3 | `collect_activations.py`, `autointerp.py` | Feature activation collection & Claude API labeling |
| 4 | `generate_graphs.py`, `motif_analysis.py`, `compare_models.py` | Attribution graphs, motif census, cross-model comparison |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Verify environment
python scripts/setup_env.py

# Run tests
pytest tests/
```

## Project Structure

```
r1_interp/                  # Core package
├── config.py               # Frozen dataclasses for all pipeline configs
├── prompts.py              # 24 benchmark prompts across 7 categories
├── model_registry.py       # R1-distill & base Qwen specs for circuit-motifs
└── utils.py                # Device detection, path helpers, timing

configs/                    # YAML training configs
├── sweep_pilot.yaml        # Phase 1: 3×3 sweep grid (expansion × topk)
├── train_full.yaml         # Phase 2: all 28 layers (update after sweep)
├── autointerp.yaml         # Phase 3: Claude API feature labeling
└── graph_generation.yaml   # Phase 4: attribution graph parameters

scripts/                    # One script per pipeline stage
├── setup_env.py            # Check deps, clone sparsify, download model
├── prepare_data.py         # Pre-tokenize OpenR1-Math corpus
├── train.py                # Thin wrapper around sparsify (CLI + Python API)
├── sweep.py                # Grid search with Pareto-optimal selection
├── evaluate.py             # CE loss increase, dead features, L0
├── collect_activations.py  # Top-N activating examples per feature
├── autointerp.py           # Rate-limited Claude API labeling with resume
├── generate_graphs.py      # circuit-tracer attribution + Neuronpedia export
├── motif_analysis.py       # circuit-motifs pipeline wrapper
└── compare_models.py       # R1 vs Qwen + think-vs-answer comparison

data/prompts/
└── benchmark_prompts.json  # 24 prompts: arithmetic, multihop, reasoning,
                            # factual_recall, safety, reasoning_failure, baseline

tests/                      # 47 tests
├── test_config.py          # Config properties, sweep generation, YAML roundtrip
├── test_prompts.py         # Unique IDs, category coverage, JSON roundtrip
├── test_model_registry.py  # Model specs, circuit-motifs registration
└── test_graph_compat.py    # Neuronpedia JSON ↔ circuit-motifs compatibility
```

## Key Dependencies

| Dependency | Purpose |
|-----------|---------|
| [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) | Transcoder training framework |
| [circuit-tracer](https://github.com/anthropics/circuit-tracer) | Attribution graph generation |
| [circuit-motifs](https://github.com/mkenney2/network-motif-analysis) | Network motif analysis for circuit comparison |
| [wandb](https://wandb.ai) | Experiment tracking |

## Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Expansion factor | 32×, 64×, 128× | Sweep candidates (of hidden dim 1536) |
| TopK | 32, 64, 128 | Sparsity parameter |
| Skip connection | Yes | Dunefsky et al. skip transcoders |
| Training tokens | 200M | Per layer |
| Faithfulness target | <0.1 nats CE increase | Ideally <0.05 |
| Dead features target | <10% | |

## Benchmark Prompts

24 prompts across 7 categories designed for circuit comparison:

- **arithmetic** (5) — addition, multiplication, algebra, order of operations
- **multihop** (4) — Dallas→Texas→Austin style multi-step retrieval
- **reasoning** (4) — snail problem, syllogism, missing dollar, bat-and-ball
- **factual_recall** (4) — simple facts for baseline comparison
- **safety** (3) — refusal triggers and boundary cases
- **reasoning_failure** (2) — problems the model is likely to get wrong
- **baseline** (2) — text completion (no reasoning required)

## Planned Comparisons

1. **R1-distill vs. base Qwen-1.5B** — Does reasoning distillation change circuit wiring?
2. **`<think>` phase vs. answer phase** — Are reasoning traces structurally different?
3. **R1-distill vs. Claude Haiku** — Cross-architecture motif comparison
4. **Correct vs. incorrect reasoning** — Circuit signatures of failure

## License

MIT
