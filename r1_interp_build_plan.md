# R1-Interp-1.5B: Full Interpretability Toolkit Build Plan

**Goal:** Build and release the first complete interpretability toolkit for a reasoning model (DeepSeek-R1-Distill-Qwen-1.5B), enabling Anthropic-style circuit analysis on a model that can run on consumer hardware.

**Estimated total cost:** $550–$1,200
**Estimated timeline:** 4–6 weeks of part-time work

---

## Phase 0: Setup & Infrastructure (Days 1–3)

### 0.1 Environment
- [ ] Set up RunPod or Lambda account with billing
- [ ] Provision an 8x A100 node (or start with 1x A100 for prototyping)
- [ ] Create a GitHub repo: `r1-interp` (or similar)
- [ ] Set up wandb for experiment tracking

### 0.2 Codebase
- [ ] Clone EleutherAI's `clt-training` (formerly `sparsify`) repo
- [ ] Verify it runs on a known model (e.g., Llama-3.2-1B) with a quick test transcoder training
- [ ] Clone Anthropic's `circuit-tracer` repo
- [ ] Verify circuit tracer runs on a known model
- [ ] Download DeepSeek-R1-Distill-Qwen-1.5B from HuggingFace
- [ ] Verify the model runs inference correctly (sanity check: give it a math problem, confirm it produces reasoning traces)

### 0.3 Data
- [ ] Select a training dataset — reasoning-heavy is ideal since this is a reasoning model
  - Options: OpenR1-Math, LMSYS-Chat-1M, or a mix of reasoning + general text
  - Aim for ~200M–500M tokens for training, plus a held-out eval set
- [ ] Pre-tokenize the dataset with the Qwen tokenizer to avoid tokenization bottleneck during training

**Estimated cost:** ~$20 (a few hours of GPU for setup/testing)

---

## Phase 1: Hyperparameter Sweep on 2–3 Layers (Days 4–10)

Don't train all 28 layers yet. First, find good hyperparameters on a small subset.

### 1.1 Pick pilot layers
- [ ] Select 3 layers: one early (layer 2), one middle (layer 14), one late (layer 26)
- [ ] These will be your development layers for all tuning

### 1.2 Key hyperparameters to sweep
- [ ] **Expansion factor**: 32x, 64x, 128x (ratio of transcoder features to hidden dim 1536)
  - 32x = 49,152 features per layer
  - 64x = 98,304 features per layer
  - 128x = 196,608 features per layer
  - Start with 64x as a reasonable default
- [ ] **Sparsity (k for TopK)**: 32, 64, 128
  - Controls how many features activate per input
- [ ] **Learning rate**: 1e-4, 3e-4, 1e-3
- [ ] **Training tokens**: 100M, 200M, 500M
  - Plot loss curves to find where returns diminish
- [ ] **Skip connection** (skip transcoder): yes/no
  - Dunefsky et al. showed skip transcoders are a Pareto improvement

### 1.3 Evaluation metrics for each config
- [ ] **Faithfulness**: Cross-entropy loss increase when substituting transcoder for MLP
  - Target: <0.1 nats increase (ideally <0.05)
- [ ] **Sparsity (L0)**: Average number of active features per input
  - Lower is better for interpretability, but too low hurts faithfulness
- [ ] **Dead features**: Percentage of features that never activate
  - Target: <10% dead features
- [ ] **Autointerp score** (on a small sample): Run automated interpretability on ~100 features from each config
  - This tells you if the features are actually interpretable, not just faithful

### 1.4 Select best config
- [ ] Plot faithfulness vs. sparsity Pareto frontier
- [ ] Pick the config that gives best faithfulness at reasonable sparsity
- [ ] Document the decision and reasoning in the repo

**Estimated cost:** ~$200–$400 (multiple training runs on 3 layers)

---

## Phase 2: Full Transcoder Training (Days 11–16)

### 2.1 Train all 28 layers
- [ ] Using the best hyperparameters from Phase 1, train transcoders for all 28 layers
- [ ] Use `--distribute_modules` flag to parallelize across GPUs
  - On 8x A100: can train 7 layers per GPU simultaneously (28/4 = 7 per pass with some overlap)
  - Or 4 layers per GPU with 2 passes if memory is tight
- [ ] Monitor training with wandb: loss curves, L0, dead features per layer

### 2.2 Validate
- [ ] Measure faithfulness on held-out eval set for each layer individually
- [ ] Measure **full-model faithfulness**: replace ALL 28 MLPs with transcoders simultaneously
  - This is the real test — how much does the model degrade with the full replacement model?
  - Anthropic reports matching outputs ~50% of the time for their CLTs on Haiku
  - For per-layer transcoders on a 1.5B, aim for reasonable degradation
- [ ] Run a qualitative check: give the replacement model a few math problems and compare outputs to the original

### 2.3 Save and upload
- [ ] Save all 28 transcoder checkpoints
- [ ] Upload to HuggingFace with proper model cards documenting:
  - Training config (hyperparameters, dataset, tokens)
  - Faithfulness metrics per layer and full-model
  - How to load and use them
  - License information

**Estimated cost:** ~$200–$400

---

## Phase 3: Feature Visualization & Autointerp (Days 17–22)

### 3.1 Collect activations
- [ ] Run the full replacement model on a diverse corpus (~100M tokens)
  - Include: math/reasoning problems, general text, code, safety-relevant prompts
- [ ] For each feature, record:
  - Top activating examples (tokens + surrounding context)
  - Activation frequency (what % of tokens activate this feature?)
  - Max activation values
  - Layer and position statistics

### 3.2 Automated interpretability
- [ ] Run autointerp on the most active/important features
  - Start with features that activate on >0.1% of tokens
  - Use Claude API (or GPT-4) to generate natural language descriptions
  - Format: show the model 10-20 top activating examples, ask it to describe what the feature detects
- [ ] Estimate: ~10,000–50,000 features worth labeling
  - At ~$0.005 per feature (a few hundred tokens of prompt/response): $50–$250
- [ ] Save labels as CSV/JSON and as a SQL database (following Goodfire's format for compatibility)

### 3.3 Quality check
- [ ] Manually inspect 50–100 feature labels across layers
  - Are early layer features detecting token-level patterns?
  - Are late layer features detecting abstract reasoning concepts?
  - Are there features specific to reasoning (e.g., "backtracking", "verification", "step enumeration")?
- [ ] Flag and document any interesting reasoning-specific features you find — these become content for the blog post

**Estimated cost:** ~$80–$300 (GPU for activation collection + API credits for autointerp)

---

## Phase 4: Attribution Graph Generation (Days 23–28)

### 4.1 Set up circuit tracer
- [ ] Integrate your transcoders with Anthropic's circuit tracer library
  - The circuit tracer expects a specific format — document any adapter code needed
  - This adapter code is itself a valuable contribution

### 4.2 Generate attribution graphs for benchmark prompts
- [ ] Create a curated prompt set covering:
  - **Math reasoning**: "What is 47 + 86?", multi-step arithmetic
  - **Multi-hop factual**: "The capital of the state containing Dallas is ___" (for direct comparison with Anthropic's work)
  - **Chain-of-thought**: Problems where R1 produces extended reasoning traces
  - **Reasoning failures**: Prompts where the model gets the answer wrong (interesting to compare circuits)
  - **Safety**: "How do I make a bomb?" (for comparison with your existing circuit-motifs analysis)
  - **General/baseline**: Simple factual recall, text completion
- [ ] Generate and save attribution graphs for each prompt
- [ ] Export in Neuronpedia-compatible format

### 4.3 Run circuit-motifs
- [ ] Run your motif analysis tool on all generated graphs
- [ ] Compare R1-distill motif profiles to:
  - Your existing 99-graph dataset (Claude Haiku)
  - Gemma results from your scaling study (if completed by this point)
  - Base Qwen-1.5B (non-reasoning) on the same prompts — this is the key comparison
- [ ] Trace top FFLs in reasoning circuits and identify processing stages
  - Do reasoning models show different cascade structures?
  - Are there motif signatures unique to chain-of-thought computation?

### 4.4 Upload graphs to Neuronpedia
- [ ] Work with Johnny Lin to get R1-distill-Qwen-1.5B added as a model on Neuronpedia
- [ ] Upload attribution graphs for the benchmark prompts
- [ ] Upload feature visualizations

**Estimated cost:** ~$30–$60

---

## Phase 5: Write-Up & Release (Days 29–35)

### 5.1 Package the release
- [ ] GitHub repo with:
  - Training scripts and configs (fully reproducible)
  - Transcoder weights (or HuggingFace links)
  - Feature labels database
  - Pre-computed attribution graphs
  - Tutorial notebook: "From prompt to circuit analysis in 5 minutes"
  - Circuit-motifs analysis scripts
  - Cost breakdown and lessons learned
- [ ] HuggingFace model card with all metrics
- [ ] Neuronpedia integration

### 5.2 Write the blog post
Structure:
1. **Why this matters**: No one has done circuit analysis on a reasoning model. Here's the first toolkit.
2. **How I built it**: Training decisions, faithfulness results, cost breakdown (~$X total). Anyone can reproduce this.
3. **What I found inside**:
   - Interesting reasoning-specific features (from autointerp)
   - Motif profile comparison: reasoning model vs. standard model
   - FFL cascade analysis on reasoning traces — how does the model "think through" a math problem?
   - Any structural differences from Anthropic's findings on Claude Haiku
4. **The toolkit**: Here's how to use it. One notebook, consumer GPU, full circuit analysis.

### 5.3 Publish and share
- [ ] Post on Substack
- [ ] Twitter thread with key figures
- [ ] Share with Anthropic interp team, Goodfire, Johnny Lin at Neuronpedia
- [ ] Submit to EA Forum
- [ ] Try LessWrong again (by this point you have community traction)
- [ ] Consider arXiv preprint if results are strong

---

## Phase 6 (Stretch): Scale to 8B (Optional, Days 36+)

If the 1.5B toolkit works well and you want to go further:

- [ ] Train transcoders for R1-Distill-Llama-8B using the same pipeline
- [ ] Compare 1.5B vs 8B circuit structures (reasoning scaling laws!)
- [ ] This becomes Project 4 / the reasoning deep dive

**Additional cost for 8B:** ~$500–$1,500

---

## Budget Summary

| Phase | Cost Estimate |
|-------|--------------|
| 0: Setup & Infrastructure | ~$20 |
| 1: Hyperparameter Sweep | $200–$400 |
| 2: Full Training | $200–$400 |
| 3: Feature Viz & Autointerp | $80–$300 |
| 4: Attribution Graphs | $30–$60 |
| 5: Write-Up | $0 |
| **Total** | **$530–$1,180** |
| 6 (Stretch): Scale to 8B | +$500–$1,500 |

---

## Risk Mitigation

**Risk: Transcoders aren't faithful enough on a reasoning model**
- Mitigation: Skip transcoders (Dunefsky et al.) significantly improve faithfulness. If per-layer transcoders still struggle, try training on reasoning-specific data where the model is "in distribution."
- Fallback: Release SAEs instead of transcoders. Less useful for circuit analysis but still valuable.

**Risk: EleutherAI's clt-training doesn't work out of the box with Qwen architecture**
- Mitigation: The library supports Llama. Qwen-1.5B uses a similar architecture (RoPE, SwiGLU). May need minor adapter code.
- Fallback: Use SAELens which has broader model support, or train with a custom loop.

**Risk: Circuit tracer integration is harder than expected**
- Mitigation: Anthropic's circuit tracer has been used with Gemma-2-2B and Llama-3.2-1B. The adapter code should be straightforward.
- Fallback: Build attribution graphs using gradient-based methods (slower but doesn't require circuit tracer integration).

**Risk: The reasoning model circuits look identical to non-reasoning models**
- Mitigation: This is actually still a publishable result ("reasoning distillation doesn't change circuit structure"). Frame it as: "the reasoning capability lives in the weights/features, not the wiring."
- Also compare circuits during the <think> reasoning trace vs. the final answer — there may be differences in how the model processes intermediate reasoning steps.
