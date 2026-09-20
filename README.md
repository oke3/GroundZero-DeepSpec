# GroundZero-DeepSpec

> Under the hood of LLM code intelligence — train and evaluate speculative-decoding draft models.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Ground Zero LLC](https://img.shields.io/badge/Built%20by-Ground%20Zero%20LLC-purple)](https://github.com/oke3)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)

DeepSpec is a full-stack codebase for training and evaluating draft models for speculative decoding.
It contains data preparation utilities, draft model implementations, training code, and evaluation
scripts — everything you need to benchmark how well small models can predict what large models will
generate, accelerating inference without sacrificing quality.

This is a Ground Zero LLC fork of the original DeepSpec research codebase, maintained for
reproducibility and extensibility.

---

## Why

Speculative decoding is one of the most practical ways to accelerate LLM inference: a small
"draft" model proposes tokens, a large "target" model verifies them in parallel. When the draft
is right, you get multiple tokens for the cost of one forward pass.

But training a good draft model is non-trivial. You need:

1. **Target cache generation** — run the target model over training data to produce hidden-state
   caches for the draft to learn from.
2. **Architecture design** — the draft must read the target's internal representations and
   propose tokens that match the target's distribution.
3. **Evaluation** — measure acceptance rates across diverse benchmarks to know if your draft
   actually helps.

DeepSpec provides all three stages in a single, reproducible pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DATA PREPARATION                                        │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Download   │─▶│ Regenerate   │─▶│ Build Target   │  │
│  │ Prompts    │  │ Target       │  │ Cache (hidden  │  │
│  │ (JSONL)    │  │ Answers      │  │ states/layer)  │  │
│  └────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  TRAINING                                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  deepspec/trainer/                                │   │
│  │  BaseTrainer → DSpark Trainer | Eagle3 Trainer    │   │
│  │  Config: config/dspark/ | config/eagle3/          │   │
│  │  Output: ~/checkpoints/<project>/<exp>/step_*     │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  EVALUATION                                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  deepspec/eval/                                   │   │
│  │  Speculative-decoding loop: propose → verify →    │   │
│  │  accept/reject across 9 benchmarks                │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Module Overview

| Module | Purpose |
|--------|---------|
| `deepspec/modeling/` | Draft model architectures (DSpark, Eagle3) |
| `deepspec/trainer/` | DDP-aware training loops, checkpoint management |
| `deepspec/eval/` | Speculative-decoding evaluation with acceptance-rate metrics |
| `deepspec/data/` | Dataset loading, target-cache datasets, CUDA prefetching |
| `deepspec/utils/` | Config loading, seeding, distributed init, sampling |
| `config/` | Per-algorithm × per-target-model training configurations |
| `scripts/` | Shell wrappers for data prep, training, and evaluation |

---

## Supported Algorithms

| Algorithm | Paper | Key Idea |
|-----------|-------|----------|
| **DSpark** | [DSpark paper](./DSpark_paper.pdf) | Markov head + confidence head on target hidden states |
| **DFlash** | [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | Flash-attention-based draft with cross-layer attention |
| **Eagle3** | [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) | Autoregressive draft with target-layer feature fusion |

Target models: **Qwen3** (4B, 8B, 14B) and **Gemma 4** (12B it).

---

## Benchmarks

| Benchmark | Samples | Domain |
|-----------|---------|--------|
| GSM8K | 500 | Grade-school math |
| MATH-500 | 500 | Competition math |
| AIME 2025 | 30 | American Invitational Math Exam |
| HumanEval | 164 | Code generation (Python) |
| MBPP | 256 | Basic Python problems |
| LiveCodeBench | 500 | Competitive programming |
| MT-Bench | 80 | Multi-turn instructions |
| Alpaca | 500 | General instructions |
| Arena-Hard-v2 | 500 | Hard prompt arena |

Each benchmark measures **acceptance rate** (draft tokens accepted by target) and **acceptance
length** (average consecutive accepted tokens).

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s) — default configs assume 8 GPUs on a single node

### Installation

```bash
git clone https://github.com/oke3/GroundZero-DeepSpec.git
cd GroundZero-DeepSpec
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

> Install the CUDA build of PyTorch matching your machine if the default wheel is wrong.

### Data Preparation

```bash
bash scripts/data/prepare.sh
```

See [scripts/data/README.md](./scripts/data/README.md) for the full pipeline. **Storage warning:**
the target cache can be ~38 TB for `Qwen/Qwen3-4B`.

### Training

```bash
bash scripts/train/train.sh --config config/dspark/dspark_qwen3_4b.py
```

Override config values with `--opts`:

```bash
bash scripts/train/train.sh --config config/dspark/dspark_qwen3_4b.py \
  --opts data.max_length 2048 train.global_batch_size 256
```

Checkpoints → `~/checkpoints/<project>/<exp>/step_*`. For fewer GPUs, set `CUDA_VISIBLE_DEVICES`.

### Evaluation

```bash
bash scripts/eval/eval.sh \
  --target_name_or_path Qwen/Qwen3-4B \
  --draft_name_or_path ~/checkpoints/deepspec/dspark_block8_qwen3_4b/step_latest
```

Or use a released Hugging Face checkpoint:

```bash
bash scripts/eval/eval.sh \
  --target_name_or_path Qwen/Qwen3-4B \
  --draft_name_or_path deepseek-ai/dspark_qwen3_4b_block7
```

Optional flags: `--max-new-tokens` (2048), `--temperature` (1.0), `--confidence-threshold` (0.0),
`--tensorboard-dir`, `--step`, `--seed` (980406).

---

## Released Checkpoints

Used for Table 1 in the [paper](./DSpark_paper.pdf). Trained on
[open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend).

| Algorithm | `Qwen3-4B` | `Qwen3-8B` | `Qwen3-14B` | `Gemma4-12B` |
| --- | --- | --- | --- | --- |
| Eagle3 | [eagle3_qwen3_4b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_4b_ttt7) | [eagle3_qwen3_8b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_8b_ttt7) | [eagle3_qwen3_14b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_14b_ttt7) | [eagle3_gemma4_12b_ttt7](https://huggingface.co/deepseek-ai/eagle3_gemma4_12b_ttt7) |
| DFlash | [dflash_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_4b_block7) | [dflash_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_8b_block7) | [dflash_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_14b_block7) | [dflash_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dflash_gemma4_12b_block7) |
| DSpark | [dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) | [dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | [dspark_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7) | [dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) |

> [!IMPORTANT]
> Align your setup with this repo's training settings when citing; otherwise comparisons are
> not meaningful. For domain-specific use, fine-tune the draft model again.

---

## Project Structure

```
GroundZero-DeepSpec/
├── config/              # Training configs (dspark/, dflash/, eagle3/)
├── deepspec/
│   ├── data/            # Dataset loading, target-cache, CUDA prefetch
│   ├── eval/            # Speculative-decoding evaluation loop
│   ├── modeling/        # Draft architectures (dspark/, eagle3/)
│   ├── trainer/         # Training loops, checkpoint management
│   └── utils/           # Config, seeding, distributed, sampling
├── eval_datasets/       # JSONL benchmark datasets
├── scripts/             # Shell wrappers (data/, eval/, train/)
├── train.py             # Training entry point
├── eval.py              # Evaluation entry point
└── DSpark_paper.pdf     # DSpark research paper
```

---

## Research

DeepSpec is designed for research reproducibility:

- **Deterministic seeding** — `seed_all()` at every stage boundary.
- **DDP-first** — `torch.multiprocessing.spawn` for training; `torch.distributed` for eval aggregation.
- **Target cache** — Pre-computed hidden states ensure draft training matches inference.
- **Config-driven** — Every hyperparameter in a Python config file, logged with checkpoints.

### Adding a New Algorithm

1. Create `deepspec/modeling/<algorithm>/`.
2. Implement a trainer subclass in `deepspec/trainer/`.
3. Implement an evaluator subclass in `deepspec/eval/`.
4. Add a config under `config/<algorithm>/`.
5. Register the evaluator in `eval.py`'s `EVALUATORS` dict.

---

## Citation

```bibtex
@misc{deepspec2026,
  title={DeepSpec: Training and Evaluating Speculative-Decoding Draft Models},
  author={The DeepSpec Authors},
  year={2026},
  publisher={GitHub},
  url={https://github.com/oke3/GroundZero-DeepSpec}
}
```

---

## Acknowledgements

DeepSpec builds on [SpecForge](https://github.com/sgl-project/SpecForge) (Apache-2.0),
[DFlash](https://github.com/z-lab/dflash) (MIT),
[Qwen3](https://github.com/QwenLM/Qwen3), and
[Gemma](https://github.com/google-deepmind/gemma). See [NOTICE](./NOTICE) for full attribution.

---

## Related Projects

| Project | What It Does |
|---------|-------------|
| [gz-context-engine](https://github.com/oke3/gz-context-engine) | Production-grade RAG context engine |
| [gz-modelrouter](https://github.com/oke3/gz-modelrouter) | Intelligent LLM cost router |
| [gz-gateway](https://github.com/oke3/gz-gateway) | OpenAI-compatible AI gateway — rate limiting, caching, failover, cost tracking |
| [gz-agent](https://github.com/oke3/gz-agent) | Production-grade agent runtime — tool calling, state machines, multi-agent coordination |
| [gz-eval](https://github.com/oke3/gz-eval) | Evaluation framework — golden test sets, quality scoring, A/B comparison |
| [gz-guardrails](https://github.com/oke3/gz-guardrails) | AI safety middleware — PII detection, prompt injection defense, content moderation |
| [hyperframes](https://github.com/oke3/hyperframes) | Agent-native video rendering |

---

---

## Enterprise Support

Need this customized for your infrastructure? We offer:

- **Integration consulting** — Wire GroundZero-DeepSpec into your research pipeline
- **Custom configuration** — Task-specific rules, models, and workflows for your team
- **Managed deployment** — We host and maintain your instance
- **Training workshops** — Hands-on sessions for your engineering team

[Book a 30-min call](https://www.grndxero.com/brief) · [See pricing](https://www.grndxero.com/pricing)

---

## License

MIT — Ground Zero LLC

---

Built by [Ground Zero LLC](https://github.com/oke3) — AI infrastructure for the agentic age.
