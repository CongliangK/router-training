# router-training

**Train a tiny vision-language model to route delegated subtasks into fast / balanced / flagship tiers — with calibration, honest evaluation, and a production quality gate.**

[中文文档](README.zh-CN.md) | [Data format & building your own dataset](docs/DATA.md) | [Agent guide](AGENTS.md)

This repository archives a complete, reproducible pipeline that fine-tunes
[`internlm/Intern-Decision-0.8B`](https://huggingface.co/internlm/Intern-Decision-0.8B)
(a 0.8B decision-scoring VLM) into a **task-difficulty router**: given the text of a
delegated subtask, it answers three fixed-schema questions — `tier` (fast / balanced /
flagship), `needs_vision` (yes / no), and `task_type` (7 classes) — which a caller can
map onto concrete model routes. The router never generates task answers; it only
scores fixed decision options from the model's logits.

## Why this exists

Multi-agent harnesses waste money and latency when every subtask goes to the largest
model, and lose quality when hard tasks go to the smallest. A 0.8B router that reads
only the task prompt and outputs a calibrated tier keeps routing overhead negligible.
The interesting engineering here is not the model — it is everything around it:

- **Expert-imitation training with honest metrics.** Labels are prompt-only expert
  judgments; the evaluation section is explicit that agreement-with-expert is *not*
  task-success, and a `quality_gate` flag refuses to claim production readiness.
- **Calibration as a first-class step.** Held-out temperature fitting plus a
  decision-side logit adjustment sweep (`tau * log(prior)`) chosen under
  recall/false-positive constraints.
- **Red-team-hardened sampling.** Class-balanced oversampling (fixes a
  majority-class collapse), split-group isolation for near-duplicate tasks,
  cluster-level bootstrap confidence intervals, calibration split into
  selection/fitting halves to avoid reuse.
- **A full failure log.** v2 → v3 → v3b → v3c iteration notes (tier collapse,
  bf16 update underflow, mixed-dtype kernel crashes, 16GB VRAM budget engineering)
  are kept in `research/` and `docs/` as historical archives.

## Final results (v3c, expert-label agreement on held-out test, n=104)

| Metric | v3c |
|---|---:|
| tier macro-F1 | 0.805 |
| flagship recall / precision | 0.933 / 0.509 |
| vision recall / precision | 0.959 / 0.887 |
| test contexts truncated | 0 |

The quality gate is **not passed** (no independent task-success benchmark yet;
flagship over-promotion remains). See the archived `artifacts/real-history-v3c/report.json`
in a full local checkout and the honest-limitations notes below.

## Repository layout

| Path | Purpose |
|---|---|
| `train_eval.py` | LoRA / full-backbone training, calibration, evaluation, quality gate |
| `model_utils.py` | Fixed-option decision logits, 8K context, shared train/serve truncation |
| `prepare.py`, `extract.py` | Read-only prompt extraction, redaction, grouped train/calibration/test splitting |
| `server.py`, `policy.py`, `routes.json` | Optional standalone FastAPI inference server |
| `scripts/train.py` | Safe launcher for the archived v3c recipe (refuses to overwrite runs) |
| `configs/v3c.json` | Exact CLI recipe of the completed run |
| `tests/` | CPU-only unit tests (run without a GPU) |
| `data/labeling/RUBRIC.md` | Tier/vision/task-type labeling rubric |
| `research/`, `docs/` | Full-FT survey, red-team notes, iteration history |
| `docs/DATA.md` | Data schemas, synthetic examples, building your own dataset |

## Getting started

Requires Python ≥ 3.11, a CUDA GPU (full fine-tuning peaks ~15GB), and a
CUDA-matched PyTorch build.

```sh
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Download the pinned base model revision into `models/intern-decision-0.8b`
(see below), then reproduce the archived run under a new name:

```sh
.venv/bin/python scripts/train.py --run v3c-reproduction --dry-run   # inspect only
.venv/bin/python scripts/train.py --run v3c-reproduction             # full 6-epoch run
.venv/bin/python scripts/train.py --run v3c-smoke --smoke            # GPU smoke test
```

Standalone inference server:

```sh
.venv/bin/python server.py --run real-history-v3c --port 19949 --routes routes.json
curl http://127.0.0.1:19949/health
```

### Base model & weights

The base is `internlm/Intern-Decision-0.8B` pinned to revision
`85a0cc5a99d67ea8d56dfe98115689212867171d`, loaded without remote model code.
Weight files (`.safetensors` etc.) are **not** distributed in this repository —
download the pinned revision from the upstream Hugging Face repo (Apache-2.0) or
fill in your local copy before training. Model config, tokenizer, inference code,
and upstream licenses are versioned under `models/intern-decision-0.8b/`.

## Data: not shipped, by design

The real training data (668 historical subtask prompts + expert labels) contains
potentially identifiable personal project details and is **not published**. The
repository instead ships:

- the complete **labeling rubric** (`data/labeling/RUBRIC.md`),
- exact **JSONL schemas** and a few **clearly-synthetic example rows**
  (`data/synthetic-examples/`),
- a guide for **extracting and labeling your own prompts** (`docs/DATA.md`).

All pipeline code runs unchanged on your own dataset once it matches the schema.

## Honest limitations

- Metrics measure **agreement with expert labels**, not real downstream task success.
- The v3c quality gate is intentionally left un-passed; reasons are recorded in
  `quality-gate.json` (missing task-success benchmark, tau gate miss, flagship
  over-promotion).
- Several training-implementation quirks of the archived run (effective initial
  LR 2e-6 vs CLI default, SWA pool semantics, non-group-isolated calibration
  halves) are documented rather than silently "fixed", so the recipe reproduces
  what actually produced the checkpoint. See README.zh-CN.md §复现要点.
- No optimizer resume; interrupted runs restart from scratch. Bit-exact
  reproduction across GPUs/driver versions is not guaranteed.

## License

Code and documentation: **Apache-2.0** (see [LICENSE](LICENSE), [NOTICE](NOTICE)).
The base model is upstream Apache-2.0; redistributed fine-tuned checkpoints inherit
its obligations. Real task data is not covered by this license because it is not
distributed.
