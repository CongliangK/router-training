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
| `calibration_groups.py` | Stdlib-only group-isolated calibration planning and out-of-fold tau/T selection (issue #2) |
| `model_utils.py` | Fixed-option decision logits, 8K context, shared train/serve truncation |
| `prepare.py`, `extract.py` | Read-only prompt extraction, redaction, grouped train/calibration/test splitting |
| `server.py`, `policy.py`, `routes.json` | Optional standalone FastAPI inference server |
| `scripts/train.py` | Safe launcher: explicit `--legacy` (archived v3c recipe) or grouped mode with pre-registered roles |
| `scripts/validate_calibration.py` | Pure-CPU anonymous structural validation of a roles config (never reads test data) |
| `configs/v3c.json` | Exact CLI recipe of the completed run |
| `tests/` | CPU-only unit tests (run without a GPU) |
| `data/labeling/RUBRIC.md` | Tier/vision/task-type labeling rubric |
| `research/`, `docs/` | Full-FT survey, red-team notes, iteration history |
| `docs/DATA.md` | Data schemas, synthetic examples, building your own dataset |

## Getting started

Requires Python ≥ 3.11. Training needs a CUDA GPU (full fine-tuning peaks ~15GB)
and a CUDA-matched PyTorch build; CPU unit tests need neither a GPU nor model weights.

```sh
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

### Public-checkout validation (release standard)

A fresh public clone must install the unchanged `requirements.txt` and pass
`python -m unittest discover -s tests -v` without private files, a GPU, or weights.
GitHub Actions runs this CPU-only check on pushes and pull requests. The pinned
CUDA-capable dependencies are retained, even on a CPU runner.

The public tests generate explicitly synthetic data in temporary directories and
exercise `prepare.py`, checking split counts, complete ID coverage, ID/group isolation,
and separation from synthetic augmentation. Fixture counts (120 primary rows and
200 supplemental rows) are not the archived 668-row dataset. Three additional
archive checks are explicitly **skipped** when their private assets are absent:
real split counts/isolation, real-versus-synthetic prompt separation, and source
checksums from `snapshot-manifest.json`. When present, these assets are validated
with the original assertions; skips do not establish private-archive integrity.
No real prompts, labels, or private fingerprints are required or distributed.

Download the pinned base model revision into `models/intern-decision-0.8b`
(see below), then reproduce the archived run under a new name (the row-split
calibration is explicit `--legacy`; it reproduces v3c exactly and is not the
default for new runs):

```sh
.venv/bin/python scripts/train.py --run v3c-reproduction --legacy --dry-run   # inspect only
.venv/bin/python scripts/train.py --run v3c-reproduction --legacy             # full 6-epoch run
.venv/bin/python scripts/train.py --run v3c-smoke --legacy --smoke            # GPU smoke test
```

### Group-isolated calibration for new runs (issue #2)

External review P1-3 / [issue #2](https://github.com/CongliangK/router-training/issues/2):
the archived recipe shuffled the calibration split **by row** into
selection/fitting halves. Task ids are unique, but near-duplicate tasks from
one source share a `split_group`, so a row bisection puts correlated rows on
both sides; with only ~8 recorded calibration groups, no re-split of rows can
fix that — reuse is not addressed by another seed.

New runs therefore default to a strict, pre-registered grouping in
`calibration_groups.py` (stdlib-only, importable without torch):

- **Roles A/B/C are pre-registered** in a roles JSON (`role_groups` plus
  optional `support_policy`/`protocol`). A = checkpoint selection rows,
  B = temperature/tau fitting rows, C = reserved for final configuration
  selection. Groups and ids are mutually exclusive across roles and must
  cover every calibration row; a missing role is a hard failure with **no
  seed-based fallback**.
- **Support gates fail closed.** Example gates (provisional, *not* frozen by
  any user decision): ≥5/6/5 groups for A/B/C, ≥5 rows and ≥2 groups per
  required-head class per region, ≥2 groups per class in every B-fold
  training union. Vision stays a required head — preflight never silently
  downgrades to tier-only. `task_type` is optional: without support its
  temperature stays at T=1 and is reported as not calibrated (the legacy
  2.747 fallback is never used).
- **Tau is selected by deterministic 3-fold GroupKFold over B** (groups
  sorted by row count, sha256 tiebreak, whole groups never split; every B row
  is a holdout exactly once). Per tau, each fold first applies
  `z − tau·log(prior)` to its training rows, *then* fits that fold's
  temperatures, then scores the holdout with the same formula; groups are
  equally weighted throughout. A holdout fold missing a class yields a null
  score (never a fake 0); if no tau is feasible, nothing is selected. After
  selection, temperatures are refit once on all of B and the result binds
  plan/prior/schema/protocol/checkpoint hashes. This is a development
  estimate, not an unbiased evaluation.
- **Priors come from real unique train rows only** (synthetic augmentation
  and repeated ids excluded; a class without support is a hard error).
- **The old row-split halves remain available only as explicit legacy**
  (`--legacy-calibration`), byte-for-byte, for v3c reproduction. Grouped dev
  runs never open the test split; a `--structural-dry-run` (or
  `scripts/train.py --dry-run` in grouped mode) runs the real CPU preflight —
  no model, no artifacts, no test access:

```sh
.venv/bin/python scripts/train.py --run v4-dev --roles roles.json --dry-run
.venv/bin/python scripts/validate_calibration.py --roles roles.json --data-dir data
```

Output summaries are counts-only (no group names, ids, labels, or prompts).
**Status:** the archived calibration pool cannot satisfy the example gates
(roughly eight groups, some classes concentrated in very few of them — see
issue #2), so no grouped run can be instantiated with current data. The
gates are not relaxed to make it pass; additional group-isolated development
data is required. Full dev-no-test for training and the approved numeric
protocol are not yet satisfied — this repo does not claim training readiness.

Standalone inference server:

```sh
.venv/bin/python server.py --run real-history-v3c --port 19949 --routes routes.json
curl http://127.0.0.1:19949/health
```

### Service contract (schema version 2)

The default `--mode diagnostic` is **advisory only**, even for a qualified model.
Both `/health` and `/route` report `mode` (`diagnostic` or `execution`),
`quality_gate_passed`, `quality_gate_mode` (`shadow` or `qualified`),
`quality_gate_status`, `execution_allowed`, and `advisory_only`. A non-null
`recommendation` repeats the last two permission flags. Its presence alone is
never permission to execute; callers must require `execution_allowed: true` on
the route response and recommendation.

`--mode execution` permits a recommendation only when that run's gate has the
literal boolean `passed: true` and `mode: qualified`, and an authorized route
exists. An unpassed, missing, unreadable, malformed, or contradictory gate
returns HTTP 503 with a fail-closed contract in `detail`, before model scoring.
No authorized recommendation also returns 503. Diagnostic mode may still show
an advisory recommendation with such a gate. The health permission describes
service readiness, not a guarantee that every request has a compatible route.
Schema 2 replaces the old response `mode=shadow/qualified` with explicit service
mode; gate status remains separate. Clients must update rather than infer
permission from the old field. The archived v3c is **shadow, not production
qualified**; a historical service switch or `ready: true` does not pass its gate.
No model, gate artifact, or deployment is changed by this contract.

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
- The archived calibration split bisects rows (groups leak across the
  selection/fitting halves); it is preserved under explicit `--legacy` for
  reproduction, and new runs use the group-isolated path above — which the
  current data cannot yet instantiate (issue #2).
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
