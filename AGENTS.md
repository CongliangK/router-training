# AGENTS.md — guidance for AI coding agents working in this repository

This file tells any LLM-based coding agent (Claude Code, Codex, DSH agents, etc.)
how to work here safely. Read it before making changes.

## What this repository is

A standalone training pipeline that fine-tunes Intern-Decision-0.8B into a
task-difficulty router (tier / vision / task_type fixed-option decisions).
It is a **curated archive plus living code**: the v3c recipe and its artifacts
are frozen history; new work happens under new run names.

## Hard rules

1. **Never publish or copy real data outside the repo.** `data/*.jsonl` (except
   `data/synthetic-examples/`), `data/labeling/batch-*`, `labels-*`, and
   `artifacts/**/*.jsonl` contain real (redacted-but-not-guaranteed) personal
   task prompts. Do not paste their contents into issues, PRs, commit messages,
   external services, or the public export.
2. **Never overwrite an existing run.** `artifacts/<run>/` directories are
   append-only; `scripts/train.py` already refuses — keep that behavior.
3. **Do not modify frozen artifacts.** `artifacts/**`, `configs/v3c.json`,
   `snapshot-manifest.json` and `model-manifest.json` are archives of what
   actually happened; checksums are asserted by `tests/test_repository.py`.
   If code must change, update the manifest deliberately and explain why.
4. **Do not enable remote model code** (`trust_remote_code`) when loading the
   base model. Pinned revision only.
5. **GPU discipline.** Full fine-tuning needs ~15GB VRAM. Do not kill or restart
   unrelated GPU services to make room; ask the human first. Smoke mode
   (`--smoke`) exists for cheap pipeline checks.
6. **Keep training/serve parity.** `model_utils.request_for` is the single
   truncation path shared by training and `server.py`; never fork it.

## Where things are

| Need | Look at |
|---|---|
| Training loop, calibration, metrics, gate | `train_eval.py` |
| Fixed-option scoring, context handling | `model_utils.py` |
| Data extraction / redaction / splitting | `extract.py`, `prepare.py` |
| Serving | `server.py`, `policy.py`, `routes.json` |
| Run recipe | `configs/v3c.json`, `scripts/train.py` |
| Labeling rules | `data/labeling/RUBRIC.md` |
| Data schemas & building a dataset | `docs/DATA.md` |
| Iteration history / why decisions were made | `research/`, `docs/OVERNIGHT-STATUS.md`, README "复现要点" |

## Conventions

- Style: this codebase is deliberately compact (no linter config). Match the
  existing dense style; do not reformat whole files.
- Tests: CPU-only, `python -m unittest discover -s tests -v`. Run them before
  declaring any change done; they need no GPU and no weights.
- Python ≥ 3.11. Dependencies are pinned in `requirements.txt`; torch must be
  a CUDA build matching the local driver — never "fix" the pin to a CPU wheel.
- Comments explaining **why** (red-team findings, numeric pitfalls like the
  bf16 underflow fix) are load-bearing. Do not delete them as "noise".
- New documentation that changes public behavior should update both
  `README.md` and `README.zh-CN.md` (bilingual pair).

## Verified commands

```sh
.venv/bin/python -m unittest discover -s tests -v          # CPU tests
.venv/bin/python scripts/train.py --run <new-name> --dry-run
.venv/bin/python server.py --run <run> --port <port> --routes routes.json
```

If a task requires real data to leave the machine, GPU service interruption,
or editing frozen artifacts — stop and ask the human.
