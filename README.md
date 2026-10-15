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

## Opt-in six-task labels (#18; public CPU subset)

Default/unversioned checkpoints still use the **unchanged seven-type v1 questions**.
Explicit `--label-schema router-label-v2` selects six task types and yes/no vision;
unknown/mixed versions or incompatible checkpoint question hashes fail closed.
Executable code/scripts/stylesheets take precedence over a creative subject;
nonprogramming creation/appreciation is creative. Ambiguous visual actions and
unresolved vision remain adjudication/excluded, never forced binary.

See the [versioned API and CLI contract](docs/LABEL-SCHEMA.md) and
[labeling rubric](data/labeling/RUBRIC.md). Prepare writes v2 only to a new separate
directory; preflight/train/model questions/run metadata/serve/evaluation carry the
selected version. V2 serving requires matching run + calibration schema metadata.
The [15 clearly synthetic creative+vision examples](data/synthetic-examples/labels-v2-creative-vision.jsonl)
(4 flagship) are **split-independent supplemental only**, not real support or
measured improvement; no automatic ingestion or calibration/test membership.
Frozen v3c recipe/artifacts are unchanged. **#18 remains OPEN**: six real uncertain
reviews, 16 creative blinded reviews, real version provenance/group support and
no-leak acceptance are unavailable. No training, smoke or model loading is implied.


Integration with merged #8 keeps its intake API separate from task semantics:
`label_schema_version` remains source integer 1; `task_label_schema_version`
identifies router-label-v1/v2. V1 frozen/unified and expert-only split-v2 remain
available. V2 supports expert-only new outputs with the frozen-v1 split algorithm;
v2 + unified/measured/prepare-preflight/split-v2 fails closed before writes.
See the [exact supported matrix](docs/LABEL-SCHEMA.md#composed-interface-matrix-8--18--24).
Upstream #8 is approved/merged (main `32ad23a`, supplied integration status);
this #18 fix still needs a fresh independent full red-team of its final SHA.

V2 dataset preflight/runtime require matching complete question hash, ordered classes
and semantic version in both prepared schema documents before sampling/output/model
work. Relative launch inputs (dataset, supplements, roles, recipe) resolve once in the
caller's directory. Prediction adapters validate input semantics before scoring and
retain train-only/source provenance; supplements cannot become evaluation support.
Grouped calibration fits fresh role-B logits joined against locked IDs/groups/labels
(the repaired stale-planning-score defect was inherited from #2). The frozen-v1
report reader rejects v2 aggregates and rows, including unscored inputs; v2 runs
record an unavailable diagnostic instead. Use the explicit versioned metrics API.

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
| tier accuracy | 0.721 |
| tier macro-F1 (fixed 3 classes) | 0.805 |
| tier NLL / Brier | 1.488 / 0.678 |
| tier ECE (archived pre-fix implementation) | 0.408 |
| flagship recall / precision | 0.933 / 0.509 |
| vision recall / precision | 0.959 / 0.887 |
| test contexts truncated | 0 |

The quality gate is **not passed** (no independent task-success benchmark yet;
flagship over-promotion remains). See the archived `artifacts/real-history-v3c/report.json`
in a full local checkout and the honest-limitations notes below.

## Evaluation metrics and limits (issue #3)

New reports (`metric_definition_version: 2`) show **accuracy, fixed-class macro-F1,
NLL, multiclass Brier, five-bin ECE, and the true × predicted confusion matrix**
together for tier, vision, and task_type. Each `*_per_class` entry has precision,
recall, F1, true support `support` (n), predicted support, and `low_support` when
n < 10 (configurable through `metrics(..., low_support_threshold=...)`). A zero
ratio denominator is `null`; an entirely unevaluated field has `null` scores,
not zero accuracy/loss. Unknown target labels are counted in `excluded_rows`.

Macro-F1 keeps the historical **fixed schema denominator** (tier 3, vision 2,
task_type 7), including unsupported classes with zero contribution, and records
its definition and low-support warning. Undefined per-class F1 is `null` but
contributes zero to this explicitly named convention; an empty field's macro-F1
is `null`. There is no silently substituted supported-class average. Never pick
or qualify a router using macro-F1 or any other single metric alone.

ECE uses disjoint [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1] bins. `ece_bins`
exposes counts, weights, accuracy, confidence, and contributions; every valid
row enters exactly one bin. The old epsilon upper edge double-counted boundary
values (all-correct confidence .4 produced 1.2 rather than .6). Frozen v3c
artifacts and the table above are unchanged historical measurements, not
recomputed results. New NLL explicitly floors true-label probability at 1e-12;
Brier sums squared errors over all classes. Invalid/nonfinite probability
vectors are rejected instead of yielding misleading metrics.

`routing_costs` reports downgrade (true higher → predicted lower) and overroute
(true lower → predicted higher) counts and weighted losses, plus every tier
transition. The default is **illustrative ordinal tier distance**: one step
costs 1, two cost 2, correct decisions cost 0. It is not dollars or measured task
success. `metrics(..., cost_matrix=...)` accepts a predeclared true × predicted
nonnegative loss matrix with zero diagonal and records it in the report.

Bootstrap intervals sample whole `split_group` clusters from the full
field-valid pool, including groups without the target class. Prediction output
preserves source groups. Missing groups, fewer than two independent groups,
or fewer than two groups supporting the statistic denominator make the interval
unavailable (`low`/`high`: `null`, with a reason). Undefined ratio resamples are
omitted and counted; valid-resample counts, seed, and scope are reported. These
95% percentile intervals are conditional on nonzero denominators; few groups
or boundary scores can understate uncertainty, so even a degenerate interval
is not a production qualification. Row IDs never silently imply independence.

Optionally pre-register `--max-tier-ece <0..1>` before evaluation via
`scripts/train.py` or `train_eval.py`. The threshold is recorded in
`effective_config.evaluation_policy` and `quality_gate.calibration_constraint`.
Exceeding it, or lacking an evaluable score, adds a veto; meeting it never turns
an unpassed gate into a pass. **No threshold is enabled by default**; 0.25 is only
an unvalidated proposal. Grouped development still does not open test data and
cannot assess this independent-evaluation constraint. Candidate task-success
evidence remains required (#26); frozen-v1/private prediction verification is
external acceptance (#27). Synthetic CPU tests establish metric behavior, not
real model improvement or production readiness.

## Unified label intake and cold start (issue #8)

`prepare.py --outcomes <full-v2-audit.json> --routes <candidate-routes.json>`
accepts gated candidate outcomes alongside explicit expert understanding labels.
Training/calibration/test remain expert-only; measured labels go into a separate
traceable shadow partition with route fingerprints and trial/gate metadata.
No understanding fields are inferred from outcomes. Measured confidence is an
interval confidence level, never an expert sample weight. A future per-candidate
success-probability field is reserved as null, not fitted from mock evidence.
Add `--preflight` for stdlib-only validation (no torch, models, splitting or
writes); `--unified` also supports expert-only intake. These flags use the
default/explicit `--split-version frozen-v1`. Combining split-v2 with any
`--unified`, `--outcomes`, `--routes`, or `--preflight` explicitly fails before
input reads/writes; measured input is never silently ignored. `--output` is
split-v2-only. See the [supported flag matrix](docs/DATA.md#cli-coexistence-unified-intake-vs-split-v2).
Unified mode requires a fresh output bundle; any existing unified output blocks
**all frozen-v1 modes**, even preflight/expert-only reruns. Migrate or change corpus/evidence in a new directory
with input files only; nothing is auto-deleted. Legacy-only runs still replace
their five regular outputs. All metadata must be finite UTF-8 JSON; validation
and serialization finish before writes (not a crash/I/O/concurrency transaction).
See [schema, exact overwrite policy and migration](docs/DATA.md#output-lifecycle-and-migration).
Legacy expert splits and frozen v3c remain unchanged. Genuine measured supply
and validity remain external acceptance in #26; synthetic CPU tests are not
real measurements or production qualification.

## Repository layout

| Path | Purpose |
|---|---|
| `train_eval.py` | LoRA / full-backbone training, calibration, evaluation, quality gate |
| `calibration_groups.py` | Stdlib-only group-isolated calibration planning and out-of-fold tau/T selection (issue #2) |
| `model_utils.py` | Fixed-option decision logits, 8K context, shared train/serve truncation |
| `prepare.py`, `extract.py` | Read-only prompt extraction, redaction, grouped train/calibration/test splitting |
| `server.py`, `policy.py`, `routes.json` | Optional standalone FastAPI inference server |
| `outcomes.py` | Dynamic-routes candidate-outcome import gated by a conservative exact Clopper-Pearson lower bound (issue #7) |
| `scripts/train.py` | Safe launcher: explicit `--legacy` (archived v3c recipe) or grouped mode with pre-registered roles; both pass the full v3c recipe explicitly, grouped deviations via `--recipe` (issue #14) |
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
calibration is explicit `--legacy`; it reproduces the v3c recipe and
calibration split — note that the issue #14 scheduler/SWA correctness fixes
mean a re-run's LR trajectory now differs from the archived run, see below):

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
  2.747 fallback is never used). The business schema has ONE source:
  `model_utils.QUESTIONS`, assembled by a single function the stdlib
  preflight reaches by parsing that same file (no torch import). A roles
  config cannot override the schema and unknown config keys are rejected,
  so preflight, launcher dry-run, and the run itself validate the same
  truth.
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
- **The selected calibration is server-consumable as-is.** A selected result
  writes `calibration.json` with `temperatures` plus `tier_adjust` =
  `tau·log(prior)` per class (and `tier_prior`, the tau, and the binding
  hashes) — exactly what `server.py` loads and `model_utils.score` subtracts
  from raw logits, so serving applies the same adjust-then-temper formula the
  temperatures were fitted on. An **insufficient** result writes **no**
  `calibration.json`: the server refuses to start for such a run instead of
  silently routing unadjusted logits, and the run's quality gate stays
  not-passed, so execution mode is never authorized.
- **The binding locks checkpoint weight bytes.** `checkpoint_ref` in the
  calibration binding is a streaming SHA-256 digest of the actual checkpoint
  files (`adapter/` or `full-checkpoint/`, relative filenames explicit) — not
  the `best-epoch.json` text. Any byte change in the shipped weights changes
  the binding; free-text or tampered digests are rejected.
- **The old row-split halves remain available only as explicit legacy**
  (`--legacy-calibration`), byte-for-byte, for v3c reproduction. Grouped dev
  runs never open the test split; a `--structural-dry-run` (or
  `scripts/train.py --dry-run` in grouped mode) runs the real CPU preflight —
  no model, no artifacts, no test access:

```sh
.venv/bin/python scripts/train.py --run v4-dev --roles roles.json --dry-run
.venv/bin/python scripts/validate_calibration.py --roles roles.json --data-dir data
```

`--data-dir` configures that preflight **only**: the actual grouped launch
always reads the repository `data/` directory (`train_eval.py` takes no
data-dir), so a non-default `--data-dir` is refused at launch and allowed
only together with `--dry-run` — a run can never preflight one dataset and
train on another. Grouped `--smoke` is honestly short: 24 train rows plus a
whole-group calibration sub-pool re-validated under the SAME support policy
(gates are never relaxed to make smoke pass; legacy smoke keeps its archived
24-row behavior).

Output summaries are counts-only (no group names, ids, labels, or prompts).
**Status:** the archived calibration pool cannot satisfy the example gates
(roughly eight groups, some classes concentrated in very few of them — see
issue #2), so no grouped run can be instantiated with current data. The
gates are not relaxed to make it pass; additional group-isolated development
data is required. Full dev-no-test for training and the approved numeric
protocol are not yet satisfied — this repo does not claim training readiness.

### Training reproducibility fixes (issue #14)

Four audited defects are fixed for **future runs**; the archived v3c artifacts
are frozen history and are not touched:

- **The scheduler counts actual optimizer updates.** `total_steps` is derived
  from the class-balanced oversampled row stream — the same precomputed lists
  the training loop iterates (one source of truth) — with each epoch's tail
  batch counted by ceil. The archived run planned 498 steps from the raw
  pre-oversampling length while actually taking 1,158 updates. Cosine progress
  is clamped to [0, 1] so the LR can never rise again past the planned end,
  and the decay floor is **10% of the initial LR** (2e-7 for fullft, matching
  the official recipe; the archived run used a 1e-3 multiplier floor — see
  `--fullft` help and the README.zh-CN §复现要点 note). Warmup stays 3%.
  **Future runs therefore follow a different LR curve than the archived v3c**
  — including `--legacy` reproductions under current code; the archived v3c
  numbers are old-scheduler history and must not be compared LR-trajectory to
  LR-trajectory with new runs (legacy reports carry this note explicitly).
- **The grouped launcher passes the complete recipe.** `scripts/train.py`
  assembles legacy and grouped commands from one shared builder reading
  `configs/v3c.json`; a grouped run no longer silently falls back to
  `train_eval.py` defaults (`synth`/`fullft`/`swa` off). Deviations from v3c
  are declared only via an explicit `--recipe <json>` override (unknown or
  wrongly-typed keys are refused), echoed to the console, and the `--dry-run`
  Command lists every recipe flag actually passed:
  `--run v4-dev --roles roles.json --recipe overrides.json --dry-run`.
- **The report records what actually ran.** `report.json` gains an
  `effective_config` block — effective vs requested LR (fullft fixes the
  initial LR at 2e-6 regardless of `--lr`), per-epoch oversampled update
  counts, tail-batch handling, scheduler shape (total updates, warmup, clamp,
  floor), and the final SWA pool members — kept separate from raw CLI input.
- **The SWA pool is a strict top-3.** Epochs that fall out of the top-3 by
  selection macro-F1 are evicted and their CPU snapshots released; the soup
  is always exactly the final top-3 average, and the pool print line is
  truthful at every epoch.
- **The optimizer is a recorded recipe choice.** `--optimizer` selects
  `adamw` (default; byte-identical to the archived v3c) or `adamw8bit`,
  declared via `--recipe {"optimizer": "adamw8bit"}` and recorded in
  `effective_config`; the archived v3c used adamw. 8-bit optimizer states
  trade exact AdamW moments for ~2.8GiB VRAM on this GPU (fp32 AdamW fullft
  OOMs 16GB; the 8-bit m/v demand lands the run at ~9.7GiB). `bitsandbytes`
  is an optional RUNTIME-side dependency of the training machine only — it
  is deliberately absent from `requirements.txt` and CI, which stays green
  without it. A real full validation run (3 epochs, same seed and data as
  v3c) confirmed these estimates end-to-end — measured stability, throughput
  parity, and VRAM headroom, plus evaluation-methodology evidence for
  issues #1–#5, are distilled in
  [research/fullft-8bit-run-lessons.md](research/fullft-8bit-run-lessons.md).

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

### Candidate outcome labels (issue #7)

External review P1-4 / [issue #7](https://github.com/CongliangK/router-training/issues/7):
`outcomes.py` used hardcoded routes and treated a 3/3 success-rate point estimate as
selectable — but 3/3 all-pass says almost nothing about true success. The import
pipeline is now fixed first (no new traffic exists to collect yet); all fixtures in
tests are purely synthetic and no real candidate calls are made by this tool.

```sh
.venv/bin/python outcomes.py --tasks tasks.jsonl --outcomes outcomes.jsonl \
    --output labels.json --routes routes.json   # any routes.json; no code change
```

- **Dynamic routes.** Candidates come from an explicit `--routes` file. The default
  resolves to the repository `routes.json`; only if that file is absent does it fall
  back to the public `DEFAULT_ROUTES` and record a warning in the output. An
  explicitly given path that is missing fails fast (server.py semantics) — never a
  silent default substitution. Route entries may declare optional `model_version`,
  `reasoning_effort`, and `route_id` metadata (validated, `route_id` unique; the
  `(provider, model)` serving identity and `server.select()` contract are unchanged);
  a record whose measured version or effort disagrees with the declaration is
  rejected as a different operating point.
- **Conservative gate.** A candidate is selectable only when the *one-sided exact
  Clopper-Pearson lower bound* of its success rate clears the threshold — never the
  point estimate. Verified anchors at 95%: 3/3 → 0.3684031499 (no label), 28/28 →
  0.898534 (still no label), 29/29 → 0.9018553723 (the smallest all-pass n that
  clears 0.9), 49/50 → 0.9086018693. `min_trials` (default 3) counts trial
  completeness only and is not a reliability gate. The bound is computed with
  lgamma log-pmfs and term-ratio recursion because naive `math.comb` tails overflow
  float at n ≥ 1030; `--min-success-lb` replaces the old `--min-success-rate`
  (semantic change: the threshold now applies to the lower bound), and `--confidence`
  is new. The thresholds 0.9/0.95 are suggested defaults carried over from the
  issue's current values — not a user-frozen production protocol.
- **Conservative cost, stated honestly.** If a candidate's true success rate is
  exactly 0.9, the probability that 29 independent executions all pass is only
  0.0471 — clearing this gate is strong evidence, and an expensive one.
- **Sequential-peeking disclosure.** After one failure the gate can still be cleared
  by continuing (L(99,100) = 0.9534401885 ≥ 0.9), so the per-candidate trial count n
  and the candidate set are pre-registered before execution (the output's
  `preregistration` field records the routes fingerprint); growing n until the bound
  clears is sequential peeking, not confirmation.
- **Coverage statement.** The bound is per-candidate marginal — no simultaneous
  guarantee across candidates and no post-selection guarantee; passing it does not
  certify a candidate as the cheapest or safest overall. Selection among gate-passers
  keeps the preregistered deterministic tie-break (mean cost, mean latency, tier
  order) over all-trials means; it deliberately does not pick the highest lower bound.
- **Strict failure semantics and audit envelope.** `passed: false` requires a
  `failure_type` (`timeout` / `acceptance` / `execution_error` / `other`); a passed
  trial cannot carry one; a timeout is a failure. Every trial carries a unique
  `execution_id` (duplicates rejected — one execution counts once), a
  `context_sha256` that must be stable within a task, `wall_time`/`timeout`, optional
  seed/randomness policy, and a 16-hex `route_fingerprint` (sha256 of
  [provider, model, tier, model_version, reasoning_effort]). Identical artifact bytes
  across distinct executions are legal — deterministic tasks can make independent
  runs coincide — and only raise a `duplicate-artifact-hash` audit flag, never a
  rejection or a sample-size discount; a nonce or fresh execution_id must never
  mutate the artifact body to fake distinct executions.
- **Privacy boundary and estimand.** `trial_details` are execution evidence for
  local audit files only; never paste them into issues, PRs, or any public artifact
  — public material may carry only purely synthetic coarse aggregates. The estimand
  is the single-execution success rate of harness-independent executions on the
  fixed evaluated task set: no extrapolation to new task populations and no
  production-benefit claim.

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
- Training-implementation defects found by later audit (scheduler step count
  from the raw length, unclamped cosine with a 1e-3 floor, fullft LR
  misreporting, over-filled SWA pool, grouped launcher dropping the recipe)
  are fixed for future runs ([issue #14](https://github.com/CongliangK/router-training/issues/14));
  the archived v3c artifacts keep the old behavior, so its LR trajectory is
  **not comparable** with new runs. The non-group-isolated calibration halves
  remain only under explicit `--legacy` (see above). See README.zh-CN.md
  §复现要点.
- No optimizer resume; interrupted runs restart from scratch. Bit-exact
  reproduction across GPUs/driver versions is not guaranteed.
- No real candidate-execution data exists yet: `outcomes.py` (issue #7) defines and
  CPU-validates the import pipeline with purely synthetic fixtures; no real success
  rates, cost savings, or production benefits are claimed.

## License

Code and documentation: **Apache-2.0** (see [LICENSE](LICENSE), [NOTICE](NOTICE)).
The base model is upstream Apache-2.0; redistributed fine-tuned checkpoints inherit
its obligations. Real task data is not covered by this license because it is not
distributed.

### Opt-in split-v2 (CPU, issue #24)

The historical/default `prepare()` and frozen outputs are unchanged. New entry
points `prepare.split_v2(tasks, labels, ...)` (pure) and
`prepare.prepare_v2(root, output, ...)` (fresh bundle) are stdlib-only:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void python prepare.py \
  --split-version split-v2 --data /path/to/new-expert-inputs \
  --output /path/to/nonexistent-split-v2-bundle
```

A new directory holds versioned `*.split-v2.jsonl` and
`split-report.split-v2.json`; all existing output paths are refused. Roots and
transitive cross-root near-duplicate components stay indivisible. Every selected
split has largest component share **≤35%**; the gate is never weakened. Scoring
adds vision=yes and task_type proportion penalties and reports targets, actual
proportions/deviations, class support and descriptive Kish concentration — **not
independent effective sample size or independence proof**. Jaccard <.88 does not
establish independence; child sessions/time windows cannot authorize root splitting.

Provable impossibility returns `infeasible`; bounded heuristic failure returns
`search_exhausted`, **not a proof**. Either writes only a new diagnostic report
(exit 2), never a relaxed split. See [DATA.md](docs/DATA.md#opt-in-split-v2-cpu-development-issue-24)
for objective weights, exhaustive/heuristic budgets and the API contract.
**A changed test set is not directly comparable to historical frozen-v1 metrics.**
No real re-split or vision improvement is claimed: #24 is synthetic CPU development;
versioned labels (#18) and real distribution/.35 feasibility/version review remain
external acceptance in #27. Training data paths are not switched automatically.

### Frozen-test sensitivity (CPU, issue #4)

New reports label the historical test set **frozen-v1, known skew** and append
`frozen_test_diagnostics`. Run `python scripts/report_split.py --split-report
 data/split-report.json` for aggregate counts; add `--predictions <fixed-test.jsonl>`
for observed group support, bootstrap, and fixed-prediction group deletion.
No predictions means unavailable metrics. This is no-retraining sensitivity,
not cross-validation or evidence of independent model quality; vision is
mainly supported by one historical task group. Never tune on frozen-v1.
The CLI uses only the standard library, exports aggregate aliases, and refuses
output overwrite. See [DATA.md](docs/DATA.md#frozen-v1-known-skew--已知偏斜)
for the input contract and limits. Real replay remains #27.
