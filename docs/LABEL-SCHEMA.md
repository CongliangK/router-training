# Versioned label contract (#18; public CPU subset, issue OPEN)

## Two versions, never an implicit migration

| Label schema | task_type order | needs_vision supervision |
|---|---|---|
| `router-label-v1` (default/unversioned legacy) | extraction, research, implementation, debugging, review, visual, creative | historical prepare accepts uncertain; required calibration rejects it |
| `router-label-v2` (explicit opt-in) | extraction, research, implementation, debugging, review, creative | yes/no only; unresolved cases excluded into adjudication ledger |

`model_utils.QUESTIONS` and `TYPES` remain the original seven-type objects.
`task_label_schema.questions_for(version, legacy_questions=None)` returns a fresh
versioned table; v2 never mutates v1. The question text **and ordered options**
are part of the model interface, not just a classifier name change. Do not score
an old checkpoint using six options, drop the visual logit, or rename it in place.

- `task_label_schema.version(None)` selects v1; unknown versions fail.
- `validate_supervised_rows(rows, version)` rejects mixed versions. V2 needs a
  top-level `task_label_schema_version` and `label.label_status: "resolved"`, legal
  tier/type and binary vision. Adjudication/excluded and supplemental rows cannot
  supply ordinary training/calibration/evaluation support.
- `reviewed_task_type(primary_action, evidence=None)` accepts
  `write_executable` → implementation, `nonprogramming_creative` → creative,
  `visual_extract` → extraction, `visual_review` → review. Visual mappings need
  nonempty action evidence; no free-text keyword guessing is performed.
- `review_legacy_label(label, vision_decision=..., vision_evidence=...,
  primary_action=..., action_evidence=...)` returns a **copy**, with v2 version,
  reviewed evidence or `label_status: "adjudication"`, null unresolved fields
  and reasons. It never silently decides uncertain or visual. The helper cannot
  verify the truth/independence of supplied reviewer evidence.
- `partition_labels` separates resolved labels from a reasoned adjudication/
  excluded ledger. It does not promote pending records to labels.

## Prepare, preflight, train, score, serve, evaluate

New v2 recovery labels explicitly carry `task_label_schema_version` and `label_status`.
Prepare requires a **new separate output directory**, refuses any existing output,
retains near-duplicate unions even through excluded tasks, and writes resolved
splits, the adjudication ledger, schema metadata and counts. It does not change
split objectives/support gates (#24 is a separate integration). No real migration
has been run by this change.

```sh
python prepare.py --data /path/to/reviewed-input --output /path/to/new-v2-dataset \
  --label-schema router-label-v2
python scripts/validate_calibration.py --data-dir /path/to/new-v2-dataset \
  --roles /path/to/preregistered-roles.json --label-schema router-label-v2
```

The preflight is stdlib-only, never reads test, and uses the same versioned ordered
classes as runtime, preserving the legacy source-AST mutation canary. V2 legality
is checked on train and calibration **before** prior/model work. Required tier/
vision support gates and optional fixed task-type T=1 are unchanged. No support
is inferred merely from legal labels.

For a future **separately authorized** v2 launch, the launcher forwards both
`--label-schema` and `--data-dir` unchanged into the actual run. The v1 custom-data
launch guard is retained. V2 requires a new recipe override `{"synth": false}`:
legacy `--synth` is incompatible, not silently reinterpreted. Optional repeatable
`--supplemental <v2.jsonl>` loads explicitly synthetic, split-independent examples
into **train only, after real prior preflight**, never into calibration/test. The
15 public examples are not automatically loaded. Counts are separately recorded
in `effective_config.supplemental_rows`; the prior still excludes synthetic rows.

`model_utils.load_engine(..., schema_version=...)` tags the engine; `request_for`
and `score` use its bound version, rejecting an explicit mismatch. The SAME
truncation implementation serves both versions and train/serve. Scoring checks
logit width; v2 predictions carry `task_label_schema_version`.

New v2 runs write `label-schema.json` and `calibration.json.label_schema`, each
containing `task_label_schema_version`, `question_sha256` (canonical full question
JSON SHA-256), and ordered `classes`. Effective run metadata records the same
contract. `resolve_run_schema(calibration, run_metadata, requested)` and
`load_run_schema(directory, requested)` reject unknown/missing-v2, mismatched
versions, changed question fingerprints or class order **before loading weights**.
Both files are required for v2. Unversioned old calibration is valid **only as v1**.

Serving defaults to v1; v2 requires `server.py --label-schema router-label-v2`.
Health/route responses expose `task_label_schema_version` separately from HTTP
`schema_version` (also distinct from outcome/source schemas). Schema validation
precedes model loading even for diagnostic mode; execution/shadow quality gates
remain untouched. This change does not launch/restart a service.

`evaluation_metrics.metrics(..., schema_version=...)`, `per_class`, and the
training calibration helpers take an explicit version (the legacy temperature
fitter refuses v2; v2 uses grouped calibration only). V1 keeps its fixed seven
class denominator; v2 uses six and records its version. V2 rejects unversioned or
illegal rows, supplemental support, and old seven-class probabilities instead of
coercing them. A v2 metric is not directly comparable to a frozen-v1 metric.
The legacy split-report CLI is not a v2 evaluator: it fails closed on scored v2 rows;
use the explicit metrics API for v2 until reporting-version integration is added.

## Supplemental examples and outstanding acceptance

The 15 invented creative+vision rows (4 flagship) are format/rubric examples,
not observations, verified image assets, independent real groups, or measured
improvement. Their single synthetic source bucket and `supplemental_only` marker
prevent any implication of 15 independent sources. Default code does not ingest
them. CPU tests may use additional invented distributions to exercise support
failure; that is never real support evidence.

**Issue #18 stays OPEN.** No private repository/data/labels/artifacts were read or
changed. Six real uncertain cases still require evidence-based adjudication or
exclusion; 16 creative cases require a preregistered prediction-blind review.
Real revised version provenance, frozen-v1 preservation, split class/group support,
missing-class reports, no-leak evidence and independent acceptance remain absent.
Legal schemas and synthetic passes do not establish any support/performance goal.

## Concurrent integration boundary

- #8 dual-source ingest: compose its source schema with `task_label_schema_version`;
  source/outcome schema numbers are NOT label versions. Preserve measured shadow
  partition and never fabricate understanding labels. Overlap: prepare, train
  reader/preflight, DATA/README docs.
- #24 splitter: feed only resolved v2 labels while preserving duplicate unions,
  immutable outputs and pending counts; its objectives/feasibility remain separate.
  Overlap: prepare and DATA/README docs. No branch was merged here.
- #6/#23 server/policy: retain the new label-schema startup gate and response field
  beside policy-cause/identity/effort schema fields; preserve execution permissions.
  Overlap: server, README docs, ASGI tests. No routes/capabilities were changed.
- Calibration/training work (#19–22): propagate selected schema/classes/hash rather
  than importing the global legacy table for v2; do not relax support gates.

## 中文摘要

默认/无版本产物仍为七类 v1，六类 v2 必须显式选择；QUESTIONS 不变。
新标签行声明版本与 resolved，uncertain/动作不明保留待裁定或排除账本。
prepare 新目录、预检、训练请求、校准、元数据、服务启动与评估共用版本边界；
服务加载前核对完整问题哈希与类别顺序。合成 15 条（旗舰 4）只作独立于 split
的补充例子，需显式授权加入训练，不能填真实支持门。六条真实 uncertain、
16 条 creative 盲审与无泄漏/组支持验收仍不可用，#18 不关闭。
