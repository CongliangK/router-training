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
legacy split objectives/support gates (task-v2 + split-v2 is explicitly unsupported). No real migration
has been run by this change.

```sh
python prepare.py --data /path/to/reviewed-input --output /path/to/new-v2-dataset \
  --label-schema router-label-v2
python scripts/validate_calibration.py --data-dir /path/to/new-v2-dataset \
  --roles /path/to/preregistered-roles.json --label-schema router-label-v2
```

The preflight is stdlib-only, never reads test, and uses the same versioned ordered
classes as runtime, preserving the legacy source-AST mutation canary. V2 legality
is checked on train and calibration **before** prior/model work. Both dataset
`label-schema.json` and `split-report.json.label_schema` must match the selected
full question hash, ordered classes and semantic version. Missing, malformed or
conflicting v2 declarations reject before output/sampling/model work. V1 metadata
may be absent; explicit conflicts still fail. Required tier/vision support gates
and optional fixed task-type T=1 are unchanged. No support is inferred merely from
legal labels. Grouped fitting joins freshly scored B logits by locked IDs, groups,
labels and version, rejecting missing/duplicate/mismatched B rows. Planning caches
never supply fitted logits (repair of an inherited #2 defect; Refs #2).

For a future **separately authorized** v2 launch, the launcher forwards both
`--label-schema` and the validated `--data-dir` into the actual run. Dataset,
supplemental, roles and recipe paths resolve once against the original caller cwd,
so changing the child cwd cannot change the validated file identities. Absent
options add no new historical command flags. The v1 custom-data launch guard is retained. V2 requires a new recipe override `{"synth": false}`:
legacy `--synth` is incompatible, not silently reinterpreted. Optional repeatable
`--supplemental <v2.jsonl>` loads explicitly synthetic, split-independent examples
into **train only, after real prior preflight**, never into calibration/test. The
15 public examples are not automatically loaded. Counts are separately recorded
in `effective_config.supplemental_rows`; the prior still excludes synthetic rows.

`model_utils.load_engine(..., schema_version=...)` tags the engine; `request_for`
and `score` use its bound version, rejecting an explicit mismatch. The SAME
truncation implementation serves both versions and train/serve. Scoring checks
logit width; v2 predictions carry `task_label_schema_version`. `predict_rows`
validates the entire input's selected semantics before any score call, then checks
score version/width without re-stamping input metadata. It retains all source,
group, synthetic and supplemental-only provenance (excluding duplicate prompt
text). Scored supplemental rows still cannot supply real calibration/metrics.

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
The legacy split-report CLI/API is not a v2 evaluator: it fails closed on v2
aggregate metadata, unscored/scored rows and all early-return paths. V2 run reports
record an honest unavailable diagnostic without calling that reader; use the
explicit metrics API for v2 until reporting-version integration is added.

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

- #8 dual-source ingest: keep its source schema distinct from `task_label_schema_version`;
  source/outcome schema numbers are NOT label versions. Preserve measured shadow
  partition and never fabricate understanding labels. Overlap: prepare, train
  reader/preflight, DATA/README docs.
- #24 splitter: feed only resolved v2 labels while preserving duplicate unions,
  immutable outputs and pending counts; its objectives/feasibility remain separate.
  Overlap: prepare and DATA/README docs. #8 intake is integrated unchanged here,
  including reviewed split-v2; upstream #8 is now approved/merged (supplied main `32ad23a`).
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

## Composed interface matrix (#8 / #18 / #24)

`label_schema.py` is the unchanged #8 intake API; `task_label_schema.py` is the
independent #18 question/semantic API. There is no class-table patching or implicit
migration. `label_schema_version: 1` is an intake/source contract integer (currently
in unified split reports); it is NEVER `router-label-v1/v2`. Semantic rows, engine,
metrics and metadata use **`task_label_schema_version`** instead. An ambiguous old
semantic string in `label_schema_version`, bool, or unsupported source integer
fails at the composed prepare/split/preflight/metrics boundaries. Complete run
metadata stays under `label_schema`, with the semantic key, hash and ordered classes.

| Task semantics | Split algorithm | Source / operation | Support |
|---|---|---|---|
| v1 (default) | frozen-v1 | expert-only, legacy five outputs | supported, historical bytes/reruns unchanged |
| v1 | frozen-v1 | unified expert or gated expert+measured; prepare/preflight | supported, #8 gate replay/fresh bundle; expert supervision + separate measured shadow |
| v1 | split-v2 | expert-only new versioned bundle | supported, .35 cap/root+transitive components/no overwrite/infeasibility unchanged |
| v1 | split-v2 | unified/outcomes/routes/prepare-preflight | unsupported; CLI refuses before input reads/writes; APIs reject keywords |
| v2 (explicit) | frozen-v1 algorithm | reviewed expert-only, new separate output | supported; pending/excluded ledger retained, never supervised; duplicate unions precede exclusion |
| v2 | frozen-v1 algorithm | unified/measured/routes or prepare --preflight | unsupported; dispatcher refuses before input reads/writes |
| v2 | split-v2 | any source | unsupported; explicit CLI refuses before reads/writes; undeclared v2 inputs fail validation before directory creation |
| v2 | already prepared expert splits | structural calibration preflight, launcher dry-run, training interfaces, score/server/metrics | supported with matching explicit semantics and complete dataset + run metadata; no actual training performed |
| v2 | any | frozen-v1 split-report CLI/API | unsupported for aggregates and unscored/scored rows; run diagnostic unavailable, explicit metrics API retained |

A split-v2 expert output can be evaluated only through an explicit future filename
adapter: current launcher reads train.jsonl/calibration.jsonl, not split-v2 names.
There is no automatic split-v2-to-training adapter. Existing unified bundles can
be read by the expert-only split-v2 path; it does not migrate their shadow/evidence.
Neither task version admits measured shadow rows into supervision or metrics.
Source API alone remains byte-for-byte upstream; version composition is owned by
the dispatcher and task-boundary validator, not a reinterpretation inside #8.

Upstream #8 is **approved and merged** (main `32ad23a`, user-supplied status;
no network recheck). This #18 fix's final SHA still needs a fresh FULL independent
red-team; no previous approval transfers. #6/#23 were NOT imported: future coexistence must compose server startup
schema gate, response semantic version, policy causes, candidate identity/effort and
execution permissions together. Original #18 remains OPEN: six uncertain reviews,
16 creative prediction-blind reviews and real provenance/support/no-leak evidence
remain external (Refs #5, #8, #24, #26, #27; calibration/loss dependencies #19–22).

中文：来源版本整数与任务语义版本分离；#8 模块/API 原样保留，#18 使用独立模块和
`task_label_schema_version`。v2 仅支持旧切分算法的专家新目录与显式后续接口；
v2+统一/实测/prepare预检/split-v2 均在写入前拒绝，不声称真实 v2 重切成功。
#8 已批准合入上游（用户提供 main `32ad23a`）；本轮集成修复的最终 SHA 必须全新独立完整红队，不能沿用批准。
预检/运行先核对两个 v2 数据集文档的完整版本/哈希/类别顺序；调用者相对路径解析一次。
新鲜 B 组分数严格关联后拟合，预测保留仅训练溯源；旧 v1 报告拒绝 v2 聚合/无分数/有分数输入，
v2 run 诊断如实不可用，显式版本 metrics API 保留。
