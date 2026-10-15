# Data format & building your own dataset

中文摘要见文末。The real training data (668 historical subtask prompts + expert
labels) is **not published** with this repository: even after automated redaction
the prompts still describe identifiable personal projects. Everything you need to
rebuild an equivalent dataset on your own task history is documented here.
An **aggregate profile** of the frozen dataset (class distributions, cross-tabs,
group structure, known skews, and a run-metrics comparison — no raw data) for
analysis without the raw rows is in [DATA-PROFILE.md](DATA-PROFILE.md).

## Dataset layout expected by the pipeline

`prepare.py` consumes a directory (default `data/`) containing:

```
data/
  tasks.jsonl               # one row per delegated subtask
  expert-recovery-1.jsonl   # expert labels, 4 recovery passes merged by id
  expert-recovery-2.jsonl
  expert-recovery-3.jsonl
  expert-recovery-4.jsonl
  synth/*.jsonl             # optional synthetic augmentation (train only)
```

and produces `train.jsonl`, `calibration.jsonl`, `test.jsonl`, `expert-labels.jsonl`,
`split-report.json`. Final splits have already been generated here and are frozen.
Legacy expert-only `prepare.py --data <dir>` still re-splits from scratch and
replaces these five regular output files, **only if no unified output exists**.
Unified bundles use the stricter fresh-directory policy below; never re-prepare
frozen data or change inputs in an existing unified bundle.

## Schema: `tasks.jsonl`

One JSON object per line:

| Field | Type | Meaning |
|---|---|---|
| `id` | str | Stable unique id (e.g. truncated SHA-256 of provenance) |
| `prompt` | str | The full delegated-task prompt text (redacted) |
| `description` | str | Short delegation description (may be empty) |
| `group` | str | Session/group id; tasks from one session share a group |
| `project` | str | Hashed working-directory (project identity) |
| `provenance` | obj | `{kind, session, call, tool}` — how the task was issued |
| `prompt_hash` | str | SHA-256 of whitespace-normalized prompt (dedup key) |

## Schema: `expert-recovery-N.jsonl` (labels)

| Field | Type | Constraints |
|---|---|---|
| `id` | str | Must match a tasks.jsonl id; ids unique across all 4 files |
| `tier` | str | `fast` / `balanced` / `flagship` |
| `task_type` | str | one of `extraction research implementation debugging review visual creative` |
| `needs_vision` | str | `yes` / `no` / `uncertain` |
| `confidence` | float | 0–1, confidence **in the tier judgment** (not difficulty) |
| `rationale` | str | one sentence; flagship must name which of the 5 rubric cases |
| `label_source` | str | must be `expert_provisional` (enforced by prepare.py) |
| `evidence` | str | `prompt_only` — labels never use model-execution outcomes |

Labeling rules: see [`data/labeling/RUBRIC.md`](../data/labeling/RUBRIC.md).
Core principle: `balanced` is the default; `flagship` only for the five enumerated
cases (architecture design, novel methodology, cross-module root cause,
safety/concurrency correctness, adversarial final review). Long prompts are never
an upgrade reason.

## Unified cold-start schema and measured shadow (issue #8)

The legacy command and its expert-only split algorithm remain unchanged. Opt in
with `--unified` (expert-only is valid), or supply `--outcomes` to enable the
unified outputs automatically. All tasks still require explicit prompt-only
expert labels in the four recovery files; measured-only tasks without these
understanding annotations fail closed. A measured selected tier is a candidate
selection outcome, **not** a task-understanding annotation. Nothing derives
`needs_vision`, `task_type`, or expert tier from success/failure, cost or routes.

```sh
# Structural preflight: standard library only, no torch/models, no writes/splits.
python -S prepare.py --data /path/to/new-data --outcomes /path/to/outcomes.json \
  --routes /path/to/candidate-routes.json --preflight
# Prepare a NEW dataset directory; ALL existing prepare outputs are refused.
python prepare.py --data /path/to/new-data --outcomes /path/to/outcomes.json \
  --routes /path/to/candidate-routes.json
# Expert-only unified output is also supported.
python prepare.py --data /path/to/new-data --unified
```

`--outcomes` accepts the **full schema-v2 audit JSON from `outcomes.py`**, not a
bare measured-label JSONL or an execution plan. Explicit `--routes` must match
the preregistered candidate-set fingerprint. Structural validation replays the
existing exact one-sided lower-bound gate from candidate metrics and trial
metadata; it checks completeness, source/evidence, candidate operating-point
fingerprints, the deterministic selection rule, and gate/audit consistency.
Missing/conflicting sources, duplicate task IDs or IDs within either label
source, incomplete expert coverage/understanding, and ungated or tampered
measured labels are rejected. The same task ID across expert and measured
sources is an intentional join, not a duplicate. Pending outcomes stay pending;
they never create labels. An envelope must cover the supplied task corpus,
including its pending IDs. Every input tree, including arbitrary nested/ignored
metadata in tasks, expert labels, routes and the full outcomes envelope, must be
finite UTF-8 JSON (no NaN/infinity, unserializable objects or lone surrogates).
Preflight checks this and output-path conflicts as well as structure, **not**
split feasibility, execution authenticity, preregistration timing or independence.

### Output lifecycle and migration

- `--unified` or `--outcomes` requires **none** of the eight named prepare outputs
  to exist: the five legacy files above plus the three files below. Even one
  unknown/partial file, directory, or dangling symlink at those names is refused;
  new unified outputs are opened exclusively, never overwritten.
- If **any** of the three unified output names exists, **all modes**, including
  ordinary expert-only and `--preflight`, refuse that directory. An expert-only
  rerun cannot silently retain old measured evidence or clear just its shadow.
  Fresh expert-only `--unified` creates empty shadow and null outcomes, and does
  not create `measured-evidence.json`.
- To migrate legacy outputs to unified schema, change corpus/evidence, or rerun a
  unified bundle, create a new directory and copy only the input `tasks.jsonl`
  and four `expert-recovery-N.jsonl` files (plus optional input `synth/`). Supply
  the intended outcomes/routes explicitly. Do **not** copy old generated outputs
  or replace inputs in the old bundle. No force flag, inferred ownership, automatic
  deletion, or destructive cleanup is provided; unrelated filenames are untouched.
- Without unified files, default expert-only runs keep the historical five-file
  overwrite behavior and byte format for archived reproduction. Symlinks and
  directories at those five output names are refused before any writes.
- All expected validation and JSON/UTF-8 serialization errors occur before any
  output is opened; every output is serialized in memory first. This is **not a
  multi-file filesystem transaction**: I/O errors, crashes or concurrent writers
  can still leave partial outputs. There is no locking or input snapshot. Use one
  writer and stable inputs per directory; inspect/preserve a partial directory
  and retry in a new one. No files are automatically removed after failure.

Additional outputs (all local data, subject to the same privacy boundary):

| Output | Contents |
|---|---|
| `unified-labels.jsonl` | One task ID per row; `schema_version: 1`, explicit expert `task_understanding` (complete original label with `label_source`), nullable original `measured_candidate_outcomes` label, `cold_start_training_source: expert_provisional`, and nullable `candidate_success_probabilities` |
| `measured-shadow.jsonl` | Only gate-passing measured rows; original task, measured `label`, separate expert `task_understanding`, `partition: measured_shadow`, expert split/group association, complete per-task candidate metrics/trial details and bound/preregistration metadata |
| `measured-evidence.json` | Complete original outcomes envelope, including pending records, routes fingerprint, trial/gate metadata and estimand (only written when `--outcomes` is supplied) |

The train/calibration/test files and `expert-labels.jsonl` remain **expert-only**,
byte-identical for the same expert/tasks/seed regardless of measured evidence.
Shadow records retain both the route operating-point `route_fingerprint` and the
candidate-set `preregistration.routes_fingerprint`. Trial details are local audit
evidence, not public examples. Existing training consumes none of these new
files; schema unification does not authorize measured losses or success heads.

Cold start trains the task-understanding layer on expert judgments, using
measured data only for validation/shadow comparison. Expert `confidence` is
confidence in the tier judgment; measured `confidence` is an interval confidence
level and **must never become an expert-confidence sample weight** (#19).
`candidate_success_probabilities: null` reserves a future conditional-success
head; empirical rates and lower bounds stay in the measured evidence and are
not silently promoted to calibrated probabilities. Genuine measured supply
and validity/independence acceptance remain in #26. Synthetic CPU fixtures
verify plumbing only and must not be described as real measurements. Frozen
v3c config/artifacts and the training/serving paths are untouched.

中文：新增 `--unified` / `--outcomes` 统一可溯源入口，默认冷启动仍只用专家标签训练，
measured 单独进入影子分区，绝不自动混入 loss。每个任务仍须四轮文件中的显式专家任务理解
标签，不能由 outcome 或实测选择的 tier 臆造 needs_vision/task_type/专家 tier。
`--outcomes` 必须提供完整 v2 审计 JSON 和显式 `--routes`；预检重放既有门禁并核对
来源、指纹、候选完整性、试验与门禁元数据，拒绝缺失/矛盾来源、源内重复 ID、缺失理解字段
及未过门禁结果。专家与实测同 ID 是有意关联。`--preflight` 仅 stdlib、无 torch/模型、
不切分也不写文件；不证明切分可行、真实性或独立性。实测 confidence 是区间置信水平而非
专家样本权重。未来条件成功率字段当前为 null。新增文件均为本地敏感数据；mock 测试不是
真实供给，真实 measured 有效性由 #26 验收。既有 v3c 与专家切分保持不变。
统一模式要求上述八个输出名均不存在；已有任意统一输出时，所有模式（含预检和普通专家
模式）均拒绝，绝不自动删除或部分清理旧数据。迁移/换语料/换证据/重跑须新建目录，仅复制
输入任务与四轮专家标签（及可选 synth），不可复制旧输出或在旧包内更换输入。无统一输出时，
默认专家模式仍按历史行为覆盖五个普通文件，但拒绝输出符号链接/目录。任意嵌套元数据也须
可序列化为有限数值 UTF-8 JSON；所有校验/序列化完成后才打开输出。并非多文件事务：I/O
失败、进程中断及并发仍可能留下部分包；无锁/输入快照，不自动回滚删除，须保留检查后换新
目录重试。

## Schema: `synth/*.jsonl` (optional augmentation)

Rows need only `prompt`, `description`, `group`, `project`, and an embedded
`label` object (same fields as above, `label_source: "synthetic_vision_augmentation"`).
Synthetic rows join **train only** — never calibration or test — so temperatures
and the tau operating point are not fitted on synthetic lexical shortcuts.

## Splitting guarantees (what `prepare.py` enforces)

- Splits are **group-disjoint**: all tasks from one session (or near-duplicate
  cluster, 5-gram Jaccard ≥ 0.88) land in exactly one of train/calibration/test.
- Tier/vision coverage constraints per split; ~70/15/15 ratio by groups.
- 12,000 random assignments are scored; the best-scoring split wins (seeded).

Worked example rows (clearly synthetic, format-only) are in
[`data/synthetic-examples/`](../data/synthetic-examples/).

## Building your own dataset

1. **Extract prompts** from your own multi-agent session logs. `extract.py`
   implements a read-only extractor for a `--manifest` of session JSONL paths:
   it pulls every delegated subtask prompt (`subagent*` / `spawn_teammate` calls
   and workflow-child initial prompts), applies credential redaction
   (private keys, tokens, API keys, emails, home paths), hashes provenance, and
   deduplicates. It never exports historical model selections.
2. **Label** the prompts with the rubric, by hand or with a strong model acting
   under `data/labeling/RUBRIC.md` — 4 passes (recovery files) let you reconcile
   disagreements; `prepare.py` validates the merged result.
3. **Split**: `python prepare.py --data <your-dir>`; check `split-report.json`
   for group disjointness and tier coverage.
4. **Train**: `python scripts/train.py --run <your-run>` (needs base weights in
   `models/intern-decision-0.8b/`).

> Automated redaction is best-effort, never a guarantee (`extract.py` says so in
> its own report). Review extracted prompts before sharing anything.

---

## 中文摘要

真实训练数据不随仓库发布。`tasks.jsonl` 每行一条委派任务（prompt/组/溯源），
`expert-recovery-{1..4}.jsonl` 为专家标签（tier/needs_vision/task_type/置信度/理由），
`synth/*.jsonl` 为仅入训练集的合成增广。`prepare.py` 做组隔离切分
（会话组 + 近重复簇不跨集），产出 train/calibration/test。自建流程：
用 `extract.py` 从自己的多智能体会话日志只读提取并脱敏 → 按
`data/labeling/RUBRIC.md` 标注四轮 → `prepare.py` 切分 → `scripts/train.py` 训练。
格式示例见 `data/synthetic-examples/`。冻结数据集的聚合画像（类分布、
交叉表、组结构、已知偏斜与运行指标对照，不含原始数据）见
[DATA-PROFILE.md](DATA-PROFILE.md)。

## frozen-v1: known skew / 已知偏斜

The current historical test split is frozen-v1, known skew. It has already
been inspected; it is not a pristine independent holdout. Do not choose a
checkpoint, hyperparameter, temperature, operating threshold, or routing
policy using its scores or deletion results. Use calibration/new groups.
Future data should support a natural-distribution set plus a separately
reported stratified challenge set; neither exists merely because this report
code exists.

The historical aggregate split report has only 4 vision-positive training
rows versus 74 test positives, and only 2 fast test rows. Issue #4 records that
79 of 104 test rows (70 vision positives) come from one group. These are
historical aggregate facts, not newly replayed results. Vision results must
be read separately: they are mainly supported by a single task group.
Resampling few groups does not create independent sources. Expert-label
agreement does not establish candidate-model task success.

CPU-only diagnostic command (standard library, no model/torch required):

```sh
python scripts/report_split.py --split-report data/split-report.json
python scripts/report_split.py --split-report /path/to/split-report.json \
  --predictions /path/to/fixed-test-predictions.jsonl --output /path/to/new-report.json
```

Without explicit rows, group sizes, performance, bootstrap, and deletion
results are unavailable. JSONL rows require `label.tier`, `label.needs_vision`,
`label.task_type`, and `split_group`. Labels alone allow group support counts;
metrics require `answers` with each supported field's `choice`, `confidence`,
and complete fixed-schema `probabilities`. Missing group IDs are never
replaced with row IDs. Optional `group` is the original root identity from
extraction; it is distinct from the merged `split_group`. Missing roots are
unavailable, never inferred. Group/root counts describe supplied provenance,
not proven statistical independence. The report exports aggregate counts
and stable report-local group/root aliases only, not prompts or source IDs.
Review even aggregate reports before sharing; small groups can be sensitive.

Every supplied group is deleted in descending size from the SAME fixed
predictions. No refit, retraining, tuning or LOGO cross-validation occurs.
Each deletion reports remaining class/group/root support, nullable metrics,
deltas and overall defined-score ranges. Absent support stays null, including
zero/single-group bootstrap cases. Vision recall is separate from tier and
task-type scores. Bootstrap uses the shared metric protocol (1,000 whole-group
resamples, seed 0), with undefined resamples and support disclosed. The legacy
metric key `independent_groups` counts observed clusters; independence is not
verified. CLI input SHA-256 digests record reproducibility, never file paths.
Existing output paths are refused. Supplied rows are explicitly test rows;
the caller must provide the right version and fixed operating point. A row
count comparison with the aggregate test count is not dataset authentication.

New training reports append `frozen_test_diagnostics`: legacy evaluation
includes baseline and tuned fixed-score sensitivity; grouped-dev attaches
aggregate-only unavailable test metrics and never reads test rows for this
appendix. Old artifacts and the split are untouched. Real private-data replay
and provenance verification remain separate acceptance in #27.
