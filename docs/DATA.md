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
`split-report.json`. Final splits have already been generated here and are frozen;
re-running `prepare.py --data <dir>` re-splits from scratch (overwrites outputs).

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
