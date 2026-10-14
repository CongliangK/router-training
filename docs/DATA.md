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
