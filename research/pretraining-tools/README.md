# Pretraining closeout — PAUSED / NOT READY

Refs #2, #18, #19, #20, #24, #25, #27. No research or production acceptance issue is completed by this archive.

## Owner decision, 2026-10-16

Pause additional collection and preparation until naturally generated visual tasks provide more genuine independent sources. An independent reviewer is the intended labeling arrangement; no reviewer identity or completed labels are assumed. No monitoring, automatic resume, GPU check, smoke run or training is scheduled. Resume requires a new owner request and explicit scope.

## Completed safe preparation

- Reviewed public source base: `cd058a8093977865691b867c360bfc64b4e29f88`, tree `83c0aa70158ab69b2e9bb08ac2b0c87d85658981`.
- A source-only private candidate contains 128 files, individually checked against that public Git tree. No private history, real dataset or model weights were attached to it. This is not an in-place private-runtime upgrade.
- Local programs inventoried source/development data and inspected three environments through distribution metadata, without importing their model libraries. Only aggregate observations leave the private workspace.
- A new development diagnostic review package has **36 pending cases**: four cached-label conflicts, 16 creative diagnostics and 16 controls. The reviewer wrapper omits prior labels, candidate predictions, root names, original IDs and selection reasons. Intrinsic prompt privacy/blinding still needs local checking. No label was assigned automatically.
- The package is **not** acceptance of the original issue-18 historical 16 cases or all six uncertain cases. No pristine holdout was established. The original test split and candidate predictions were not opened by these preparation helpers; the full existing source-label pool was counted.
- Local repaired preparation-tool variant: **19 CPU cases passed**. The unchanged public training source was separately rechecked with the native offline/GPU-hidden suite: **407 cases, 404 passed, three optional-private skips, exit 0**. These results are not real-model compatibility or quality evidence.

Initial tool review found three robustness defects: duplicate/foreign review IDs distorted completion counts, altered integrity-sidecar keys bypassed read confinement, and recovery wrapping dropped explicit synthetic flags. The repaired revision restricts completion to unique expected IDs, validates the complete input path scope before byte reads, and preserves explicit task/recovery/nested synthetic flags. Nine regression cases supplement the original ten tests. Initial evidence remains private and is marked superseded, not erased. No tested defect authorized training; the checker remains explicitly BLOCKED.

## Why training stays blocked

| Local aggregate observation | Consequence |
|---|---|
| 668 source tasks, 37 original root labels; vision=yes occurs in only five original roots | Current default A/B/C policy already requires at least six distinct positive roots across the three roles, before train/test reservations and B-fold constraints. Six is only a necessary lower bound, not sufficient support. Root labels do not authenticate independence. |
| Existing split-v2 train/calibration share six original roots despite disjoint IDs/fine components | Fine-component counts are not independent original roots. One train root was partitioned into 88 fine components. Do not adopt this bundle or fragment roots to pass gates. |
| Four cached semantic-label differences from current recovery labels, split 2 train / 2 calibration | Newer labels are not automatically correct; independent evidence and version/provenance review remain necessary. |
| All 36 diagnostic decisions remain pending; existing source labels lack explicit semantic review metadata | Material generation is not label validation or completed issue-18 review. |
| No inspected environment matches all pinned requirements; the documented private default environment was absent | No dependency edits, installation or download was performed. Metadata is not driver/kernel/model compatibility. |
| Semantics, entrypoint, whole-root bundle, roles, numeric policy and matched budget unapproved | Do not adopt the 256-row development example or combine taxonomy/loss/sampling/DoRA changes as one factor. Six-class research/split integration is still unsupported in the recorded base. |
| No actual model/adapter/reload/peak-memory test | Future GPU compatibility checks and any training need separate permission. No quality improvement is established. |

The source-minus-historical-development complement was counted only; it is not verified test membership. No private split, historical run, frozen recipe or manifest was overwritten. Aggregate counts and metadata cannot certify authenticity, causal effect, calibration validity or production qualification.

## Source-only historical tools — not installed training commands

This folder archives five helper sources, **not private materials or a new supported public training interface**:

- [Inventory](audit_local.py): fixed-vocabulary counts and confined local reads.
- [Material builder](build_preparation.py): pending-only development review packaging, metadata inventory and a non-executable proposal.
- [Source exporter](export_candidate.py): fixed-base source-only Git-blob verification; no model weights or private history.
- [Readiness checker](check_readiness.py): snapshot consistency and explicit BLOCKED output. It never launches training and is not a global interception of old launchers.
- [Tests](test_preparation.py): six toy cases, nine repair regressions and four private-material consistency cases.

Four Python files are byte-identical to the repaired local sources. The builder differs only in two deliberately disclosed path substitutions: an individual home directory becomes `Path.home() / 'miniconda3'`, and a historical roles path becomes `pretraining-approved-roles.json` under the private project root. No personal paths, actual review rows/IDs, private integrity sidecars, original data, logs, model files or private Git history are distributed.

The sources retain the session's fixed base SHA and private-layout assumptions. They require a future reviewed placement under a NEW private `pretraining/<case>/` directory with an authorized sibling public checkout at the recorded base; they are not portable production CLIs. **Do not execute helper mains against data from this public archive or change guards simply to force a run.** A later preparation session must review integration and create a new immutable material version. Generated outputs are intentionally excluded from this archive.

Synthetic-only checks, from this folder:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python -B -m unittest test_preparation.ToyOnly test_preparation.RepairRegressions -v
```

Do not run the four private-material cases here: their intentionally undistributed fixtures are absent. A normal local readiness rejection exits **1**; execution/integrity errors exit **2**. Neither is a training PASS. Expected-file hash matching certifies the recorded 128 files, not absence of additional files elsewhere. Reviewer identity/truth and source independence are not machine-authenticated.

---

## 中文交接

按维护者要求暂停，等待今后自然产生的真实视觉任务；独立审核者尚待指定。今天固化的是安全准备工具源码、19 项工具测试和明确的 NO-GO 结论，不是可开训证明。

现有五个视觉正例根无法满足默认 A/B/C 至少六根的必要条件；已有切分共享六个原始根，不能用拆细组或合成样本补支持。36 条开发材料仍全待审，也不是原历史案例审核。真实数据、私有历史、旧运行和原文盲审包均保留本机，未随源码归档发布。

恢复前需补真实独立来源、完成本机独立盲审、批准语义/分组/预算及环境方案。任何 GPU 兼容性小检或训练另行授权；不承诺重训质量提升。本次暂停不安排后台补采或自动开训。
