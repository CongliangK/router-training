# Synthetic snapshot-to-OOF draft adapter

Refs #43, Refs #21, Refs #42. These scientific/real-data issues remain open.

## Scope and hard boundary

`synthetic_oof_adapter.py` is a **draft-only software adapter** for explicitly
invented in-memory fixtures. It has no CLI, file/path argument, file discovery,
training, sampling, model loading, model-byte verification, network, approval or
qualification operation. It does not change the existing research, OOF, source,
calibration, serving or policy interfaces. #25 and PR #38 remain paused.

Two APIs are available:

- `build_synthetic_draft(payloads, config, protocol)` validates the supplied
  synthetic input profile and produces a bound draft.
- `validate_synthetic_records(payloads, config, protocol, draft, records)` rebuilds
  that draft from all original inputs and checks complete invented per-row,
  per-epoch OOF lineage, including all three v2 head widths. It reports software
  consistency only, with no accuracy, risk bound, improvement or real support claim.

All outputs retain `qualified: false`, `approval_verified: false`,
`real_root_audit_verified: false`, `training_allowed: false` and
`formal_admission: "NO-GO"`. The builder only accepts `status: "draft"` and literal
`protocol_approved: false`; adding an approval string or relabeling the version
cannot enable formal use. Its output is neither a valid research training config
nor an original calibration roles config. There is no approved adapter version.

The synthetic declaration is an input trust boundary, not proof of origin. A
metadata-only function cannot recognize real rows dishonestly supplied as invented
fixtures. Do not call it with real/private assets. Fixture expert tags simulate
PR #41's schema; they do not certify actual expert review. The adapter never
removes provenance or converts synthetic/measured labels into expert supervision.

## Input profile and source/rejection matrix

`payloads` must have exactly five keys whose values are already supplied `bytes`:
`train.jsonl`, `calibration.jsonl`, `label-schema.json`, `split-report.json`, and
`roles.json`. The caller owns those invented bytes. No paths are accepted and no
files are opened. One local mapping of immutable bytes is retained before hashing
and parsing, so later caller mapping edits cannot swap the consumed snapshot.
Each JSON/JSONL document is decoded strictly: duplicate object
keys, invalid UTF-8, nonfinite numbers and invalid Unicode are refused, including
metadata not used for scoring.

| Field / evidence | Source and binding | Rejection boundary |
| --- | --- | --- |
| Five input identities | SHA-256 of the exact supplied bytes equals `config.input_sha256` | Missing/extra keys, path/string inputs, or byte changes, even whitespace |
| Research config | Existing PR #41 validator, explicit research-v2/task6-v2/router-label-v2; `protocol.config_sha256` uses existing `research_training.config_hash` | Cross-version schema/config, changed config, unknown config keys |
| Task semantic identity | Both schema documents and complete question fingerprint/ordered classes | V1, inferred six-column identity, reordered classes or incompatible declarations |
| Source integer | `label_schema_version` retains the existing independent integer-1 meaning | Boolean or semantic-version substitution |
| Original task identity | Required id, prompt, group, split_group, task_label_schema_version and label | Missing/duplicate IDs, undeclared fields, illegal/excluded labels |
| Expert provenance | Existing PR #41 validator; `expert_provisional` / `expert_reviewed`, resolved, prompt_only, finite confidence | Unknown/measured/outcome sources, unsupported confidence semantics, conflicting task/label identity, synthetic/supplemental primary flags; confidence-off does not bypass it |
| Original source split | Exact containing snapshot key `train.jsonl` or `calibration.jsonl` | Extra test input, explicit alternative source fields or audit/source mismatch |
| Original role plan | Original `roles.json` is hashed and validated with existing A/B/C support and isolation | Unknown roles config, component/root leakage, insufficient original support |
| Audited root candidate | Separately supplied per-ID synthetic root declaration | Missing evidence/rows/roots, duplicate/foreign IDs, wrong group/component/source, one component split across roots, one root crossing source splits or distinct original calibration components (including within one role) |
| Nested roles | Separately supplied per-outer-fold fit/A/B/C root lists, checked by existing `oof.nested_plan` | Missing, empty, overlapping, foreign or outer-validation roots; insufficient class-root support |
| Epochs | Ordered complete positive integer range 1..config.training.epochs | Missing, duplicate, boolean, reordered or changed schedule |
| Model identities | Explicit `(fold, epoch, model_digest, source)` registry, SHA-256-shaped placeholders | Missing/duplicate/foreign fold-epoch, noninteger fields, unknown/real source, bad hash |
| OOF records | Exact row/source/root, nested plan digest, draft digest, model registry, fit roots, task schema | Missing/duplicate/foreign row-epoch, validation-root training leakage, changed model or lineage |
| Logits | Explicit finite tier=3, vision=2, task_type=6 arrays in bound class order | Missing/extra heads, boolean/nonfinite values, old seven-task vectors |

Root names are never inferred from `group` or `split_group`. A caller can explicitly
supply the same string, but equality is not evidence of independence. Audit IDs,
evidence references and matching hashes only establish declared consistency. Real
root/near-duplicate audit and human approval remain absent and formal use blocked.

The narrow calibration profile requires a one-root/one-component correspondence.
An audited root spanning two distinct original calibration `split_group` components
is rejected even if both components belong to the same A, B or C role. Otherwise
the old B fit/holdout folds could share that root and old component support counts
could overstate audited-root support. The adapter does not recompute or certify
that incompatible original plan; its error requires review of the original
component/role snapshot. Multiple original `group` values within one component
and its one declared root remain supported. This is a synthetic structural rule,
not a successful scientific or independent-source audit.

### Deliberately bounded snapshot profile

This first draft accepts the following row fields. Required fields are listed in
the matrix. Optional task fields are `label_schema_version`, `label_source`,
`evidence`, `label_status`, `synthetic`, and `supplemental_only`. Labels require
`tier`, `needs_vision`, `task_type`, `label_source`, `label_status`, `evidence`, and
`confidence`; optional label fields are `id`, `rationale`,
`task_label_schema_version`, `label_schema_version`, `synthetic`, and
`supplemental_only`. Existing PR #41 checks apply to all allowed provenance fields.
Every other field is rejected, including cached logits or unknown root declarations.

`split-report.json` has exactly `label_schema`, `task_label_schema_version`,
`label_schema_version`, and `splits`. `splits` has exactly `train` and
`calibration`, each with one integer `rows` matching the supplied snapshot.
This is an invented development declaration, not a full preparation-report
migration. Full reports with held-out summaries or other fields are unsupported
and fail closed. The adapter does not read, trim, reinterpret or approve such a
report; it must not be used to prepare a real-data bridge by dropping fields.

The original `roles.json` bytes are retained. This adapter imposes a local strict
subset of the inherited override schema, then runs the unchanged inherited
provisional policy/protocol and role/support validation. Optional `protocol` and
`support_policy` must be objects; omitted or empty objects keep existing defaults,
while explicit null, wrong types and unknown keys are rejected, never removed.

- `protocol` permits only `tau_grid` (nonempty list of finite numbers >= 0),
  `temperature_bounds` (two finite numbers with 0 < low < high),
  `fit_max_iterations` (positive integer), `fit_tolerance` (positive finite number)
  and `probability_epsilon` (finite number strictly between 0 and 1).
- `support_policy` permits only `min_groups_per_role` (exact A/B/C positive-integer
  mapping), `min_rows_per_class`, `min_groups_per_class`,
  `min_fold_train_groups_per_class` (positive integers), and `n_folds` (integer >= 2).
- Legal partial numeric overrides retain existing semantics and remain unapproved.
  Booleans are not integers/numbers. Approval/qualification fields and callable
  `fit`/`score`/`selector` extensions are not supported by this draft profile.

No shared validator, gate or fallback is changed. The original A/B/C roles and the
new nested fit/A/B/C proposal are distinct plans. The latter is never generated
from the former; approving either remains human work.

### Draft protocol schema

Every key is required and unknown keys are rejected:

- `version`: `synthetic-oof-draft-v1`
- `status`: `draft`
- `synthetic_fixture`: literal `true`
- `protocol_approved`: literal `false`
- `config_sha256`: exact existing research config hash
- `root_audit`: object described below
- `n_folds`, `min_class_roots`: existing positive-integer OOF feasibility inputs
- `nested_role_roots`: one exact `{fit, A, B, C}` root-list object per outer fold
- `epochs`: complete ordered schedule matching the bound config
- `models`: exactly one object per fold/epoch, with integer `fold`, integer `epoch`,
  `model_digest` and `source: "synthetic-placeholder"`

`root_audit` has exactly `version: "synthetic-root-declaration-v1"`,
`status: "synthetic-unverified"`, nonempty `evidence_ref`, exact five-entry
`input_sha256` and `entries`. Every entry has exactly `id`, `source_split`,
`group`, `split_group`, and explicit `root_id`, with complete one-to-one row
coverage. The evidence reference is an invented metadata reference; it is never
opened and is not independently verified. Missing declaration blocks the software
adapter; absent real audit always blocks formal admission.

### Record schema and invalidation

Records have exactly `id`, `source_split`, `root_id`, integer `fold` and `epoch`,
`plan_digest` (the nested plan), `draft_sha256` (the adapter envelope),
`model_digest`, `training_roots`, `task_label_schema_version: "router-label-v2"`,
and `logits`. The fit-root list must exactly match the canonical nested plan.
The adapter validates all three heads, then delegates tier/vision lineage and
coverage checks to the existing OOF validator. Task type remains diagnostic.

The draft binds every input byte, complete config, audit, original role plan,
nested roles, model registry and epoch schedule. Validation reconstructs the
expected draft and compares the entire canonical value, so editing a draft and
recomputing its hash is insufficient. Editing/rebinding the original inputs creates
a different unapproved draft and invalidates old records. Nothing is automatically
repinned. Hashes are neither signatures nor a trusted approval/timestamp system.

## CPU validation and remaining acceptance

`tests/test_synthetic_oof_adapter.py` contains a fully invented fixture and actual
positive/negative calls to both APIs. The group layout is chosen only for software
coverage. It is not the historical development pool and supplies none of the real
uncertain/creative/collection/independent-root quotas.

Run `python -m unittest discover -s tests -p test_synthetic_oof_adapter.py -v` and
the repository's full CPU suite. Tests check immutable inputs, all binding changes,
unknown/missing fields, malformed JSON, source/schema drift, original and nested
role leakage, support failure, complete coverage, invalid logit vectors, failed
approval escalation and file/network/model/sampling traps. Optional tensor and
private-archive skips must be reported separately; a skip is not an actual pass.

- Software A: synthetic draft contract and applicable CPU checks only; final
  publication still requires exact-head independent review and actual verification.
- Environment B: real root audit, authentic labels, checkpoint/OOF execution,
  fresh fixed-model calibration, independent confirmation and parity are UNRUN.
- Scientific C: formal role/support/statistical/budget protocol is unapproved.
  #2/#18/#21/#42 and their original scientific criteria remain open. #44 may use
  this software field contract only after review/freeze; that does not approve
  a scientific protocol or authorize a real matched training experiment.

## 中文边界

本实现只把显式传入的五份内存合成快照、配置和独立的合成根审核声明，适配为未批准的
OOF 软件草案。原 calibration A/B/C 与每折 nested fit/A/B/C 分别绑定，不能互相改名；
根来自逐 ID 明示声明，不从 group/split_group 推断。新审核只要把原 calibration 的不同组件合成同根就拒绝，包括同一 A/B/C 角色内，
防止原 B 折隔离或支持度失效；同组件内多个原始 group 保留。roles 的 protocol 与
support_policy 只允许本文列出的原数值字段及合法部分覆盖，拒绝未知/批准字段，不裁剪输入。
输入、来源、配置、审核、角色、模型或 epoch 改动均使旧草案/记录失效，所有未知字段拒绝。

当前仅支持本文的窄合成输入 profile；完整 prepare 报告（含 test 摘要或额外字段）不支持，
不能自行删字段做真实适配。合成 envelope 声明和专家标签只是虚构 fixture 元数据，代码
不能证明真实性；不得把真实私有输入传入此 API。没有文件发现、test 读取、模型下载、
采样、训练、正式批准或资格升级入口。始终 qualified=false、approval_verified=false、
real_root_audit_verified=false、formal_admission=NO-GO。真实审核/科学协议仍需人审；
合成测试不抵真实配额，不代表模型收益，不恢复 #25 / PR #38。
