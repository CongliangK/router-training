# Expert-only task-v2 research bridge

Refs #18, Refs #19, Refs #20. These issues remain open.

## Contract and boundaries

- `research-training-v1` / `legacy7-v1` remains unchanged and requires default or
  explicit `router-label-v1`. Unversioned checkpoints remain seven-task products.
- `research-training-v2` / `task6-v2` requires explicit `router-label-v2` and the
  full `label_schema` metadata: question SHA-256 and ordered head classes. Neither
  the source-provenance integer `label_schema_version: 1` nor a six-column vector
  can implicitly opt into new semantics. No seven-task checkpoint is reinterpreted.
- `configs/research-v2.json` is an intentionally unbound **template**. Its five
  zero `input_sha256` values must be replaced in a reviewed new config with the
  SHA-256 of the exact bytes of `train.jsonl`, `calibration.jsonl`,
  `label-schema.json`, `split-report.json`, and the selected roles file (key
  `roles.json`). Only these five files are read by the shared input loader; no
  held-out/test data is opened or hashed. Review existing approved inputs before
  hashing; this interface does not authorize collection, preparation or training.
- Preserve `data_version` and `evaluation_protocol` as explicit identifiers.
  Neither the template identifiers nor user-supplied strings establish protocol
  approval. Changing an input, including whitespace or order, fails closed with
  instructions to review the change and regenerate the bindings in a new config.
  Validation never updates an approved config automatically.
- Launcher and runtime use `load_v2_inputs`: read each bound file once, hash those
  same bytes, parse and validate that snapshot, and retain those exact rows/roles
  for sampling and any separately authorized run. The launcher additionally sends
  the canonical config hash as `--research-config-sha256`; a config changed between
  launcher and runtime is refused. Runtime training recipe flags must match the
  config exactly, including budget, seed and optimizer choices.
- Every primary row needs an explicit v2 semantic declaration, nonempty id,
  prompt, original `group` and `split_group`, and resolved legal labels. Both
  train and calibration require `expert_provisional` or `expert_reviewed`,
  `evidence: prompt_only` and finite expert confidence in [0,1]. Measured shadow,
  outcome fields, conflicting source/status declarations and synthetic or
  supplemental flags at either task or label boundary are refused. Disabling
  confidence weighting does not disable provenance validation.
  An optional nested `label.id` must be a nonempty string identical to the outer
  task id. Optional outer `evidence` must also be `prompt_only`; conflicting
  declarations are never ignored. The existing expert source schema defines no
  explicit `confidence_semantics` marker, so any such declaration on the task or
  label is unsupported and rejected. In particular, measured
  `interval_confidence_level_not_sample_weight` cannot become an expert weight,
  even if an expert source tag was copied onto it or weighting is disabled.
  Direct weighting of explicitly v2 rows checks the same provenance contract.
  No source metadata is removed or rewritten, and omitted optional identity or
  evidence fields are not manufactured. This correction is v2-only.
- PR #40 isolation checks run before sampling: train/calibration id and component
  overlap, partial or malformed root declarations, and one original root spanning
  components are rejected. Existing A/B/C role, fold and support gates remain.
  Declared consistency is not proof of genuine independent roots or authentic
  labels. Real unique training support for every tier and vision class is
  mandatory even with prior smoothing or expected-sampler priors.
- **All v2 research `--supplemental` combinations are explicitly refused with the
  same error in both entry points, before opening the supplemental file.** The
  existing non-research v2 supplemental path is unchanged. The multiplier remains
  a versioned sampler setting, but v2 primary-only data contains no synthetic rows
  on which it could act. Synthetic test fixtures never count toward real quotas.

For a reviewed synthetic fixture, launch structural validation with:

```sh
python scripts/train.py --run new-unique-name --label-schema router-label-v2 \
  --research-config reviewed-research-v2.json --data-dir synthetic-fixture \
  --roles synthetic-fixture/roles.json --dry-run
```

The printed runtime command carries the complete recipe. Add
`--structural-dry-run` to that runtime command to exercise the same input contract
without a model or artifact creation. Structural v2 validation also works for
staged8bit configs without instantiating an optimizer; actual runtime availability
checks remain in place for separately authorized non-structural runs.

## Evidence scope and remaining acceptance

CPU synthetic tests cover compatible/crossed versions, full question identity and
class order, input/config fingerprints, original-root isolation, row provenance,
real-support preconditions, deterministic sampling and actual CLI/runtime
structural execution. Test-file read, model-load and artifact-write traps are
asserted. Optional toy tensor tests exercise six-class loss/gradients only when a
CPU torch installation already exists. Test skips must be reported separately.

This bridge does **not** complete blind real-label review or group-support audit
(#18), matched real training ablations or causal claims (#19), pinned-backbone,
PEFT/8-bit forward/backward/save/reload, smoke, GPU memory, speed or quality
acceptance (#20). Smoke and legacy calibration remain incompatible with research;
weight interpolation remains unavailable; no protocol or qualification gate is
relaxed. Task type remains diagnostic, quality gates remain shadow/unqualified,
and the #25 / PR #38 collection/preparation pause is unchanged.

## 中文契约与验收边界

Refs #18、Refs #19、Refs #20；不关闭这些研究任务。

`research-training-v2` / `task6-v2` 必须显式选择 `router-label-v2`，并绑定完整问题
指纹与类别顺序。v1/default/冻结配方和旧七分类 checkpoint 含义不变。来源整数
`label_schema_version: 1` 与任务语义字符串分属不同命名空间。

模板中的五个全零哈希故意不能执行。经审查后，在新配置中填写 train、calibration、
两个 schema 声明以及所选 roles 文件的精确字节 SHA-256；不读取或哈希 test。输入任何
变化都拒绝并要求重新审查/生成绑定，绝不自动改写批准配置。共享加载器只读取一次，
对同一字节快照做哈希、解析、校验，并将同一数据/角色用于采样及另行授权的运行。
启动器还绑定配置哈希，防止预检后配置被替换；运行时完整配方必须与配置一致。

train 与 calibration 都要求显式 v2、完整 id/prompt/group/split_group、已裁定标签、
专家 prompt_only 来源及有限置信度。measured 影子、执行结果来源、冲突声明、任何层级的
合成/补充主数据标记均拒绝；关闭置信度权重不能绕过来源检查。可选嵌套 `label.id`
必须是与任务 id 完全相同的非空字符串；显式外层 evidence 也必须为 `prompt_only`。
现有专家来源契约没有定义 `confidence_semantics` 标记，因此任务或标签上的任何显式
声明都拒绝；尤其不能把 measured 区间覆盖水平当作专家权重，关闭权重也不能绕过。
直接调用显式 v2 行的置信度加权同样校验来源，不删改元数据，也不补造可选字段。
这些修复仅作用于 v2。保留 PR #40 的行/组件/
原始根隔离、A/B/C 角色/划折与支持门槛；一致声明不是独立性或真实性证明。tier 和 vision
的每类都必须有唯一真实训练支持，平滑及采样先验不能补造支持。

该有界版本在两个入口以相同提示拒绝所有研究 `--supplemental` 组合，拒绝发生在文件
读取前；非研究 v2 补充数据路径不变。模板中的合成暴露乘数对纯专家主数据不生效。
所有测试仅为合成软件证据，不能计入真实标签或独立组配额。

CPU 结构校验不代表协议获批或训练/生产就绪；不进行真实标签复核、模型下载、训练、
GPU/显存/速度/质量实验。#18 的真实盲审与支持度、#19 的匹配消融和因果结论、#20 的真实
骨干/PEFT/8-bit 兼容性与 smoke/训练仍需独立验收。研究 smoke/legacy 不兼容、插值不可用、
shadow 资格门槛以及 #25 / PR #38 的采集/制备暂停保持不变。
