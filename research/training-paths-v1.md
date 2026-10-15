# Research training paths v1 — development subset, not results

Refs [#19](https://github.com/CongliangK/router-training/issues/19),
[#20](https://github.com/CongliangK/router-training/issues/20),
[#5](https://github.com/CongliangK/router-training/issues/5). **All remain OPEN.**
This public CPU development implements candidate mechanics, not the forbidden
actual training/smoke, model loading, private-data diagnosis or GPU validation.
No measured quality, speed, memory, parameter count or backbone compatibility is
claimed. The shipped data/protocol identifiers are explicitly unapproved
synthetic-development placeholders, and budget 256 is a configurable example,
not the historical data size or an approved matched-experiment budget.

## Explicit API / preservation boundary

The opt-in launcher flag is `--research-config configs/research-v1.json`.
It is grouped-only, mutually exclusive with `--legacy`, `--recipe`, `--smoke`.
No flag means existing behavior, including the complete frozen v3c recipe,
remains unchanged. Neither global QUESTIONS nor `model_utils.request_for`,
serving, manifests, archived configurations/artifacts or dependency pins change.
`train_eval.py --research-config PATH` also accepts the flag, but **every existing
training CLI value must match** the config's complete `training` object; use the
launcher to avoid mismatch. Unknown keys, versions and nonfinite numbers fail.

Safe structural example (requires caller-owned synthetic fixtures, not real data):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/train.py --run new-research-preflight --roles /synthetic/roles.json \
  --data-dir /synthetic --research-config configs/research-v1.json --dry-run
```

This validates grouped roles and the same sampler/prior contract as runtime,
prints a command, and never invokes it, opens test, creates a run or checks
backbone compatibility. A nondefault `--data-dir` remains preflight-only. No
actual launch command is recommended until external authorization and protocol
freeze. Research v1 intentionally refuses smoke rather than implying those
acceptance paths were exercised.

Public Python APIs (stdlib import unless tensor execution is requested):
`research_training.load_config`, `sample_epochs`, `sampler_plan`, `additive_bias`,
`expected_cost`, `confidence_weight`, `head_weights`, `parameter_groups`,
`row_loss`; `research_capacity.enumerate_dora_targets`, `dora_config`,
`verify_dora_injection`, `staged_parameters`, `apply_stage`, `install_neftune`.
The tensor loss and capacity operations import torch lazily.

## #19: versioned distribution and objective

- `sqrt-group-v1`: with-replacement row weight `n_split_group ** -0.5`.
  Expected group mass is proportional to sqrt(group size), **not** equal groups,
  natural uniform rows or tier balancing. `epoch_rows` controls the actual row
  stream. Local seeded RNG is deterministic; repetition/missing rows within an
  epoch are possible. No hardcoded 669 or tier quota is used.
- Explicit synthetic rows marked `synthetic: true` with expert labels and
  balanced/vision=yes receive probability multiplier 1.5 (configurable). This is
  sampling exposure only, never another loss multiplier. This deliberately
  exposes balanced+vision without making vision a balanced-tier rule. The old
  synthetic join loader is disabled on research paths; supply a new reviewed
  train pool, never synthetic calibration/test. All required classes need real
  unique expert train support even for a synthetic structural fixture.
- Reports record natural uniform probabilities, target per-row probabilities,
  achieved per-epoch frequencies, row-order digest, group count and prior
  provenance. They omit row/group IDs and prompts (digests still require sharing
  review). Default `unique-natural-expert` priors exclude synthetic rows; duplicate
  IDs are rejected before sampling. Optional `expected-sampler` priors describe
  the expected reweighted distribution **including** synthetic exposure. Both
  record pseudocount smoothing in count units and data version. This is a
  declared candidate choice, not an assertion one prior is universally correct.
- `additive-la-v1`: CE uses **z + tau * log(pi)**, with tier tau=1 and vision
  tau=.75 candidates. For zero binary logits and pi=(.8,.2), adjusted probabilities
  are (.8,.2), proving the training sign. This is not inference's subtraction;
  raw logits continue into existing grouped calibration. Decision-side tau and
  its natural prior remain separately recorded/selected; combined compensation
  needs matched ablations, not an automatic claim of eliminating double correction.
- `asymmetric-tier-v1`: full true-row × predicted-column relative-regret matrix:

  | true \\ predicted | fast | balanced | flagship |
  |---|---:|---:|---:|
  | fast | 0 | 1 | 3 |
  | balanced | 1 | 0 | 1.5 |
  | flagship | 5 | 5 | 0 |

  Diagonal is no error; fast→balanced costs one provisional budget unit,
  fast→flagship three for an unnecessary two-tier escalation;
  balanced→fast costs one for a bounded-execution miss;
  balanced→flagship 1.5 for the observed overpromotion concern;
  either flagship downgrade costs five because both miss the requested top-tier
  capability. Equal 5/5 deliberately does not assume a measured severity gradient.
  These are **preregisterable hypotheses, not dollars, prices or task-success
  estimates**. Changing cells requires a new version, not silent v1 editing.
  Cost term is `lambda * sum_j softmax(raw_z)[j] * C[y,j]`, lambda=.25 candidate.
  Example p=(.2,.3,.5): fast expected regret 1.8, balanced .95, flagship 2.5.
  CE uses adjusted logits, cost uses raw discriminant logits; both retain gradients.
- Expert certainty multiplies the entire per-row objective by
  `max(.5, confidence)**2`; config can disable it. Only explicit
  `expert_provisional`/`expert_reviewed` sources qualify; measured confidence is
  never used. The helper returns weight 1 for nonexpert sources; the v1 training
  reader additionally rejects measured/untyped rows altogether.
- `heads-v1` candidate coefficients are tier/vision/task_type=(1,.5,.15), without
  legacy effective-number class multipliers. No-decay whitelist v1 assigns zero
  decay to 1D, bias, norm, A_log, dt_bias; other eligible parameters decay .01.
  Tied identities occur once. Scheduler counts actual sampled rows with per-epoch
  ceil tails; research tails divide by their actual accumulation size. Constant
  adapter LR versus fullft cosine semantics otherwise remain existing behavior.

`effective_config.research` records full config/hash, sampling, head weights,
capacity checks, noise and unavailable interpolation; optimizer groups have
named coverage and sampled row counts never masquerade as oversampling.

## #20: capacity candidates, not proven executable backbone support

DoRA candidate uses r16/alpha32/dropout .05. At authorized runtime, exact pinned
`config._commit_hash` is required. A missing local commit hash fails closed even
if a manifest names the revision: an external verified provenance bridge would
need review. Enumerate actual language projection names and shapes, excluding
vision/projector/embedding/lm_head. The issue's '11 classes' enumerates 12 suffixes
when k_proj is counted; v1 explicitly includes all 12 listed in the helper. Hybrid
backbones need not have every suffix, but every encountered target must be plain
`torch.nn.Linear`, have compatible matrix dimensions and support rank16. Unknown
language linear projections and custom/quantized classes fail; no permissive
'all-linear' fallback. Installed PEFT must expose and retain `use_dora`. After
injection, every target must have correctly shaped trainable A/B/magnitude state.
Configuration/version pins or toy mocks do **not** prove forward/backward, custom
kernels, checkpoint reload, merge or resource compatibility. Failures propagate;
no silent LoRA replacement. Actual support remains unverified without weights.

`capacity.mode=staged8bit` is a **separate control**: set training fullft=true,
optimizer=adamw8bit, epochs=2 in a new config. It is not DoRA and does not quantize
the backbone. Language layout must have known contiguous layers and >=4 layers.
First ceil(.15 * total optimizer updates) updates expose only the top four
language layers; then all eligible language parameters unfreeze (vision/projector,
embeddings and lm_head stay frozen). This is **not strict linear-probe LP-FT**.
All eventual eligible parameters, including initially frozen ones, enter optimizer
no-decay groups at construction. grad=None prevents early AdamW state/decay;
no optimizer reset or scheduler restart is used on unfreeze. Very short budgets
may have no full-language phase; achieved fraction is recorded, not hidden.
Missing bitsandbytes still fails loudly, never falls back to AdamW.

NEFTune attaches one removable input-embedding hook; floating [B,S,H] output gets
uniform noise in +/- alpha/sqrt(S*H), alpha=5 candidate, only in module.train().
Evaluation is exact/no-noise; unsupported embedding types/shapes and duplicate
hooks fail. Alpha=0 disables the hook. No service hook is introduced.

**Post-weight interpolation is unavailable.** The .65 WiSE-FT candidate is a
separate future experiment requiring compatible merged full parameter states,
base/revision/key/shape/dtype and merge semantics checks, new checkpoint,
independent calibration and resource validation. Averaging adapter factors,
in-run SWA, and logit mixing are not substitutes; numeric interpolation values
are refused by this config instead of offering a fake executable path.

## #5: preregistered auxiliary protocol, not a causal conclusion

`loss.auxiliary` supports `candidate` (1,.5,.15), `task-type-zero` (1,.5,0),
`current-03` (1,.5,.3), and literal `tier-only` (1,0,0). Issue #5 calls task_type=0
"tier-only"; use **task-type-zero** to isolate task_type while preserving vision.
`current-03` means current task_type coefficient only, not the archived .7 vision
coefficient or frozen full recipe. No task_type coefficient is silently active
when zero. The production policy remains tier+vision; task_type is diagnostic,
not a production selector or a reason to spend core deployment budget.

An independent-head/two-stage **protocol** (not an implemented mode):
1. Freeze a reviewed #18 label/schema version, approved #21 A/B/C protocol,
   train groups, seeds, row exposure/optimizer budget, calibration rules and
   primary tier metrics before runs. No test selection. Blinded label review
   first examines creative/implementation definition drift, class support and
   group overlap, without assuming drift explains collapse.
2. Compare matched task-type-zero, current-03, candidate, and optionally literal
   tier-only; use the same sampled streams and report exact effective budgets.
   Report tier acc/macro-F1/NLL/Brier/ECE, flagship recall/precision, vision FN,
   relative cost, creative per-class support/F1/confusion and group uncertainty.
3. The current architecture shares a causal LM's vocabulary logits; those are
   not separable independent classifier heads. A real independent-head arm needs
   a separately reviewed pooling/representation contract and explicit classifier
   architecture. Stage A trains tier+vision with task_type disabled; stage B
   freezes the selected backbone and routing path, then trains only that new
   task_type classifier with its own declared budget. Verify zero backbone/tier
   gradient and bit-identical routing predictions after stage B, plus serving/
   serialization/calibration contracts. Do not relabel staged partial-unfreeze
   as this two-stage classifier experiment. Config rejects independent-head mode.
4. Keep classifier-stage cost separate from matched routing-training budget;
   report negative/null outcomes and uncertainty. Only actual controlled runs
   and blinded diagnosis can answer benefit/harm or creative-collapse causality.

## Concurrent integration and blocked acceptance ledger

This branch is independently based on public baseline 8c04cd7. #18 concurrently
owns schema/QUESTIONS/train/serve changes; no merge or global schema mutation is
performed here. Research v1 accepts **legacy7-v1 only** and checks actual question
ordering at runtime/preflight. #18's six-type schema cannot be inferred from an
old checkpoint or silently substituted. Integration overlaps in train_eval's
reader/schema/run setup and the README pair; resolve deliberately and introduce
an explicit supported recipe schema version with shared tests. #8 provenance
translation likewise needs a reviewed interface before measured rows can enter.
No branch integration is claimed; #19 real runs depend on #18, #20 runs depend on
#19 and frozen #21 protocol (not all future OOF research).

External and still-blocked acceptance:
- #19 real smoke; authorized matched individual/combined ablations with clean
  private labels, uncertainty, NLL/ECE/promotions/cost, actual timing (no 2.3x claim).
- #20 actual pinned module/shape/PEFT/runtime/trainable-count enumeration; both
  real paths' smoke, checkpoint/reload/forward/backward verification, peak <16GB,
  two-epoch matched DoRA/control comparison, separately calibrated .65 weight
  interpolation if implemented/authorized. None was attempted here.
- #5 actual controlled runs, independent-head architecture arm, blinded creative
  diagnosis and evidence-backed benefit/null/harm conclusion. Existing #3 metrics
  are reused rather than duplicated; class support alone is not causal evidence.
- Label/data/protocol approval, independent outcome evidence and production gates
  remain external; synthetic tests and a mergeable CPU subset close none of these.

## 中文摘要与边界

这是 #19/#20/#5 的**显式 opt-in 公共 CPU 开发子集**，三项 issue 全部保持 OPEN；
未加载/下载模型、未训练/真实 smoke、未调用 GPU，也没有私有数据或性能结论。
无 `--research-config` 时旧行为不变；冻结 v3c、全局 QUESTIONS、共享截断与服务不变。
配置仅支持 legacy7-v1；#18 六类 schema 与 train_eval/README 重叠必须单独审查集成，
不能从依赖版本或旧 checkpoint 猜测兼容。新数据/协议标识和 256 行预算均为示例，非批准方案。

平方根组采样按 n_g^-0.5 有放回抽样，合成 balanced+vision=yes 仅在采样概率乘 1.5，
不再叠乘 loss。明确记录自然/目标/实际概率和先验来源；默认先验排除合成行，
可选 expected-sampler 则包括其暴露，需预注册。训练 CE 加 **+tau log pi**，
序数代价用 raw logits 概率的完整 3x3 期望相对损失，不是钱或成功率。
专家置信度是 max(.5,c)^2；measured 区间置信水平不可代用，v1 拒绝此类训练行。
候选头权重 1/.5/.15，1D/bias/norm/A_log/dt_bias 不衰减；scheduler 按实际采样更新计数。

DoRA r16/alpha32/dropout.05 只提供 fail-closed 枚举/注入检查，真实骨干支持仍未验证；
缺少固定 revision 元数据、未知模块/形状/PEFT 能力都拒绝，不能靠 pin 声称可运行。
8-bit 是独立两 epoch 对照，前 ceil(15% 更新) 仅顶层四层，再解冻符合条件的语言骨干，
不是严格 LP-FT；optimizer 从一开始包含后续解冻参数，grad=None 不提前更新/衰减。
NEFTune alpha5 只在训练 hook 加噪，eval 无噪。训练后 .65 权重插值明确 **unavailable**，
不能用 SWA/adapter 因子平均/logit mixing 冒充。

#5 的 task-type-zero 保留 vision、只把 task_type 置零；literal tier-only 连 vision 也关闭；
current-03 是 task_type=.3 而 vision 仍 .5。独立头两阶段仅有明确协议，当前共享 LM 头
不能安全伪装成独立分类器。task_type 只用于诊断，不参与生产路由。真正因果答案、
creative 塌缩归因、私有净标签匹配对照、双路径 smoke 与显存均待外部授权验收。
