# Offline tier calibration protocols — issues #1 / #22

**Tooling, not real evaluation or issue closure.** No private inputs, weights,
models, training (including smoke), GPU, network, or frozen-test replay were used
for this implementation. Only invented fixtures are tested. These tools do not
modify training, A/B/C calibration, serving, policy, preparation or frozen runs.

- [#22](https://github.com/CongliangK/router-training/issues/22):
  [wise_ft_calibrate.py](../scripts/wise_ft_calibrate.py) performs a **logit
  ensemble** `(1-alpha)*z_base + alpha*z_ft`, not
  [WiSE-FT](https://arxiv.org/abs/2109.01903) **weight interpolation**. They are
  generally different for nonlinear networks. Weight soup/checkpoint production
  is deliberately skipped; this tool produces no serving calibration file.
- [#1](https://github.com/CongliangK/router-training/issues/1):
  [tau_ablation.py](../scripts/tau_ablation.py) isolates inference-side
  `z_adjusted = z_raw - tau*log(train_prior)`, then divides by a **fixed,
  predeclared** temperature. No T is fitted during this tau-only comparison.
  Fixed trained weights cannot undo oversampling or identify its causal effect;
  that requires matched training ablations under #19, outside this work.
- Both use [raw_logit_contract.py](../raw_logit_contract.py) and
  [offline_calibration.py](../offline_calibration.py), standard-library only.
  No implicit repository data discovery. **Tier only:** no vision/task-type
  calibration, capability-aware policy costs, or task-success claims.

## 1. Shared aligned raw-logit boundary (v1)

Each source is one JSON object, not legacy prediction JSONL. An authorized local
extractor must construct this boundary from the raw pre-adjustment scores; the
public tools intentionally do not open historical prediction rows or weights.
**Never copy private raw logits into a public checkout.** Public fixtures must be
invented, not scrambled/rescaled real rows. `private-local` is reserved for a
separately authorized private environment, not permission to import data here.

All keys shown below are mandatory; unknown fields at every level are rejected.
The one-row fragment illustrates the schema, **not adequate calibration support**:

```json
{
  "schema_version": "router-raw-tier-logits/v1",
  "data_kind": "synthetic",
  "split": "calibration",
  "head": "tier",
  "class_order": ["fast", "balanced", "flagship"],
  "logit_stage": "raw-pre-temperature-pre-prior",
  "dataset_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "scoring_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "model_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "rows": [
    {"id": "invented-0", "split_group": "invented-group-0", "label": "fast", "logits": [3.0, -1.0, -2.0]}
  ]
}
```

Contract requirements:

- `data_kind`: `synthetic` or `private-local`; `split`: **only** `development` or
  `calibration`. Train/test inputs are rejected; there is no test-selection mode.
- Fixed tier order is parsed from the existing QUESTIONS source. Shared metric
  helpers import only schema constants; backend/model loading stays lazy, and
  neither CLI imports torch, transformers or peft. Exactly three finite numeric scores, no booleans, with
  absolute value <= 1e6 (explicit numerical guard). At most 2,000 rows for this
  bounded CPU research prototype. No probability-vector auto-conversion.
- Unique, nonempty UTF-8 string IDs and nonempty whole-source `split_group`
  identities (escaped lone surrogates are rejected with sanitized errors);
  exactly one legal tier label per row. Groups must reflect upstream correlated
  sources, not invented row IDs. The software cannot establish independence.
- `dataset_sha256`: canonical ID/group/target snapshot binding from extraction;
  `scoring_sha256`: shared fixed-option ordering, tokenizer revision, request
  truncation/context, and score-extraction procedure binding; `model_sha256`:
  actual checkpoint-byte digest supplied by extraction. The example hashes above
  are fake. **No weights are read or hashed by these tools.**
- Ensemble sources must match every field except `model_sha256` and logit values.
  ID sets, groups, labels, dataset/scoring bindings and split must agree exactly.
  Alignment is by ID, never row order. Duplicate IDs/JSON keys, extra/missing rows,
  schema changes, NaN/Infinity, and metadata mismatches fail closed.
- Provenance/hashes/logit stage are **assertions by the extractor**, not an
  authenticated attestation. Arbitrary finite scores cannot reveal whether an
  upstream producer already tempered them, saw test labels, or trained on a
  held-out group. Audit upstream model/data isolation separately. These guards
  do not make in-sample model logits into held-out predictions.

Only exact requested inputs are opened. Error messages omit supplied values,
paths and row identities. Reports contain aggregates and bindings, not prompts,
raw logits, per-row labels or group IDs. Private-local output is **not automatically
publication approved**: small support cells and hashes still need privacy review.

## 2. Nested LOGO alpha / T / continuous thresholds

Every protocol field is explicit; there are no silently enabled defaults.
The following is a **synthetic example, not an approved production protocol**:

```json
{
  "schema_version": "router-nested-logo/v1",
  "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
  "temperature_bounds": [0.25, 8.0],
  "fit_iterations": 100,
  "fit_tolerance": 0.000001,
  "min_fit_groups": 2,
  "min_class_rows": 2,
  "min_class_groups": 2,
  "min_predicted_groups": 2,
  "min_flagship_recall": 0.75,
  "max_flagship_fpr": 0.10,
  "min_fast_precision": 0.75
}
```

All group/class/predicted-group minima must be >=2; a smaller number is not a
supported fallback. With these example gates at least four groups are necessary,
but four (or eight, or 95 rows) is **not sufficient**: every outer and inner fitting
union must retain the registered per-class rows and distinct groups. No row-level
split, A/B/C weakening, class dropping, or borrowed holdout labels is allowed.
Numeric gates are feasibility checks, not error-rate or recall guarantees.

For **each outer held-out group**:

1. Remove that entire group before alpha, temperature or thresholds are selected.
   Preflight all outer-training and inner-training unions; inadequate support
   makes the whole report insufficient before fitting anything.
2. For each alpha, run genuine **inner LOGO over outer-training groups**. Fit T
   by bounded group-equal mean NLL on inner-training logits using the existing
   deterministic temperature helper. Fit both thresholds on those same
   inner-training probabilities. No inner-validation labels enter either fit.
3. Threshold rule: flagship if `p_flag >= theta_flag`; else fast if
   `p_fast >= theta_fast`; else balanced. Flagship takes precedence on overlaps
   and exact ties. Search empirical continuous breakpoints, not a coarse grid:
   all distinct flagship probabilities plus 0 and `nextafter(1,+inf)`, and all
   distinct non-flagship fast-probability partitions. Select the lowest
   **group-equal decision error**, subject to group-equal flagship recall/FPR,
   fast precision, and distinct predicted fast/flagship group minima. Ties use
   smaller flagship then fast threshold. Empty fast/flagship decisions fail
   support, rather than creating undefined precision or a trivial all-balanced
   fallback. Exactness is over finite empirical partitions, not population risk.
4. Score each inner-validation group with that fitted T/threshold pair. An alpha
   is eligible only if all inner fits succeeded and the complete inner OOF pool
   meets the registered constraints/support. Choose alpha by group-equal inner
   OOF error, then group-equal NLL, then smaller alpha.
5. Refit T and thresholds for that chosen alpha on **outer training only**.
   Apply the locked parameters once to the outer held-out group. Report fitting
   parameters, but **never describe fitted training metrics as generalization**.

No feasible candidate or threshold pair in **any** outer fold => insufficient,
no pooled partial-OOF score, no fallback alpha/T, no deployment configuration.
A complete report pools exactly one outer prediction per row. It estimates the
nested selection **procedure**; fold-specific parameters are not one locked
candidate. No full-pool refit or in-pool validation score is emitted. T fitting
has a finite bounded numerical budget, not a claimed global-optimization proof.
Threshold-selected-class probability defines decision-confidence ECE (it need
not be the argmax probability). Class-null and fixed-3-class macro-F1 conventions,
NLL floor 1e-12, Brier sum and disjoint five-bin ECE match metric definition v2;
additional group-equal statistics are named separately. The adapter directly
reuses shared `per_class`/`calibration_bins` helpers; CPU tests also check it
against the complete existing `evaluation_metrics.metrics` implementation,
including null class ratios, zero-probability NLL, ECE boundaries and non-argmax
threshold decisions. Its ordinal error-cost helper is intentionally **not**
reused as simulated route spend: correct routes still cost money/relative units.

## 3. Tau-only grid, costs and group uncertainty

A separate **bound train-prior** input is mandatory. It must use unique training
rows **before oversampling**, never dev/test-estimated frequencies. Smoothing,
if used, is part of that externally locked prior/extraction protocol. Exact schema:

```json
{
  "schema_version": "router-tier-prior/v1",
  "source": "unique-training-before-oversampling",
  "model_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "class_order": ["fast", "balanced", "flagship"],
  "probabilities": [0.1, 0.8, 0.1]
}
```

The model binding must match; positive finite probabilities sum to one. The toy
prior above is invented, not reconstructed from an archived report.

The relative cost table is explicit, versioned and mandatory. Example only:

```json
{
  "schema_version": "router-relative-cost/v1",
  "version": "invented-relative-v1",
  "units": "relative-simulation-not-billing",
  "relative_cost_by_predicted_tier": {"fast": 1.0, "balanced": 3.0, "flagship": 9.0}
}
```

Version is a 1–64-character symbolic identifier (letters/digits/underscore/dot/
hyphen). All costs must be positive finite nonboolean numbers. This is simulated
**spend per predicted tier**, including correct routes, not ordinal error loss,
actual billing, policy-adjusted routes, or measured task success. No costs are
inferred from the capability/identity route table. A changed table changes the
binding and requires a new declared analysis.

Cost means count only actually predicted routes. Integer costs are exact; parsed
floating costs mean their exact binary values (not ideal decimal literals).
Route counts and row/group weights are rational until each final mean is rounded
once to binary64, nearest with ties to even. Group-equal cost is the mean of exact
within-group means, not an average of already rounded group scores. This avoids
both intermediate sum overflow and loss of positive small/subnormal spend from
scaling by a huge unused route. No epsilon, zero substitution, cost-table range
restriction or clipping is used; nonfinite/nonrepresentable arithmetic fails
closed. Bootstrap samples use this same calculation and distinct cluster-draw
identities; paired deltas subtract the reported binary64 means as before (a zero
difference does not mean zero spend). Other metrics and selection rules are
unchanged.

Example explicit tau protocol (again provisional):

```json
{
  "schema_version": "router-tau-ablation/v1",
  "tau_grid": [0.0, 0.5, 1.0, 1.5, 2.0],
  "fixed_temperature": 1.0,
  "comparison_tau": 0.0,
  "min_class_rows": 2,
  "min_class_groups": 2,
  "min_uncertainty_groups": 8,
  "bootstrap_samples": 1000,
  "bootstrap_seed": 0,
  "min_flagship_recall": 0.75,
  "max_flagship_fpr": 0.10
}
```

- Every tau reports confusion, class precision/recall/F1/support, fixed-class
  macro-F1, accuracy, NLL, Brier, ECE, and mean relative cost, with group-equal
  NLL/error/recall/FPR/cost alongside row-weighted metrics. Undefined class ratios
  are null. These are **descriptive development** scores, not held-out evaluation.
- Development selection: minimum group-equal NLL among supported candidates
  satisfying the predeclared flagship recall/FPR; ties choose smaller tau.
  No feasible candidate => selected tau null / insufficient. No relaxed gate.
- Register **one** smaller `comparison_tau` before reading results. For tau=2
  minus that reference, paired whole-group 95% percentile bootstrap diagnostics
  cover group-equal NLL, error and relative cost. Whole clusters retain all rows;
  repeated draws count with multiplicity. The same group draws are paired across
  both tau values; seeds/budgets are explicit. This does not quantify uncertainty
  of a sweep-selected winner or refit the selection procedure in each draw.
- Group count below the registered uncertainty minimum or inadequate per-class
  support => null intervals and uncertain real-data conclusion. Synthetic input
  **always** yields `synthetic_only_no_model_conclusion`. A positive NLL interval
  can at most yield exploratory evidence against tau=2 vs the registered
  reference; **no formal significance, simultaneous coverage, independence,
  population, causal or production guarantee** is claimed. Few groups, shared
  model selection and boundary scores can understate uncertainty. Accuracy,
  cost and class behavior may disagree with NLL; inspect the full table.

## 4. Running safely / 安全运行

Use Python >=3.11; no extra packages required by these tools. No installation,
pin changes, model loading or training is needed. Use synthetic inputs only in
public development; put outputs in a new external research/evidence location.

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/wise_ft_calibrate.py --base synthetic-base.json --finetuned synthetic-ft.json \
  --protocol nested-protocol.json --output new-nested-report.json
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/tau_ablation.py --logits synthetic-ft.json --prior synthetic-prior.json \
  --costs relative-costs.json --protocol tau-protocol.json --output new-tau-report.json
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

Exit 0 = complete report/candidate; 2 = insufficient support or infeasible gates;
1 = invalid input/IO. Output is exclusive-create, never overwrite. A 0 exit on
synthetic data certifies neither model quality nor the real-data issue goals.

中文摘要：公开目录只允许纯合成 fixture，不读取或复制私有逐行 logits。两个命令
共用严格的原始 tier-logit 契约，拒绝 test/train、错序/错组/错标签、非有限值及未知字段。
外折完整留组；内折也真正按组隔离，alpha/T/双阈值都不能看当前外折标签，内折验证组
不参与 T/阈值拟合。支持不足/无解即不可行，不回退、不改门槛、不把全池拟合当泛化。
此处为 **logit 集成，不是 WiSE-FT 权重插值**，不产出 checkpoint 或服务配置。
Tau 固定预注册 T，只改推理侧调整；显式训练 prior 与版本化相对档位成本，成本不是账单。
按组配对 bootstrap 仅为探索性诊断，支持不足结论不确定；合成值不能证明真实改善。
[#1 协议/不可用结果模板](../research/tau-ablation.md) 保持真实回放和 #19 因果训练未完成。

## 5. External acceptance remains open

[#1 protocol/unavailable table](../research/tau-ablation.md) deliberately has no
invented results. #22's real aligned private logits and independent authorized
replay are unavailable here. Freeze checkpoint identities, input/extraction
bindings, protocol/cost/prior and final dev candidate **before** any private
frozen-test descriptive review. Test review belongs to a separate authorized
protocol; these CLIs do not accept test inputs. Historical frozen-v1 was viewed
repeatedly: a new preregistration cannot restore pristine holdout status or
certify independent generalization. Research targets (accuracy/F1/recall/
precision/ECE) remain unverified, not guarantees or acceptance already met.
