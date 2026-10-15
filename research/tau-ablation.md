# Tau ablation — protocol and unavailable-results template (#1)

**Status: CPU tooling only; no real evaluation, no causal finding, no issue
closure.** All implementation tests use invented logits. Private raw scores,
labels, weights and frozen-test replay have not been read or run here. An
aggregate archived report cannot reconstruct per-row logits.

## Reproducible implementation

- [Tau CLI](../scripts/tau_ablation.py),
  [shared raw-logit contract](../raw_logit_contract.py), and
  [full protocol/inputs/exit semantics](../docs/OFFLINE-CALIBRATION.md).
- Fixed checkpoint and fixed predeclared T; scan **tau = 0, 0.5, 1, 1.5, 2** with
  `(z_raw - tau*log(unique_train_prior))/T`. Positive rare-class boosts from
  subtraction are intentional; do not accidentally reverse the archived sign.
- Explicit input bindings for checkpoint, scoring/extraction, dataset, train
  prior, versioned relative-cost table and full selection/comparison protocol.
- Development selection is minimum group-equal NLL under predeclared flagship
  recall/FPR constraints and class/group support; tie => smaller tau. Gate
  failure => no selection, never an automatic tau=0/relaxed-recall fallback.
- Quality/cost table is descriptive development evidence. Spend is the mean of
  **predicted-tier** relative costs, not an error penalty or actual billing.
- One smaller tau is registered before the scan as the comparison against tau=2.
  Paired whole-group bootstrap diagnostics report NLL, error and cost deltas,
  seed, support, draw budget and warnings. No post-sweep significance claim.

## Before any separately authorized private replay

- [ ] Approve scope/environment; keep private raw inputs outside public checkouts.
- [ ] Verify checkpoint-byte identity and raw score extraction parity. Confirm
      upstream model/epoch selection did not use the evaluation pool's groups.
- [ ] Lock source/extraction/dataset digests and class order; reject calibrated
      probabilities, already-adjusted scores, and ID/group/label mismatches.
- [ ] Lock unique pre-oversampling training prior, any smoothing convention,
      fixed T, relative cost table version/units and protocol hashes.
- [ ] Lock support thresholds, uncertainty minimum, group definition, one smaller
      comparison tau, primary NLL diagnostic, bootstrap seed/budget and tie rules.
- [ ] Run development/calibration only; inspect confusion and class support, not
      macro-F1 alone. Record unsupported/null denominators and any infeasibility.
- [ ] Lock final development candidate and a separate one-time descriptive
      frozen-test protocol before test review. Do not use test to choose tau.
- [ ] Release only reviewed coarse aggregates, no prompts, labels, prediction
      rows, IDs, group names, private paths or raw logs.

## Tau × quality × simulated-cost table (real results UNAVAILABLE)

| tau | confusion / class F1 | accuracy / macro-F1 | NLL / Brier / ECE | relative cost | class/group support | feasible dev point |
|---:|---|---|---|---|---|---|
| 0.0 | unavailable | unavailable | unavailable | unavailable | unavailable | not assessed |
| 0.5 | unavailable | unavailable | unavailable | unavailable | unavailable | not assessed |
| 1.0 | unavailable | unavailable | unavailable | unavailable | unavailable | not assessed |
| 1.5 | unavailable | unavailable | unavailable | unavailable | unavailable | not assessed |
| 2.0 | unavailable | unavailable | unavailable | unavailable | unavailable | not assessed |

| Locked comparison / evidence | Current state |
|---|---|
| checkpoint / input / extraction bindings | unavailable |
| train prior / fixed T / cost version | not approved for real replay |
| registered smaller reference / protocol hash | not approved for real replay |
| paired group NLL/error/cost intervals | unavailable |
| supported groups and per-class groups | unavailable |
| selected development tau | unavailable; no test tuning |
| is tau=2 materially/significantly worse? | **uncertain; no real paired evidence or formal significance guarantee** |
| frozen-v1 descriptive confirmation | not executed; historically viewed, not pristine holdout |

Do not fill cells from synthetic fixture outputs or infer them from historical
aggregate reports. Even available group-bootstrap intervals remain approximate,
conditional on fixed predictions and the asserted independent group structure;
small support, boundary scores and adaptive development selection limit them.
A positive NLL delta is not a claim of better task success, lower billing, or
uniformly better class behavior. An unavailable/unsupported comparison remains
uncertain rather than a fabricated zero-width interval or a guaranteed win.

## Causal boundary and #22 relationship

Inference-side tau replay **cannot remove training oversampling**. The suspected
double compensation mechanism still requires #19's matched training-side
ablation, controlled for seed, data, loss and model selection. No causal training
conclusion is available here.

[#22 nested LOGO CLI](../scripts/wise_ft_calibrate.py) is separate tier **logit
ensemble** tooling: every outer group's labels are excluded for alpha/T/dual
threshold selection, and inner validation groups fit neither T nor thresholds.
It emits no full-pool fitted claim, deployment configuration, or checkpoint.
It is not WiSE-FT weight interpolation; optional weight soup is skipped. Real
private logit replay and the issue's research metric targets remain unverified.

中文结论：当前仅交付可复现 CPU 工具与协议，未读取私有 logits、未执行真实回放，
不填造数值、不关闭 #1/#22。tau=2 是否显著劣于较小值目前**不确定**；支持不足必须
保留不确定。相对成本为显式版本化模拟单位，不是账单。训练过采样的因果关系仍须
#19 匹配训练消融；固定权重改推理配置不能声称已移除过采样。冻结测试不用于调参，
历史反复查看的 frozen-v1 也不能因新预注册恢复独立 holdout 身份。
