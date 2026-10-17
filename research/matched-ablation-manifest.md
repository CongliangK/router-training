# Synthetic matched-ablation manifest

Refs [#44](https://github.com/CongliangK/router-training/issues/44),
[#43](https://github.com/CongliangK/router-training/issues/43),
[#42](https://github.com/CongliangK/router-training/issues/42) and
[#21](https://github.com/CongliangK/router-training/issues/21).

This is an isolated **invented-input software draft**, not a runnable configuration,
experiment executor, approved budget, scientific protocol, measured result or
qualification. All outputs remain `status=draft`, `qualified=false`,
`approval_verified=false`, `real_root_audit_verified=false`,
`training_allowed=false`, `formal_admission=NO-GO`, real execution `UNRUN`.
The validator's successful status is `synthetic-software-check-only` and retains
all false/NO-GO/UNRUN flags. No research issue is scientifically satisfied by it.

## Dependency and exact scope

This candidate is stacked on the independently reviewed #43 source commit
`e9873635928a83c2cbfe0669f807f4317d89118c`, tree
`31d6120d3c36b4c92fa59c0d767a7bfbb8a98e40`, whose main base is
`169d343d2f1fc374bc78c4262f677a6c532526c6`. #43 is separately published as
[Draft PR #45](https://github.com/CongliangK/router-training/pull/45), with the
same complete source tree at remote commit
`15a81c8065df53b0d82fafce79d1355230bf4489`. This does **not** say #43 is merged.
A later publication must explicitly map the complete reviewed tree onto that
remote parent and reverify the published final head; a changed source/interface
requires new independent review. The earlier #43 `44a77ed` draft is superseded.

The product write set is this document, `matched_ablation_manifest.py`, its new
CPU test file, and a pointer in each README. No shared validators, sampler/loss,
training/serving entry point, config, dependency, CI/security setting or archive
is modified. #43's final strict override boundary and calibration component/root
guard are inherited by calling its public API, not copied or weakened.

## APIs and strict inputs

- `build_synthetic_manifest(payloads, arms, comparison)` returns the draft.
- `validate_synthetic_manifest(payloads, arms, comparison, manifest)` rebuilds
  every arm and the complete result, then requires exact canonical equality.

No CLI, file/path discovery, writes, network, model loading, optimizer, fitting,
qualification or config-export API exists. Sampling the supplied invented fit
rows is the only new computation. Neither API accepts logits/OOF records.
Any future records must pass #43's own validator against their own arm's exact
original bytes/config/protocol/draft; a manifest never rebinds old records.

`payloads` is exactly #43's five immutable byte snapshots:
`train.jsonl`, `calibration.jsonl`, `label-schema.json`, `split-report.json`,
`roles.json`. One local byte mapping is retained before any validation and used
for all arms and sampling. #43 validates its exact narrow development-only input
profile, including strict JSON, expert provenance and ordered 3/2/6 class metadata.
Full preparation reports, test inputs, unknown extensions and path strings are
unsupported. No fields are silently projected away. Config/protocol/comparison
metadata is copied as plain, finite UTF-8 JSON; tuples, callable/custom objects,
non-string keys, NaN/infinities and invalid Unicode are rejected.

Each `arms` entry has exactly:

| Field | Contract |
| --- | --- |
| `arm_id` | Unique nonempty string |
| `config` | Complete existing `research-training-v2` config, including all five exact byte hashes |
| `protocol` | Complete explicit #43 synthetic protocol, including its own matching config hash, audit, roles, epochs and models |

A comparison needs at least two arms as a structural requirement. This is not a
scientific replicate/seed quota. Arms are never generated or repinned by the API.
Each arm is first independently rebuilt with `build_synthetic_draft`. Only after
**all** have passed are comparison metadata checked and fit rows sampled.

Complete config, protocol and draft hashes legitimately differ across arms.
The invariant projection removes **only** the three permitted factor fields.
Protocol comparison removes only `config_sha256` and `models`. Exact input bytes,
audit declarations, original role plan, nested role/fold assignments, support,
epoch lattice and every other field remain identical across arms. Reordering
nested declarations is conservatively treated as a changed proposal even if its
sets would yield the same plan. Epochs must be exactly `1..training.epochs`.

Models are explicit `(arm_id, fold, epoch, model_digest, source)` bindings in the
output, sorted by fold/epoch. The original #43 registry must cover every slot.
This narrow comparator additionally rejects a digest copied **across arms** so
an ablated arm cannot silently inherit another arm's output registry. Within an
arm, #43's allowance for repeated placeholder digests across slots is retained.
Distinct placeholder identities prove neither distinct weights nor actual runs.

## Common initialization and selection/calibration description

`comparison` has exactly `version`, `status`, `synthetic_fixture`,
`protocol_approved`, `baseline_arm_id`, `contrasts`, `initialization`,
`calibration_recipe`. Its four envelope values are respectively
`synthetic-matched-ablation-v1`, `draft`, `true`, `false`.

`initialization` has exactly:

- `version=synthetic-common-initialization-v1`
- `source=synthetic-placeholder`
- Explicit nonempty `base_model_id`, `revision`
- Lowercase SHA-256 `initialization_sha256`, `tokenizer_sha256`,
  `scorer_request_sha256`

These identifiers describe a shared starting point; #43's per-fold/epoch model
registry describes outputs instead. No checkpoint/tokenizer/scorer bytes are
opened or authenticated. There are no implicit model defaults or downloads.

`calibration_recipe` has exactly:

- `version=synthetic-selection-calibration-v1`, `status=draft`,
  `source=synthetic-placeholder`
- `A` and `C`: each exactly `selection_rule_id`, `tie_break_rule_id`, both
  explicit nonempty method identifiers
- `B`: exactly `fit_rule_id`, `threshold_rule_id`, `tie_break_rule_id` (nonempty
  identifiers), plus `decision_prior`
- `B.decision_prior`: exactly `source=unique-training-before-oversampling` and
  explicit finite nonnegative numeric `smoothing_pseudocount`
- `numeric_protocol`: all five effective original-role settings:
  `tau_grid`, `temperature_bounds`, `fit_max_iterations`, `fit_tolerance`,
  `probability_epsilon`
- `support_policy`: all five effective original-role settings:
  `min_groups_per_role` (exact A/B/C integer map), `min_rows_per_class`,
  `min_groups_per_class`, `min_fold_train_groups_per_class`, `n_folds`
- `fixed_model_required=true`, `fresh_calibration_required=true`,
  `independent_confirmation_status=UNRUN`

No omitted numeric defaults are accepted in the descriptor. Its complete numeric
and support maps must exactly equal the existing provisional defaults expanded
with the validated original `roles.json` overrides. Expansion records what the
unchanged validator used; it does not add or approve support/statistical policy.
Unknown fields at every new nesting level fail. Booleans cannot substitute for
numbers or the required flags. The original roles' overrides still pass #43's
own strict key/type/range/support checks.

The rule IDs are **unexecuted caller-authored synthetic descriptions**, not proof
that an algorithm is fully specified or scientifically suitable. This narrow
schema exposes the existing numeric protocol only; it cannot describe additional
method-specific tunable numbers. Such an extension needs a new reviewed contract,
not an ignored field or a claim that an opaque ID proves a complete protocol.
The B threshold identifier describes the declared selection procedure over the
explicit grid. No thresholds, fitted temperatures, selected epochs/configurations,
CRC risks or scientific metrics are computed. Fix the method across arms, not the
future model-specific fitted values. C remains selection, not independent final
evaluation. No default new selector or numerical scientific target is chosen.

The B decision-prior source is explicitly distinct from the **training** prior
factor `unique-natural-expert` / `expected-sampler`; the latter is never renamed
into the inference raw-logit contract. Actual inference priors are not computed
here. The common descriptor is part of the manifest's full revalidation identity;
changing it produces a new unapproved draft and rejects the old one. Per-arm
initialization or recipe overrides are unsupported and rejected as unknown keys.

## Factor matrix and declared contrasts

| `loss.auxiliary` | tier | vision | task_type |
| --- | ---: | ---: | ---: |
| `candidate` | 1 | 0.5 | 0.15 |
| `task-type-zero` | 1 | 0.5 | 0 |
| `tier-only` (literal tier-only) | 1 | 0 | 0 |
| `current-03` | 1 | 0.5 | 0.3 |

Weights come from existing `research_training.head_weights`; current-03 means
this coefficient mode, not an exact reconstruction of the entire archived recipe.
Independent-head/two-stage classifiers are not implemented by setting a zero.

The only other allowed factors are:

- `loss.confidence_weighting`: explicit boolean. On uses existing
  `max(0.5, expert_confidence)^2`; off uses 1. Expert-source/confidence/label
  validation remains mandatory when off. Measured confidence is never imported.
- `sampler.prior_source`: explicit `unique-natural-expert` or
  `expected-sampler`. Per-fit tier/vision priors and provenance are recomputed
  with the existing sampler. Changing this factor alone changes the chosen prior
  distribution where the pool warrants it, never the RNG stream/probabilities.

One `contrasts` entry is required for every nonbaseline arm, with exactly
`arm_id`, `changed_factors` (distinct supported path names) and `kind`.
`changed_factors` must equal the actual nonempty differences against the named
baseline; `kind=single-factor` for one change, `kind=combined` for more. A combined
change cannot be labeled a single-factor experiment. This is only a structural
comparison, not evidence of effects or causal identification.

All other config values are frozen: data/protocol versions and schema, training
seed/epochs/accumulation/LR/beta/rank/optimizer/flags, label smoothing, capacity and
noise, sampler algorithm/row budget/multiplier/smoothing, taus, cost matrix/lambda
and head/loss versions. An undeclared difference is rejected even if sampled
counts happen to match. All configs still pass the unchanged research validator. The local comparator also
rejects boolean cost-matrix entries rather than relying on Python's `True == 1`
and `False == 0` numeric equality; it does not change the shared validator.

## Planned synthetic budgets, not measured resources

For each arm/fold, filter the original **train byte row order followed by
calibration byte row order** to that nested fold's exact `fit` IDs. The fit role
can contain declared development rows from either source. Never sample A/B/C,
the outer validation fold, or the entire outer complement. Join back to those
retained validated row bytes, not fabricated prompt-less rows. Roots are never
inferred from names. Original A/B/C and nested fit/A/B/C are separate contracts.

Call existing `sample_epochs` with that fit pool, the explicit training seed and
complete epoch count. There is no seed search or fold offset. The original sampler
still uses its documented component weighting, with replacement. Its algorithm
is not reimplemented as an audited-root sampler. Report:

- Fit row/root counts, original ordered-row fingerprint, fit-order and natural /
  target probability digests, original sampler row-order hash
- Per-fit actual calculated training priors and provenance (not learned metrics)
- For every epoch: exact stream digest, achieved per-ID count digest,
  unique sampled-row count, planned row exposure, optimizer updates and final
  update size; IDs themselves and prompt/label rows are not emitted
- Per-fold and per-arm planned totals, separately from confidence-weighted and
  per-head weighted exposure

Full ordered per-epoch ID lists and probability arrays are compared **in memory**
across arms, not merely frequency summaries. Only their digests appear in output.
Equal frequencies with different update order fail. The current Python
implementation/version and `stdlib-random.Random.choices` identity are bound in
the output; seeds alone do not prove cross-version RNG equality.

For each epoch with stream length `n` and accumulation `a`:

- Planned updates: `(n + a - 1) // a`
- Final update rows: `(n - 1) % a + 1` when nonempty
- Fold total: **sum of per-epoch ceilings**, never ceiling of the pooled rows

For example, an invented fixture's three epochs of 19 rows at accumulation 8
produce 9 planned updates per fold, not `ceil(57/8)=8`. These numbers are test
fixtures, not a proposed real budget. All folds are totaled explicitly.
Confidence exposure is the sum of existing confidence weights over the same
stream; each head's exposure multiplies that sum by its coefficient. These
expected differences are not changes to sampling or update budgets and are not
counts of effective independent data or proof of loss/gradient magnitude.

Capacity config and rank are reported separately. This shared-LM coefficient
change does not create/remove a separate classifier or prove parameter savings.
For each arm, `measurement.status=UNRUN`; actual trainable parameters, optimizer
updates, training tokens/FLOPs, wall time, GPU peak memory and monetary cost are
all `null` (including both trainable and total parameter counts). No numeric runtime, speedup, savings, quality or money limit is inferred.

## Revalidation, limits and tests

Any changed byte, row order, config, root audit, original/nested roles, model,
epoch, common descriptor, runtime, budget or output flag must match a complete
rebuild from current explicit inputs. Self-rehashing an edited envelope cannot
approve or launder it. Old records remain bound to their original #43 draft.
The result is rejected by both original research-config and roles-config loaders.
It emits no full arm config, rows/labels, command or runnable training artifact.

A declaration and matching hash cannot prove that a dishonest caller supplied
invented data, authentic roots, real model weights or genuine human approval.
Those are explicit trust boundaries. This API is not an access-control system
preventing callers from using their already-held configs elsewhere. No real,
private, held-out or model assets are authorized by this software.

Run the focused tests with the existing environment (no installation):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_matched_ablation_manifest.py -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Coverage includes every factor, exact contrasts, fit-only stream/order/probability
and per-epoch tail accounting, all immutable bindings, caller swaps, model/epoch
registry failures, full numeric expansion, #43's repaired root/component and
unknown-override guards, provenance with confidence off, unknown keys/types,
self-rehash/approval/resource claims and file/network/model/training/fitting traps.
Tests are invented CPU evidence only. Exact-head source review, independent actual
verification and hosted checks are separate gates; skipped tests are not passes.

## Acceptance and remaining gates

| Source | Bounded software contribution | Unfinished external gate |
| --- | --- | --- |
| #44 | Explicit task-type-zero vs literal tier-only; confidence/prior choices; fixed factors, model identities and synthetic planned budgets; non-runnable draft | Independent final source review, actual verifier, exact published head checks and dependency-aware PR process |
| #43 | Every arm rebuilds its original five bytes/config/protocol; strict root/role/model/epoch and override guards preserved | Authentic independent roots and actual human protocol review remain absent |
| #19 / #5 | Individual/combined factor declarations and same ordered fit exposure | Real matched smoke/training, creative diagnosis, independent/two-stage paths, causal/quality/cost/NLL/ECE evidence remain UNRUN |
| #21 / #2 | Dev-no-test, source/fold/fit/A/B/C lineage and epoch/model bindings preserved | Approved roles/support/statistical/sampling/budget protocol, real independent support, actual OOF, fixed-model fresh calibration, confirmation, soup and deployment parity remain blocked/UNRUN |
| #42 | Identity/budget and admission draft with false qualification/NO-GO | The complete AC01–AC21 contract below remains in force |

All #42 inherited obligations are preserved without lowering their scope:

- AC01: only old calendar dates retired; stage work and unfinished goals remain.
- AC02: original targets remain unverified: Phase 0 tier accuracy ≥.78,
  macro-F1 ≥.82, flagship recall/precision ≥.90/≥.62, ECE ≤.10, no training;
  Phase 1 ≥.83/≥.85/≥.93/≥.68 and creative recall recovery; Phase 2 accuracy
  ≥.86, flagship miss ≤.05, vision FN ≤.04 and decidable quality gate.
- AC03: seven historical runs, diagnostics and prior read-only findings remain
  historical hypotheses/evidence, not new empirical or causal results.
- AC04: #1/#22 real aligned raw logits, extractor/prior/scorer/label/cost evidence,
  frozen development selection and separately authorized one-time descriptive
  test; logit ensemble and actual weight interpolation remain separate UNRUN work.
- AC05: approved role/support/statistical/sampling/budget protocols, authentic
  independent roots, new runs/OOF, fresh fixed-model calibration/confirmation and
  actual soup remain blocked/UNRUN. Historical 564 rows/30 groups are not automatic
  independent evidence. Keep #21 protocol → #20 execution → #21 real OOF ordering.
- AC06: six actual uncertain adjudications remain outstanding; pending/excluded
  or invented examples do not fulfill any real review quota.
- AC07: 16 preregistered creative blinded reviews, frozen old labels, visual/action
  migration, real provenance/root/prior/support/leakage and checkpoint/serve parity
  remain outstanding. No independent classifier/two-stage implementation is claimed.
- AC08: real #19 matched training/causal tests for oversampling, logit adjustment,
  ordinal cost, confidence and auxiliary losses remain UNRUN.
- AC09: #20 real DoRA r16 and staged 8-bit model paths, pinned dependencies/revision,
  modules/shapes/parameters, forward/backward/train/smoke/save/reload/parity and
  measured memory/speed remain UNRUN; no model/GPU work is performed here.
- AC10: #25's 35 real targeted samples (20 fast plus 15 cross-root vision=yes),
  local collection/adjudication/provenance/support/allocation remain unmet and
  PAUSED. Synthetic examples do not count and collection is not resumed.
- AC11: #26/#8 45-anchor real execution, bias/correlation/sampling correction,
  budgets, independent confirmation/trial integrity, actual success/savings,
  cost/quality/shadow/qualification gates remain UNRUN. No task-level measured
  labels are inferred from root means.
- AC12–AC16: closed CPU #3/#4/#6/#23/#24 retain their real frozen-v1 metrics and
  denominators/cost, private deletion-sensitivity/skew, policy reasons/threshold/
  cost, candidate vision/effort/private replay, and new labels/splits/root isolation/
  35% cap/infeasibility/comparability acceptance. None is fulfilled here.
- AC17: freeze exact final base/head/tree; fresh independent source/red-team review,
  then separate actual applicable CPU verifier; both required original hosted CPU
  event checks; issue-linked draft PR, authorized serial exact-head merge and
  merged-tree equality/post-main verification. Any source/interface/conflict or
  integration change invalidates transfer of an old PASS. A green test is not
  scientific or merge approval. Stacked #43 dependency remains explicit.
- AC18: atomic research issues and their original acceptance stay separate. Use
  `Refs` in research PRs, without automatic issue-closing directives. #27's prior
  replacement is not research completion and is not changed here.
- AC19: PR #38 remains PAUSED/NOT READY; no collection/preparation/model download/
  training/smoke/GPU/backbone work resumes. Admission stays NO-GO/insufficient.
- AC20: reviewed public source, invented fixtures and fingerprints only; no real
  prompt/label/prediction/mapping/original-byte sidecars, private paths/logs/history
  or model assets. Missing inputs and unavailable checks remain explicit; no
  installation, service, permission or security changes.
- AC21: historical failures, repair evidence and review versions remain retained.
  Author, independent reviewer and verifier executions are distinct runs, not
  additive unique test counts or substitutes for private-asset acceptance.

A (software draft), B (real execution, blocked/UNRUN) and C (scientific approval,
unapproved) must always be reported separately. No successful draft, hash, test,
source review or PR changes B/C or authorizes paused work.
