# CPU protocol prototypes — Refs #21, Refs #26

**Status: independently reviewable development prototypes, not production
qualifications. Both issues remain open.** No training (including smoke), model
loading, weight averaging, real outcome execution or test-data access is part of
this implementation. Existing training, serving, policy and outcomes gates are
unchanged. All results carry `qualified: false`. Approval strings/digests are
metadata declarations, never proof that an audit or approval happened. The
`preregister` API serializes a **draft** plan and detects accidental mutation;
its hash is not a trusted timestamp, signature, approval, or execution record.
`approval_verified` and `assumptions_verified` remain false wherever reported.

## #21: explicit development/no-test and lineage contract

`oof_protocol.py` accepts in-memory metadata, never file paths or a data directory.
Every row must explicitly say `source_split: train` or `calibration`; missing,
implicit development, and test sources are rejected. It requires unique row IDs,
`split_group`, audited `root_id`, tier and vision labels. A split group cannot span
roots. The caller must first close root/near-duplicate relationships and supply an
audit reference; distinct root names are not evidence of independence. Dishonestly
relabeled test metadata cannot be detected by this metadata-only prototype.

Historical 469+95=564 rows/30 groups describe the issue's proposed development
pool, not assets read or independently verified here. Neither historic row counts
nor split-group counts establish independent root counts. The frozen 104 test
rows are outside development and must remain sealed until an externally approved
locked final evaluation; this code cannot open them.

1. `development_folds` counts class-supporting **roots**, including both vision
   classes and all three tier classes, before constructing five whole-root folds.
   `min_class_roots` is an explicit provisional feasibility parameter, not a
   mathematical qualification threshold. At least five independent positive roots
   are necessary for five positive holdouts at minimum support one.
2. Assignment calls the existing `calibration_groups.group_folds` on root row
   counts. It is deterministic greedy row balancing, NOT stratification and does
   not promise six groups per fold. Each actual validation/training complement
   is checked again. Missing class support fails closed: no split/duplicate roots,
   seed search or relaxed gate. A rejected greedy allocation does not prove no
   possible allocation exists. Any stratified alternative/reduced fold count/new
   collection requires separately preregistered approval; it is not a fallback.
3. `nested_plan` requires a complete, disjoint, nonempty `fit/A/B/C` root
   partition inside EACH outer complement, with class support per role. Only
   `fit` trains the fold model; A selects its epoch, B fits temperature/threshold
   on that selected fixed model, C selects the final configuration. All roles
   are excluded from the outer validation fold. Actual fit-root counts and all
   per-role row IDs/sources are retained. This is a deliberately conservative
   independent-role design, NOT training on all outside rows while pretending
   their A/B/C evidence is fresh. C is final **selection**, not final evaluation.
4. `validate_oof_records` requires every development row once at every declared
   epoch, matching plan/fold/root/original split, exact fit-root training lineage,
   one frozen model digest per fold/epoch and finite tier/vision logits. The plan
   digest binds labels and source metadata as well as allocation. No model byte
   verification/training occurs. Such private per-row provenance must not be
   published; public evidence contains synthetic fixtures/aggregate counts only.

Different outer fits share training roots. OOF logits are development evidence,
NOT 564 fresh IID observations and NOT interchangeable outputs from a fixed
model. No pooled OOF risk guarantee, nested fitting implementation, continuous
optimizer, or OOF execution has been delivered. External execution must record
its A-selected checkpoint, B fitting procedure, C decision, priors computed only
from fit rows, all recipe/code identities and effective training roots. Any use
of C or outer results to revise the recipe requires a new independent final
confirmation, not repeated claims of untouched validation.

### Restricted conformal risk control (CRC) solver

The only implemented family is a predeclared step loss, NOT arbitrary callables:

- one task preselected per audited root, fixed model score `s` in [0,1];
- safety parameter lambda in [0,1], predict flagship when `lambda >= 1-s`;
- `L(lambda) = 1{true flagship AND lambda < 1-s}` is bounded [0,1], monotone
  nonincreasing and right-continuous; equality promotes, lambda=1 is zero loss;
- choose the smallest breakpoint with `(sum(L)+1)/(n+1) <= alpha`.

The exact finite-step solver evaluates 0, all loss jumps, and 1; rational comparison
handles the n=19, alpha=.05 boundary without floating-point drift. If the corrected
criterion cannot be met (including empty or too-small samples), output is
`infeasible`, threshold null, with safe extreme identified separately. Even a
known zero-loss extreme is not misrepresented as adequate empirical evidence.

[CRC Theorem 1](https://arxiv.org/html/2208.02814v4) concerns **expected risk** under
exchangeability of these calibration units and the new unit, fixed model/loss
before calibration, bounded monotone right-continuous losses and an appropriate
safe extreme. It is NOT a high-probability guarantee, a guarantee conditional on
each group, or a statement about correlated/adaptively reused OOF predictions.
All listed declarations, fixed model identity, correct population and external
approval reference are needed even to show an `assumption_conditional` statement;
otherwise arithmetic is exploratory and the risk statement is null. The formal
statement is `E[L_new(lambda_hat)] <= alpha`, averaging over calibration and new
unit randomness, NOT a bound on the realized conditional risk of this particular
calibration sample. `coverage_claim: false` records that no actual experiment's
coverage is certified. No result qualifies a model; this code cannot establish
those assumptions empirically.

`joint_flagship_miss` estimates P(true flagship AND miss) over all sampled units.
`conditional_flagship_miss` requires sampling only true-flagship units from that
population, estimating P(miss | true flagship). A low joint risk does not establish
conditional risk. Both use equal-root, one-preselected-task loss. They are NOT
arbitrary row-weighted losses; a group-average target, size-weighted task target,
or repeated tasks per root requires a newly justified loss/sampling contract.
5% remains a proposal requiring approval, not a default production target.

### Deployed-model bridge and soup refusal boundary

[Model Soup](https://arxiv.org/abs/2203.05482) is a separate experiment, not an OOF
calibration trick. `soup_compatibility` compares pinned base/revision,
architecture, tokenizer and scoring identities, exact parameter keys, shapes and
dtypes using descriptors only. Raw LoRA/DoRA factors and quantized states are
refused. An adapter descriptor must instead attest independently reviewed
materialization of its delta into the identical dense base, including DoRA's
nonlinear/norm semantics. A digest/flag alone cannot verify this materialization.
Pinned base revisions accept 40- or 64-hex immutable commit IDs (not branch
names); model/tokenizer/scorer evidence uses SHA-256 declarations. No weights are
opened, loaded, merged or averaged by this prototype.

A fold model, full-development refit, or materialized soup is a DIFFERENT fixed
model. `deployment_bridge` rejects pooled OOF thresholds and different-model
calibration outright. Freeze the final model/recipe/scorer first, then acquire
fresh root-disjoint B calibration not used in training or model selection, and
independent confirmation roots. Train, selection, calibration and confirmation
regions cannot overlap. Refitting after calibration invalidates the binding.
External byte-level model identity, independent calibration evidence, approved
loss/target, and verified train/serve parity are still required. Even a complete
metadata bridge remains shadow and never inherits OOF guarantees. If all available
roots were consumed by full-pool refitting, acquire new independent roots or use a
separately approved alternative; do not recycle OOF thresholds to certify soup.

## #26: preregisterable anchor/cascade/confirmation contract

`cascade_protocol.py` samples audited roots uniformly without replacement, then
one or two tasks uniformly within each selected root, using a recorded seed/frame
digest. Sampling metadata includes root, split group and near-duplicate cluster;
clusters cannot cross root IDs. Screening and reserved confirmation roots are
disjoint. IDs alone cannot validate correlation; between-root audit remains
external. The frame is development/synthetic only in this prototype, never test.

The proposed 45 anchors / 15 flagship + 20 balanced + 10 fast is an external
feasibility target, not a promised output. Actual tier composition is counted,
not manufactured. These uniform-root samples are NOT label-stratified quota
samples. Requiring those quotas needs a new feasible stratified-root sampling
and weighting design, approved before outcomes. The purported 37 split IDs must
first be audited into genuine roots and near-duplicate closures.

A preregistration must lock BEFORE outcomes:

- sampling frame, root audit, selection seed, task counts, root inclusion design;
- estimand: **finite-frame equal-root mean of the predeclared bounded acceptance
  protocol score**, equal tasks within root, equal fixed trials within task;
- frozen ordered upgrading candidate chain (one per tier), including provider,
  model revision, effort, context policy and acceptance-spec digests;
- per-candidate/stage maxima and global cap, fixed exploration/confirmation counts,
  precommitted independent-root task/trial randomness schedule reference, alpha,
  reliability target, all-candidate comparison family and stopping rules;
- external approval, execution audit, shared-shock/independence review and private
  evidence handling. Any change produces a new protocol identity.

Cascade behavior for each screening anchor:

| Initial three trials | Action |
|---|---|
| >=2 failures | Eliminate that candidate here and upgrade to next tier |
| 3 passes | Fixed optional supplement, **exploration only**, stop chain |
| 2 passes | Ambiguous: stop shadow, no expansion or forced success |
| Exhaust all tiers | Shadow; no candidate certified |

Independent confirmation evaluates ALL preregistered candidates on reserved roots,
with fixed tasks/trials and no peeking, early success, screening-data pooling,
selective supplements or outcome-dependent enlargement. A missing or duplicated
trial is an error. Unused budget does not authorize extra testing. Worst-case cap
is `screen_anchors*(3*candidates + explore_trials) +
confirmation_anchors*confirmation_trials*candidates`; per-candidate stage caps are
also recorded. 270 executions is only an earlier budget sketch, and neither the
45-anchor count nor this prototype establishes 97% savings.

### Statistical boundary and synthetic evidence

The harness refuses real outcomes. Strong within-root correlation is simulated
with a shared latent uniform per root/candidate/stage; repeated trials never
increase `n_root`. Domain-separated, counter-addressable pseudo-random streams
bind root allocation, each frame root's task schedule and every candidate/stage
outcome separately. Changing screening branches, execution order or supplement
budgets cannot consume/change confirmation draws. Root task schedules are defined
for the whole frame before allocation. This reproducible CPU construction is not
proof of independence in a real executor. No IID binomial CP is applied to
correlated tasks or selectively expanded samples.

For diagnostic arithmetic, form each root's bounded [0,1] fixed-protocol score,
then its equal-root mean. The conservative root-level lower bound is
`max(0, mean - sqrt(log(M/alpha)/(2*R)))`, Bonferroni over ALL M preregistered
candidates. A defensible interpretation requires simple random root sampling
without replacement from the registered finite frame of fixed potential root
protocol scores (the Hoeffding sampling-without-replacement inequality), with
candidate/task/acceptance/trial schedules fixed before root/outcome inspection.
Task/trial random schedules must be precommitted independently of root selection;
conditioning on those schedules defines the finite-frame potential scores.
Unmodeled shared execution shocks or data-dependent task selection invalidate
that design. Without externally defensible root/design evidence, the bound is
null. To see the restricted design argument, condition on independently fixed
whole-frame schedules, giving fixed scores `z[r,c] in [0,1]`. Marginally over the
root allocation (not conditional on favorable screening), the R confirmation
roots form a simple random sample of the frame. The sampling-without-replacement
Hoeffding upper-tail inequality is
`P(mean_sample[c] - mean_frame[c] >= e) <= exp(-2*R*e^2)`.
Setting `e=sqrt(log(M/alpha)/(2*R))` and union-bounding the M fixed candidates
motivates the displayed diagnostic. Screening-dependent candidate changes,
conditioning on favorable screening, shared/adaptive task schedules or a new
population target are outside that argument. The simulator does not even supply
unobserved-frame outcomes, so it does NOT verify this coverage statement.
No superpopulation or long-run per-task reliability extrapolation is made.
Changing the estimand, sampling weights, root dependency or stopping rule requires
new statistical review; retaining the same formula does not preserve coverage.

`simulation_stress` counts diagnostic false accepts against the stated synthetic
sampled-root probability reference. It is a Monte Carlo code check, not evidence
that a real protocol has coverage. All synthetic reports are shadow with
`coverage_claim: false`, `qualified: false`, empty measured labels and no gate
updates, even if a diagnostic bound meets the proposal target. Population/root
reliability must NEVER be converted to each task's measured training label. Real
#8 labels require their own per-task acceptance evidence contract; existing
`outcomes.py` gates are neither bypassed nor weakened here.

## Acceptance ledger / reproducible CPU checks

Run `scripts/protocol_demo.py` for a wholly invented aggregate demonstration; it
accepts no model/data paths and cannot execute a candidate. Tests cover root
closure, deterministic folds, per-row/epoch source lineage, all nested roles,
insufficient class support, exact CRC boundaries/degenerates, soup semantic
refusals, budgets, fixed confirmation, multiplicity and correlated repetitions.

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/protocol_demo.py
```

Still external/unfulfilled: approved statistical/root/sampling/loss/5%/budget
protocols; actual development-pool/vision support audit; #18 data version and
#19/#20 training readiness; real per-epoch OOF coverage; model-byte/adapter merge
verification; soup inference, fresh fixed-model calibration and train/serve
parity; frozen candidate/route/acceptance identities; authorized GPU/API budgets
and actual independent task-success executions; valid independent confirmation
and per-task #8 measured evidence. Synthetic success closes none of these and
does not close #2, #21 or #26. No public issue/PR/status mutation is made here.
