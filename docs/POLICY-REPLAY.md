# Policy observation and local replay (#6 / #23)

## Compatibility and safety

`select(prediction, routes, has_image=False, minimum_confidence=.9)` remains valid.
The conservative `legacy-v1` default retains the choice + confidence fallback:
`choice == yes` or vision confidence below `minimum_confidence` requires vision.
This is a conservative policy, not a sign error. On coherent binary predictions,
a confidence cutoff of .9 effectively flags positive probabilities above .1
(the equality case still follows the original strict confidence comparison).
There is no newly calibrated default and no change to actual route capabilities.

Opt in explicitly with keyword arguments `vision_policy='positive-v2'` and
`vision_positive_threshold=<predeclared value in [0,1]>`. The rule is
`has_image or P(vision=yes) >= threshold`, including equality. A threshold without
this version, or this version without an explicit threshold, is rejected. The
server exposes equivalent `--vision-policy positive-v2
--vision-positive-threshold <value>` flags. `.35` is an unvalidated experiment,
not a recommended production threshold; reducing forced upgrades can increase
vision false negatives. `minimum_confidence` is never repurposed as this threshold.

Both versions require vision for absent/invalid/nonfinite/out-of-range or boolean
positive probabilities. They do not infer a missing probability from confidence.
`vision_probability` is `null` for invalid values, so JSON never emits NaN/Inf.
`has_image=True` always requires vision. Capability filtering never downgrades
below the required tier; truncation still requires flagship. No candidate means
refusal, not permission. The shared model truncation path is unchanged.

## Additive decision contract

The existing server permission envelope remains `schema_version: 2` with unchanged
shadow/qualified and diagnostic/execution gates. `decision_schema_version: 1` adds:

- `model_tier`: classifier argmax; `required_tier`: after context policy;
  `selected_tier`: actual candidate tier, or `null` on refusal.
- `reason_details`: ordered objects with `category`, `code`, and `upgraded`.
  Stable category ordering is `model_choice`, `vision_forced`,
  `context_truncation`, `risk_policy`. Multiple reasons coexist.
- `vision_policy`, `vision_positive_threshold`, and sanitized `vision_probability`.

`model_choice` always records the classifier tier, including an active flagship
choice. `vision_forced` compares selection/refusal with eligible candidates at the
same required tier **without** the vision filter; a missing intermediate tier or
context upgrade is not falsely attributed to vision. A capability refusal carries
`vision-capability-refusal` with `upgraded: false`. Other candidate-tier gaps and
refusals are `risk_policy`. Existing human-readable `reasons` remain compatible.
The historical low-confidence string mentions effort compensation; its structured
counterpart explicitly says `compensation_applied: false`. This server performs
**no effort compensation or downstream model execution**.

`recommendation_contract_version: 1` preserves unique `(provider, model)` identity
and forwards already declared optional `route_id`, `model_version`, and
`reasoning_effort` unchanged. This is metadata pass-through, not evidence of backend
support or execution. Missing fields remain absent. Duplicate provider/model pairs
remain rejected even with different efforts or route IDs; no multi-operating-point
schema is introduced. Existing outcomes fingerprints and declared-effort/version
checks remain unchanged. Consumers must validate backend support separately.

The `/route` response and observation record carry the decision contract. An
execution-mode no-candidate 503 also includes decision fields, without granting
execution. Pre-scoring gate failures have no policy decision and remain 503.

## CPU replay entry

No model, weights, service, network, or GPU is used:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void python policy_replay.py \
  --synthetic --vision-positive-threshold .35 --output /tmp/policy-synthetic.json
```

The output path must not exist. `--synthetic` is explicitly illustrative, not actual
private/frozen prediction replay. Output includes the legacy confidence sensitivity
at .8/.85/.9/.95, all-decision reason prevalence, upgrade-attributed reason shares,
refusal/upgrade rates, and a simulated balanced-vision capability comparison.
The simulation changes balanced capability declarations **in memory only** (or adds
an explicitly synthetic candidate if no balanced tier exists). It never authorizes
or verifies actual candidates. Reasons overlap; shares need not sum to one. Refusals
are not upgrades or savings. Empty denominators return `null`.

For an authorized private/local operator, use `--input /local/predictions.jsonl`
(and optionally `--routes /local/routes.json`), **keeping inputs and outputs outside
public checkouts**. Each JSONL row has:

```json
{"prediction":{"answers":{"tier":{"choice":"balanced","confidence":0.95},"vision":{"choice":"no","confidence":0.8,"probabilities":{"yes":0.2}}},"context_truncated":false},"has_image":false,"vision_label":"no"}
```

This example is purely synthetic. `has_image` is required and boolean. Optional
`vision_label` is `yes` or `no`; only `yes` rows enter vision FN/refusal denominators.
No labels means those rates are `null`, not zero. A vision false negative here means
`vision_label=yes` but policy `needs_vision=false`; it is not measured task failure.
Local inputs are labeled `local-unverified`; the tool cannot certify their origin.
Extra prompt/ID fields are ignored and never written into aggregate output.

Optional `--costs /local/costs.json` supplies explicit per-selected-decision estimates:
`[{"provider":"synthetic","model":"one","cost":1.0}]`. Cost units and assumptions
belong to the operator; they are not model prices or ordinal tier-distance losses.
Total estimated cost is `null` unless every selected decision has cost coverage;
coverage counts remain visible. Report refusals alongside cost and never interpret
refusal-induced cost reduction as savings. The synthetic invocation omits costs
rather than inventing backend prices. Review even aggregate private output before
publication; the CLI never emits prompts, IDs, or individual predictions.

## Acceptance boundary

CPU tests cover schema migration, threshold boundaries, high-confidence text,
mandatory images, invalid probabilities, no candidate, truncation, duplicate
identity, declared metadata pass-through and unchanged shadow/execution gating.
These satisfy development behavior, not empirical risk or capability validation.
Real private replay/reason rates and confidence sensitivity, authorized candidate
capability confirmation, backend effort support, threshold calibration/risk-cost
tradeoffs and independent production qualification remain #27 external acceptance.
No backend request or effort compensation is implemented. Reference #6/#23; do not
claim synthetic ratios are real savings or that forced upgrades must approach zero.
