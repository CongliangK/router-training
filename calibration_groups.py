"""Group-isolated calibration planning and out-of-fold selection (issue #2).

Stdlib-only on purpose: this module is shared by CPU structural preflight and
the public CI path, so it must not require torch, peft, or model weights, and
it must stay importable on a clean checkout.

Why this module exists (issue #2, external review P1-3): the archived recipe
shuffled the calibration split BY ROW into selection/fitting halves. Task ids
are unique, but near-duplicate tasks from one source share a split_group, so a
row bisection puts correlated rows on both sides and the halves leak groups.
With only ~8 recorded calibration groups (issue #2) no re-split of rows can
produce group-clean halves with usable class support, so the fix is not
another seed: new runs must pre-register whole groups into explicit roles.

Roles (pre-registered, never derived from a run seed):
- A: checkpoint / epoch selection rows.
- B: temperature and tau fitting rows, evaluated by deterministic GroupKFold
  out-of-fold selection (issue #2: "use LOGO/GroupKFold to select tau instead
  of treating rows as independent observations").
- C: reserved for final configuration selection; validated and then left
  untouched by this module (it must not silently borrow A/B groups).

Safety conventions: every validation error carries a reason and counts only,
never row ids, group names, or label values. Every numeric gate below marked
PROVISIONAL is an example value for synthetic validation; the production
numbers must come from an approved external protocol and are not frozen by
any user decision yet (see research/E2r1 discussion; it is NOT approved).
"""
from __future__ import annotations
import collections
import hashlib
import itertools
import json
import math

DEFAULT_SALT = 'router-calibration-diagnostic-v1'
ROLES = ('A', 'B', 'C')

# Router head schema mirroring model_utils.QUESTIONS (kept in sync by a unit
# test; this module must not import model_utils because that pulls in torch).
DEFAULT_SCHEMA = {'heads': {
    'tier': {'label_key': 'tier', 'classes': ['fast', 'balanced', 'flagship'],
             'required': True, 'calibration': 'temperature', 'adjust': True},
    'vision': {'label_key': 'needs_vision', 'classes': ['no', 'yes'],
               'required': True, 'calibration': 'temperature', 'adjust': False},
    'task_type': {'label_key': 'task_type',
                  'classes': ['extraction', 'research', 'implementation', 'debugging', 'review', 'visual', 'creative'],
                  'required': False, 'calibration': 'fixed', 'fixed_T': 1.0},
}}

# PROVISIONAL example support gates (NOT user-frozen): A5/B6/C5 role groups,
# >=5 rows and >=2 groups per required-head class per region, >=2 groups per
# class in every B-fold training union. A run that misses these fails hard;
# the implementation never downgrades vision to tier-only and never quietly
# lowers a gate to make data fit.
DEFAULT_SUPPORT_POLICY = {
    'min_groups_per_role': {'A': 5, 'B': 6, 'C': 5},
    'min_rows_per_class': 5,
    'min_groups_per_class': 2,
    'min_fold_train_groups_per_class': 2,
    'n_folds': 3,
}

# PROVISIONAL example numeric protocol (NOT user-frozen): tau grid, fitted
# temperature bounds, bounded-fit iteration/tolerance budget, and probability
# clip. Production runs must receive these numbers from the approved protocol
# in the roles config; comparisons across models must reuse one protocol.
DEFAULT_PROTOCOL = {
    'tau_grid': [0.0, 0.5, 1.0, 1.5, 2.0],
    'temperature_bounds': [0.25, 8.0],
    'fit_max_iterations': 100,
    'fit_tolerance': 1e-6,
    'probability_epsilon': 1e-12,
}


class PlanError(ValueError):
    """Safe validation failure: reason and counts only, no original values."""


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _hash(obj):
    return _sha(_canonical(obj))


def _logsumexp(values):
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def _numeric_core(protocol):
    # Hashable protocol identity: numbers only. Injectable callables (fit /
    # score / selector) are code, pinned by the repository, not by the file.
    return {k: v for k, v in protocol.items() if k not in ('fit', 'score', 'selector')}


def _merge(defaults, override):
    if override is None:
        return dict(defaults)
    if not isinstance(override, dict):
        raise PlanError('policy/protocol override must be an object')
    merged = dict(defaults)
    merged.update(override)
    return merged


def _merge_policy(override):
    policy = _merge(DEFAULT_SUPPORT_POLICY, override)
    counts = policy.get('min_groups_per_role')
    if not isinstance(counts, dict) or set(counts) != set(ROLES) or not all(isinstance(counts[r], int) and counts[r] >= 1 for r in ROLES):
        raise PlanError('support policy min_groups_per_role must map A/B/C to positive integers')
    for key in ('min_rows_per_class', 'min_groups_per_class', 'min_fold_train_groups_per_class'):
        value = policy.get(key)
        if not isinstance(value, int) or value < 1:
            raise PlanError(f'support policy {key} must be a positive integer')
    folds = policy.get('n_folds')
    if not isinstance(folds, int) or folds < 2:
        raise PlanError('support policy n_folds must be an integer >= 2')
    return policy


def _merge_protocol(override):
    protocol = _merge(DEFAULT_PROTOCOL, override)
    grid = protocol.get('tau_grid')
    if not isinstance(grid, list) or not grid or not all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in grid):
        raise PlanError('protocol tau_grid must be a non-empty list of finite numbers >= 0')
    lo, hi = protocol.get('temperature_bounds', [0, 0])
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and 0 < lo < hi):
        raise PlanError('protocol temperature_bounds must satisfy 0 < low < high')
    if not isinstance(protocol.get('fit_max_iterations'), int) or protocol['fit_max_iterations'] < 1:
        raise PlanError('protocol fit_max_iterations must be a positive integer')
    if not isinstance(protocol.get('fit_tolerance'), (int, float)) or protocol['fit_tolerance'] <= 0:
        raise PlanError('protocol fit_tolerance must be > 0')
    epsilon = protocol.get('probability_epsilon')
    if not isinstance(epsilon, (int, float)) or not (0 < epsilon < 1):
        raise PlanError('protocol probability_epsilon must be in (0, 1)')
    return protocol


def validate_rows(rows, schema):
    """Every row needs a unique non-empty string id, a non-empty string
    split_group, and legal labels on every required head. Optional heads may
    lack or hold out-of-schema labels; such rows simply do not support them
    (support accounting reflects that; nothing is silently relabeled)."""
    if not isinstance(rows, list) or not rows:
        raise PlanError('rows must be a non-empty list')
    heads = schema['heads']
    seen = set()
    bad_id = bad_group = bad_label = duplicates = 0
    for row in rows:
        rid = row.get('id') if isinstance(row, dict) else None
        if not isinstance(rid, str) or not rid:
            bad_id += 1
            continue
        if rid in seen:
            duplicates += 1
            continue
        seen.add(rid)
        group = row.get('split_group')
        if not isinstance(group, str) or not group:
            bad_group += 1
            continue
        label = row.get('label') if isinstance(row.get('label'), dict) else {}
        for head, spec in heads.items():
            if spec.get('required') and label.get(spec['label_key']) not in spec['classes']:
                bad_label += 1
    if bad_id or bad_group or bad_label or duplicates:
        raise PlanError(f'row validation failed: invalid_id={bad_id} invalid_split_group={bad_group} '
                        f'illegal_required_label={bad_label} duplicate_id={duplicates} (values withheld)')


def _label(row, head, spec):
    label = row.get('label') if isinstance(row.get('label'), dict) else {}
    return label.get(spec['label_key'])


def _support(rows, head, spec):
    """Per-class row counts and distinct-group counts for one head."""
    row_counts = collections.Counter()
    group_sets = collections.defaultdict(set)
    for row in rows:
        value = _label(row, head, spec)
        if value in spec['classes']:
            row_counts[value] += 1
            group_sets[value].add(row['split_group'])
    return {c: {'rows': row_counts[c], 'groups': len(group_sets[c])} for c in spec['classes']}


def _group_row_counts(rows):
    counts = collections.Counter(r['split_group'] for r in rows)
    return dict(counts)


def group_folds(group_counts, n_folds=3):
    """Deterministic whole-group fold assignment: groups sorted by row count
    descending with sha256(group) as the stable tiebreak, each placed whole
    into the fold with the fewest rows so far (ties to the lowest index).
    Groups are never split across folds, so every B row is a holdout row in
    exactly one fold."""
    order = sorted(group_counts, key=lambda g: (-group_counts[g], _sha(g)))
    folds = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for group in order:
        index = sizes.index(min(sizes))
        folds[index].append(group)
        sizes[index] += group_counts[group]
    return folds


def _normalize_roles(role_groups, group_counts):
    if not isinstance(role_groups, dict) or set(role_groups) != set(ROLES):
        raise PlanError('role_groups must pre-register exactly roles A (checkpoint selection), '
                        'B (temperature/tau fitting), and C (reserved); strict mode has no '
                        'seed-based fallback re-split')
    normalized = {}
    empty = dup_within = 0
    for role in ROLES:
        groups = role_groups[role]
        if not isinstance(groups, list) or not groups:
            raise PlanError(f'role {role} must list at least one group (strict preflight hard-fail; no fallback)')
        clean = []
        for group in groups:
            if not isinstance(group, str) or not group:
                empty += 1
            elif group in clean:
                dup_within += 1
            else:
                clean.append(group)
        normalized[role] = clean
    if empty or dup_within:
        raise PlanError(f'role_groups invalid: empty_group_name={empty} duplicated_within_role={dup_within} (values withheld)')
    overlap = sum(len(set(normalized[a]) & set(normalized[b])) for a, b in itertools.combinations(ROLES, 2))
    if overlap:
        raise PlanError(f'role_groups overlap: {overlap} group(s) appear in more than one role; '
                        'A/B/C must be mutually exclusive (values withheld)')
    missing_rows = sum(1 for role in ROLES for g in normalized[role] if group_counts.get(g, 0) == 0)
    if missing_rows:
        raise PlanError(f'role_groups references {missing_rows} group(s) with no rows in the pool (values withheld)')
    return normalized


def build_plan(rows, role_groups, schema, support_policy=None, protocol=None):
    """Validate and lock a strict/protocol calibration plan.

    Checks (all hard failures; nothing is downgraded or re-split by a run
    seed): unique/legal rows, pre-registered disjoint A/B/C groups covering
    every calibration row exactly once, per-role group minima, per-class row
    and group minima per required head per region, B>=3 groups, deterministic
    whole-group folds, and per-fold training-union class coverage. Optional
    temperature heads that lack fold coverage are downgraded to T=1 with
    calibrated=False (reported, never silently recalibrated).
    """
    policy = _merge_policy(support_policy)
    proto = _merge_protocol(protocol)
    validate_rows(rows, schema)
    counts = _group_row_counts(rows)
    roles = _normalize_roles(role_groups, counts)

    known = set().union(*(set(roles[r]) for r in ROLES))
    uncovered = [r for r in rows if r['split_group'] not in known]
    unknown_groups = {r['split_group'] for r in uncovered}
    if uncovered:
        raise PlanError(f'plan does not cover the pool: uncovered_rows={len(uncovered)} '
                        f'unknown_groups={len(unknown_groups)}; A/B/C must fully cover all calibration groups '
                        '(values withheld)')

    # Canonical id order inside every role: the plan is byte-stable under any
    # input row permutation (fold assignment and hashes already are).
    rows_by_role = {role: sorted((r for r in rows if r['split_group'] in set(roles[role])),
                                 key=lambda r: r['id']) for role in ROLES}

    below_role = [r for r in ROLES if len(roles[r]) < policy['min_groups_per_role'][r]]
    if below_role:
        raise PlanError(f'role group minima missed for {len(below_role)} role(s) '
                        f'(policy A>={policy["min_groups_per_role"]["A"]} B>={policy["min_groups_per_role"]["B"]} '
                        f'C>={policy["min_groups_per_role"]["C"]})')
    if len(roles['B']) < 3:
        raise PlanError('role B needs at least 3 groups for grouped-fold tau selection (issue #2)')
    n_folds = policy['n_folds']
    if len(roles['B']) < n_folds:
        raise PlanError(f'role B has fewer groups ({len(roles["B"])}) than n_folds ({n_folds})')

    support = {role: {head: _support(rows_by_role[role], head, spec) for head, spec in schema['heads'].items()} for role in ROLES}
    for role in ROLES:
        for head, spec in schema['heads'].items():
            if not spec.get('required'):
                continue
            table = support[role][head]
            below_rows = sum(1 for c in spec['classes'] if table[c]['rows'] < policy['min_rows_per_class'])
            below_groups = sum(1 for c in spec['classes'] if table[c]['groups'] < policy['min_groups_per_class'])
            if below_rows or below_groups:
                raise PlanError(f'role {role} head {head}: classes_below_rows_minimum={below_rows} '
                                f'classes_below_groups_minimum={below_groups} '
                                f'(policy rows>={policy["min_rows_per_class"]} groups>={policy["min_groups_per_class"]}); '
                                'strict gates are not silently relaxed')

    b_counts = {g: counts[g] for g in roles['B']}
    folds = group_folds(b_counts, n_folds)
    fold_sets = [set(f) for f in folds]
    for head, spec in schema['heads'].items():
        if not spec.get('required'):
            continue
        failing = 0
        for fold_set in fold_sets:
            train_rows = [r for r in rows_by_role['B'] if r['split_group'] not in fold_set]
            table = _support(train_rows, head, spec)
            failing += sum(1 for c in spec['classes'] if table[c]['groups'] < policy['min_fold_train_groups_per_class'])
        if failing:
            raise PlanError(f'head {head}: fold_training_classes_below_minimum={failing} '
                            f'(policy groups>={policy["min_fold_train_groups_per_class"]} per fold); '
                            'group-isolated calibration refuses to train a fold without required-class support')

    optional_head_status = {}
    for head, spec in schema['heads'].items():
        if spec.get('required'):
            optional_head_status[head] = {'mode': 'required', 'calibratable': True}
            continue
        if spec.get('calibration') == 'fixed':
            optional_head_status[head] = {'mode': 'fixed', 'calibratable': False,
                                          'fixed_T': spec.get('fixed_T', 1.0),
                                          'reason': 'fixed_by_protocol'}
            continue
        affected = 0
        for fold_set in fold_sets:
            train_rows = [r for r in rows_by_role['B'] if r['split_group'] not in fold_set]
            table = _support(train_rows, head, spec)
            affected += 1 if any(table[c]['groups'] < 1 for c in spec['classes']) else 0
        calibratable = affected == 0
        optional_head_status[head] = {'mode': 'optional-temperature', 'calibratable': calibratable,
                                      'reason': None if calibratable else 'fold_training_class_missing',
                                      'folds_affected': affected}

    plan = {
        'mode': 'protocol',
        'diagnostic_only': False,
        'schema': schema,
        'policy': policy,
        'protocol': proto,
        'roles': roles,
        'rows_by_role': rows_by_role,
        'folds': folds,
        'support': support,
        'optional_head_status': optional_head_status,
    }
    plan['plan_hash'] = _hash({'schema': schema, 'policy': policy, 'protocol': _numeric_core(proto),
                               'roles': {r: sorted(roles[r]) for r in ROLES}})
    plan['protocol_hash'] = _hash(_numeric_core(proto))
    return plan


def diagnostic_halves(rows, schema, seed='', salt=DEFAULT_SALT, support_policy=None):
    """DIAGNOSTIC-ONLY default grouping: SHA256(seed + fixed salt + group)
    splits recorded groups into two whole-group halves and reports the plan
    and its insufficiencies. It can never produce selected tau/T or any
    calibration product: the returned report is marked diagnostic_only and
    select_tau_oof refuses such plans. This exists to show, honestly and
    ahead of training, why a pool cannot support the strict gates."""
    policy = _merge_policy(support_policy)
    validate_rows(rows, schema)
    selection, fitting = [], []
    for group in sorted({r['split_group'] for r in rows}):
        bit = int(_sha(f'{seed}{salt}{group}')[:8], 16) % 2
        (selection if bit else fitting).append(group)
    support = {'selection': {head: _support([r for r in rows if r['split_group'] in set(selection)], head, spec)
                             for head, spec in schema['heads'].items()},
               'fitting': {head: _support([r for r in rows if r['split_group'] in set(fitting)], head, spec)
                           for head, spec in schema['heads'].items()}}
    insufficient = []
    for half, table in support.items():
        groups = len(selection if half == 'selection' else fitting)
        mins = {'A': policy['min_groups_per_role']['A'], 'B': policy['min_groups_per_role']['B']}
        if groups < max(mins.values()):
            insufficient.append(f'{half}: group_count_below_example_gates')
        for head, spec in schema['heads'].items():
            if not spec.get('required'):
                continue
            per = table[head]
            if any(per[c]['rows'] < policy['min_rows_per_class'] for c in spec['classes']):
                insufficient.append(f'{half}:head {head}: classes_below_rows_minimum')
            if any(per[c]['groups'] < policy['min_groups_per_class'] for c in spec['classes']):
                insufficient.append(f'{half}:head {head}: classes_below_groups_minimum')
    return {'mode': 'diagnostic', 'diagnostic_only': True, 'seed': seed, 'salt': salt,
            'selection_groups': selection, 'fitting_groups': fitting,
            'selection_rows': sum(1 for r in rows if r['split_group'] in set(selection)),
            'fitting_rows': sum(1 for r in rows if r['split_group'] in set(fitting)),
            'support': support, 'insufficient': insufficient}


def compute_prior(rows, head, schema, add_one=True):
    """Real-unique-train class prior for an adjusted head: drops repeated ids,
    excludes rows flagged synthetic, and REFUSES when any class has no unique
    real support (a silent zero-count prior is not acceptable for new runs;
    the archived legacy path keeps its own original prior untouched).
    add-one smoothing is a PROVISIONAL protocol choice."""
    spec = schema['heads'][head]
    classes = spec['classes']
    seen = set()
    counts = collections.Counter()
    excluded_synth = excluded_dup = 0
    for row in rows:
        rid = row.get('id')
        if rid in seen:
            excluded_dup += 1
            continue
        seen.add(rid)
        if row.get('synthetic'):
            excluded_synth += 1
            continue
        value = _label(row, head, spec)
        if value in classes:
            counts[value] += 1
    missing = sum(1 for c in classes if counts[c] == 0)
    if missing:
        raise PlanError(f'prior head {head}: {missing} class(es) without unique real train support; '
                        'refusing to calibrate against an unsupported prior')
    total = sum(counts.values())
    if add_one:
        probs = {c: (counts[c] + 1) / (total + len(classes)) for c in classes}
    else:
        probs = {c: counts[c] / total for c in classes}
    return {'head': head, 'counts': {c: counts[c] for c in classes}, 'add_one': bool(add_one),
            'excluded_synthetic': excluded_synth, 'excluded_duplicate': excluded_dup,
            'probabilities': probs}


def fit_temperature(logits, class_indices, groups, bounds=(0.25, 8.0), max_iterations=100,
                    tolerance=1e-6, epsilon=1e-12):
    """Deterministic bounded golden-section fit of ONE temperature to
    ALREADY-ADJUSTED logits (the issue #2 contract is adjust-first: never fit
    T on raw logits and apply tau afterwards). Objective: group-equal mean NLL
    — each group contributes its mean row NLL equally, because groups (not
    rows) are the assumed statistical unit; near-duplicate rows inside one
    group are correlated and must not count as independent observations.
    Bounded deterministic search with fixed iteration/tolerance budget; NOT a
    global-optimum guarantee for pathological objectives. Selection rule,
    tolerance, and iteration count are protocol numbers, documented here."""
    if not logits:
        raise PlanError('temperature fit requires at least one row')
    lo, hi = float(bounds[0]), float(bounds[1])
    by_group = collections.defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)

    def nll(temperature):
        total = 0.0
        for members in by_group.values():
            mean = 0.0
            for i in members:
                scaled = [v / temperature for v in logits[i]]
                lp = scaled[class_indices[i]] - _logsumexp(scaled)
                mean += -max(lp, math.log(epsilon))
            total += mean / len(members)
        return total / len(by_group)

    a, b = lo, hi
    golden = (math.sqrt(5) - 1) / 2
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(int(max_iterations)):
        if b - a <= tolerance:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = nll(d)
    return (a + b) / 2


def score_holdout(logits, class_indices, groups, temperature, epsilon=1e-12):
    """Group-equal mean NLL of holdout rows under (adjusted logits)/T. Holdout
    rows are only scored, never fitted. Missing-class nullability is decided
    by the caller (select_tau_oof returns None for that head/fold); this
    function never fabricates a 0 score."""
    by_group = collections.defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    total = 0.0
    for members in by_group.values():
        mean = 0.0
        for i in members:
            scaled = [v / temperature for v in logits[i]]
            lp = scaled[class_indices[i]] - _logsumexp(scaled)
            mean += -max(lp, math.log(epsilon))
        total += mean / len(members)
    return total / len(by_group)


def default_selector(candidates):
    """PROVISIONAL default tau selector: among candidates whose required heads
    all scored out-of-fold, pick the lowest pooled group-equal NLL, breaking
    ties toward the smaller tau. This is a development estimate under the
    example protocol, not the approved production rule; an approved protocol
    can inject its own selector. If nothing is feasible, no tau is selected."""
    feasible = [c for c in candidates if c['feasible'] and c['pooled'] is not None]
    if not feasible:
        return None
    return min(feasible, key=lambda c: (c['pooled'], c['tau']))['tau']


def select_tau_oof(plan, rows, prior, checkpoint_ref, protocol=None):
    """Out-of-fold tau/T selection over role B (the issue #2 mechanism).

    Order of operations is fixed: the A-side checkpoint must already be
    locked (checkpoint_ref is mandatory); for every tau candidate each fold
    first applies z - tau*log(prior) to its TRAINING rows, then fits that
    fold's temperatures on those adjusted training rows, then scores the
    holdout fold with the same (z - tau*log(prior))/T formula. Groups are
    equally weighted everywhere (groups, not rows, are the statistical
    unit). Holdout folds missing a required head's class yield a null score
    for that head (never a fake 0) and make the tau infeasible; if no tau is
    feasible nothing is selected and no temperatures are produced. After a
    tau is chosen, temperatures are refitted once on the full B pool with the
    SAME adjust-then-fit order, and the result binds plan/prior/schema/
    protocol/checkpoint hashes so compared models provably shared one plan,
    folds, grid, budget, and order. Tau selection is a development estimate;
    this procedure does not make the whole evaluation unbiased."""
    if not isinstance(plan, dict) or plan.get('mode') != 'protocol' or plan.get('diagnostic_only'):
        raise PlanError('out-of-fold selection requires a pre-registered protocol plan; '
                        'diagnostic halves cannot produce calibration products')
    if not isinstance(checkpoint_ref, str) or not checkpoint_ref.strip():
        raise PlanError('checkpoint must be locked by role A before tau selection (adjust-then-select order)')
    proto = _merge_protocol(protocol or plan['protocol'])
    schema = plan['schema']
    heads = schema['heads']
    b_rows = plan['rows_by_role']['B']
    b_ids = [r['id'] for r in b_rows]

    if not isinstance(rows, list):
        raise PlanError('scored rows must be a list')
    scored = {}
    for row in rows:
        rid = row.get('id') if isinstance(row, dict) else None
        if rid in scored:
            raise PlanError('scored rows contain duplicate ids (values withheld)')
        scored[rid] = row
    missing = sum(1 for rid in b_ids if rid not in scored)
    extra = sum(1 for rid in scored if rid not in set(b_ids))
    if missing or extra:
        raise PlanError(f'scored rows do not match role B exactly: missing={missing} extra={extra}; '
                        'roles A/C and any test rows must never reach temperature/tau fitting')
    for row in rows:
        for head, spec in heads.items():
            if spec.get('calibration') != 'temperature':
                continue
            values = (row.get('logits') or {}).get(head)
            if not isinstance(values, list) or len(values) != len(spec['classes']) \
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise PlanError(f'scored logits invalid for head {head} (row id withheld)')

    for head, spec in heads.items():
        if spec.get('calibration') == 'temperature' and spec.get('adjust'):
            probs = (prior or {}).get(head)
            if not isinstance(probs, dict) or set(probs) != set(spec['classes']) \
                    or not all(isinstance(p, (int, float)) and 0 < p <= 1 for p in probs.values()):
                raise PlanError(f'prior missing or malformed for adjusted head {head}')

    fit = proto.get('fit') or fit_temperature
    score = proto.get('score') or score_holdout
    select = proto.get('selector') or default_selector
    bounds = tuple(proto['temperature_bounds'])
    iterations = proto['fit_max_iterations']
    tolerance = proto['fit_tolerance']
    epsilon = proto['probability_epsilon']
    fold_sets = [set(f) for f in plan['folds']]
    b_group_count = len(plan['roles']['B'])
    status = plan['optional_head_status']
    temperature_heads = [h for h, spec in heads.items()
                         if spec.get('calibration') == 'temperature' and status[h]['calibratable']]
    required_temperature_heads = [h for h in temperature_heads if heads[h].get('required')]

    def adjusted(head, spec, values, tau):
        if not spec.get('adjust'):
            return [float(v) for v in values]
        probs = prior[head]
        return [float(v) - tau * math.log(probs[c]) for c, v in zip(spec['classes'], values)]

    candidates = []
    for tau in proto['tau_grid']:
        folds_report = []
        feasible = True
        for fold_set in fold_sets:
            train = [r for r in b_rows if r['split_group'] not in fold_set]
            holdout = [r for r in b_rows if r['split_group'] in fold_set]
            temperatures = {}
            head_scores = {}
            for head in temperature_heads:
                spec = heads[head]
                train_logits = [adjusted(head, spec, r['logits'][head], tau) for r in train]
                train_cls = [spec['classes'].index(_label(r, head, spec)) for r in train]
                train_groups = [r['split_group'] for r in train]
                temperature = fit(train_logits, train_cls, train_groups, bounds, iterations, tolerance, epsilon)
                temperatures[head] = temperature
                present = {_label(r, head, spec) for r in holdout}
                if present != set(spec['classes']):
                    # Holdout fold lacks a class for this head: report null, never a fake 0.
                    head_scores[head] = None
                    if spec.get('required'):
                        feasible = False
                    continue
                holdout_logits = [adjusted(head, spec, r['logits'][head], tau) for r in holdout]
                holdout_cls = [spec['classes'].index(_label(r, head, spec)) for r in holdout]
                holdout_groups = [r['split_group'] for r in holdout]
                head_scores[head] = score(holdout_logits, holdout_cls, holdout_groups, temperature, epsilon)
            folds_report.append({'holdout_groups': len(fold_set), 'holdout_rows': len(holdout),
                                 'train_rows': len(train), 'temperatures': temperatures,
                                 'head_scores': head_scores})
        pooled_by_head = {}
        for head in temperature_heads:
            if head not in required_temperature_heads:
                continue
            values = [(f['head_scores'][head], f['holdout_groups']) for f in folds_report]
            if any(v is None for v, _ in values):
                pooled_by_head[head] = None
            else:
                # Group-equal pooling: weight each fold score by its holdout group count.
                pooled_by_head[head] = sum(v * n for v, n in values) / sum(n for _, n in values)
        pooled = (None if any(v is None for v in pooled_by_head.values())
                  else sum(pooled_by_head.values()) / max(1, len(pooled_by_head)))
        candidates.append({'tau': tau, 'feasible': feasible, 'pooled': pooled,
                           'pooled_by_head': pooled_by_head, 'folds': folds_report})

    binding = {'plan_hash': plan['plan_hash'], 'protocol_hash': _hash(_numeric_core(proto)),
               'prior_hash': _hash(prior), 'schema_hash': _hash(schema), 'checkpoint_ref': checkpoint_ref}
    chosen = select(candidates)
    if chosen is None:
        return {'status': 'insufficient', 'selected_tau': None, 'candidates': candidates,
                'binding': binding,
                'note': 'no feasible tau under the registered protocol; nothing selected or refit'}
    if not any(math.isclose(c['tau'], chosen, rel_tol=0, abs_tol=1e-12) for c in candidates):
        raise PlanError('selector returned a tau outside the registered grid (protocol violation)')

    temperatures = {}
    calibrated = {}
    for head, spec in heads.items():
        state = status[head]
        if spec.get('calibration') != 'temperature' or not state['calibratable']:
            fixed = state.get('fixed_T', 1.0)
            temperatures[head] = fixed
            calibrated[head] = False
            continue
        logits = [adjusted(head, spec, r['logits'][head], chosen) for r in b_rows]
        cls = [spec['classes'].index(_label(r, head, spec)) for r in b_rows]
        groups = [r['split_group'] for r in b_rows]
        temperatures[head] = fit(logits, cls, groups, bounds, iterations, tolerance, epsilon)
        calibrated[head] = True
    return {'status': 'selected', 'selected_tau': chosen, 'temperatures': temperatures,
            'calibrated': calibrated, 'candidates': candidates, 'binding': binding,
            'note': 'development estimate over role B folds; group-equal weighting; not an unbiased final evaluation'}


def anonymous_summary(plan):
    """Counts-only plan summary safe for public logs: role/fold group and row
    counts, per-head class support, hashes. Never group names, ids, labels,
    or prompts. Used by train_eval, scripts/train.py --dry-run, and
    scripts/validate_calibration.py so all three print the same truth."""
    roles = plan['roles']
    rows_by_role = plan['rows_by_role']
    counts = _group_row_counts([r for role in ROLES for r in rows_by_role[role]])
    folds = [{'fold': i, 'groups': len(f), 'rows': sum(counts[g] for g in f)}
             for i, f in enumerate(plan['folds'])]
    return {'mode': plan['mode'], 'roles': {r: {'groups': len(roles[r]), 'rows': len(rows_by_role[r])} for r in ROLES},
            'folds': folds, 'support': plan['support'],
            'optional_head_status': plan['optional_head_status'],
            'plan_hash': plan['plan_hash'], 'protocol_hash': plan['protocol_hash']}


def preflight_roles(calibration_rows, roles_config):
    """Shared preflight used by every entry point: builds and validates a
    protocol plan from a roles config object ({'role_groups': {...},
    'support_policy': {...}?, 'protocol': {...}?, 'schema': {...}?})."""
    if not isinstance(roles_config, dict) or 'role_groups' not in roles_config:
        raise PlanError("roles config must be an object with a 'role_groups' key")
    schema = roles_config.get('schema') or DEFAULT_SCHEMA
    return build_plan(rows=calibration_rows, role_groups=roles_config['role_groups'], schema=schema,
                      support_policy=roles_config.get('support_policy'),
                      protocol=roles_config.get('protocol'))


def smoke_subpool(rows, role_groups, schema, support_policy=None, protocol=None):
    """--smoke for grouped runs must not slice rows across pre-registered
    groups: that would rebuild the issue #2 leak in miniature. Whole groups
    are dropped (smallest-support first, deterministic order) and the SAME
    policy re-validates the sub-pool; if no whole-group sub-pool satisfies
    the gates, this fails — gates are never relaxed for smoke."""
    policy = _merge_policy(support_policy)
    counts = _group_row_counts(rows)
    roles = _normalize_roles(role_groups, counts)
    ordered = {role: sorted(groups, key=lambda g: (counts[g], _sha(g))) for role, groups in roles.items()}
    ranges = [range(policy['min_groups_per_role'][r], len(ordered[r]) + 1) for r in ROLES]
    for keep_a, keep_b, keep_c in itertools.product(*ranges):
        chosen = {'A': ordered['A'][:keep_a], 'B': ordered['B'][:keep_b], 'C': ordered['C'][:keep_c]}
        keep = set(chosen['A']) | set(chosen['B']) | set(chosen['C'])
        subset = [r for r in rows if r['split_group'] in keep]
        try:
            return build_plan(subset, chosen, schema, support_policy=support_policy, protocol=protocol)
        except PlanError:
            continue
    raise PlanError('no whole-group smoke sub-pool satisfies the support policy; '
                    'refusing to relax strict gates (issue #2)')
