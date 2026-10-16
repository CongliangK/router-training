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
numbers must come from an approved external protocol that is still PENDING
(none of these example numbers is frozen by any user decision).
"""
from __future__ import annotations
import ast
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path

DEFAULT_SALT = 'router-calibration-diagnostic-v1'
ROLES = ('A', 'B', 'C')

# Calibration-side flags per business head. Classes ALWAYS come from
# model_utils.QUESTIONS — the single source of truth — via
# schema_from_questions: train_eval passes the runtime QUESTIONS object, the
# stdlib preflight passes the same file parsed by questions_from_source
# (AST constants only; this module must not import model_utils because that
# pulls in torch). A unit test asserts both assemblies agree and that the
# parser follows the real source (mutation canary), so the two entry points
# cannot drift apart and no hand-maintained schema copy exists here.
BUSINESS_HEAD_FLAGS = {
    'tier': {'label_key': 'tier', 'required': True, 'calibration': 'temperature', 'adjust': True},
    'vision': {'label_key': 'needs_vision', 'required': True, 'calibration': 'temperature', 'adjust': False},
    'task_type': {'label_key': 'task_type', 'required': False, 'calibration': 'fixed', 'fixed_T': 1.0},
}


def schema_from_questions(questions, schema_version=None):
    """Assemble the calibration schema from a QUESTIONS-shaped mapping
    ({head: {'criteria': {class: ...}}}). Single assembly point shared by the
    runtime path (train_eval.grouped_schema) and the stdlib source-parsed
    path (default_business_schema); vision stays REQUIRED here — nothing may
    downgrade it to tier-only."""
    heads = {}
    for head, flags in BUSINESS_HEAD_FLAGS.items():
        spec = dict(flags)
        criteria = (questions or {}).get(head, {}).get('criteria')
        if not isinstance(criteria, dict) or not criteria \
                or not all(isinstance(c, str) and c for c in criteria):
            raise PlanError(f'business schema needs an ordered non-empty class table for head {head}')
        spec['classes'] = list(criteria)
        heads[head] = spec
    result={'heads': heads}
    if schema_version is not None:
        from task_label_schema import version, V2, SIX_TYPES
        if version(schema_version)==V2:
            if heads['task_type']['classes']!=list(SIX_TYPES):
                raise PlanError('v2 business schema needs exactly the ordered six-task classes')
            result['task_label_schema_version']=V2
    return result


def questions_from_source(path=None):
    """Load ONLY the QUESTIONS/TIERS/TYPES constants from the real
    model_utils.py source via the ast module (never imports torch/peft).
    Returns {head: {'criteria': {class: None}}} preserving class order;
    raises PlanError when the file or its literal structure is missing, so a
    broken source fails loudly instead of falling back to a stale copy."""
    source = Path(path) if path is not None else Path(__file__).resolve().parent / 'model_utils.py'
    try:
        tree = ast.parse(source.read_text(encoding='utf-8'))
    except (OSError, SyntaxError, ValueError) as error:
        raise PlanError(f'cannot read model_utils source for the business schema ({type(error).__name__}); '
                        'the stdlib preflight refuses to guess a schema')
    lists = {}
    questions_node = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('TIERS', 'TYPES'):
                try:
                    lists[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    raise PlanError(f'model_utils {target.id} is not a literal list')
            if isinstance(target, ast.Name) and target.id == 'QUESTIONS':
                questions_node = node.value
    if questions_node is None or not isinstance(questions_node, ast.Dict):
        raise PlanError('model_utils has no literal QUESTIONS table')
    questions = {}
    for key_node, value_node in zip(questions_node.keys, questions_node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise PlanError('QUESTIONS field names must be string literals')
        head = key_node.value
        if not isinstance(value_node, ast.Dict):
            raise PlanError(f'QUESTIONS field {head} is not a literal object')
        criteria_node = None
        for sub_key, sub_value in zip(value_node.keys, value_node.values):
            if isinstance(sub_key, ast.Constant) and sub_key.value == 'criteria':
                criteria_node = sub_value
        if criteria_node is None:
            raise PlanError(f'QUESTIONS field {head} has no criteria table')
        classes = None
        if isinstance(criteria_node, ast.Dict):  # tier / vision: literal keys
            classes = [k.value for k in criteria_node.keys
                       if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        else:
            # task_type: dict(zip(TYPES, [...])) — classes come from TYPES.
            call = criteria_node
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'dict' \
                    and len(call.args) == 1:
                call = call.args[0]  # unwrap dict(zip(...))
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'zip' \
                    and call.args and isinstance(call.args[0], ast.Name):
                classes = lists.get(call.args[0].id)
        if not classes:
            raise PlanError(f'cannot resolve ordered classes for QUESTIONS field {head}')
        questions[head] = {'criteria': dict.fromkeys(classes)}
    return questions


def default_business_schema(path=None, schema_version=None):
    """Legacy AST source remains authoritative; v2 is an explicit derivation."""
    from task_label_schema import version, questions_for
    selected=version(schema_version)
    return schema_from_questions(questions_for(selected,questions_from_source(path)),selected)

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


def _plain_int(value):
    # bool is an int subclass in Python; True must never pass an integer gate.
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value):
    # Rejects bools and non-finite floats (NaN/inf) alike: a NaN bound or a
    # True temperature silently poisons the calibration math downstream.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _merge_policy(override):
    policy = _merge(DEFAULT_SUPPORT_POLICY, override)
    counts = policy.get('min_groups_per_role')
    if not isinstance(counts, dict) or set(counts) != set(ROLES) \
            or not all(_plain_int(counts[r]) and counts[r] >= 1 for r in ROLES):
        raise PlanError('support policy min_groups_per_role must map A/B/C to positive integers')
    for key in ('min_rows_per_class', 'min_groups_per_class', 'min_fold_train_groups_per_class'):
        value = policy.get(key)
        if not _plain_int(value) or value < 1:
            raise PlanError(f'support policy {key} must be a positive integer')
    folds = policy.get('n_folds')
    if not _plain_int(folds) or folds < 2:
        raise PlanError('support policy n_folds must be an integer >= 2')
    return policy


def _merge_protocol(override):
    protocol = _merge(DEFAULT_PROTOCOL, override)
    grid = protocol.get('tau_grid')
    if not isinstance(grid, list) or not grid or not all(_finite_number(t) and t >= 0 for t in grid):
        raise PlanError('protocol tau_grid must be a non-empty list of finite numbers >= 0 '
                        '(bools, NaN, and infinity are rejected)')
    lo, hi = protocol.get('temperature_bounds', [0, 0])
    if not (_finite_number(lo) and _finite_number(hi) and 0 < lo < hi):
        raise PlanError('protocol temperature_bounds must be two finite numbers with 0 < low < high')
    if not _plain_int(protocol.get('fit_max_iterations')) or protocol['fit_max_iterations'] < 1:
        raise PlanError('protocol fit_max_iterations must be a positive integer')
    if not _finite_number(protocol.get('fit_tolerance')) or protocol['fit_tolerance'] <= 0:
        raise PlanError('protocol fit_tolerance must be a finite number > 0')
    epsilon = protocol.get('probability_epsilon')
    if not _finite_number(epsilon) or not (0 < epsilon < 1):
        raise PlanError('protocol probability_epsilon must be a finite number in (0, 1)')
    return protocol


def validate_declared_roots(rows):
    """Check supplied original roots, never infer them from prompts or names.

    Issue #2: split_group is a component that may join several original group
    roots, but one root cannot be fragmented into several components. Otherwise
    distinct A/B/C groups (or B folds) can still share the same source. Absent
    roots remain legacy-compatible; partial or malformed declarations fail
    closed. Even complete declarations are not independent-source evidence.
    """
    declared = sum('group' in row for row in rows)
    if not declared:
        return {'status': 'unavailable', 'independence': 'unverified'}
    missing = len(rows) - declared
    malformed = sum('group' in row and
                    (not isinstance(row['group'], str) or not row['group'].strip())
                    for row in rows)
    if missing or malformed:
        raise PlanError(f'original root declarations incomplete or malformed: '
                        f'missing_group={missing} invalid_group={malformed}; '
                        'supply non-empty group strings on every consumed row (values withheld)')
    components = collections.defaultdict(set)
    for row in rows:
        components[row['group']].add(row['split_group'])
    fragmented = sum(len(groups) > 1 for groups in components.values())
    if fragmented:
        raise PlanError(f'original root isolation failed: fragmented_roots={fragmented}; '
                        'one group must belong to only one split_group (values withheld)')
    return {'status': 'declared_consistent', 'independence': 'unverified'}


def validate_partition_isolation(train_rows, calibration_rows):
    """Shared consumed-input boundary; never opens held-out/test data.

    Check BEFORE filtering synthetic rows, deduplicating priors or sampling:
    otherwise a discarded row can hide a declared train/calibration collision.
    Repeated train IDs retain the existing prior-deduplication semantics.
    """
    for rows in (train_rows, calibration_rows):
        if not isinstance(rows, list) or not rows:
            raise PlanError('train and calibration must be non-empty row lists')
        invalid = sum(not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ('id', 'split_group')) for row in rows)
        if invalid:
            raise PlanError(f'partition identity validation failed: invalid_rows={invalid}; '
                            'id and split_group must be non-empty strings (values withheld)')
    overlaps = {key: len({r[key] for r in train_rows} & {r[key] for r in calibration_rows})
                for key in ('id', 'split_group')}
    if any(overlaps.values()):
        raise PlanError(f'train/calibration isolation failed: shared_id={overlaps["id"]} '
                        f'shared_split_group={overlaps["split_group"]} (values withheld)')
    return validate_declared_roots(train_rows + calibration_rows)


def validate_rows(rows, schema):
    """Every row needs a unique non-empty string id, a non-empty string
    split_group, and legal labels on every required head. Optional heads may
    lack or hold out-of-schema labels; such rows simply do not support them
    (support accounting reflects that; nothing is silently relabeled)."""
    if not isinstance(rows, list) or not rows:
        raise PlanError('rows must be a non-empty list')
    from task_label_schema import validate_supervised_rows
    try:validate_supervised_rows(rows,schema.get('task_label_schema_version'))
    except ValueError as error:raise PlanError(str(error)) from error
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
    validate_declared_roots(rows)
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
    from task_label_schema import validate_supervised_rows
    try:validate_supervised_rows(rows,schema.get('task_label_schema_version'),allow_supplemental=True)
    except ValueError as error:raise PlanError(str(error)) from error
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


def _is_sha256_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def checkpoint_digest(directory):
    """Streaming SHA-256 binding of the ACTUAL checkpoint weight files
    (works for both LoRA adapter/ and full-checkpoint/ products, weights are
    never loaded — bytes are hashed in 1 MiB chunks). Relative filenames are
    explicit so the binding names exactly which bytes were locked; the
    combined digest is order-independent (sorted names)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PlanError('checkpoint digest requires the checkpoint directory (adapter/ or full-checkpoint/)')
    file_digests = {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        file_digests[str(path.relative_to(directory)).replace('\\', '/')] = digest.hexdigest()
    if not file_digests:
        raise PlanError('checkpoint directory has no files to bind')
    combined = hashlib.sha256(''.join(f'{name}:{file_digests[name]}\n'
                                      for name in sorted(file_digests)).encode()).hexdigest()
    return {'algorithm': 'sha256', 'file_count': len(file_digests),
            'file_digests': file_digests, 'digest': combined}


def verify_checkpoint_ref(checkpoint_ref):
    """checkpoint_ref must be the JSON output of checkpoint_digest (or an
    equivalent sha256 file-digest binding): a binding to the checkpoint
    WEIGHT BYTES. Free text (a best-epoch.json snippet, an epoch id, any
    other unvalidated selection event) and self-inconsistent (tampered)
    digests are rejected — the calibration binding must be verifiable, not
    asserted."""
    try:
        binding = json.loads(checkpoint_ref)
    except ValueError:
        raise PlanError('checkpoint_ref must be a JSON checkpoint digest (checkpoint_digest output), '
                        'not free text or a selection-event label')
    if not isinstance(binding, dict) or binding.get('algorithm') != 'sha256' \
            or not isinstance(binding.get('file_digests'), dict) or not binding['file_digests'] \
            or not _is_sha256_hex(binding.get('digest')):
        raise PlanError('checkpoint_ref digest must carry algorithm=sha256, non-empty file_digests, and a digest')
    for name, digest in binding['file_digests'].items():
        if not isinstance(name, str) or not name or name.startswith(('/', '..')) or '\\' in name \
                or not _is_sha256_hex(digest):
            raise PlanError('checkpoint_ref file entries must be explicit relative filenames with sha256 hex digests')
    combined = hashlib.sha256(''.join(f'{name}:{binding["file_digests"][name]}\n'
                                      for name in sorted(binding['file_digests'])).encode()).hexdigest()
    if combined != binding['digest']:
        raise PlanError('checkpoint_ref digest does not match its file digests (tampered binding rejected)')


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
    folds, grid, budget, and order; checkpoint_ref must be a checkpoint_digest
    binding of the actual checkpoint weight bytes (verify_checkpoint_ref).
    The selected result additionally carries the server-consumable payload
    (temperatures plus tier_adjust = tau*log(prior) in the exact form
    model_utils.score subtracts at serve time), so training-side adjustment
    and serving-side adjustment cannot silently diverge. Tau selection is a
    development estimate; this procedure does not make the whole evaluation
    unbiased."""
    if not isinstance(plan, dict) or plan.get('mode') != 'protocol' or plan.get('diagnostic_only'):
        raise PlanError('out-of-fold selection requires a pre-registered protocol plan; '
                        'diagnostic halves cannot produce calibration products')
    if not isinstance(checkpoint_ref, str) or not checkpoint_ref.strip():
        raise PlanError('checkpoint must be locked by role A before tau selection (adjust-then-select order)')
    verify_checkpoint_ref(checkpoint_ref)
    proto = _merge_protocol(protocol or plan['protocol'])
    schema = plan['schema']
    heads = schema['heads']
    b_rows = plan['rows_by_role']['B']
    b_ids = [r['id'] for r in b_rows]

    if not isinstance(rows, list):
        raise PlanError('scored rows must be a list')
    from task_label_schema import validate_supervised_rows
    try:validate_supervised_rows(rows,schema.get('task_label_schema_version'))
    except ValueError as error:raise PlanError(str(error)) from error
    scored = {}
    for row in rows:
        rid = row.get('id') if isinstance(row, dict) else None
        if not isinstance(rid,str) or not rid:
            raise PlanError('scored rows require non-empty string ids (values withheld)')
        if rid in scored:
            raise PlanError('scored rows contain duplicate ids (values withheld)')
        scored[rid] = row
    missing = sum(1 for rid in b_ids if rid not in scored)
    extra = sum(1 for rid in scored if rid not in set(b_ids))
    if missing or extra:
        raise PlanError(f'scored rows do not match role B exactly: missing={missing} extra={extra}; '
                        'roles A/C and any test rows must never reach temperature/tau fitting')
    for planned in b_rows:
        fresh=scored[planned['id']]
        if fresh.get('split_group')!=planned['split_group'] or fresh.get('label')!=planned.get('label'):
            raise PlanError('scored role-B group/label differs from the locked plan (values withheld)')
        if 'group' in fresh and fresh['group']!=planned.get('group'):
            raise PlanError('scored role-B root differs from the locked plan (values withheld)')
    for row in rows:
        for head, spec in heads.items():
            values = (row.get('logits') or {}).get(head)
            if spec.get('calibration') != 'temperature' and values is None:
                continue  # fixed optional heads may omit logits, never mis-size supplied ones
            if not isinstance(values, list) or len(values) != len(spec['classes']) \
                    or not all(_finite_number(v) for v in values):
                raise PlanError(f'scored logits invalid for head {head} '
                                '(bools, NaN, and infinity are rejected; row id withheld)')

    for head, spec in heads.items():
        if spec.get('calibration') == 'temperature' and spec.get('adjust'):
            probs = (prior or {}).get(head)
            if not isinstance(probs, dict) or set(probs) != set(spec['classes']) \
                    or not all(_finite_number(p) and 0 < p <= 1 for p in probs.values()):
                raise PlanError(f'prior missing or malformed for adjusted head {head} '
                                '(probabilities must be finite numbers in (0, 1]; bools rejected)')

    # Red-team #18 F1 / Refs #2: the inherited selector validated fresh scores
    # but fitted raw/cached planning rows. Join ONLY after exact identity checks;
    # plan labels/groups stay locked, submitted logits alone are authoritative.
    b_rows=[{**row,'logits':scored[row['id']]['logits']} for row in b_rows]
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
    # Server-consumable adjustment (F1, train/serve parity): temperatures were
    # fitted on z - tau*log(prior), so serving must subtract the SAME
    # tau*log(prior) from raw logits before dividing by T. server.py loads
    # calibration['tier_adjust'] into model_utils.score's adjustments, whose
    # formula (values[index] - adjustment) is exactly this value.
    adjustments = {head: {c: chosen * math.log(prior[head][c]) for c in heads[head]['classes']}
                   for head in temperature_heads if heads[head].get('adjust')}
    return {'status': 'selected', 'selected_tau': chosen, 'temperatures': temperatures,
            'tier_adjust': adjustments.get('tier'), 'tier_prior': (prior or {}).get('tier'),
            'adjustments': adjustments,
            'calibrated': calibrated, 'candidates': candidates, 'binding': binding,
            'note': 'development estimate over role B folds; group-equal weighting; not an unbiased evaluation'}


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
    from task_label_schema import version
    return {'mode': plan['mode'], 'task_label_schema_version':version(plan['schema'].get('task_label_schema_version')), 'roles': {r: {'groups': len(roles[r]), 'rows': len(rows_by_role[r])} for r in ROLES},
            'folds': folds, 'support': plan['support'],
            'optional_head_status': plan['optional_head_status'],
            'plan_hash': plan['plan_hash'], 'protocol_hash': plan['protocol_hash']}


ROLES_CONFIG_KEYS = ('role_groups', 'support_policy', 'protocol')


def check_roles_config(roles_config):
    """Shared roles-config contract for EVERY entry point (train_eval.run,
    scripts/train.py dry-run, scripts/validate_calibration.py). The config may
    pre-register groups and numeric policy/protocol values only: it must NOT
    override the business schema — heads/classes/required flags come from
    model_utils.QUESTIONS through schema_from_questions, so a config can never
    downgrade vision to optional or otherwise let a preflight pass that the
    actual run would reject (or worse, accept with different meaning)."""
    if not isinstance(roles_config, dict) or 'role_groups' not in roles_config:
        raise PlanError("roles config must be an object with a 'role_groups' key")
    if 'schema' in roles_config:
        raise PlanError('roles config must not override the business schema: heads, classes, and '
                        'required flags are fixed by model_utils.QUESTIONS (vision cannot be '
                        'downgraded to optional; preflight and run share one schema source)')
    unknown = sorted(set(roles_config) - set(ROLES_CONFIG_KEYS))
    if unknown:
        raise PlanError(f"roles config has unknown key(s) {unknown}; allowed keys are "
                        f"{list(ROLES_CONFIG_KEYS)} — a key the run would ignore cannot pass preflight")


def preflight_roles(calibration_rows, roles_config, schema=None, schema_version=None):
    """Shared preflight used by every entry point: builds and validates a
    protocol plan from a roles config object ({'role_groups': {...},
    'support_policy': {...}?, 'protocol': {...}?}). The schema is NOT
    configurable here: it always comes from the business schema source
    (default_business_schema, parsed from the same model_utils.py the run
    imports), so preflight and run cannot validate different truths."""
    check_roles_config(roles_config)
    return build_plan(rows=calibration_rows, role_groups=roles_config['role_groups'],
                      schema=schema or default_business_schema(schema_version=schema_version),
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
