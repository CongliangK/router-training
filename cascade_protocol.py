"""Preregisterable root-unit cascade and synthetic-only CPU harness, Refs #26.

Not an outcomes importer. No API calls, model I/O, measured labels or gate
updates. Even favorable synthetic bounds remain shadow, never qualified.
"""
from __future__ import annotations
import collections
import math
import random
from oof_protocol import ProtocolError, digest, _hash, _number, _positive, _text

TIERS = ('fast', 'balanced', 'flagship')
RANDOMNESS = 'domain_separated_root_allocation_task_and_candidate_stage_v1'


def _rng(seed, *domain):
    # Stable independent pseudo-random streams: a screening branch must never
    # consume confirmation randomness, nor allocation change a root's tasks.
    # This is reproducible simulation, NOT proof of real execution independence.
    return random.Random(int(digest([seed, *domain]), 16))


def _frame(rows):
    if not isinstance(rows, list) or not rows:
        raise ProtocolError('nonempty public/synthetic development sampling metadata required')
    ids, splits, duplicates = set(), {}, {}
    for row in rows:
        if not isinstance(row, dict) or any(not _text(row.get(k)) for k in
                ('id', 'root_id', 'split_group', 'duplicate_cluster')):
            raise ProtocolError('audited root/split/near-duplicate identities required')
        if row.get('source_split') not in ('train', 'calibration', 'synthetic'):
            raise ProtocolError('prototype never accepts test-source sampling metadata')
        if row.get('tier') not in TIERS or row['id'] in ids:
            raise ProtocolError('invalid tier or duplicate sampling task')
        ids.add(row['id'])
        for table, key in ((splits, 'split_group'), (duplicates, 'duplicate_cluster')):
            if table.setdefault(row[key], row['root_id']) != row['root_id']:
                raise ProtocolError('root audit failed: split/near-duplicate cluster spans roots')


def sample_anchors(frame, *, screening_roots, confirmation_roots, tasks_per_root, seed):
    """Uniform roots without replacement, then uniform tasks within each root.

    Labels do not drive sampling. The historical 45 / 15:20:10 proposal is NOT
    asserted feasible. Root counts, not split-group IDs, determine support.
    All selections precede outcomes; reserved confirmation roots never screen.
    """
    _frame(frame)
    if any(not _positive(x) for x in (screening_roots, confirmation_roots, tasks_per_root)) \
            or tasks_per_root > 2 or type(seed) is not int:
        raise ProtocolError('positive root counts, 1-2 tasks per root and integer seed required')
    groups = collections.defaultdict(list)
    for row in sorted(frame, key=lambda r: r['id']):
        groups[row['root_id']].append(row)
    if len(groups) < screening_roots + confirmation_roots:
        raise ProtocolError('insufficient audited roots; never split/duplicate to manufacture support')
    roots = sorted(groups)
    _rng(seed, 'root_allocation').shuffle(roots)
    # Task schedules can be defined for EVERY frame root before allocation;
    # changing the split or selected-root count cannot change a root's tasks.
    tasks_by_root = {}
    for root in sorted(groups):
        tasks = list(groups[root])
        _rng(seed, 'tasks', root).shuffle(tasks)
        tasks_by_root[root] = tasks[:tasks_per_root]
    # Unequal root sizes are legal. Never duplicate a task to reach two.
    selected = [tasks_by_root[root] for root in roots[:screening_roots + confirmation_roots]]
    screen = [r for tasks in selected[:screening_roots] for r in tasks]
    confirm = [r for tasks in selected[screening_roots:] for r in tasks]
    return {'screening': screen, 'confirmation': confirm,
            'sampling': {'design': 'uniform_roots_without_replacement_then_uniform_tasks',
                         'frame_digest': digest(sorted(frame, key=lambda r: r['id'])),
                         'frame_root_count': len(groups), 'seed': seed,
                         'randomness_policy': RANDOMNESS,
                         'tasks_per_root': tasks_per_root,
                         'screening_root_count': screening_roots,
                         'confirmation_root_count': confirmation_roots},
            'observed_anchor_tiers': dict(collections.Counter(r['tier'] for r in screen + confirm))}


def preregister(selection, candidates, *, protocol_id, root_audit_ref, acceptance_ref,
                confirmation_schedule_ref, explore_trials, confirmation_trials, alpha, reliability_target,
                approval_ref=None):
    """Freeze identities, root allocation, stages, multiplicity, stopping and caps.

    Estimand: equally weighted finite-frame ROOT mean of the predeclared task/
    trial acceptance protocol, not each task's long-run reliability. Candidate
    identities include provider/model/revision/effort and acceptance semantics.
    Confirmation evaluates ALL registered candidates, never selected repeats.
    """
    if not isinstance(selection, dict) or not isinstance(selection.get('sampling'), dict):
        raise ProtocolError('pre-outcome sampling selection required')
    screen, confirm = selection.get('screening'), selection.get('confirmation')
    _frame(screen)
    _frame(confirm)
    if {r['root_id'] for r in screen}.intersection(r['root_id'] for r in confirm) \
            or {r['id'] for r in screen}.intersection(r['id'] for r in confirm):
        raise ProtocolError('screening and independent confirmation roots/tasks overlap')
    _frame(screen + confirm)  # Also catches cross-region near-duplicate aliases.
    sampling = selection['sampling']
    if sampling.get('design') != 'uniform_roots_without_replacement_then_uniform_tasks' \
            or sampling.get('randomness_policy') != RANDOMNESS \
            or not _hash(sampling.get('frame_digest')) or type(sampling.get('seed')) is not int \
            or any(not _positive(sampling.get(k)) for k in
                   ('frame_root_count', 'tasks_per_root', 'screening_root_count', 'confirmation_root_count')) \
            or sampling['tasks_per_root'] > 2:
        raise ProtocolError('sampling frame/design identity invalid')
    for rows, field in ((screen, 'screening_root_count'), (confirm, 'confirmation_root_count')):
        counts = collections.Counter(r['root_id'] for r in rows)
        if len(counts) != sampling[field] or max(counts.values()) > sampling['tasks_per_root']:
            raise ProtocolError('selected anchors do not match preregistered root/task counts')
    if sampling['frame_root_count'] < sampling['screening_root_count'] + sampling['confirmation_root_count']:
        raise ProtocolError('sampling counts exceed audited root frame')
    if not all(_text(x) for x in (protocol_id, root_audit_ref, acceptance_ref, confirmation_schedule_ref)):
        raise ProtocolError('protocol/root audit/acceptance/precommitted confirmation schedule references required')
    if not isinstance(candidates, list) or not candidates or len(candidates) > len(TIERS):
        raise ProtocolError('nonempty unique strictly upgrading candidate chain required')
    previous, keys = -1, set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or any(not _text(candidate.get(k)) for k in
                ('id', 'provider', 'model', 'revision', 'effort', 'tier')) \
                or not _hash(candidate.get('acceptance_digest')) \
                or not _hash(candidate.get('context_policy_digest')):
            raise ProtocolError('complete frozen candidate operating-point identity required')
        if candidate['tier'] not in TIERS or TIERS.index(candidate['tier']) <= previous \
                or candidate['id'] in keys:
            raise ProtocolError('candidate chain must strictly upgrade tiers with unique identities')
        previous = TIERS.index(candidate['tier'])
        keys.add(candidate['id'])
    if type(explore_trials) is not int or explore_trials < 0 or not _positive(confirmation_trials):
        raise ProtocolError('exploration >=0 and fixed confirmation trial count >0 required')
    if not _number(alpha) or not 0 < alpha < 1 or not _number(reliability_target) \
            or not 0 < reliability_target < 1:
        raise ProtocolError('finite alpha and reliability target in (0,1) required')
    caps = {c['id']: {'screening': 3 * len(screen), 'exploration': explore_trials * len(screen),
                      'confirmation': confirmation_trials * len(confirm)} for c in candidates}
    plan = {'protocol_id': protocol_id, 'root_audit_ref': root_audit_ref,
            'acceptance_ref': acceptance_ref, 'approval_ref': approval_ref,
            'confirmation_randomness': {'policy': RANDOMNESS,
                                        'schedule_ref': confirmation_schedule_ref},
            'selection': selection, 'candidates': candidates,
            'assumptions_verified': False, 'approval_verified': False,
            'estimand': 'finite_frame_equal_root_mean_predeclared_acceptance_score',
            'weights': 'equal_roots_then_equal_sampled_tasks_then_equal_fixed_trials',
            'stages': {'screening': 3, 'exploration': explore_trials, 'confirmation': confirmation_trials},
            'stopping': {'at_least_two_screen_failures': 'eliminate_here_upgrade_next',
                         'three_screen_passes': 'exploratory_supplement_then_stop_chain',
                         'two_screen_passes': 'ambiguous_stop_shadow_no_expansion',
                         'confirmation': 'all_registered_candidates_fixed_n_no_peeking_no_early_success',
                         'exhausted_chain': 'shadow_no_candidate'},
            'multiplicity': {'method': 'bonferroni_all_preregistered_candidates',
                             'family_size': len(candidates), 'alpha': alpha},
            'reliability_target': reliability_target, 'per_candidate_caps': caps,
            # Worst chain: all candidates screened and only the final candidate
            # supplemented. Summing all exploration caps is a looser bound.
            'global_max_executions': len(screen) * (3 * len(candidates) + explore_trials)
                                     + len(confirm) * confirmation_trials * len(candidates),
            'qualified': False, 'status': 'shadow'}
    # Freeze a JSON copy: subsequent caller mutation cannot alter the plan.
    import json
    plan = json.loads(json.dumps(plan, allow_nan=False))
    plan['plan_digest'] = digest(plan)
    return plan


def _bound_plan(plan):
    if not isinstance(plan, dict) or plan.get('plan_digest') != digest(
            {k: v for k, v in plan.items() if k != 'plan_digest'}):
        raise ProtocolError('preregistered protocol binding missing or changed')


def confirmation_summary(plan, records, *, independent_root_evidence=False):
    """Synthetic fixed-n root-score Hoeffding diagnostic, NOT a quality gate.

    Conditional coverage requires audited whole-root sampling, bounded root
    potential scores/independent root execution randomness and no shared shocks,
    fixed candidates/acceptance/tasks/trials and no outcome-dependent expansion.
    The conservative finite-population SRS-without-replacement Hoeffding bound
    applies to fixed root scores; no IID Bernoulli/CP model is used. If these
    design assumptions cannot be defended, only empirical root means remain.
    """
    _bound_plan(plan)
    if type(independent_root_evidence) is not bool or not isinstance(records, list):
        raise ProtocolError('synthetic records and explicit root-evidence flag required')
    anchors = {r['id']: r for r in plan['selection']['confirmation']}
    candidates = {c['id'] for c in plan['candidates']}
    count = plan['stages']['confirmation']
    seen, values = set(), collections.defaultdict(list)
    for record in records:
        if not isinstance(record, dict) or record.get('synthetic') is not True \
                or type(record.get('passed')) is not bool:
            raise ProtocolError('harness accepts explicitly synthetic boolean verdicts only')
        task, candidate, trial = record.get('task_id'), record.get('candidate_id'), record.get('trial')
        if not _text(task) or not _text(candidate) or task not in anchors or candidate not in candidates \
                or type(trial) is not int \
                or not 0 <= trial < count or record.get('stage') != 'confirmation' \
                or record.get('root_id') != anchors[task]['root_id']:
            raise ProtocolError('confirmation record violates fixed independent allocation')
        key = task, candidate, trial
        if key in seen:
            raise ProtocolError('duplicate confirmation trial')
        seen.add(key)
        values[candidate, record['root_id']].append(int(record['passed']))
    if len(seen) != len(anchors) * len(candidates) * count:
        raise ProtocolError('incomplete confirmation; no selective early stopping permitted')
    summaries = []
    roots = {r['root_id'] for r in anchors.values()}
    alpha = plan['multiplicity']['alpha']
    family = plan['multiplicity']['family_size']
    for candidate in sorted(candidates):
        scores = [sum(values[candidate, root]) / len(values[candidate, root]) for root in sorted(roots)]
        mean = sum(scores) / len(scores)
        # Within-root repeats NEVER increase n_root. Equal fixed trial counts
        # per task imply equal sampled-task weighting WITHIN each root, even
        # when different roots contain different numbers of sampled tasks.
        lower = max(0.0, mean - math.sqrt(math.log(family / alpha) / (2 * len(roots)))) \
            if independent_root_evidence else None
        summaries.append({'candidate_id': candidate, 'n_root': len(roots), 'n_execution': len(anchors) * count,
                          'root_mean': mean, 'diagnostic_lower_bound': lower,
                          'diagnostic_accept': lower is not None and lower >= plan['reliability_target']})
    return {'status': 'shadow', 'qualified': False, 'synthetic': True, 'coverage_claim': False,
            'measured_labels': [], 'gate_updates': [], 'candidate_summaries': summaries,
            'bound_method': 'bounded_root_hoeffding_bonferroni_diagnostic_not_iid_cp',
            'warning': 'population/root reliability cannot become a per-task measured label'}


def simulate(plan, probabilities, *, seed, within_root_shared=True):
    """Generate CPU synthetic cascades only; no executor/callback/network hook.

    Shared latent uniform per root/candidate/stage stresses arbitrarily strong
    within-root correlation. Confirmation randomness is independent of screen.
    `probabilities` maps every root to every candidate's synthetic probability.
    """
    _bound_plan(plan)
    if type(seed) is not int or type(within_root_shared) is not bool or not isinstance(probabilities, dict):
        raise ProtocolError('explicit synthetic probability table, seed and correlation mode required')
    roots = {r['root_id'] for stage in ('screening', 'confirmation') for r in plan['selection'][stage]}
    candidates = [c['id'] for c in plan['candidates']]
    if set(probabilities) != roots or any(not isinstance(probabilities[r], dict) or
            set(probabilities[r]) != set(candidates) or any(not _number(p) or not 0 <= p <= 1
                for p in probabilities[r].values()) for r in roots):
        raise ProtocolError('complete finite [0,1] root/candidate probability table required')
    records, decisions = [], []
    def trial(anchor, candidate, stage, index):
        # Counter-addressable potential outcomes: no execution-order or
        # screening-outcome dependence in confirmation. In shared mode all
        # tasks/trials inside a root/candidate/stage use ONE latent uniform.
        domain = ('execution', anchor['root_id'], candidate, stage)
        if not within_root_shared:
            domain += (anchor['id'], index)
        uniform = _rng(seed, *domain).random()
        passed = uniform < probabilities[anchor['root_id']][candidate]
        records.append({'synthetic': True, 'stage': stage, 'task_id': anchor['id'],
                        'root_id': anchor['root_id'], 'candidate_id': candidate,
                        'trial': index, 'passed': passed})
        return passed
    for anchor in plan['selection']['screening']:
        decision = 'shadow_no_candidate'
        for candidate in candidates:
            passed = sum(trial(anchor, candidate, 'screening', i) for i in range(3))
            if passed <= 1:
                continue
            if passed == 3:
                for i in range(plan['stages']['exploration']):
                    trial(anchor, candidate, 'exploration', i)
                decision = 'exploratory_only'
            else:
                decision = 'ambiguous_shadow'
            break
        decisions.append(decision)
    # Confirmation is unconditional and fixed for ALL registered candidates.
    for anchor in plan['selection']['confirmation']:
        for candidate in candidates:
            for i in range(plan['stages']['confirmation']):
                trial(anchor, candidate, 'confirmation', i)
    counts = {candidate: {stage: sum(r['candidate_id'] == candidate and r['stage'] == stage for r in records)
                          for stage in ('screening', 'exploration', 'confirmation')} for candidate in candidates}
    if len(records) > plan['global_max_executions'] or any(
            counts[c][s] > plan['per_candidate_caps'][c][s] for c in candidates for s in counts[c]):
        raise ProtocolError('synthetic execution cap exceeded')
    summary = confirmation_summary(plan, [r for r in records if r['stage'] == 'confirmation'],
                                   independent_root_evidence=True)
    summary.update(execution_count=len(records), stage_counts=counts,
                   screening_decisions=dict(collections.Counter(decisions)),
                   budget_cap=plan['global_max_executions'], plan_digest=plan['plan_digest'])
    return summary


def simulation_stress(plan, probabilities, *, repetitions, seed):
    """Monte Carlo code check, not validation of real coverage or model quality."""
    _bound_plan(plan)
    if not _positive(repetitions) or type(seed) is not int:
        raise ProtocolError('positive fixed simulation count and integer seed required')
    counts = collections.Counter()
    roots = {r['root_id'] for r in plan['selection']['confirmation']}
    for i in range(repetitions):
        result = simulate(plan, probabilities, seed=seed + i, within_root_shared=True)
        for candidate in result['candidate_summaries']:
            cid = candidate['candidate_id']
            # A deliberately stated simulation reference over the sampled roots;
            # not an estimator/claim about unobserved real frame reliability.
            reference = sum(probabilities[r][cid] for r in roots) / len(roots)
            if candidate['diagnostic_accept'] and reference < plan['reliability_target']:
                counts[cid] += 1
    return {'synthetic': True, 'status': 'shadow', 'qualified': False, 'repetitions': repetitions,
            'diagnostic_false_accept_counts': {c['id']: counts[c['id']] for c in plan['candidates']},
            'coverage_claim': False, 'measured_labels': []}
