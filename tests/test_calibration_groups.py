"""CPU-only tests for group-isolated calibration (issue #2).

All fixtures are invented synthetic distributions (arbitrary group sizes and
opaque ids); they deliberately do NOT mirror the archived dataset's group
sizes, ids, or hashes, and no real prompts or labels appear anywhere.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import calibration_groups as cg

TIERS = ['fast', 'balanced', 'flagship']
SEVEN = ['extraction', 'research', 'implementation', 'debugging', 'review', 'visual', 'creative']
SCHEMA = cg.DEFAULT_SCHEMA


def make_pool(n_groups=16, seed=11):
    """Invented synthetic pool: varied group sizes (5-7 rows), tier/vision/
    task_type cycling. Duplicate split_group values are the point (issue #2):
    many rows share one recorded group."""
    rows = []
    for g in range(n_groups):
        group = f'dev-source-{g:02d}'
        for i in range(5 + g % 3):
            n = g * 10 + i
            rows.append({'id': f'fixture-row-{n:04d}', 'split_group': group,
                         'prompt': f'synthetic calibration fixture prompt {n} {seed}',
                         'synthetic': True,
                         'label': {'tier': TIERS[n % 3], 'needs_vision': ['no', 'yes'][n % 2],
                                   'task_type': SEVEN[n % 7]}})
    return rows


def make_roles(rows, a=5, b=6):
    groups = sorted({r['split_group'] for r in rows})
    return {'role_groups': {'A': groups[:a], 'B': groups[a:a + b], 'C': groups[a + b:]}}


def make_train_rows(n=40):
    rows = []
    for g in range(n):
        rows.append({'id': f'fixture-train-{g:04d}', 'split_group': f'dev-train-{g:08d}',
                     'label': {'tier': TIERS[g % 3], 'needs_vision': ['no', 'yes'][g % 2],
                               'task_type': SEVEN[g % 7]}})
    return rows


def add_logits(rows):
    for r in rows:
        ci = TIERS.index(r['label']['tier'])
        r['logits'] = {'tier': [0.2 * i + (1.1 if i == ci else 0.0) for i in range(3)]}
        vi = 0 if r['label']['needs_vision'] == 'no' else 1
        r['logits']['vision'] = [0.3 * (1 - vi), 0.9 * vi]
    return rows


def build_happy_plan():
    rows = add_logits(make_pool())
    roles = make_roles(rows)
    plan = cg.build_plan(rows, roles['role_groups'], SCHEMA)
    return plan, rows, roles


class Recorder:
    def __init__(self):
        self.fit_calls = []
        self.score_calls = []
        self.selected = None

    def protocol(self):
        def fit(logits, cls, groups, bounds, iters, tol, eps):
            self.fit_calls.append({'groups': tuple(groups), 'logits': [list(x) for x in logits], 'width': len(logits[0])})
            return cg.fit_temperature(logits, cls, groups, bounds, iters, tol, eps)

        def score(logits, cls, groups, temperature, eps):
            self.score_calls.append({'groups': tuple(groups), 'logits': [list(x) for x in logits], 'width': len(logits[0])})
            return cg.score_holdout(logits, cls, groups, temperature, eps)

        def selector(candidates):
            chosen = cg.default_selector(candidates)
            self.selected = chosen
            return chosen

        return {'fit': fit, 'score': score, 'selector': selector}


class RowValidationTests(unittest.TestCase):
    def test_missing_id_group_empty_group_and_cross_group_duplicate_are_anonymous_errors(self):
        base = make_pool(n_groups=8)
        cases = {
            'missing id': lambda rows: rows[0].__delitem__('id'),
            'missing group': lambda rows: rows[0].__delitem__('split_group'),
            'empty group string': lambda rows: rows[0].__setitem__('split_group', ''),
            'duplicate id across groups': lambda rows: rows[6].__setitem__('id', rows[0]['id']),
            'illegal required label': lambda rows: rows[0]['label'].__setitem__('tier', 'ultra'),
        }
        for name, mutate in cases.items():
            rows = make_pool(n_groups=8)
            mutate(rows)
            with self.subTest(case=name):
                with self.assertRaises(cg.PlanError) as ctx:
                    cg.validate_rows(rows, SCHEMA)
                message = str(ctx.exception)
                self.assertIn('row validation failed', message)
                # Anonymity: original values never appear in the error.
                self.assertNotIn('fixture-row-0000', message)
                self.assertNotIn('dev-source-', message)
                self.assertNotIn('ultra', message)

    def test_plan_rejects_uncovered_rows_and_unknown_role_groups(self):
        rows = make_pool(n_groups=16)
        roles = make_roles(rows)
        roles['role_groups']['C'] = roles['role_groups']['C'][:-1]  # leave one group unregistered
        with self.assertRaises(cg.PlanError) as ctx:
            cg.build_plan(rows, roles['role_groups'], SCHEMA)
        self.assertIn('does not cover the pool', str(ctx.exception))
        self.assertNotIn('dev-source-', str(ctx.exception))
        roles['role_groups']['C'].append('dev-source-99')  # registered group absent from pool
        with self.assertRaises(cg.PlanError) as ctx:
            cg.build_plan(rows, roles['role_groups'], SCHEMA)
        self.assertIn('with no rows in the pool', str(ctx.exception))


class PlanIsolationTests(unittest.TestCase):
    def test_selection_fitting_groups_and_ids_disjoint_with_duplicate_split_groups(self):
        # Issue #2 acceptance: calibration data with duplicate split_group must
        # yield selection (A) / fitting (B) halves with NO group overlap.
        plan, rows, roles = build_happy_plan()
        self.assertTrue(any(len([r for r in rows if r['split_group'] == g]) > 1
                            for g in roles['role_groups']['A']))  # duplicate split_group present
        selection_groups = set(plan['roles']['A'])
        fitting_groups = set(plan['roles']['B'])
        self.assertFalse(selection_groups & fitting_groups)
        selection_ids = {r['id'] for r in plan['rows_by_role']['A']}
        fitting_ids = {r['id'] for r in plan['rows_by_role']['B']}
        self.assertFalse(selection_ids & fitting_ids)
        # Full coverage: every calibration row belongs to exactly one role.
        covered = [r['id'] for role in 'ABC' for r in plan['rows_by_role'][role]]
        self.assertEqual(sorted(covered), sorted({r['id'] for r in rows}))
        self.assertFalse(set(plan['roles']['A']) & set(plan['roles']['C']))
        self.assertFalse(set(plan['roles']['B']) & set(plan['roles']['C']))

    def test_strict_mode_rejects_missing_roles_overlap_and_has_no_seed_fallback(self):
        rows = make_pool(n_groups=16)
        roles = make_roles(rows)
        for broken in ({'A': roles['role_groups']['A'], 'B': roles['role_groups']['B']},  # C missing
                       {'A': roles['role_groups']['A'], 'B': roles['role_groups']['B'], 'C': []},  # C empty
                       {'A': [], 'B': roles['role_groups']['B'], 'C': roles['role_groups']['C']}):  # A empty
            with self.subTest(role_keys=sorted(broken)):
                with self.assertRaises(cg.PlanError) as ctx:
                    cg.build_plan(rows, broken, SCHEMA)
                # Either the pre-registration shape check or the per-role
                # hard-fail fires; neither silently falls back to a re-split.
                self.assertRegex(str(ctx.exception), r'pre-register|must list at least one group')
        overlap = {'role_groups': {'A': roles['role_groups']['A'] + [roles['role_groups']['B'][0]],
                                   'B': roles['role_groups']['B'],
                                   'C': roles['role_groups']['C']}}
        with self.assertRaises(cg.PlanError) as ctx:
            cg.build_plan(rows, overlap['role_groups'], SCHEMA)
        self.assertIn('overlap', str(ctx.exception))
        # No seed parameter exists: plans come from pre-registration, never a
        # seed-based re-split, and failures never fall back to one.
        self.assertNotIn('seed', _signature_params(cg.build_plan))

    def test_plan_is_deterministic_under_input_row_order_shuffle(self):
        plan, rows, roles = build_happy_plan()
        shuffled = rows[:]
        random.Random(99).shuffle(shuffled)
        plan2 = cg.build_plan(shuffled, roles['role_groups'], SCHEMA)
        self.assertEqual(plan['plan_hash'], plan2['plan_hash'])
        self.assertEqual(plan['folds'], plan2['folds'])
        self.assertEqual([r['id'] for r in plan['rows_by_role']['B']],
                         [r['id'] for r in plan2['rows_by_role']['B']])
        # Diagnostic halves are likewise order-invariant for a fixed seed.
        d1 = cg.diagnostic_halves(rows, SCHEMA, seed='7')
        d2 = cg.diagnostic_halves(shuffled, SCHEMA, seed='7')
        self.assertEqual(d1['selection_groups'], d2['selection_groups'])
        self.assertEqual(d1['fitting_groups'], d2['fitting_groups'])


def _signature_params(function):
    import inspect
    return inspect.signature(function).parameters


class SupportPolicyTests(unittest.TestCase):
    def test_role_group_minimum_enforced(self):
        rows = make_pool(n_groups=15)  # C would get 4 groups
        roles = make_roles(rows, a=5, b=6)
        with self.assertRaises(cg.PlanError) as ctx:
            cg.build_plan(rows, roles['role_groups'], SCHEMA)
        self.assertIn('role group minima', str(ctx.exception))

    def test_rows_per_class_minimum_enforced(self):
        rows = make_pool(n_groups=16)
        # Starve 'fast' inside role A to 4 rows (spread over >=2 groups so the
        # group minimum is NOT the trigger): the row minimum must fail.
        a_groups = set(sorted({r['split_group'] for r in rows})[:5])
        keep = [r for r in rows if r['split_group'] in a_groups and r['label']['tier'] == 'fast'][:4]
        for r in rows:
            if r['split_group'] in a_groups and r['label']['tier'] == 'fast' and r not in keep:
                r['label']['tier'] = 'balanced'
        fast_rows_a = [r for r in rows if r['split_group'] in a_groups and r['label']['tier'] == 'fast']
        self.assertEqual(len(fast_rows_a), 4)
        self.assertGreaterEqual(len({r['split_group'] for r in fast_rows_a}), 2)
        roles = make_roles(rows)
        with self.assertRaises(cg.PlanError) as ctx:
            cg.build_plan(rows, roles['role_groups'], SCHEMA)
        self.assertIn('classes_below_rows_minimum=1', str(ctx.exception))

    def test_groups_per_class_minimum_enforced(self):
        rows = make_pool(n_groups=16)
        # Confine vision='yes' to a single group of role A, then duplicate its
        # rows: row counts grow but INDEPENDENT group support stays 1.
        target = 'dev-source-00'
        for r in rows:
            if r['split_group'] != target and r['split_group'] < 'dev-source-05':
                r['label']['needs_vision'] = 'no'
        yes_rows = [r for r in rows if r['split_group'] == target and r['label']['needs_vision'] == 'yes']
        clones = []
        for i, r in enumerate(yes_rows):
            for k in range(3):
                clone = json.loads(json.dumps(r))
                clone['id'] = f'fixture-clone-{i}-{k}'
                clones.append(clone)
        rows.extend(clones)
        roles = make_roles(rows)
        with self.assertRaises(cg.PlanError) as ctx:
            cg.build_plan(rows, roles['role_groups'], SCHEMA)
        message = str(ctx.exception)
        self.assertIn('classes_below_groups_minimum=1', message)
        self.assertIn('classes_below_rows_minimum=0', message)  # duplication inflated rows, not support

    def test_fold_training_group_minimum_enforced(self):
        rows = make_pool(n_groups=16)
        roles = make_roles(rows)
        b_groups = roles['role_groups']['B']
        # 'fast' confined to exactly 2 B groups (region gates still pass:
        # >=5 rows, >=2 groups) so no fold training union can reach 2 groups.
        keep = set(b_groups[:2])
        for r in rows:
            if r['split_group'] in set(b_groups) and r['split_group'] not in keep:
                r['label']['tier'] = 'balanced' if r['label']['tier'] == 'fast' else r['label']['tier']
        for r in rows:
            if r['split_group'] in keep and r['label']['tier'] != 'fast':
                r['label']['tier'] = 'fast'
        with self.assertRaises(cg.PlanError) as ctx:
            cg.build_plan(rows, roles['role_groups'], SCHEMA)
        self.assertIn('fold_training_classes_below_minimum', str(ctx.exception))

    def test_smoke_subpool_keeps_whole_groups_and_never_relaxes_gates(self):
        rows = make_pool(n_groups=20)
        roles = make_roles(rows, a=6, b=8)
        relaxed_policy = {'min_groups_per_role': {'A': 3, 'B': 6, 'C': 3}, 'min_rows_per_class': 3,
                          'min_groups_per_class': 2, 'min_fold_train_groups_per_class': 2, 'n_folds': 3}
        plan = cg.smoke_subpool(rows, roles['role_groups'], SCHEMA, support_policy=relaxed_policy)
        kept = set(plan['roles']['A']) | set(plan['roles']['B']) | set(plan['roles']['C'])
        self.assertLess(len(kept), len({r['split_group'] for r in rows}))
        self.assertEqual(sorted(kept), sorted({r['split_group'] for r in rows
                                               if r['split_group'] in kept}))
        impossible = {'min_groups_per_role': {'A': 9, 'B': 9, 'C': 9}}
        with self.assertRaises(cg.PlanError) as ctx:
            cg.smoke_subpool(rows, roles['role_groups'], SCHEMA, support_policy=impossible)
        self.assertIn('refusing to relax', str(ctx.exception))


class FoldTests(unittest.TestCase):
    def test_group_folds_deterministic_whole_groups(self):
        counts = {'a': 10, 'b': 9, 'c': 8, 'd': 7, 'e': 6, 'f': 5}
        folds = cg.group_folds(counts, 3)
        self.assertEqual([sorted(f) for f in folds], [['a', 'f'], ['b', 'e'], ['c', 'd']])
        self.assertEqual([sum(counts[g] for g in f) for f in folds], [15, 15, 15])
        self.assertEqual(folds, cg.group_folds(counts, 3))

    def test_every_b_row_is_holdout_exactly_once_and_fit_holdout_mutually_exclusive(self):
        plan, rows, roles = build_happy_plan()
        rec = Recorder()
        prior = cg.compute_prior(make_train_rows(), 'tier', SCHEMA)
        b_rows = [r for r in rows if r['split_group'] in set(plan['roles']['B'])]
        result = cg.select_tau_oof(plan, b_rows, {'tier': prior['probabilities']},
                                   checkpoint_ref='ckpt-1', protocol=rec.protocol())
        self.assertEqual(result['status'], 'selected')
        b_groups = set(plan['roles']['B'])
        all_ids = {r['id'] for r in b_rows}
        taus = [c['tau'] for c in result['candidates']]
        per_tau_scores = len(taus) * len(plan['folds']) * 2  # tier+vision scored per fold per tau
        self.assertEqual(len(rec.score_calls), per_tau_scores)
        for start in range(0, len(rec.score_calls), 6):  # 3 folds x 2 heads per tau
            window = rec.score_calls[start:start + 6]
            group_sets = {frozenset(call['groups']) for call in window}
            self.assertEqual(len(group_sets), 3)  # tier+vision share each fold's groups
            union = set().union(*group_sets)
            self.assertEqual(union, b_groups)  # folds partition B exactly once per tau
            for a in group_sets:
                for b in group_sets - {a}:
                    self.assertFalse(a & b)
        scored_ids = {r['id'] for r in b_rows if r['split_group'] in
                      {g for call in rec.score_calls[:6] for g in call['groups']}}
        self.assertEqual(scored_ids, all_ids)  # every B row is a holdout exactly once
        for call in rec.fit_calls + rec.score_calls:
            self.assertTrue(set(call['groups']) <= b_groups)  # never A/C/other rows
            self.assertNotIn('dev-source-00', call['groups'])  # an A group sentinel
        # Fit and holdout are exact complements within each tau window: for
        # every holdout fold there is a fit call on precisely the remaining
        # B groups (so a fold's holdout rows are never fitted for that fold;
        # cross-fold overlap is expected and correct under GroupKFold). The
        # final two fit calls (full-B refit) are outside these windows.
        for start in range(0, len(rec.fit_calls) - 2, 6):
            fit_sets = {frozenset(c['groups']) for c in rec.fit_calls[start:start + 6]}
            holdout_sets = {frozenset(c['groups']) for c in rec.score_calls[start:start + 6]}
            self.assertEqual(len(fit_sets), 3)
            self.assertEqual(len(holdout_sets), 3)
            for holdout_set in holdout_sets:
                complement = b_groups - holdout_set
                self.assertIn(complement, fit_sets)
            self.assertEqual(fit_sets, {frozenset(b_groups - h) for h in holdout_sets})

    def test_candidates_adjust_before_fold_fit_and_full_b_refit_uses_selected_tau(self):
        plan, rows, roles = build_happy_plan()
        rec = Recorder()
        prior = cg.compute_prior(make_train_rows(), 'tier', SCHEMA)
        probs = prior['probabilities']
        b_rows = [r for r in rows if r['split_group'] in set(plan['roles']['B'])]
        result = cg.select_tau_oof(plan, b_rows, {'tier': probs}, checkpoint_ref='ckpt-1',
                                   protocol=rec.protocol())
        chosen = result['selected_tau']
        taus = [c['tau'] for c in result['candidates']]
        # Each tau window contributes 6 fit calls (3 folds x tier+vision).
        self.assertEqual(len(rec.fit_calls), len(taus) * 6 + 2)  # + final tier/vision refit
        for window_start, tau in zip(range(0, len(rec.fit_calls) - 2, 6), taus):
            window = rec.fit_calls[window_start:window_start + 6]
            tier_calls = [c for c in window if c['width'] == 3]
            vision_calls = [c for c in window if c['width'] == 2]
            self.assertEqual(len(tier_calls), 3)
            self.assertEqual(len(vision_calls), 3)
            # Adjust BEFORE fit: tier training logits equal raw - tau*log(prior).
            call = tier_calls[0]
            group = call['groups'][0]
            expected = [[v - tau * _log(probs[c]) for v, c in zip(r['logits']['tier'], TIERS)]
                        for r in b_rows if r['split_group'] == group]
            actual = [row for row, g in zip(call['logits'], call['groups']) if g == group]
            self.assertEqual(len(expected), len(actual))
            for exp, act in zip(expected, actual):
                for e, a in zip(exp, act):
                    self.assertAlmostEqual(e, a, places=9)
            # Non-adjusted heads (vision) fit raw logits.
            vcall = vision_calls[0]
            vgroup = vcall['groups'][0]
            vexpected = [list(map(float, r['logits']['vision'])) for r in b_rows if r['split_group'] == vgroup]
            vactual = [row for row, g in zip(vcall['logits'], vcall['groups']) if g == vgroup]
            self.assertEqual(vexpected, vactual)
        # Final refit: last two fit calls (tier, vision) cover ALL B groups and
        # use the CHOSEN tau's adjustment.
        refit_tier = rec.fit_calls[-2]
        self.assertEqual(set(refit_tier['groups']), set(plan['roles']['B']))
        some_group = refit_tier['groups'][0]
        expected = [[v - chosen * _log(probs[c]) for v, c in zip(r['logits']['tier'], TIERS)]
                    for r in b_rows if r['split_group'] == some_group]
        actual = [row for row, g in zip(refit_tier['logits'], refit_tier['groups']) if g == some_group]
        self.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            self.assertAlmostEqual(exp[0], act[0], places=9)
        self.assertEqual(rec.selected, chosen)
        self.assertEqual(result['binding']['checkpoint_ref'], 'ckpt-1')
        self.assertEqual(result['binding']['plan_hash'], plan['plan_hash'])

    def test_holdout_missing_class_returns_null_and_no_tau_is_selected(self):
        rows = make_pool(n_groups=16)
        roles = make_roles(rows)
        counts = {g: sum(1 for r in rows if r['split_group'] == g) for g in roles['role_groups']['B']}
        folds = cg.group_folds(counts, 3)
        # Remove 'flagship' from the first fold's holdout groups (region gates
        # still pass: >=5 rows / >=2 groups remain in B and every fold-train).
        for r in rows:
            if r['split_group'] in set(folds[0]) and r['label']['tier'] == 'flagship':
                r['label']['tier'] = 'fast'
        add_logits(rows)
        plan = cg.build_plan(rows, roles['role_groups'], SCHEMA)
        prior = cg.compute_prior(make_train_rows(), 'tier', SCHEMA)
        b_rows = [r for r in rows if r['split_group'] in set(plan['roles']['B'])]
        result = cg.select_tau_oof(plan, b_rows, {'tier': prior['probabilities']}, checkpoint_ref='ckpt-2')
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['selected_tau'])
        self.assertNotIn('temperatures', result)  # nothing calibrated on unsupported support
        for candidate in result['candidates']:
            self.assertFalse(candidate['feasible'])
            self.assertIsNone(candidate['pooled'])


def _log(value):
    import math
    return math.log(value)


class DiagnosticTests(unittest.TestCase):
    def test_diagnostic_halves_never_produce_selected_products(self):
        rows = make_pool(n_groups=16)
        report = cg.diagnostic_halves(rows, SCHEMA, seed='42')
        self.assertTrue(report['diagnostic_only'])
        selection, fitting = set(report['selection_groups']), set(report['fitting_groups'])
        self.assertFalse(selection & fitting)  # whole-group halves, no row bisection
        self.assertEqual(selection | fitting, {r['split_group'] for r in rows})
        for forbidden in ('selected_tau', 'temperatures', 'calibrated'):
            self.assertNotIn(forbidden, report)
        # A diagnostic plan can never be consumed by selection.
        with self.assertRaises(cg.PlanError) as ctx:
            cg.select_tau_oof(report, [], {}, checkpoint_ref='ckpt')
        self.assertIn('diagnostic', str(ctx.exception))
        # Thin pools report insufficiency instead of silently passing.
        thin = make_pool(n_groups=4)
        thin_report = cg.diagnostic_halves(thin, SCHEMA, seed='42')
        self.assertTrue(thin_report['insufficient'])

    def test_diagnostic_halves_are_hash_based_and_seed_sensitivity_is_explicit(self):
        rows = make_pool(n_groups=16)
        a = cg.diagnostic_halves(rows, SCHEMA, seed='1')
        b = cg.diagnostic_halves(rows, SCHEMA, seed='1')
        c = cg.diagnostic_halves(rows, SCHEMA, seed='2')
        self.assertEqual(a['selection_groups'], b['selection_groups'])
        self.assertNotEqual((a['selection_groups'], a['fitting_groups']),
                            (c['selection_groups'], c['fitting_groups']))


class PriorTests(unittest.TestCase):
    def test_prior_excludes_synthetic_and_repeats_and_refuses_missing_support(self):
        rows = make_train_rows(n=30)
        rows[0]['synthetic'] = True
        rows.append(dict(rows[1]))  # repeated id
        prior = cg.compute_prior(rows, 'tier', SCHEMA)
        self.assertEqual(prior['excluded_synthetic'], 1)
        self.assertEqual(prior['excluded_duplicate'], 1)
        self.assertEqual(sum(prior['probabilities'].values()), 1.0)
        total = sum(prior['counts'].values())
        for c in TIERS:
            self.assertAlmostEqual(prior['probabilities'][c], (prior['counts'][c] + 1) / (total + 3))
        starved = [r for r in rows if r['label']['tier'] != 'flagship' and not r.get('synthetic')]
        starved = [r for r in starved if r['id'] != rows[1]['id']][:10]
        with self.assertRaises(cg.PlanError) as ctx:
            cg.compute_prior(starved, 'tier', SCHEMA)
        self.assertIn('without unique real train support', str(ctx.exception))


class OptionalHeadTests(unittest.TestCase):
    def test_task_type_is_fixed_at_t1_and_never_uses_legacy_fallback(self):
        plan, rows, roles = build_happy_plan()
        prior = cg.compute_prior(make_train_rows(), 'tier', SCHEMA)
        b_rows = [r for r in rows if r['split_group'] in set(plan['roles']['B'])]
        result = cg.select_tau_oof(plan, b_rows, {'tier': prior['probabilities']}, checkpoint_ref='ckpt-3')
        self.assertEqual(result['temperatures']['task_type'], 1.0)
        self.assertFalse(result['calibrated']['task_type'])
        self.assertNotIn('2.747', json.dumps(result))  # legacy checkpoint fallback never leaks in

    def test_optional_temperature_head_downgrades_to_t1_when_fold_coverage_missing(self):
        schema = {'heads': {
            'tier': dict(SCHEMA['heads']['tier']),
            'vision': dict(SCHEMA['heads']['vision']),
            'task_type': {'label_key': 'task_type', 'classes': list(SEVEN), 'required': False,
                          'calibration': 'temperature', 'adjust': False}}}
        rows = make_pool(n_groups=16)
        # Give task_type full region support but starve one class inside the
        # two B groups that deterministic folding places... simpler: confine
        # 'visual' to a single B group; region gates don't apply (optional),
        # but fold coverage fails -> downgraded to T=1, calibrated=False.
        for r in rows:
            if r['label']['task_type'] == 'visual' and r['split_group'] != 'dev-source-05':
                r['label']['task_type'] = 'research'
        for r in rows:
            if r['split_group'] == 'dev-source-05':
                r['label']['task_type'] = 'visual'
        add_logits(rows)
        for r in rows:
            r['logits']['task_type'] = [0.1 * (i == SEVEN.index(r['label']['task_type'])) for i in range(7)]
        roles = make_roles(rows)
        plan = cg.build_plan(rows, roles['role_groups'], schema)
        self.assertFalse(plan['optional_head_status']['task_type']['calibratable'])
        prior = cg.compute_prior(make_train_rows(), 'tier', SCHEMA)
        b_rows = [r for r in rows if r['split_group'] in set(plan['roles']['B'])]
        for r in b_rows:
            r['logits'].setdefault('task_type', [0.0] * 7)
        result = cg.select_tau_oof(plan, b_rows, {'tier': prior['probabilities']}, checkpoint_ref='ckpt-4')
        self.assertEqual(result['temperatures']['task_type'], 1.0)
        self.assertFalse(result['calibrated']['task_type'])
        self.assertEqual(result['status'], 'selected')  # optional head never blocks required calibration


class NumericTests(unittest.TestCase):
    def test_fit_temperature_is_deterministic_bounded_and_stable(self):
        rows = add_logits(make_pool(n_groups=6))
        logits = [r['logits']['tier'] for r in rows]
        cls = [TIERS.index(r['label']['tier']) for r in rows]
        groups = [r['split_group'] for r in rows]
        t1 = cg.fit_temperature(logits, cls, groups)
        t2 = cg.fit_temperature(logits, cls, groups)
        self.assertEqual(t1, t2)
        self.assertGreaterEqual(t1, cg.DEFAULT_PROTOCOL['temperature_bounds'][0])
        self.assertLessEqual(t1, cg.DEFAULT_PROTOCOL['temperature_bounds'][1])
        # Strongly-correct logits prefer the sharpest allowed temperature.
        sharp = [[5.0, 0.0, 0.0]] * 6
        self.assertAlmostEqual(cg.fit_temperature(sharp, [0] * 6, [f'g{i}' for i in range(6)]),
                               cg.DEFAULT_PROTOCOL['temperature_bounds'][0], places=3)

    def test_score_holdout_is_group_equal(self):
        # One group holds a single confident row; the other holds four
        # maximally-uncertain rows. Group-equal scoring weighs them 50/50
        # (row-level averaging would give the uncertain rows 4/5 of the vote).
        import math
        logits = [[4.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        cls = [0, 1, 1, 1, 1]
        groups = ['g0', 'g1', 'g1', 'g1', 'g1']
        value = cg.score_holdout(logits, cls, groups, temperature=1.0)
        good = -math.log(math.exp(4) / (math.exp(4) + 1.0))
        uncertain = math.log(2)
        self.assertAlmostEqual(value, (good + uncertain) / 2, places=9)


class ImportPurityTests(unittest.TestCase):
    def test_calibration_modules_import_without_torch(self):
        script_dir = ROOT / 'scripts'
        code = ('import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); '
                'import calibration_groups, validate_calibration; '
                "assert 'torch' not in sys.modules, 'torch leaked into structural validation'; "
                "print('PURE')" % (str(script_dir), str(ROOT)))
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PURE', result.stdout)


class StructuralDryRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='router-structural-')
        self.data_dir = Path(self.directory.name)
        write_jsonl(self.data_dir / 'train.jsonl', make_train_rows(n=30))
        write_jsonl(self.data_dir / 'calibration.jsonl', make_pool(n_groups=16))
        # Canary: if any code path opens and parses the test split, this invalid
        # JSON makes it crash. The structural check must never read test data.
        (self.data_dir / 'test.jsonl').write_text('NOT-VALID-JSON-SENTINEL\n')
        self.roles_path = self.data_dir / 'roles.json'
        self.roles_path.write_text(json.dumps(make_roles(make_pool(n_groups=16))))

    def tearDown(self):
        self.directory.cleanup()

    def test_validate_calibration_is_anonymous_and_never_opens_test(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_calibration.py'),
                                 '--data-dir', str(self.data_dir), '--roles', str(self.roles_path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload['ok'])
        self.assertNotIn('dev-source-', result.stdout)  # anonymous: no group names
        self.assertNotIn('fixture-row-', result.stdout)  # no ids either
        self.assertIn('plan_hash', payload['plan'])

    def test_validate_calibration_reports_violations_with_count_only_errors(self):
        broken = make_roles(make_pool(n_groups=16))
        broken['role_groups'].pop('C')
        path = self.data_dir / 'broken-roles.json'
        path.write_text(json.dumps(broken))
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_calibration.py'),
                                 '--data-dir', str(self.data_dir), '--roles', str(path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('pre-register', result.stdout)
        self.assertNotIn('dev-source-', result.stdout)

    def test_grouped_dry_run_runs_real_preflight_without_model_artifacts_or_test(self):
        run_name = 'calibration-structural-only'
        artifacts = ROOT / 'artifacts' / run_name
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'), '--run', run_name,
                                 '--roles', str(self.roles_path), '--data-dir', str(self.data_dir),
                                 '--dry-run'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Structural preflight', result.stdout)
        self.assertIn('plan_hash', result.stdout)
        self.assertNotIn('dev-source-', result.stdout)
        self.assertFalse(artifacts.exists())  # no output directory created
        # Violations make the grouped dry-run fail closed.
        broken = self.data_dir / 'broken-roles.json'
        broken.write_text(json.dumps({'role_groups': {'A': ['x'], 'B': ['y'], 'C': ['z']}}))
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'), '--run', run_name,
                                 '--roles', str(broken), '--data-dir', str(self.data_dir),
                                 '--dry-run'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


class TrainEvalAdapterTests(unittest.TestCase):
    """The actual train_eval adapters (the ones run() calls) must route only
    role-B rows into calibration_groups — tested with heavy libraries mocked,
    not by merely checking the helpers exist."""

    def _import_train_eval(self):
        torch = types.ModuleType('torch')
        torch.nn = types.ModuleType('torch.nn')
        torch.nn.functional = types.ModuleType('torch.nn.functional')
        torch.manual_seed = lambda *a, **k: None
        torch.set_num_threads = lambda *a, **k: None
        peft = types.ModuleType('peft')
        peft.LoraConfig = object
        peft.get_peft_model = lambda *a, **k: None
        model_utils = types.ModuleType('model_utils')
        model_utils.ROOT = ROOT
        model_utils.TIERS = TIERS
        model_utils.TYPES = SEVEN
        model_utils.QUESTIONS = {head: {'criteria': dict.fromkeys(SCHEMA['heads'][head]['classes'])}
                                 for head in SCHEMA['heads']}
        model_utils.load_engine = lambda *a, **k: None
        model_utils.read_jsonl = lambda *a, **k: []
        model_utils.request_for = lambda *a, **k: ({}, False)
        model_utils.field_logits = lambda *a, **k: ({}, 0)
        model_utils.score = lambda *a, **k: {}
        with mock.patch.dict(sys.modules, {'torch': torch, 'torch.nn': torch.nn,
                                           'torch.nn.functional': torch.nn.functional,
                                           'peft': peft, 'model_utils': model_utils}):
            sys.modules.pop('train_eval', None)
            import train_eval
            return train_eval

    def test_run_refuses_grouped_mode_without_roles_before_touching_torch(self):
        train_eval = self._import_train_eval()
        args = argparse.Namespace(run='fixture-run', seed=1, legacy_calibration=False,
                                  calibration_roles=None, structural_dry_run=False, smoke=False)
        with self.assertRaises(SystemExit) as ctx:
            train_eval.run(args)
        self.assertIn('--calibration-roles', str(ctx.exception))
        args.legacy_calibration = True
        args.structural_dry_run = True
        with self.assertRaises(SystemExit):
            train_eval.run(args)

    def test_adapter_routes_only_role_b_rows_and_requires_a_locked_checkpoint(self):
        train_eval = self._import_train_eval()
        schema = train_eval.grouped_schema()
        self.assertEqual(schema['heads']['tier']['classes'], TIERS)
        self.assertTrue(schema['heads']['vision']['required'])  # vision never silently downgraded
        rows = add_logits(make_pool(n_groups=16))
        roles_config = make_roles(rows)
        plan = train_eval.build_grouped_plan(rows, roles_config)
        prior = cg.compute_prior(make_train_rows(), 'tier', SCHEMA)
        scored = [{'id': r['id'], 'label': dict(r['label']),
                   'answers': {'tier': {'logits': r['logits']['tier']},
                               'vision': {'logits': r['logits']['vision']}},
                   'latency_ms': 1.0}
                  for r in rows]
        sentinels = [{'id': 'sentinel-A', 'label': {}, 'answers': {}, 'latency_ms': 0},
                     {'id': 'sentinel-C', 'label': {}, 'answers': {}, 'latency_ms': 0},
                     {'id': 'sentinel-test', 'label': {}, 'answers': {}, 'latency_ms': 0}]
        with mock.patch('calibration_groups.select_tau_oof') as mocked:
            mocked.return_value = {'status': 'selected'}
            result = train_eval.run_grouped_calibration(plan, scored + sentinels,
                                                        {'tier': prior['probabilities']},
                                                        protocol=plan['protocol'],
                                                        checkpoint_ref='ckpt-A-locked')
            self.assertEqual(result, {'status': 'selected'})
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            self.assertEqual(sorted(r['id'] for r in kwargs['rows']),
                             sorted(r['id'] for r in plan['rows_by_role']['B']))
            self.assertEqual(kwargs['checkpoint_ref'], 'ckpt-A-locked')
            self.assertEqual(kwargs['plan'], plan)
        # Without mocks the real adapter delegates end-to-end.
        result = train_eval.run_grouped_calibration(plan, scored, {'tier': prior['probabilities']},
                                                    protocol=plan['protocol'],
                                                    checkpoint_ref='ckpt-A-locked')
        self.assertEqual(result['status'], 'selected')
        self.assertIn('binding', result)
        with self.assertRaises(cg.PlanError):
            train_eval.run_grouped_calibration(plan, scored, {'tier': prior['probabilities']},
                                               protocol=plan['protocol'], checkpoint_ref='  ')

    def test_legacy_calibration_halves_preserve_archived_row_split(self):
        train_eval = self._import_train_eval()
        rows = make_pool(n_groups=8)
        selection, fitting = train_eval.legacy_calibration_halves(rows, seed=20261010)
        # Byte-equivalent reimplementation of the archived v3c statements.
        rng = random.Random(20261010)
        calib = rows[:]
        rng.shuffle(calib)
        half = max(1, len(calib) // 2)
        self.assertEqual([r['id'] for r in selection], [r['id'] for r in calib[:half]])
        self.assertEqual([r['id'] for r in fitting], [r['id'] for r in (calib[half:] or calib)])
        # Documented legacy behavior (the issue #2 leak): halves can share groups.
        self.assertTrue(set(r['split_group'] for r in selection) &
                        set(r['split_group'] for r in fitting))
        again = train_eval.legacy_calibration_halves(rows, seed=20261010)
        self.assertEqual([r['id'] for r in selection], [r['id'] for r in again[0]])


if __name__ == '__main__':
    unittest.main()
