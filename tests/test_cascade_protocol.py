"""Pure synthetic CPU contract tests, Refs #26; no executors or task labels."""
import copy
import math
import json
import subprocess
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cascade_protocol as c
from oof_protocol import ProtocolError


def frame(n=16, tasks=2):
    return [{'id': f'synthetic-{i}-{j}', 'root_id': f'r-{i}', 'split_group': f's-{i}',
             'duplicate_cluster': f'd-{i}-{j}', 'source_split': 'synthetic',
             'tier': ('fast', 'balanced', 'flagship')[i % 3]}
            for i in range(n) for j in range(tasks)]


def candidates():
    return [{'id': tier, 'provider': 'synthetic', 'model': f'synthetic-{tier}',
             'revision': 'synthetic-pinned-v1', 'effort': 'fixed', 'tier': tier,
             'acceptance_digest': 'a'*64, 'context_policy_digest': 'b'*64} for tier in c.TIERS]


def selection(n=16, tasks=2):
    return c.sample_anchors(frame(n, tasks), screening_roots=4, confirmation_roots=n-4,
                            tasks_per_root=tasks, seed=7)


def plan(sel=None, **overrides):
    args = dict(protocol_id='synthetic-v1', root_audit_ref='synthetic-audit',
                acceptance_ref='synthetic-acceptance', confirmation_schedule_ref='synthetic-schedule', explore_trials=2,
                confirmation_trials=3, alpha=.05, reliability_target=.9)
    args.update(overrides)
    return c.preregister(selection() if sel is None else sel, candidates(), **args)


def probabilities(pl, p=.7):
    return {r['root_id']: {candidate['id']: p for candidate in pl['candidates']}
            for stage in ('screening', 'confirmation') for r in pl['selection'][stage]}


def records(pl, passed=True):
    return [{'synthetic': True, 'stage': 'confirmation', 'task_id': a['id'], 'root_id': a['root_id'],
             'candidate_id': candidate['id'], 'trial': i, 'passed': passed}
            for a in pl['selection']['confirmation'] for candidate in pl['candidates']
            for i in range(pl['stages']['confirmation'])]


class SamplingTests(unittest.TestCase):
    def test_root_first_determinism_disjoint_no_duplicate_fill(self):
        a = c.sample_anchors(frame(), screening_roots=4, confirmation_roots=6, tasks_per_root=2, seed=11)
        b = c.sample_anchors(list(reversed(frame())), screening_roots=4, confirmation_roots=6,
                             tasks_per_root=2, seed=11)
        self.assertEqual(a, b)
        self.assertFalse({r['root_id'] for r in a['screening']} & {r['root_id'] for r in a['confirmation']})
        tiny = c.sample_anchors(frame(tasks=1), screening_roots=4, confirmation_roots=6,
                                tasks_per_root=2, seed=11)
        self.assertEqual(len(tiny['screening']), 4)
        self.assertEqual(len(a['screening']), 8)

    def test_task_schedule_does_not_depend_on_root_allocation(self):
        rows = frame(16, 5)
        a = c.sample_anchors(rows, screening_roots=4, confirmation_roots=12, tasks_per_root=2, seed=11)
        real_rng = c._rng
        def reversed_allocation(seed, *domain):
            if domain == ('root_allocation',):
                return type('Reverse', (), {'shuffle': lambda self, values: values.reverse()})()
            return real_rng(seed, *domain)
        with mock.patch.object(c, '_rng', side_effect=reversed_allocation):
            b = c.sample_anchors(rows, screening_roots=4, confirmation_roots=12, tasks_per_root=2, seed=11)
        def tasks(selection):
            return {r['root_id']: sorted(x['id'] for stage in ('screening', 'confirmation')
                    for x in selection[stage] if x['root_id'] == r['root_id'])
                    for stage in ('screening', 'confirmation') for r in selection[stage]}
        self.assertEqual(tasks(a), tasks(b))
        self.assertEqual(a['sampling']['randomness_policy'], c.RANDOMNESS)

    def test_tiers_are_descriptive_not_sampling_quotas(self):
        rows = frame()
        a = c.sample_anchors(rows, screening_roots=4, confirmation_roots=6, tasks_per_root=1, seed=3)
        for row in rows:
            row['tier'] = 'fast'
        b = c.sample_anchors(rows, screening_roots=4, confirmation_roots=6, tasks_per_root=1, seed=3)
        for stage in ('screening', 'confirmation'):
            self.assertEqual([r['id'] for r in a[stage]], [r['id'] for r in b[stage]])

    def test_split_ids_do_not_prove_root_independence(self):
        rows = frame()
        for r in rows:
            r['root_id'] = 'one-correlated-root'
        with self.assertRaisesRegex(ProtocolError, 'insufficient audited roots'):
            c.sample_anchors(rows, screening_roots=4, confirmation_roots=6, tasks_per_root=1, seed=1)

    def test_root_alias_and_near_duplicate_leak_refused(self):
        for field in ('split_group', 'duplicate_cluster'):
            rows = frame()
            rows[2][field] = rows[0][field]
            with self.assertRaises(ProtocolError):
                c.sample_anchors(rows, screening_roots=4, confirmation_roots=6, tasks_per_root=1, seed=1)

    def test_test_source_empty_invalid_counts_and_tier(self):
        for change in (lambda r: r[0].update(source_split='test'),
                       lambda r: r[0].update(tier='other'), lambda r: r.append(r[0])):
            rows = frame()
            change(rows)
            with self.assertRaises(ProtocolError):
                c.sample_anchors(rows, screening_roots=4, confirmation_roots=6, tasks_per_root=1, seed=1)
        for values in ((0, 1), (True, 1), (4, 3)):
            with self.assertRaises(ProtocolError):
                c.sample_anchors(frame(), screening_roots=values[0], confirmation_roots=6,
                                 tasks_per_root=values[1], seed=1)


class PreregistrationTests(unittest.TestCase):
    def test_frozen_caps_stopping_multiplicity_and_no_budget_claim(self):
        sel = selection()
        pl = plan(sel)
        ns, nc = len(sel['screening']), len(sel['confirmation'])
        self.assertEqual(pl['global_max_executions'], ns*(3*3+2)+nc*3*3)
        self.assertEqual(pl['multiplicity']['family_size'], 3)
        self.assertEqual(pl['stopping']['two_screen_passes'], 'ambiguous_stop_shadow_no_expansion')
        sel['screening'][0]['id'] = 'caller-mutated'
        self.assertNotEqual(sel['screening'][0]['id'], pl['selection']['screening'][0]['id'])
        self.assertFalse(pl['qualified'])
        self.assertNotEqual(pl['global_max_executions'], 270)

    def test_duplicate_chain_candidate_revision_effort_and_acceptance_required(self):
        for change in (lambda cs: cs[0].pop('effort'), lambda cs: cs[0].pop('revision'),
                       lambda cs: cs[1].update(id=cs[0]['id']),
                       lambda cs: cs[1].update(tier='fast'),
                       lambda cs: cs[0].update(acceptance_digest='')):
            cs = candidates()
            change(cs)
            with self.assertRaises(ProtocolError):
                c.preregister(selection(), cs, protocol_id='x', root_audit_ref='audit',
                              acceptance_ref='a', confirmation_schedule_ref='synthetic-schedule',
                              explore_trials=0, confirmation_trials=1,
                              alpha=.05, reliability_target=.9)

    def test_confirmation_overlap_and_cross_region_near_duplicates_refused(self):
        sel = selection()
        sel['confirmation'][0] = sel['screening'][0]
        with self.assertRaises(ProtocolError):
            plan(sel)
        sel = selection()
        sel['confirmation'][0]['duplicate_cluster'] = sel['screening'][0]['duplicate_cluster']
        with self.assertRaises(ProtocolError):
            plan(sel)

    def test_changed_selected_root_count_invalid_or_nonfinite_config(self):
        sel = selection()
        sel['sampling']['confirmation_root_count'] = 1
        with self.assertRaises(ProtocolError):
            plan(sel)
        for kw in ({'alpha': math.nan}, {'alpha': True}, {'confirmation_trials': True},
                   {'explore_trials': -1}, {'reliability_target': 1}, {'root_audit_ref': ''}):
            with self.assertRaises(ProtocolError):
                plan(**kw)


class ConfirmationTests(unittest.TestCase):
    def test_root_not_execution_n_and_equal_root_weights(self):
        sel = selection()
        # Make one root's task count smaller; its equal-root weight is unchanged.
        root = sel['confirmation'][0]['root_id']
        sel['confirmation'] = [r for i, r in enumerate(sel['confirmation'])
                               if r['root_id'] != root or i == 0]
        pl = plan(sel)
        rs = records(pl)
        for r in rs:
            r['passed'] = r['root_id'] != root
        report = c.confirmation_summary(pl, rs, independent_root_evidence=True)
        for s in report['candidate_summaries']:
            self.assertEqual(s['n_root'], 12)
            self.assertAlmostEqual(s['root_mean'], 11/12)
            expected = max(0., 11/12-math.sqrt(math.log(3/.05)/24))
            self.assertAlmostEqual(s['diagnostic_lower_bound'], expected)
        self.assertEqual(report['measured_labels'], [])
        self.assertEqual(report['gate_updates'], [])
        self.assertFalse(report['qualified'] or report['coverage_claim'])

    def test_repeated_correlated_successes_do_not_increase_root_n(self):
        pl1, pl2 = plan(confirmation_trials=1), plan(confirmation_trials=29)
        r1 = c.confirmation_summary(pl1, records(pl1), independent_root_evidence=True)
        r2 = c.confirmation_summary(pl2, records(pl2), independent_root_evidence=True)
        self.assertEqual(r1['candidate_summaries'][0]['diagnostic_lower_bound'],
                         r2['candidate_summaries'][0]['diagnostic_lower_bound'])
        self.assertLess(r2['candidate_summaries'][0]['diagnostic_lower_bound'], .9)

    def test_unproven_between_root_independence_returns_no_bound(self):
        pl = plan()
        r = c.confirmation_summary(pl, records(pl))
        self.assertIsNone(r['candidate_summaries'][0]['diagnostic_lower_bound'])
        self.assertFalse(r['candidate_summaries'][0]['diagnostic_accept'])

    def test_fixed_n_no_peeking_duplicate_selective_or_real_outcomes(self):
        pl = plan()
        for mutate in (lambda rs: rs.pop(), lambda rs: rs.append(rs[0]),
                       lambda rs: rs[0].update(stage='exploration'),
                       lambda rs: rs[0].update(synthetic=False),
                       lambda rs: rs[0].update(passed=1),
                       lambda rs: rs[0].update(trial=3)):
            rs = records(pl)
            mutate(rs)
            with self.assertRaises(ProtocolError):
                c.confirmation_summary(pl, rs, independent_root_evidence=True)

    def test_tampering_reliability_or_candidate_family_refused(self):
        pl = plan()
        rs = records(pl)
        pl['multiplicity']['family_size'] = 1
        with self.assertRaises(ProtocolError):
            c.confirmation_summary(pl, rs, independent_root_evidence=True)

    def test_zero_success_and_single_root_are_not_qualified(self):
        sel = c.sample_anchors(frame(2), screening_roots=1, confirmation_roots=1,
                               tasks_per_root=2, seed=1)
        pl = plan(sel)
        result = c.confirmation_summary(pl, records(pl, False), independent_root_evidence=True)
        self.assertTrue(all(s['diagnostic_lower_bound'] == 0 for s in result['candidate_summaries']))
        self.assertFalse(result['qualified'])


class SimulationTests(unittest.TestCase):
    def test_synthetic_cli_never_qualifies_or_emits_labels(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run([sys.executable, str(root/'scripts/protocol_demo.py'), '--repetitions', '3'],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertFalse(report['qualified'])
        self.assertEqual(report['crc']['status'], 'exploratory')
        self.assertIsNone(report['crc']['risk_statement'])
        self.assertEqual(report['cascade']['measured_labels'], [])
        self.assertFalse(report['null_stress']['coverage_claim'])

    def test_failures_upgrade_then_final_exploration_budget(self):
        pl = plan()
        probs = probabilities(pl, 0.)
        for table in probs.values():
            table['flagship'] = 1.
        report = c.simulate(pl, probs, seed=3)
        self.assertEqual(report['execution_count'], report['budget_cap'])
        self.assertEqual(report['stage_counts']['fast']['exploration'], 0)
        self.assertEqual(report['stage_counts']['flagship']['exploration'], 16)
        self.assertEqual(report['screening_decisions'], {'exploratory_only': 8})
        self.assertFalse(report['qualified'])

    def test_all_pass_supplements_only_first_but_confirms_all(self):
        pl = plan()
        result = c.simulate(pl, probabilities(pl, 1.), seed=1)
        self.assertEqual(result['stage_counts']['balanced']['screening'], 0)
        self.assertEqual(result['stage_counts']['balanced']['confirmation'], 72)
        self.assertEqual(result['measured_labels'], [])
        self.assertEqual(result, c.simulate(pl, probabilities(pl, 1.), seed=1))

    def test_ambiguous_screen_stops_without_expansion(self):
        pl = plan()
        # A deterministic search through CPU pseudo-random seeds exercises
        # the 2/3 branch without injecting an execution callback.
        found = False
        for seed in range(20):
            r = c.simulate(pl, probabilities(pl, .7), seed=seed, within_root_shared=False)
            if r['screening_decisions'].get('ambiguous_shadow', 0):
                found = True
                break
        self.assertTrue(found)

    def test_null_stress_counts_not_population_guarantees(self):
        pl = plan()
        report = c.simulation_stress(pl, probabilities(pl, .5), repetitions=30, seed=3)
        self.assertEqual(report['repetitions'], 30)
        self.assertEqual(report['diagnostic_false_accept_counts'], {t: 0 for t in c.TIERS})
        self.assertFalse(report['qualified'] or report['coverage_claim'])
        self.assertEqual(report['measured_labels'], [])

    def test_confirmation_invariant_to_screen_branches_and_supplement_budget(self):
        for shared in (False, True):
            pl = plan()
            base = probabilities(pl, .6)
            a = c.simulate(pl, base, seed=17, within_root_shared=shared)
            changed = copy.deepcopy(base)
            for row in pl['selection']['screening']:
                changed[row['root_id']] = {candidate['id']: 0. for candidate in pl['candidates']}
            b = c.simulate(pl, changed, seed=17, within_root_shared=shared)
            longer = plan(explore_trials=13)
            d = c.simulate(longer, base, seed=17, within_root_shared=shared)
            self.assertEqual(a['candidate_summaries'], b['candidate_summaries'])
            self.assertEqual(a['candidate_summaries'], d['candidate_summaries'])

    def test_nonvacuous_diagnostic_can_accept_without_qualifying(self):
        pl = plan(selection(n=44), reliability_target=.5, confirmation_trials=1)
        result = c.simulate(pl, probabilities(pl, 1.), seed=3)
        self.assertTrue(all(s['diagnostic_accept'] for s in result['candidate_summaries']))
        self.assertFalse(result['qualified'] or result['coverage_claim'])
        self.assertEqual(result['measured_labels'], [])
        stress = c.simulation_stress(pl, probabilities(pl, .49), repetitions=20, seed=3)
        self.assertEqual(stress['diagnostic_false_accept_counts'], {t: 0 for t in c.TIERS})
        self.assertFalse(stress['coverage_claim'])

    def test_invalid_probability_tables_and_degenerate_trials(self):
        pl = plan()
        for p in (math.nan, True, -1, 1.1):
            with self.assertRaises(ProtocolError):
                c.simulate(pl, probabilities(pl, p), seed=1)
        with self.assertRaises(ProtocolError):
            c.simulate(pl, {}, seed=1)
        with self.assertRaises(ProtocolError):
            c.simulation_stress(pl, probabilities(pl), repetitions=0, seed=1)


if __name__ == '__main__':
    unittest.main()
