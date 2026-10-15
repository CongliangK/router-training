"""Invented CPU-only metadata; no real data/weights/training, Refs #21."""
import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oof_protocol as p


def pool(n=30):
    return [{'id': f'synthetic-{g}-{i}', 'root_id': f'root-{g}', 'split_group': f'split-{g}',
             'source_split': 'train' if g % 2 else 'calibration',
             'label': {'tier': ('fast', 'balanced', 'flagship')[i],
                       'needs_vision': 'yes' if i == 0 else 'no'}}
            for g in range(n) for i in range(3)]


def plan():
    rows = pool()
    outer = p.development_folds(rows, root_audit_ref='synthetic-audit')
    roles = []
    for fold in outer['folds']:
        roots = fold['outside_roots']
        roles.append({'fit': roots[:-3], 'A': roots[-3:-2], 'B': roots[-2:-1], 'C': roots[-1:]})
    return p.nested_plan(rows, roles, root_audit_ref='synthetic-audit')


def predictions(pl):
    return [{'id': rid, 'root_id': row['root_id'], 'source_split': row['source_split'],
             'fold': f['fold'], 'epoch': epoch, 'plan_digest': pl['plan_digest'],
             'model_digest': p.digest([f['fold'], epoch]), 'training_roots': f['role_roots']['fit'],
             'logits': {'tier': [0., 1., 2.], 'vision': [0., 1.]}}
            for f in pl['folds'] for rid in f['validation_ids'] for epoch in (1, 2)
            for row in pl['rows'] if row['id'] == rid]


def context(conditional=False):
    return {'fixed_before_calibration': True, 'exchangeable_roots': True,
            'one_task_per_root_preselected': True, 'loss_preregistered': True,
            'no_calibration_label_selection': True, 'prediction_source': 'independent_fixed_model',
            'model_digest': 'a' * 64, 'approval_ref': 'synthetic-declaration-not-real-approval',
            'sampling_population': 'true_flagship_units' if conditional else 'all_units'}


def units(n, score=.8):
    return [{'root_id': f'u-{i}', 'true_flagship': True, 'score': score} for i in range(n)]


def model(identity):
    return {'model_digest': identity * 64, 'representation': 'dense_full',
            'base_id': 'synthetic-base', 'base_revision': 'c' * 64, 'architecture': 'synthetic',
            'tokenizer_digest': 'd' * 64, 'scoring_contract_digest': 'e' * 64,
            'parameters': {'weight': {'shape': [2, 3], 'dtype': 'float32'},
                           'bias': {'shape': [3], 'dtype': 'float32'}}}


class FoldsTests(unittest.TestCase):
    def test_deterministic_permuted_pool_whole_roots(self):
        a = p.development_folds(pool(), root_audit_ref='audit')
        b = p.development_folds(list(reversed(pool())), root_audit_ref='audit')
        self.assertEqual(a, b)
        held = [r for f in a['folds'] for r in f['validation_roots']]
        self.assertEqual(len(held), len(set(held)))
        for f in a['folds']:
            self.assertFalse(set(f['validation_roots']) & set(f['outside_roots']))
        self.assertFalse(a['qualified'])

    def test_explicit_no_test_source_and_missing_sources(self):
        for split in ('test', '', None, 'development'):
            rows = pool()
            rows[0]['source_split'] = split
            with self.assertRaises(p.ProtocolError):
                p.development_folds(rows, root_audit_ref='audit')

    def test_root_closure_over_split_groups_and_source_provenance(self):
        rows = pool()
        rows[1]['root_id'] = 'alias'
        with self.assertRaises(p.ProtocolError):
            p.development_folds(rows, root_audit_ref='audit')
        rows = pool()
        rows[0]['source_split'] = 'train'
        # Historical train/calibration roots may overlap; pooled development
        # still assigns the entire audited root to ONE outer fold.
        self.assertEqual(p.development_folds(rows, root_audit_ref='audit')['root_count'], 30)

    def test_duplicate_empty_invalid_metadata(self):
        for mutate in (lambda r: r.append(r[0]), lambda r: r[0].update(root_id=''),
                       lambda r: r[0].update(label={'tier': 'unknown', 'needs_vision': 'yes'})):
            rows = pool()
            mutate(rows)
            with self.assertRaises(p.ProtocolError):
                p.development_folds(rows, root_audit_ref='audit')
        with self.assertRaises(p.ProtocolError):
            p.development_folds([], root_audit_ref='audit')
        with self.assertRaises(p.ProtocolError):
            p.development_folds(pool())

    def test_support_fail_closed_before_folding(self):
        rows = pool()
        for r in rows:
            if int(r['root_id'].split('-')[1]) >= 4:
                r['label']['needs_vision'] = 'no'
        with self.assertRaisesRegex(p.ProtocolError, 'before fold'):
            p.development_folds(rows, root_audit_ref='audit')

    def test_greedy_is_not_stratified_and_may_be_infeasible(self):
        rows = pool()
        folds = p.development_folds(rows, root_audit_ref='audit')['folds']
        # Keep global positive support >=5, but none in the first greedy fold.
        missing = set(folds[0]['validation_roots'])
        for r in rows:
            if r['root_id'] in missing:
                r['label']['needs_vision'] = 'no'
        with self.assertRaisesRegex(p.ProtocolError, 'greedy fold'):
            p.development_folds(rows, root_audit_ref='audit')

    def test_fold_count_and_policy_not_booleans(self):
        for kw in ({'n_folds': True}, {'n_folds': 1}, {'min_class_roots': 0}, {'min_class_roots': True}):
            with self.assertRaises(p.ProtocolError):
                p.development_folds(pool(), root_audit_ref='audit', **kw)

    def test_independent_roles_and_actual_training_lineage(self):
        pl = plan()
        for f in pl['folds']:
            self.assertLess(f['actual_training_root_count'], f['eligible_outside_root_count'])
            self.assertEqual(set(f['role_roots']), {'fit', 'A', 'B', 'C'})
            self.assertEqual(len({r for v in f['role_roots'].values() for r in v}), len(f['outside_roots']))
        report = p.validate_oof_records(pl, predictions(pl), [1, 2])
        self.assertEqual(report['row_epoch_count'], 180)
        self.assertIsNone(report['independent_calibration_units'])
        self.assertFalse(report['qualified'])

    def test_reused_roles_and_outer_leakage_refused(self):
        pl = plan()
        roles = [f['role_roots'] for f in pl['folds']]
        roles[0]['B'] = roles[0]['A']
        with self.assertRaises(p.ProtocolError):
            p.nested_plan(pool(), roles, root_audit_ref='audit')
        roles = [f['role_roots'] for f in plan()['folds']]
        roles[0]['C'] = pl['folds'][0]['validation_roots'][:1]
        with self.assertRaises(p.ProtocolError):
            p.nested_plan(pool(), roles, root_audit_ref='audit')

    def test_prediction_duplicates_missing_mutated_plan_and_leaky_train(self):
        pl = plan()
        for mutate in (lambda rs: rs.pop(), lambda rs: rs.append(rs[0]),
                       lambda rs: rs[0].update(training_roots=pl['folds'][0]['outside_roots']),
                       lambda rs: rs[0].update(logits={'tier': [math.nan, 0, 0], 'vision': [0, 0]}),
                       lambda rs: rs[0].update(source_split='test'),
                       lambda rs: rs[0].update(model_digest='bad')):
            records = predictions(pl)
            mutate(records)
            with self.assertRaises(p.ProtocolError):
                p.validate_oof_records(pl, records, [1, 2])
        pl['rows'][0]['source_split'] = 'test'
        with self.assertRaises(p.ProtocolError):
            p.validate_oof_records(pl, [], [1, 2])


class CRCTests(unittest.TestCase):
    def solve(self, us, alpha=.05, ctx=None, estimand='joint_flagship_miss'):
        return p.crc_threshold(us, alpha=alpha, estimand=estimand, context=context() if ctx is None else ctx)

    def test_exact_five_percent_boundary_and_insufficient(self):
        self.assertEqual(self.solve(units(18))['status'], 'infeasible')
        r = self.solve(units(19))
        self.assertEqual(r['corrected_empirical_risk'], .05)
        self.assertEqual(r['threshold'], 1.0 - .8)
        self.assertEqual(r['empirical_misses'], 0)
        self.assertEqual(r['status'], 'assumption_conditional')
        self.assertFalse(r['qualified'])
        self.assertNotIn('high-probability', r['risk_statement'])

    def test_smallest_right_continuous_threshold(self):
        us = units(3, .5)
        us[0]['score'] = 1.
        us[2]['score'] = 0.
        r = self.solve(us, alpha=.5)
        self.assertEqual(r['threshold'], .5)
        self.assertEqual(r['empirical_misses'], 1)  # equality is NOT a miss
        self.assertEqual(self.solve(units(19, 0.))['threshold'], 1.)
        self.assertEqual(self.solve(units(19, 1.))['threshold'], 0.)

    def test_bruteforce_finite_step_solver(self):
        us = units(9)
        for i, u in enumerate(us):
            u['score'] = i / 8
        for alpha in (.1, .2, .35, .8):
            r = self.solve(us, alpha=alpha)
            feasible = [t for t in sorted({0., 1.} | {1-u['score'] for u in us})
                        if (sum(t < 1-u['score'] for u in us) + 1) / 10 <= alpha]
            self.assertEqual(r['threshold'], min(feasible))

    def test_expected_loss_for_exchangeable_finite_permutations(self):
        # Exhaust a tiny exchangeable (NOT independent) calibration/new-unit
        # experiment. This checks algebra, never a real-data coverage claim.
        population = units(4)
        for u, score in zip(population, (0., .25, .75, 1.)):
            u['score'] = score
        for alpha in (.25, .5, .75):
            total, count = 0, 0
            for permutation in itertools.permutations(population):
                r = self.solve(list(permutation[:3]), alpha=alpha)
                total += r['threshold'] < 1 - permutation[3]['score']
                count += 1
                self.assertFalse(r['coverage_claim'] or r['assumptions_verified'] or r['approval_verified'])
            self.assertLessEqual(total / count, alpha)

    def test_joint_is_not_conditional_risk(self):
        us = units(19, 1.)
        us[0]['true_flagship'] = False
        self.assertEqual(self.solve(us)['threshold'], 0.)
        with self.assertRaises(p.ProtocolError):
            self.solve(us, estimand='conditional_flagship_miss', ctx=context(True))
        r = self.solve(units(19), estimand='conditional_flagship_miss', ctx=context(True))
        self.assertEqual(r['status'], 'assumption_conditional')
        self.assertEqual(self.solve(units(19), ctx=context(True))['status'], 'exploratory')

    def test_oof_correlated_unapproved_never_inherits_guarantee(self):
        for key, value in (('prediction_source', 'oof'), ('exchangeable_roots', False),
                           ('approval_ref', None), ('fixed_before_calibration', False),
                           ('no_calibration_label_selection', False)):
            ctx = context()
            ctx[key] = value
            r = self.solve(units(19), ctx=ctx)
            self.assertEqual(r['status'], 'exploratory')
            self.assertIsNone(r['risk_statement'])
            self.assertFalse(r['qualified'])

    def test_empty_invalid_and_duplicate_units(self):
        self.assertEqual(self.solve([])['status'], 'infeasible')
        for bad in (math.nan, math.inf, True, -1., 1.1):
            us = units(19)
            us[0]['score'] = bad
            with self.assertRaises(p.ProtocolError):
                self.solve(us)
        us = units(19)
        us[1]['root_id'] = us[0]['root_id']
        with self.assertRaises(p.ProtocolError):
            self.solve(us)
        for alpha in (0., 1., True, math.nan):
            with self.assertRaises(p.ProtocolError):
                self.solve(units(19), alpha=alpha)


class SoupBridgeTests(unittest.TestCase):
    def test_compatible_metadata_does_not_average_load_or_qualify(self):
        r = p.soup_compatibility([model('a'), model('b')])
        self.assertEqual(r['status'], 'metadata_compatible_only')
        self.assertFalse(r['weights_loaded'] or r['weights_averaged'] or r['qualified'])

    def test_keys_shapes_base_dtype_semantic_refusals(self):
        mutations = [lambda m: m.update(base_revision='f'*64),
                     lambda m: m['parameters'].pop('bias'),
                     lambda m: m['parameters']['weight'].update(shape=[3, 2]),
                     lambda m: m['parameters']['weight'].update(dtype='int8'),
                     lambda m: m.update(representation='lora_factors'),
                     lambda m: m.update(representation='dora_factors'),
                     lambda m: m.update(representation='adapter_merged_dense'),
                     lambda m: m.update(tokenizer_digest='f'*64)]
        for change in mutations:
            m = model('b')
            change(m)
            with self.assertRaises(p.ProtocolError):
                p.soup_compatibility([model('a'), m])

    def test_base_revision_accepts_pinned_git_hashes_not_floating_names(self):
        members = [model('a'), model('b')]
        for m in members:
            m['base_revision'] = 'c' * 40
        self.assertEqual(p.soup_compatibility(members)['status'], 'metadata_compatible_only')
        for bad in ('main', '', 'z' * 40, 'c' * 39):
            members[1]['base_revision'] = bad
            with self.assertRaises(p.ProtocolError):
                p.soup_compatibility(members)

    def test_materialized_dense_merge_requires_reviewed_semantics(self):
        m = model('b')
        m.update(representation='adapter_merged_dense',
                 merge_semantics='materialized_delta_into_identical_base', merge_evidence_digest='f'*64)
        self.assertFalse(p.soup_compatibility([model('a'), m])['qualified'])
        m['merge_semantics'] = 'average_adapter_factors'
        with self.assertRaises(p.ProtocolError):
            p.soup_compatibility([model('a'), m])

    def bridge(self, **kw):
        args = dict(model_digest='a'*64, training_roots=['train'], selection_roots=['selection'],
                    calibration={'source': 'independent_fixed_model', 'model_digest': 'a'*64,
                                 'roots': ['fresh-B'], 'evidence_ref': 'external-calibration-required'},
                    confirmation_roots=['fresh-confirmation'], parity_ref='external-parity-required',
                    approval_ref='external-approval-required')
        args.update(kw)
        return p.deployment_bridge(**args)

    def test_bridge_never_certifies_and_requires_fresh_fixed_model_calibration(self):
        self.assertEqual(self.bridge()['status'], 'shadow')
        self.assertFalse(self.bridge()['qualified'])
        for source, model_digest in (('oof', 'a'*64), ('independent_fixed_model', 'b'*64)):
            with self.assertRaises(p.ProtocolError):
                self.bridge(calibration={'source': source, 'model_digest': model_digest, 'roots': ['B']})
        with self.assertRaises(p.ProtocolError):
            self.bridge(confirmation_roots=['fresh-B'])
        self.assertFalse(self.bridge(parity_ref=None)['structurally_complete'])


if __name__ == '__main__':
    unittest.main()
