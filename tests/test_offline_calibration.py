"""Issues #1/#22: invented logits only; no models loaded or private IO."""
import copy
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import offline_calibration as oc
from calibration_groups import PlanError
from raw_logit_contract import (RAW_SCHEMA, TIERS, align_sources, load_json, source_binding,
                                validate_costs, validate_prior, validate_source, write_report)


def source(groups=5, weak=False):
    rows = []
    for g in range(groups):
        for i, c in enumerate(TIERS):
            z = [-2., -2., -2.]
            z[i] = 3. + g / 20
            if weak:
                z = [0., 0., 0.]
                z[(i+1) % 3] = 1.
            rows.append({'id': f'invented-row-{g}-{i}', 'split_group': f'invented-group-{g}', 'label': c, 'logits': z})
    return {'schema_version': RAW_SCHEMA, 'data_kind': 'synthetic', 'split': 'calibration', 'head': 'tier',
            'class_order': list(TIERS), 'logit_stage': 'raw-pre-temperature-pre-prior',
            'dataset_sha256': 'a'*64, 'scoring_sha256': 'b'*64, 'model_sha256': ('c' if weak else 'd')*64,
            'rows': rows}


def wise_protocol():
    # Explicit provisional fixture gates, NOT a real approved risk protocol.
    return {'schema_version': 'router-nested-logo/v1', 'alpha_grid': [0., .5, 1.],
            'temperature_bounds': [.25, 4.], 'fit_iterations': 30, 'fit_tolerance': 1e-5,
            'min_fit_groups': 2, 'min_class_rows': 2, 'min_class_groups': 2, 'min_predicted_groups': 2,
            'min_flagship_recall': .75, 'max_flagship_fpr': .1, 'min_fast_precision': .75}


def tau_protocol():
    return {'schema_version': 'router-tau-ablation/v1', 'tau_grid': [0., .5, 1., 1.5, 2.],
            'fixed_temperature': 1., 'comparison_tau': 0., 'min_class_rows': 2, 'min_class_groups': 2,
            'min_uncertainty_groups': 4, 'bootstrap_samples': 100, 'bootstrap_seed': 7,
            'min_flagship_recall': .75, 'max_flagship_fpr': .1}


def prior(src):
    return {'schema_version': 'router-tier-prior/v1', 'source': 'unique-training-before-oversampling',
            'model_sha256': src['model_sha256'], 'class_order': list(TIERS), 'probabilities': [.1, .8, .1]}


def costs():
    return {'schema_version': 'router-relative-cost/v1', 'version': 'invented-v1',
            'units': 'relative-simulation-not-billing',
            'relative_cost_by_predicted_tier': {'fast': 1., 'balanced': 3., 'flagship': 9.}}


class ContractTests(unittest.TestCase):
    def test_order_independent_alignment_and_binding(self):
        a, b = source(), source(weak=True)
        a['rows'].reverse()
        left, right = align_sources(a, b)
        self.assertEqual([r['id'] for r in left], [r['id'] for r in right])
        c = copy.deepcopy(a);c['rows'].reverse()
        self.assertEqual(source_binding(a), source_binding(c))

    def test_reject_unknown_fields_any_level(self):
        for location in ('top', 'row'):
            s = source()
            (s if location == 'top' else s['rows'][0])['prompt'] = 'not allowed even if synthetic'
            with self.assertRaises(PlanError):validate_source(s)

    def test_test_train_and_calibrated_inputs_rejected(self):
        for key, values in {'split': ['test', 'train', 'validation', None],
                            'logit_stage': ['probabilities', 'calibrated', None],
                            'head': ['vision'], 'class_order': [list(reversed(TIERS))],
                            'schema_version': [True, 1], 'data_kind': ['unknown']}.items():
            for value in values:
                s = source();s[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(PlanError):validate_source(s)

    def test_logits_finite_numeric_shape_and_range(self):
        for value in [True, float('nan'), float('inf'), -float('inf'), '3', None, 1e7, 10**400]:
            s = source();s['rows'][0]['logits'][0] = value
            with self.subTest(value=value), self.assertRaises(PlanError):validate_source(s)
        for value in [[], [1, 2], [1, 2, 3, 4], {}]:
            s = source();s['rows'][0]['logits'] = value
            with self.assertRaises(PlanError):validate_source(s)

    def test_empty_duplicate_and_illegal_identity_label(self):
        s = source();s['rows'] = []
        with self.assertRaises(PlanError):validate_source(s)
        s = source();s['rows'].append(copy.deepcopy(s['rows'][0]))
        with self.assertRaises(PlanError):validate_source(s)
        for key, value in [('id', ''), ('split_group', ' '), ('split_group', None), ('label', 'unknown')]:
            s = source();s['rows'][0][key] = value
            with self.assertRaises(PlanError):validate_source(s)

    def test_non_utf8_identities_fail_closed_before_digesting(self):
        for key in ('id', 'split_group'):
            with self.subTest(key=key):
                s = source();s['rows'][0][key] = 'withheld-\ud800'
                with self.assertRaises(PlanError) as error:source_binding(s)
                self.assertNotIn('withheld-', str(error.exception))

    def test_alignment_rejects_missing_extra_and_mismatched_rows_metadata(self):
        for key, value in [('dataset_sha256', 'e'*64), ('scoring_sha256', 'e'*64), ('split', 'development'), ('data_kind', 'private-local')]:
            a, b = source(), source();b[key] = value
            with self.assertRaises(PlanError):align_sources(a, b)
        for key, value in [('id', 'other'), ('split_group', 'other'), ('label', 'balanced')]:
            a, b = source(), source();b['rows'][0][key] = value
            with self.assertRaises(PlanError):align_sources(a, b)
        a, b = source(), source();b['rows'].pop()
        with self.assertRaises(PlanError):align_sources(a, b)

    def test_model_binding_and_prior_provenance(self):
        s = source();p = prior(s)
        validate_prior(p, s)
        for key, value in [('source', 'calibration-estimated'), ('model_sha256', 'f'*64),
                            ('probabilities', [.1, .1, .1]), ('probabilities', [0., .5, .5]),
                            ('probabilities', [True, 0., 0.])]:
            bad = copy.deepcopy(p);bad[key] = value
            with self.assertRaises(PlanError):validate_prior(bad, s)

    def test_versioned_relative_costs_not_ordinal_error_or_routes(self):
        c = costs();self.assertEqual(validate_costs(c)['flagship'], 9.)
        for key, value in [('version', ''), ('units', 'USD'), ('schema_version', 'routes/v1'),
                            ('relative_cost_by_predicted_tier', {})]:
            bad = copy.deepcopy(c);bad[key] = value
            with self.assertRaises(PlanError):validate_costs(bad)
        for v in [True, 0, -1, float('nan'), float('inf')]:
            bad = costs();bad['relative_cost_by_predicted_tier']['fast'] = v
            with self.assertRaises(PlanError):validate_costs(bad)

    def test_json_duplicate_keys_and_nonfinite_rejected_with_safe_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'not-a-public-identifier.json'
            for text in ['{"rows":1,"rows":2}', '{"x":NaN}', '{bad']:
                path.write_text(text)
                with self.assertRaises(PlanError) as error:load_json(path)
                self.assertNotIn(str(path), str(error.exception))
            with self.assertRaises(PlanError):load_json(Path(d) / 'missing')

    def test_append_only_report_and_strict_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'report.json';write_report(p, {'a': 1})
            with self.assertRaises(PlanError):write_report(p, {'a': 2})
            self.assertEqual(json.loads(p.read_text()), {'a': 1})
            with self.assertRaises(ValueError):write_report(Path(d) / 'bad.json', {'a': float('nan')})
            self.assertFalse((Path(d) / 'bad.json').exists())


class ThresholdTests(unittest.TestCase):
    def test_continuous_breakpoints_and_flagship_overlap_priority(self):
        rows = source(4)['rows'];p = wise_protocol()
        probs = [[.631234, .2, .168766] if r['label'] == 'fast' else
                 [.12, .721111, .158889] if r['label'] == 'balanced' else [.1, .17, .73] for r in rows]
        thresholds = oc.choose_thresholds(rows, probs, p)
        self.assertIsNotNone(thresholds)
        self.assertIn(thresholds['theta_fast'], {x[0] for x in probs})
        self.assertIn(thresholds['theta_flag'], {x[2] for x in probs})
        pred = oc.predict([{'label': 'fast', 'split_group': 'g', 'logits': [0., 0., 0.]}], 1,
                          {'theta_flag': .3, 'theta_fast': .3})
        self.assertEqual(pred[0]['prediction'], 'flagship')

    def test_exact_solver_matches_brute_force_empirical_partitions(self):
        rows = source(4)['rows'];p = wise_protocol()
        probs = [oc.probabilities(r['logits'], 2) for r in rows]
        result = oc.choose_thresholds(rows, probs, p)
        candidates = []
        for tf, tx in itertools.product(sorted({0., math.nextafter(1., math.inf), *(v[2] for v in probs)}),
                                       sorted({0., math.nextafter(1., math.inf), *(v[0] for v in probs)})):
            predicted = [{**r, 'probabilities': v, 'prediction': 'flagship' if v[2] >= tf else 'fast' if v[0] >= tx else 'balanced'}
                         for r, v in zip(rows, probs)]
            m = oc.summarize(predicted)['group_equal']
            if oc._constraints(m, p, fast=True) and all(len({r['split_group'] for r in predicted if r['prediction'] == c}) >= 2
                                                      for c in ('fast', 'flagship')):
                candidates.append(m['error'])
        selected = [{**r, 'probabilities': v, 'prediction': 'flagship' if v[2] >= result['theta_flag'] else
                     'fast' if v[0] >= result['theta_fast'] else 'balanced'} for r, v in zip(rows, probs)]
        self.assertAlmostEqual(oc.summarize(selected)['group_equal']['error'], min(candidates))

    def test_no_feasible_thresholds_never_relaxes_constraints(self):
        rows = source(4)['rows'];p = wise_protocol()
        self.assertIsNone(oc.choose_thresholds(rows, [[1/3]*3 for _ in rows], p))

    def test_support_failure_before_any_temperature_fitting(self):
        a = source(3)
        with mock.patch.object(oc, 'fit_temperature', side_effect=AssertionError('fit must not happen')):
            report = oc.nested_logo(a, copy.deepcopy(a), wise_protocol())
        self.assertEqual(report['status'], 'insufficient');self.assertIsNone(report['outer_oof'])

    def test_class_concentration_cannot_borrow_outer_labels(self):
        a = source(5)
        for r in a['rows']:
            if r['label'] == 'fast':r['split_group'] = 'concentrated'
        report = oc.nested_logo(a, copy.deepcopy(a), wise_protocol())
        self.assertEqual(report['status'], 'insufficient')
        self.assertIsNone(report['deployment_configuration'])


class NestedTests(unittest.TestCase):
    def test_nested_complete_synthetic_and_no_fitted_deployment_claim(self):
        a = source();report = oc.nested_logo(a, copy.deepcopy(a), wise_protocol())
        self.assertEqual(report['status'], 'available')
        self.assertEqual(report['outer_oof']['rows'], len(a['rows']))
        self.assertEqual(len(report['outer_folds']), 5)
        # Even this separable fixture can miss a held-out confidence breakpoint;
        # do not turn synthetic mechanics into a guaranteed accuracy claim.
        self.assertEqual(sum(sum(v.values()) for v in report['outer_oof']['confusion'].values()), len(a['rows']))
        self.assertTrue(0 <= report['outer_oof']['accuracy'] <= 1)
        self.assertIsNone(report['deployment_configuration']);self.assertFalse(report['test_used'])
        for f in report['outer_folds']:
            self.assertEqual(f['fit_groups'], 4)
            self.assertEqual(f['holdout_groups'], 1)
            self.assertEqual(f['selection']['alpha'], 0.)
            for c in f['candidates']:
                self.assertEqual(len(c['inner_folds']), 4)
                self.assertTrue(all(x['fit_groups'] == 3 and x['validation_groups'] == 1 for x in c['inner_folds']))
        serialized = json.dumps(report, allow_nan=False)
        for r in a['rows']:
            self.assertNotIn(r['id'], serialized);self.assertNotIn(r['split_group'], serialized)
        self.assertNotIn('logits', serialized)

    def test_outer_labels_cannot_change_its_selection(self):
        a, b = source(5, weak=True), source()
        first = oc.nested_logo(a, b, wise_protocol())
        # Preserve support while changing ONLY the first outer group's labels.
        for src in (a, b):
            for r in src['rows'][:3]:r['label'] = TIERS[(TIERS.index(r['label'])+1) % 3]
        second = oc.nested_logo(a, b, wise_protocol())
        self.assertEqual(first['outer_folds'][0]['selection'], second['outer_folds'][0]['selection'])
        self.assertEqual(first['outer_folds'][0]['candidates'], second['outer_folds'][0]['candidates'])
        self.assertNotEqual(first['outer_folds'][0]['holdout_metrics'], second['outer_folds'][0]['holdout_metrics'])

    def test_inner_validation_never_fits_temperature_or_thresholds(self):
        a = source();rows = validate_source(a)
        original = oc._fit;fit_groups = []
        def fit(train, p):
            fit_groups.append({r['split_group'] for r in train})
            return original(train, p)
        p = wise_protocol();p['alpha_grid'] = [0.]
        with mock.patch.object(oc, '_fit', side_effect=fit):oc.select_outer_training(rows, rows, p)
        groups = sorted({r['split_group'] for r in rows})
        self.assertEqual(fit_groups[:-1], [set(groups)-{g} for g in groups])
        self.assertEqual(fit_groups[-1], set(groups))

    def test_every_outer_group_is_absent_from_all_tuning_fits(self):
        a, b = source(), source(weak=True);p = wise_protocol()
        all_groups = {r['split_group'] for r in a['rows']};seen = set()
        original_select, original_fit = oc.select_outer_training, oc._fit
        def select(left, right, protocol):
            training_groups = {r['split_group'] for r in left}
            held_out = all_groups - training_groups
            self.assertEqual(len(held_out), 1);seen.update(held_out)
            self.assertEqual([r['id'] for r in left], [r['id'] for r in right])
            self.assertEqual(training_groups, {r['split_group'] for r in right})
            def fit(rows, registered):
                self.assertTrue({r['split_group'] for r in rows} <= training_groups)
                self.assertTrue(all(r['split_group'] not in held_out for r in rows))
                return original_fit(rows, registered)
            with mock.patch.object(oc, '_fit', side_effect=fit):
                return original_select(left, right, protocol)
        with mock.patch.object(oc, 'select_outer_training', side_effect=select):
            report = oc.nested_logo(a, b, p)
        self.assertEqual(report['status'], 'available')
        self.assertEqual(seen, all_groups)

    def test_no_candidate_is_insufficient_not_tau_or_temperature_fallback(self):
        a = source()
        for r in a['rows']:r['logits'] = [0., 0., 0.]
        report = oc.nested_logo(a, copy.deepcopy(a), wise_protocol())
        self.assertEqual(report['status'], 'insufficient');self.assertIsNone(report['outer_oof'])
        self.assertIsNone(report['outer_folds'][0]['selection'])

    def test_permutation_invariance_and_determinism(self):
        a, b = source(), source(weak=True);p = wise_protocol()
        first = oc.nested_logo(a, b, p)
        a['rows'].reverse();b['rows'] = b['rows'][1:] + b['rows'][:1]
        self.assertEqual(first, oc.nested_logo(a, b, p))

    def test_protocol_rejects_unknown_bool_nonfinite_and_silent_defaults(self):
        for key, value in [('alpha_grid', [True]), ('alpha_grid', [.5, .5]), ('temperature_bounds', [0., 1.]),
                            ('fit_iterations', True), ('fit_tolerance', float('nan')), ('min_fit_groups', 1),
                            ('min_class_groups', 0), ('min_predicted_groups', 1), ('min_flagship_recall', 0),
                            ('min_fast_precision', -1), ('max_flagship_fpr', float('inf'))]:
            p = wise_protocol();p[key] = value
            with self.assertRaises(PlanError):oc.nested_logo(source(), source(), p)
        p = wise_protocol();p['unregistered'] = 1
        with self.assertRaises(PlanError):oc.validate_wise_protocol(p)


class TauTests(unittest.TestCase):
    def test_requested_grid_fixed_T_sign_cost_and_metrics(self):
        s = source();report = oc.tau_ablation(s, prior(s), costs(), tau_protocol())
        self.assertEqual([c['tau'] for c in report['candidates']], [0., .5, 1., 1.5, 2.])
        for c in report['candidates']:
            m = c['metrics']
            for key in ('confusion', 'per_class', 'nll', 'brier', 'ece_5_bins', 'mean_relative_cost'):self.assertIn(key, m)
            self.assertAlmostEqual(m['mean_relative_cost'], 13/3)
        self.assertFalse(report['test_used'])
        self.assertIn('matched #19', report['training_causal_conclusion'])
        self.assertEqual(report['comparison']['conclusion'], 'synthetic_only_no_model_conclusion')
        self.assertEqual(report['cost_table'], costs())
        # Equal raw scores and unequal prior: z - tau*log(prior) boosts rare fast/flagship.
        r = {'label': 'balanced', 'split_group': 'g', 'logits': [0., 0., 0.]}
        adjusted = {**r, 'logits': [-2*math.log(x) for x in [.1, .8, .1]]}
        self.assertEqual(oc.predict([adjusted], 1)[0]['prediction'], 'fast')

    def test_hand_computed_metrics_and_predicted_probability_ece(self):
        rows = [{'label': 'fast', 'split_group': 'g0', 'prediction': 'fast', 'probabilities': [.8, .1, .1]},
                {'label': 'balanced', 'split_group': 'g1', 'prediction': 'fast', 'probabilities': [.6, .2, .2]}]
        m = oc.summarize(rows, costs()['relative_cost_by_predicted_tier'])
        self.assertAlmostEqual(m['accuracy'], .5)
        self.assertAlmostEqual(m['macro_f1'], (2/3)/3)
        self.assertAlmostEqual(m['nll'], (-math.log(.8)-math.log(.2))/2)
        self.assertAlmostEqual(m['brier'], .55)
        self.assertAlmostEqual(m['ece_5_bins'], .4)
        self.assertEqual(m['mean_relative_cost'], 1.)
        self.assertIsNone(m['per_class']['flagship']['recall'])

    def test_metric_parity_with_existing_shared_v2_implementation(self):
        # The CLI reuses shared metric helpers without loading torch/weights.
        # Compare the complete tier result too. Do not reuse routing_costs as
        # simulated spend: its zero-diagonal ordinal error loss is a different quantity.
        from evaluation_metrics import metrics
        boundary = [{'label': 'fast' if i % 2 else 'balanced', 'split_group': f'g{i % 3}',
                     'prediction': 'fast', 'probabilities': [v, (1-v)/2, (1-v)/2]}
                    for i, v in enumerate([0., .2, .4, .6, .8, 1.])]
        regular = oc.predict(source()['rows'], 1.7, {'theta_flag': .95, 'theta_fast': .7})
        for rows in (regular, boundary, boundary[:1]):
            shared_rows = [{'label': {'tier': r['label'], 'needs_vision': None, 'task_type': None},
                            'split_group': r['split_group'], 'context_truncated': False, 'latency_ms': 0.,
                            'answers': {'tier': {'choice': r['prediction'],
                                                'confidence': r['probabilities'][TIERS.index(r['prediction'])],
                                                'probabilities': dict(zip(TIERS, r['probabilities']))}}} for r in rows]
            expected, actual = metrics(shared_rows), oc.summarize(rows)
            self.assertEqual(expected['metric_definition_version'], 2)
            self.assertEqual(actual['confusion'], expected['tier']['confusion'])
            for key in ('accuracy', 'macro_f1', 'nll', 'brier', 'ece_5_bins'):
                self.assertAlmostEqual(actual[key], expected['tier'][key], places=12)
            self.assertEqual(actual['ece_bin_counts'], [b['count'] for b in expected['tier']['ece_bins']])
            for c in TIERS:
                for key, value in actual['per_class'][c].items():
                    self.assertEqual(value, expected['tier_per_class'][c][key])

    def test_group_equal_statistics_ignore_within_group_replication(self):
        rows = [{'label': 'fast', 'split_group': 'g0', 'prediction': 'fast', 'probabilities': [.8, .1, .1]},
                {'label': 'balanced', 'split_group': 'g1', 'prediction': 'fast', 'probabilities': [.6, .2, .2]}]
        a = oc.summarize(rows, costs()['relative_cost_by_predicted_tier'])
        b = oc.summarize(rows + [dict(rows[0]) for _ in range(9)], costs()['relative_cost_by_predicted_tier'])
        self.assertNotEqual(a['accuracy'], b['accuracy'])
        for key, value in a['group_equal'].items():
            if value is None:self.assertIsNone(b['group_equal'][key])
            else:self.assertAlmostEqual(value, b['group_equal'][key], places=12)

    def test_tau_nll_uses_existing_adjust_then_temperature_semantics(self):
        from calibration_groups import score_holdout
        s = source();p = tau_protocol();p['fixed_temperature'] = 2.7
        report = oc.tau_ablation(s, prior(s), costs(), p)
        for candidate in report['candidates']:
            adjusted = [[z-candidate['tau']*math.log(v) for z, v in zip(r['logits'], prior(s)['probabilities'])]
                        for r in s['rows']]
            expected = score_holdout(adjusted, [TIERS.index(r['label']) for r in s['rows']],
                                     [r['split_group'] for r in s['rows']], p['fixed_temperature'])
            self.assertAlmostEqual(candidate['metrics']['group_equal']['nll'], expected, places=12)

    def test_cost_simulation_is_predicted_route_spend_not_error_loss(self):
        s = source();c = costs()
        c['relative_cost_by_predicted_tier'] = {'fast': 2., 'balanced': 2., 'flagship': 2.}
        r = oc.tau_ablation(s, prior(s), c, tau_protocol())
        self.assertTrue(all(x['metrics']['mean_relative_cost'] == 2. for x in r['candidates']))

    def test_large_finite_cost_units_do_not_overflow_aggregation(self):
        s = source();c = costs()
        c['relative_cost_by_predicted_tier'] = dict.fromkeys(TIERS, 1e308)
        report = oc.tau_ablation(s, prior(s), c, tau_protocol())
        json.dumps(report, allow_nan=False)
        for candidate in report['candidates']:
            self.assertEqual(candidate['metrics']['mean_relative_cost'], 1e308)
            self.assertTrue(math.isfinite(candidate['metrics']['group_equal']['mean_relative_cost']))

    def test_ece_boundary_neighbors_are_disjoint(self):
        for boundary, index in [(0., 0), (.2, 1), (.4, 2), (.6, 3), (.8, 4), (1., 4)]:
            for value, expected in [(boundary, index), (math.nextafter(boundary, 0.), index-1 if 0 < boundary < 1 else index)]:
                row = {'label': 'fast', 'split_group': 'g', 'prediction': 'fast',
                       'probabilities': [value, (1-value)/2, (1-value)/2]}
                m = oc.summarize([row])
                self.assertEqual(m['ece_bin_counts'][expected], 1)
                self.assertEqual(sum(m['ece_bin_counts']), 1)
                self.assertAlmostEqual(m['ece_5_bins'], 1-value)

    def test_group_support_and_infeasible_no_selection(self):
        s = source(1);r = oc.tau_ablation(s, prior(s), costs(), tau_protocol())
        self.assertEqual(r['status'], 'insufficient');self.assertIsNone(r['selected_development_tau'])
        self.assertEqual(r['comparison']['intervals']['nll']['status'], 'unavailable')
        p = tau_protocol();p['max_flagship_fpr'] = 0.
        for row in s['rows']:row['logits'] = [0., 0., 4.]
        self.assertIsNone(oc.tau_ablation(s, prior(s), costs(), p)['selected_development_tau'])

    def test_paired_bootstrap_cluster_multiplicity_matches_reference(self):
        rows = [{'label': 'fast', 'split_group': 'a', 'prediction': 'fast', 'probabilities': [.8, .1, .1]},
                {'label': 'fast', 'split_group': 'a', 'prediction': 'fast', 'probabilities': [.8, .1, .1]},
                {'label': 'balanced', 'split_group': 'b', 'prediction': 'fast', 'probabilities': [.4, .5, .1]}]
        other = [{**r, 'probabilities': [.5, .4, .1]} for r in rows]
        p = tau_protocol();p.update(min_uncertainty_groups=2)
        # Gates are independently tested; isolate paired sampling arithmetic.
        with mock.patch.object(oc, 'supported', return_value=True):
            result = oc.paired_group_interval(rows, other, p, lambda rs:oc.summarize(rs)['group_equal']['nll'])
        # Draws [a,a] / [b,b] must reproduce each cluster's delta despite row counts.
        self.assertAlmostEqual(result['low'], -math.log(.8)+math.log(.5))
        self.assertAlmostEqual(result['high'], -math.log(.5)+math.log(.4))
        self.assertEqual(result['valid_resamples'], 100)

    def test_uncertainty_support_not_claimed_from_selection_support(self):
        s = source(3);s['data_kind'] = 'private-local'  # Still entirely invented values.
        r = oc.tau_ablation(s, prior(s), costs(), tau_protocol())
        self.assertEqual(r['status'], 'available')
        self.assertEqual(r['comparison']['conclusion'], 'uncertain')
        self.assertTrue(all(x['status'] == 'unavailable' for x in r['comparison']['intervals'].values()))

    def test_tau_protocol_explicit_and_numeric(self):
        for key, value in [('tau_grid', [0, 2]), ('comparison_tau', 2.), ('fixed_temperature', True),
                            ('bootstrap_samples', 1), ('bootstrap_seed', -1), ('min_class_rows', True),
                            ('min_class_groups', 1), ('min_uncertainty_groups', 1)]:
            p = tau_protocol();p[key] = value
            with self.assertRaises(PlanError):oc.validate_tau_protocol(p)


class CliTests(unittest.TestCase):
    def run_cli(self, script, args):
        return subprocess.run([sys.executable, str(ROOT/'scripts'/script), *args], capture_output=True, text=True)

    def test_both_clis_new_report_no_overwrite_and_no_test_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);s = source()
            for name, obj in [('base', s), ('ft', copy.deepcopy(s)), ('wise', wise_protocol()),
                              ('tau', tau_protocol()), ('prior', prior(s)), ('costs', costs())]:
                (root/f'{name}.json').write_text(json.dumps(obj))
            cases = [('wise_ft_calibrate.py', ['--base', str(root/'base.json'), '--finetuned', str(root/'ft.json'),
                                              '--protocol', str(root/'wise.json')]),
                     ('tau_ablation.py', ['--logits', str(root/'base.json'), '--prior', str(root/'prior.json'),
                                          '--costs', str(root/'costs.json'), '--protocol', str(root/'tau.json')])]
            for i, (script, args) in enumerate(cases):
                output = root/f'report-{i}.json';args += ['--output', str(output)]
                result = self.run_cli(script, args)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                before = output.read_bytes()
                result = self.run_cli(script, args)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(output.read_bytes(), before)
                self.assertNotIn('invented-row', output.read_text())
                result = self.run_cli(script, [*args, '--test'])
                self.assertEqual(result.returncode, 2)

    def test_cli_invalid_sanitized_and_infeasible_exit_two(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);s = source(3)
            for name, obj in [('s', s), ('p', wise_protocol())]:(root/f'{name}.json').write_text(json.dumps(obj))
            args = ['--base', str(root/'s.json'), '--finetuned', str(root/'s.json'), '--protocol', str(root/'p.json'),
                    '--output', str(root/'output.json')]
            result = self.run_cli('wise_ft_calibrate.py', args)
            self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
            self.assertEqual(json.loads((root/'output.json').read_text())['status'], 'insufficient')
            s['split'] = 'test';(root/'s.json').write_text(json.dumps(s))
            result = self.run_cli('wise_ft_calibrate.py', args)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn(str(root), result.stdout)

    def test_cli_imports_no_torch_or_model_backend(self):
        result = subprocess.run([sys.executable, '-c',
                                 "import offline_calibration, raw_logit_contract, sys; assert 'torch' not in sys.modules; "
                                 "assert 'transformers' not in sys.modules; assert 'peft' not in sys.modules; "
                                 "assert 'intern_decision_reference' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
