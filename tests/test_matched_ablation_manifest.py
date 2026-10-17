"""Invented CPU fixtures only; successful drafts remain unapproved and UNRUN."""
import builtins
import collections
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import calibration_groups as cg
import matched_ablation_manifest as manifest
import oof_protocol as oof
import research_training as research
import synthetic_oof_adapter as adapter
from test_synthetic_oof_adapter import fixture as adapter_fixture, decode, encode, repin


def arm_copy(arm, arm_id, **changes):
    result = copy.deepcopy(arm)
    result['arm_id'] = arm_id
    for path, value in changes.items():
        section, key = path.split('.')
        result['config'][section][key] = value
    result['protocol']['config_sha256'] = research.config_hash(result['config'])
    for model in result['protocol']['models']:
        model['model_digest'] = oof.digest(['invented-output', arm_id, model['fold'], model['epoch']])
    return result


def contrasts(arms, comparison):
    baseline = next(a for a in arms if a['arm_id'] == comparison['baseline_arm_id'])
    result = []
    for arm in arms:
        if arm is baseline:
            continue
        fields = [path for path in manifest.FACTORS for section, key in [path.split('.')]
                  if arm['config'][section][key] != baseline['config'][section][key]]
        result.append({'arm_id': arm['arm_id'], 'changed_factors': fields,
                       'kind': 'single-factor' if len(fields) == 1 else 'combined'})
    comparison['contrasts'] = result


def fixture():
    payloads, config, protocol = adapter_fixture()
    documents = decode(payloads)
    # Uneven explicit component sizes and label/confidence mixture, all invented.
    for i in range(3):
        row = copy.deepcopy(documents['train.jsonl'][0])
        row['id'] = 'invented-extra-fast-' + str(i)
        row['prompt'] = 'Invented uneven-group fixture ' + str(i)
        row['label']['confidence'] = .25 + i * .2
        documents['train.jsonl'].append(row)
        entry = copy.deepcopy(protocol['root_audit']['entries'][0])
        entry['id'] = row['id']
        protocol['root_audit']['entries'].append(entry)
    documents['split-report.json']['splits']['train']['rows'] += 3
    payloads = encode(documents)
    config['training'].update(epochs=3, accumulation=8, seed=48271)
    config['sampler']['epoch_rows'] = 19
    protocol['epochs'] = [1, 2, 3]
    protocol['models'] = [{'fold': fold, 'epoch': epoch,
        'model_digest': oof.digest(['invented-base', fold, epoch]), 'source': 'synthetic-placeholder'}
        for fold in range(5) for epoch in (1, 2, 3)]
    # Uneven train component changes greedy outer allocation, so explicitly
    # supply a fresh proposal rather than asking product code to repair it.
    entries = {e['id']: e for e in protocol['root_audit']['entries']}
    rows = [{**row, 'source_split': source, 'root_id': entries[row['id']]['root_id']}
        for source in ('train', 'calibration') for row in documents[source + '.jsonl']]
    outer = oof.development_folds(rows, root_audit_ref='invented-uneven-proposal')
    protocol['nested_role_roots'] = [{'fit': f['outside_roots'][:-3],
        'A': f['outside_roots'][-3:-2], 'B': f['outside_roots'][-2:-1],
        'C': f['outside_roots'][-1:]} for f in outer['folds']]
    repin(payloads, config, protocol)
    base = arm_copy({'arm_id': 'baseline', 'config': config, 'protocol': protocol}, 'baseline')
    arms = [base, arm_copy(base, 'task-zero', **{'loss.auxiliary': 'task-type-zero'})]
    comparison = {'version': manifest.VERSION, 'status': 'draft', 'synthetic_fixture': True,
        'protocol_approved': False, 'baseline_arm_id': 'baseline', 'contrasts': [],
        'initialization': {'version': manifest.INITIALIZATION_VERSION, 'source': 'synthetic-placeholder',
            'base_model_id': 'invented-common-base', 'revision': 'invented-revision-v1',
            'initialization_sha256': oof.digest('invented-initialization'),
            'tokenizer_sha256': oof.digest('invented-tokenizer'),
            'scorer_request_sha256': oof.digest('invented-scorer-request')},
        'calibration_recipe': {'version': manifest.RECIPE_VERSION, 'status': 'draft',
            'source': 'synthetic-placeholder',
            'A': {'selection_rule_id': 'invented-A-selection-v1', 'tie_break_rule_id': 'invented-A-ties-v1'},
            'B': {'fit_rule_id': 'invented-adjust-then-T-v1', 'threshold_rule_id': 'invented-tau-selection-v1',
                  'tie_break_rule_id': 'invented-B-ties-v1',
                  'decision_prior': {'source': 'unique-training-before-oversampling', 'smoothing_pseudocount': 1.0}},
            'C': {'selection_rule_id': 'invented-C-selection-v1', 'tie_break_rule_id': 'invented-C-ties-v1'},
            'numeric_protocol': copy.deepcopy(cg.DEFAULT_PROTOCOL),
            'support_policy': copy.deepcopy(cg.DEFAULT_SUPPORT_POLICY),
            'fixed_model_required': True, 'fresh_calibration_required': True,
            'independent_confirmation_status': 'UNRUN'}}
    contrasts(arms, comparison)
    return payloads, arms, comparison


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.arms, self.comparison = fixture()

    def build(self):
        return manifest.build_synthetic_manifest(self.payloads, self.arms, self.comparison)

    def check(self, value):
        return manifest.validate_synthetic_manifest(self.payloads, self.arms, self.comparison, value)

    def reject(self):
        with self.assertRaises(manifest.ManifestError):
            self.build()

    def pin(self):
        for arm in self.arms:
            repin(self.payloads, arm['config'], arm['protocol'])

    def test_complete_rebuild_and_input_output_immutability(self):
        original = copy.deepcopy((self.payloads, self.arms, self.comparison))
        with mock.patch.object(adapter, 'build_synthetic_draft', wraps=adapter.build_synthetic_draft) as build:
            value = self.build()
        self.assertEqual(build.call_count, 2)
        self.assertEqual(value, self.build())
        self.assertEqual(original, (self.payloads, self.arms, self.comparison))
        self.assertEqual(self.check(value)['status'], 'synthetic-software-check-only')
        for output in (value, self.check(value)):
            for field in ('qualified', 'approval_verified', 'real_root_audit_verified', 'training_allowed'):
                self.assertIs(output[field], False)
            self.assertEqual(output['formal_admission'], 'NO-GO')
            self.assertEqual(output['real_execution_status'], 'UNRUN')
            self.assertIsNone(output['scientific_result'])
        self.assertEqual(value['status'], 'draft')
        value['comparison']['initialization']['base_model_id'] = 'changed-output'
        self.assertEqual(original, (self.payloads, self.arms, self.comparison))
        with self.assertRaises(manifest.ManifestError):
            self.check(value)

    def test_all_auxiliary_confidence_and_prior_choices_are_explicit(self):
        base = self.arms[0]
        self.arms += [arm_copy(base, 'literal-tier-only', **{'loss.auxiliary': 'tier-only'}),
            arm_copy(base, 'current-03', **{'loss.auxiliary': 'current-03'}),
            arm_copy(base, 'no-confidence', **{'loss.confidence_weighting': False}),
            arm_copy(base, 'expected-prior', **{'sampler.prior_source': 'expected-sampler'}),
            arm_copy(base, 'combined', **{'loss.auxiliary': 'tier-only',
                'loss.confidence_weighting': False, 'sampler.prior_source': 'expected-sampler'})]
        contrasts(self.arms, self.comparison)
        value = self.build()
        reports = {a['arm_id']: a for a in value['arms']}
        for arm in self.arms:
            self.assertEqual(reports[arm['arm_id']]['head_coefficients'], research.head_weights(arm['config']))
        self.assertEqual(reports['task-zero']['head_coefficients'], {'tier': 1., 'vision': .5, 'task_type': 0.})
        self.assertEqual(reports['literal-tier-only']['head_coefficients'], {'tier': 1., 'vision': 0., 'task_type': 0.})
        self.assertEqual(len({a['config_sha256'] for a in value['arms']}), len(self.arms))
        self.assertEqual(len({a['draft_sha256'] for a in value['arms']}), len(self.arms))
        self.assertEqual(len({a['protocol_sha256'] for a in value['arms']}), len(self.arms))
        prior_differences = 0
        for fold in range(5):
            original = reports['baseline']['fold_budgets'][fold]
            prior = reports['expected-prior']['fold_budgets'][fold]
            prior_differences += original['training_priors'] != prior['training_priors']
            for arm in value['arms']:
                current = arm['fold_budgets'][fold]
                self.assertEqual(original['probability_sha256'], current['probability_sha256'])
                for left, right in zip(original['epochs'], current['epochs']):
                    self.assertEqual(left['ordered_stream_sha256'], right['ordered_stream_sha256'])
                    self.assertEqual(left['achieved_row_counts_sha256'], right['achieved_row_counts_sha256'])
            off = reports['no-confidence']['fold_budgets'][fold]['epochs'][0]
            self.assertEqual(off['confidence_weighted_exposure'], 19)
            self.assertLess(original['epochs'][0]['confidence_weighted_exposure'], 19)
            self.assertEqual(reports['task-zero']['fold_budgets'][fold]['epochs'][0]['head_weighted_exposure']['task_type'], 0)
        self.assertGreater(prior_differences, 0)

    def test_per_epoch_tail_ceil_and_unknown_actual_resources(self):
        value = self.build()
        # Three epochs of 19 rows, accumulation 8: 3+3+3, never ceil(57/8)=8.
        self.assertEqual(value['training_controls'], {'epochs': 3, 'accumulation': 8, 'seed': 48271})
        for arm in value['arms']:
            self.assertEqual(arm['planned_row_exposure'], 5 * 3 * 19)
            self.assertEqual(arm['planned_optimizer_updates'], 5 * 3 * 3)
            for fold in arm['fold_budgets']:
                self.assertEqual(fold['planned_optimizer_updates'], 9)
                self.assertNotEqual(fold['planned_optimizer_updates'], math.ceil(57 / 8))
                self.assertTrue(all(e['final_update_rows'] == 3 for e in fold['epochs']))
            self.assertEqual(arm['measurement']['status'], 'UNRUN')
            self.assertTrue(all(v is None for k, v in arm['measurement'].items() if k != 'status'))

    def test_budget_divisor_boundaries_match_existing_epoch_formula(self):
        for accumulation in (1, 8, 19, 20):
            with self.subTest(accumulation=accumulation):
                self.setUp()
                for arm in self.arms: arm['config']['training']['accumulation'] = accumulation
                self.pin()
                value = self.build()
                for arm in value['arms']:
                    for fold in arm['fold_budgets']:
                        for epoch in fold['epochs']:
                            self.assertEqual(epoch['planned_optimizer_updates'], -(-19 // accumulation))
                            self.assertEqual(epoch['final_update_rows'], (19 - 1) % accumulation + 1)

    def test_numeric_cost_controls_reject_bool_even_in_all_arms(self):
        for value, column in ((False, 0), (True, 1)):
            self.setUp()
            for arm in self.arms: arm['config']['loss']['cost_matrix'][0][column] = value
            self.pin(); self.reject()

    def test_import_is_stdlib_only_without_framework_loading(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import matched_ablation_manifest; "
            "assert not {'torch', 'transformers', 'peft', 'train_eval', 'server'} & set(sys.modules)"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_actual_nested_fit_rows_reach_sampler_in_snapshot_order(self):
        expected = adapter.build_synthetic_draft(self.payloads, self.arms[0]['config'], self.arms[0]['protocol'])
        docs = decode(self.payloads)
        ordered = docs['train.jsonl'] + docs['calibration.jsonl']
        calls = []
        real = research.sample_epochs
        def sampling(rows, config, seed, epochs):
            fold = expected['oof_plan']['folds'][len(calls) % 5]
            wanted = set(fold['role_ids']['fit'])
            self.assertEqual([r['id'] for r in rows], [r['id'] for r in ordered if r['id'] in wanted])
            for role in ('A', 'B', 'C'):
                self.assertFalse(wanted.intersection(fold['role_ids'][role]))
            self.assertFalse(wanted.intersection(fold['validation_ids']))
            self.assertEqual(seed, 48271)  # No hidden fold offsets.
            self.assertEqual(epochs, 3)
            calls.append(rows)
            return real(rows, config, seed, epochs)
        with mock.patch.object(research, 'sample_epochs', side_effect=sampling):
            self.build()
        self.assertEqual(len(calls), 10)

    def test_complete_stream_comparison_refuses_order_only_change(self):
        real = research.sample_epochs
        def reverse(rows, config, seed, epochs):
            streams, plan = real(rows, config, seed, epochs)
            if config['loss']['auxiliary'] == 'task-type-zero':
                streams[0] = list(reversed(streams[0]))
            return streams, plan
        with mock.patch.object(research, 'sample_epochs', side_effect=reverse):
            self.reject()

    def test_equal_pooled_exposure_different_epoch_tails_refused(self):
        real = research.sample_epochs
        def redistribute(rows, config, seed, epochs):
            streams, plan = real(rows, config, seed, epochs)
            if config['loss']['auxiliary'] == 'task-type-zero':
                streams[1] = streams[0][-3:] + streams[1]
                streams[0] = streams[0][:-3]  # 16,22,19, same pooled exposure.
            return streams, plan
        with mock.patch.object(research, 'sample_epochs', side_effect=redistribute):
            self.reject()

    def test_changed_probabilities_rejected_even_with_same_streams(self):
        real = research.sample_epochs
        def mutate(rows, config, seed, epochs):
            streams, plan = real(rows, config, seed, epochs)
            if config['loss']['auxiliary'] == 'task-type-zero':
                plan['target_row_probabilities'][0] += .001
                plan['target_row_probabilities'][1] -= .001
            return streams, plan
        with mock.patch.object(research, 'sample_epochs', side_effect=mutate):
            self.reject()

    def test_every_arm_must_pass_adapter_before_sampling(self):
        del self.arms[1]['protocol']['root_audit']
        with mock.patch.object(research, 'sample_epochs', side_effect=AssertionError('sampling before validation')):
            self.reject()

    def test_arm_model_identity_registry_and_epoch_rejections(self):
        for change in ('copy', 'missing', 'duplicate', 'wrong-fold', 'wrong-epoch', 'wrong-source', 'bad-digest', 'bool', 'reorder-epochs'):
            with self.subTest(change=change):
                self.setUp()
                p = self.arms[1]['protocol']
                if change == 'copy': p['models'] = copy.deepcopy(self.arms[0]['protocol']['models'])
                if change == 'missing': p['models'].pop()
                if change == 'duplicate': p['models'].append(copy.deepcopy(p['models'][0]))
                if change == 'wrong-fold': p['models'][0]['fold'] = 9
                if change == 'wrong-epoch': p['models'][0]['epoch'] = 9
                if change == 'wrong-source': p['models'][0]['source'] = 'real-trained'
                if change == 'bad-digest': p['models'][0]['model_digest'] = 'not-a-hash'
                if change == 'bool': p['models'][0]['fold'] = False
                if change == 'reorder-epochs': p['epochs'].reverse()
                self.reject()

    def test_model_order_is_normalized_but_binding_preserved(self):
        self.arms[1]['protocol']['models'].reverse()
        value = self.build()
        self.assertEqual([(m['fold'], m['epoch']) for m in value['arms'][1]['models']],
                         [(f, e) for f in range(5) for e in (1, 2, 3)])
        self.assertTrue(all(m['arm_id'] == 'task-zero' for m in value['arms'][1]['models']))

    def test_single_and_combined_axes_exactness(self):
        attacks = [lambda c: c['contrasts'][0].update(changed_factors=[]),
            lambda c: c['contrasts'][0].update(changed_factors=['loss.auxiliary', 'loss.auxiliary']),
            lambda c: c['contrasts'][0].update(changed_factors=['loss.confidence_weighting']),
            lambda c: c['contrasts'][0].update(changed_factors=['training.seed']),
            lambda c: c['contrasts'][0].update(kind='combined'),
            lambda c: c.update(baseline_arm_id='foreign'),
            lambda c: c['contrasts'].append(copy.deepcopy(c['contrasts'][0])),
            lambda c: c['contrasts'][0].update(arm_id='baseline')]
        for attack in attacks:
            with self.subTest(attack=attack):
                self.setUp(); attack(self.comparison); self.reject()
        self.setUp()
        self.arms[1]['config']['loss']['confidence_weighting'] = False
        self.pin(); self.reject()
        contrasts(self.arms, self.comparison)
        self.build()

    def test_every_nonfactor_config_knob_is_fixed(self):
        edits = [('training', 'seed', 11), ('training', 'accumulation', 7),
            ('training', 'lr', .0002), ('training', 'optimizer', 'adamw8bit'),
            ('training', 'beta', .9), ('training', 'label_smoothing', .1),
            ('sampler', 'epoch_rows', 20), ('sampler', 'prior_smoothing', 2.),
            ('sampler', 'synthetic_balanced_vision_multiplier', 2.),
            ('loss', 'tier_tau', 2.), ('loss', 'vision_tau', .8), ('loss', 'cost_lambda', .5),
            ('capacity', 'mode', 'lora'), ('capacity', 'neftune_alpha', 6.)]
        for section, key, value in edits:
            with self.subTest(section=section, key=key):
                self.setUp(); self.arms[1]['config'][section][key] = value; self.pin(); self.reject()
        for key in ('data_version', 'evaluation_protocol'):
            self.setUp(); self.arms[1]['config'][key] += '-changed'; self.pin(); self.reject()

    def test_role_root_fold_epoch_and_audit_changes_are_not_factors(self):
        for change in ('audit-ref', 'swap-roles', 'epochs', 'min-support'):
            with self.subTest(change=change):
                self.setUp(); p = self.arms[1]['protocol']
                if change == 'audit-ref': p['root_audit']['evidence_ref'] += '-new'
                if change == 'swap-roles':
                    r = p['nested_role_roots'][0]; r['A'], r['C'] = r['C'], r['A']
                if change == 'epochs':
                    self.arms[1]['config']['training']['epochs'] = 2
                    p['epochs'] = [1, 2]; p['models'] = [m for m in p['models'] if m['epoch'] < 3]
                if change == 'min-support': p['min_class_roots'] = 2
                self.pin(); self.reject()

    def test_five_exact_bytes_no_paths_extra_inputs_or_repairs(self):
        for name in research.INPUT_FILES:
            with self.subTest(name=name):
                self.setUp(); self.payloads[name] += b' '; self.reject()
                self.setUp(); self.payloads[name] = '/private/never-open'; self.reject()
                self.setUp(); self.payloads[name] = bytearray(self.payloads[name]); self.reject()
                self.setUp(); del self.payloads[name]; self.reject()
        self.setUp(); self.payloads['test.jsonl'] = b'never parse'; self.reject()

    def test_original_manifest_rejected_after_complete_semantic_repin(self):
        for change in ('bytes', 'order', 'seed', 'model', 'root-name', 'roles', 'epoch'):
            with self.subTest(change=change):
                self.setUp(); value = self.build()
                if change == 'bytes': self.payloads['roles.json'] += b'\n'
                if change == 'order':
                    docs = decode(self.payloads); docs['train.jsonl'].reverse(); self.payloads = encode(docs)
                if change == 'seed':
                    for arm in self.arms: arm['config']['training']['seed'] += 1
                if change == 'model': self.arms[1]['protocol']['models'][0]['model_digest'] = 'f' * 64
                if change == 'root-name':
                    old = self.arms[0]['protocol']['root_audit']['entries'][0]['root_id']
                    for arm in self.arms:
                        p = arm['protocol']
                        for entry in p['root_audit']['entries']:
                            if entry['root_id'] == old: entry['root_id'] = old + '-renamed'
                        for roles in p['nested_role_roots']:
                            for role, roots in roles.items(): roles[role] = [r + '-renamed' if r == old else r for r in roots]
                if change == 'roles':
                    docs = decode(self.payloads); r = docs['roles.json']['role_groups']; r['A'], r['C'] = r['C'], r['A']; self.payloads = encode(docs)
                if change == 'epoch':
                    for arm in self.arms:
                        arm['config']['training']['epochs'] = 2; arm['protocol']['epochs'] = [1, 2]
                        arm['protocol']['models'] = [m for m in arm['protocol']['models'] if m['epoch'] < 3]
                self.pin()
                # A changed root name can change deterministic fold tie order;
                # fail-closed is also valid if the old proposal becomes infeasible.
                try: changed = self.build()
                except manifest.ManifestError:
                    if change != 'root-name': raise
                else: self.assertNotEqual(value['manifest_sha256'], changed['manifest_sha256'])
                with self.assertRaises(manifest.ManifestError): self.check(value)

    def test_common_initialization_tokenizer_scoring_and_recipe_rebinding(self):
        changes = [(('initialization', 'base_model_id'), 'new-invented-base'),
            (('initialization', 'revision'), 'new-invented-revision'),
            (('initialization', 'initialization_sha256'), 'e' * 64),
            (('initialization', 'tokenizer_sha256'), 'e' * 64),
            (('initialization', 'scorer_request_sha256'), 'e' * 64),
            (('calibration_recipe', 'A', 'selection_rule_id'), 'new-A-rule'),
            (('calibration_recipe', 'B', 'threshold_rule_id'), 'new-B-rule'),
            (('calibration_recipe', 'C', 'tie_break_rule_id'), 'new-C-tie'),
            (('calibration_recipe', 'B', 'decision_prior', 'smoothing_pseudocount'), 2.)]
        for path, replacement in changes:
            with self.subTest(path=path):
                self.setUp(); value = self.build(); target = self.comparison
                for key in path[:-1]: target = target[key]
                target[path[-1]] = replacement
                self.assertNotEqual(value['manifest_sha256'], self.build()['manifest_sha256'])
                with self.assertRaises(manifest.ManifestError): self.check(value)

    def test_original_calibration_protocol_expansion_is_explicit(self):
        docs = decode(self.payloads)
        docs['roles.json']['protocol'] = {'tau_grid': [0., 1.], 'fit_max_iterations': 23}
        docs['roles.json']['support_policy'] = {'n_folds': 2}
        self.payloads = encode(docs); self.pin(); self.reject()
        self.comparison['calibration_recipe']['numeric_protocol'].update(docs['roles.json']['protocol'])
        self.comparison['calibration_recipe']['support_policy'].update(docs['roles.json']['support_policy'])
        self.build()
        self.comparison['calibration_recipe']['numeric_protocol']['fit_max_iterations'] = True
        self.reject()

    def test_final43_strict_roles_override_guards_are_inherited(self):
        for section in ('protocol', 'support_policy'):
            for key in ('qualified', 'approval_verified', 'approval_ref', 'fit', 'score', 'selector', 'future_field'):
                with self.subTest(section=section, key=key):
                    self.setUp(); docs = decode(self.payloads)
                    docs['roles.json'][section] = {key: True}; self.payloads = encode(docs); self.pin(); self.reject()

    def test_final43_original_calibration_component_collapse_is_inherited(self):
        for role in ('A', 'B', 'C'):
            with self.subTest(role=role):
                self.setUp()
                docs = decode(self.payloads)
                components = docs['roles.json']['role_groups'][role][:2]
                for arm in self.arms:
                    entries = arm['protocol']['root_audit']['entries']
                    target = next(e['root_id'] for e in entries if e['split_group'] == components[0])
                    for entry in entries:
                        if entry['split_group'] == components[1]: entry['root_id'] = target
                # Root guard runs before nested-plan work; no new support policy.
                with mock.patch.object(oof, 'nested_plan', side_effect=AssertionError('root collapse escaped')):
                    self.reject()

    def test_root_source_leakage_missing_support_and_class_order_fail_closed(self):
        for change in ('audit-missing', 'source', 'role-overlap', 'outer-leak', 'support', 'classes'):
            with self.subTest(change=change):
                self.setUp(); p = self.arms[0]['protocol']
                if change == 'audit-missing': p['root_audit']['entries'].pop()
                if change == 'source': p['root_audit']['entries'][0]['source_split'] = 'calibration'
                if change == 'role-overlap': p['nested_role_roots'][0]['A'] = p['nested_role_roots'][0]['B']
                if change == 'outer-leak':
                    d = adapter.build_synthetic_draft(self.payloads, self.arms[0]['config'], p)
                    p['nested_role_roots'][0]['fit'].append(d['oof_plan']['folds'][0]['validation_roots'][0])
                if change == 'support': p['min_class_roots'] = 100
                if change == 'classes': self.arms[0]['config']['label_schema']['classes']['task_type'].reverse(); self.pin()
                self.reject()

    def test_confidence_off_does_not_launder_expert_provenance(self):
        for key, bad in [('label_source', 'measured'), ('evidence', 'execution'),
            ('confidence', True), ('confidence', 2.), ('label_status', 'pending'),
            ('task_type', 'visual'), ('confidence_semantics', 'coverage')]:
            with self.subTest(key=key):
                self.setUp(); docs = decode(self.payloads)
                docs['train.jsonl'][0]['label'][key] = bad
                self.payloads = encode(docs)
                for arm in self.arms: arm['config']['loss']['confidence_weighting'] = False
                self.pin(); self.reject()

    def test_all_unknown_nested_fields_and_approval_attempts_rejected(self):
        paths = [(), ('initialization',), ('calibration_recipe',), ('calibration_recipe', 'A'),
            ('calibration_recipe', 'B'), ('calibration_recipe', 'B', 'decision_prior'),
            ('calibration_recipe', 'C'), ('calibration_recipe', 'numeric_protocol'),
            ('calibration_recipe', 'support_policy'),
            ('calibration_recipe', 'support_policy', 'min_groups_per_role')]
        for path in paths:
            with self.subTest(path=path):
                self.setUp(); obj = self.comparison
                for key in path: obj = obj[key]
                obj['future_ignored_field'] = 'approval'; self.reject()
        for key, val in [('status', 'approved'), ('synthetic_fixture', 1), ('protocol_approved', 0)]:
            self.setUp(); self.comparison[key] = val; self.reject()
        for key, val in [('fixed_model_required', False), ('fresh_calibration_required', 1),
                         ('independent_confirmation_status', 'PASS')]:
            self.setUp(); self.comparison['calibration_recipe'][key] = val; self.reject()
        for field in ('config', 'protocol'):
            self.setUp(); self.arms[1][field]['qualified'] = True; self.reject()
        self.setUp(); self.arms[1]['calibration_recipe'] = self.comparison['calibration_recipe']; self.reject()

    def test_missing_types_nonfinite_unicode_and_unsupported_modes(self):
        for bad in (None, True, 1, [], 'path'):
            self.setUp(); self.comparison['initialization'] = bad; self.reject()
        for bad in (float('nan'), float('inf'), '\ud800', object(), ('tuple',), {1: 'coerced'}, 10 ** 400):
            self.setUp(); self.comparison['initialization']['revision'] = bad; self.reject()
        for bad in (True, '1', -1):
            self.setUp(); self.comparison['calibration_recipe']['B']['decision_prior']['smoothing_pseudocount'] = bad; self.reject()
        for bad in ('independent-head', 'two-stage'):
            self.setUp(); self.arms[1]['config']['loss']['auxiliary'] = bad; self.pin(); self.reject()
        self.setUp(); self.arms[1]['arm_id'] = 'baseline'; self.reject()
        self.setUp(); del self.comparison['calibration_recipe']['A']; self.reject()
        self.setUp(); self.arms = self.arms[:1]; self.reject()

    def test_strict_original_json_is_not_silently_reparsed(self):
        for payload in (b'{"role_groups":{},"role_groups":{}}', b'{"bad":NaN}',
                        b'{"bad":"\\ud800"}', b'\xff', b'null'):
            self.setUp(); self.payloads['roles.json'] = payload; self.pin(); self.reject()

    def test_retained_copies_survive_caller_mapping_and_structure_swaps(self):
        value = self.build(); real = adapter.build_synthetic_draft
        def swap(payloads, config, protocol):
            self.payloads['train.jsonl'] = b'invalid changed caller mapping'
            self.arms[1]['config']['training']['seed'] = -111
            self.comparison['initialization']['revision'] = 'changed-original'
            return real(payloads, config, protocol)
        with mock.patch.object(adapter, 'build_synthetic_draft', side_effect=swap):
            self.assertEqual(value, self.build())
        self.reject()

    def test_output_tampering_self_rehash_and_measured_claims_refused(self):
        value = self.build()
        attacks = [lambda v: v.update(qualified=True), lambda v: v.update(approval_verified=True),
            lambda v: v.update(real_root_audit_verified=True), lambda v: v.update(training_allowed=True),
            lambda v: v.update(formal_admission='GO'), lambda v: v.update(status='approved'),
            lambda v: v.update(approval_ref='invented-human'),
            lambda v: v['arms'][0]['head_coefficients'].update(vision=0),
            lambda v: v['arms'][0]['measurement'].update(actual_trainable_parameters=123),
            lambda v: v['arms'][0]['measurement'].update(wall_seconds=1),
            lambda v: v['arms'][0]['measurement'].update(monetary_cost=0),
            lambda v: v['arms'][0].update(planned_optimizer_updates=8),
            lambda v: v['oof_binding']['folds'][0].update(actual_training_root_count=999)]
        for attack in attacks:
            with self.subTest(attack=attack):
                changed = copy.deepcopy(value); attack(changed)
                changed['manifest_sha256'] = manifest._digest({k: v for k, v in changed.items() if k != 'manifest_sha256'})
                with self.assertRaises(manifest.ManifestError): self.check(changed)

    def test_envelope_is_not_runnable_and_has_no_row_ids_or_labels(self):
        value = self.build()
        with self.assertRaises(ValueError): research.validate_config(value)
        with self.assertRaises(ValueError): cg.check_roles_config(value)
        text = json.dumps(value)
        for token in ('invented-extra-fast-', 'invented-train-', 'Invented fixture', 'invented-audit-root-', '"prompt":', '"label":', 'train_command'):
            self.assertNotIn(token, text)
        self.assertNotIn('config', value['arms'][0])
        self.assertNotIn('protocol', value['arms'][0])

    def test_no_file_network_model_training_or_calibration_effects(self):
        value = self.build(); real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft', 'train_eval', 'server', 'requests'}:
                raise AssertionError('forbidden framework/model import')
            return real_import(name, *args, **kwargs)
        with mock.patch.object(builtins, 'open', side_effect=AssertionError('file open')), \
                mock.patch.object(io, 'open', side_effect=AssertionError('file open')), \
                mock.patch.object(Path, 'glob', side_effect=AssertionError('discovery')), \
                mock.patch.object(Path, 'rglob', side_effect=AssertionError('discovery')), \
                mock.patch.object(socket, 'socket', side_effect=AssertionError('network')), \
                mock.patch.object(socket, 'create_connection', side_effect=AssertionError('network')), \
                mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('process')), \
                mock.patch.object(research, 'load_v2_inputs', side_effect=AssertionError('real loader')), \
                mock.patch.object(research, 'row_loss', side_effect=AssertionError('training')), \
                mock.patch.object(cg, 'fit_temperature', side_effect=AssertionError('fit')), \
                mock.patch.object(cg, 'select_tau_oof', side_effect=AssertionError('select')), \
                mock.patch.object(builtins, '__import__', side_effect=guarded_import):
            self.assertEqual(value, self.build())
            self.assertEqual(self.check(value)['formal_admission'], 'NO-GO')


if __name__ == '__main__':
    unittest.main()
