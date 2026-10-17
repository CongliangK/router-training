"""Invented-only adapter tests. No historical row counts, private data or weights."""
import builtins
import copy
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calibration_groups as cg
import model_utils
import oof_protocol as oof
import research_training as research
import synthetic_oof_adapter as adapter
import task_label_schema as schema


def fixture():
    """A tiny invented software layout; its counts satisfy no scientific quota."""
    data = {'train': [], 'calibration': []}
    entries = []
    for source, count in (('train', 5), ('calibration', 16)):
        for group in range(count):
            for i in range(6):
                row = {'id': f'invented-{source}-{group}-{i}',
                    'prompt': f'Invented fixture {source} {group} {i}',
                    'group': f'invented-original-{source}-{group}',
                    'split_group': f'invented-component-{source}-{group}',
                    'task_label_schema_version': schema.V2, 'label_schema_version': 1,
                    'label': {'tier': research.TIERS[i % 3], 'needs_vision': research.VISION[i % 2],
                        'task_type': schema.SIX_TYPES[i], 'label_source': 'expert_provisional',
                        'label_status': 'resolved', 'evidence': 'prompt_only', 'confidence': .8,
                        'task_label_schema_version': schema.V2, 'label_schema_version': 1}}
                data[source].append(row)
                entries.append({k: row[k] for k in ('id', 'group', 'split_group')} |
                    {'source_split': source, 'root_id': f'invented-audit-root-{source}-{group}'})
    groups = sorted({r['split_group'] for r in data['calibration']})
    documents = {'train.jsonl': data['train'], 'calibration.jsonl': data['calibration'],
        'label-schema.json': schema.metadata(schema.V2),
        'split-report.json': {'label_schema': schema.metadata(schema.V2),
            'task_label_schema_version': schema.V2, 'label_schema_version': 1,
            'splits': {source: {'rows': len(rows)} for source, rows in data.items()}},
        'roles.json': {'role_groups': {'A': groups[:5], 'B': groups[5:11], 'C': groups[11:]}}}
    config = json.loads((ROOT/'configs/research-v2.json').read_text())
    payloads = encode(documents)
    config['input_sha256'] = fingerprints(payloads)
    protocol = {'version': adapter.VERSION, 'status': 'draft', 'synthetic_fixture': True,
        'protocol_approved': False, 'config_sha256': research.config_hash(config),
        'root_audit': {'version': adapter.AUDIT_VERSION, 'status': 'synthetic-unverified',
            'evidence_ref': 'invented-audit-declaration-not-real-evidence',
            'input_sha256': fingerprints(payloads), 'entries': entries},
        'n_folds': 5, 'min_class_roots': 1, 'nested_role_roots': [], 'epochs': [1, 2], 'models': []}
    metadata = [{**{k: v for k, v in row.items() if k != 'prompt'},
                 'source_split': source, 'root_id': entry['root_id']}
                for source, rows in data.items() for row in rows
                for entry in entries if entry['id'] == row['id']]
    outer = oof.development_folds(metadata, root_audit_ref='invented-fixture-layout')
    protocol['nested_role_roots'] = [{'fit': f['outside_roots'][:-3],
        'A': f['outside_roots'][-3:-2], 'B': f['outside_roots'][-2:-1],
        'C': f['outside_roots'][-1:]} for f in outer['folds']]
    protocol['models'] = [{'fold': fold, 'epoch': epoch,
        'model_digest': oof.digest(['invented-not-model-bytes', fold, epoch]),
        'source': 'synthetic-placeholder'} for fold in range(5) for epoch in (1, 2)]
    return payloads, config, protocol


def encode(documents):
    return {name: (''.join(json.dumps(r) + '\n' for r in value) if name.endswith('.jsonl')
                   else json.dumps(value)).encode() for name, value in documents.items()}


def fingerprints(payloads):
    return {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}


def decode(payloads):
    return {name: [json.loads(line) for line in value.splitlines()] if name.endswith('.jsonl')
            else json.loads(value) for name, value in payloads.items()}


def repin(payloads, config, protocol):
    config['input_sha256'] = fingerprints(payloads)
    protocol['config_sha256'] = research.config_hash(config)
    protocol['root_audit']['input_sha256'] = fingerprints(payloads)


def predictions(draft):
    plan = draft['oof_plan']
    rows = {r['id']: r for r in plan['rows']}
    models = {(m['fold'], m['epoch']): m['model_digest'] for m in draft['models']}
    return [{'id': rid, 'source_split': rows[rid]['source_split'], 'root_id': rows[rid]['root_id'],
        'fold': fold['fold'], 'epoch': epoch, 'plan_digest': plan['plan_digest'],
        'draft_sha256': draft['draft_sha256'], 'model_digest': models[fold['fold'], epoch],
        'training_roots': list(fold['role_roots']['fit']), 'task_label_schema_version': schema.V2,
        'logits': {'tier': [0., .5, -.5], 'vision': [.25, -.25], 'task_type': [0.] * 6}}
        for fold in plan['folds'] for rid in fold['validation_ids'] for epoch in draft['epochs']]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.config, self.protocol = fixture()

    def build(self):
        return adapter.build_synthetic_draft(self.payloads, self.config, self.protocol)

    def reject(self):
        with self.assertRaises(adapter.AdapterError):
            self.build()

    def mutate_document(self, name, change, *, pin=True):
        documents = decode(self.payloads)
        change(documents[name])
        self.payloads = encode(documents)
        if pin:
            repin(self.payloads, self.config, self.protocol)

    def test_complete_mapping_no_approval_and_no_input_mutation(self):
        original = copy.deepcopy((self.payloads, self.config, self.protocol))
        draft = self.build()
        self.assertEqual(draft, self.build())
        self.assertEqual(original, (self.payloads, self.config, self.protocol))
        self.assertEqual(draft['status'], 'draft')
        self.assertEqual(draft['formal_admission'], 'NO-GO')
        for key in ('qualified', 'approval_verified', 'real_root_audit_verified', 'training_allowed'):
            self.assertIs(draft[key], False)
        for row in draft['oof_plan']['rows']:
            self.assertNotEqual(row['root_id'], row['group'])
            self.assertNotEqual(row['root_id'], row['split_group'])
            self.assertNotIn('prompt', row)
            self.assertEqual(row['label']['label_source'], 'expert_provisional')
        checked = adapter.validate_synthetic_records(self.payloads, self.config, self.protocol,
                                                     draft, predictions(draft))
        self.assertEqual(checked['row_epoch_count'], 252)
        self.assertIsNone(checked['scientific_metrics'])
        self.assertIsNone(checked['independent_calibration_units'])
        self.assertFalse(checked['qualified'])
        # Returned mutable values cannot alter the original arguments.
        draft['models'][0]['model_digest'] = 'f' * 64
        self.assertEqual(original, (self.payloads, self.config, self.protocol))

    def test_formal_and_unknown_protocol_fields_fail_closed(self):
        for field, value in [('version', 'approved-oof-v1'), ('status', 'approved'),
                             ('synthetic_fixture', False), ('synthetic_fixture', 1),
                             ('protocol_approved', True), ('protocol_approved', 0),
                             ('approval_ref', 'human-approved')]:
            with self.subTest(field=field, value=value):
                _, _, self.protocol = fixture()
                self.protocol[field] = value
                self.reject()
        _, _, self.protocol = fixture()
        del self.protocol['root_audit']
        self.reject()

    def test_draft_is_not_training_or_original_roles_config(self):
        draft = self.build()
        with self.assertRaises(ValueError):
            research.validate_config(draft)
        with self.assertRaises(ValueError):
            cg.check_roles_config(draft)
        for claimed in ('qualified', 'approval_verified', 'real_root_audit_verified'):
            changed = copy.deepcopy(draft)
            changed[claimed] = True
            changed['draft_sha256'] = oof.digest({k: v for k, v in changed.items() if k != 'draft_sha256'})
            with self.assertRaises(adapter.AdapterError):
                adapter.validate_synthetic_records(self.payloads, self.config, self.protocol,
                                                   changed, predictions(draft))

    def test_exact_five_memory_bytes_only(self):
        for name in adapter.INPUT_NAMES:
            with self.subTest(name=name):
                payloads = dict(self.payloads)
                del payloads[name]
                with self.assertRaises(adapter.AdapterError):
                    adapter.build_synthetic_draft(payloads, self.config, self.protocol)
                payloads = dict(self.payloads)
                payloads[name] = '/private/test.jsonl'
                with self.assertRaises(adapter.AdapterError):
                    adapter.build_synthetic_draft(payloads, self.config, self.protocol)
        self.payloads['test.jsonl'] = b'never parsed'
        self.reject()

    def test_each_input_byte_change_is_rejected_without_repins(self):
        for name in adapter.INPUT_NAMES:
            payloads = dict(self.payloads)
            payloads[name] += b' '
            with self.subTest(name=name), self.assertRaises(adapter.AdapterError):
                adapter.build_synthetic_draft(payloads, self.config, self.protocol)

    def test_hashing_and_parsing_retain_one_byte_snapshot(self):
        expected = self.build()
        original_parse = adapter._parse
        def mutate_caller_mapping(payload):
            self.payloads['calibration.jsonl'] = b'INVALID REPLACEMENT AFTER HASHING'
            self.payloads['roles.json'] = b'INVALID REPLACEMENT AFTER HASHING'
            return original_parse(payload)
        with mock.patch.object(adapter, '_parse', side_effect=mutate_caller_mapping):
            self.assertEqual(expected, self.build())
        self.reject()  # The next call observes the changed mapping and refuses it.

    def test_config_and_root_audit_each_bind_the_snapshot(self):
        self.config['training']['seed'] += 1
        self.reject()
        self.protocol['config_sha256'] = research.config_hash(self.config)
        self.build()  # A new draft is permitted, never an approved configuration.
        self.payloads['roles.json'] += b'\n'
        self.config['input_sha256'] = fingerprints(self.payloads)
        self.protocol['config_sha256'] = research.config_hash(self.config)
        self.reject()  # Audit still binds old original-role bytes.

    def test_strict_json_duplicates_nonfinite_encoding_and_unknown_keys(self):
        for value in (b'{"role_groups":{},"role_groups":{}}', b'{"x":NaN}',
                      b'{"x":Infinity}', b'{"x":"\\ud800"}', b'\xff', b'null'):
            with self.subTest(value=value):
                self.payloads, self.config, self.protocol = fixture()
                self.payloads['roles.json'] = value
                repin(self.payloads, self.config, self.protocol)
                self.reject()
        for name, change in [('train.jsonl', lambda rows: rows[0].update(unknown='field')),
                             ('train.jsonl', lambda rows: rows[0]['label'].update(unknown='field')),
                             ('split-report.json', lambda report: report.update(unknown='field')),
                             ('roles.json', lambda roles: roles.update(unknown='field'))]:
            self.payloads, self.config, self.protocol = fixture()
            self.mutate_document(name, change)
            self.reject()

    def test_schema_class_order_versions_and_label_sources(self):
        changes = [lambda rows: rows[0].update(task_label_schema_version=schema.LEGACY),
            lambda rows: rows[0].update(label_schema_version=True),
            lambda rows: rows[0]['label'].update(task_type='visual'),
            lambda rows: rows[0]['label'].update(needs_vision='uncertain'),
            lambda rows: rows[0]['label'].update(label_source='unknown'),
            lambda rows: rows[0]['label'].update(label_source='measured_candidate_outcomes'),
            lambda rows: rows[0]['label'].update(evidence='model_execution'),
            lambda rows: rows[0]['label'].update(label_status='pending'),
            lambda rows: rows[0]['label'].update(confidence=True),
            lambda rows: rows[0]['label'].update(id='foreign'),
            lambda rows: rows[0]['label'].update(confidence_semantics='interval'),
            lambda rows: rows[0].update(synthetic=True),
            lambda rows: rows[0]['label'].update(supplemental_only=True),
            lambda rows: rows[0].update(source_split='test')]
        for change in changes:
            self.payloads, self.config, self.protocol = fixture()
            self.mutate_document('train.jsonl', change)
            self.reject()
        self.payloads, self.config, self.protocol = fixture()
        self.mutate_document('label-schema.json', lambda meta: meta['classes']['task_type'].reverse())
        self.reject()

    def test_unknown_source_still_rejects_with_confidence_disabled(self):
        self.config['loss']['confidence_weighting'] = False
        self.mutate_document('train.jsonl', lambda rows: rows[0]['label'].update(label_source='unknown'))
        self.reject()

    def test_duplicate_ids_and_original_source_component_isolation(self):
        changes = [lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows[1].update(id=rows[0]['id']),
            lambda rows: rows[1].update(split_group='fragmented-component'),
            lambda rows: rows[0].update(group='invented-original-calibration-0'),
            lambda rows: rows[0].update(split_group='invented-component-calibration-0'),
            lambda rows: rows[0].update(id='invented-calibration-0-0')]
        for change in changes:
            self.payloads, self.config, self.protocol = fixture()
            self.mutate_document('train.jsonl', change)
            self.reject()

    def test_root_audit_missing_foreign_duplicate_and_source_binding(self):
        mutations = [lambda a: a.update(evidence_ref=''), lambda a: a.update(status='approved'),
            lambda a: a.update(entries=[]), lambda a: a['entries'].pop(),
            lambda a: a['entries'].append(copy.deepcopy(a['entries'][0])),
            lambda a: a['entries'][0].update(id='foreign'),
            lambda a: a['entries'][0].pop('root_id'),
            lambda a: a['entries'][0].update(root_id=''),
            lambda a: a['entries'][0].update(source_split='test'),
            lambda a: a['entries'][0].update(source_split='calibration'),
            lambda a: a['entries'][0].update(group='unbound'),
            lambda a: a['entries'][0].update(split_group='unbound'),
            lambda a: a['entries'][0].update(root_id='split-one-component'),
            lambda a: a['entries'][0].update(unknown=True)]
        for mutate in mutations:
            self.payloads, self.config, self.protocol = fixture()
            mutate(self.protocol['root_audit'])
            self.reject()
        self.payloads, self.config, self.protocol = fixture()
        for entry in self.protocol['root_audit']['entries']:
            if entry['source_split'] == 'train' and entry['group'].endswith('-0'):
                entry['root_id'] = 'invented-audit-root-calibration-0'
        self.reject()

    def test_original_roles_not_reinterpreted_as_nested_roles(self):
        self.mutate_document('roles.json', lambda roles: roles['role_groups']['B'].append(
            roles['role_groups']['A'][0]))
        self.reject()
        self.payloads, self.config, self.protocol = fixture()
        self.protocol['nested_role_roots'] = decode(self.payloads)['roles.json']['role_groups']
        self.reject()

    def test_new_audit_cannot_hide_root_overlap_in_original_roles(self):
        roles = decode(self.payloads)['roles.json']['role_groups']
        joined_components = {roles['A'][0], roles['B'][0]}
        for entry in self.protocol['root_audit']['entries']:
            if entry['split_group'] in joined_components:
                entry['root_id'] = 'invented-correlated-original-roles'
        # The audit reveals a correlation the original components did not.
        # Fail before construction of any new nested roles or support plan.
        with mock.patch.object(oof, 'nested_plan', side_effect=AssertionError('too late')):
            self.reject()

    def test_new_audit_cannot_collapse_components_within_any_original_role(self):
        for role in ('A', 'B', 'C'):
            self.payloads, self.config, self.protocol = fixture()
            groups = decode(self.payloads)['roles.json']['role_groups'][role]
            for entry in self.protocol['root_audit']['entries']:
                if entry['split_group'] in groups:
                    entry['root_id'] = 'invented-one-root-for-old-components'
            # Original B folds/support cannot survive this new audit. Refuse
            # the old snapshot before consulting any proposed new nested plan.
            with self.subTest(role=role), \
                 mock.patch.object(oof, 'nested_plan', side_effect=AssertionError('too late')), \
                 self.assertRaisesRegex(adapter.AdapterError, 'original component/role snapshot requires review'):
                self.build()

    def test_multiple_original_groups_inside_one_component_remain_supported(self):
        documents = decode(self.payloads)
        rows = documents['calibration.jsonl']
        component = rows[0]['split_group']
        changed_ids = {row['id'] for row in rows if row['split_group'] == component} - {rows[0]['id']}
        for row in rows:
            if row['id'] in changed_ids:
                row['group'] = 'invented-second-original-group-inside-component'
        for entry in self.protocol['root_audit']['entries']:
            if entry['id'] in changed_ids:
                entry['group'] = 'invented-second-original-group-inside-component'
        self.payloads = encode(documents)
        repin(self.payloads, self.config, self.protocol)
        draft = self.build()
        matching = [r for r in draft['oof_plan']['rows'] if r['split_group'] == component]
        self.assertEqual(len({r['group'] for r in matching}), 2)
        self.assertEqual(len({r['root_id'] for r in matching}), 1)
        checked = adapter.validate_synthetic_records(self.payloads, self.config, self.protocol,
                                                     draft, predictions(draft))
        self.assertEqual(checked['row_epoch_count'], 252)
        self.assertEqual(checked['formal_admission'], 'NO-GO')
        self.assertIs(checked['real_root_audit_verified'], False)

    def test_local_role_override_allowlists_match_supported_numeric_fields(self):
        self.assertEqual(adapter.ROLE_PROTOCOL_KEYS, set(cg.DEFAULT_PROTOCOL))
        self.assertEqual(adapter.ROLE_SUPPORT_KEYS, set(cg.DEFAULT_SUPPORT_POLICY))
        valid = {'protocol': {'tau_grid': [0, .75], 'temperature_bounds': [1, 4.],
                    'fit_max_iterations': 20, 'fit_tolerance': 1e-5, 'probability_epsilon': 1e-8},
                 'support_policy': {'min_groups_per_role': {'A': 5, 'B': 6, 'C': 5},
                    'min_rows_per_class': 6, 'min_groups_per_class': 2,
                    'min_fold_train_groups_per_class': 2, 'n_folds': 2}}
        for field, values in valid.items():
            for key, value in values.items():
                self.payloads, self.config, self.protocol = fixture()
                self.mutate_document('roles.json', lambda roles: roles.update({field: {key: value}}))
                before = copy.deepcopy((self.payloads, self.config, self.protocol))
                with self.subTest(field=field, key=key):
                    draft = self.build()
                    checked = adapter.validate_synthetic_records(self.payloads, self.config, self.protocol,
                                                                 draft, predictions(draft))
                    self.assertEqual(checked['formal_admission'], 'NO-GO')
                    self.assertEqual(before, (self.payloads, self.config, self.protocol))
        self.payloads, self.config, self.protocol = fixture()
        self.mutate_document('roles.json', lambda roles: roles.update(valid))
        self.assertEqual(self.build()['formal_admission'], 'NO-GO')
        for field in valid:
            self.payloads, self.config, self.protocol = fixture()
            self.mutate_document('roles.json', lambda roles: roles.update({field: {}}))
            self.build()  # An empty explicit override keeps inherited defaults.

    def test_fully_repinned_unknown_role_override_fields_are_rejected(self):
        for field in ('protocol', 'support_policy'):
            for key in ('qualified', 'approval_verified', 'approval_ref', 'future_ignored_field',
                        'fit', 'score', 'selector'):
                self.payloads, self.config, self.protocol = fixture()
                self.mutate_document('roles.json', lambda roles: roles.update({field: {key: True}}))
                with self.subTest(field=field, key=key):
                    self.reject()
            for wrong_map in (None, [], True, 1, 'unknown'):
                self.payloads, self.config, self.protocol = fixture()
                self.mutate_document('roles.json', lambda roles: roles.update({field: wrong_map}))
                self.reject()
        self.payloads, self.config, self.protocol = fixture()
        self.mutate_document('roles.json', lambda roles: roles.update(support_policy={
            'min_groups_per_role': {'A': 5, 'B': 6, 'C': 5, 'qualified': True}}))
        self.reject()

    def test_known_role_override_types_shapes_and_ranges_still_fail_closed(self):
        wrong_protocol = {'tau_grid': (True, '0', [], [True], [-1], [10**400]),
            'temperature_bounds': (None, [], [1], [1, 2, 3], [True, 2], [0, 2], [2, 1]),
            'fit_max_iterations': (True, 1.0, '3', 0),
            'fit_tolerance': (True, '0.1', 0, -1, 10**400),
            'probability_epsilon': (True, '0.1', 0, 1, 10**400)}
        wrong_support = {'min_groups_per_role': (None, [], {'A': 5},
                    {'A': True, 'B': 6, 'C': 5}, {'A': 0, 'B': 6, 'C': 5}),
            'min_rows_per_class': (True, '5', 5.0, 0),
            'min_groups_per_class': (True, '2', 2.0, -1),
            'min_fold_train_groups_per_class': (True, None, 2.0, 0),
            'n_folds': (True, '3', 3.0, 1)}
        for field, fields in (('protocol', wrong_protocol), ('support_policy', wrong_support)):
            for key, values in fields.items():
                for value in values:
                    self.payloads, self.config, self.protocol = fixture()
                    self.mutate_document('roles.json', lambda roles: roles.update({field: {key: value}}))
                    with self.subTest(field=field, key=key, value=value):
                        self.reject()

    def test_oversized_numeric_values_fail_with_adapter_error(self):
        self.config['training']['lr'] = 10**400
        self.protocol['config_sha256'] = research.config_hash(self.config)
        self.reject()
        self.payloads, self.config, self.protocol = fixture()
        draft = self.build()
        for head in ('tier', 'vision', 'task_type'):
            records = predictions(draft)
            records[0]['logits'][head][0] = 10**400
            with self.subTest(head=head), self.assertRaises(adapter.AdapterError):
                adapter.validate_synthetic_records(self.payloads, self.config, self.protocol, draft, records)
        self.config['training']['epochs'] = 10**400
        self.protocol['config_sha256'] = research.config_hash(self.config)
        self.reject()  # Do not allocate an unbounded range for an invalid schedule.

    def test_nested_roles_overlap_validation_leak_and_support_fail_closed(self):
        for mode in ('overlap', 'empty', 'holdout', 'missing', 'unknown', 'support', 'fold_type'):
            self.payloads, self.config, self.protocol = fixture()
            roles = self.protocol['nested_role_roots'][0]
            if mode == 'overlap': roles['A'].append(roles['B'][0])
            if mode == 'empty': roles['C'] = []
            if mode == 'holdout': roles['fit'].append(self.build()['oof_plan']['folds'][0]['validation_roots'][0])
            if mode == 'missing': self.protocol['nested_role_roots'].pop()
            if mode == 'unknown': roles['D'] = roles['C']
            if mode == 'support': self.protocol['min_class_roots'] = 100
            if mode == 'fold_type': self.protocol['n_folds'] = True
            self.reject()

    def test_epoch_registry_missing_duplicates_mismatch_unknown_and_unapproved(self):
        for mutate in [lambda p: p.update(epochs=[1]), lambda p: p.update(epochs=[1, 1, 2]),
            lambda p: p.update(epochs=[True, 2]), lambda p: p.update(epochs=[2, 1]),
            lambda p: p['models'].pop(), lambda p: p['models'].append(copy.deepcopy(p['models'][0])),
            lambda p: p['models'][0].update(model_digest='not-a-digest'),
            lambda p: p['models'][0].update(source='real-checkpoint'),
            lambda p: p['models'][0].update(fold=True),
            lambda p: p['models'][0].update(epoch=3),
            lambda p: p['models'][0].update(approval='yes')]:
            self.payloads, self.config, self.protocol = fixture()
            mutate(self.protocol)
            self.reject()

    def test_changed_context_cannot_validate_old_draft_even_with_fresh_bindings(self):
        for field in ('input', 'config', 'audit', 'roles', 'model'):
            self.payloads, self.config, self.protocol = fixture()
            draft = self.build()
            if field == 'input':
                self.mutate_document('train.jsonl', lambda rows: rows[0].update(prompt='New invented content'))
            if field == 'config':
                self.config['training']['seed'] += 1
                self.protocol['config_sha256'] = research.config_hash(self.config)
            if field == 'audit': self.protocol['root_audit']['evidence_ref'] = 'different-invented-audit'
            if field == 'roles':
                roles = self.protocol['nested_role_roots'][0]
                roles['A'], roles['B'] = roles['B'], roles['A']
            if field == 'model': self.protocol['models'][0]['model_digest'] = 'f' * 64
            with self.subTest(field=field), self.assertRaises(adapter.AdapterError):
                adapter.validate_synthetic_records(self.payloads, self.config, self.protocol,
                                                   draft, predictions(draft))

    def test_oof_prediction_lineage_coverage_and_all_heads(self):
        draft = self.build()
        mutations = [lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])),
            lambda r: r[0].update(id='foreign'), lambda r: r[0].update(epoch=True),
            lambda r: r[0].update(epoch=3), lambda r: r[0].update(fold=8),
            lambda r: r[0].update(source_split='test'), lambda r: r[0].update(root_id='foreign'),
            lambda r: r[0].update(plan_digest='0' * 64), lambda r: r[0].update(draft_sha256='0' * 64),
            lambda r: r[0].update(model_digest='f' * 64),
            lambda r: r[0]['training_roots'].append(r[0]['root_id']),
            lambda r: r[0].update(task_label_schema_version=schema.LEGACY),
            lambda r: r[0].update(unknown=True), lambda r: r[0]['logits'].update(task_type=[0.] * 7),
            lambda r: r[0]['logits'].update(task_type=[0.] * 5 + [True]),
            lambda r: r[0]['logits'].update(tier=[0., float('nan'), 1.]),
            lambda r: r[0]['logits'].update(vision=[0., float('inf')]),
            lambda r: r[0]['logits'].pop('task_type'),
            lambda r: r[0]['logits'].update(unknown=[0.])]
        for mutate in mutations:
            records = predictions(draft)
            mutate(records)
            with self.assertRaises(adapter.AdapterError):
                adapter.validate_synthetic_records(self.payloads, self.config, self.protocol, draft, records)
        # A consistent replacement across all records must still match the registry.
        records = predictions(draft)
        for record in records:
            record['model_digest'] = 'f' * 64
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_synthetic_records(self.payloads, self.config, self.protocol, draft, records)

    def test_no_io_network_sampling_model_loading_or_hidden_file_discovery(self):
        draft = self.build()
        records = predictions(draft)
        original_import = builtins.__import__
        def safe_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'peft', 'transformers', 'huggingface_hub'):
                raise AssertionError('model/framework import forbidden')
            return original_import(name, *args, **kwargs)
        with mock.patch('builtins.open', side_effect=AssertionError('file access forbidden')), \
             mock.patch('io.open', side_effect=AssertionError('file access forbidden')), \
             mock.patch.object(Path, 'glob', side_effect=AssertionError('discovery forbidden')), \
             mock.patch.object(Path, 'mkdir', side_effect=AssertionError('output forbidden')), \
             mock.patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')), \
             mock.patch.object(research, 'sample_epochs', side_effect=AssertionError('sampling forbidden')), \
             mock.patch.object(model_utils, 'load_engine', side_effect=AssertionError('model forbidden')), \
             mock.patch('builtins.__import__', side_effect=safe_import):
            self.assertEqual(draft, self.build())
            checked = adapter.validate_synthetic_records(self.payloads, self.config, self.protocol, draft, records)
            self.assertEqual(checked['formal_admission'], 'NO-GO')


if __name__ == '__main__':
    unittest.main()
