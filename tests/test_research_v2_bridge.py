"""Research-v2 bridge: invented CPU-only inputs and real entrypoint preflights.

No archived data, prediction artifacts, weights, backbone, downloads or GPU are
used. Bad fixtures are re-pinned deliberately: a hash failure must not stand in
for validating expert provenance, genuine prior support, or root isolation.
"""
import argparse
import ast
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calibration_groups as cg
import model_utils as mu
import research_training as rt
import task_label_schema as ls

_spec = importlib.util.spec_from_file_location(
    'research_v2_calibration_fixtures', ROOT/'tests/test_calibration_groups.py')
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

INPUT_NAMES = ('train.jsonl', 'calibration.jsonl', 'label-schema.json',
               'split-report.json', 'roles.json')
V2_VERSION = 'research-training-v2'
V2_SCHEMA = 'task6-v2'


def invented_rows():
    """Reuse only invented group layouts, with explicit invented provenance."""
    train, calibration = fixtures.make_train_rows(42), fixtures.make_pool(16)
    for rows in (train, calibration):
        for index, row in enumerate(rows):
            row.pop('synthetic', None)
            row['group'] = row['split_group']
            row['prompt'] = 'Invented research-v2 fixture prompt ' + row['id']
            row['task_label_schema_version'] = ls.V2
            row['label_schema_version'] = 1
            row['label'].update(task_type=ls.SIX_TYPES[index % 6],
                label_source='expert_reviewed' if index % 2 else 'expert_provisional',
                label_status='resolved', evidence='prompt_only', confidence=.8,
                task_label_schema_version=ls.V2, label_schema_version=1)
    return train, calibration


def config_template():
    config = json.loads((ROOT/'configs/research-v1.json').read_text())
    config.update(version=V2_VERSION, schema_version=V2_SCHEMA,
                  label_schema=ls.metadata(ls.V2),
                  input_sha256=dict.fromkeys(INPUT_NAMES, '0'*64))
    config['sampler']['epoch_rows'] = 31
    return config


class FixtureCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='router-research-v2-fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root/'data'
        self.data.mkdir()
        self.roles_path = self.root/'roles.json'
        self.config_path = self.root/'research.json'
        self.train, self.calibration = invented_rows()
        self.roles = fixtures.make_roles(self.calibration)
        self.metadata = ls.metadata(ls.V2)
        self.report = {'label_schema': copy.deepcopy(self.metadata),
                       'task_label_schema_version': ls.V2,
                       'label_schema_version': 1, 'splits': {}}
        self.config = config_template()
        self.questions = ls.questions_for(ls.V2, mu.QUESTIONS)
        self.persist()
        (self.data/'test.jsonl').write_text('FORBIDDEN TEST SPLIT READ, INVALID JSON')

    def input_path(self, name):
        return self.roles_path if name == 'roles.json' else self.data/name

    def pin(self):
        self.config['input_sha256'] = {
            name: hashlib.sha256(self.input_path(name).read_bytes()).hexdigest()
            for name in INPUT_NAMES}
        self.config_path.write_text(json.dumps(self.config), encoding='utf-8')

    def persist(self):
        fixtures.write_jsonl(self.data/'train.jsonl', self.train)
        fixtures.write_jsonl(self.data/'calibration.jsonl', self.calibration)
        for path, value in [(self.roles_path, self.roles),
                            (self.data/'label-schema.json', self.metadata),
                            (self.data/'split-report.json', self.report)]:
            path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        self.pin()

    @contextlib.contextmanager
    def guarded_reads(self, forbidden=()):
        """Trap every normal Path/io/builtins read of test/private/model files."""
        original_open = io.open
        recorded = []
        forbidden = {Path(path).resolve() for path in forbidden}

        def checked_open(file, mode='r', *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(file)).resolve()
                if (path.name == 'test.jsonl' or path in forbidden or
                    any(path == ROOT/name or ROOT/name in path.parents
                        for name in ('artifacts', 'models', 'data'))):
                    raise AssertionError('forbidden private/model/test input access')
                recorded.append((path, mode))
            return original_open(file, mode, *args, **kwargs)

        with mock.patch('io.open', side_effect=checked_open), \
             mock.patch('builtins.open', side_effect=checked_open), \
             mock.patch.object(Path, 'mkdir', side_effect=AssertionError('artifact/output creation forbidden')):
            yield recorded

    def load(self, **kwargs):
        return rt.load_v2_inputs(self.config, self.data, self.roles_path,
                                 self.questions, **kwargs)

    def runtime(self, **overrides):
        runtime = fixtures.import_train_eval_with_stubs()
        # The shared legacy test stub contains class names only. Research v2
        # binds instruction/criterion meaning too, so use the actual constants.
        runtime.QUESTIONS = copy.deepcopy(mu.QUESTIONS)
        options = dict(self.config['training'], run='research-v2-no-output',
            label_schema=ls.V2, legacy_calibration=False, smoke=False,
            research_config=str(self.config_path),
            calibration_roles=str(self.roles_path), data_dir=str(self.data),
            structural_dry_run=True, supplemental=None)
        options.update(overrides)
        with mock.patch.object(runtime, 'ROOT', self.root), \
             mock.patch.object(runtime, 'load_engine', side_effect=AssertionError('model load forbidden')), \
             mock.patch.object(runtime, 'oversample_epoch_rows', side_effect=AssertionError('legacy sampler forbidden')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            runtime.run(argparse.Namespace(**options))
        self.assertFalse((self.root/'artifacts').exists())
        return output.getvalue()

    def launcher(self, *extra):
        # Run the actual CLI, with independent process-level canaries. The
        # fixture directory is outside ROOT; no real dataset can be touched.
        bootstrap = r'''
import builtins, io, os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
real_open = io.open
real_import = builtins.__import__
def guarded_open(file, mode='r', *args, **kwargs):
    if isinstance(file, (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(file)).resolve()
        if path.name == 'test.jsonl' or path.name == 'forbidden-supplement.jsonl' or any(
                path == root/name or root/name in path.parents
                for name in ('data', 'artifacts', 'models')):
            raise AssertionError('forbidden private/model/test/supplement input access')
        if any(flag in mode for flag in 'wax+'):
            raise AssertionError('structural CLI wrote a file')
    return real_open(file, mode, *args, **kwargs)
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'peft', 'transformers', 'huggingface_hub'):
        raise AssertionError('model/framework import forbidden during CPU preflight')
    return real_import(name, *args, **kwargs)
def forbidden_mkdir(*args, **kwargs):
    raise AssertionError('structural CLI created output')
io.open = builtins.open = guarded_open
builtins.__import__ = guarded_import
pathlib.Path.mkdir = forbidden_mkdir
sys.dont_write_bytecode = True
sys.argv = [str(root/'scripts/train.py')] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        command = [sys.executable, '-c', bootstrap, str(ROOT),
            '--run', 'research-v2-synthetic-bridge-canary', '--dry-run',
            '--roles', str(self.roles_path), '--data-dir', str(self.data),
            '--research-config', str(self.config_path), '--label-schema', ls.V2]
        return subprocess.run(command + list(extra), cwd=self.root,
            capture_output=True, text=True, timeout=30)

    def assert_entrypoints_reject(self, supplemental=None):
        kwargs = {'supplemental': supplemental} if supplemental else {}
        with self.guarded_reads(forbidden=supplemental or []):
            with self.assertRaises((ValueError, SystemExit)) as direct:
                self.load(**kwargs)
            with self.assertRaises((ValueError, SystemExit)) as actual:
                self.runtime(**kwargs)
        self.assertEqual(str(actual.exception), str(direct.exception))
        cli = self.launcher(*(['--supplemental', str(supplemental[0])] if supplemental else []))
        self.assertNotEqual(cli.returncode, 0, cli.stdout)
        self.assertIn(str(direct.exception), cli.stdout + cli.stderr)
        self.assertNotIn('Command:', cli.stdout)
        self.assertNotIn('AssertionError', cli.stderr)
        self.assertFalse((self.root/'artifacts').exists())
        return str(direct.exception)


class ResearchV2ConfigTests(FixtureCase):
    def test_versioned_config_and_explicit_schema_optin(self):
        self.assertEqual(rt.V2_VERSION, V2_VERSION)
        self.assertEqual(rt.V2_SCHEMA, V2_SCHEMA)
        self.assertEqual(rt.validate_config(self.config), self.config)
        self.assertEqual(rt.load_config(self.config_path), self.config)
        self.assertEqual(rt.check_task_schema(ls.V2, self.config), V2_SCHEMA)
        self.assertEqual(rt.check_task_schema(), rt.SCHEMA)
        legacy = rt.load_config(ROOT/'configs/research-v1.json')
        self.assertEqual(rt.check_task_schema(ls.LEGACY, legacy), rt.SCHEMA)
        for selected, config in [(ls.V2, None), (ls.V2, legacy),
                                 (ls.LEGACY, self.config), (None, self.config)]:
            with self.subTest(selected=selected, version=config and config['version']), self.assertRaises(ValueError):
                rt.check_task_schema(selected, config)

    def test_shipping_template_is_explicit_and_unbound(self):
        template = rt.load_config(ROOT/'configs/research-v2.json')
        self.assertEqual(template['version'], V2_VERSION)
        self.assertEqual(template['schema_version'], V2_SCHEMA)
        self.assertEqual(template['label_schema'], ls.metadata(ls.V2))
        self.assertEqual(template['input_sha256'], dict.fromkeys(INPUT_NAMES, '0'*64))
        with self.guarded_reads(), self.assertRaises(ValueError):
            rt.load_v2_inputs(template, self.data, self.roles_path, self.questions)

    def test_exact_versioned_config_metadata_and_hash_keys(self):
        changes = [lambda c: c.pop('label_schema'), lambda c: c.pop('input_sha256'),
            lambda c: c.update(unknown='ignored?'),
            lambda c: c.update(schema_version=rt.SCHEMA),
            lambda c: c.update(version=rt.VERSION),
            lambda c: c['label_schema'].update(question_sha256='f'*64),
            lambda c: c['label_schema']['classes']['task_type'].reverse(),
            lambda c: c['label_schema'].update(extra=True),
            lambda c: c['input_sha256'].pop('roles.json'),
            lambda c: c['input_sha256'].update(**{'test.jsonl': '0'*64})]
        for change in changes:
            config = copy.deepcopy(self.config); change(config)
            with self.subTest(change=changes.index(change)), self.assertRaises(ValueError):
                rt.validate_config(config)
        for invalid in [None, 1, True, [], '', 'a'*63, 'g'*64]:
            config = copy.deepcopy(self.config)
            config['input_sha256']['train.jsonl'] = invalid
            with self.subTest(hash=invalid), self.assertRaises(ValueError):
                rt.validate_config(config)

    def test_questions_require_full_meaning_and_order_for_every_head(self):
        rt.check_questions(self.questions, self.config)
        changes = [lambda q: q['task_type']['criteria'].update(visual='legacy'),
            lambda q: q['task_type'].update(criteria=dict(reversed(list(q['task_type']['criteria'].items())))),
            lambda q: q['task_type'].update(instructions='Different primary-action rule'),
            lambda q: q['task_type']['criteria'].update(creative='Executable software'),
            lambda q: q['tier'].update(instructions='Pick the most expensive model'),
            lambda q: q['vision']['criteria'].update(yes='Text is sufficient'),
            lambda q: q['vision'].update(type='free-text'),
            lambda q: q['tier'].update(extra='not part of bound question')]
        for change in changes:
            questions = copy.deepcopy(self.questions); change(questions)
            with self.subTest(change=changes.index(change)), self.assertRaises(ValueError):
                rt.check_questions(questions, self.config)
        with self.assertRaises(ValueError): rt.check_questions(cg.questions_from_source(), self.config)
        with self.assertRaises(ValueError): rt.check_questions(mu.QUESTIONS, self.config)
        with self.assertRaises(ValueError): rt.check_questions(self.questions)
        rt.check_questions(mu.QUESTIONS)  # Seven-class v1 remains the default.

    def test_config_cli_crosses_and_changed_recipe_are_rejected_before_reads(self):
        args = argparse.Namespace(**self.config['training'], label_schema=ls.V2,
                                  legacy_calibration=False, smoke=False)
        rt.check_args(self.config, args)
        for key, value in [('label_schema', ls.LEGACY), ('epochs', 3),
                           ('smoke', True), ('legacy_calibration', True)]:
            bad = copy.copy(args); setattr(bad, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): rt.check_args(self.config, bad)
        for selected in (ls.LEGACY, None):
            with self.guarded_reads(forbidden=[self.input_path(n) for n in INPUT_NAMES]), self.assertRaises(ValueError):
                self.runtime(label_schema=selected)
        cli = self.launcher('--label-schema', ls.LEGACY)
        self.assertNotEqual(cli.returncode, 0)
        self.assertNotIn('Command:', cli.stdout)


class ResearchV2InputTests(FixtureCase):
    def test_review_f1_rejects_conflicting_expert_identity_in_both_partitions(self):
        original=copy.deepcopy((self.train,self.calibration))
        # Exact independent-review counterexample, re-pinned so hashes cannot
        # hide an incorrect task/expert join. Both real entrypoints must reject.
        for partition in (0,1):
            self.train,self.calibration=copy.deepcopy(original)
            (self.train,self.calibration)[partition][0]['label']['id']='different-invented-task'
            self.persist()
            with self.subTest(partition=partition):
                error=self.assert_entrypoints_reject()
                self.assertIn('identity',error)
                self.assertNotIn('different-invented-task',error)

    def test_review_f2_rejects_interval_confidence_at_both_provenance_levels(self):
        original=copy.deepcopy((self.train,self.calibration))
        # Four exact reviewer counterexamples, repeated with weighting disabled:
        # interval coverage cannot become expert support or an expert weight.
        for enabled in (True,False):
            self.config['loss']['confidence_weighting']=enabled
            for partition in (0,1):
                for nested in (False,True):
                    self.train,self.calibration=copy.deepcopy(original)
                    row=(self.train,self.calibration)[partition][0]
                    (row['label'] if nested else row)['confidence_semantics']='interval_confidence_level_not_sample_weight'
                    self.persist()
                    with self.subTest(enabled=enabled,partition=partition,nested=nested):
                        self.assertIn('confidence_semantics',self.assert_entrypoints_reject())
                        with self.assertRaisesRegex(ValueError,'confidence_semantics'):
                            rt.confidence_weight(row,enabled)

    def test_conflicting_outer_evidence_cannot_be_masked_by_prompt_only_label(self):
        original=copy.deepcopy((self.train,self.calibration))
        for enabled in (True,False):
            self.config['loss']['confidence_weighting']=enabled
            for partition in (0,1):
                self.train,self.calibration=copy.deepcopy(original)
                row=(self.train,self.calibration)[partition][0]
                row['evidence']='acceptance_tests';self.persist()
                with self.subTest(enabled=enabled,partition=partition):
                    self.assertIn('evidence provenance',self.assert_entrypoints_reject())
                    with self.assertRaisesRegex(ValueError,'evidence provenance'):
                        rt.confidence_weight(row,enabled)

    def test_optional_expert_identity_and_evidence_are_validated_without_rewriting(self):
        for rows in (self.train,self.calibration):
            for row in rows:
                row['label']['id']=row['id'];row['evidence']='prompt_only'
        self.persist();before=copy.deepcopy((self.train,self.calibration))
        with self.guarded_reads():
            loaded=self.load();self.runtime()
        cli=self.launcher();self.assertEqual(cli.returncode,0,cli.stderr)
        self.assertEqual(loaded['datasets'],dict(zip(('train','calibration'),before)))
        for row in self.train+self.calibration:
            self.assertAlmostEqual(rt.confidence_weight(row),.64)
            self.assertEqual(rt.confidence_weight(row,False),1.)
        # Absence remains supported; no id/evidence/semantics is synthesized.
        for rows in (self.train,self.calibration):
            for row in rows:
                row['label'].pop('id');row.pop('evidence')
        self.persist();before=copy.deepcopy((self.train,self.calibration))
        with self.guarded_reads():loaded=self.load()
        self.assertEqual(loaded['datasets'],dict(zip(('train','calibration'),before)))

    def test_malformed_optional_provenance_rejects_without_changing_v1(self):
        row=copy.deepcopy(self.train[0])
        for value in (None,True,0,'',' ',[],{},'other-task'):
            bad=copy.deepcopy(row);bad['label']['id']=value
            with self.subTest(id=value),self.assertRaisesRegex(ValueError,'identity'):
                rt.validate_v2_experts([bad])
        for field in ('confidence_semantics','evidence'):
            for value in (None,True,0,'','expert_annotation_certainty',[],{}):
                for nested in (False,True):
                    bad=copy.deepcopy(row);(bad['label'] if nested else bad)[field]=value
                    with self.subTest(field=field,value=value,nested=nested),self.assertRaises(ValueError):
                        rt.validate_v2_experts([bad])
        # The focused correction does not reinterpret the legacy contract.
        legacy=copy.deepcopy(self.train)
        for item in legacy:
            item['task_label_schema_version']=ls.LEGACY
            item['label']['task_label_schema_version']=ls.LEGACY
        config=rt.load_config(ROOT/'configs/research-v1.json')
        expected=rt.sampler_plan(legacy,config)
        legacy[0].update(evidence='acceptance_tests',confidence_semantics='interval_confidence_level_not_sample_weight')
        legacy[0]['label']['id']='different-invented-task'
        self.assertEqual(rt.sampler_plan(legacy,config),expected)
        self.assertAlmostEqual(rt.confidence_weight(legacy[0]),.64)
        self.assertEqual(rt.confidence_weight(legacy[0],False),1.)

    def test_valid_snapshot_builds_grouped_plan_priors_and_deterministic_streams(self):
        before = copy.deepcopy((self.config, self.questions, self.train, self.calibration))
        with self.guarded_reads() as reads:
            loaded = self.load()
        self.assertTrue({'datasets', 'roles', 'plan', 'prior', 'streams', 'sampling',
                         'isolation', 'summary'} <= set(loaded))
        self.assertEqual(loaded['datasets'], {'train': self.train, 'calibration': self.calibration})
        self.assertEqual(loaded['roles'], self.roles)
        self.assertEqual(loaded['isolation'], {'status': 'declared_consistent', 'independence': 'unverified'})
        self.assertEqual(loaded['plan']['schema']['heads']['task_type']['classes'], list(ls.SIX_TYPES))
        expected_streams, expected_sampling = rt.sample_epochs(self.train, self.config,
            self.config['training']['seed'], self.config['training']['epochs'])
        expected_sampling['input_sha256'] = self.config['input_sha256']
        self.assertEqual((loaded['streams'], loaded['sampling']), (expected_streams, expected_sampling))
        self.assertEqual([len(rows) for rows in loaded['streams']], [31, 31])
        self.assertEqual(loaded['prior']['counts'], dict.fromkeys(rt.TIERS, 14))
        self.assertEqual(loaded['sampling']['prior_provenance']['unique_nonsynthetic_expert_support'],
                         {'tier': dict.fromkeys(rt.TIERS, 14), 'vision': dict.fromkeys(rt.VISION, 21)})
        self.assertEqual(before, (self.config, self.questions, self.train, self.calibration))
        consumed = collections.Counter(path for path, mode in reads if 'r' in mode)
        self.assertEqual(consumed, collections.Counter(self.input_path(n).resolve() for n in INPUT_NAMES))
        with self.guarded_reads():
            repeated = self.load()
        self.assertEqual(loaded['summary'], repeated['summary'])
        self.assertEqual(loaded['streams'], repeated['streams'])
        report = json.dumps(loaded['summary'], ensure_ascii=False)
        for row in self.train + self.calibration:
            for key in ('id', 'group', 'split_group', 'prompt'):
                self.assertNotIn(row[key], report)

    def test_snapshot_parsing_never_reopens_changed_input(self):
        original = Path.read_bytes
        target = self.data/'train.jsonl'
        changed = False
        def swap_after_read(path):
            nonlocal changed
            content = original(path)
            if path == target and not changed:
                changed = True
                target.write_bytes(b'INVALID UNPINNED REPLACEMENT')
            return content
        with self.guarded_reads(), mock.patch.object(Path, 'read_bytes', swap_after_read):
            loaded = self.load()
        self.assertTrue(changed)
        self.assertEqual(loaded['datasets']['train'], self.train)

    def test_every_pinned_file_rejects_even_whitespace_byte_drift(self):
        for name in INPUT_NAMES:
            self.persist()
            path = self.input_path(name); path.write_bytes(path.read_bytes() + b'\n')
            with self.subTest(input=name), self.guarded_reads(), self.assertRaises(ValueError):
                self.load()

    def test_exact_byte_binding_is_used_by_launcher_and_actual_runtime(self):
        for name in INPUT_NAMES:
            self.persist()
            path = self.input_path(name); path.write_bytes(path.read_bytes() + b' ')
            with self.subTest(input=name):
                self.assert_entrypoints_reject()

    def test_missing_or_corrupt_metadata_is_not_reconstructed(self):
        mutations = [lambda m: m.update(question_sha256='a'*64),
                     lambda m: m['classes']['task_type'].reverse(),
                     lambda m: m.update(task_label_schema_version=ls.LEGACY),
                     lambda m: m.pop('question_sha256'), lambda m: m.update(extra='not bound')]
        for name in ('label-schema.json', 'split-report.json'):
            for mutate in mutations:
                self.persist()
                document = copy.deepcopy(self.metadata)
                mutate(document)
                if name == 'split-report.json': document = {'label_schema': document}
                self.input_path(name).write_text(json.dumps(document))
                self.pin()
                with self.subTest(file=name, mutation=mutations.index(mutate)), self.guarded_reads(), self.assertRaises(ValueError):
                    self.load()
            self.persist()
            self.input_path(name).unlink()
            with self.subTest(missing=name), self.guarded_reads(), self.assertRaises((ValueError, OSError)):
                self.load()
        for document in ({}, {'label_schema': self.metadata, 'task_label_schema_version': ls.LEGACY},
                         {'label_schema': self.metadata, 'label_schema_version': True}):
            self.persist()
            (self.data/'split-report.json').write_text(json.dumps(document)); self.pin()
            with self.subTest(report=document), self.guarded_reads(), self.assertRaises(ValueError): self.load()

    def test_both_partitions_require_expert_prompt_only_resolved_provenance(self):
        mutations = [lambda r: r['label'].pop('label_source'),
            lambda r: r['label'].update(label_source='measured_candidate_outcomes'),
            lambda r: r['label'].update(label_source='auto_generated'),
            lambda r: r['label'].pop('evidence'),
            lambda r: r['label'].update(evidence='outcome_based'),
            lambda r: r['label'].update(label_status='adjudication'),
            lambda r: r['label'].update(label_status='excluded'),
            lambda r: r['label'].pop('label_status'),
            lambda r: r.update(label_source='measured_candidate_outcomes'),
            lambda r: r.update(label_status='adjudication'),
            lambda r: r.update(partition='measured_shadow'),
            lambda r: r['label'].update(partition='measured_shadow')]
        original = copy.deepcopy((self.train, self.calibration))
        for partition in (0, 1):
            for mutate in mutations:
                self.train, self.calibration = copy.deepcopy(original)
                mutate((self.train, self.calibration)[partition][0]); self.persist()
                with self.subTest(partition=partition, mutation=mutations.index(mutate)), self.guarded_reads(), self.assertRaises(ValueError): self.load()

    def test_expert_confidence_is_required_even_without_weighting(self):
        original = copy.deepcopy((self.train, self.calibration))
        self.config['loss']['confidence_weighting'] = False
        for partition in (0, 1):
            for value in (None, True, '0.8', -0.1, 1.1, float('nan'), float('inf')):
                self.train, self.calibration = copy.deepcopy(original)
                (self.train, self.calibration)[partition][0]['label']['confidence'] = value
                self.persist()
                with self.subTest(partition=partition, confidence=value), self.guarded_reads(), self.assertRaises(ValueError): self.load()
            self.train, self.calibration = copy.deepcopy(original)
            (self.train, self.calibration)[partition][0]['label'].pop('confidence'); self.persist()
            with self.subTest(partition=partition, missing=True), self.guarded_reads(), self.assertRaises(ValueError): self.load()

    def test_source_integer_and_semantic_string_namespaces_are_not_interchangeable(self):
        original = copy.deepcopy((self.train, self.calibration))
        for partition in (0, 1):
            for nested in (False, True):
                for key, value in [('label_schema_version', ls.V2), ('label_schema_version', True),
                                   ('label_schema_version', 2), ('task_label_schema_version', 1),
                                   ('task_label_schema_version', ls.LEGACY)]:
                    self.train, self.calibration = copy.deepcopy(original)
                    row = (self.train, self.calibration)[partition][0]
                    (row['label'] if nested else row)[key] = value; self.persist()
                    with self.subTest(partition=partition, nested=nested, key=key, value=value), self.guarded_reads(), self.assertRaises(ValueError): self.load()
            self.train, self.calibration = copy.deepcopy(original)
            (self.train, self.calibration)[partition][0].pop('task_label_schema_version'); self.persist()
            with self.subTest(partition=partition, missing_top_level=True), self.guarded_reads(), self.assertRaises(ValueError): self.load()

    def test_primary_data_rejects_outer_and_nested_synthetic_supplemental_flags(self):
        original = copy.deepcopy((self.train, self.calibration))
        for partition in (0, 1):
            for nested in (False, True):
                for key in ('synthetic', 'supplemental_only'):
                    for value in (True, 'false', 0, None):
                        self.train, self.calibration = copy.deepcopy(original)
                        row = (self.train, self.calibration)[partition][0]
                        (row['label'] if nested else row)[key] = value; self.persist()
                        with self.subTest(partition=partition, nested=nested, key=key, value=value), self.guarded_reads(), self.assertRaises(ValueError): self.load()
        self.train, self.calibration = copy.deepcopy(original)
        for row in self.train + self.calibration:
            for obj in (row, row['label']): obj.update(synthetic=False, supplemental_only=False)
        self.persist()
        with self.guarded_reads(): self.load()

    def test_outcome_fields_cannot_be_laundered_as_expert_supervision(self):
        original = copy.deepcopy((self.train, self.calibration))
        for partition in (0, 1):
            for nested in (False, True):
                for field in ('model', 'provider', 'historical_model', 'route_fingerprint', 'success_lb', 'success_rate'):
                    self.train, self.calibration = copy.deepcopy(original)
                    row = (self.train, self.calibration)[partition][0]
                    (row['label'] if nested else row)[field] = 'invented outcome marker'; self.persist()
                    with self.subTest(partition=partition, nested=nested, field=field), self.guarded_reads(), self.assertRaises(ValueError): self.load()

    def test_nonempty_prepared_identity_and_binary_six_class_labels_required(self):
        original = copy.deepcopy((self.train, self.calibration))
        for partition in (0, 1):
            for key in ('id', 'group', 'split_group', 'prompt'):
                for value in (None, '', ' ', 1, [], {}):
                    self.train, self.calibration = copy.deepcopy(original)
                    (self.train, self.calibration)[partition][0][key] = value; self.persist()
                    with self.subTest(partition=partition, key=key, value=value), self.guarded_reads(), self.assertRaises(ValueError): self.load()
            for key, value in [('tier', 'unknown'), ('needs_vision', 'uncertain'), ('task_type', 'visual')]:
                self.train, self.calibration = copy.deepcopy(original)
                (self.train, self.calibration)[partition][0]['label'][key] = value; self.persist()
                with self.subTest(partition=partition, key=key), self.guarded_reads(), self.assertRaises(ValueError): self.load()

    def test_duplicate_ids_and_missing_genuine_prior_support_are_rejected(self):
        original = copy.deepcopy((self.train, self.calibration))
        for partition in (0, 1):
            self.train, self.calibration = copy.deepcopy(original)
            rows = (self.train, self.calibration)[partition]; rows.append(copy.deepcopy(rows[0])); self.persist()
            with self.subTest(duplicate_partition=partition): self.assert_entrypoints_reject()
        for key, value in [('tier', 'balanced'), ('needs_vision', 'no')]:
            self.train, self.calibration = copy.deepcopy(original)
            for row in self.train: row['label'][key] = value
            self.persist()
            with self.subTest(missing_prior=key): self.assert_entrypoints_reject()

    def test_pr40_partition_and_root_isolation_cannot_be_bypassed(self):
        original = copy.deepcopy((self.train, self.calibration))
        def root_across_roles(train, calibration):
            for row in calibration: row['group'] = 'invented-fragmented-root'
        def root_inside_b_folds(train, calibration):
            groups = self.roles['role_groups']['B'][:2]
            for row in calibration:
                if row['split_group'] in groups: row['group'] = 'invented-b-fold-alias'
        mutations = [lambda t,c: t[0].update(id=c[0]['id']),
                     lambda t,c: t[0].update(split_group=c[0]['split_group']),
                     lambda t,c: t[0].update(group=c[0]['group']),
                     root_across_roles, root_inside_b_folds]
        for mutate in mutations:
            self.train, self.calibration = copy.deepcopy(original)
            mutate(self.train, self.calibration); self.persist()
            with self.subTest(mutation=mutations.index(mutate)):
                error = self.assert_entrypoints_reject()
                for row in self.train + self.calibration:
                    for key in ('id', 'group', 'split_group', 'prompt'): self.assertNotIn(row[key], error)

    def test_group_role_and_class_support_gates_still_apply(self):
        baseline = copy.deepcopy((self.roles, self.calibration))
        for case in ('missing_group', 'duplicated_role', 'unknown_group', 'schema_override', 'missing_vision_class'):
            self.roles, self.calibration = copy.deepcopy(baseline)
            if case == 'missing_group': self.roles['role_groups']['C'].pop()
            elif case == 'duplicated_role': self.roles['role_groups']['A'].append(self.roles['role_groups']['B'][0])
            elif case == 'unknown_group': self.roles['role_groups']['A'].append('invented-unknown-role')
            elif case == 'schema_override': self.roles['schema'] = {'heads': {}}
            else:
                for row in self.calibration:
                    if row['split_group'] in self.roles['role_groups']['A']: row['label']['needs_vision'] = 'no'
            self.persist()
            with self.subTest(case=case): self.assert_entrypoints_reject()


class ResearchV2EntrypointTests(FixtureCase):
    def test_real_cli_and_runtime_share_preflight_before_model_or_artifacts(self):
        with self.guarded_reads(), mock.patch.object(rt, 'load_v2_inputs', wraps=rt.load_v2_inputs) as shared:
            actual = self.runtime()
        self.assertEqual(shared.call_count, 1)
        cli = self.launcher()
        self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
        self.assertIn('STRUCTURAL DRY RUN', actual)
        self.assertIn('Research preflight', cli.stdout)
        self.assertIn('not backbone compatibility', cli.stdout)
        self.assertIn('--data-dir', cli.stdout)
        self.assertIn(str(self.data), cli.stdout)
        self.assertIn('--research-config', cli.stdout)
        self.assertIn(ls.V2, cli.stdout)
        self.assertNotIn("'--synth'", cli.stdout)
        self.assertNotIn("'--fullft'", cli.stdout)
        expected = self.load()['summary']
        # The shared object is reported directly, possibly inside an existing
        # runtime summary envelope; compare parsed JSON rather than whitespace.
        decoder = json.JSONDecoder()
        def objects(output):
            found = []
            for index, char in enumerate(output):
                if char == '{':
                    try: value, _ = decoder.raw_decode(output[index:])
                    except ValueError: continue
                    found.append(value)
            return found
        self.assertIn(expected, objects(actual))
        self.assertIn(expected, objects(cli.stdout))
        for output in (actual, cli.stdout):
            for row in self.train + self.calibration:
                for key in ('id', 'group', 'split_group', 'prompt'): self.assertNotIn(row[key], output)

    def test_launcher_command_runs_actual_runtime_main_and_argparse_on_cpu(self):
        cli = self.launcher()
        self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
        command = ast.literal_eval(next(line[len('Command: '):]
            for line in cli.stdout.splitlines() if line.startswith('Command: ')))
        self.assertEqual(Path(command[1]), ROOT/'train_eval.py')
        self.assertIn('--research-config-sha256', command)
        bootstrap = r'''
import builtins, io, os, pathlib, runpy, sys, types
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
sys.dont_write_bytecode = True
real_open = io.open
real_import = builtins.__import__
def forbidden(*args, **kwargs):
    raise AssertionError('model, optimizer, GPU, or output operation forbidden')
def guarded_open(file, mode='r', *args, **kwargs):
    if isinstance(file, (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(file)).resolve()
        if path.name == 'test.jsonl' or any(path == root/name or root/name in path.parents
                for name in ('data', 'artifacts', 'models')):
            raise AssertionError('forbidden private/model/test input access')
        if any(flag in mode for flag in 'wax+'):
            raise AssertionError('structural runtime wrote a file')
    return real_open(file, mode, *args, **kwargs)
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'peft', 'transformers', 'huggingface_hub', 'bitsandbytes') and name not in sys.modules:
        raise AssertionError('real model/framework import forbidden')
    return real_import(name, *args, **kwargs)
io.open = builtins.open = guarded_open
builtins.__import__ = guarded_import
pathlib.Path.mkdir = forbidden
# Only the import surface is stubbed. Real model_utils owns the questions,
# schema fingerprint and readers; neither a copied schema nor fake runtime
# parser can make this integration test pass.
import model_utils
model_utils.load_engine = forbidden
model_utils.reference_module = forbidden
torch = types.ModuleType('torch')
torch.nn = types.ModuleType('torch.nn')
torch.nn.functional = types.ModuleType('torch.nn.functional')
torch.manual_seed = torch.set_num_threads = forbidden
peft = types.ModuleType('peft')
peft.LoraConfig = peft.get_peft_model = forbidden
sys.modules.update({'torch': torch, 'torch.nn': torch.nn,
                    'torch.nn.functional': torch.nn.functional, 'peft': peft})
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-c', bootstrap, str(ROOT)] +
            command[1:] + ['--structural-dry-run'], cwd=self.root,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('STRUCTURAL DRY RUN', result.stdout)
        reported = json.loads(result.stdout[result.stdout.index('{'):])
        self.assertEqual(reported, self.load()['summary'])
        self.assertFalse((self.root/'artifacts').exists())
        self.assertFalse((ROOT/'artifacts/research-v2-synthetic-bridge-canary').exists())

    def test_launcher_binds_exact_config_and_runtime_rejects_post_preflight_changes(self):
        cli = self.launcher()
        self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
        command_lines = [line[len('Command: '):] for line in cli.stdout.splitlines()
                         if line.startswith('Command: ')]
        self.assertEqual(len(command_lines), 1)
        command = ast.literal_eval(command_lines[0])
        self.assertEqual(command.count('--research-config-sha256'), 1)
        expected = command[command.index('--research-config-sha256') + 1]
        self.assertEqual(expected, rt.config_hash(self.config))
        with self.guarded_reads(), mock.patch.object(rt, 'load_v2_inputs', wraps=rt.load_v2_inputs) as loader:
            output = self.runtime(research_config_sha256=expected)
        self.assertEqual(loader.call_count, 1)
        self.assertIn('STRUCTURAL DRY RUN', output)
        # Same dataset and valid recipe, but a changed protocol declaration:
        # runtime must not silently use a config different from preflight.
        self.config['evaluation_protocol'] = 'invented-changed-protocol-v2'
        self.config_path.write_text(json.dumps(self.config))
        self.assertNotEqual(expected, rt.config_hash(self.config))
        with self.guarded_reads(forbidden=[self.input_path(name) for name in INPUT_NAMES]), \
             mock.patch.object(rt, 'load_v2_inputs', side_effect=AssertionError('input loader forbidden')) as loader, \
             self.assertRaisesRegex(ValueError, 'config fingerprint changed'):
            self.runtime(research_config_sha256=expected)
        loader.assert_not_called()
        self.assertFalse((self.root/'artifacts').exists())

    def test_staged8bit_structural_preflight_does_not_require_optimizer_dependencies(self):
        self.config['capacity']['mode'] = 'staged8bit'
        self.config['training'].update(fullft=True, optimizer='adamw8bit', epochs=2)
        self.persist()
        runtime = fixtures.import_train_eval_with_stubs()
        with mock.patch.object(fixtures, 'import_train_eval_with_stubs', return_value=runtime), \
             mock.patch.object(runtime, 'ensure_optimizer_available', side_effect=AssertionError('optimizer dependency check forbidden')) as optimizer, \
             self.guarded_reads():
            output = self.runtime(research_config_sha256=rt.config_hash(self.config))
        optimizer.assert_not_called()
        self.assertIn('STRUCTURAL DRY RUN', output)
        cli = self.launcher()
        self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
        self.assertIn("'--optimizer', 'adamw8bit'", cli.stdout)
        self.assertIn("'--fullft'", cli.stdout)
        self.assertIn(rt.config_hash(self.config), cli.stdout)

    def test_supplemental_is_refused_identically_before_any_supplemental_read(self):
        forbidden = self.root/'forbidden-supplement.jsonl'
        forbidden.write_text('FORBIDDEN SUPPLEMENTAL CANARY')
        message = self.assert_entrypoints_reject(supplemental=[forbidden])
        self.assertIn('supplement', message.lower())
        self.assertIn('v2', message.lower())

    def test_rejected_calibration_expert_provenance_reaches_both_entrypoints(self):
        for source in ('measured_candidate_outcomes', 'untyped', None):
            self.calibration[0]['label']['label_source'] = source; self.persist()
            with self.subTest(source=source): self.assert_entrypoints_reject()

    def test_rejected_primary_nested_synthetic_reaches_both_entrypoints(self):
        for rows in (self.train, self.calibration):
            rows[0]['label']['synthetic'] = True; self.persist()
            self.assert_entrypoints_reject()
            rows[0]['label'].pop('synthetic')

    def test_missing_full_metadata_reaches_both_entrypoints(self):
        self.report.pop('label_schema'); self.persist()
        self.assert_entrypoints_reject()

    def test_research_v1_and_nonresearch_v2_remain_separate(self):
        legacy = rt.load_config(ROOT/'configs/research-v1.json')
        rt.check_questions(mu.QUESTIONS, legacy)
        with self.assertRaises(ValueError): rt.check_task_schema(ls.V2, legacy)
        # Existing general v2 training remains opt-in without any research
        # config or hash binding and continues using its own structural path.
        with self.guarded_reads(), mock.patch.object(rt, 'load_v2_inputs', side_effect=AssertionError('research-only path')):
            output = self.runtime(research_config=None)
        self.assertIn('STRUCTURAL DRY RUN', output)


class ResearchV2TensorTests(FixtureCase):
    def test_six_class_cpu_loss_backpropagates_all_active_heads(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional CPU tensor test; torch is not installed')
        plan = self.load()['sampling']
        logits = {head: torch.zeros(len(spec['criteria']), requires_grad=True)
                  for head, spec in self.questions.items()}
        row = self.train[0]
        loss = rt.row_loss(logits, row, self.questions, self.config, plan,
                           self.config['training']['label_smoothing'])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for head, values in logits.items():
            self.assertIsNotNone(values.grad)
            self.assertTrue(torch.isfinite(values.grad).all().item())
            self.assertGreater(values.grad.abs().sum().item(), 0)
        self.assertEqual(len(logits['task_type'].grad), 6)
        self.assertLess(logits['task_type'].grad[ls.SIX_TYPES.index(row['label']['task_type'])].item(), 0)


if __name__ == '__main__':
    unittest.main()
