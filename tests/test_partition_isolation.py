"""Issue #2: invented CPU-only identity/root boundary regressions."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_calibration_groups as fixtures
import calibration_groups as cg
import task_label_schema as ls

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('isolation_validator', ROOT/'scripts/validate_calibration.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def rows_for(selected):
    train, pool = fixtures.make_train_rows(), fixtures.make_pool()
    for row in train + pool:
        row['group'] = row['split_group']
        row.pop('synthetic', None)
        if selected == ls.V2:
            row['task_label_schema_version'] = ls.V2
            row['label']['label_status'] = 'resolved'
            if row['label']['task_type'] == 'visual':
                row['label']['task_type'] = 'extraction'
    return train, pool


class IsolationTests(unittest.TestCase):
    def exercise(self, selected, mutate, accepted, launcher=False):
        train, pool = rows_for(selected)
        roles = fixtures.make_roles(pool)
        mutate(train, pool)
        original = copy.deepcopy((train, pool))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root/'data'; data.mkdir()
            for name, rows in [('train', train), ('calibration', pool)]:
                fixtures.write_jsonl(data/(name+'.jsonl'), rows)
            (data/'test.jsonl').write_text('FORBIDDEN TEST READ')
            role_path = root/'roles.json'; role_path.write_text(json.dumps(roles))
            reads = []; runtime = fixtures.import_train_eval_with_stubs(reads)
            # Actual question metadata (no model loaded); subprocess must agree.
            with contextlib.nullcontext():
                if selected == ls.V2:
                    meta = ls.metadata(selected)
                    (data/'label-schema.json').write_text(json.dumps(meta))
                    (data/'split-report.json').write_text(json.dumps({'label_schema': meta}))
                ok, message = validator.run_validation(data, role_path, selected)
                self.assertEqual(ok, accepted, message)
                args = argparse.Namespace(run='isolation', seed=1, label_schema=selected,
                    legacy_calibration=False, calibration_roles=str(role_path),
                    structural_dry_run=True, smoke=False, synth=False)
                with mock.patch.object(runtime, 'ROOT', root), \
                     mock.patch.object(runtime, 'load_engine', side_effect=AssertionError('model forbidden')), \
                     mock.patch.object(runtime, 'oversample_epoch_rows', side_effect=AssertionError('sampler forbidden')), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    if accepted:
                        runtime.run(args)
                        summary = json.loads(message)['plan']
                        self.assertIn(json.dumps(summary['root_isolation']), output.getvalue())
                    else:
                        with self.assertRaises(cg.PlanError) as error:
                            runtime.run(args)
                        self.assertEqual(json.loads(message)['error'], str(error.exception))
                        for row in train + pool:
                            for key in ('id', 'split_group', 'group'):
                                if isinstance(row.get(key), str) and row[key].strip():
                                    self.assertNotIn(row[key], str(error.exception))
            self.assertFalse((root/'artifacts').exists())
            self.assertEqual({Path(path).name for path in reads}, {'train.jsonl', 'calibration.jsonl'})
            if launcher:
                command = [sys.executable, str(ROOT/'scripts/train.py'), '--run', 'isolation',
                           '--dry-run', '--data-dir', str(data), '--roles', str(role_path),
                           '--label-schema', selected]
                recipe = root/'recipe.json'; recipe.write_text(json.dumps({'synth': False}))
                command += ['--recipe', str(recipe)]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
                if not accepted:
                    self.assertIn(json.loads(message)['error'], result.stdout + result.stderr)
                self.assertFalse((root/'artifacts').exists())
        self.assertEqual((train, pool), original)

    def test_entrypoint_parity(self):
        def no_roots(t, c):
            for row in t+c: row.pop('group')
        def joined_roots(t, c):
            # Multiple original roots legitimately share a split component.
            c[0]['group'] = 'another-original-root'
        def id_overlap(t, c): t[0]['id'] = c[0]['id']
        def split_overlap(t, c): t[0]['split_group'] = c[0]['split_group']
        def cross_root(t, c): t[0]['group'] = c[0]['group']
        def role_alias(t, c):
            for row in c: row['group'] = 'fragmented-original-root'
        def fold_alias(t, c):
            groups = fixtures.make_roles(c)['role_groups']['B'][:2]
            for row in c:
                if row['split_group'] in groups: row['group'] = 'fragmented-b-root'
        def partial(t, c): t[0].pop('group')
        cases = [(lambda t,c: None, True), (no_roots, True), (joined_roots, True),
                 (id_overlap, False), (split_overlap, False), (cross_root, False),
                 (role_alias, False), (fold_alias, False), (partial, False)]
        for selected in ls.VERSIONS:
            for mutate, accepted in cases:
                with self.subTest(schema=selected, case=mutate.__name__):
                    self.exercise(selected, mutate, accepted, launcher=True)
            for value in (None, '', '  ', [], {}, 1, True):
                with self.subTest(schema=selected, malformed=value):
                    self.exercise(selected, lambda t,c: c[0].update(group=value), False)

    def test_direct_plan_rejects_root_alias_without_changing_roles(self):
        for selected in ls.VERSIONS:
            train, pool = rows_for(selected); roles = fixtures.make_roles(pool)
            for row in pool: row['group'] = 'aliased-root'
            with self.assertRaisesRegex(cg.PlanError, 'fragmented_roots=1'):
                cg.preflight_roles(pool, roles, schema_version=selected)

    def test_permutation_and_train_dedup_semantics(self):
        train, pool = rows_for(ls.LEGACY)
        train.append(copy.deepcopy(train[0]))
        self.assertEqual(cg.validate_partition_isolation(train, pool),
                         cg.validate_partition_isolation(train[::-1], pool[::-1]))
        train[0]['group'] = pool[0]['group']
        for t,c in [(train,pool), (train[::-1],pool[::-1])]:
            with self.assertRaisesRegex(cg.PlanError, 'fragmented_roots=1'):
                cg.validate_partition_isolation(t,c)

    def test_malformed_identity_has_native_error(self):
        for key in ('id', 'split_group'):
            for value in (None, [], {}, True, '', ' '):
                train, pool = rows_for(ls.LEGACY); train[0][key] = value
                with self.assertRaisesRegex(cg.PlanError, 'invalid_rows=1'):
                    cg.validate_partition_isolation(train, pool)


if __name__ == '__main__': unittest.main()
