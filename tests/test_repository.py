"""CPU-only checks for the standalone snapshot and recipe launcher."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic_data import synthetic_splits, supplemental_rows, SPLIT_NAMES


class RepositoryTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'snapshot-manifest.json').is_file(),
                         'Optional private archive validation: snapshot manifest is not distributed')
    def test_training_core_matches_archived_source(self):
        manifest = json.loads((ROOT / 'snapshot-manifest.json').read_text())
        for name in ['train_eval.py', 'model_utils.py', 'prepare.py']:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                             manifest['files'][name]['sha256'])

    def test_recipe_matches_completed_run(self):
        config = json.loads((ROOT / 'configs/v3c.json').read_text())
        report = json.loads((ROOT / 'artifacts/real-history-v3c/report.json').read_text())
        for key in ['epochs', 'rank', 'lr', 'accumulation']:
            self.assertEqual(config[key], report['training'][key])
        self.assertEqual(config['seed'], report['seed'])
        self.assertTrue(config['fullft'] and config['synth'] and config['swa'])

    def test_dry_run_does_not_start_gpu_or_create_output(self):
        # --legacy is explicit (issue #2): the archived v3c recipe keeps its
        # original row-split calibration; new runs default to grouped mode.
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                                 '--run', 'repository-validation-only', '--legacy', '--dry-run'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ['--fullft', '--synth', '--swa', '--legacy-calibration']:
            self.assertIn(flag, result.stdout)
        # Issue #14: the dry-run Command must show the COMPLETE recipe actually
        # passed (numerics included), with each flag bound to its v3c value.
        import ast
        command = ast.literal_eval(
            result.stdout.split('Command: ', 1)[1].strip().splitlines()[0])
        recipe = json.loads((ROOT / 'configs/v3c.json').read_text())
        for key in ['epochs', 'rank', 'lr', 'accumulation', 'seed', 'beta', 'label_smoothing']:
            flag = '--' + key.replace('_', '-')
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], str(recipe[key]), key)
        self.assertNotIn('--mlp', command)
        self.assertFalse((ROOT / 'artifacts/repository-validation-only').exists())

    def test_launcher_refuses_overwriting_winner(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                                 '--run', 'real-history-v3c', '--legacy', '--dry-run'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output already exists', result.stderr)

    def test_grouped_launch_requires_roles_and_never_falls_back_to_legacy(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                                 '--run', 'repository-validation-only', '--dry-run'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--roles', result.stderr)
        self.assertIn('--legacy', result.stderr)
        self.assertFalse((ROOT / 'artifacts/repository-validation-only').exists())
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                                 '--run', 'repository-validation-only', '--legacy',
                                 '--roles', 'unused.json', '--dry-run'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--legacy reproduces the archived', result.stderr)

    def test_supplemental_rows_are_separate_from_generated_splits(self):
        tasks, splits, _ = synthetic_splits()
        primary = [row for rows in splits.values() for row in rows]
        supplemental = supplemental_rows()
        self.assertEqual(len(primary), len(tasks))
        self.assertEqual(len(supplemental), 200)
        self.assertEqual(len({r['id'] for r in supplemental}), 200)
        self.assertEqual(len({r['prompt'] for r in supplemental}), 200)
        self.assertFalse({r['prompt'] for r in primary} & {r['prompt'] for r in supplemental})
        self.assertFalse({r['id'] for r in primary} & {r['id'] for r in supplemental})
        self.assertTrue(all(r['synthetic'] for r in primary + supplemental))

    @unittest.skipUnless(all((ROOT / 'data' / f'{name}.jsonl').is_file() for name in SPLIT_NAMES)
                         and (ROOT / 'data/synth').is_dir(),
                         'Optional private archive validation: real splits and synthetic augmentation are not distributed')
    def test_synthetic_rows_are_separate_from_real_splits(self):
        real = [json.loads(line) for name in ['train', 'calibration', 'test']
                for line in (ROOT / 'data' / (name + '.jsonl')).read_text().splitlines()]
        synthetic = [json.loads(line) for path in (ROOT / 'data/synth').glob('*.jsonl')
                     for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(synthetic), 200)
        self.assertFalse({r['prompt'] for r in real} & {r['prompt'] for r in synthetic})


if __name__ == '__main__':
    unittest.main()
