"""CPU-only checks for the standalone snapshot and recipe launcher."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
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
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                                 '--run', 'repository-validation-only', '--dry-run'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ['--fullft', '--synth', '--swa']:
            self.assertIn(flag, result.stdout)
        self.assertFalse((ROOT / 'artifacts/repository-validation-only').exists())

    def test_launcher_refuses_overwriting_winner(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                                 '--run', 'real-history-v3c', '--dry-run'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output already exists', result.stderr)

    def test_synthetic_rows_are_separate_from_real_splits(self):
        real = [json.loads(line) for name in ['train', 'calibration', 'test']
                for line in (ROOT / 'data' / (name + '.jsonl')).read_text().splitlines()]
        synthetic = [json.loads(line) for path in (ROOT / 'data/synth').glob('*.jsonl')
                     for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(synthetic), 200)
        self.assertFalse({r['prompt'] for r in real} & {r['prompt'] for r in synthetic})


if __name__ == '__main__':
    unittest.main()
