"""Launch the archived v3c recipe without overwriting the winner."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='New output name; existing runs are refused')
    parser.add_argument('--smoke', action='store_true', help='Short GPU pipeline check (still evaluates real test rows)')
    parser.add_argument('--dry-run', action='store_true', help='Print command without loading the model')
    args = parser.parse_args()
    if not args.run.replace('-', '').replace('_', '').isalnum():
        parser.error('Invalid run name')
    if (ROOT / 'artifacts' / args.run).exists():
        parser.error('Output already exists; choose a new run name')
    recipe = json.loads((ROOT / 'configs/v3c.json').read_text())
    command = [sys.executable, str(ROOT / 'train_eval.py'), '--run', args.run]
    for key in ['epochs', 'rank', 'lr', 'accumulation', 'seed', 'beta', 'label_smoothing']:
        command.extend(['--' + key.replace('_', '-'), str(recipe[key])])
    for key in ['synth', 'mlp', 'fullft', 'swa']:
        if recipe[key]:
            command.append('--' + key)
    if args.smoke:
        command.append('--smoke')
    print('Command:', command, flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('PYTHONUNBUFFERED', '1')
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
