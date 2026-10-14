"""Launch training runs without overwriting existing artifacts.

Two explicit modes (issue #2; no silent fallback between them):
- --legacy: the archived v3c recipe with its ORIGINAL row-split calibration,
  passed through as --legacy-calibration. Reproduction only.
- default (grouped): future runs use group-isolated calibration with
  PRE-REGISTERED A/B/C role groups (--roles). --dry-run in this mode performs
  the real CPU structural preflight via scripts/validate_calibration.py —
  no model, no artifacts, no test access — instead of just printing a command.
"""
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
    parser.add_argument('--smoke', action='store_true', help='Short GPU pipeline check (still evaluates real test rows in legacy mode)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Legacy: print the command. Grouped: run the real CPU structural preflight (no model, no artifacts, no test access)')
    parser.add_argument('--legacy', action='store_true',
                        help='Reproduce the archived v3c recipe with the original row-split calibration (explicit legacy; issue #2)')
    parser.add_argument('--roles', default=None,
                        help='Roles config JSON with pre-registered A/B/C groups; required for grouped (non-legacy) runs')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'),
                        help='Data directory for the grouped structural preflight (train.jsonl + calibration.jsonl; test.jsonl is never read)')
    args = parser.parse_args()
    if not args.run.replace('-', '').replace('_', '').isalnum():
        parser.error('Invalid run name')
    if (ROOT / 'artifacts' / args.run).exists():
        parser.error('Output already exists; choose a new run name')

    if args.legacy:
        if args.roles:
            parser.error('--roles configures grouped runs; --legacy reproduces the archived v3c recipe unchanged')
        recipe = json.loads((ROOT / 'configs/v3c.json').read_text())
        command = [sys.executable, str(ROOT / 'train_eval.py'), '--run', args.run, '--legacy-calibration']
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

    if not args.roles:
        parser.error('New runs use group-isolated calibration (issue #2): pass --roles <roles.json>, '
                     'or pass --legacy to reproduce the archived v3c recipe')
    roles_path = Path(args.roles)
    if not roles_path.is_file():
        parser.error(f'Roles config not found: {roles_path}')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_calibration import run_validation
    ok, text = run_validation(args.data_dir, roles_path)
    print('Structural preflight (CPU, anonymous, no model/artifacts/test):')
    print(text, flush=True)
    command = [sys.executable, str(ROOT / 'train_eval.py'), '--run', args.run,
               '--calibration-roles', str(roles_path)]
    if args.smoke:
        command.append('--smoke')
    print('Command:', command, flush=True)
    if args.dry_run:
        return 0 if ok else 1
    if not ok:
        return 1
    env = os.environ.copy()
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('PYTHONUNBUFFERED', '1')
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
