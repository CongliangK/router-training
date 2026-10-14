"""Launch training runs without overwriting existing artifacts.

Two explicit modes (issue #2; no silent fallback between them):
- --legacy: the archived v3c recipe with its ORIGINAL row-split calibration,
  passed through as --legacy-calibration. Reproduction only.
- default (grouped): future runs use group-isolated calibration with
  PRE-REGISTERED A/B/C role groups (--roles). --dry-run in this mode performs
  the real CPU structural preflight via scripts/validate_calibration.py —
  no model, no artifacts, no test access — instead of just printing a command.
  --data-dir configures that preflight only; an actual launch always reads
  the repository data/ directory (train_eval.py takes no data-dir), so a
  non-default --data-dir is refused at launch and allowed only with --dry-run.

Issue #14: BOTH modes now pass the complete v3c training recipe explicitly
through one shared builder reading configs/v3c.json. The grouped entry point
previously forwarded only --run/--calibration-roles and silently trained on
train_eval.py defaults (synth/fullft/swa off); a grouped run now declares
deviations from v3c ONLY through an explicit --recipe JSON override (unknown
or wrongly-typed keys are refused), the deviations are echoed to the console,
and the --dry-run Command lists every recipe flag actually passed.

Issue #14b: the optimizer is a recorded recipe choice too — --recipe may
declare {"optimizer": "adamw8bit"} (enum-validated). The key is absent from
configs/v3c.json (frozen archive; the v3c run used train_eval.py's adamw
default), so the flag is passed ONLY when declared and archived-reproduction
commands stay byte-identical.
"""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / 'data'
RECIPE_PATH = ROOT / 'configs/v3c.json'
# configs/v3c.json keys that map to train_eval.py CLI flags. Everything else
# in the file (description, run name, manifests, prompt budgets) is archive
# metadata, not overridable training input.
RECIPE_SCALAR_KEYS = ['epochs', 'rank', 'lr', 'accumulation', 'seed', 'beta', 'label_smoothing']
RECIPE_FLAG_KEYS = ['synth', 'mlp', 'fullft', 'swa']
# Issue #14b enum keys: NOT present in configs/v3c.json (frozen archive), so
# each entry carries the value the archived v3c effectively used (the
# train_eval.py default) plus the allowed values. The flag reaches
# train_eval.py only when explicitly declared via --recipe.
RECIPE_ENUM_KEYS = {'optimizer': ('adamw', ('adamw', 'adamw8bit'))}


def data_dir_guard(args, parser):
    """Red-team F4: --data-dir feeds ONLY the structural preflight;
    train_eval.py always reads ROOT/data. An actual grouped launch with a
    non-default --data-dir would preflight one dataset and train on another,
    so it is refused outright. --dry-run (preflight only) may use any data
    directory, e.g. a synthetic fixture."""
    if not args.legacy and not args.dry_run \
            and Path(args.data_dir).resolve() != DEFAULT_DATA_DIR.resolve():
        parser.error('--data-dir is preflight-only: the actual grouped run always reads the '
                     f"repository data directory ({DEFAULT_DATA_DIR}); train_eval.py takes no data-dir. "
                     'Use the default for real launches, or combine a custom --data-dir with --dry-run')


def recipe_flags(recipe):
    """The COMPLETE training recipe as explicit train_eval.py flags. ONE
    builder assembles both launcher modes' commands (issue #14), so neither
    the legacy nor the grouped entry point can silently fall back to
    train_eval.py defaults again."""
    flags = []
    for key in RECIPE_SCALAR_KEYS:
        flags.extend(['--' + key.replace('_', '-'), str(recipe[key])])
    for key in RECIPE_FLAG_KEYS:
        if recipe[key]:
            flags.append('--' + key)
    # Issue #14b enum keys: emitted ONLY when declared (they have no v3c.json
    # entry to default from), so undeclared runs keep the train_eval.py
    # default — adamw — and archived-reproduction commands stay unchanged.
    for key in RECIPE_ENUM_KEYS:
        if key in recipe:
            flags.extend(['--' + key.replace('_', '-'), recipe[key]])
    return flags


def merged_recipe(recipe, overrides_path, parser):
    """v3c defaults + explicitly declared deviations (issue #14). The override
    file may carry only known recipe keys, each matching the v3c value's type
    (bools are never confused with ints) and finite (JSON's NaN/Infinity
    literals are refused); anything else is refused rather than guessed.
    Returns the merged recipe and the declared deviations."""
    if overrides_path is None:
        return recipe, {}
    try:
        overrides = json.loads(Path(overrides_path).read_text())
    except (OSError, ValueError) as error:
        parser.error(f'--recipe must be a readable JSON object: {error}')
    if not isinstance(overrides, dict) or not overrides:
        parser.error('--recipe must be a non-empty JSON object of recipe overrides')
    allowed = set(RECIPE_SCALAR_KEYS) | set(RECIPE_FLAG_KEYS) | set(RECIPE_ENUM_KEYS)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        parser.error(f'--recipe has unknown keys {unknown}; overridable keys: {sorted(allowed)}')
    merged = dict(recipe)
    deviations = {}
    for key, value in overrides.items():
        if key in RECIPE_ENUM_KEYS:
            # Issue #14b: enum keys validate against their allowed values; a
            # wrong value or wrong type is refused rather than guessed.
            v3c_value, choices = RECIPE_ENUM_KEYS[key]
            if not (isinstance(value, str) and value in choices):
                parser.error(f'--recipe key {key!r} must be one of {list(choices)}')
            merged[key] = value
            deviations[key] = {'v3c': v3c_value, 'declared': value}
            continue
        expected = recipe[key]
        if isinstance(expected, bool):
            valid = isinstance(value, bool)
        elif isinstance(expected, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:  # float keys (lr, beta, label_smoothing): plain ints are fine
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid:
            parser.error(f'--recipe key {key!r} must match the v3c value type '
                         f'({expected!r}: {type(expected).__name__})')
        # Red-team LOW-1: json.loads happily parses the JSON NaN/Infinity
        # literals into float nan/inf, which would silently poison optimizer
        # hyperparameters (lr=inf) and later report JSON. Ints cannot be
        # non-finite, so only float values need the check. The value itself is
        # deliberately NOT echoed — only the key and the reason — so a crafted
        # override file cannot inject arbitrary text into the error path.
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f'--recipe key {key!r} must be a finite number; '
                         'NaN and Infinity are not valid recipe values')
        merged[key] = value
        deviations[key] = {'v3c': expected, 'declared': value}
    return merged, deviations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='New output name; existing runs are refused')
    parser.add_argument('--smoke', action='store_true',
                        help='Short GPU pipeline check (legacy: 24 train rows + real test rows; '
                             'grouped: 24 train rows + whole-group calibration sub-pool, gates unchanged)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Legacy: print the command. Grouped: run the real CPU structural preflight (no model, no artifacts, no test access)')
    parser.add_argument('--legacy', action='store_true',
                        help='Reproduce the archived v3c recipe with the original row-split calibration (explicit legacy; issue #2)')
    parser.add_argument('--roles', default=None,
                        help='Roles config JSON with pre-registered A/B/C groups; required for grouped (non-legacy) runs')
    parser.add_argument('--recipe', default=None,
                        help='Grouped runs only: JSON file of recipe overrides over the v3c defaults from '
                             'configs/v3c.json (e.g. {"lr": 1e-4} or {"optimizer": "adamw8bit"}); deviations '
                             'are echoed and appear in the printed Command. Refused with --legacy, which '
                             'reproduces v3c unchanged')
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR),
                        help='Data directory for the grouped structural preflight ONLY (train.jsonl + '
                             'calibration.jsonl; test.jsonl is never read). The actual run always reads the '
                             'repository data/ directory, so a non-default --data-dir is refused at launch '
                             'and allowed only together with --dry-run')
    args = parser.parse_args()
    if not args.run.replace('-', '').replace('_', '').isalnum():
        parser.error('Invalid run name')
    if (ROOT / 'artifacts' / args.run).exists():
        parser.error('Output already exists; choose a new run name')
    data_dir_guard(args, parser)

    if args.legacy:
        if args.roles:
            parser.error('--roles configures grouped runs; --legacy reproduces the archived v3c recipe unchanged')
        if args.recipe:
            parser.error('--recipe declares grouped-run deviations; --legacy reproduces the archived v3c recipe unchanged')
        recipe = json.loads(RECIPE_PATH.read_text())
        command = [sys.executable, str(ROOT / 'train_eval.py'), '--run', args.run, '--legacy-calibration']
        command.extend(recipe_flags(recipe))
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
    # Issue #14: the grouped command carries the COMPLETE recipe — v3c defaults
    # plus any explicitly declared --recipe deviations — so a grouped run can
    # never silently train on train_eval.py defaults again.
    recipe, deviations = merged_recipe(json.loads(RECIPE_PATH.read_text()), args.recipe, parser)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_calibration import run_validation
    ok, text = run_validation(args.data_dir, roles_path)
    print('Structural preflight (CPU, anonymous, no model/artifacts/test):')
    print(text, flush=True)
    command = [sys.executable, str(ROOT / 'train_eval.py'), '--run', args.run,
               '--calibration-roles', str(roles_path)]
    command.extend(recipe_flags(recipe))
    if args.smoke:
        command.append('--smoke')
    if deviations:
        print('Recipe overrides vs configs/v3c.json:', json.dumps(deviations, sort_keys=True), flush=True)
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
