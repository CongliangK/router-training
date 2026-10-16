"""Pure-CPU anonymous structural validation of a calibration roles config.

Loads ONLY train.jsonl and calibration.jsonl from --data-dir (it never opens
test.jsonl), validates rows / pre-registered A/B/C roles / support policy /
deterministic folds through calibration_groups, and additionally checks that
the real unique train rows support every adjusted head's prior. The business
schema is NOT configurable: it is parsed from the same model_utils.py source
the run imports (questions_from_source), and a roles config that tries to
override the schema (or carries unknown keys) is rejected — preflight and
run share one truth. Output is a counts-only JSON summary (no group names,
ids, labels, prompts, or private paths). No model, no GPU, no artifacts, no
network. Exit 0 = structurally valid; exit 1 = violation (the error carries
reasons and counts only).

This is the same preflight that train_eval.py runs before load_engine() and
that scripts/train.py --dry-run performs for grouped launches, so all three
entry points report the same truth.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_label_schema import LEGACY, VERSIONS, validate_supervised_rows, validate_dataset_schema, training_supplemental
from calibration_groups import PlanError, compute_prior, preflight_roles, anonymous_summary, validate_partition_isolation


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_validation(data_dir, roles_path, schema_version=LEGACY, supplemental_paths=None):
    """Returns (ok, text). data_dir must contain train.jsonl and
    calibration.jsonl; test.jsonl is intentionally never read."""
    data_dir = Path(data_dir)
    roles_path = Path(roles_path)
    try:
        validate_dataset_schema(data_dir,schema_version)
        train_rows = _read_jsonl(data_dir / 'train.jsonl')
        calibration_rows = _read_jsonl(data_dir / 'calibration.jsonl')
        isolation = validate_partition_isolation(train_rows, calibration_rows)
        roles_config = json.loads(roles_path.read_text())
        validate_supervised_rows(train_rows,schema_version)
        additions=training_supplemental(supplemental_paths,train_rows+calibration_rows,schema_version)
        plan = preflight_roles(calibration_rows, roles_config,schema_version=schema_version)
        priors = {}
        for head, spec in plan['schema']['heads'].items():
            if spec.get('adjust'):
                prior = compute_prior(train_rows, head, plan['schema'])
                priors[head] = {'counts': prior['counts'], 'add_one': prior['add_one'],
                                'excluded_synthetic': prior['excluded_synthetic'],
                                'excluded_duplicate': prior['excluded_duplicate']}
        summary = anonymous_summary(plan)
        summary['root_isolation'] = isolation
        summary['priors'] = priors
        summary['supplemental_rows']=len(additions)
        return True, json.dumps({'ok': True, 'plan': summary}, ensure_ascii=False, indent=2)
    except OSError:
        return False,json.dumps({'ok':False,'error':'preflight input cannot be read (path withheld)'})
    except ValueError as error:
        return False, json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(ROOT / 'data'),
                        help='Directory holding train.jsonl and calibration.jsonl (test.jsonl is never read)')
    parser.add_argument('--roles', required=True, help='Roles config JSON with pre-registered A/B/C groups')
    parser.add_argument('--label-schema',choices=VERSIONS,default=LEGACY)
    parser.add_argument('--supplemental',type=Path,action='append',help='Validate explicit v2 train-only supplements without using them for support')
    args = parser.parse_args()
    ok, text = run_validation(args.data_dir, args.roles,args.label_schema,args.supplemental)
    print(text)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
