"""CPU fixed-weight tau-only development sweep (#1); no weights are loaded.

Requires strict raw logits, a bound pre-oversampling train prior, a versioned
relative tier cost table, and an explicit protocol. No test mode or implicit
repository asset reads. This cannot establish a training-side causal effect.
Exit 0: a development candidate selected; 2: registered gates infeasible;
1: invalid inputs or exclusive-output failure. See docs/OFFLINE-CALIBRATION.md.
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_groups import PlanError
from raw_logit_contract import load_json, write_report
from offline_calibration import tau_ablation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logits', required=True, help='Strict raw tier-logit source JSON')
    parser.add_argument('--prior', required=True, help='Bound pre-oversampling train-prior JSON')
    parser.add_argument('--costs', required=True, help='Versioned explicit relative tier cost JSON (not billing)')
    parser.add_argument('--protocol', required=True, help='Predeclared tau comparison/selection protocol JSON')
    parser.add_argument('--output', required=True, help='New aggregate report JSON; overwrite refused')
    args = parser.parse_args(argv)
    try:
        report = tau_ablation(load_json(args.logits), load_json(args.prior), load_json(args.costs), load_json(args.protocol))
        write_report(args.output, report)
    except PlanError as error:
        print(json.dumps({'status': 'invalid', 'reason': str(error)}))
        return 1
    print(json.dumps({'status': report['status'], 'test_used': False}))
    return 0 if report['status'] == 'available' else 2


if __name__ == '__main__':
    raise SystemExit(main())
