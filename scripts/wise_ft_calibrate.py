"""CPU nested LOGO logit ensemble research (#22), NOT WiSE-FT weight interpolation.

Explicit raw-logit JSON inputs only; no model, default data path, training,
network, test mode, serving product, or weight soup. See docs/OFFLINE-CALIBRATION.md.
Exit 0: complete nested development estimate; 2: support/constraint infeasible;
1: invalid inputs or exclusive-output failure. Reports contain aggregates only.
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_groups import PlanError
from raw_logit_contract import load_json, write_report
from offline_calibration import nested_logo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Strict raw tier-logit source JSON')
    parser.add_argument('--finetuned', required=True, help='Exactly aligned raw tier-logit source JSON')
    parser.add_argument('--protocol', required=True, help='Explicit versioned nested LOGO protocol JSON')
    parser.add_argument('--output', required=True, help='New aggregate report JSON; overwrite refused')
    args = parser.parse_args(argv)
    try:
        report = nested_logo(load_json(args.base), load_json(args.finetuned), load_json(args.protocol))
        write_report(args.output, report)
    except PlanError as error:
        print(json.dumps({'status': 'invalid', 'reason': str(error)}))
        return 1
    print(json.dumps({'status': report['status'], 'test_used': False}))
    return 0 if report['status'] == 'available' else 2


if __name__ == '__main__':
    raise SystemExit(main())
