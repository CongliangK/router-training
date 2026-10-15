#!/usr/bin/env python3
"""Wholly invented CPU protocol demonstration, Refs #21, Refs #26.

No data/model path arguments, external executors, artifacts or gate writes.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oof_protocol import development_folds, nested_plan, crc_threshold
from cascade_protocol import sample_anchors, preregister, simulate, simulation_stress


def demonstration(repetitions=50):
    rows = [{'id': f'synthetic-{i}-{j}', 'root_id': f'root-{i}', 'split_group': f'split-{i}',
             'source_split': 'train' if i % 2 else 'calibration',
             'label': {'tier': ('fast', 'balanced', 'flagship')[j],
                       'needs_vision': 'yes' if j == 0 else 'no'}} for i in range(30) for j in range(3)]
    outer = development_folds(rows, root_audit_ref='synthetic-declaration-only')
    roles = []
    for fold in outer['folds']:
        roots = fold['outside_roots']
        roles.append({'fit': roots[:-3], 'A': roots[-3:-2], 'B': roots[-2:-1], 'C': roots[-1:]})
    nested = nested_plan(rows, roles, root_audit_ref='synthetic-declaration-only')
    crc = crc_threshold([{'root_id': f'unit-{i}', 'score': .8, 'true_flagship': True} for i in range(19)],
                        alpha=.05, estimand='joint_flagship_miss', context={'prediction_source': 'oof'})
    frame = [{'id': f'anchor-{i}-{j}', 'root_id': f'root-{i}', 'split_group': f'split-{i}',
              'duplicate_cluster': f'duplicate-{i}-{j}', 'source_split': 'synthetic',
              'tier': ('fast', 'balanced', 'flagship')[i % 3]} for i in range(16) for j in range(2)]
    selected = sample_anchors(frame, screening_roots=4, confirmation_roots=12, tasks_per_root=2, seed=17)
    candidates = [{'id': tier, 'provider': 'synthetic', 'model': f'synthetic-{tier}',
                   'revision': 'synthetic-v1', 'effort': 'fixed', 'tier': tier,
                   'acceptance_digest': 'a'*64, 'context_policy_digest': 'b'*64}
                  for tier in ('fast', 'balanced', 'flagship')]
    cascade = preregister(selected, candidates, protocol_id='synthetic-demo-v1',
                          root_audit_ref='synthetic-only', acceptance_ref='synthetic-only',
                          confirmation_schedule_ref='synthetic-schedule-only', explore_trials=2, confirmation_trials=3, alpha=.05, reliability_target=.9)
    probabilities = {f'root-{i}': {c['id']: .6 for c in candidates} for i in range(16)}
    return {'synthetic': True, 'qualified': False,
            'oof': {'status': 'exploratory', 'qualified': False, 'row_count': nested['row_count'],
                    'root_count': nested['root_count'],
                    'fold_training_root_counts': [f['actual_training_root_count'] for f in nested['folds']]},
            'crc': crc, 'cascade': simulate(cascade, probabilities, seed=23),
            'null_stress': simulation_stress(cascade, probabilities, repetitions=repetitions, seed=31)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10000:
        parser.error('repetitions must be in [1,10000]')
    print(json.dumps(demonstration(args.repetitions), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
