"""Local-only pretraining inventory. Never emits task text, IDs or root names.
No model imports, test-split reads, training, network or CUDA operations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC = ROOT.parent / 'router-training-public'
ENUMS = {
    'tier': ('fast', 'balanced', 'flagship'),
    'needs_vision': ('yes', 'no', 'uncertain'),
    'task_type': ('extraction', 'research', 'implementation', 'debugging', 'review', 'visual', 'creative'),
    'label_source': ('expert_provisional', 'expert_reviewed', 'measured'),
    'label_status': ('resolved', 'adjudication', 'excluded', 'pending'),
    'task_label_schema_version': ('router-label-v1', 'router-label-v2'),
}

def safe_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or 'test.jsonl' in path.name:
        raise ValueError('unauthorized input')
    return path

def read_rows(relative):
    path = safe_path(relative)
    if not path.is_file():
        return None
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('invalid row type')
                rows.append(row)
    return rows

def label(row):
    return row['label'] if isinstance(row.get('label'), dict) else row

def explicit_synthetic(row):
    """Preserve an explicit true flag at either record or nested-label level."""
    return row.get('synthetic') is True or label(row).get('synthetic') is True

def recovery_rows(task_rows, recovery):
    by_id = {row.get('id'): row for row in task_rows}
    result = []
    for row in recovery:
        task = by_id.get(row.get('id'), {})
        declarations = [value for value in (task.get('task_label_schema_version'), row.get('task_label_schema_version')) if value is not None]
        selected = declarations[0] if declarations else None
        if any(value != selected for value in declarations):
            selected = 'conflicting'
        result.append({'id': row.get('id'), 'group': task.get('group'), 'label': row,
                       'synthetic': explicit_synthetic(task) or explicit_synthetic(row),
                       'task_label_schema_version': selected})
    return result

def counts(rows, field):
    result = Counter()
    for row in rows:
        value = row.get(field) if field == 'task_label_schema_version' else label(row).get(field)
        key = value if isinstance(value, str) and value in ENUMS[field] else ('missing' if value is None else 'other')
        result[key] += 1
    return dict(sorted(result.items()))

def identities(rows, key):
    return [row[key] for row in rows if isinstance(row.get(key), str) and row[key]]

def summary(rows):
    ids = identities(rows, 'id')
    groups = identities(rows, 'group')
    components = identities(rows, 'split_group')
    group_sizes = Counter(groups)
    component_sizes = Counter(components)
    support = {}
    for field in ('tier', 'needs_vision', 'task_type'):
        support[field] = {}
        for value in ENUMS[field]:
            matching = [row for row in rows if label(row).get(field) == value and not explicit_synthetic(row)]
            support[field][value] = {'rows': len(matching), 'roots': len(set(identities(matching, 'group'))), 'components': len(set(identities(matching, 'split_group')))}
    return {
        'rows': len(rows), 'duplicate_id_rows': len(ids) - len(set(ids)),
        'missing_ids': len(rows) - len(ids), 'roots': len(set(groups)),
        'components': len(set(components)), 'missing_component_rows': len(rows) - len(components),
        'largest_root_rows': max(group_sizes.values(), default=0),
        'largest_component_rows': max(component_sizes.values(), default=0),
        'synthetic_true_rows': sum(explicit_synthetic(row) for row in rows),
        'labels': {field: counts(rows, field) for field in ENUMS},
        'non_synthetic_support_not_authentication': support,
    }

def inventory():
    task_rows = read_rows('data/tasks.jsonl') or []
    recovery = []
    for n in range(1, 5):
        recovery.extend(read_rows(f'data/expert-recovery-{n}.jsonl') or [])
    task_ids, label_ids = set(identities(task_rows, 'id')), set(identities(recovery, 'id'))
    report = {
        'formal_training_ready': False, 'test_split_opened': False,
        'model_or_weights_loaded': False, 'training_or_cuda_executed': False,
        'source_tasks': {'rows': len(task_rows), 'roots': len(set(identities(task_rows, 'group'))), 'largest_root_rows': max(Counter(identities(task_rows, 'group')).values(), default=0)},
        'expert_recovery': summary(recovery_rows(task_rows, recovery)),
        'full_existing_source_label_pool_counted': True,
        'new_pristine_holdout_not_established': True,
        'source_coverage': {'tasks_missing_label': len(task_ids-label_ids), 'labels_missing_task': len(label_ids-task_ids)},
        'datasets': {},
    }
    for relative in ('data', 'data/split-v2'):
        train = read_rows(relative + '/train.jsonl')
        calibration = read_rows(relative + '/calibration.jsonl')
        if train is None or calibration is None:
            report['datasets'][relative] = {'available': False}
            continue
        report['datasets'][relative] = {
            'available': True, 'train': summary(train), 'calibration': summary(calibration),
            'overlap_counts_train_calibration': {key: len(set(identities(train, key)) & set(identities(calibration, key))) for key in ('id', 'group', 'split_group')},
        }
    code_files = ('train_eval.py', 'model_utils.py', 'prepare.py', 'calibration_groups.py', 'scripts/train.py', 'requirements.txt')
    report['private_public_code_bytes_equal'] = {name: (ROOT / name).is_file() and (PUBLIC / name).is_file() and (ROOT / name).read_bytes() == (PUBLIC / name).read_bytes() for name in code_files}
    # Private integrity evidence stays local and is not printed; no test bytes read.
    fixed = ['configs/v3c.json', 'snapshot-manifest.json', 'model-manifest.json']
    data_paths = ['data/tasks.jsonl'] + [f'data/expert-recovery-{n}.jsonl' for n in range(1,5)]
    data_paths += [base + '/' + name + '.jsonl' for base in ('data', 'data/split-v2') for name in ('train', 'calibration')]
    integrity = {name: hashlib.sha256(safe_path(name).read_bytes()).hexdigest() for name in fixed + data_paths if safe_path(name).is_file()}
    for name, value in [('local-integrity-before.json', integrity), ('aggregate-audit.json', report)]:
        with (HERE / name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

def environment():
    packages = ('torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'safetensors', 'huggingface-hub', 'tokenizers', 'zstandard', 'fastapi', 'uvicorn', 'pydantic', 'Pillow', 'numpy', 'bitsandbytes')
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {'python': sys.version.split()[0], 'versions': versions, 'method': 'distribution metadata only; no torch/model/CUDA imports', 'requirements_match': {}}
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            name, expected = line.strip().split('==', 1)
            result['requirements_match'][name] = versions.get(name) == expected
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', action='store_true')
    args = parser.parse_args()
    try:
        environment() if args.environment else inventory()
    except Exception as error:
        # No exception messages/tracebacks: they may contain raw private row data.
        print(json.dumps({'ok': False, 'error_type': type(error).__name__, 'detail_withheld_for_privacy': True}))
        raise SystemExit(1)
