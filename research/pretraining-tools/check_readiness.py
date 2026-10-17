"""Verify this preparation snapshot; BLOCKED is expected, never launches anything."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def original_inputs_unchanged(before):
    allowed = {'configs/v3c.json', 'snapshot-manifest.json', 'model-manifest.json',
               'data/tasks.jsonl', *(f'data/expert-recovery-{n}.jsonl' for n in range(1,5)),
               *(base+'/'+split+'.jsonl' for base in ('data','data/split-v2') for split in ('train','calibration'))}
    if not isinstance(before, dict) or set(before) != allowed:
        raise ValueError('integrity manifest input scope differs')
    checked = []
    for name in sorted(allowed):
        path = ROOT/name
        resolved = path.resolve()
        if resolved != path or not resolved.is_relative_to(ROOT) or 'test.jsonl' in resolved.name:
            raise ValueError('integrity input path refused')
        digest = before[name]
        if not isinstance(digest,str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('integrity digest malformed')
        checked.append((path,digest))
    # Validate every key/path/digest BEFORE opening any original input bytes.
    return all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path,digest in checked)

def decision_progress(expected, reviewed):
    from collections import Counter
    identifiers = [row.get('review_id') if isinstance(row,dict) else None for row in reviewed]
    permitted = [key for key in identifiers if isinstance(key,str) and key in expected]
    frequencies = Counter(permitted)
    identities_valid = len(permitted) == len(reviewed) and len(frequencies) == len(reviewed)
    completed = set()
    for row in reviewed:
        if not isinstance(row,dict):
            continue
        key = row.get('review_id')
        if not isinstance(key,str) or key not in expected or frequencies[key] != 1:
            continue
        valid = (row.get('label_status') == 'resolved' and row.get('tier') in ('fast','balanced','flagship')
                 and row.get('needs_vision') in ('yes','no')
                 and row.get('task_type') in ('extraction','research','implementation','debugging','review','creative')
                 and isinstance(row.get('reviewer'),str) and bool(row['reviewer'].strip())
                 and isinstance(row.get('vision_evidence'),str) and bool(row['vision_evidence'].strip())
                 and isinstance(row.get('action_evidence'),str) and bool(row['action_evidence'].strip()))
        if valid:
            completed.add(key)
    return identities_valid, len(completed)

def assess():
    prepared = HERE / 'prepared'
    lock = json.loads((prepared/'SOURCE-LOCK.json').read_text())
    import subprocess
    public = ROOT.parent/'router-training-public'
    reviewed_sha = 'cd058a8093977865691b867c360bfc64b4e29f88'
    tree = subprocess.check_output(['git','-C',str(public),'ls-tree','-rz',reviewed_sha])
    expected_blobs = {}
    for entry in tree.split(b'\0'):
        if entry:
            metadata, name = entry.split(b'\t',1)
            expected_blobs[name.decode()] = metadata.decode().split()[2]
    source_exact = lock['public_source_sha'] == reviewed_sha and set(expected_blobs) == set(lock['sha256_by_file'])
    candidate = (prepared/'code-candidate').resolve()
    for name, blob in expected_blobs.items():
        path = candidate/name
        if not path.resolve().is_relative_to(candidate) or path.is_symlink():
            source_exact = False
            continue
        data = path.read_bytes()
        source_exact = source_exact and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == blob
        source_exact = source_exact and hashlib.sha256(data).hexdigest() == lock['sha256_by_file'][name]
    before = json.loads((HERE/'local-integrity-before.json').read_text())
    originals_unchanged = original_inputs_unchanged(before)
    tasks = rows(prepared/'review/tasks-blinded.PRIVATE.jsonl')
    template = rows(prepared/'review/decisions-pending.jsonl')
    expected = {row['review_id'] for row in template}
    wrapper_valid = all(set(row) == {'review_id','prompt'} for row in tasks)
    template_valid = len(expected) == len(template) == len(tasks) and {row['review_id'] for row in tasks} == expected
    decisions_path = prepared/'review/decisions-reviewed.PRIVATE.jsonl'
    reviewed = rows(decisions_path) if decisions_path.exists() else []
    valid_decisions, resolved = decision_progress(expected, reviewed)
    appendix = json.loads((prepared/'aggregate-appendix.json').read_text())
    status = {
        'status': 'BLOCKED', 'formal_training_ready': False,
        'source_128_files_exact': source_exact and lock['files'] == 128,
        'original_input_and_frozen_manifest_bytes_unchanged': originals_unchanged,
        'blind_wrapper_and_identity_integrity': wrapper_valid and template_valid and valid_decisions,
        'development_review_total': len(expected), 'development_reviews_resolved_structurally_only': resolved,
        'development_reviews_pending': len(expected)-resolved,
        'reviewer_truth_and_independence_not_machine_authenticated': True,
        'original_issue18_test_cases_not_reviewed_by_this_package': True,
        'positive_original_roots': appendix['default_protocol']['current_positive_original_roots_in_full_source'],
        'ABC_minimum_positive_roots_necessary_lower_bound': 6,
        'audited_environments_matching_all_pins': [name for name, item in appendix['environment_metadata'].items() if item['all_pinned_packages_present_and_version_match']],
        'blockers': [
            'Real root-support deficit: no approved whole-root A/B/C plan; root fragmentation forbidden.',
            'Conflicting cached labels and unverified provenance; independent local review pending.',
            'Semantic schema, supported entrypoint, dataset version, numeric policy and matched budget not approved.',
            'No inspected environment matches all pinned requirements; private .venv absent at audit.',
            'Actual model/adapter/reload/CUDA memory compatibility not tested or authorized.',
            'Formal training not authorized by a request for preparation; no quality improvement proven.'
        ],
        'cannot_turn_this_snapshot_green_by_editing_approval_booleans': True,
        'no_test_evaluation_no_training_no_GPU': True,
    }
    return status

if __name__ == '__main__':
    try:
        print(json.dumps(assess(), ensure_ascii=False, indent=2))
        raise SystemExit(1)  # Deliberate fail-closed: this snapshot is not an approved experiment.
    except Exception as error:
        print(json.dumps({'status':'BLOCKED','error_type':type(error).__name__,'details_withheld_for_privacy':True}))
        raise SystemExit(2)
