"""Build local-only, non-launchable preparation evidence and blinded dev review.
Never reads test.jsonl, scores a model, assigns a label, or exports private rows.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('preparation_inventory', HERE / 'audit_local.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
ROOT, PUBLIC = audit.ROOT, audit.PUBLIC
FIELDS = ('tier', 'needs_vision', 'task_type')

def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

def spread(ids, tasks, count, rng):
    buckets = defaultdict(list)
    for key in sorted(ids):
        buckets[tasks[key]['group']].append(key)
    roots = sorted(buckets)
    rng.shuffle(roots)
    for values in buckets.values():
        rng.shuffle(values)
    result = []
    while len(result) < count:
        progressed = False
        for root in roots:
            if buckets[root] and len(result) < count:
                result.append(buckets[root].pop())
                progressed = True
        if not progressed:
            break
    return result

def main():
    import subprocess
    public_head = subprocess.check_output(['git', '-C', str(PUBLIC), 'rev-parse', 'HEAD'], text=True).strip()
    public_status = subprocess.check_output(['git', '-C', str(PUBLIC), 'status', '--porcelain'], text=True).strip()
    if public_head != 'cd058a8093977865691b867c360bfc64b4e29f88' or public_status:
        raise ValueError('public source binding changed')
    integrity = json.loads((HERE / 'local-integrity-before.json').read_text())
    for relative, digest in integrity.items():
        if hashlib.sha256(audit.safe_path(relative).read_bytes()).hexdigest() != digest:
            raise ValueError('input changed since inventory')
    output = HERE / 'prepared'
    if output.exists():
        raise FileExistsError('fresh output required')
    output.mkdir(mode=0o700)
    tasks_list = audit.read_rows('data/tasks.jsonl')
    labels_list = sum((audit.read_rows(f'data/expert-recovery-{n}.jsonl') for n in range(1,5)), [])
    tasks = {row['id']: row for row in tasks_list}
    labels = {row['id']: row for row in labels_list}
    if len(tasks) != len(tasks_list) or len(labels) != len(labels_list) or set(tasks) != set(labels):
        raise ValueError('source identity invalid')
    legacy_train = audit.read_rows('data/train.jsonl')
    legacy_cal = audit.read_rows('data/calibration.jsonl')
    development_ids = {row['id'] for row in legacy_train + legacy_cal}
    eligible_development_ids = {key for key in development_ids if key in tasks and key in labels
                                and not audit.explicit_synthetic(tasks[key]) and not audit.explicit_synthetic(labels[key])}
    if not development_ids <= tasks.keys():
        raise ValueError('development identity mismatch')
    source_complement = tasks.keys() - development_ids
    report = {'formal_training_ready': False, 'test_split_opened': False,
              'complement_is_not_verified_test_membership': True,
              'full_existing_source_labels_counted_not_pristine_holdout': True,
              'datasets': {}, 'default_protocol': {}}
    critical_ids = set()
    reason_counts = Counter()
    for relative in ('data', 'data/split-v2'):
        train = audit.read_rows(relative + '/train.jsonl')
        calibration = audit.read_rows(relative + '/calibration.jsonl')
        detail = {}
        for split, rows in [('train', train), ('calibration', calibration)]:
            mismatches = Counter()
            roots_to_components = defaultdict(set)
            different_any = 0
            missing_source = 0
            prompt_changes = 0
            for row in rows:
                key = row.get('id')
                current = labels.get(key)
                if current is None:
                    missing_source += 1
                    continue
                different = [field for field in FIELDS if audit.label(row).get(field) != current.get(field)]
                mismatches.update(different)
                different_any += bool(different)
                prompt_changes += row.get('prompt') != tasks[key].get('prompt')
                roots_to_components[row.get('group')].add(row.get('split_group'))
                if relative == 'data' and key in eligible_development_ids and (different or audit.label(row).get('needs_vision') not in ('yes', 'no') or audit.label(row).get('task_type') == 'visual'):
                    critical_ids.add(key)
            detail[split] = {'rows': len(rows), 'rows_semantically_different_from_current_recovery': different_any,
                             'field_mismatch_counts': dict(mismatches), 'rows_without_current_recovery': missing_source,
                             'prompt_bytes_differ_from_current_source': prompt_changes,
                             'roots_partitioned_into_multiple_components': sum(len(values) > 1 for values in roots_to_components.values()),
                             'max_components_inside_one_root': max(map(len, roots_to_components.values()), default=0),
                             'rows_from_source_complement_of_historical_development': sum(row.get('id') in source_complement for row in rows)}
        report['datasets'][relative] = detail
    positive_roots = {tasks[key]['group'] for key in tasks if labels[key].get('needs_vision') == 'yes'
                      and not audit.explicit_synthetic(tasks[key]) and not audit.explicit_synthetic(labels[key])}
    report['default_protocol'] = {
        'current_positive_original_roots_in_full_source': len(positive_roots),
        'minimum_calibration_positive_roots_necessary_lower_bound': 3 * 2,
        'whole_root_ABC_feasible_under_current_default_policy': False if len(positive_roots) < 6 else None,
        'lower_bound_only_not_sufficient': True,
        'training_test_and_B_fold_requirements_can_need_more_roots': True,
        'policy_is_provisional_not_owner_approved': True,
        'never_fragment_roots_or_synthesize_support_to_pass': True,
    }
    # New development-only diagnostic selection, NOT the original issue18 16-case signoff.
    rng = random.Random(20261016)
    creative = {key for key in eligible_development_ids if labels[key].get('task_type') == 'creative'}
    controls = eligible_development_ids - creative
    creative_selected = spread(creative, tasks, 16, rng)
    controls_selected = spread(controls, tasks, 16, rng)
    selected = sorted(critical_ids | set(creative_selected) | set(controls_selected))
    rng.shuffle(selected)
    review_dir = output / 'review'
    review_dir.mkdir(mode=0o700)
    blinded, templates, mapping = [], [], []
    for index, key in enumerate(selected, 1):
        review_id = f'review-{index:04d}'
        blinded.append({'review_id': review_id, 'prompt': tasks[key]['prompt']})
        templates.append({'review_id': review_id, 'tier': None, 'needs_vision': None, 'task_type': None,
                          'primary_action': None, 'vision_evidence': '', 'action_evidence': '',
                          'reviewer': None, 'label_status': 'pending', 'unresolved_reason': None})
        reasons = []
        if key in critical_ids: reasons.append('development_label_conflict_or_legacy_ambiguity')
        if key in creative_selected: reasons.append('new_development_creative_diagnostic')
        if key in controls_selected: reasons.append('new_development_control')
        reason_counts.update(reasons)
        mapping.append({'review_id': review_id, 'task_id': key, 'selection_reasons': reasons})
    assert set(selected) <= development_ids
    write_jsonl(review_dir / 'tasks-blinded.PRIVATE.jsonl', blinded)
    write_jsonl(review_dir / 'decisions-pending.jsonl', templates)
    write_jsonl(review_dir / 'mapping-PRIVATE-NOT-FOR-REVIEWER.jsonl', mapping)
    report['review_package'] = {'selected': len(selected), 'pending': len(selected),
                                'reason_counts_overlap': dict(reason_counts), 'selected_from_historical_development_only': True,
                                'reviewer_fields': ['review_id', 'prompt'], 'prior_labels_and_predictions_in_reviewer_wrapper': False,
                                'candidate_predictions_never_read': True, 'original_issue18_16_cases_not_claimed_reviewed': True,
                                'intrinsic_prompt_content_requires_local_privacy_and_blinding_review': True,
                                'source_pool_has_no_explicit_split_components_so_zero_means_missing_not_zero_independent_roots': True}
    # Distribution metadata inspection only; do not execute/import any package in target environments.
    names = {'torch','torchvision','transformers','peft','accelerate','safetensors','huggingface-hub','tokenizers','zstandard','fastapi','uvicorn','pydantic','pillow','numpy','bitsandbytes'}
    from importlib.metadata import distributions
    environments = {}
    prefix = Path.home() / 'miniconda3'
    for name, env in [('base', prefix), ('langchain', prefix/'envs/langchain'), ('aitoolkit', prefix/'envs/aitoolkit')]:
        sites = list(env.glob('lib/python*/site-packages'))
        found = {}
        for dist in distributions(path=[str(path) for path in sites]):
            dist_name = (dist.metadata.get('Name') or '').lower().replace('_','-')
            if dist_name in names:
                found[dist_name] = dist.version
        matches = {}
        for line in (ROOT/'requirements.txt').read_text().splitlines():
            if '==' in line and not line.lstrip().startswith('#'):
                package, expected = line.strip().split('==', 1)
                actual = found.get(package.lower().replace('_','-'))
                matches[package] = bool(actual and actual.split('+',1)[0] == expected)
        environments[name] = {'distribution_versions': found, 'requirements_match': matches,
                              'all_pinned_packages_present_and_version_match': all(matches.values()),
                              'target_interpreter_or_libraries_not_executed': True,
                              'cuda_driver_kernel_memory_compatibility_not_checked': True}
    report['environment_metadata'] = environments
    roles = ROOT / 'pretraining-approved-roles.json'
    if roles.is_file():
        public_spec = importlib.util.spec_from_file_location('pretraining_public_validator', PUBLIC/'scripts/validate_calibration.py')
        validator = importlib.util.module_from_spec(public_spec)
        public_spec.loader.exec_module(validator)
        results = {}
        for relative in ('data', 'data/split-v2'):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ok, text = validator.run_validation(ROOT/relative, roles)
            with (output / ('preflight-detail-' + relative.replace('/', '-') + '.PRIVATE.json')).open('x', encoding='utf-8') as handle:
                handle.write(text + '\n')
            results[relative] = {'ok': ok, 'detail_kept_local_not_exported': True,
                                 'historical_roles_not_approved_for_new_experiment': True}
        report['existing_public_structural_preflight'] = results
    proposal = {
        'status': 'DRAFT_BLOCKED_NOT_A_TRAINING_CONFIG', 'training_authorized': False,
        'issue_refs': [2,18,19,20,24,25,27], 'semantic_schema_selected': None,
        'data_version_approved': None, 'role_protocol_approved': None, 'matched_update_budget_approved': None,
        'recommended_sequence': ['resolve label/provenance conflicts locally', 'collect genuinely independent vision-positive roots',
                                 'freeze whole-root development/heldout boundaries and metadata', 'approve six-class vs legacy-schema entrypoint',
                                 'freeze source/base/environment/data/roles and matched baseline', 'separately authorize model/resource compatibility checks',
                                 'only then consider a prospective one-factor experiment'],
        'one_factor_candidate_only': {'factor': 'ordinal_cost_lambda', 'baseline': 0, 'candidate': 0.25,
                                      'all_other_factors_must_match': True, 'conditional_on_explicit_supported_research_schema': True},
        'unapproved_example_budget_256_not_adopted': True,
        'no_DoRA_sampling_noise_taxonomy_loss_stack_as_one_factor': True,
    }
    write_json(output/'protocol-proposal.BLOCKED.json', proposal)
    write_json(output/'aggregate-appendix.json', report)
    for relative, digest in integrity.items():
        if hashlib.sha256(audit.safe_path(relative).read_bytes()).hexdigest() != digest:
            raise ValueError('input changed during preparation')
    write_json(output/'BUILD-COMPLETE.json', {'complete': True, 'formal_training_ready': False, 'review_cases_pending': len(selected), 'input_hashes_unchanged': True})
    # Aggregate-only terminal output; never print reviewer rows/mapping/private errors.
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'ok': False, 'error_type': type(error).__name__, 'details_withheld_for_privacy': True}))
        raise SystemExit(1)
