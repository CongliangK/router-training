"""Explicit synthetic matched-ablation drafts, Refs #44/#43/#42/#21.

Pure in-memory metadata and invented-fit sampling; no executor, CLI, config export,
model I/O, scientific qualification or approval. Supplied provenance is a claim.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import re
import sys

import calibration_groups as cg
import research_training as research
import synthetic_oof_adapter as adapter

VERSION = 'synthetic-matched-ablation-v1'
FACTORS = ('loss.auxiliary', 'loss.confidence_weighting', 'sampler.prior_source')
INITIALIZATION_VERSION = 'synthetic-common-initialization-v1'
RECIPE_VERSION = 'synthetic-selection-calibration-v1'


class ManifestError(ValueError):
    """Safe failure without supplied identifiers, rows or values."""


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ManifestError('exact supported manifest fields required')


def _json(value):
    # Reject key coercion, tuples, custom objects and bool-as-number surprises
    # before JSON normalization. Caller inputs cannot carry callables/extensions.
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ManifestError('JSON object keys must be strings')
        for item in value.values():
            _json(item)
    elif type(value) is list:
        for item in value:
            _json(item)
    elif type(value) in (int, float):
        try:
            if not math.isfinite(value):
                raise ManifestError('finite JSON required')
        except OverflowError:
            raise ManifestError('finite JSON required') from None
    elif value is not None and type(value) not in (str, bool):
        raise ManifestError('plain finite JSON metadata required')


def _canonical(value):
    try:
        _json(value)
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        raise ManifestError('plain finite UTF-8 JSON metadata required') from None


def _copy(value):
    return json.loads(_canonical(value))


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _equal(left, right):
    return _canonical(left) == _canonical(right)


def _text(value):
    return type(value) is str and bool(value.strip())


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _settings(comparison, roles):
    """Bind descriptions, not executable methods or invented approval defaults."""
    _keys(comparison, {'version', 'status', 'synthetic_fixture', 'protocol_approved',
        'baseline_arm_id', 'contrasts', 'initialization', 'calibration_recipe'})
    if comparison['version'] != VERSION or comparison['status'] != 'draft' or \
            comparison['synthetic_fixture'] is not True or comparison['protocol_approved'] is not False:
        raise ManifestError('only explicit unapproved synthetic comparison drafts supported')
    init = comparison['initialization']
    _keys(init, {'version', 'source', 'base_model_id', 'revision',
                 'initialization_sha256', 'tokenizer_sha256', 'scorer_request_sha256'})
    if init['version'] != INITIALIZATION_VERSION or init['source'] != 'synthetic-placeholder' or \
            any(not _text(init[k]) for k in ('base_model_id', 'revision')) or \
            any(not _hash(init[k]) for k in
                ('initialization_sha256', 'tokenizer_sha256', 'scorer_request_sha256')):
        raise ManifestError('explicit common synthetic initialization/tokenizer/scorer identity required')
    recipe = comparison['calibration_recipe']
    _keys(recipe, {'version', 'status', 'source', 'A', 'B', 'C', 'numeric_protocol',
        'support_policy', 'fixed_model_required', 'fresh_calibration_required',
        'independent_confirmation_status'})
    if recipe['version'] != RECIPE_VERSION or recipe['status'] != 'draft' or \
            recipe['source'] != 'synthetic-placeholder' or \
            recipe['fixed_model_required'] is not True or recipe['fresh_calibration_required'] is not True or \
            recipe['independent_confirmation_status'] != 'UNRUN':
        raise ManifestError('unapproved fixed-model fresh-calibration recipe required')
    for role in ('A', 'C'):
        _keys(recipe[role], {'selection_rule_id', 'tie_break_rule_id'})
        if any(not _text(value) for value in recipe[role].values()):
            raise ManifestError('explicit selection/tie-break identities required')
    _keys(recipe['B'], {'fit_rule_id', 'threshold_rule_id', 'tie_break_rule_id', 'decision_prior'})
    if any(not _text(recipe['B'][key]) for key in ('fit_rule_id', 'threshold_rule_id', 'tie_break_rule_id')):
        raise ManifestError('explicit calibration method identities required')
    prior = recipe['B']['decision_prior']
    _keys(prior, {'source', 'smoothing_pseudocount'})
    if prior['source'] != 'unique-training-before-oversampling' or \
            type(prior['smoothing_pseudocount']) not in (int, float) or prior['smoothing_pseudocount'] < 0:
        raise ManifestError('explicit decision prior distinct from training prior required')
    # #43 has already checked strict overrides, their types/ranges and support.
    # Expand the SAME inherited defaults, not a new validation/approval policy.
    effective = {'numeric_protocol': {**cg.DEFAULT_PROTOCOL, **roles.get('protocol', {})},
                 'support_policy': {**cg.DEFAULT_SUPPORT_POLICY, **roles.get('support_policy', {})}}
    for name, values in effective.items():
        if not _equal(recipe[name], values):
            raise ManifestError('explicit complete recipe numbers must match original effective roles settings')


def _projection(config):
    invariant = _copy(config)
    for path in FACTORS:
        section, key = path.split('.')
        del invariant[section][key]
    return invariant


def _factors(config):
    return {path: config[section][key] for path in FACTORS for section, key in [path.split('.')]}


def _contrasts(arms, comparison):
    by_id = {arm['arm_id']: arm for arm in arms}
    baseline_id = comparison['baseline_arm_id']
    if not _text(baseline_id) or baseline_id not in by_id:
        raise ManifestError('comparison baseline must identify a supplied arm')
    contrasts = comparison['contrasts']
    if type(contrasts) is not list or len(contrasts) != len(arms) - 1:
        raise ManifestError('one explicit baseline contrast required for each other arm')
    seen, baseline = set(), _factors(by_id[baseline_id]['config'])
    for contrast in contrasts:
        _keys(contrast, {'arm_id', 'changed_factors', 'kind'})
        arm_id, declared = contrast['arm_id'], contrast['changed_factors']
        if not _text(arm_id) or arm_id not in by_id or arm_id == baseline_id or arm_id in seen:
            raise ManifestError('unique nonbaseline contrast arms required')
        seen.add(arm_id)
        if type(declared) is not list or not declared or any(type(p) is not str for p in declared) or \
                len(set(declared)) != len(declared) or set(declared) - set(FACTORS):
            raise ManifestError('explicit distinct supported comparison factors required')
        current = _factors(by_id[arm_id]['config'])
        actual = {path for path in FACTORS if not _equal(baseline[path], current[path])}
        if actual != set(declared) or contrast['kind'] != ('single-factor' if len(actual) == 1 else 'combined'):
            raise ManifestError('declared contrast must equal all actual factor changes')


def _oof_binding(draft):
    plan = draft['oof_plan']
    return {'plan_digest': plan['plan_digest'], 'pool_digest': plan['pool_digest'],
        'root_audit_sha256': draft['root_audit_sha256'],
        'original_roles_plan_hash': draft['original_roles_plan_hash'],
        'root_count': plan['root_count'], 'row_count': plan['row_count'],
        'epochs': draft['epochs'],
        'folds': [{'fold': fold['fold'],
            'validation_ids_sha256': _digest(fold['validation_ids']),
            'validation_roots_sha256': _digest(fold['validation_roots']),
            'role_ids_sha256': {r: _digest(ids) for r, ids in fold['role_ids'].items()},
            'role_roots_sha256': {r: _digest(roots) for r, roots in fold['role_roots'].items()},
            'role_row_counts': {r: len(ids) for r, ids in fold['role_ids'].items()},
            'role_root_counts': {r: len(roots) for r, roots in fold['role_roots'].items()},
            'actual_training_root_count': fold['actual_training_root_count'],
            'eligible_outside_root_count': fold['eligible_outside_root_count']}
            for fold in plan['folds']]}


def _budgets(rows, config, draft):
    """Only validated fit rows, original train-then-calibration byte row order.

    Count the flush at EACH epoch boundary. Never pool rows before ceil. Full
    ordered streams remain private in-memory comparison values, not just counts.
    """
    reports, signatures = [], []
    accumulation = config['training']['accumulation']
    weights = research.head_weights(config)
    enabled = config['loss']['confidence_weighting']
    for fold in draft['oof_plan']['folds']:
        fit_ids = set(fold['role_ids']['fit'])
        fit = [row for row in rows if row['id'] in fit_ids]
        streams, sampling = research.sample_epochs(fit, config,
            config['training']['seed'], len(draft['epochs']))
        order = [row['id'] for row in fit]
        sequence = [[row['id'] for row in stream] for stream in streams]
        frequency = [dict(sorted(collections.Counter(ids).items())) for ids in sequence]
        probability = {'row_order': order,
            'natural': sampling['natural_row_probabilities'],
            'target': sampling['target_row_probabilities']}
        epoch_reports = []
        for epoch, stream, ids, counts in zip(draft['epochs'], streams, sequence, frequency):
            n = len(stream)
            confidence = math.fsum(research.confidence_weight(row, enabled) for row in stream)
            epoch_reports.append({'epoch': epoch, 'planned_row_exposure': n,
                'planned_optimizer_updates': (n + accumulation - 1) // accumulation,
                'final_update_rows': (n - 1) % accumulation + 1 if n else 0,
                'ordered_stream_sha256': _digest(ids), 'achieved_row_counts_sha256': _digest(counts),
                'unique_sampled_rows': len(counts), 'confidence_weighted_exposure': confidence,
                'head_weighted_exposure': {head: coefficient * confidence for head, coefficient in weights.items()}})
        report = {'fold': fold['fold'], 'fit_row_count': len(fit),
            'fit_root_count': fold['actual_training_root_count'],
            'fit_order_sha256': _digest(order),
            'probability_sha256': _digest(probability),
            'natural_probabilities_sha256': _digest(probability['natural']),
            'target_probabilities_sha256': _digest(probability['target']),
            'sampler_row_order_hash': sampling['row_order_hash'],
            'ordered_fit_rows_sha256': sampling['row_provenance']['ordered_rows_sha256'],
            'training_priors': sampling['priors'], 'prior_provenance': sampling['prior_provenance'],
            'epochs': epoch_reports,
            'planned_row_exposure': sum(e['planned_row_exposure'] for e in epoch_reports),
            'planned_optimizer_updates': sum(e['planned_optimizer_updates'] for e in epoch_reports)}
        reports.append(report)
        signatures.append({'probability': probability, 'sequence': sequence,
            'budgets': [{k: e[k] for k in ('epoch', 'planned_row_exposure',
                'planned_optimizer_updates', 'final_update_rows')} for e in epoch_reports]})
    return reports, signatures


def build_synthetic_manifest(payloads, arms, comparison):
    """Rebuild #43 for every explicit arm and return a draft, never a config.

    Supported changes are auxiliary coefficient mode, confidence weighting and
    training prior source. All other config/protocol fields must match. Model
    outputs are supplied independently for each arm; no hashes are repaired.
    """
    _keys(payloads, research.INPUT_FILES)
    payloads = dict(payloads)  # One retained immutable byte snapshot for ALL arms.
    if any(type(value) is not bytes for value in payloads.values()):
        raise ManifestError('five explicit byte snapshots required; no paths supported')
    arms, comparison = _copy(arms), _copy(comparison)
    if type(arms) is not list or len(arms) < 2:
        raise ManifestError('comparison requires at least two explicit arms, not a replicate quota')
    seen, drafts, output_digests = set(), [], set()
    for arm in arms:
        _keys(arm, {'arm_id', 'config', 'protocol'})
        if not _text(arm['arm_id']) or arm['arm_id'] in seen:
            raise ManifestError('unique nonempty arm identities required')
        seen.add(arm['arm_id'])
        try:
            draft = adapter.build_synthetic_draft(payloads, arm['config'], arm['protocol'])
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ManifestError('arm failed the unchanged synthetic OOF adapter contract') from None
        # The inherited cost-matrix equality check uses Python numeric equality;
        # False/True must not masquerade as its 0/1 numeric controls locally.
        if any(type(value) not in (int, float) for row in arm['config']['loss']['cost_matrix'] for value in row):
            raise ManifestError('cost matrix controls must be numbers, not booleans')
        digests = {m['model_digest'] for m in draft['models']}
        if output_digests.intersection(digests):
            raise ManifestError('synthetic output model identities must be arm-local, not copied across arms')
        output_digests.update(digests)
        drafts.append(draft)
    # Parsing only already-validated retained bytes; no new schema validator or
    # projection of unsupported #43 input profiles. Its strict JSON check ran first.
    roles = json.loads(payloads['roles.json'])
    _settings(comparison, roles)
    _contrasts(arms, comparison)
    invariant = _projection(arms[0]['config'])
    fixed_protocol = {k: v for k, v in arms[0]['protocol'].items() if k not in ('models', 'config_sha256')}
    for arm, draft in zip(arms[1:], drafts[1:]):
        protocol = {k: v for k, v in arm['protocol'].items() if k not in ('models', 'config_sha256')}
        if not _equal(_projection(arm['config']), invariant) or not _equal(protocol, fixed_protocol) or \
                not _equal(draft['oof_plan'], drafts[0]['oof_plan']) or \
                draft['original_roles_plan_hash'] != drafts[0]['original_roles_plan_hash']:
            raise ManifestError('nonfactor config, root/role/fold/epoch or protocol changed across arms')
    rows = [json.loads(line) for name in ('train.jsonl', 'calibration.jsonl')
            for line in payloads[name].decode('utf-8').splitlines() if line.strip()]
    reports, reference = [], None
    for arm, draft in zip(arms, drafts):
        try:
            budgets, signature = _budgets(rows, arm['config'], draft)
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ManifestError('synthetic fit-only budget calculation failed') from None
        if reference is not None and not _equal(reference, signature):
            raise ManifestError('ordered fit sampling stream, probabilities or per-epoch budget changed')
        reference = signature
        reports.append({'arm_id': arm['arm_id'], 'factors': _factors(arm['config']),
            'head_coefficients': research.head_weights(arm['config']),
            'config_sha256': draft['config_sha256'], 'protocol_sha256': draft['protocol_sha256'],
            'draft_sha256': draft['draft_sha256'],
            'models': [{'arm_id': arm['arm_id'], **model} for model in draft['models']],
            'fold_budgets': budgets,
            'planned_row_exposure': sum(b['planned_row_exposure'] for b in budgets),
            'planned_optimizer_updates': sum(b['planned_optimizer_updates'] for b in budgets),
            'measurement': {'status': 'UNRUN', 'actual_trainable_parameters': None,
                'actual_total_parameters': None,
                'actual_optimizer_updates': None, 'training_tokens': None, 'training_flops': None,
                'wall_seconds': None, 'gpu_peak_bytes': None, 'monetary_cost': None}})
    result = {'version': VERSION, 'status': 'draft', 'synthetic_fixture': True,
        'qualified': False, 'approval_verified': False, 'real_root_audit_verified': False,
        'training_allowed': False, 'formal_admission': 'NO-GO',
        'real_execution_status': 'UNRUN', 'scientific_result': None,
        'input_sha256': drafts[0]['input_sha256'], 'label_schema': drafts[0]['label_schema'],
        'invariant_config_sha256': _digest(invariant), 'fixed_protocol_sha256': _digest(fixed_protocol),
        'capacity': _copy(arms[0]['config']['capacity']),
        'training_rank': arms[0]['config']['training']['rank'],
        'training_controls': {key: arms[0]['config']['training'][key]
            for key in ('seed', 'epochs', 'accumulation')},
        'sampler_controls': {key: value for key, value in arms[0]['config']['sampler'].items()
            if key != 'prior_source'},
        'comparison': comparison, 'comparison_sha256': _digest(comparison),
        'oof_binding': _oof_binding(drafts[0]), 'arms': reports,
        'budget_formula': 'per-fold sum-over-epochs ceil(sampled_rows/accumulation); final tail flushed',
        'sampling_order': 'train byte row order then calibration byte row order, filtered to nested fit IDs',
        'runtime': {'python_implementation': sys.implementation.name,
            'python_version': list(sys.version_info[:3]), 'engine': 'stdlib-random.Random.choices',
            'sampler_version': arms[0]['config']['sampler']['version']},
        'blocked_on': ['human_role_support_statistical_and_budget_protocol',
            'real_independent_root_and_label_audit', 'authorized_real_model_and_matched_training',
            'fresh_fixed_model_calibration_and_independent_confirmation', 'independent_final_source_and_execution_review']}
    result['manifest_sha256'] = _digest(result)
    return result


def validate_synthetic_manifest(payloads, arms, comparison, manifest):
    """Recompute the entire envelope. Editing and self-rehashing cannot approve it."""
    expected = build_synthetic_manifest(payloads, arms, comparison)
    if not _equal(manifest, expected):
        raise ManifestError('manifest differs from its fully rebuilt explicit inputs')
    return {'version': VERSION, 'status': 'synthetic-software-check-only',
        'manifest_sha256': expected['manifest_sha256'], 'qualified': False,
        'approval_verified': False, 'real_root_audit_verified': False,
        'training_allowed': False, 'formal_admission': 'NO-GO',
        'real_execution_status': 'UNRUN', 'scientific_result': None}
