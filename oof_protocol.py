"""CPU metadata/step-loss prototypes, Refs #21. No training or model I/O.

All plans/results remain unqualified. Declarations and digest strings do not
prove root independence, model provenance, approval, or train/serve parity.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import re
from fractions import Fraction
from calibration_groups import group_folds


class ProtocolError(ValueError):
    """Fail closed; errors deliberately omit input identifiers/labels."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _positive(value):
    return type(value) is int and value > 0


def _pool(rows):
    if not isinstance(rows, list) or not rows:
        raise ProtocolError('nonempty development metadata required')
    ids, split_roots, root_sources = set(), {}, {}
    for row in rows:
        if not isinstance(row, dict) or any(not _text(row.get(k)) for k in
                ('id', 'root_id', 'split_group')):
            raise ProtocolError('row identity and audited root required')
        # An explicit source is mandatory: no implicit development default and
        # no test argument/file discovery API. This does not prove honest input.
        if row.get('source_split') not in ('train', 'calibration'):
            raise ProtocolError('dev-no-test requires explicit train/calibration source')
        if row['id'] in ids:
            raise ProtocolError('duplicate development row')
        ids.add(row['id'])
        if split_roots.setdefault(row['split_group'], row['root_id']) != row['root_id']:
            raise ProtocolError('split group spans audited roots')
        root_sources.setdefault(row['root_id'], set()).add(row['source_split'])
        label = row.get('label', {})
        if not isinstance(label, dict) or label.get('tier') not in ('fast', 'balanced', 'flagship') \
                or label.get('needs_vision') not in ('yes', 'no'):
            raise ProtocolError('required tier/vision support metadata invalid')
    return root_sources


def _support(rows):
    return {f'{head}:{value}': len({r['root_id'] for r in rows if r['label'][head] == value})
            for head, values in (('tier', ('fast', 'balanced', 'flagship')),
                                 ('needs_vision', ('yes', 'no'))) for value in values}


def development_folds(rows, *, n_folds=5, min_class_roots=1, root_audit_ref=None):
    """Whole ROOT folds using existing row-count greedy group_folds, not stratification.

    Support gates are user-declared feasibility requirements, NOT statistical
    qualification. Impossible greedy allocations fail without splitting roots,
    duplicating positives, lowering support, or searching seeds.
    """
    sources = _pool(rows)
    if not _positive(n_folds) or n_folds < 2 or not _positive(min_class_roots):
        raise ProtocolError('fold/support counts must be positive integers, folds >=2')
    if not _text(root_audit_ref):
        raise ProtocolError('root/near-duplicate audit reference required; IDs alone are insufficient')
    if len(sources) < n_folds or any(n < n_folds * min_class_roots for n in _support(rows).values()):
        raise ProtocolError('infeasible independent-root class support before fold construction')
    roots = group_folds(dict(collections.Counter(r['root_id'] for r in rows)), n_folds)
    folds = []
    for i, held in enumerate(roots):
        validation = [r for r in rows if r['root_id'] in held]
        training = [r for r in rows if r['root_id'] not in held]
        if any(n < min_class_roots for part in (validation, training) for n in _support(part).values()):
            raise ProtocolError('infeasible greedy fold class support; approved alternative needed')
        folds.append({'fold': i, 'validation_roots': sorted(held),
                      'outside_roots': sorted(set(sources) - set(held)),
                      'validation_ids': sorted(r['id'] for r in validation),
                      'outside_ids': sorted(r['id'] for r in training),
                      'validation_support': _support(validation), 'outside_support': _support(training)})
    ordered = sorted(rows, key=lambda r: r['id'])
    return {'kind': 'development-no-test', 'qualified': False, 'status': 'exploratory',
            'root_audit_ref': root_audit_ref, 'pool_digest': digest(ordered),
            'folds': folds, 'rows': ordered, 'min_class_roots': min_class_roots,
            'root_count': len(sources), 'row_count': len(rows)}


def nested_plan(rows, role_roots, *, n_folds=5, min_class_roots=1, root_audit_ref=None):
    """Outer OOF with independent fit/A/B/C groups inside EACH outer complement.

    A selects epochs, B fits T/threshold on the selected fixed model, C chooses
    the final configuration. C is selection, NOT independent final evaluation.
    Models only train on fit, not the entire outer complement. Refit invalidates
    old B calibration. No callable fitting/training occurs here.
    """
    plan = development_folds(rows, n_folds=n_folds, min_class_roots=min_class_roots,
                             root_audit_ref=root_audit_ref)
    if not isinstance(role_roots, list) or len(role_roots) != n_folds:
        raise ProtocolError('explicit roles needed for every outer fold')
    for fold, roles in zip(plan['folds'], role_roots):
        if not isinstance(roles, dict) or set(roles) != {'fit', 'A', 'B', 'C'}:
            raise ProtocolError('exactly independent fit/A/B/C roles required')
        union = set()
        for role, roots in roles.items():
            if not isinstance(roots, list) or not roots or not all(_text(r) for r in roots) \
                    or len(set(roots)) != len(roots) or union.intersection(roots):
                raise ProtocolError('nested roles must be nonempty unique disjoint roots')
            union.update(roots)
            region = [r for r in rows if r['root_id'] in roots]
            if any(n < min_class_roots for n in _support(region).values()):
                raise ProtocolError('infeasible nested role support')
        if union != set(fold['outside_roots']):
            raise ProtocolError('roles must partition only the outer complement')
        fold['role_roots'] = {k: sorted(v) for k, v in roles.items()}
        fold['role_ids'] = {k: sorted(r['id'] for r in rows if r['root_id'] in v)
                            for k, v in roles.items()}
        fold['actual_training_root_count'] = len(roles['fit'])
        fold['eligible_outside_root_count'] = len(fold['outside_roots'])
    plan['plan_digest'] = digest(plan)
    return plan


def validate_oof_records(plan, records, epochs):
    """Check every per-row/per-epoch external prediction's declared lineage.

    Each fold/epoch has one frozen weight digest and exact fit-root lineage.
    This is metadata evidence, not independent verification of training bytes.
    Sensitive returned records/plans belong in private execution evidence only.
    """
    if not isinstance(plan, dict) or plan.get('plan_digest') != digest(
            {k: v for k, v in plan.items() if k != 'plan_digest'}):
        raise ProtocolError('nested plan binding missing or changed')
    if not isinstance(epochs, list) or not epochs or any(not _positive(e) for e in epochs) \
            or len(set(epochs)) != len(epochs) or not isinstance(records, list):
        raise ProtocolError('unique positive epoch schedule and records required')
    by_id = {r['id']: r for r in plan['rows']}
    folds = {rid: f for f in plan['folds'] for rid in f['validation_ids']}
    seen, weights = set(), {}
    for record in records:
        if not isinstance(record, dict):
            raise ProtocolError('prediction record must be metadata object')
        rid, epoch = record.get('id'), record.get('epoch')
        if not _text(rid) or rid not in by_id or type(epoch) is not int or epoch not in epochs:
            raise ProtocolError('unknown OOF row/epoch')
        if (rid, epoch) in seen:
            raise ProtocolError('duplicate OOF row/epoch')
        seen.add((rid, epoch))
        fold = folds[rid]
        if type(record.get('fold')) is not int or record['fold'] != fold['fold'] \
                or record.get('source_split') != by_id[rid]['source_split'] \
                or record.get('root_id') != by_id[rid]['root_id'] \
                or record.get('plan_digest') != plan['plan_digest']:
            raise ProtocolError('prediction validation lineage mismatch')
        if record.get('training_roots') != fold['role_roots']['fit']:
            raise ProtocolError('prediction training roots must exactly match isolated fit role')
        if not _hash(record.get('model_digest')):
            raise ProtocolError('frozen model digest required')
        key = fold['fold'], epoch
        if weights.setdefault(key, record['model_digest']) != record['model_digest']:
            raise ProtocolError('inconsistent frozen model within fold/epoch')
        logits = record.get('logits')
        if not isinstance(logits, dict) or set(logits) != {'tier', 'vision'}:
            raise ProtocolError('both required-head logits required')
        for head, size in (('tier', 3), ('vision', 2)):
            if not isinstance(logits[head], list) or len(logits[head]) != size \
                    or not all(_number(v) for v in logits[head]):
                raise ProtocolError('invalid OOF logits')
    if len(seen) != len(by_id) * len(epochs):
        raise ProtocolError('incomplete per-row/per-epoch OOF coverage')
    return {'status': 'exploratory', 'qualified': False, 'row_epoch_count': len(seen),
            'plan_digest': plan['plan_digest'], 'independent_calibration_units': None,
            'warning': 'OOF models share training roots; OOF rows are not fresh CRC units'}


def crc_threshold(units, *, alpha, estimand, context):
    """Restricted CRC for L(lambda)=1{true flagship AND lambda < 1-score}.

    One preselected task per audited exchangeable root, score in [0,1]. Larger
    lambda promotes more tasks. Loss is bounded by 1, decreasing and RIGHT
    continuous (equality promotes); lambda=1 has deterministic zero loss.
    Uses (sum L + 1)/(n+1) <= alpha, NOT a high-probability risk bound.
    For conditional miss risk, sample ONLY true-flagship units from that
    population. Joint miss risk over all units is a different estimand.
    """
    if not _number(alpha) or not 0 < alpha < 1:
        raise ProtocolError('alpha must be finite in (0,1)')
    if estimand not in ('joint_flagship_miss', 'conditional_flagship_miss'):
        raise ProtocolError('explicit joint/conditional estimand required')
    if not isinstance(context, dict) or not isinstance(units, list):
        raise ProtocolError('units and assumption context required')
    seen, breaks = set(), []
    for unit in units:
        if not isinstance(unit, dict) or not _text(unit.get('root_id')) \
                or unit['root_id'] in seen or type(unit.get('true_flagship')) is not bool \
                or not _number(unit.get('score')) or not 0 <= unit['score'] <= 1:
            raise ProtocolError('one valid bounded-score task per unique root required')
        seen.add(unit['root_id'])
        if estimand == 'conditional_flagship_miss' and not unit['true_flagship']:
            raise ProtocolError('conditional risk needs true-flagship-only sampling')
        if unit['true_flagship']:
            breaks.append(1.0 - unit['score'])
    reasons = []
    for field in ('fixed_before_calibration', 'exchangeable_roots', 'one_task_per_root_preselected',
                  'loss_preregistered', 'no_calibration_label_selection'):
        if context.get(field) is not True:
            reasons.append(field)
    if context.get('prediction_source') != 'independent_fixed_model':
        reasons.append('not_independent_fixed_model_predictions')
    if not _hash(context.get('model_digest')):
        reasons.append('model_digest')
    if not _text(context.get('approval_ref')):
        reasons.append('protocol_not_approved')
    expected_population = 'true_flagship_units' if estimand.startswith('conditional') else 'all_units'
    if context.get('sampling_population') != expected_population:
        reasons.append('sampling_population_mismatch')
    result = {'qualified': False, 'status': 'exploratory', 'coverage_claim': False,
              'assumptions_verified': False, 'approval_verified': False, 'n_root_units': len(units),
              'estimand': estimand, 'alpha': alpha, 'threshold': None, 'safe_extreme': 1.0,
              'risk_statement': None, 'missing_assumptions': reasons,
              'scope': 'equal-root one-preselected-task loss; not arbitrary row-weighted risk'}
    if not units:
        result['status'] = 'infeasible'
        return result
    # Fraction avoids 5% boundary drift at n=19. Breakpoints are computed once
    # and reused, so exact equality implements the stated right continuity.
    for threshold in sorted(set([0.0, 1.0] + breaks)):
        losses = sum(threshold < b for b in breaks)
        corrected = Fraction(losses + 1, len(units) + 1)
        if corrected <= Fraction(str(alpha)):
            result.update(threshold=threshold, empirical_misses=losses,
                          corrected_empirical_risk=float(corrected))
            if not reasons:
                result['status'] = 'assumption_conditional'
                result['risk_statement'] = ('If declared theorem assumptions actually hold, '
                                            'E[L_new(threshold)] <= alpha; expectation over calibration/new unit only')
            return result
    result['status'] = 'infeasible'
    return result


def soup_compatibility(members):
    """Validate dense parameter metadata only. NEVER load or average weights.

    Raw LoRA/DoRA factor averaging is not equivalent to dense weight averaging.
    Quantized tensors also require a separately reviewed materialization path.
    """
    if not isinstance(members, list) or len(members) < 2:
        raise ProtocolError('at least two dense model descriptors required')
    reference, seen = None, set()
    for model in members:
        if not isinstance(model, dict):
            raise ProtocolError('model descriptor required')
        if model.get('representation') not in ('dense_full', 'adapter_merged_dense'):
            raise ProtocolError('raw adapter factors/quantized weights cannot be averaged semantically')
        if model.get('representation') == 'adapter_merged_dense' and (
                model.get('merge_semantics') != 'materialized_delta_into_identical_base' or
                not _hash(model.get('merge_evidence_digest'))):
            raise ProtocolError('adapter requires independently verified materialized dense merge')
        if not _hash(model.get('model_digest')) or model['model_digest'] in seen:
            raise ProtocolError('distinct frozen model identities required')
        seen.add(model['model_digest'])
        fields = ('base_id', 'base_revision', 'architecture', 'tokenizer_digest', 'scoring_contract_digest')
        if any(not _text(model.get(k)) for k in fields) \
                or re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', model['base_revision']) is None \
                or any(not _hash(model.get(k)) for k in ('tokenizer_digest', 'scoring_contract_digest')):
            raise ProtocolError('pinned base commit and tokenizer/scoring digests required')
        params = model.get('parameters')
        if not isinstance(params, dict) or not params or any(not _text(k) for k in params):
            raise ProtocolError('nonempty parameter key/shape/dtype metadata required')
        for value in params.values():
            if not isinstance(value, dict) or set(value) != {'shape', 'dtype'} \
                    or not isinstance(value['shape'], list) \
                    or any(not _positive(d) for d in value['shape']) \
                    or value['dtype'] not in ('float32', 'float64', 'float16', 'bfloat16'):
                raise ProtocolError('invalid dense tensor shape/dtype')
        signature = {k: model[k] for k in fields} | {'parameters': params}
        if reference is not None and reference != signature:
            raise ProtocolError('incompatible base/keys/shapes/dtypes/tokenizer/scoring semantics')
        reference = signature
    return {'status': 'metadata_compatible_only', 'qualified': False, 'member_count': len(members),
            'weights_loaded': False, 'weights_averaged': False,
            'requires': 'new fixed-soup calibration and independent validation; no OOF threshold transfer'}


def deployment_bridge(*, model_digest, training_roots, selection_roots, calibration, confirmation_roots,
                      parity_ref=None, approval_ref=None):
    """Check necessary independent evidence for a final fold/refit/soup model.

    No guarantee is inherited from OOF. Even a structurally complete bridge
    remains shadow until external approval, execution and gates are satisfied.
    """
    if not _hash(model_digest) or not isinstance(calibration, dict):
        raise ProtocolError('fixed deployment and calibration identities required')
    if calibration.get('source') != 'independent_fixed_model' \
            or calibration.get('model_digest') != model_digest:
        raise ProtocolError('OOF thresholds or different-model calibration cannot certify deployment/soup')
    regions = (training_roots, selection_roots, calibration.get('roots'), confirmation_roots)
    seen = set()
    for roots in regions:
        if not isinstance(roots, list) or not roots or not all(_text(r) for r in roots) \
                or len(set(roots)) != len(roots) or seen.intersection(roots):
            raise ProtocolError('deployment train/selection/calibration/confirmation roots must be disjoint')
        seen.update(roots)
    missing = [k for k, value in (('parity_ref', parity_ref), ('approval_ref', approval_ref),
                                 ('calibration_evidence_ref', calibration.get('evidence_ref')))
               if not _text(value)]
    return {'status': 'shadow', 'qualified': False, 'missing_external_evidence': missing,
            'structurally_complete': not missing, 'inherits_oof_guarantee': False}
