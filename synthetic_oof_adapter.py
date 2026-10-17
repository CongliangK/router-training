"""Synthetic-only snapshot -> OOF metadata draft, Refs #43 and #21.

No paths, discovery, training, model I/O, approval or qualification API. A caller's
synthetic/audit declaration cannot establish authentic provenance. This bounded
candidate accepts only its documented synthetic input profile, not real assets.
"""
from __future__ import annotations
import hashlib
import json
import math
import re

import calibration_groups as cg
import oof_protocol as oof
import research_training as research
import task_label_schema as schema

VERSION = 'synthetic-oof-draft-v1'
AUDIT_VERSION = 'synthetic-root-declaration-v1'
INPUT_NAMES = research.INPUT_FILES
ROW_KEYS = {'id', 'prompt', 'group', 'split_group', 'label', 'task_label_schema_version'}
ROW_OPTIONAL = {'label_schema_version', 'label_source', 'evidence', 'label_status',
                'synthetic', 'supplemental_only'}
LABEL_KEYS = {'tier', 'needs_vision', 'task_type', 'label_source', 'label_status',
              'evidence', 'confidence'}
LABEL_OPTIONAL = {'id', 'rationale', 'task_label_schema_version', 'label_schema_version',
                  'synthetic', 'supplemental_only'}
PROTOCOL_KEYS = {'version', 'status', 'synthetic_fixture', 'protocol_approved',
                 'config_sha256', 'root_audit', 'n_folds', 'min_class_roots',
                 'nested_role_roots', 'epochs', 'models'}
RECORD_KEYS = {'id', 'source_split', 'root_id', 'fold', 'epoch', 'plan_digest',
               'draft_sha256', 'model_digest', 'training_roots', 'logits',
               'task_label_schema_version'}
ROLE_PROTOCOL_KEYS = {'tau_grid', 'temperature_bounds', 'fit_max_iterations',
                      'fit_tolerance', 'probability_epsilon'}
ROLE_SUPPORT_KEYS = {'min_groups_per_role', 'min_rows_per_class',
                     'min_groups_per_class', 'min_fold_train_groups_per_class', 'n_folds'}


class AdapterError(ValueError):
    """Values/identifiers are withheld from errors."""


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or \
            set(value) - set(required) - set(optional):
        raise AdapterError('missing or unknown fields in synthetic draft input')


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _role_overrides(roles):
    """The draft's strict nested schema is local; inherited APIs stay unchanged.

    Shared preflight merges override dictionaries, including unrecognized keys.
    This profile permits only its existing numeric fields, never approval or
    callable extensions. Shared preflight still validates all numeric ranges.
    """
    _keys(roles, {'role_groups'}, {'protocol', 'support_policy'})
    if 'protocol' in roles:
        override = roles['protocol']
        _keys(override, set(), ROLE_PROTOCOL_KEYS)
        for key, value in override.items():
            if key in ('tau_grid', 'temperature_bounds'):
                if not isinstance(value, list) or not value or \
                        (key == 'temperature_bounds' and len(value) != 2) or \
                        any(not _finite(number) for number in value):
                    raise AdapterError('roles protocol needs finite numeric lists of supported shape')
            elif key == 'fit_max_iterations':
                if type(value) is not int:
                    raise AdapterError('roles fit iteration count must be an integer')
            elif not _finite(value):
                raise AdapterError('roles protocol needs finite numeric values')
    if 'support_policy' in roles:
        override = roles['support_policy']
        _keys(override, set(), ROLE_SUPPORT_KEYS)
        for key, value in override.items():
            if key == 'min_groups_per_role':
                _keys(value, {'A', 'B', 'C'})
                if any(type(count) is not int for count in value.values()):
                    raise AdapterError('role support counts must be integers')
            elif type(value) is not int:
                raise AdapterError('support policy counts must be integers')


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          allow_nan=False, ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        raise AdapterError('finite UTF-8 JSON metadata required') from None


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value):
    return json.loads(_canonical(value))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AdapterError('duplicate JSON object keys are forbidden')
        result[key] = value
    return result


def _parse(payload):
    try:
        value = json.loads(payload, object_pairs_hook=_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               AdapterError('nonfinite JSON is forbidden')))
        _canonical(value)
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise AdapterError('strict finite UTF-8 JSON snapshot required') from None


def _snapshot(payloads, config):
    _keys(payloads, INPUT_NAMES)
    # Retain one local mapping of immutable bytes before hashing or parsing.
    # Later caller-side mapping edits must not swap the parsed snapshot.
    payloads = dict(payloads)
    if any(type(value) is not bytes for value in payloads.values()):
        raise AdapterError('five explicit in-memory byte snapshots required; paths are unsupported')
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    try:
        research.validate_config(config)
        research.check_task_schema(schema.V2, config)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise AdapterError('matching research-v2 configuration required') from None
    if hashes != config['input_sha256']:
        raise AdapterError('snapshot/config fingerprints changed; no bindings are repaired')
    parsed = {}
    for name, payload in payloads.items():
        try:
            text = payload.decode('utf-8')
        except UnicodeError:
            raise AdapterError('UTF-8 snapshots required') from None
        parsed[name] = ([_parse(line) for line in text.splitlines() if line.strip()]
                        if name.endswith('.jsonl') else _parse(text))
    train, calibration = parsed['train.jsonl'], parsed['calibration.jsonl']
    all_rows = train + calibration
    ids = set()
    for row in all_rows:
        _keys(row, ROW_KEYS, ROW_OPTIONAL)
        _keys(row['label'], LABEL_KEYS, LABEL_OPTIONAL)
        if not _text(row['id']) or row['id'] in ids:
            raise AdapterError('unique nonempty row identities required across both sources')
        ids.add(row['id'])
    report = parsed['split-report.json']
    # This draft deliberately accepts an explicit development-only declaration,
    # not full preparation reports containing held-out rows or unknown fields.
    _keys(report, {'label_schema', 'task_label_schema_version', 'label_schema_version', 'splits'})
    _keys(report['splits'], {'train', 'calibration'})
    for name, rows in (('train', train), ('calibration', calibration)):
        _keys(report['splits'][name], {'rows'})
        count = report['splits'][name]['rows']
        if type(count) is not int or count != len(rows):
            raise AdapterError('development split declaration differs from snapshot')
    try:
        schema.validate_dataset_documents({name: parsed[name] for name in
            ('label-schema.json', 'split-report.json')}, schema.V2)
        research.validate_v2_experts(train)
        research.validate_v2_experts(calibration)
        cg.validate_partition_isolation(train, calibration)
        _role_overrides(parsed['roles.json'])
        original_roles = cg.preflight_roles(calibration, parsed['roles.json'],
            schema=cg.schema_from_questions(schema.questions_for(schema.V2), schema.V2))
    except (ValueError, TypeError, KeyError, OverflowError):
        raise AdapterError('snapshot schema, expert provenance, isolation or original roles invalid') from None
    return parsed, hashes, original_roles['plan_hash']


def _audited_rows(parsed, hashes, audit):
    _keys(audit, {'version', 'status', 'evidence_ref', 'input_sha256', 'entries'})
    if audit['version'] != AUDIT_VERSION or audit['status'] != 'synthetic-unverified' or \
            not _text(audit['evidence_ref']) or audit['input_sha256'] != hashes:
        raise AdapterError('snapshot-bound synthetic root audit declaration required')
    if not isinstance(audit['entries'], list) or not audit['entries']:
        raise AdapterError('explicit root audit entries required; groups are never inferred roots')
    entries = {}
    for entry in audit['entries']:
        _keys(entry, {'id', 'source_split', 'group', 'split_group', 'root_id'})
        if any(not _text(value) for value in entry.values()) or entry['id'] in entries:
            raise AdapterError('root audit entries require unique complete identities')
        entries[entry['id']] = entry
    rows, root_sources, component_roots = [], {}, {}
    for source in ('train', 'calibration'):
        for row in parsed[source + '.jsonl']:
            entry = entries.get(row['id'])
            if entry is None or any(entry[k] != row[k] for k in ('id', 'group', 'split_group')) \
                    or entry['source_split'] != source:
                raise AdapterError('root audit coverage or source/component binding differs')
            root = entry['root_id']
            if root_sources.setdefault(root, source) != source:
                raise AdapterError('audited root crosses train/calibration sources')
            if component_roots.setdefault(row['split_group'], root) != root:
                raise AdapterError('one duplicate component cannot be split across audited roots')
            rows.append({**{key: value for key, value in row.items() if key != 'prompt'},
                         'source_split': source, 'root_id': root})
    if set(entries) != {row['id'] for row in rows}:
        raise AdapterError('root audit contains foreign identities')
    # The additional root audit can reveal correlations not expressed by old
    # components. Preserve original A/B/C isolation too; do not let a fresh
    # nested proposal launder an invalid original-role snapshot.
    group_roles = {group: role for role, groups in parsed['roles.json']['role_groups'].items()
                   for group in groups}
    root_roles, root_components = {}, {}
    for row in rows:
        if row['source_split'] == 'calibration':
            role = group_roles[row['split_group']]
            if root_roles.setdefault(row['root_id'], role) != role:
                raise AdapterError('audited root overlaps original calibration roles')
            # F1: even within B, joining old components can leak across its
            # old folds or erase support. Require a one-root/one-component
            # calibration mapping instead of silently rebuilding that plan.
            if root_components.setdefault(row['root_id'], row['split_group']) != row['split_group']:
                raise AdapterError('audited root spans original calibration components; '
                                   'original component/role snapshot requires review')
    return rows


def build_synthetic_draft(payloads, config, protocol):
    """Validate a supplied invented fixture and return an unapproved draft.

    The original calibration roles remain bound and validated. Nested OOF roles
    are a separate explicit proposal, NEVER derived from those roles. Digests
    detect changes, not audit authenticity or human approval. No files are read.
    """
    config, protocol = _copy(config), _copy(protocol)
    _keys(protocol, PROTOCOL_KEYS)
    if protocol['version'] != VERSION or protocol['status'] != 'draft' or \
            protocol['synthetic_fixture'] is not True or protocol['protocol_approved'] is not False:
        raise AdapterError('only explicitly synthetic unapproved drafts are supported; formal use is NO-GO')
    if not _hash(protocol['config_sha256']) or protocol['config_sha256'] != research.config_hash(config):
        raise AdapterError('draft config binding changed')
    parsed, hashes, original_role_hash = _snapshot(payloads, config)
    rows = _audited_rows(parsed, hashes, protocol['root_audit'])
    epochs = protocol['epochs']
    if not isinstance(epochs, list) or not epochs or any(type(e) is not int or e < 1 for e in epochs) \
            or len(epochs) != config['training']['epochs'] or \
            epochs != list(range(1, len(epochs) + 1)):
        raise AdapterError('complete ordered epoch schedule must match bound config')
    try:
        plan = oof.nested_plan(rows, protocol['nested_role_roots'],
            n_folds=protocol['n_folds'], min_class_roots=protocol['min_class_roots'],
            root_audit_ref=_digest(protocol['root_audit']))
    except (ValueError, TypeError, KeyError, OverflowError):
        raise AdapterError('explicit nested roles or independent-root support infeasible') from None
    if not isinstance(protocol['models'], list):
        raise AdapterError('explicit synthetic fold/epoch model registry required')
    expected = {(fold['fold'], epoch) for fold in plan['folds'] for epoch in epochs}
    seen = set()
    for model in protocol['models']:
        _keys(model, {'fold', 'epoch', 'model_digest', 'source'})
        if type(model['fold']) is not int or type(model['epoch']) is not int:
            raise AdapterError('model fold/epoch must be integers')
        key = model['fold'], model['epoch']
        if key not in expected or key in seen or not _hash(model['model_digest']) or \
                model['source'] != 'synthetic-placeholder':
            raise AdapterError('synthetic model registry is incomplete, duplicate or incompatible')
        seen.add(key)
    if seen != expected:
        raise AdapterError('synthetic model registry must cover every fold/epoch exactly')
    result = {'version': VERSION, 'status': 'draft', 'synthetic_fixture': True,
        'qualified': False, 'approval_verified': False, 'real_root_audit_verified': False,
        'formal_admission': 'NO-GO', 'training_allowed': False,
        'label_schema': _copy(config['label_schema']), 'input_sha256': hashes,
        'config_sha256': protocol['config_sha256'], 'protocol_sha256': _digest(protocol),
        'root_audit_sha256': _digest(protocol['root_audit']),
        'original_roles_plan_hash': original_role_hash, 'oof_plan': plan,
        'epochs': epochs, 'models': sorted(protocol['models'], key=lambda m: (m['fold'], m['epoch'])),
        'blocked_on': ['human_protocol_approval', 'real_independent_root_audit',
                       'authorized_real_model_execution', 'fresh_calibration_and_confirmation']}
    result['draft_sha256'] = _digest(result)
    return result


def validate_synthetic_records(payloads, config, protocol, draft, records):
    """Check invented OOF lineage, all v2 head widths and prebound model IDs.

    Rebuild from all original inputs; neither rehashing an edited draft nor a
    matching digest string proves evidence/approval. No metrics are estimated.
    """
    expected = build_synthetic_draft(payloads, config, protocol)
    if _canonical(draft) != _canonical(expected):
        raise AdapterError('draft differs from its fully revalidated bound inputs')
    records = _copy(records)
    if not isinstance(records, list):
        raise AdapterError('explicit synthetic OOF record list required')
    models = {(m['fold'], m['epoch']): m['model_digest'] for m in expected['models']}
    projected = []
    for record in records:
        _keys(record, RECORD_KEYS)
        if record['draft_sha256'] != expected['draft_sha256'] or \
                record['task_label_schema_version'] != schema.V2:
            raise AdapterError('record draft or task-schema identity differs')
        if type(record['fold']) is not int or type(record['epoch']) is not int or \
                models.get((record['fold'], record['epoch'])) != record['model_digest']:
            raise AdapterError('record model/fold/epoch differs from prebound registry')
        logits = record['logits']
        _keys(logits, expected['label_schema']['classes'])
        for head, classes in expected['label_schema']['classes'].items():
            values = logits[head]
            if not isinstance(values, list) or len(values) != len(classes) or any(
                    not _finite(v) for v in values):
                raise AdapterError('finite exact-width v2 logits required')
        projected.append({**{k: v for k, v in record.items()
                             if k not in ('draft_sha256', 'task_label_schema_version')},
                          'logits': {k: logits[k] for k in ('tier', 'vision')}})
    try:
        checked = oof.validate_oof_records(expected['oof_plan'], projected, expected['epochs'])
    except (ValueError, TypeError, KeyError, OverflowError):
        raise AdapterError('OOF identity, source, role, model or coverage lineage invalid') from None
    return {'version': VERSION, 'status': 'synthetic-software-check-only',
            'qualified': False, 'approval_verified': False, 'real_root_audit_verified': False,
            'formal_admission': 'NO-GO', 'training_allowed': False,
            'draft_sha256': expected['draft_sha256'], 'records_sha256': _digest(records),
            'row_epoch_count': checked['row_epoch_count'], 'independent_calibration_units': None,
            'scientific_metrics': None}
