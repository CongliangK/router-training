"""Strict tier-only offline input boundary for issues #1/#22 (stdlib only).

No prompts, probabilities masquerading as logits, unknown metadata, or test
splits. Provenance is an extractor assertion, not independently authenticated.
Errors deliberately withhold paths, row IDs, group names, and supplied values.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
from calibration_groups import PlanError, questions_from_source

TIERS = tuple(questions_from_source()['tier']['criteria'])
RAW_SCHEMA = 'router-raw-tier-logits/v1'
MAX_ROWS = 2000


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def exact(obj, keys, where):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise PlanError(f'{where}: exact schema required; missing/unknown fields rejected (values withheld)')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanError('duplicate JSON object key (values withheld)')
        result[key] = value
    return result


def load_json(path):
    """Read only the explicitly requested JSON; never discover adjacent assets."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(PlanError('nonfinite JSON number')))
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, PlanError):
            raise
        raise PlanError('cannot read valid JSON input (path and contents withheld)') from None


def validate_source(source):
    exact(source, ('schema_version', 'data_kind', 'split', 'head', 'class_order',
                   'logit_stage', 'dataset_sha256', 'scoring_sha256', 'model_sha256', 'rows'), 'raw logits')
    if source['schema_version'] != RAW_SCHEMA or source['head'] != 'tier' or source['class_order'] != list(TIERS):
        raise PlanError('unsupported raw-logit version/head/class order')
    if source['data_kind'] not in ('synthetic', 'private-local'):
        raise PlanError('explicit synthetic or private-local provenance required')
    if source['split'] not in ('development', 'calibration'):
        raise PlanError('only development/calibration allowed; test and train selection prohibited')
    if source['logit_stage'] != 'raw-pre-temperature-pre-prior':
        raise PlanError('raw pre-temperature pre-prior logits required; calibrated inputs prohibited')
    if not all(sha(source[k]) for k in ('dataset_sha256', 'scoring_sha256', 'model_sha256')):
        raise PlanError('dataset/scoring/model SHA-256 bindings required')
    rows = source['rows']
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise PlanError(f'raw-logit pool requires 1..{MAX_ROWS} rows (bounded CPU prototype)')
    seen = set()
    for row in rows:
        exact(row, ('id', 'split_group', 'label', 'logits'), 'raw-logit row')
        if not all(isinstance(row[k], str) and row[k].strip() for k in ('id', 'split_group')):
            raise PlanError('nonempty row and whole-source group identities required (values withheld)')
        # JSON permits escaped lone surrogates; canonical UTF-8 bindings do not.
        # Reject them here instead of leaking an unsanitized encoding traceback.
        try:
            for key in ('id', 'split_group'):
                row[key].encode('utf-8')
        except UnicodeError:
            raise PlanError('row/group identities must be valid UTF-8 (values withheld)') from None
        if row['id'] in seen:
            raise PlanError('duplicate row identity (values withheld)')
        seen.add(row['id'])
        if row['label'] not in TIERS:
            raise PlanError('tier label outside fixed schema (value withheld)')
        values = row['logits']
        if not isinstance(values, list) or len(values) != len(TIERS) or not all(number(v) and abs(v) <= 1e6 for v in values):
            raise PlanError('logits must be three finite nonboolean raw scores with absolute value <= 1e6')
    return sorted(rows, key=lambda r: r['id'])


def align_sources(base, finetuned):
    """Join by exact identity, not row position; all target/group metadata agree."""
    left, right = validate_source(base), validate_source(finetuned)
    for key in ('schema_version', 'data_kind', 'split', 'head', 'class_order', 'logit_stage',
                'dataset_sha256', 'scoring_sha256'):
        if base[key] != finetuned[key]:
            raise PlanError('source metadata mismatch; logits are not aligned')
    if len(left) != len(right) or any(tuple(a[k] for k in ('id', 'split_group', 'label')) !=
                                    tuple(b[k] for k in ('id', 'split_group', 'label'))
                                    for a, b in zip(left, right)):
        raise PlanError('source identity/group/label mismatch; exact aligned pools required (values withheld)')
    return left, right


def source_binding(source):
    rows = validate_source(source)
    canonical = {**source, 'rows': rows}
    return {'input_sha256': digest(canonical), 'model_sha256': source['model_sha256'],
            'dataset_sha256': source['dataset_sha256'], 'scoring_sha256': source['scoring_sha256'],
            'data_kind': source['data_kind'], 'split': source['split']}


def validate_prior(prior, source):
    exact(prior, ('schema_version', 'source', 'model_sha256', 'class_order', 'probabilities'), 'prior')
    if prior['schema_version'] != 'router-tier-prior/v1' or prior['source'] != 'unique-training-before-oversampling' \
            or prior['model_sha256'] != source['model_sha256'] or prior['class_order'] != list(TIERS):
        raise PlanError('prior must bind this model and unique pre-oversampling training provenance')
    probs = prior['probabilities']
    if not isinstance(probs, list) or len(probs) != len(TIERS) or not all(number(p) and 0 < p <= 1 for p in probs) \
            or not math.isclose(sum(probs), 1, rel_tol=0, abs_tol=1e-9):
        raise PlanError('training prior must contain normalized strictly positive finite probabilities')
    return probs


def validate_costs(costs):
    exact(costs, ('schema_version', 'version', 'units', 'relative_cost_by_predicted_tier'), 'relative costs')
    if costs['schema_version'] != 'router-relative-cost/v1' or costs['units'] != 'relative-simulation-not-billing' \
            or not isinstance(costs['version'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', costs['version']):
        raise PlanError('explicit versioned relative simulation cost table required (not billing)')
    table = costs['relative_cost_by_predicted_tier']
    exact(table, TIERS, 'relative tier costs')
    if not all(number(v) and v > 0 for v in table.values()):
        raise PlanError('relative route costs must be finite positive nonboolean numbers')
    return table


def write_report(path, report):
    """Exclusive-create only, with no partial JSON or overwrite on validation errors."""
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    try:
        with Path(path).open('x', encoding='utf-8') as handle:
            handle.write(text)
    except OSError:
        raise PlanError('cannot exclusively create output; existing reports are never overwritten') from None
