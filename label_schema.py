"""Traceable cold-start labels (issue #8). Standard library only; never a trainer.

Measured confidence is a bound's confidence level, NOT an expert sample weight.
A successful execution cannot supply task-understanding annotations. The expert
record remains authoritative for all three cold-start heads, even on shadow rows.
"""
from __future__ import annotations
import copy, json, math
from outcomes import import_outcomes

TIERS = ['fast', 'balanced', 'flagship']
TYPES = ['extraction', 'research', 'implementation', 'debugging', 'review', 'visual', 'creative']
EXPERT = 'expert_provisional'
MEASURED = 'measured_candidate_outcomes'


def _require(condition, message):
    if not condition:raise ValueError(message)


def _index(rows, kind):
    result={}
    for row in rows:
        _require(isinstance(row,dict), f'{kind}: expected object')
        identity=row.get('id')
        _require(isinstance(identity,str) and bool(identity.strip()), f'{kind}: missing id')
        _require(identity not in result, f'{kind}: duplicate id')
        result[identity]=row
    return result


def validate_experts(tasks, labels):
    """Require explicit understanding coverage, never fill it from outcome tiers."""
    tasks_by_id=_index(tasks,'tasks'); experts=_index(labels,'expert labels')
    _require(bool(tasks_by_id), 'empty task corpus')
    _require(tasks_by_id.keys()==experts.keys(), 'expert task-understanding coverage differs from task ids')
    for task in tasks:
        _require(all(isinstance(task.get(k),str) and task[k].strip() for k in ('prompt','group')), 'task requires prompt and group')
    for label in labels:
        _require(label.get('label_source')==EXPERT, 'missing or conflicting expert label_source')
        _require(label.get('evidence')=='prompt_only', 'conflicting expert evidence source')
        _require(label.get('tier') in TIERS and label.get('task_type') in TYPES and label.get('needs_vision') in ('yes','no','uncertain'), 'missing or invalid task-understanding fields')
        confidence=label.get('confidence')
        _require(type(confidence) in (int,float) and math.isfinite(confidence) and 0<=confidence<=1, 'invalid expert confidence')
        _require(isinstance(label.get('rationale'),str) and bool(label['rationale'].strip()), 'missing expert rationale')
        _require(not any(k in label for k in ('model','provider','historical_model','route_fingerprint','success_lb','success_rate')), 'conflicting expert outcome source')
    return experts


def validate_measured(tasks, envelope, routes):
    """Replay the v2 gate from the audit envelope against explicit routes.

    This validates internal consistency, not authenticity, independence, or a
    production guarantee. Local evidence still requires review under issue #26.
    Bare measured label lists are deliberately insufficient evidence.
    """
    try:
        _require(isinstance(envelope,dict) and envelope.get('schema_version')==2, 'measured evidence requires outcomes schema v2')
        _require(routes is not None, 'measured evidence requires explicit routes')
        _require(envelope.get('simulated',False) is False, 'simulated evidence is not measured')
        labels=_index(envelope['labels'],'measured labels')
        _require(set(labels)<=set(t['id'] for t in tasks), 'unknown measured task id')
        for label in labels.values():
            _require(label.get('label_source')==MEASURED and label.get('evidence')=='acceptance_tests', 'missing or conflicting measured label_source')
            _require(not any(k in label for k in ('needs_vision','task_type','task_understanding','rationale')), 'outcomes cannot supply task-understanding fields')
        metrics={}
        for metric in envelope['candidate_metrics']:
            key=(metric['task_id'],metric['route_fingerprint'])
            _require(key not in metrics, 'duplicate candidate metric identity')
            metrics[key]=metric
        records=[]
        for trial in envelope['trial_details']:
            metric=metrics[(trial['task_id'],trial['route_fingerprint'])]
            records.append({**trial, 'provider':metric['provider'], 'model':metric['model'],
                'reasoning_effort':metric['reasoning_effort'], 'model_version':metric['model_version'],
                'acceptance_spec_sha256':metric['acceptance_spec_sha256'], 'evidence_kind':'acceptance_tests'})
        replay=import_outcomes(tasks,records,routes,envelope['minimum_trials'],envelope['minimum_success_lb'],envelope['confidence'])
        # Compare canonical JSON so bool cannot impersonate an integer. Replay
        # also checks all candidates' completeness, fingerprints and tie-breaks.
        for key in ('schema_version','labels','pending','candidate_metrics','trial_details',
                    'minimum_trials','minimum_success_lb','confidence','bound','preregistration','estimand'):
            _require(json.dumps(envelope[key],sort_keys=True,allow_nan=False)==json.dumps(replay[key],sort_keys=True,allow_nan=False), f'measured gate/audit mismatch: {key}')
        return labels
    except (KeyError,TypeError,IndexError,AttributeError) as error:
        raise ValueError('missing or malformed measured gate/trial metadata') from error


def unify(tasks, experts, measured=None, routes=None):
    """Return unified task records and a separate measured-only shadow partition.

    A task may have both sources: unique IDs are required *within* each source;
    the cross-source same-ID association is an explicit join, not a duplicate.
    Candidate success probabilities reserve a future conditional-success head;
    observed rates/bounds remain evidence, never calibrated head targets.
    """
    by_id=validate_experts(tasks,experts)
    measured_by_id=validate_measured(tasks,measured,routes) if measured is not None else {}
    unified=[];shadow=[]
    for task in tasks:
        identity=task['id']; expert=copy.deepcopy(by_id[identity]); outcome=copy.deepcopy(measured_by_id.get(identity))
        unified.append({'schema_version':1,'id':identity,'task_understanding':expert,
            'measured_candidate_outcomes':outcome,'candidate_success_probabilities':None,
            'cold_start_training_source':EXPERT})
        if outcome is not None:
            shadow.append({**copy.deepcopy(task),'label':outcome,'task_understanding':expert,
                'partition':'measured_shadow','confidence_semantics':'interval_confidence_level_not_sample_weight',
                'candidate_success_probabilities':None,
                'outcome_metadata':{key:copy.deepcopy(measured[key]) for key in
                    ('schema_version','minimum_trials','minimum_success_lb','confidence','bound','preregistration','estimand')},
                'candidate_metrics':[copy.deepcopy(r) for r in measured['candidate_metrics'] if r['task_id']==identity],
                'trial_details':[copy.deepcopy(r) for r in measured['trial_details'] if r['task_id']==identity]})
    return unified,shadow
