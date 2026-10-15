"""Aggregate-only frozen-v1 diagnostics from explicitly supplied, fixed rows.

Never selects a model, fits temperatures, changes splits, or loads weights.
Source group/root identifiers and row contents are deliberately not exported.
"""
import collections
from evaluation_metrics import metrics, target
from model_utils import QUESTIONS
from task_label_schema import LEGACY, check_row_versions, resolve_run_schema

NOTICE='frozen-v1, known skew / frozen-v1，已知偏斜'
LIMITS=[
    'Exploratory expert-label agreement only; frozen-v1 has already been inspected.',
    'Do not select models, hyperparameters, temperatures or thresholds using this test report; use calibration/new groups.',
    'Fixed-prediction group deletion is sensitivity analysis, not retraining or leave-one-group-out cross-validation.',
    'Bootstrap resamples supplied split_group clusters; it cannot create independent sources or establish their independence.',
    'Root provenance is the supplied group field; split_group may combine roots. Neither count establishes independent sources.',
    'Real frozen-v1 replay acceptance is separate (#27); report code/synthetic tests do not establish model quality.',
]

def unavailable(reason):return {'status':'unavailable','reason':reason}


def counts(rows):
    out={'rows':len(rows),'classes':{},'excluded_labels':{}}
    for field,spec in QUESTIONS.items():
        labels=list(spec['criteria']);values=[target(r,field) for r in rows]
        out['classes'][field]={label:values.count(label) for label in labels}
        out['excluded_labels'][field]=sum(v not in labels for v in values)
    return out


def valid_id(value):return isinstance(value,str) and bool(value.strip())


def aggregate_counts(source):
    """Whitelist numeric aggregates; never echo arbitrary strings or row data."""
    out={}
    for split in ['train','calibration','test']:
        item=source.get('splits',{}).get(split)
        if item is None:out[split]=unavailable('split absent from aggregate input');continue
        result={}
        for field,key in [('tier','tiers'),('vision','vision'),('task_type','task_types')]:
            values=item.get(key)
            if values is None:result[field]=unavailable('category counts absent');continue
            if not isinstance(values,dict) or any(type(v) is not int or v<0 for v in values.values()):
                raise ValueError('aggregate category counts must be nonnegative integers')
            result[field]={label:values.get(label,0) for label in QUESTIONS[field]['criteria']}
            result[field]['excluded_label_count']=sum(v for k,v in values.items() if k not in QUESTIONS[field]['criteria'])
        n=item.get('rows')
        if n is not None and (type(n) is not int or n<0):raise ValueError('aggregate rows must be a nonnegative integer')
        result['rows']=n
        result['group_sizes']=unavailable('aggregate group IDs do not provide per-group row counts')
        out[split]=result
    return out


def provenance(rows,reference_rows=None):
    groups={r['split_group'] for r in rows}
    roots_complete=all(valid_id(r.get('group')) for r in rows)
    roots=sorted({r['group'] for r in rows}) if roots_complete else []
    reference_rows=rows if reference_rows is None else reference_rows
    reference_roots=sorted({r['group'] for r in reference_rows if valid_id(r.get('group'))})
    root_alias={g:f'root_{i+1}' for i,g in enumerate(reference_roots)}
    group_alias={g:f'group_{i+1}' for i,g in enumerate(sorted({r['split_group'] for r in reference_rows}))}
    entries=[]
    for g in sorted(groups,key=lambda g:(-sum(r['split_group']==g for r in rows),g)):
        subset=[r for r in rows if r['split_group']==g]
        entry={'group':group_alias[g],**counts(subset)}
        entry['roots']=sorted({root_alias[r['group']] for r in subset}) if roots_complete else None
        entries.append(entry)
    return {'observed_split_groups':len(groups),'groups':entries,
            'root_provenance':{'status':'available' if roots_complete and rows else 'unavailable',
                               'field':'group','observed_roots':len(roots) if roots_complete and rows else None,
                               'reason':None if roots_complete and rows else 'no rows or missing explicit root group on at least one row'},
            'independence':'unverified; counts are observed identities, not independent-source evidence'},group_alias


def summary(evaluation):
    # Deliberately keep vision separate from tier and task-type scores.
    return {'tier_accuracy':evaluation['tier']['accuracy'],'tier_macro_f1':evaluation['tier']['macro_f1'],
            'flagship_recall':evaluation['tier_per_class']['flagship']['recall'],
            'vision_recall':evaluation['vision_per_class']['yes']['recall'],
            'task_type_macro_f1':evaluation['task_type']['macro_f1']}


def split_report(source,rows=None):
    # This reader makes frozen-v1/seven-class claims even without predictions.
    # Guard BEFORE aggregate counting and ALL early returns, not only metrics().
    if not isinstance(source,dict):raise ValueError('split report input must be an object')
    check_row_versions([source],LEGACY)
    if 'label_schema' in source:
        if not isinstance(source['label_schema'],dict):raise ValueError('complete label schema metadata required')
        resolve_run_schema({'label_schema':source['label_schema']},requested=LEGACY)
    if rows is not None:
        rows=list(rows);check_row_versions(rows,LEGACY)
    result={'schema_version':1,'dataset_notice':NOTICE,'limitations':list(LIMITS),
            'aggregate_input_counts':aggregate_counts(source),
            'prediction_scope':'caller-supplied test rows only; frozen scores, no fitting or selection',
            'vision_notice':'Historical frozen-v1 vision positives are mainly supported by one task group; this is not an overall quality score.',
            'bootstrap_protocol':{'seed':0,'samples':1000,'unit':'split_group','interpretation':'descriptive cluster resampling; not proof of source independence'}}
    if rows is None:
        for key in ['row_support','group_provenance','evaluation','group_deletion','deletion_metric_ranges']:
            result[key]=unavailable('no explicit per-row labels/predictions supplied')
        return result
    rows=list(rows)
    if any(not isinstance(r,dict) or not isinstance(r.get('label'),dict) or
           any(k not in r['label'] for k in ['tier','needs_vision','task_type']) for r in rows):
        raise ValueError('each row requires explicit tier, needs_vision and task_type labels')
    result['row_support']=counts(rows)
    declared=result['aggregate_input_counts'].get('test',{}).get('rows')
    result['aggregate_row_count_matches']=None if declared is None else declared==len(rows)
    result['row_input_note']='Counts and scores below describe only supplied rows; aggregate counts do not reconstruct missing predictions.'
    if any(not valid_id(r.get('split_group')) for r in rows):
        for key in ['group_provenance','evaluation','group_deletion','deletion_metric_ranges']:
            result[key]=unavailable('missing valid split_group; grouping is not inferred from row IDs or roots')
        return result
    result['group_provenance'],aliases=provenance(rows)
    if any('answers' not in r for r in rows):
        for key in ['evaluation','group_deletion','deletion_metric_ranges']:result[key]=unavailable('prediction answers absent on at least one supplied row')
        return result
    # Auxiliary latency/context are not required for prediction-only input and
    # are not fabricated as observations. Shared metric helpers expect them.
    normalized=[{**r,'context_truncated':False,'latency_ms':0} for r in rows]
    def evaluate(subset):
        m=metrics(subset)
        for key in ['context_truncated','mean_latency_ms']:m.pop(key)
        return m
    baseline=evaluate(normalized);result['evaluation']=baseline
    base_summary=summary(baseline);deletions=[]
    for entry in result['group_provenance']['groups']:
        subset=[r for r in normalized if aliases[r['split_group']]!=entry['group']]
        evaluation=evaluate(subset);values=summary(evaluation)
        remaining,_=provenance(subset,normalized)
        deletions.append({'removed_group':entry['group'],'removed_rows':entry['rows'],
                          'remaining_support':counts(subset),'remaining_provenance':remaining,
                          'metrics':evaluation,'change_from_full':{k:None if v is None or base_summary[k] is None else v-base_summary[k] for k,v in values.items()}})
    result['group_deletion']={'status':'available' if rows else 'unavailable','reason':None if rows else 'no groups to delete',
                              'method':'delete each observed split_group from fixed predictions, descending size; no retraining',
                              'results':deletions}
    result['deletion_metric_ranges']={}
    for name in base_summary:
        values=[summary(d['metrics'])[name] for d in deletions];defined=[v for v in values if v is not None]
        result['deletion_metric_ranges'][name]={'min':min(defined) if defined else None,'max':max(defined) if defined else None,
                                               'defined_deletions':len(defined),'undefined_deletions':len(values)-len(defined)}
    return result
