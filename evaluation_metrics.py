"""Shared CPU-only evaluation metrics; no training or model imports."""
import collections, math, random
from model_utils import TIERS, QUESTIONS
from task_label_schema import LEGACY, V2, version, questions_for, check_row_versions, validate_supervised_rows, validate_prediction_rows

def target(row,field):
    return row['label'][{'tier':'tier','vision':'needs_vision','task_type':'task_type'}[field]]


def per_class(valid,field,low_support_threshold=10,schema_version=LEGACY):
    """Fixed-schema macro F1; undefined class ratios are null, never evidence.

    Historical macro-F1 includes EVERY schema class with zero_division=0.
    Keep that convention explicit, even when a class F1 itself is undefined.
    An entirely empty evaluation has no macro score.
    """
    if type(low_support_threshold) is not int or low_support_threshold<1:
        raise ValueError('low_support_threshold must be a positive integer')
    check_row_versions(valid,schema_version)
    if version(schema_version)==V2:validate_prediction_rows(valid,schema_version)
    labels=list(questions_for(schema_version,QUESTIONS)[field]['criteria']);n=len(valid);out={}
    for c in labels:
        tp=sum(1 for r in valid if r['answers'][field]['choice']==c and target(r,field)==c)
        fp=sum(1 for r in valid if r['answers'][field]['choice']==c and target(r,field)!=c)
        fn=sum(1 for r in valid if r['answers'][field]['choice']!=c and target(r,field)==c)
        out[c]={'tp':tp,'fp':fp,'fn':fn,'tn':n-tp-fp-fn,'support':tp+fn,'predicted_support':tp+fp,
                'low_support':tp+fn<low_support_threshold,
                'precision':tp/(tp+fp) if tp+fp else None,
                'recall':tp/(tp+fn) if tp+fn else None,
                'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    out['macro_f1']=sum(out[c]['f1'] or 0 for c in labels)/len(labels) if n else None
    out['macro_f1_definition']={'classes':labels,'denominator':len(labels),'zero_division':0,
                                'empty_evaluation':None,'excluded_classes':[]}
    out['low_support_threshold']=low_support_threshold
    out['low_support_classes']=[c for c in labels if out[c]['low_support']]
    out['warning']='Low-support classes remain in fixed-class macro-F1; do not select a model by this metric alone.' if out['low_support_classes'] else None
    return out


def bootstrap_ci(valid,field,statistic,samples=1000,seed=0):
    """95% percentile interval, resampling the FULL field-valid group pool.

    Do not prefilter to a target class before sampling: whole groups are the
    independent units. Undefined ratio draws are omitted, explicitly counted,
    and intervals are conditional on nonzero denominators. Fewer than two
    groups cannot support an empirical cluster interval. Missing group IDs do
    not establish independence, so never silently substitute row IDs.
    """
    if type(samples) is not int or samples<2:raise ValueError('samples must be an integer >= 2')
    out={'low':None,'high':None,'status':'unavailable','method':'whole-group percentile bootstrap',
         'confidence_level':.95,'field':field,'evaluated_rows':len(valid),'independent_groups':0,
         'supporting_groups':0,'requested_resamples':samples,'valid_resamples':0,'undefined_resamples':0,'seed':seed,
         'scope':'all field-valid rows; conditional on a nonzero statistic denominator'}
    if not valid:out['reason']='no evaluated rows';return out
    groups=collections.defaultdict(list)
    for r in valid:
        group=r.get('split_group')
        if not isinstance(group,str) or not group.strip():
            out['reason']='missing valid split_group; row independence is not assumed';return out
        groups[group].append(r)
    buckets=list(groups.values());out['independent_groups']=len(buckets)
    out['supporting_groups']=sum(statistic(bucket) is not None for bucket in buckets)
    if len(buckets)<2:out['reason']='fewer than two independent groups';return out
    if statistic(valid) is None:out['reason']='statistic has no denominator support';return out
    if out['supporting_groups']<2:out['reason']='fewer than two groups support the statistic denominator';return out
    rng=random.Random(seed);values=[]
    for _ in range(samples):
        draw=[row for bucket in (rng.choice(buckets) for _ in buckets) for row in bucket]
        value=statistic(draw)
        if value is None:out['undefined_resamples']+=1
        elif not math.isfinite(value):raise ValueError('bootstrap statistic must be finite or None')
        else:values.append(value)
    out['valid_resamples']=len(values)
    if len(values)<2:out['reason']='fewer than two defined resamples';return out
    values.sort()
    out.update(low=values[int(.025*len(values))],high=values[min(len(values)-1,int(.975*len(values)))],status='available',
               warning='Few groups and conditional resampling can understate uncertainty; this interval is not a production qualification.')
    return out


def validate_answer(answer,labels):
    """Fail closed on malformed probabilities rather than emit invalid scores."""
    probs=answer['probabilities'];confidence=answer['confidence']
    if answer['choice'] not in labels or set(probs)!=set(labels):
        raise ValueError('prediction must use the complete fixed class schema')
    for value in [*probs.values(),confidence]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('probabilities and confidence must be finite numbers in [0,1]')
    if not math.isclose(sum(probs.values()),1,rel_tol=0,abs_tol=1e-6):
        raise ValueError('probabilities must sum to one')
    if not math.isclose(confidence,probs[answer['choice']],rel_tol=0,abs_tol=1e-6):
        raise ValueError('confidence must equal the chosen class probability')


def calibration_bins(valid,field):
    # No epsilon on an upper edge: it double-counted internal boundaries
    # (all correct at confidence .4 gave ECE 1.2 instead of .6; issue #3).
    from bisect import bisect_right
    edges=[0,.2,.4,.6,.8,1.];buckets=[[] for _ in range(5)];out=[]
    for row in valid:buckets[bisect_right(edges[1:-1],row['answers'][field]['confidence'])].append(row)
    for i,bucket in enumerate(buckets):
        n=len(bucket)
        confidence=sum(r['answers'][field]['confidence'] for r in bucket)/n if n else None
        accuracy=sum(r['answers'][field]['choice']==target(r,field) for r in bucket)/n if n else None
        weight=n/len(valid) if valid else None
        out.append({'lower':edges[i],'upper':edges[i+1],'upper_inclusive':i==4,'count':n,
                    'weight':weight,'mean_confidence':confidence,'accuracy':accuracy,
                    'contribution':weight*abs(confidence-accuracy) if n else (0 if valid else None)})
    return out


def routing_costs(valid,cost_matrix=None):
    """Synthetic ordinal cost, NOT money or measured generation/task success.

    Matrix axes are true tier -> predicted tier. A caller may supply a
    predeclared finite nonnegative loss matrix with a zero diagonal.
    """
    labels=list(TIERS)
    default=cost_matrix is None
    if default:cost_matrix={a:{b:abs(i-j) for j,b in enumerate(labels)} for i,a in enumerate(labels)}
    if not isinstance(cost_matrix,dict) or set(cost_matrix)!=set(labels):raise ValueError('cost matrix must contain every true tier')
    for a in labels:
        if not isinstance(cost_matrix[a],dict) or set(cost_matrix[a])!=set(labels):raise ValueError('cost matrix must contain every predicted tier')
        for b,value in cost_matrix[a].items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (a==b and value!=0):
                raise ValueError('cost matrix must be finite, nonnegative, with a zero diagonal')
    out={'evaluated_rows':len(valid),'cost_basis':'ordinal tier distance (illustrative, not currency)' if default else 'caller-supplied loss units (not validated task success)',
         'cost_matrix':cost_matrix,'downgrade':{'count':0,'weighted_cost':0},'overroute':{'count':0,'weighted_cost':0},
         'by_transition':{a:{b:{'count':0,'weighted_cost':0} for b in labels} for a in labels}}
    for row in valid:
        true=target(row,'tier');pred=row['answers']['tier']['choice'];cost=cost_matrix[true][pred]
        cell=out['by_transition'][true][pred];cell['count']+=1;cell['weighted_cost']+=cost
        if true!=pred:
            kind='downgrade' if labels.index(pred)<labels.index(true) else 'overroute'
            out[kind]['count']+=1;out[kind]['weighted_cost']+=cost
    out['total_weighted_cost']=out['downgrade']['weighted_cost']+out['overroute']['weighted_cost']
    out['mean_cost']=out['total_weighted_cost']/len(valid) if valid else None
    return out


def metrics(rows,low_support_threshold=10,cost_matrix=None,schema_version=LEGACY):
    selected=version(schema_version)
    validate_prediction_rows(rows,selected)
    questions=questions_for(selected,QUESTIONS)
    result={'rows':len(rows),'context_truncated':sum(r['context_truncated'] for r in rows),
            'mean_latency_ms':sum(r['latency_ms'] for r in rows)/len(rows) if rows else None,
            'metric_definition_version':2,'nll_probability_floor':1e-12}
    valid_by_field={}
    result['task_label_schema_version']=selected
    for field in questions:
        valid=[r for r in rows if target(r,field) in questions[field]['criteria']];labels=list(questions[field]['criteria'])
        valid_by_field[field]=valid
        matrix={label:dict.fromkeys(labels,0) for label in labels};nll=brier=0;n=len(valid)
        for row in valid:
            expected=target(row,field);answer=row['answers'][field];validate_answer(answer,labels)
            matrix[expected][answer['choice']]+=1
            nll-=math.log(max(1e-12,answer['probabilities'][expected]))
            brier+=sum((answer['probabilities'][label]-(label==expected))**2 for label in labels)
        correct=sum(matrix[l][l] for l in labels);bins=calibration_bins(valid,field)
        pc=per_class(valid,field,low_support_threshold,selected)
        result[field]={'evaluated_rows':n,'excluded_rows':len(rows)-n,'status':'available' if n else 'unavailable',
                       'accuracy':correct/n if n else None,'macro_f1':pc['macro_f1'],'nll':nll/n if n else None,
                       'brier':brier/n if n else None,'ece_5_bins':sum(b['contribution'] for b in bins) if n else None,
                       'ece_bins':bins,'confusion':matrix}
        result[field+'_per_class']=pc
    tier_valid=valid_by_field['tier'];vision_valid=valid_by_field['vision']
    flagship=[r for r in tier_valid if target(r,'tier')=='flagship'];vision=[r for r in vision_valid if target(r,'vision')=='yes']
    result['flagship_underroute_rate']=sum(r['answers']['tier']['choice']!='flagship' for r in flagship)/len(flagship) if flagship else None
    result['vision_false_negative_rate']=sum(r['answers']['vision']['choice']!='yes' for r in vision)/len(vision) if vision else None
    result['flagship_examples']=len(flagship);result['vision_positive_examples']=len(vision)
    result['routing_costs']=routing_costs(tier_valid,cost_matrix)
    def ratio(rs,field,label,precision=False):
        subset=[r for r in rs if (r['answers'][field]['choice'] if precision else target(r,field))==label]
        return sum(r['answers'][field]['choice']==target(r,field) for r in subset)/len(subset) if subset else None
    result['bootstrap']={
        'tier_accuracy':bootstrap_ci(tier_valid,'tier',lambda rs:sum(r['answers']['tier']['choice']==target(r,'tier') for r in rs)/len(rs) if rs else None),
        'flagship_recall':bootstrap_ci(tier_valid,'tier',lambda rs:ratio(rs,'tier','flagship')),
        'flagship_precision':bootstrap_ci(tier_valid,'tier',lambda rs:ratio(rs,'tier','flagship',True)),
        'vision_recall':bootstrap_ci(vision_valid,'vision',lambda rs:ratio(rs,'vision','yes')),
    }
    return result
