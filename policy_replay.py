"""Stdlib-only policy replay. Local predictions stay local; outputs contain aggregates only."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from policy import DEFAULT_ROUTES,TIERS,select,validate_policy,validate_routes

CATEGORIES=('model_choice','vision_forced','context_truncation','risk_policy')


def synthetic_rows():
    """Illustrative cases, NOT sampled from private data or calibrated evidence."""
    result=[]
    for p,image,truncated,label in [(.02,False,False,'no'),(.12,False,False,'no'),
            (.16,False,False,'no'),(.3,False,False,'yes'),(.45,False,False,'yes'),
            (.9,False,False,'yes'),(.01,True,False,'yes'),(.01,False,True,'no')]:
        result.append({'prediction':{'answers':{'tier':{'choice':'balanced','confidence':.95},
            'vision':{'choice':'yes' if p>=.5 else 'no','confidence':max(p,1-p),'probabilities':{'yes':p}}},
            'context_truncated':truncated},'has_image':image,'vision_label':label})
    return result


def replay(rows,routes,minimum_confidence=.9,*,vision_policy='legacy-v1',vision_positive_threshold=None,costs=None):
    # Empty evidence still requires the exact configuration contract used by select.
    validate_policy(minimum_confidence,vision_policy,vision_positive_threshold)
    if routes!=[]:validate_routes(routes)
    # Cost inputs are explicit per-decision estimates, not tier-distance or measured savings.
    cost_index={}
    for item in costs or []:
        pair=(item['provider'],item['model']);value=item['cost']
        if pair in cost_index or type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Invalid replay cost declaration')
        cost_index[pair]=value
    n=upgrades=refusals=vision_n=fn=vision_refusals=cost_n=selected_n=0;total_cost=0.
    counts=dict.fromkeys(CATEGORIES,0);upgraded=dict.fromkeys(CATEGORIES,0)
    for row in rows:
        if 'has_image' not in row:raise ValueError('Replay requires explicit has_image')
        label=row.get('vision_label')
        if label is not None and label not in ('yes','no'):raise ValueError('Invalid vision label')
        d=select(row['prediction'],routes,row['has_image'],minimum_confidence,vision_policy=vision_policy,vision_positive_threshold=vision_positive_threshold)
        n+=1;refusals+=d['recommendation'] is None
        is_upgrade=d['selected_tier'] is not None and TIERS.index(d['selected_tier'])>TIERS.index(d['model_tier'])
        upgrades+=is_upgrade
        for category in CATEGORIES:
            counts[category]+=any(e['category']==category for e in d['reason_details'])
            upgraded[category]+=is_upgrade and any(e['category']==category and e['upgraded'] for e in d['reason_details'])
        if label=='yes':
            vision_n+=1;fn+=not d['needs_vision'];vision_refusals+=d['recommendation'] is None
        if d['recommendation'] is not None:
            selected_n+=1;r=d['recommendation'];cost=cost_index.get((r['provider'],r['model']))
            if cost is not None:cost_n+=1;total_cost+=cost
    ratio=lambda x,den:x/den if den else None
    if not math.isfinite(total_cost):raise ValueError('Nonfinite aggregate cost')
    return {'rows':n,'upgrades':upgrades,'upgrade_rate':ratio(upgrades,n),'refusals':refusals,'refusal_rate':ratio(refusals,n),
        'reason_counts':counts,'reason_rates':{k:ratio(v,n) for k,v in counts.items()},
        'upgrade_reason_counts':upgraded,'upgrade_reason_shares':{k:ratio(v,upgrades) for k,v in upgraded.items()},
        'vision_positive_labels':vision_n,'vision_false_negatives':fn,'vision_false_negative_rate':ratio(fn,vision_n),
        'vision_refusals':vision_refusals,'vision_refusal_rate':ratio(vision_refusals,vision_n),
        'selected_decisions':selected_n,'cost_supported_decisions':cost_n,
        'estimated_total_cost':total_cost if selected_n and cost_n==selected_n else None,
        'cost_note':'Explicit per-selected-decision estimates only; null unless every selection has cost support. Refusals are not savings.',
        'minimum_confidence':minimum_confidence,'vision_policy':vision_policy,'vision_positive_threshold':vision_positive_threshold}


def study(rows,routes,*,input_kind='local-unverified',vision_positive_threshold=None,costs=None):
    if vision_positive_threshold is not None:
        validate_policy(vision_policy='positive-v2',vision_positive_threshold=vision_positive_threshold)
    rows=list(rows)
    baseline=replay(rows,routes,costs=costs)
    # Simulation changes capabilities only in memory. It authorizes no real candidate.
    simulated=[{**r,'vision':True} if r['tier']=='balanced' else dict(r) for r in routes]
    if not any(r['tier']=='balanced' for r in simulated):
        provider='synthetic-policy-replay'
        while any(r['provider']==provider for r in simulated):provider+='-simulation'
        simulated.append({'tier':'balanced','provider':provider,'model':'balanced-vision','vision':True})
    comparison=replay(rows,simulated,costs=costs)
    return {'replay_schema_version':1,'input_kind':input_kind,'execution_performed':False,
        'note':'Synthetic/local policy replay only, not actual model replay, backend capability evidence, calibrated threshold selection, or production savings. Reason categories overlap; shares need not sum to one.',
        'baseline':baseline,'legacy_confidence_sensitivity':[replay(rows,routes,c,costs=costs) for c in (.8,.85,.9,.95)],
        'simulated_balanced_vision':comparison,
        'simulated_upgrade_rate_reduction':None if baseline['upgrade_rate'] is None else baseline['upgrade_rate']-comparison['upgrade_rate'],
        'positive_threshold_experiment':None if vision_positive_threshold is None else replay(rows,routes,vision_policy='positive-v2',vision_positive_threshold=vision_positive_threshold,costs=costs)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--synthetic',action='store_true');source.add_argument('--input',type=Path)
    p.add_argument('--routes',type=Path);p.add_argument('--costs',type=Path);p.add_argument('--vision-positive-threshold',type=float);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=synthetic_rows() if a.synthetic else [json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    routes=json.loads(a.routes.read_text()) if a.routes else DEFAULT_ROUTES
    costs=json.loads(a.costs.read_text()) if a.costs else None
    result=study(rows,routes,input_kind='synthetic' if a.synthetic else 'local-unverified',vision_positive_threshold=a.vision_positive_threshold,costs=costs)
    # Serialize and encode before reserving a name: expected failures leave no empty output.
    payload=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode('utf-8')
    # Refuse overwrites and never echo paths, prompts, IDs, or prediction rows.
    with a.output.open('xb') as file:file.write(payload)
    print('Policy replay complete; aggregate-only output written. No model or backend executed.')


if __name__=='__main__':main()
