"""Import new candidate execution evidence; never derive labels from inherited routes."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, re
from pathlib import Path
from policy import DEFAULT_ROUTES,TIERS

def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def import_outcomes(tasks,records,min_trials=3,min_success_rate=.9):
    ids={row['id'] for row in tasks};routes={(row['provider'],row['model']):row for row in DEFAULT_ROUTES};grouped=collections.defaultdict(list);seen=set()
    for record in records:
        if record['task_id'] not in ids:raise ValueError('Unknown task id')
        pair=(record['provider'],record['model'])
        if pair not in routes:raise ValueError('Unknown candidate route')
        if type(record['passed']) is not bool:raise ValueError('Acceptance verdict must be boolean')
        effort=record.get('reasoning_effort')
        if effort is not None and (not isinstance(effort,str) or not effort.strip() or len(effort)>200):raise ValueError('Invalid measured reasoning effort')
        if record.get('evidence_kind')!='acceptance_tests':raise ValueError('Task-success evidence requires acceptance tests/rubric, not a model preference')
        if record.get('simulated',False):raise ValueError('Simulated results cannot produce measured labels')
        for key in ['acceptance_spec_sha256','artifact_sha256']:
            if not re.fullmatch('[a-f0-9]{64}',record.get(key,'')):raise ValueError('Evidence requires acceptance and artifact hashes')
        for key in ['cost','latency_ms','input_tokens','output_tokens']:
            value=record.get(key)
            if type(value) not in [float,int] or not math.isfinite(value) or value<0:raise ValueError('Invalid outcome resource metric')
        identity=(record['task_id'],*pair,record['trial'])
        if identity in seen:raise ValueError('Duplicate execution trial')
        seen.add(identity);grouped[record['task_id'],pair].append(record)
    labels=[];pending=[];stats=[]
    for task_id in sorted(ids):
        candidates=[];complete=True
        for pair,route in routes.items():
            attempts=grouped[task_id,pair];count=len(attempts)
            if count<min_trials:complete=False
            specs={r['acceptance_spec_sha256'] for r in attempts}
            if len(specs)>1:raise ValueError('Acceptance criteria changed across trials')
            if not attempts:continue
            efforts={r.get('reasoning_effort') for r in attempts}
            if len(efforts)>1:raise ValueError('Reasoning effort changed across trials; import separate operating points')
            effort=next(iter(efforts))
            success=sum(row['passed'] for row in attempts)/count
            row={'task_id':task_id,'provider':pair[0],'model':pair[1],'tier':route['tier'],'reasoning_effort':effort,'trials':count,'success_rate':success,'mean_cost':sum(r['cost'] for r in attempts)/count,'mean_latency_ms':sum(r['latency_ms'] for r in attempts)/count,'acceptance_spec_sha256':next(iter(specs))}
            stats.append(row)
            if count>=min_trials and success>=min_success_rate:candidates.append(row)
        all_specs={r['acceptance_spec_sha256'] for (tid,_),attempts in grouped.items() if tid==task_id for r in attempts}
        if len(all_specs)>1:raise ValueError('Candidates were judged with different criteria')
        if not complete or not candidates:
            pending.append({'id':task_id,'reason':'incomplete-candidate-trials' if not complete else 'no-candidate-passed'});continue
        selected=min(candidates,key=lambda r:(r['mean_cost'],r['mean_latency_ms'],TIERS.index(r['tier'])))
        labels.append({'id':task_id,'tier':selected['tier'],'provider':selected['provider'],'model':selected['model'],'reasoning_effort':selected['reasoning_effort'],'label_source':'measured_candidate_outcomes','evidence':'acceptance_tests','success_rate':selected['success_rate'],'trials':selected['trials'],'acceptance_spec_sha256':selected['acceptance_spec_sha256']})
    return {'labels':labels,'pending':pending,'candidate_metrics':stats,'minimum_trials':min_trials,'minimum_success_rate':min_success_rate,'note':'Small trial counts are preliminary measurements, not calibrated production reliability.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--outcomes',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--min-trials',type=int,default=3);p.add_argument('--min-success-rate',type=float,default=.9)
    a=p.parse_args()
    if a.min_trials<1 or not 0<a.min_success_rate<=1:raise SystemExit('Invalid outcome acceptance thresholds')
    tasks=rows(a.tasks)
    if a.outcomes:
        result=import_outcomes(tasks,rows(a.outcomes),a.min_trials,a.min_success_rate)
    else:
        # A plan is not execution evidence. Tasks need snapshots and acceptance specs.
        result={'status':'planned-not-executed','task_corpus_sha256':hashlib.sha256(a.tasks.read_bytes()).hexdigest(),'candidates':DEFAULT_ROUTES,'required_trials_per_candidate':a.min_trials,'tasks':[{'id':r['id'],'prompt':r['prompt'],'required_context_snapshot':True,'required_acceptance_spec':True} for r in tasks]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('outcome import' if a.outcomes else 'execution plan',a.output)
