"""Validate fresh expert hints and create leakage-resistant grouped splits."""
from __future__ import annotations
import argparse, collections, hashlib, json, random, re
from pathlib import Path

TIERS = ['fast', 'balanced', 'flagship']
TYPES = ['extraction', 'research', 'implementation', 'debugging', 'review', 'visual', 'creative']

def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def grams(prompt):
    text = re.sub(r'\s+', '', prompt.lower())
    return {text[i:i+5] for i in range(max(0,len(text)-4))}

def prepare(root: Path, seed=20261010, similarity=.88, schema_version=None, output=None):
    from task_label_schema import LEGACY, V2, version, partition_labels, metadata
    selected=version(schema_version)
    destination=root if output is None else Path(output)
    if selected==V2 and (output is None or destination.resolve()==root.resolve()):
        raise ValueError('v2 prepare needs a separate new output directory; frozen inputs are immutable')
    if output is not None and destination.exists():raise FileExistsError('prepare output already exists')
    tasks=load(root/'tasks.jsonl'); labels=[]
    for n in range(1,5): labels.extend(load(root/f'expert-recovery-{n}.jsonl'))
    assert len({r['id'] for r in labels})==len(labels), 'duplicate expert labels'
    by_id={r['id']:r for r in labels}
    assert set(by_id)=={r['id'] for r in tasks}, 'label coverage differs from task ids'
    labels,pending=partition_labels(labels,selected)
    by_id={r['id']:r for r in labels}
    for label in labels:
        assert label['tier'] in TIERS and label['task_type'] in (TYPES if selected==LEGACY else metadata(selected)['classes']['task_type'])
        assert label['needs_vision'] in (['yes','no','uncertain'] if selected==LEGACY else ['yes','no'])
        assert label['label_source']=='expert_provisional' and label['evidence']=='prompt_only'
        assert 0<=label['confidence']<=1 and label['rationale']
        assert not any(key in label for key in ['model','provider','historical_model'])
    parent={r['group']:r['group'] for r in tasks}
    def find(g):
        while parent[g]!=g: parent[g]=parent[parent[g]];g=parent[g]
        return g
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    sets=[grams(r['prompt']) for r in tasks]; near=[]
    for i,a in enumerate(tasks):
        for j in range(i):
            b=tasks[j]
            if a['group']==b['group']:continue
            x,y=sets[i],sets[j]
            if min(len(x),len(y))<similarity*max(len(x),len(y)):continue
            score=len(x&y)/max(1,len(x|y))
            if score>=similarity:union(a['group'],b['group']);near.append([a['id'],b['id'],round(score,4)])
    # Exclude unresolved supervision only AFTER duplicate unions: pending rows
    # may still connect two active roots and must not erase leakage edges.
    tasks=[row for row in tasks if row['id'] in by_id]
    grouped=collections.defaultdict(list)
    for row in tasks:
        label=by_id[row['id']]
        joined={**row,'label':label,'split_group':find(row['group'])}
        if selected==V2:
            from task_label_schema import validate_supervised_rows
            joined['task_label_schema_version']=selected
            validate_supervised_rows([joined],selected)
        grouped[find(row['group'])].append(joined)
    assert len(grouped)>=3,'Too few independent groups for train/calibration/test'
    keys=sorted(grouped); rng=random.Random(seed); best=None
    for _ in range(12000):
        assignment={key:rng.choices(['train','calibration','test'],weights=[.7,.15,.15])[0] for key in keys}
        splits={name:[row for key in keys if assignment[key]==name for row in grouped[key]] for name in ['train','calibration','test']}
        if any(len(rows)<20 for rows in splits.values()):continue
        # Only two roots contain visual tasks: prioritize train/test coverage;
        # do not manufacture a third independent visual calibration group.
        if any(not any(r['label']['needs_vision']=='yes' for r in splits[name]) for name in ['train','test']):continue
        score=sum(abs(len(splits[name])/len(tasks)-target) for name,target in [('train',.7),('calibration',.15),('test',.15)])
        for name,rows in splits.items():
            counts=collections.Counter(r['label']['tier'] for r in rows)
            score+=sum(2 for label in TIERS if counts[label]==0)
            score+=sum(.5 for label in ['yes','no'] if not any(r['label']['needs_vision']==label for r in rows))
            # Prefer distributions close to the whole corpus without mixing groups.
            for label in TIERS:
                whole=sum(r['tier']==label for r in labels)/len(labels)
                score+=.15*abs(counts[label]/len(rows)-whole)
        if best is None or score<best[0]:best=(score,splits)
    assert best is not None
    splits=best[1];report={'seed':seed,'label_basis':'new expert hints; no candidate-model execution evidence', 'near_duplicate_threshold':similarity,'cross_root_near_duplicate_edges':len(near),'independent_groups':len(keys),'splits':{}}
    if output is not None:destination.mkdir(parents=True,exist_ok=False)
    if selected==V2:
        report['label_schema']=metadata(selected)
        report['label_review']={'resolved':len(labels),'adjudication_or_excluded':len(pending),
                                'support_claim':'schema legality is not independent-group support'}
        (destination/'label-schema.json').write_text(json.dumps(metadata(selected),ensure_ascii=False,indent=2)+'\n')
        (destination/'adjudication.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pending))
    for name,rows in splits.items():
        rows.sort(key=lambda r:r['id'])
        (destination/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        report['splits'][name]={'rows':len(rows),'groups':sorted({r['split_group'] for r in rows}),'tiers':dict(collections.Counter(r['label']['tier'] for r in rows)),'vision':dict(collections.Counter(r['label']['needs_vision'] for r in rows)),'task_types':dict(collections.Counter(r['label']['task_type'] for r in rows))}
    names=list(splits)
    for i,a in enumerate(names):
        for b in names[:i]:assert not ({r['split_group'] for r in splits[a]} & {r['split_group'] for r in splits[b]})
    report['dataset_sha256']=hashlib.sha256((root/'tasks.jsonl').read_bytes()).hexdigest()
    report['group_disjoint']=True; report['historical_model_labels_used']=False
    report['limitations']=(['Only two independent roots contain vision-positive tasks; visual confidence calibration is not supported by a fully independent third root.'] if selected==LEGACY else ['Label legality does not establish A/B/C feasibility; independent-group support must be audited separately.'])+['Expert imitation metrics do not establish candidate-model task success.']
    (destination/'split-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (destination/'expert-labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(labels,key=lambda r:r['id'])))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).parent/'data');p.add_argument('--seed',type=int,default=20261010)
    from task_label_schema import VERSIONS, LEGACY
    p.add_argument('--label-schema',choices=VERSIONS,default=LEGACY)
    p.add_argument('--output',type=Path,help='required new output directory for v2; never overwrite frozen inputs')
    a=p.parse_args();prepare(a.data,a.seed,schema_version=a.label_schema,output=a.output)
