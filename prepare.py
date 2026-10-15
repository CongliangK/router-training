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

LEGACY_OUTPUTS = ('train.jsonl','calibration.jsonl','test.jsonl','split-report.json','expert-labels.jsonl')
UNIFIED_OUTPUTS = ('unified-labels.jsonl','measured-shadow.jsonl','measured-evidence.json')


def check_outputs(root, unified):
    # Never infer ownership from old contents or remove stale user files. New
    # unified bundles are exclusive; legacy-only reproduction keeps its 5 writes.
    for name in (*LEGACY_OUTPUTS,*UNIFIED_OUTPUTS) if unified else UNIFIED_OUTPUTS:
        path=root/name
        if path.exists() or path.is_symlink():
            raise ValueError(f'existing prepare output {name}; use a new dataset directory (no files removed)')
    if not unified:
        for name in LEGACY_OUTPUTS:
            path=root/name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f'legacy output {name} must be a regular file, not a symlink or directory')


def prepare(root: Path, seed=20261010, similarity=.88, *, outcomes=None, routes=None, unified=False, preflight=False, schema_version=None, output=None):
    from task_label_schema import version, V2
    selected=version(schema_version)
    if selected==V2:
        if outcomes is not None or routes is not None or unified or preflight:
            raise ValueError("router-label-v2 does not support unified/measured intake or prepare preflight")
        from prepare_task_v2 import prepare_task_v2
        return prepare_task_v2(root,seed,similarity,schema_version=selected,output=output)
    if output is not None:raise ValueError("v1 frozen prepare does not support separate output")
    tasks=load(root/'tasks.jsonl'); labels=[]
    for n in range(1,5): labels.extend(load(root/f'expert-recovery-{n}.jsonl'))
    from label_schema import finite_json, unify
    finite_json({'seed':seed,'similarity':similarity},context='split parameters')
    from task_label_schema import check_row_versions
    check_row_versions(tasks)
    check_row_versions([{'label':label} for label in labels])
    records,shadow=unify(tasks,labels,outcomes,routes)
    unified=unified or outcomes is not None
    check_outputs(root,unified)
    if preflight:
        summary={'status':'structurally_valid','tasks':len(tasks),'expert_training_rows':len(labels),
                 'measured_shadow_rows':len(shadow),'models_loaded':False,'outputs_written':False}
        print(json.dumps(summary,sort_keys=True));return summary
    assert len({r['id'] for r in labels})==len(labels), 'duplicate expert labels'
    by_id={r['id']:r for r in labels}
    assert set(by_id)=={r['id'] for r in tasks}, 'label coverage differs from task ids'
    for label in labels:
        assert label['tier'] in TIERS and label['task_type'] in TYPES
        assert label['needs_vision'] in ['yes','no','uncertain']
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
    grouped=collections.defaultdict(list)
    for row in tasks:
        label=by_id[row['id']]
        grouped[find(row['group'])].append({**row,'label':label,'split_group':find(row['group'])})
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
    payloads={}
    for name,rows in splits.items():
        rows.sort(key=lambda r:r['id'])
        payloads[f'{name}.jsonl']=''.join(finite_json(r)+'\n' for r in rows)
        report['splits'][name]={'rows':len(rows),'groups':sorted({r['split_group'] for r in rows}),'tiers':dict(collections.Counter(r['label']['tier'] for r in rows)),'vision':dict(collections.Counter(r['label']['needs_vision'] for r in rows)),'task_types':dict(collections.Counter(r['label']['task_type'] for r in rows))}
    names=list(splits)
    for i,a in enumerate(names):
        for b in names[:i]:assert not ({r['split_group'] for r in splits[a]} & {r['split_group'] for r in splits[b]})
    report['dataset_sha256']=hashlib.sha256((root/'tasks.jsonl').read_bytes()).hexdigest()
    report['group_disjoint']=True; report['historical_model_labels_used']=False
    report['limitations']=['Only two independent roots contain vision-positive tasks; visual confidence calibration is not supported by a fully independent third root.', 'Expert imitation metrics do not establish candidate-model task success.']
    if unified:
        report['label_schema_version']=1
        report['cold_start_training_source']='expert_provisional'
        report['measured_shadow_rows']=len(shadow)
        split_by_id={r['id']:(name,r['split_group']) for name,rows in splits.items() for r in rows}
        for row in shadow:
            row['expert_split'],row['split_group']=split_by_id[row['id']]
        for filename,rows in [('unified-labels.jsonl',records),('measured-shadow.jsonl',shadow)]:
            payloads[filename]=''.join(finite_json(r)+'\n' for r in sorted(rows,key=lambda r:r['id']))
        if outcomes is not None:
            payloads['measured-evidence.json']=finite_json(outcomes,indent=2)+'\n'
    report_json=finite_json(report,indent=2)
    payloads['split-report.json']=report_json+'\n'
    payloads['expert-labels.jsonl']=''.join(finite_json(r)+'\n' for r in sorted(labels,key=lambda r:r['id']))
    # All expected validation/serialization failures precede ANY output opens.
    # This is not a multi-file filesystem transaction: I/O errors/crashes may
    # leave a partial bundle. Keep it for inspection; never auto-delete it.
    check_outputs(root,unified)
    for filename,text in payloads.items():
        with (root/filename).open('x' if unified else 'w',encoding='utf-8') as output:output.write(text)
    print(report_json)

# Explicit new entry points; the legacy split algorithm/defaults above stay unchanged.
from split_v2 import prepare_v2, split_v2


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=None);p.add_argument('--seed',type=int,default=20261010)
    p.add_argument('--split-version',choices=['frozen-v1','split-v2'],default='frozen-v1')
    p.add_argument('--output',type=Path,help='split-v2 only: fresh, nonexistent bundle directory (parent must exist)')
    p.add_argument('--unified',action='store_true',help='frozen-v1 only: create a fresh output bundle with unified labels and a separate measured shadow partition')
    p.add_argument('--outcomes',type=Path,help='frozen-v1 only: full outcomes.py v2 audit JSON, never a bare measured label list')
    p.add_argument('--routes',type=Path,help='frozen-v1 only: explicit candidate routes matching the measured preregistration')
    p.add_argument('--preflight',action='store_true',help='frozen-v1 only: validate finite JSON, structure/gate and output conflicts; no splitting, writes, torch or models')
    from task_label_schema import LEGACY, V2, VERSIONS
    p.add_argument('--label-schema',choices=VERSIONS,default=LEGACY)
    a=p.parse_args(argv)
    if a.label_schema==V2:
        if a.split_version!='frozen-v1' or a.unified or a.outcomes or a.routes or a.preflight:
            p.error('router-label-v2 does not support split-v2, unified/measured intake or prepare preflight')
        if a.data is None or a.output is None:p.error('router-label-v2 requires explicit --data and --output')
        prepare(a.data,a.seed,schema_version=a.label_schema,output=a.output)
        return 0
    if a.split_version=='split-v2':
        # These contracts have distinct output lifecycles. Never silently drop
        # measured evidence or turn a requested no-write preflight into a split.
        unsupported=[flag for flag,value in (('--unified',a.unified),('--outcomes',a.outcomes),
                     ('--routes',a.routes),('--preflight',a.preflight)) if value]
        if unsupported:p.error('split-v2 does not support '+', '.join(unsupported)+'; use --split-version frozen-v1 for unified intake/preflight')
        if a.data is None or a.output is None:p.error('split-v2 requires explicit --data and --output')
        report=prepare_v2(a.data,a.output,seed=a.seed)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if report['status']=='feasible' else 2
    if a.output is not None:p.error('--output requires --split-version split-v2')
    if a.outcomes and not a.routes:p.error('--outcomes requires --routes')
    if a.routes and not a.outcomes:p.error('--routes requires --outcomes')
    evidence=json.loads(a.outcomes.read_text()) if a.outcomes else None
    if a.outcomes and not isinstance(evidence,dict):p.error('--outcomes requires a full v2 audit object')
    prepare(a.data if a.data is not None else Path(__file__).parent/'data',a.seed,outcomes=evidence,
            routes=json.loads(a.routes.read_text()) if a.routes else None,unified=a.unified,preflight=a.preflight)
    return 0


if __name__=='__main__':raise SystemExit(main())
