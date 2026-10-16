"""Explicit expert-only router-label-v2 / split-v2 bridge (#18).

No measured intake, label inference, supplemental evaluation, or research bridge.
The shared optimizer retains the fixed component cap and whole-root isolation.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from label_schema import finite_json, validate_experts
from task_label_schema import V2, version, metadata, check_row_versions, partition_labels, validate_supervised_rows
from split_v2 import _components, _assign

CONTRACT='router-label-v2+split-v2/expert-v1'


def split_task_v2(tasks,labels,*,schema_version,seed=20261010,similarity=.88,attempts=12000):
    if version(schema_version)!=V2:raise ValueError('combined split requires explicit router-label-v2')
    tasks=list(tasks);labels=list(labels)
    finite_json(tasks);finite_json(labels)
    check_row_versions(tasks,V2)
    if not tasks:raise ValueError('tasks must be nonempty')
    for row in tasks:
        if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('id','group','prompt')):
            raise ValueError('task requires nonempty id, group and prompt')
    if any(not isinstance(label,dict) or not isinstance(label.get('id'),str) or not label['id'].strip() for label in labels):
        raise ValueError('label requires nonempty id')
    ids=[r['id'] for r in tasks];label_ids=[r['id'] for r in labels]
    if len(set(ids))!=len(ids) or len(set(label_ids))!=len(label_ids):raise ValueError('duplicate task/label ids')
    if set(ids)!=set(label_ids):raise ValueError('label coverage differs from task ids')
    # Check provenance even for pending/excluded records: exclusion is not a
    # route for measured or mixed-version input to enter an expert-only bundle.
    check_row_versions(labels,V2)
    for label in labels:
        if label.get('label_source')!='expert_provisional' or label.get('evidence')!='prompt_only' or any(
                k in label for k in ('model','provider','historical_model','route_fingerprint','success_lb','success_rate')):
            raise ValueError('combined split requires expert_provisional / prompt_only provenance')
    active,pending=partition_labels(labels,V2)
    active_ids={r['id'] for r in active}
    active_tasks=[r for r in tasks if r['id'] in active_ids]
    if active:validate_experts(active_tasks,active)
    if type(seed) is not int or type(attempts) is not int or attempts<1:raise ValueError('integer seed and positive attempts required')
    if type(similarity) not in (float,int) or not math.isfinite(similarity) or not 0<similarity<=1:
        raise ValueError('similarity must be finite in (0,1]')
    tasks=sorted(tasks,key=lambda r:r['id']);labels=sorted(labels,key=lambda r:r['id'])
    groups,edges,roots=_components(tasks,{r['id']:r for r in labels},similarity)
    all_components=len(groups)
    # Build closure on ALL rows first. Even a pending-only root can bridge two
    # supervised roots; never recompute components after removing supervision.
    groups=[[r for r in group if r['id'] in active_ids] for group in groups]
    groups=[group for group in groups if group]
    for group in groups:validate_supervised_rows(group,V2)
    # Question head 'vision' is stored as label field 'needs_vision'.
    classes={('needs_vision' if head=='vision' else head):values for head,values in metadata(V2)['classes'].items()}
    splits,report=_assign(tasks,labels,groups,edges,roots,seed,similarity,attempts,classes)
    report.update(bundle_contract=CONTRACT,label_schema=metadata(V2),
                  cold_start_training_source='expert_provisional',
                  label_review={'resolved':len(active),'adjudication_or_excluded':len(pending),
                                'support_claim':'schema legality is not independent-group support'})
    report['grouping']['components_before_supervision_exclusion']=all_components
    report['limitations'].append('Expert imitation and synthetic software tests do not establish candidate success, real-label review, or calibration support.')
    return splits,report,pending


def prepare_task_split_v2(root,output,*,schema_version,**options):
    root=Path(root);output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('combined output already exists; choose a fresh directory')
    if root.resolve()==output.resolve():raise ValueError('inputs must remain immutable')
    def load(name):return [json.loads(line) for line in (root/name).read_text(encoding='utf-8').splitlines() if line.strip()]
    tasks=load('tasks.jsonl');labels=[]
    for n in range(1,5):labels.extend(load(f'expert-recovery-{n}.jsonl'))
    splits,report,pending=split_task_v2(tasks,labels,schema_version=schema_version,**options)
    report['outputs']={'report':'split-report.json','schema':'label-schema.json',
                       'adjudication':'adjudication.jsonl','splits':{},'labels':None}
    payloads={'label-schema.json':finite_json(metadata(V2),indent=2)+'\n',
              'adjudication.jsonl':''.join(finite_json(r)+'\n' for r in sorted(pending,key=lambda r:r['id']))}
    if splits is not None:
        for name,rows in splits.items():
            filename=f'{name}.jsonl';report['outputs']['splits'][name]=filename
            payloads[filename]=''.join(finite_json(r)+'\n' for r in rows)
        report['outputs']['labels']='expert-labels.jsonl'
        payloads['expert-labels.jsonl']=''.join(finite_json(r)+'\n' for r in sorted(labels,key=lambda r:r['id']) if r['label_status']=='resolved')
    payloads['split-report.json']=finite_json(report,indent=2)+'\n'
    # Validate/serialize before exclusive reservation. I/O failure may leave a
    # partial bundle for inspection; never remove or overwrite existing output.
    output.mkdir()
    for name,text in payloads.items():
        with (output/name).open('x',encoding='utf-8') as stream:stream.write(text)
    return report
