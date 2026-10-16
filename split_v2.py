"""Opt-in, stdlib-only split-v2 (#24); never weakens the atomic-group cap.

Root identity plus cross-root near-duplicate edges define indivisible connected
components. Counts/Kish describe concentration, not source independence.
"""
from __future__ import annotations
import collections, hashlib, itertools, json, math, random, re
from pathlib import Path

VERSION='split-v2'
NAMES=('train','calibration','test')
RATIOS=(.7,.15,.15)
CLASSES={'tier':('fast','balanced','flagship'), 'needs_vision':('yes','no','uncertain'),
         'task_type':('extraction','research','implementation','debugging','review','visual','creative')}
WEIGHTS={'row_fraction':1.,'tier':.15,'vision_yes':1.,'task_type':.5}
MAX_GROUP_SHARE=.35  # Fixed candidate constraint, NOT a tunable score/CLI threshold.
LIMITS=[
    'Root sessions and near-duplicate connected components are indivisible, including transitive links across roots.',
    '5-gram Jaccard below the threshold does not prove semantic/context independence.',
    'Kish n^2/sum(group_size^2) is descriptive group-size concentration, not independent effective sample size or proof of independence.',
    'Subsessions/time windows do not authorize splitting roots; finer units require external provenance review and a new protocol.',
    'Ratios and class proportions are optimization targets, not guarantees; rare classes may not cover every split.',
    'Real re-splitting requires versioned labels (#18) and separate distribution/35% feasibility/version acceptance (#27).',
    'A changed test set is not directly comparable with historical frozen-v1 metrics; no historical output is replaced.',
]


def _valid_text(value):return isinstance(value,str) and bool(value.strip())


def _canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _validate(tasks,labels):
    from task_label_schema import check_row_versions
    check_row_versions(tasks)
    check_row_versions([{'label':label} for label in labels])
    if not tasks:raise ValueError('tasks must be nonempty')
    for row in tasks:
        if not isinstance(row,dict) or any(not _valid_text(row.get(k)) for k in ('id','group','prompt')):
            raise ValueError('each task requires nonempty id, group and prompt strings')
    for label in labels:
        if not isinstance(label,dict) or not _valid_text(label.get('id')):
            raise ValueError('each label requires a nonempty id')
        if any(label.get(k) not in values for k,values in CLASSES.items()):raise ValueError('invalid label class')
        confidence=label.get('confidence')
        if (type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1
                or not _valid_text(label.get('rationale'))):raise ValueError('invalid label confidence/rationale')
        if label.get('label_source')!='expert_provisional' or label.get('evidence')!='prompt_only':
            raise ValueError('split-v2 requires explicit expert_provisional / prompt_only labels')
        if any(k in label for k in ('model','provider','historical_model')):raise ValueError('historical model label metadata forbidden')
    ids=[r['id'] for r in tasks];label_ids=[r['id'] for r in labels]
    if len(set(ids))!=len(ids) or len(set(label_ids))!=len(label_ids):raise ValueError('duplicate task/label ids')
    if set(ids)!=set(label_ids):raise ValueError('label coverage differs from task ids')


def _components(tasks,by_id,similarity):
    parent={r['group']:r['group'] for r in tasks}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    texts=[re.sub(r'\s+','',r['prompt'].lower()) for r in tasks]
    grams=[{text[i:i+5] for i in range(max(0,len(text)-4))} for text in texts]
    edges=0
    for i,a in enumerate(tasks):
        for j in range(i):
            b=tasks[j]
            if a['group']==b['group']:continue
            x,y=grams[i],grams[j]
            # Exact normalized short prompts also link; two empty gram sets alone do not.
            same=texts[i]==texts[j]
            if not same and (not x or not y or min(len(x),len(y))<similarity*max(len(x),len(y))):continue
            if same or len(x&y)/len(x|y)>=similarity:
                edges+=1;u,v=find(a['group']),find(b['group'])
                if u!=v:parent[max(u,v)]=min(u,v)
    grouped=collections.defaultdict(list)
    for row in tasks:
        group=find(row['group'])
        grouped[group].append({**row,'label':by_id[row['id']],'split_group':group})
    return [grouped[k] for k in sorted(grouped)],edges,len(parent)


def _vector(rows,classes=CLASSES):
    return {'rows':len(rows), **{k:collections.Counter(r['label'][k] for r in rows) for k in classes}}


def _summaries(vectors,assignment,total,targets,classes=CLASSES):
    result={};score={k:0. for k in WEIGHTS};valid=True
    for s,name in enumerate(NAMES):
        chosen=[v for v,a in zip(vectors,assignment) if a==s]
        sizes=[v['rows'] for v in chosen];n=sum(sizes);largest=max(sizes,default=0)
        # Integer arithmetic keeps the .35 boundary exact. Empty splits fail closed.
        passes=bool(n) and 100*largest<=35*n
        valid=valid and passes
        counts={k:{c:sum(v[k][c] for v in chosen) for c in values} for k,values in classes.items()}
        props={k:{c:count/n if n else None for c,count in values.items()} for k,values in counts.items()}
        deviation={k:{c:None if not n else props[k][c]-targets[k][c] for c in values} for k,values in classes.items()}
        row_deviation=n/total-RATIOS[s]
        score['row_fraction']+=abs(row_deviation)
        for k in ('tier','task_type'):
            score[k]+=sum(abs(v) if v is not None else 1. for v in deviation[k].values())
        yes=deviation['needs_vision']['yes'];score['vision_yes']+=abs(yes) if yes is not None else 1.
        share=largest/n if n else None
        result[name]={'rows':n,'observed_components':len(chosen),'group_sizes':sorted(sizes,reverse=True),
                      'counts':counts,'proportions':props,'row_fraction':n/total,
                      'deviation_from_target':{'row_fraction':row_deviation,**deviation},
                      'max_group_share':share,'max_group_share_excess':max(0.,share-MAX_GROUP_SHARE) if n else None,
                      'constraint_satisfied':passes,
                      'kish_group_size_concentration':n*n/sum(v*v for v in sizes) if n else None}
    terms={k:score[k]*WEIGHTS[k] for k in WEIGHTS}
    return valid,sum(terms.values()),result,terms


def _candidates(count,seed,attempts):
    if count<=9:
        yield from itertools.product(range(3),repeat=count)
        return
    rng=random.Random(seed)
    # A balanced whole-component starting point helps cap feasibility, not a guarantee.
    order=list(range(count));rng.shuffle(order)
    initial=[0]*count
    for i,g in enumerate(order):initial[g]=i%3
    yield tuple(initial)
    for _ in range(attempts-1):
        yield tuple(rng.choices(range(3),weights=RATIOS,k=count))


def split_v2(tasks,labels,*,seed=20261010,similarity=.88,attempts=12000):
    """Return (splits or None, aggregate report); no I/O or model/dependency use.

    Exhaustive search for <=9 components; bounded seeded heuristic otherwise.
    Failure of bounded search is NOT a mathematical infeasibility certificate.
    """
    tasks=list(tasks);labels=list(labels)
    _validate(tasks,labels)
    if type(seed) is not int or type(attempts) is not int or attempts<1:raise ValueError('integer seed and positive attempts required')
    if type(similarity) not in (float,int) or not math.isfinite(similarity) or not 0<similarity<=1:
        raise ValueError('similarity must be finite in (0,1]')
    tasks=sorted(tasks,key=lambda r:r['id']);labels=sorted(labels,key=lambda r:r['id'])
    groups,edges,roots=_components(tasks,{r['id']:r for r in labels},similarity)
    return _assign(tasks,labels,groups,edges,roots,seed,similarity,attempts)


def _assign(tasks,labels,groups,edges,roots,seed,similarity,attempts,classes=CLASSES):
    """Shared atomic optimizer; callers own schema validation and exclusion."""
    vectors=[_vector(g,classes) for g in groups];whole=_vector([r for group in groups for r in group],classes);n=whole['rows']
    proportions={k:{c:whole[k][c]/n if n else 0. for c in values} for k,values in classes.items()}
    report={'schema_version':2,'split_version':VERSION,'seed':seed,
            'input_sha256':hashlib.sha256(_canonical({'tasks':tasks,'labels':labels}).encode()).hexdigest(),
            'grouping':{'observed_roots':roots,'observed_components':len(groups),'cross_root_near_duplicate_edges':edges,
                        'near_duplicate_threshold':similarity,'unit':'root + near-duplicate connected component'},
            'targets':{'row_fractions':dict(zip(NAMES,RATIOS)),'class_proportions':proportions,
                       'max_group_share':MAX_GROUP_SHARE,'nonempty_splits':True},
            'objective_weights':dict(WEIGHTS),'limitations':list(LIMITS),'class_component_support':{},'splits':{}}
    for k,values in classes.items():
        report['class_component_support'][k]={c:{'rows':whole[k][c], 'components':sum(v[k][c]>0 for v in vectors),
                                                'fewer_than_three_components':sum(v[k][c]>0 for v in vectors)<3} for c in values}
    search={'method':'exhaustive' if len(groups)<=9 else 'bounded_seeded_heuristic',
            'budget':3**len(groups) if len(groups)<=9 else attempts,'evaluated':0,'feasible_candidates':0}
    report['search']=search
    reason=None
    if len(groups)<9:reason='At least 9 components are necessary: each of 3 nonempty splits needs at least 3 at the fixed .35 cap.'
    if n and 100*max(v['rows'] for v in vectors)>35*n:
        reason='The largest indivisible component exceeds .35 of the ENTIRE corpus; it cannot fit in any split.'
    best=None;rejected=None
    if reason and n:
        # A diagnostic only, never exported as a split or presented as feasible.
        _,_,stats,_=_summaries(vectors,(0,)*len(groups),n,proportions,classes)
        report['diagnostic_candidate']={'selected':False,'splits':stats}
    elif not reason:
        for assignment in _candidates(len(groups),seed,attempts):
            search['evaluated']+=1
            sizes=[sum(v['rows'] for v,a in zip(vectors,assignment) if a==s) for s in range(3)]
            maxima=[max((v['rows'] for v,a in zip(vectors,assignment) if a==s),default=0) for s in range(3)]
            valid=all(size and 100*largest<=35*size for size,largest in zip(sizes,maxima))
            if not valid:
                # Retain one rejected witness with honest target/actual deviations.
                if rejected is None:rejected=assignment
                continue
            search['feasible_candidates']+=1
            _,score,stats,terms=_summaries(vectors,assignment,n,proportions,classes)
            if best is None or (score,assignment)<(best[0],best[1]):best=(score,assignment,stats,terms)
    if best is None:
        proven=reason is not None or search['method']=='exhaustive'
        report['status']='infeasible' if proven else 'search_exhausted'
        report['feasibility']={'proven_infeasible':proven,'reason':reason or (
            'All atomic assignments were checked; none meets the nonempty/.35 constraints.' if proven else
            'No feasible candidate found within the search budget; infeasibility is NOT proven. Increase search/review or collect new roots; never split roots or weaken the cap.')}
        if rejected is not None:
            _,_,stats,_=_summaries(vectors,rejected,n,proportions,classes)
            report['diagnostic_candidate']={'selected':False,'splits':stats}
        return None,report
    score,assignment,stats,terms=best
    splits={name:sorted([r for group,a in zip(groups,assignment) if a==s for r in group],key=lambda r:r['id']) for s,name in enumerate(NAMES)}
    report.update(status='feasible',feasibility={'proven_infeasible':False,'reason':'A candidate satisfies the unchanged atomic/nonempty/.35 constraints; objective optimality only if search was exhaustive.'},
                  splits=stats,objective={'total':score,'weighted_terms':terms},group_disjoint=True,root_disjoint=True)
    return splits,report


def prepare_v2(root: Path,output: Path,**options):
    """Read explicit expert inputs and write exclusively to a NEW bundle directory.

    Existing directories/files/symlinks are refused before reading inputs. Only a
    versioned diagnostic report is written when constraints cannot be satisfied.
    """
    root=Path(root);output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('split-v2 output already exists; choose a fresh bundle directory')
    def load(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    tasks=load(root/'tasks.jsonl');labels=[]
    for i in range(1,5):labels.extend(load(root/f'expert-recovery-{i}.jsonl'))
    splits,report=split_v2(tasks,labels,**options)
    # mkdir is the exclusive reservation, including against concurrent writers.
    # Never clean up/replace a preexisting directory, even if a previous run failed.
    output.mkdir()
    report['outputs']={'report':f'split-report.{VERSION}.json','splits':{},'labels':None}
    if splits is not None:
        for name,rows in splits.items():
            filename=f'{name}.{VERSION}.jsonl'
            with (output/filename).open('x',encoding='utf-8') as out:
                out.write(''.join(_canonical(r)+'\n' for r in rows))
            report['outputs']['splits'][name]=filename
        filename=f'expert-labels.{VERSION}.jsonl'
        with (output/filename).open('x',encoding='utf-8') as out:
            out.write(''.join(_canonical(r)+'\n' for r in sorted(labels,key=lambda r:r['id'])))
        report['outputs']['labels']=filename
    with (output/report['outputs']['report']).open('x',encoding='utf-8') as out:
        out.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report
