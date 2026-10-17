"""Opt-in versioned research recipes (#18/#19/#20); stdlib, no model effects.

Legacy training deliberately does not use these helpers. Probability vectors are
in input-row order; reports contain no prompts or row/group identities.
"""
from __future__ import annotations
import collections, hashlib, json, math, random
from pathlib import Path

VERSION='research-training-v1'
SCHEMA='legacy7-v1'
V2_VERSION='research-training-v2'
V2_SCHEMA='task6-v2'
INPUT_FILES=('train.jsonl','calibration.jsonl','label-schema.json','split-report.json','roles.json')
SUPPLEMENTAL_ERROR='research-training-v2 refuses --supplemental; use reviewed expert-only primary inputs, or the separate non-research v2 supplemental path'
TIERS=('fast','balanced','flagship')
VISION=('no','yes')
TYPES=('extraction','research','implementation','debugging','review','visual','creative')
EXPERT_SOURCES={'expert_provisional','expert_reviewed'}
COST_VERSION='asymmetric-tier-v1'
# Rows = true tier, columns = predicted tier; relative regret, NOT money.
COST=((0.,1.,3.),(1.,0.,1.5),(5.,5.,0.))
TRAIN_KEYS={'epochs','rank','lr','accumulation','seed','beta','label_smoothing','synth','mlp','fullft','swa','optimizer'}


def _number(value,low=0.,high=math.inf):
    return type(value) in (int,float) and math.isfinite(value) and low<=value<=high


def _keys(value,expected,name):
    if not isinstance(value,dict) or set(value)!=set(expected):raise ValueError(f'{name}: exact versioned keys required')


def validate_config(c):
    is_v2=isinstance(c,dict) and c.get('version')==V2_VERSION
    _keys(c,{'version','schema_version','data_version','evaluation_protocol','training','sampler','loss','capacity'}|({'label_schema','input_sha256'} if is_v2 else set()),'research config')
    if (c['version'],c['schema_version']) not in ((VERSION,SCHEMA),(V2_VERSION,V2_SCHEMA)):
        raise ValueError('unsupported research/schema version; select an explicit matching versioned contract')
    if is_v2:
        from task_label_schema import V2, metadata
        if c['label_schema']!=metadata(V2):raise ValueError('research v2 question fingerprint/ordered classes mismatch; review and regenerate the versioned config')
        _keys(c['input_sha256'],INPUT_FILES,'research v2 input fingerprints')
        if any(not isinstance(v,str) or len(v)!=64 or any(x not in '0123456789abcdef' for x in v) for v in c['input_sha256'].values()):
            raise ValueError('research v2 input fingerprints require lowercase SHA-256 hex')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in ('data_version','evaluation_protocol')):raise ValueError('data/protocol version required')
    t=c['training'];_keys(t,TRAIN_KEYS,'training')
    for k in ('epochs','rank','accumulation'):
        if type(t[k]) is not int or t[k]<1:raise ValueError(f'{k} must be a positive integer')
    if type(t['seed']) is not int:raise ValueError('seed must be an integer')
    for k in ('synth','mlp','fullft','swa'):
        if type(t[k]) is not bool:raise ValueError(f'{k} must be boolean')
    if not _number(t['lr'],1e-12) or not _number(t['beta'],0.,1.) or not _number(t['label_smoothing'],0.,1.):raise ValueError('invalid numeric training setting')
    if t['optimizer'] not in ('adamw','adamw8bit'):raise ValueError('unknown optimizer')
    # No legacy augmentation loader: v1 synthetic rows must be explicit in the
    # train pool; v2 is primary-expert-only and refuses all supplements.
    if t['synth'] or t['swa'] or t['mlp']:raise ValueError('research v1 forbids legacy synth/SWA/MLP flags')
    s=c['sampler'];_keys(s,{'version','epoch_rows','synthetic_balanced_vision_multiplier','prior_source','prior_smoothing'},'sampler')
    if s['version']!='sqrt-group-v1' or type(s['epoch_rows']) is not int or s['epoch_rows']<1:raise ValueError('invalid sampler version/budget')
    if not _number(s['synthetic_balanced_vision_multiplier'],1e-12) or not _number(s['prior_smoothing'],0.):raise ValueError('invalid sampler multiplier/smoothing')
    if s['prior_source'] not in ('unique-natural-expert','expected-sampler'):raise ValueError('unknown prior source')
    l=c['loss'];_keys(l,{'version','tier_tau','vision_tau','cost_version','cost_matrix','cost_lambda','confidence_weighting','head_version','auxiliary'},'loss')
    if l['version']!='additive-la-v1' or l['head_version']!='heads-v1' or l['cost_version']!=COST_VERSION or l['cost_matrix']!=[list(r) for r in COST]:raise ValueError('unknown loss/head/cost version or matrix')
    if any(not _number(l[k]) for k in ('tier_tau','vision_tau','cost_lambda')) or type(l['confidence_weighting']) is not bool:raise ValueError('invalid loss settings')
    if l['auxiliary'] not in ('candidate','task-type-zero','tier-only','current-03'):raise ValueError('unsupported auxiliary mode; independent-head/two-stage is protocol-only')
    cap=c['capacity'];_keys(cap,{'mode','neftune_alpha','interpolation'},'capacity')
    if cap['mode'] not in ('lora','dora','staged8bit') or not _number(cap['neftune_alpha']):raise ValueError('invalid capacity mode/noise')
    if cap['interpolation']!='unavailable':raise ValueError('post-weight interpolation is unavailable; SWA/logit mixing are not substitutes')
    if cap['mode']=='staged8bit':
        if not t['fullft'] or t['optimizer']!='adamw8bit' or t['epochs']!=2:raise ValueError('staged8bit control requires fullft, adamw8bit, 2 epochs')
    elif t['fullft']:raise ValueError('adapter candidate must not enable fullft')
    if cap['mode']=='dora' and t['rank']!=16:raise ValueError('DoRA candidate requires rank 16 (alpha 32/dropout .05)')
    return c


def load_config(path):return validate_config(json.loads(Path(path).read_text()))


def config_hash(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_task_schema(schema_version=None,c=None):
    """Map semantic v1 to the research recipe, never source provenance integers."""
    from task_label_schema import LEGACY, V2, version
    if c is not None:validate_config(c)
    expected=V2 if c is not None and c['version']==V2_VERSION else LEGACY
    if version(schema_version)!=expected:
        raise ValueError('unsupported research/task schema combination; research-training-v1 requires router-label-v1, research-training-v2 requires explicit router-label-v2')
    return V2_SCHEMA if expected==V2 else SCHEMA


def check_args(c,args):
    check_task_schema(getattr(args,'label_schema',None),c)
    expected=getattr(args,'research_config_sha256',None)
    if expected is not None and expected!=config_hash(c):raise ValueError('research config fingerprint changed since launcher preflight; review and repeat preflight; no config was changed')
    if getattr(args,'legacy_calibration',False) or getattr(args,'smoke',False):raise ValueError('research config cannot combine with legacy calibration or smoke; separate authorization/protocol required')
    check_supplemental(c,getattr(args,'supplemental',None))
    for key,value in c['training'].items():
        if getattr(args,key,None)!=value:raise ValueError(f'research config/CLI mismatch: {key}')


def check_supplemental(c,paths):
    if c['version']==V2_VERSION and paths:raise ValueError(SUPPLEMENTAL_ERROR)


def check_questions(questions,c=None):
    from task_label_schema import V2, SIX_TYPES, questions_for
    is_v2=c is not None and c['version']==V2_VERSION
    expected={'tier':TIERS,'vision':VISION,'task_type':SIX_TYPES if is_v2 else TYPES}
    if set(questions)!=set(expected) or any(tuple(questions[k]['criteria'])!=v for k,v in expected.items()):raise ValueError('research question order/classes differ from the selected versioned schema')
    if is_v2:
        validate_config(c)
        if tuple(questions)!=tuple(expected) or questions!=questions_for(V2):raise ValueError('research v2 question identity mismatch; review and regenerate the versioned config')


def validate_v2_experts(rows):
    """Keep task, source and partition provenance independent and immutable."""
    from task_label_schema import V2, validate_supervised_rows
    from label_schema import finite_json
    finite_json(rows,context='research v2 inputs');validate_supervised_rows(rows,V2)
    for row in rows:
        if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('id','prompt','group','split_group')):
            raise ValueError('research v2 primary rows require id, prompt, group and split_group (values withheld)')
        label=row['label']
        if label.get('label_source') not in EXPERT_SOURCES or label.get('evidence')!='prompt_only':
            raise ValueError('research v2 requires expert_provisional/expert_reviewed prompt_only provenance')
        if not _number(label.get('confidence'),0.,1.):raise ValueError('expert confidence must be finite in [0,1]')
        for obj in (row,label):
            if obj.get('partition')=='measured_shadow' or any(k in obj for k in ('model','provider','historical_model','route_fingerprint','success_lb','success_rate')):
                raise ValueError('research v2 refuses measured shadow or outcome provenance')
            if 'label_source' in obj and obj['label_source']!=label['label_source']:
                raise ValueError('research v2 has conflicting expert source provenance')
            if 'label_status' in obj and obj['label_status']!='resolved':
                raise ValueError('research v2 refuses unresolved supervision')


def load_v2_inputs(c,data_dir,roles_path,questions,supplemental=None):
    """Shared launcher/runtime byte snapshot; hashes bind exactly what is used.

    Only train/calibration and their declarations are opened. Never test,
    supplemental files, models or outputs. Config drift is never auto-repaired.
    """
    from task_label_schema import V2, validate_dataset_documents
    from calibration_groups import validate_partition_isolation, preflight_roles, schema_from_questions, compute_prior, anonymous_summary
    validate_config(c)
    if c['version']!=V2_VERSION:raise ValueError('v2 input loader requires research-training-v2')
    check_task_schema(V2,c);check_questions(questions,c);check_supplemental(c,supplemental)
    paths={name:Path(data_dir)/name for name in INPUT_FILES};paths['roles.json']=Path(roles_path)
    try:payloads={name:path.read_bytes() for name,path in paths.items()}
    except OSError:raise ValueError('research v2 input cannot be read (path withheld)') from None
    fingerprints={name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()}
    if fingerprints!=c['input_sha256']:
        raise ValueError('research v2 input fingerprint mismatch; independently review changed inputs, then regenerate input_sha256 in a new versioned config; no config was changed')
    try:
        datasets={name:[json.loads(line) for line in payloads[name+'.jsonl'].decode('utf-8').splitlines() if line.strip()] for name in ('train','calibration')}
        documents={name:json.loads(payloads[name].decode('utf-8')) for name in ('label-schema.json','split-report.json')}
        roles=json.loads(payloads['roles.json'].decode('utf-8'))
    except (ValueError,UnicodeError):raise ValueError('research v2 inputs must be readable UTF-8 JSON (values withheld)') from None
    validate_dataset_documents(documents,V2)
    isolation=validate_partition_isolation(datasets['train'],datasets['calibration'])
    for rows in datasets.values():validate_v2_experts(rows)
    plan=preflight_roles(datasets['calibration'],roles,schema=schema_from_questions(questions,V2),schema_version=V2)
    prior=compute_prior(datasets['train'],'tier',plan['schema'])
    streams,sampling=sample_epochs(datasets['train'],c,c['training']['seed'],c['training']['epochs'])
    sampling['input_sha256']=fingerprints
    summary={**anonymous_summary(plan),'root_isolation':isolation,'research':{
        'config_sha256':config_hash(c),'label_schema':c['label_schema'],'sampling':sampling,
        'qualification':'structural only; protocol unapproved, backbone/training/production unverified'}}
    return {'datasets':datasets,'roles':roles,'plan':plan,'prior':prior,'streams':streams,
            'sampling':sampling,'isolation':isolation,'summary':summary,'split_report':documents['split-report.json']}


def head_weights(c):
    return {'candidate':{'tier':1.,'vision':.5,'task_type':.15},
            'task-type-zero':{'tier':1.,'vision':.5,'task_type':0.},
            'tier-only':{'tier':1.,'vision':0.,'task_type':0.},
            'current-03':{'tier':1.,'vision':.5,'task_type':.3}}[c['loss']['auxiliary']]


def confidence_weight(row,enabled=True):
    label=row['label']
    # Measured confidence is an interval coverage level, NOT expert certainty.
    if not enabled or label.get('label_source') not in EXPERT_SOURCES:return 1.
    confidence=label.get('confidence')
    if not _number(confidence,0.,1.):raise ValueError('expert confidence must be finite in [0,1]')
    return max(.5,confidence)**2


def sampler_plan(rows,c):
    validate_config(c)
    # #8 source contract integer != #18 task semantics string. Reject mixed
    # rows and shadow partitions even if an expert source was copied onto them.
    from task_label_schema import LEGACY, SIX_TYPES, validate_supervised_rows
    is_v2=c['version']==V2_VERSION;types=SIX_TYPES if is_v2 else TYPES
    if is_v2:validate_v2_experts(rows)
    else:validate_supervised_rows(rows,LEGACY)
    if not rows:raise ValueError('empty train pool')
    ids=[r.get('id') for r in rows]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('unique nonempty train ids required')
    groups=[]
    for r in rows:
        g=r.get('split_group')
        if not isinstance(g,str) or not g:raise ValueError('explicit split_group required; no row-id independence fallback')
        groups.append(g)
        if type(r.get('synthetic',False)) is not bool:raise ValueError('synthetic flag must be boolean')
        label=r['label']
        if label.get('label_source') not in EXPERT_SOURCES:raise ValueError('research v1 is expert-only; measured/untyped rows need #8/#18 integration')
        if label.get('tier') not in TIERS or label.get('needs_vision') not in VISION or label.get('task_type') not in types:raise ValueError('research label/schema mismatch or unresolved label')
        confidence_weight(r,c['loss']['confidence_weighting'])
    counts=collections.Counter(groups);s=c['sampler']
    natural=[1/len(rows)]*len(rows)
    weights=[counts[g]**-.5*(s['synthetic_balanced_vision_multiplier'] if r.get('synthetic',False) and r['label']['tier']=='balanced' and r['label']['needs_vision']=='yes' else 1.) for r,g in zip(rows,groups)]
    total=sum(weights)
    if not math.isfinite(total) or total<=0:raise ValueError('sampling weight total must be finite and positive')
    probabilities=[w/total for w in weights]
    if any(not math.isfinite(p) or p<=0 for p in probabilities):raise ValueError('sampling probabilities must be finite and positive')
    priors={};support={}
    for field,key,classes in [('tier','tier',TIERS),('vision','needs_vision',VISION)]:
        eligible=[i for i,r in enumerate(rows) if not r.get('synthetic',False)]
        observed=collections.Counter(rows[i]['label'][key] for i in eligible)
        if set(observed)!=set(classes):raise ValueError('prior requires unique non-synthetic expert support for every tier/vision class')
        support[field]=dict(observed)
        if s['prior_source']=='unique-natural-expert':mass={v:float(observed[v]) for v in classes}
        else:
            # Expected distribution includes explicit synthetic exposure. Scale to
            # N rows before applying pseudocounts, so smoothing has count units.
            mass={v:len(rows)*sum(p for p,r in zip(probabilities,rows) if r['label'][key]==v) for v in classes}
        a=s['prior_smoothing'];den=sum(mass.values())+a*len(classes)
        if not math.isfinite(den) or den<=0:raise ValueError('prior normalizer must be finite and positive')
        priors[field]={v:(mass[v]+a)/den for v in classes}
    plan={'version':s['version'],'epoch_rows':s['epoch_rows'],'group_count':len(counts),
            'natural_row_probabilities':natural,'target_row_probabilities':probabilities,
            'row_order_hash':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'priors':priors,'prior_provenance':{'source':s['prior_source'],'smoothing_pseudocount':s['prior_smoothing'],
                'data_version':c['data_version'],'unique_nonsynthetic_expert_support':support,
                'expected_sampler_includes_synthetic':s['prior_source']=='expected-sampler'},
            'sampling':'with replacement; row weight n_group^-0.5 times explicit synthetic exposure; no tier balance and no loss exposure multiplier'}
    if is_v2:
        plan['research_version']=V2_VERSION;plan['label_schema']=c['label_schema']
        plan['row_provenance']={'source_counts':dict(collections.Counter(r['label']['label_source'] for r in rows)),
            'primary_only':True,'resolved_only':True,'evidence':'prompt_only',
            'ordered_rows_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
    return plan


def sample_epochs(rows,c,seed,epochs):
    plan=sampler_plan(rows,c);rng=random.Random(seed);streams=[];actual=[]
    for _ in range(epochs):
        indices=rng.choices(range(len(rows)),weights=plan['target_row_probabilities'],k=plan['epoch_rows'])
        counts=collections.Counter(indices)
        streams.append([rows[i] for i in indices]);actual.append([counts[i]/len(indices) for i in range(len(rows))])
    plan['actual_epoch_row_frequencies']=actual
    return streams,plan


def additive_bias(prior,labels,tau):
    values=[prior[k] for k in labels]
    if not _number(tau) or any(not _number(p,1e-300,1.) for p in values) or not math.isclose(sum(values),1.,abs_tol=1e-9):raise ValueError('invalid logit adjustment prior/tau')
    return [tau*math.log(p) for p in values]


def expected_cost(probabilities,true_index):
    if type(true_index) is not int or not 0<=true_index<3 or len(probabilities)!=3 or any(not _number(p,0.,1.) for p in probabilities) or not math.isclose(sum(probabilities),1.,abs_tol=1e-9):raise ValueError('invalid cost probabilities/target')
    return sum(p*v for p,v in zip(probabilities,COST[true_index]))


def row_loss(logits,row,questions,c,plan,label_smoothing=0.):
    """Differentiable CE(z + tau log pi) + lambda E_softmax(z)[C[y,:]].

    Plus sign is TRAINING only; no inference adjustment is added here. Cost
    uses raw discriminant probabilities, not the training-biased probabilities.
    """
    import torch
    import torch.nn.functional as F
    loss=logits['tier'].sum()*0
    for field,coefficient in head_weights(c).items():
        if coefficient==0:continue
        labels=list(questions[field]['criteria']);key='needs_vision' if field=='vision' else field
        index=labels.index(row['label'][key]);values=logits[field]
        adjusted=values
        if field in ('tier','vision'):
            bias=additive_bias(plan['priors'][field],labels,c['loss'][field+'_tau'])
            adjusted=values+values.new_tensor(bias)
        term=F.cross_entropy(adjusted.unsqueeze(0),torch.tensor([index],device=values.device),label_smoothing=label_smoothing)
        if field=='tier':term=term+c['loss']['cost_lambda']*(values.softmax(-1)*values.new_tensor(COST[index])).sum()
        loss=loss+coefficient*term
    return loss*confidence_weight(row,c['loss']['confidence_weighting'])


def parameter_groups(named_parameters,eligible_ids=None):
    """Include stage-frozen eligible parameters from optimizer construction onward.

    AdamW skips grad=None parameters: no state/decay until unfreeze. Whitelist
    1D, bias, norm, A_log and dt_bias. No duplicated tied tensors.
    """
    groups=[{'params':[],'weight_decay':.01},{'params':[],'weight_decay':0.}];names=[[],[]];seen=set()
    for name,p in named_parameters:
        if id(p) in seen:continue
        seen.add(id(p))
        if not (id(p) in eligible_ids if eligible_ids is not None else p.requires_grad):continue
        leaf=name.rsplit('.',1)[-1];lower=name.lower()
        no_decay=p.ndim==1 or leaf.endswith('bias') or 'norm' in lower or leaf in ('A_log','dt_bias')
        index=int(no_decay);groups[index]['params'].append(p);names[index].append(name)
    if not any(g['params'] for g in groups):raise ValueError('no optimizer parameters')
    return groups,{'version':'no-decay-whitelist-v1','decay_names':names[0],'no_decay_names':names[1],
                   'includes_initially_frozen':eligible_ids is not None}
