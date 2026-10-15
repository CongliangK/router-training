"""Opt-in research recipe v1 (#19/#5); stdlib import, no data/model side effects.

Legacy training deliberately does not use these helpers. Probability vectors are
in input-row order; reports contain no prompts or row/group identities.
"""
from __future__ import annotations
import collections, hashlib, json, math, random
from pathlib import Path

VERSION='research-training-v1'
SCHEMA='legacy7-v1'
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
    _keys(c,{'version','schema_version','data_version','evaluation_protocol','training','sampler','loss','capacity'},'research config')
    if c['version']!=VERSION or c['schema_version']!=SCHEMA:raise ValueError('unsupported research/schema version; #18 integration requires explicit review')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in ('data_version','evaluation_protocol')):raise ValueError('data/protocol version required')
    t=c['training'];_keys(t,TRAIN_KEYS,'training')
    for k in ('epochs','rank','accumulation'):
        if type(t[k]) is not int or t[k]<1:raise ValueError(f'{k} must be a positive integer')
    if type(t['seed']) is not int:raise ValueError('seed must be an integer')
    for k in ('synth','mlp','fullft','swa'):
        if type(t[k]) is not bool:raise ValueError(f'{k} must be boolean')
    if not _number(t['lr'],1e-12) or not _number(t['beta'],0.,1.) or not _number(t['label_smoothing'],0.,1.):raise ValueError('invalid numeric training setting')
    if t['optimizer'] not in ('adamw','adamw8bit'):raise ValueError('unknown optimizer')
    # No legacy augmentation loader on this path: synthetic rows must be explicit
    # in the versioned train pool and marked synthetic, never in calibration.
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


def check_task_schema(schema_version=None):
    """Map semantic v1 to the research recipe, never source provenance integers."""
    from task_label_schema import LEGACY, version
    if version(schema_version)!=LEGACY:
        raise ValueError('router-label-v2 research is unsupported; research v1 requires legacy7-v1 (router-label-v1)')
    return SCHEMA


def check_args(c,args):
    check_task_schema(getattr(args,'label_schema',None))
    if getattr(args,'legacy_calibration',False) or getattr(args,'smoke',False):raise ValueError('research config cannot combine with legacy calibration or smoke; separate authorization/protocol required')
    for key,value in c['training'].items():
        if getattr(args,key,None)!=value:raise ValueError(f'research config/CLI mismatch: {key}')


def check_questions(questions):
    expected={'tier':TIERS,'vision':VISION,'task_type':TYPES}
    if set(questions)!=set(expected) or any(tuple(questions[k]['criteria'])!=v for k,v in expected.items()):raise ValueError('research v1 requires explicit legacy7-v1 schema; do not silently mix #18')


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
    # #8 source contract integer != #18 task semantics string. Reject mixed/v2
    # rows and shadow partitions even if an expert source was copied onto them.
    from task_label_schema import LEGACY, validate_supervised_rows
    validate_supervised_rows(rows,LEGACY)
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
        if label.get('tier') not in TIERS or label.get('needs_vision') not in VISION or label.get('task_type') not in TYPES:raise ValueError('research v1 label/schema mismatch or unresolved label')
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
    return {'version':s['version'],'epoch_rows':s['epoch_rows'],'group_count':len(counts),
            'natural_row_probabilities':natural,'target_row_probabilities':probabilities,
            'row_order_hash':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'priors':priors,'prior_provenance':{'source':s['prior_source'],'smoothing_pseudocount':s['prior_smoothing'],
                'data_version':c['data_version'],'unique_nonsynthetic_expert_support':support,
                'expected_sampler_includes_synthetic':s['prior_source']=='expected-sampler'},
            'sampling':'with replacement; row weight n_group^-0.5 times explicit synthetic exposure; no tier balance and no loss exposure multiplier'}


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
