"""Fail-closed capacity candidates (#20). No imports of torch/PEFT at import time.

Toy tests do not prove Intern-Decision compatibility. Runtime checks deliberately
reject unknown revisions, module layouts/types/shapes and unsupported adapters.
"""
from __future__ import annotations
import inspect, math, re

PINNED_REVISION='85a0cc5a99d67ea8d56dfe98115689212867171d'
# Original issue says '11 classes' but lists 12 when k_proj is included. List
# explicitly rather than silently omitting k_proj; absent hybrid projections
# are allowed, but every selected actual module is enumerated and validated.
PROJECTIONS={'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj','gate_proj','up_proj','down_proj'}
LAYERS=re.compile(r'(?:^|\.)language_model\.layers\.(\d+)\.')
EXCLUDED={'visual','projector','lm_head','embed_tokens'}


def _language(name):
    return 'language_model' in name.split('.') and not EXCLUDED.intersection(name.split('.'))


def require_revision(model):
    config=getattr(model,'config',None)
    if getattr(config,'_commit_hash',None)!=PINNED_REVISION:
        raise ValueError('runtime pinned-backbone identity unverified; exact config._commit_hash required, no manifest-only compatibility claim')


def enumerate_dora_targets(model,rank=16):
    import torch
    require_revision(model)
    selected=[]
    for name,module in model.named_modules():
        if not name or not _language(name):continue
        leaf=name.rsplit('.',1)[-1]
        if leaf not in PROJECTIONS:
            if isinstance(module,torch.nn.Linear):raise ValueError('unclassified language linear projection; explicit support review required')
            continue
        # Quantized/custom linear implementations are not assumed compatible.
        if type(module) is not torch.nn.Linear:raise ValueError('DoRA target is not supported plain torch.nn.Linear')
        shape=tuple(module.weight.shape)
        if len(shape)!=2 or shape!=(module.out_features,module.in_features) or min(shape)<rank:raise ValueError('DoRA target shape/rank unsupported')
        selected.append({'name':name,'shape':list(shape),'type':'torch.nn.Linear'})
    if not selected:raise ValueError('no supported language projections; runtime layout unsupported')
    return selected


def dora_config(model,lora_config_cls):
    targets=enumerate_dora_targets(model)
    if 'use_dora' not in inspect.signature(lora_config_cls).parameters:raise ValueError('installed PEFT lacks explicit DoRA configuration support')
    config=lora_config_cls(r=16,lora_alpha=32,lora_dropout=.05,target_modules=[x['name'] for x in targets],bias='none',use_dora=True)
    if not getattr(config,'use_dora',False):raise ValueError('PEFT did not retain use_dora; no fallback permitted')
    return config,{'version':'dora-candidate-v1','targets':targets,'r':16,'alpha':32,'dropout':.05,
                   'runtime_status':'targets checked; injection verification still required','pinned_revision':PINNED_REVISION}


def verify_dora_injection(model,report,adapter='default'):
    modules=dict(model.named_modules())
    for target in report['targets']:
        matches=[m for n,m in modules.items() if n==target['name'] or n.endswith('.'+target['name'])]
        if len(matches)!=1:raise ValueError('DoRA injection target missing/ambiguous')
        m=matches[0];out_features,in_features=target['shape']
        try:
            a=m.lora_A[adapter].weight;b=m.lora_B[adapter].weight
            magnitude=m.lora_magnitude_vector[adapter].weight
            enabled=m.use_dora[adapter]
        except (AttributeError,KeyError,TypeError) as error:raise ValueError('DoRA injection unsupported; magnitude/A/B state absent') from error
        if not enabled or tuple(a.shape)!=(16,in_features) or tuple(b.shape)!=(out_features,16) or tuple(magnitude.shape)!=(out_features,):raise ValueError('DoRA injected state shape/support mismatch')
        if not all(p.requires_grad for p in (a,b,magnitude)):raise ValueError('DoRA injected parameters are not trainable')
    report['runtime_status']='enumeration and adapter injection checked; forward/backward/merge support NOT established'
    report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return report


def staged_parameters(model):
    require_revision(model)
    eligible=[];layers=set()
    for name,p in model.named_parameters():
        if not _language(name):continue
        match=LAYERS.search(name)
        if match:layers.add(int(match.group(1)))
        elif not any('norm' in part.lower() for part in name.split('.')):
            raise ValueError('unclassified language parameter outside layers/norm; staged control unsupported')
        eligible.append((name,p))
    if len(layers)<4 or sorted(layers)!=list(range(max(layers)+1)):raise ValueError('staged control requires contiguous known language layers, at least four')
    if not eligible:raise ValueError('no staged language parameters')
    return eligible,set(sorted(layers)[-4:])


def apply_stage(eligible,top_layers,completed_updates,total_updates):
    if type(total_updates) is not int or total_updates<1 or type(completed_updates) is not int or not 0<=completed_updates<=total_updates:raise ValueError('invalid optimizer update schedule')
    boundary=math.ceil(total_updates*.15)
    full=completed_updates>=boundary
    for name,p in eligible:
        match=LAYERS.search(name)
        p.requires_grad=full or bool(match and int(match.group(1)) in top_layers)
    return {'version':'top4-then-full-v1','phase':'full-language' if full else 'top4-only',
            'initial_updates':boundary,'achieved_initial_fraction':boundary/total_updates,
            'requested_initial_fraction':.15,'top_layers':sorted(top_layers),
            'not_strict_lp_ft':True,'completed_optimizer_updates':completed_updates}


def install_neftune(model,alpha=5.):
    import torch
    if type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<0:raise ValueError('invalid NEFTune alpha')
    embedding=model.get_input_embeddings()
    if type(embedding) is not torch.nn.Embedding:raise ValueError('NEFTune input embedding type unsupported')
    if getattr(embedding,'_research_neftune',False):raise ValueError('NEFTune hook already installed')
    def hook(module,inputs,output):
        if not module.training or alpha==0:return output
        if not torch.is_tensor(output) or output.ndim!=3 or not output.is_floating_point():raise ValueError('NEFTune requires [batch,sequence,hidden] floating embeddings')
        scale=alpha/math.sqrt(output.shape[-2]*output.shape[-1])
        # rand_like preserves device/dtype; no CUDA API is invoked by the hook.
        return output+torch.empty_like(output).uniform_(-scale,scale)
    handle=embedding.register_forward_hook(hook);embedding._research_neftune=True
    class Handle:
        def remove(self):
            handle.remove();embedding._research_neftune=False
    return Handle()
