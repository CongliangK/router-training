"""Actual LoRA training, held-out temperature calibration, and expert-imitation evaluation."""
from __future__ import annotations
import argparse, collections, json, math, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import LoraConfig,get_peft_model
from model_utils import ROOT,TIERS,TYPES,QUESTIONS,load_engine,read_jsonl,request_for,field_logits,score


def predict_rows(engine,rows):
    output=[]
    for i,row in enumerate(rows):
        start=time.perf_counter(); prediction=score(engine,row['prompt'])
        output.append({'id':row['id'],'label':row['label'],**prediction,'latency_ms':(time.perf_counter()-start)*1000})
        if (i+1)%25==0:print('scored',i+1,'/',len(rows),flush=True)
    return output


def target(row,field):
    return row['label'][{'tier':'tier','vision':'needs_vision','task_type':'task_type'}[field]]


def fit_temperatures(rows):
    fitted={}; diagnostics={}
    for field in QUESTIONS:
        labels=list(QUESTIONS[field]['criteria']);valid=[r for r in rows if target(r,field) in labels]
        supported=set(target(r,field) for r in valid)==set(labels)
        if not supported:
            fitted[field]=2.747760550703;diagnostics[field]='missing classes; retained checkpoint temperature (not locally calibrated)';continue
        options=[.5,.75,1,1.5,2,2.75,3.5,5,7,10]
        def nll(t):return sum(-math.log(max(1e-12,torch.softmax(torch.tensor(r['answers'][field]['logits'])/t,0)[labels.index(target(r,field))].item())) for r in valid)/len(valid)
        fitted[field]=min(options,key=nll);diagnostics[field]={'nll':nll(fitted[field]),'rows':len(valid)}
    return fitted,diagnostics


def calibrated(rows,temperatures,adjustments=None):
    rows=json.loads(json.dumps(rows))
    for row in rows:
        for field,answer in row['answers'].items():
            labels=list(QUESTIONS[field]['criteria'])
            values=torch.tensor(answer['logits'])
            if adjustments and field in adjustments:
                for index,label in enumerate(labels):
                    if label in adjustments[field]:values[index]-=adjustments[field][label]
            probs=torch.softmax(values/temperatures[field],0).tolist();best=max(range(len(labels)),key=lambda i:probs[i])
            answer.update(choice=labels[best],confidence=probs[best],probabilities=dict(zip(labels,probs)))
    return rows


def per_class(valid,field):
    """TP/FP/FN/TN, precision, recall, F1 per class for one field."""
    labels=list(QUESTIONS[field]['criteria']);n=len(valid);out={}
    for c in labels:
        tp=sum(1 for r in valid if r['answers'][field]['choice']==c and target(r,field)==c)
        fp=sum(1 for r in valid if r['answers'][field]['choice']==c and target(r,field)!=c)
        fn=sum(1 for r in valid if r['answers'][field]['choice']!=c and target(r,field)==c)
        prec=tp/max(1,tp+fp);rec=tp/max(1,tp+fn)
        out[c]={'tp':tp,'fp':fp,'fn':fn,'tn':n-tp-fp-fn,'precision':prec,'recall':rec,'f1':2*prec*rec/max(1e-9,prec+rec)}
    out['macro_f1']=sum(out[c]['f1'] for c in labels)/len(labels)
    return out


def bootstrap_ci(valid,field,statistic,samples=1000,seed=0):
    """95% bootstrap CI for a statistic(valid). Resamples independent split_group
    clusters, not rows: near-duplicate tasks inside one group are correlated and
    row-level resampling would understate the interval (red-team finding)."""
    import collections as _c
    groups=_c.defaultdict(list)
    for r in valid:groups[r.get('split_group',r['id'])].append(r)
    buckets=list(groups.values());rng=random.Random(seed);values=[]
    for _ in range(samples):
        draw=[row for bucket in (rng.choice(buckets) for _ in buckets) for row in bucket];values.append(statistic(draw))
    values.sort()
    return {'low':values[int(.025*len(values))],'high':values[min(len(values)-1,int(.975*len(values)))]}


def metrics(rows):
    result={'rows':len(rows),'context_truncated':sum(r['context_truncated'] for r in rows),'mean_latency_ms':sum(r['latency_ms'] for r in rows)/max(1,len(rows))}
    for field in QUESTIONS:
        valid=[r for r in rows if target(r,field) in QUESTIONS[field]['criteria']];labels=list(QUESTIONS[field]['criteria'])
        matrix={label:dict.fromkeys(labels,0) for label in labels};nll=brier=0
        for row in valid:
            expected=target(row,field);answer=row['answers'][field];matrix[expected][answer['choice']]+=1
            nll-=math.log(max(1e-12,answer['probabilities'][expected]))
            brier+=sum((p-(label==expected))**2 for label,p in answer['probabilities'].items())
        correct=sum(matrix[l][l] for l in labels);ece=0
        for lo in [0,.2,.4,.6,.8]:
            bucket=[r for r in valid if lo<=r['answers'][field]['confidence']<(lo+.2+1e-8)]
            if bucket:ece+=len(bucket)/len(valid)*abs(sum(r['answers'][field]['confidence'] for r in bucket)/len(bucket)-sum(r['answers'][field]['choice']==target(r,field) for r in bucket)/len(bucket))
        result[field]={'evaluated_rows':len(valid),'accuracy':correct/max(1,len(valid)),'nll':nll/max(1,len(valid)),'brier':brier/max(1,len(valid)),'ece_5_bins':ece,'confusion':matrix}
    flagship=[r for r in rows if r['label']['tier']=='flagship'];vision=[r for r in rows if r['label']['needs_vision']=='yes']
    result['flagship_underroute_rate']=sum(r['answers']['tier']['choice']!='flagship' for r in flagship)/len(flagship) if flagship else None
    result['vision_false_negative_rate']=sum(r['answers']['vision']['choice']!='yes' for r in vision)/len(vision) if vision else None
    result['flagship_examples']=len(flagship);result['vision_positive_examples']=len(vision)
    result['tier_per_class']=per_class([r for r in rows if target(r,'tier') in QUESTIONS['tier']['criteria']],'tier')
    result['vision_per_class']=per_class([r for r in rows if target(r,'vision') in QUESTIONS['vision']['criteria']],'vision')
    tier_valid=[r for r in rows if target(r,'tier') in QUESTIONS['tier']['criteria']]
    result['bootstrap']={
        'tier_accuracy':bootstrap_ci(tier_valid,'tier',lambda rs:sum(r['answers']['tier']['choice']==target(r,'tier') for r in rs)/max(1,len(rs))),
        'flagship_recall':bootstrap_ci(flagship,'tier',lambda rs:sum(r['answers']['tier']['choice']=='flagship' for r in rs)/max(1,len(rs))) if flagship else None,
        'flagship_precision':bootstrap_ci(tier_valid,'tier',lambda rs:(lambda tp,fp:tp/max(1,tp+fp))(sum(r['answers']['tier']['choice']=='flagship' and target(r,'tier')=='flagship' for r in rs),sum(r['answers']['tier']['choice']=='flagship' and target(r,'tier')!='flagship' for r in rs))),
        'vision_recall':bootstrap_ci(vision,'vision',lambda rs:sum(r['answers']['vision']['choice']=='yes' for r in rs)/max(1,len(rs))) if vision else None,
    }
    return result


def write_predictions(path,rows):path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))


def run(args):
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    output=ROOT/'artifacts'/args.run;output.mkdir(parents=True,exist_ok=True)
    datasets={name:read_jsonl(ROOT/'data'/f'{name}.jsonl') for name in ['train','calibration','test']}
    # Synthetic augmentation (vision-boundary focus) joins train/calibration only;
    # the test split stays real-history-only so reported metrics stay honest.
    if args.synth:
        synth_rows=[]
        for path in sorted((ROOT/'data'/'synth').glob('*.jsonl')):
            synth_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        import hashlib as _hashlib
        for row in synth_rows:
            row['id']=_hashlib.sha256(row['prompt'].encode()).hexdigest()[:20]
            row.setdefault('split_group',row['group'])
        # Red-team fix: synthetic rows join TRAIN ONLY. The calibration split must
        # stay real-history-only so temperatures and the tau operating point are
        # not fitted on synthetic lexical shortcuts.
        datasets['train'].extend(synth_rows)
        print('synthetic augmentation:',len(synth_rows),'rows -> train only',flush=True)
    # Red-team fix (calibration reuse): split the real calibration rows into a
    # selection half (early-stop macro-F1) and a fitting half (temperature/tau).
    # Fitting artifacts and epoch selection no longer share rows.
    rng=random.Random(args.seed);calib=datasets['calibration'][:] ;rng.shuffle(calib)
    half=max(1,len(calib)//2)
    selection_rows=calib[:half];fitting_rows=calib[half:] or calib
    print('calibration split: selection',len(selection_rows),'fitting',len(fitting_rows),flush=True)
    if args.smoke:
        datasets['train']=datasets['train'][:24];selection_rows=selection_rows[:6];fitting_rows=fitting_rows[:6];args.epochs=1
        print('SMOKE MODE: 24 train rows, 6+6 calibration rows, 1 epoch',flush=True)
    engine=load_engine();baseline={}
    for name in ['calibration','test']:
        print('baseline',name,flush=True);baseline[name]=predict_rows(engine,datasets[name]);write_predictions(output/f'baseline-{name}.jsonl',baseline[name])
    baseline_temps,baseline_temp_notes=fit_temperatures(baseline['calibration'])
    baseline_metrics=metrics(calibrated(baseline['test'],baseline_temps))
    print('baseline metrics',json.dumps(baseline_metrics),flush=True)
    targets=['q_proj','v_proj','in_proj_qkv','in_proj_z']
    if args.mlp:targets=targets+['o_proj','out_proj','in_proj_a','in_proj_b','gate_proj','up_proj','down_proj']
    lora=LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=.05,target_modules=targets,bias='none')
    if not args.fullft:
        engine.backend.model=get_peft_model(engine.backend.model,lora)
    else:
        # Deepest modification: unfreeze the language backbone entirely, keep the
        # vision tower and projector frozen (the official Intern-Decision recipe).
        # embed_tokens (tied lm_head) is ~1/3 of all params and cannot be learned
        # from 669 rows; freezing it saves ~3GB of fp32 grads+optimizer state and
        # keeps the 16GB fullft budget viable. Vision tower/projector stay frozen
        # per the official recipe.
        frozen={'visual','projector','lm_head','embed_tokens'}
        for name,param in engine.backend.model.named_parameters():
            param.requires_grad=not any(part in name for part in frozen)
        # P0 numeric fix: pure-bf16 full fine-tuning underflows (AdamW updates at
        # lr 2e-6 are smaller than bf16 quantization steps for |w|~0.02 weights,
        # so most updates round to zero and full FT looks spuriously ineffective).
        # Trainable params live in fp32 as master weights; the training forward
        # runs under bf16 autocast (see the loss step). Frozen vision stays bf16.
        for param in engine.backend.model.parameters():
            if param.requires_grad:param.data=param.data.float()
        # Custom linear-attention kernels bypass autocast and crash on mixed
        # dtypes (bf16 vision output x fp32 language weights), so promote the
        # WHOLE model to fp32; standard matmuls still run bf16 under autocast.
        for param in engine.backend.model.parameters():
            if not param.requires_grad:param.data=param.data.float()
        print('full fine-tune: trainable tensors',sum(1 for p in engine.backend.model.parameters() if p.requires_grad),'(uniform fp32 model)',flush=True)
    model=engine.backend.model;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Official XTuner recipe (configs/training/qwen35.py): lr 2e-6, cosine decay
    # to 2e-7, warmup 3%, wd 0.01. Our fullft path follows it; LoRA paths keep lr 1e-4.
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=(2e-6 if args.fullft else args.lr),weight_decay=.01,foreach=False)
    scheduler=None
    if args.fullft:
        import math as _m
        total_steps=max(1,args.epochs*(len(datasets['train'])//args.accumulation))
        warmup=max(1,int(total_steps*.03))
        def _lr_lambda(step):
            if step<warmup:return step/warmup
            progress=(step-warmup)/max(1,total_steps-warmup)
            return max(1e-3,(1+_m.cos(_m.pi*progress))/2)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,_lr_lambda)
    # Effective-number class weights (Cui et al. 2019, beta=0.999) replace the
    # earlier sqrt inverse-frequency: the soft sqrt version under-corrected the
    # 76%-majority balanced class and left the vision positive class unweighted.
    tier_counts=collections.Counter(r['label']['tier'] for r in datasets['train'])
    vision_counts=collections.Counter(r['label']['needs_vision'] for r in datasets['train'])
    beta=args.beta
    # Class-aware round-robin sampling already equalizes tier exposure per epoch,
    # so stacking effective-number tier weights on top would double-correct
    # (red-team finding). Tier loss weights stay 1.0; vision keeps effective-number
    # weights because sampling does not balance the vision field.
    weights={tier:1.0 for tier in TIERS}
    vision_weights={choice:(1-beta)/(1-beta**max(vision_counts.get(choice,0),1)) for choice in QUESTIONS['vision']['criteria']}
    vscale=2/sum(vision_weights.values());vision_weights={c:round(w*vscale,4) for c,w in vision_weights.items()}
    print('tier weights',weights,'vision weights',vision_weights,flush=True)
    history=[];best_score=-1;start=time.perf_counter();updates=0;swa_pool=[]
    for epoch in range(args.epochs):
        # Class-balanced OVERSAMPLING: repeat minority-tier rows to equal per-class
        # counts, then fully shuffle. The earlier round-robin drained minority
        # buckets first and left a pure-majority tail whose recency dominated the
        # final gradient steps of every epoch (the v3/v3b balanced-collapse cause).
        buckets=collections.defaultdict(list)
        for row in datasets['train']:buckets[row['label']['tier']].append(row)
        quota=max(len(b) for b in buckets.values())
        rows=[]
        for tier,bucket in buckets.items():
            random.shuffle(bucket)
            rows.extend(bucket*(quota//len(bucket))+random.sample(bucket,quota%len(bucket)))
        random.shuffle(rows)
        model.train();optimizer.zero_grad();running=0
        for i,row in enumerate(rows):
            request,truncated=request_for(engine,row['prompt'])
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.fullft):
                logits,_=field_logits(engine,request,train=True)
            loss=0
            for field,coefficient in [('tier',1.),('vision',.7),('task_type',.3)]:
                expected=target(row,field);labels=list(QUESTIONS[field]['criteria'])
                if expected not in labels:continue
                field_loss=F.cross_entropy(logits[field].unsqueeze(0),torch.tensor([labels.index(expected)],device=logits[field].device),label_smoothing=args.label_smoothing)
                if field=='tier':field_loss=field_loss*weights[expected]
                if field=='vision':field_loss=field_loss*vision_weights[expected]
                loss=loss+coefficient*field_loss
            (loss/args.accumulation).backward();running+=loss.detach().item()
            if (i+1)%args.accumulation==0 or i+1==len(rows):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                optimizer.step();optimizer.zero_grad();updates+=1
                if scheduler is not None:scheduler.step()
            if (i+1)%25==0:print('train epoch',epoch+1,'sample',i+1,'/',len(rows),'loss',round(running/(i+1),4),flush=True)
        model.eval();validation=predict_rows(engine,selection_rows);validation_metrics=metrics(validation)
        # Epoch selection now maximizes tier macro-F1 (imbalance-aware) instead of
        # NLL, which favored the majority class and locked in late conservative epochs.
        score=validation_metrics['tier_per_class']['macro_f1']
        history.append({'epoch':epoch+1,'train_loss':running/len(rows),'calibration_macro_f1':score,'calibration_tier_nll':validation_metrics['tier']['nll'],'calibration_vision_recall':None if validation_metrics['vision_false_negative_rate'] is None else round(1-validation_metrics['vision_false_negative_rate'],4)})
        if score>best_score:
            best_score=score
            if args.fullft:
                engine.backend.model.save_pretrained(output/'full-checkpoint',safe_serialization=True)
                # HFBackend loads the tokenizer from the checkpoint dir: ship the
                # base tokenizer artifacts alongside the fine-tuned weights.
                import shutil as _shutil
                for _f in (ROOT/'models/intern-decision-0.8b').iterdir():
                    if _f.is_file() and _f.suffix in {'.json','.txt','.jinja','.model'} and _f.name!='config.json':
                        _shutil.copy2(_f,output/'full-checkpoint'/_f.name)
            else:model.save_pretrained(output/'adapter')
            (output/'best-epoch.json').write_text(json.dumps(history[-1],indent=2)+'\n')
        print('epoch completed',history[-1],flush=True)
        if args.swa:
            # Model-soup guardrail: keep CPU fp32 copies of the 3 best epochs.
            top_ids={h['epoch'] for h in sorted(history,key=lambda h:h['calibration_macro_f1'],reverse=True)[:3]}
            if history[-1]['epoch'] in top_ids:
                snap={k:v.detach().to('cpu',torch.float32).clone() for k,v in engine.backend.model.state_dict().items() if v.is_floating_point()}
                swa_pool=[pair for pair in swa_pool if pair[0]!=history[-1]['epoch']]
                swa_pool.append((history[-1]['epoch'],snap));swa_pool.sort()
                print('swa pool epochs:',[pair[0] for pair in swa_pool],flush=True)
    # Model soup: average the pooled epoch states, then keep whichever wins on
    # the selection half (soup vs single best epoch). Same split, honest comparison.
    if args.swa and len(swa_pool)>=2:
        print('building soup from epochs',[pair[0] for pair in swa_pool],flush=True)
        avg={}
        for _,snap in swa_pool:
            for k,v in snap.items():avg[k]=avg.get(k,0)+v/len(swa_pool)
        current=engine.backend.model.state_dict()
        soup_state={k:(avg[k].to(current[k].dtype).to(current[k].device) if k in avg else current[k]) for k in current}
        engine.backend.model.load_state_dict(soup_state);engine.backend.model.eval()
        soup_metrics=metrics(predict_rows(engine,selection_rows))
        soup_f1=soup_metrics['tier_per_class']['macro_f1']
        print('soup macro_f1',round(soup_f1,4),'vs single-best',round(best_score,4),flush=True)
        if soup_f1>=best_score:
            if args.fullft:
                engine.backend.model.save_pretrained(output/'full-checkpoint',safe_serialization=True)
                import shutil as _shutil
                for _f in (ROOT/'models/intern-decision-0.8b').iterdir():
                    if _f.is_file() and _f.suffix in {'.json','.txt','.jinja','.model'} and _f.name!='config.json':
                        _shutil.copy2(_f,output/'full-checkpoint'/_f.name)
            else:model.save_pretrained(output/'adapter')
            (output/'best-epoch.json').write_text(json.dumps({'selection':'soup','epochs':[p[0] for p in swa_pool],'calibration_macro_f1':soup_f1},indent=2)+'\n')
        else:
            # Reload the saved single-best product before final evaluation.
            if not args.fullft:
                engine.backend.model.load_adapter(output/'adapter',adapter_name='selected');engine.backend.model.set_adapter('selected')
            else:
                from transformers import Qwen3_5ForConditionalGeneration as _M2
                engine.backend.model.load_state_dict(_M2.from_pretrained(output/'full-checkpoint',dtype=torch.bfloat16,local_files_only=True).state_dict())
    # Restore the best checkpoint selected ONLY with the calibration split.
    elif not args.fullft:
        model.load_adapter(output/'adapter',adapter_name='selected');model.set_adapter('selected')
    else:
        from transformers import Qwen3_5ForConditionalGeneration as _M
        state=_M.from_pretrained(output/'full-checkpoint',dtype=torch.bfloat16,local_files_only=True).state_dict()
        engine.backend.model.load_state_dict(state)
    model.eval()
    tuned={}
    for name in ['calibration','test']:
        print('finetuned',name,flush=True);tuned[name]=predict_rows(engine,datasets[name]);write_predictions(output/f'finetuned-{name}.jsonl',tuned[name])
    # Red-team fix: temperatures and the tau operating point are fitted on the
    # fitting half only; the selection half chose the epoch and stays untouched.
    tuned_fit=[r for r in tuned['calibration'] if r['id'] in {row['id'] for row in fitting_rows}]
    temps,temp_notes=fit_temperatures(tuned_fit)
    # Decision-side logit adjustment sweep: tau*log(train prior) subtracted from
    # each tier logit, chosen on the fitting split to maximize macro-F1
    # subject to flagship recall >= 0.75 and flagship false-positive rate <= 0.10.
    import math as _math
    tier_prior={tier:tier_counts[tier]/sum(tier_counts.values()) for tier in TIERS}
    calib_valid=[r for r in tuned_fit if target(r,'tier') in TIERS]
    sweep=[]
    for tau in [0,.25,.5,.75,1.,1.25,1.5,1.75,2.]:
        adj={tier:tau*_math.log(tier_prior[tier]) for tier in TIERS}
        adjusted=calibrated(calib_valid,temps,{'tier':adj})
        pc=per_class(adjusted,'tier')
        fp_rate=pc['flagship']['fp']/max(1,pc['flagship']['fp']+pc['flagship']['tn'])
        sweep.append({'tau':tau,'macro_f1':round(pc['macro_f1'],4),'flagship_recall':round(pc['flagship']['recall'],4),'flagship_fp_rate':round(fp_rate,4)})
    feasible=[s for s in sweep if s['flagship_recall']>=.75 and s['flagship_fp_rate']<=.10]
    tau_infeasible=not feasible
    if tau_infeasible:
        # Honest degradation instead of a hard exit: keep the report flowing, flag
        # the miss, and prefer the highest macro-F1 with recall>=0.6 & FP<=0.15.
        relaxed=[s for s in sweep if s['flagship_recall']>=.6 and s['flagship_fp_rate']<=.15] or sweep
        chosen=max(relaxed,key=lambda s:s['macro_f1'])
        print('TAU GATE MISSED: no point satisfies recall>=0.75 & FP<=0.10; relaxed pick',json.dumps(chosen),flush=True)
    else:
        chosen=max(feasible,key=lambda s:s['macro_f1'])
    tier_adjust={tier:chosen['tau']*_math.log(tier_prior[tier]) for tier in TIERS}
    print('logit adjustment sweep',json.dumps(sweep),'chosen',json.dumps(chosen),flush=True)
    tuned_metrics=metrics(calibrated(tuned['test'],temps,{'tier':tier_adjust}))
    # Expert agreement is insufficient for production: outcome evidence is mandatory.
    gate_reasons=['No independent candidate-model task-success benchmark has been collected.']
    if tau_infeasible:gate_reasons.append('Tau operating point missed the recall>=0.75 / FP<=0.10 gate; relaxed point in use.')
    if tuned_metrics['flagship_examples']<30:gate_reasons.append('Fewer than 30 independent flagship test examples.')
    if tuned_metrics['vision_positive_examples']<20:gate_reasons.append('Fewer than 20 independent vision-positive test examples.')
    if (tuned_metrics['flagship_underroute_rate'] or 0)>.05:gate_reasons.append('Flagship under-routing exceeds 5%.')
    if (tuned_metrics['vision_false_negative_rate'] or 0)>.05:gate_reasons.append('Vision false-negative rate exceeds 5%.')
    report={'run':args.run,'seed':args.seed,'base_model':json.loads((ROOT/'model-manifest.json').read_text()),'training':{'epochs':args.epochs,'rank':args.rank,'lr':args.lr,'accumulation':args.accumulation,'trainable_parameters':trainable,'optimizer_updates':updates,'seconds':time.perf_counter()-start,'history':history},'evaluation_basis':'agreement with fresh expert prompt-only labels; NOT real generation task success','baseline':baseline_metrics,'finetuned':tuned_metrics,'baseline_temperatures':baseline_temps,'baseline_calibration_notes':baseline_temp_notes,'temperatures':temps,'calibration_notes':temp_notes,'logit_adjustment':{'sweep':sweep,'chosen':chosen},'tier_class_weights':weights,'vision_class_weights':vision_weights,'quality_gate':{'passed':False,'mode':'shadow','reasons':gate_reasons},'split_report':json.loads((ROOT/'data/split-report.json').read_text())}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'calibration.json').write_text(json.dumps({'temperatures':temps,'tier_adjust':tier_adjust,'tier_adjust_tau':chosen['tau'],'tier_prior':tier_prior,'notes':temp_notes},ensure_ascii=False,indent=2)+'\n')
    (output/'quality-gate.json').write_text(json.dumps(report['quality_gate'],ensure_ascii=False,indent=2)+'\n')
    print('DONE',output/'report.json',json.dumps({'baseline_accuracy':baseline_metrics['tier']['accuracy'],'finetuned_accuracy':tuned_metrics['tier']['accuracy'],'quality_gate':report['quality_gate']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='real-history-v1');p.add_argument('--epochs',type=int,default=6);p.add_argument('--rank',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--seed',type=int,default=20261010);p.add_argument('--synth',action='store_true',help='merge data/synth/*.jsonl into train/calibration');p.add_argument('--beta',type=float,default=.999);p.add_argument('--mlp',action='store_true',help='extend LoRA to MLP and all attention output/input projections (deeper modification)');p.add_argument('--fullft',action='store_true',help='full fine-tune of the language backbone (deepest modification; vision tower stays frozen via LoRA-free path)');p.add_argument('--label-smoothing',type=float,default=.05);p.add_argument('--swa',action='store_true',help='average the 3 best epochs by selection macro-F1 (model soup guardrail)');p.add_argument('--smoke',action='store_true',help='24-sample 1-epoch pipeline validation run')
    args=p.parse_args()
    if not args.run.replace('-','').replace('_','').isalnum():raise SystemExit('Invalid run name')
    run(args)
