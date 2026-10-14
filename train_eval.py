"""Actual LoRA training, held-out temperature calibration, and expert-imitation evaluation."""
from __future__ import annotations
import argparse, collections, json, math, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import LoraConfig,get_peft_model
from model_utils import ROOT,TIERS,TYPES,QUESTIONS,load_engine,read_jsonl,request_for,field_logits,score

# --- Issue #14: training-reproducibility constants (scheduler / SWA audit) ---
# Official XTuner recipe (configs/training/qwen35.py): fullft lr 2e-6 with
# cosine decay to 2e-7 and 3% warmup. 2e-7 = 10% of the initial LR, so the
# cosine multiplier floor is 0.1 — the archived v3c run decayed only to a
# 1e-3 multiplier; the difference is declared in both READMEs and in
# effective_config so archived/new LR trajectories are never silently mixed.
FULLFT_INITIAL_LR=2e-6
LR_FLOOR_FRACTION=.1
SWA_POOL_SIZE=3
# Issue #14b: bnb.optim.AdamW8bit quantizes only m/v state tensors with at
# least this many elements to 8-bit; smaller tensors (heads, anchors) keep
# exact 32-bit state. One constant shared by construction and effective_config.
OPTIMIZER_MIN_8BIT_SIZE=16384


def oversample_epoch_rows(train_rows):
    # Class-balanced OVERSAMPLING: repeat minority-tier rows to equal per-class
    # counts, then fully shuffle. The earlier round-robin drained minority
    # buckets first and left a pure-majority tail whose recency dominated the
    # final gradient steps of every epoch (the v3/v3b balanced-collapse cause).
    # Issue #14: this helper is the ONE definition of the per-epoch training
    # row stream — the training loop iterates its output and the LR scheduler
    # counts optimizer updates from the same lists, so step accounting can
    # never drift from actual training again (the archived run planned 498
    # steps from the raw length while actually taking 1158 updates).
    buckets=collections.defaultdict(list)
    for row in train_rows:buckets[row['label']['tier']].append(row)
    quota=max(len(b) for b in buckets.values())
    rows=[]
    for tier,bucket in buckets.items():
        random.shuffle(bucket)
        rows.extend(bucket*(quota//len(bucket))+random.sample(bucket,quota%len(bucket)))
    random.shuffle(rows)
    return rows


def optimizer_updates(row_count,accumulation):
    # Optimizer updates for one epoch of row_count oversampled rows: the
    # training loop steps every `accumulation` rows AND on the final possibly
    # short tail batch — ceil(row_count/accumulation). run() asserts the real
    # loop counter against this formula after training (issue #14 invariant).
    return -(-row_count//max(1,accumulation))


def cosine_lr_multiplier(step,total_steps,warmup):
    # Cosine-with-warmup LR multiplier (fullft scheduler only). Issue #14
    # correctness: progress is CLAMPED to [0,1] so steps past the planned
    # total can never turn the cosine back up (the old curve rose again after
    # the end), and the decay floor is 10% of the initial LR.
    if step<warmup:return step/warmup
    progress=(step-warmup)/max(1,total_steps-warmup)
    progress=min(1.,max(0.,progress))
    return max(LR_FLOOR_FRACTION,(1+math.cos(math.pi*progress))/2)


def fullft_scheduler_plan(planned_updates,effective_lr):
    # Scheduler plan + audit report for the fullft cosine path (issue #14).
    # The .03 is the warmup TARGET fraction: after int() truncation and the
    # max(1,..) floor the ACHIEVED fraction differs (e.g. the archived v3c
    # shape — 6*193 = 1158 updates, warmup int(1158*.03) = 34 — achieves
    # 34/1158 ≈ 0.0294), so the report records warmup/total_steps, never a
    # hardcoded .03 that lied about every floored or truncated run
    # (red-team LOW-2).
    total_steps=max(1,planned_updates)
    warmup=max(1,int(total_steps*.03))
    report={'type':'cosine_with_warmup','total_optimizer_updates':total_steps,'warmup_updates':warmup,'warmup_fraction_of_total':warmup/total_steps,'cosine_progress_clamp':[0.,1.],'lr_floor_fraction_of_initial':LR_FLOOR_FRACTION,'lr_floor':effective_lr*LR_FLOOR_FRACTION,'counting_basis':'optimizer updates over class-balanced oversampled rows; per-epoch tail batch counted by ceil'}
    return total_steps,warmup,report


def effective_initial_lr(args):
    # The LR the optimizer ACTUALLY starts from: fullft fixes it at 2e-6
    # (official recipe) regardless of --lr; LoRA paths use --lr directly.
    # Reports must record this alongside the requested value (issue #14:
    # args.lr was previously reported as if it were in effect under fullft).
    # Identical for BOTH optimizers (issue #14b): --optimizer never changes
    # LR semantics, only how the m/v moments are stored.
    return FULLFT_INITIAL_LR if args.fullft else args.lr


def ensure_optimizer_available(optimizer_name):
    # Issue #14b fail-fast gate, run BEFORE any data/model/artifact work: a
    # machine without bitsandbytes must neither load the 0.8B model nor create
    # the append-only artifacts/<run>/ directory whose existence would then
    # refuse the re-run. Never a silent fallback to adamw — that would
    # impersonate the declared recipe in the report.
    if optimizer_name=='adamw8bit':
        try:import bitsandbytes  # noqa: F401 -- availability check only
        except ImportError as error:raise SystemExit('bitsandbytes not installed; --optimizer adamw8bit requires it') from error


def build_optimizer(optimizer_name,params,effective_lr):
    """Issue #14b optimizer construction. adamw (the default) is byte-identical
    to the pre-flag call: torch.optim.AdamW(lr=effective_lr, weight_decay=.01,
    foreach=False). adamw8bit swaps the fp32 AdamW m/v moments for
    bitsandbytes' 8-bit states (m/v ~3,800 -> ~950MiB on this 16GB GPU; fp32
    AdamW fullft OOMs >15.8GiB, 8-bit lands ~9.7GiB) under IDENTICAL LR
    semantics (fullft still fixes the initial LR at 2e-6). A missing
    bitsandbytes install fails loudly instead of silently impersonating adamw."""
    if optimizer_name=='adamw8bit':
        try:import bitsandbytes as bnb
        except ImportError as error:raise SystemExit('bitsandbytes not installed; --optimizer adamw8bit requires it') from error
        return bnb.optim.AdamW8bit(params,lr=effective_lr,weight_decay=.01,min_8bit_size=OPTIMIZER_MIN_8BIT_SIZE,foreach=False)
    return torch.optim.AdamW(params,lr=effective_lr,weight_decay=.01,foreach=False)


def top_swa_epochs(history):
    # The SWA pool's single definition of "best": top-K epochs by selection
    # macro-F1 (stable sort — an earlier epoch wins a tie, exactly like the
    # archived sorted() expression).
    return [h['epoch'] for h in sorted(history,key=lambda h:h['calibration_macro_f1'],reverse=True)[:SWA_POOL_SIZE]]


def update_swa_pool(swa_pool,history,snapshot):
    # Model-soup guardrail, strict Top-K (issue #14): after each epoch,
    # recompute the current top-3 by selection macro-F1, ADMIT the
    # just-finished epoch only if it qualifies, and EVICT pooled epochs that
    # fell out of the top-3 (dropping their CPU fp32 snapshot references —
    # the old pool over-filled and averaged 4+ epochs while claiming "3 best").
    # An evicted epoch can never re-enter (its score is fixed, competitors
    # only grow), so the final pool IS the final top-3 and the soup is their
    # honest average. snapshot(epoch) is only invoked for a newly admitted
    # epoch, so no wasted CPU copy is ever made.
    keep=set(top_swa_epochs(history))
    pool=[pair for pair in swa_pool if pair[0] in keep]
    last=history[-1]['epoch']
    if last in keep and last not in {pair[0] for pair in pool}:
        pool.append((last,snapshot(last)))
    pool.sort()
    return pool


def average_state_dicts(snaps):
    # Model soup: element-wise mean of the pooled fp32 CPU state dicts.
    avg={}
    for snap in snaps:
        for k,v in snap.items():avg[k]=avg.get(k,0)+v/len(snaps)
    return avg


def soup_state_dict(current,avg):
    # Cast the averaged fp32 soup back onto the live model's dtypes/devices,
    # keeping non-floating keys from the current state verbatim.
    return {k:(avg[k].to(current[k].dtype).to(current[k].device) if k in avg else current[k]) for k in current}


def effective_config(args,raw_train_rows,epoch_row_counts,scheduler_report,swa_pool):
    """Issue #14 audit block: what ACTUALLY ran, recorded separately from the
    raw CLI input. Values here are EFFECTIVE (the loop, optimizer, and
    scheduler really used them); requested values are echoed under lr_* so
    effective and requested can never be conflated again."""
    initial=effective_initial_lr(args)
    optimizer_name=getattr(args,'optimizer','adamw')
    block={'note':'effective vs requested: these values are what the run actually used; raw CLI input is preserved separately',
           'recipe':{'epochs':args.epochs,'rank':args.rank,'accumulation':args.accumulation,'seed':args.seed,'beta':args.beta,'label_smoothing':args.label_smoothing,'synth':bool(args.synth),'mlp':bool(args.mlp),'fullft':bool(args.fullft),'swa':bool(args.swa),'optimizer':optimizer_name},
           'optimizer':{'name':optimizer_name,'weight_decay':.01,
                        # min_8bit_size only exists on the 8-bit path; None on
                        # adamw so the field never implies quantization ran.
                        'min_8bit_size':(OPTIMIZER_MIN_8BIT_SIZE if optimizer_name=='adamw8bit' else None)},
           'lr_effective':initial,'lr_requested':args.lr,
           'lr_note':('fullft fixes the initial LR at 2e-6 (official recipe); --lr is ignored under --fullft' if args.fullft else '--lr is used directly (LoRA path)'),
           'train_rows':{'pre_oversampling':raw_train_rows,'oversampled_per_epoch':epoch_row_counts},
           'epoch_optimizer_updates':[optimizer_updates(n,args.accumulation) for n in epoch_row_counts],
           'accumulation_tail_batch':'ceil per epoch: a final batch shorter than --accumulation still triggers an optimizer update',
           'scheduler':scheduler_report,
           'swa_pool_final_epochs':([pair[0] for pair in swa_pool] if args.swa else None)}
    if getattr(args,'legacy_calibration',False):
        block['archived_v3c_divergence_note']=('legacy reproduction shares the issue #14 correctness fixes (oversampled-step scheduler, clamped cosine, 10% LR floor, strict top-3 SWA pool); the archived v3c numbers were produced under the OLD scheduler behavior, so LR trajectories are not comparable')
    return block


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


def legacy_calibration_halves(calibration_rows,seed):
    # Archived v3c calibration split, preserved VERBATIM for reproduction.
    # Why it stays (issue #2, external review P1-3): a seeded row shuffle
    # bisects the calibration pool, so near-duplicate tasks sharing a
    # split_group can land on both sides — task ids are unique but sources
    # are not, and with only ~8 recorded calibration groups the halves
    # inevitably share groups. Re-splitting rows with another seed cannot
    # fix that, which is why new runs pre-register whole groups (roles A/B/C
    # in calibration_groups.py) instead. Explicit --legacy-calibration only.
    rng=random.Random(seed);calib=calibration_rows[:];rng.shuffle(calib)
    half=max(1,len(calib)//2)
    return calib[:half],calib[half:] or calib


def grouped_schema():
    # Runtime business schema: assembled from the LIVE model_utils.QUESTIONS by
    # the SAME schema_from_questions the stdlib preflight uses (it parses this
    # file's source via questions_from_source), so preflight and run cannot
    # validate different truths. Vision stays a REQUIRED head: strict preflight
    # must not silently downgrade to tier-only when support is thin. task_type
    # is optional and fixed at T=1 (reported as not calibrated) — the legacy
    # 2.747 checkpoint fallback is never used by new runs.
    from calibration_groups import schema_from_questions
    return schema_from_questions(QUESTIONS)


def build_grouped_plan(calibration_rows,roles_config):
    # Issue #2 wiring: the pre-registered plan (A/B/C groups, support policy,
    # numeric protocol) is validated by the stdlib module BEFORE any model is
    # loaded or any artifact directory is created. check_roles_config is the
    # SAME contract preflight_roles enforces (no schema override, no unknown
    # keys): a config that passes scripts/train.py --dry-run must mean exactly
    # the same thing here, and a roles schema override can never downgrade
    # vision to optional.
    from calibration_groups import build_plan,check_roles_config,schema_from_questions
    check_roles_config(roles_config)
    return build_plan(rows=calibration_rows,role_groups=roles_config['role_groups'],
                      schema=schema_from_questions(QUESTIONS),
                      support_policy=roles_config.get('support_policy'),protocol=roles_config.get('protocol'))


def run_grouped_calibration(plan,scored_rows,prior,protocol,checkpoint_ref):
    # Issue #2 adapter between the torch training loop and the stdlib OOF
    # calibration: re-attaches split_group by id and forwards ONLY role-B
    # rows. Roles A/C and any test rows must never reach temperature/tau
    # fitting; calibration_groups.select_tau_oof additionally rejects any
    # row set that is not exactly role B, so a wiring mistake fails loudly.
    from calibration_groups import select_tau_oof
    by_id={row['id']:row for row in plan['rows_by_role']['B']}
    logits_rows=[{'id':row['id'],'split_group':by_id[row['id']]['split_group'],'label':row['label'],
                  'logits':{field:answer['logits'] for field,answer in row['answers'].items()}}
                 for row in scored_rows if row['id'] in by_id]
    return select_tau_oof(plan=plan,rows=logits_rows,prior=prior,protocol=protocol,checkpoint_ref=checkpoint_ref)


def servable_calibration(calibration):
    """The server-consumable calibration.json payload of a grouped run, or
    None when nothing is servable. server.py's lifespan reads exactly
    temperatures + tier_adjust (subtracted from raw logits by
    model_utils.score, the same z - tau*log(prior) the temperatures were
    fitted on), so a 'selected' result MUST carry both — serving a selected
    calibration without tier_adjust would silently route raw logits through
    a temperature fitted on adjusted ones. An 'insufficient' result produces
    NO artifact: the run stays unservable (server startup refuses on the
    missing file) instead of being filled with placeholder temperatures, and
    its quality gate stays not-passed so execution mode never authorizes."""
    if not isinstance(calibration,dict) or calibration.get('status')!='selected':
        return None
    if not calibration.get('temperatures') or 'tier_adjust' not in calibration:
        raise ValueError('selected calibration is missing server-consumable temperatures/tier_adjust')
    return {'status':'selected','selected_tau':calibration['selected_tau'],
            'temperatures':calibration['temperatures'],'tier_adjust':calibration['tier_adjust'],
            'tier_prior':calibration.get('tier_prior'),'calibrated':calibration['calibrated'],
            'binding':calibration['binding'],'note':calibration['note']}


def grouped_smoke_pool(datasets,roles):
    # Grouped --smoke, honest and short (red-team F7): calibration keeps
    # WHOLE groups (smallest-support first) re-validated under the SAME
    # support policy — slicing rows would rebuild the issue #2 leak in
    # miniature and no gate is ever relaxed — and TRAIN is shrunk to the
    # same 24 rows as legacy smoke so the run is actually short. The prior
    # is unaffected: run() captured real_train_rows before this slice.
    from calibration_groups import smoke_subpool
    plan=smoke_subpool(datasets['calibration'],roles['role_groups'],grouped_schema(),
                       roles.get('support_policy'),roles.get('protocol'))
    return plan,datasets['train'][:24]


def run(args):
    # Issue #2 mode gate, checked before anything else: new runs default to
    # group-isolated calibration with PRE-REGISTERED A/B/C role groups; the
    # archived row-split calibration is reachable only through the explicit
    # --legacy-calibration flag. No new run may silently fall back to legacy.
    legacy=bool(getattr(args,'legacy_calibration',False))
    roles_path=getattr(args,'calibration_roles',None)
    if legacy and roles_path:raise SystemExit('--legacy-calibration and --calibration-roles are mutually exclusive')
    if not legacy and not roles_path:raise SystemExit('New runs default to group-isolated calibration (issue #2): pass --calibration-roles with pre-registered A/B/C groups, or pass --legacy-calibration to reproduce the archived v3c recipe')
    if getattr(args,'structural_dry_run',False) and legacy:raise SystemExit('--structural-dry-run applies to grouped runs only')
    # Issue #14b: refuse an un-runnable recipe before touching the model or
    # creating the append-only artifacts/<run>/ directory.
    ensure_optimizer_available(getattr(args,'optimizer','adamw'))
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    output=ROOT/'artifacts'/args.run
    if legacy:output.mkdir(parents=True,exist_ok=True)
    # Grouped runs create the directory only AFTER preflight passes: a
    # support violation must fail without touching disk or the GPU, and the
    # run directory is strictly append-only (exist_ok=False).
    # Grouped dev runs never open the test split (dev-no-test). The legacy
    # path still reads test for its historical report; the complete
    # dev-no-test rework for training is NOT satisfied by this change.
    datasets={name:read_jsonl(ROOT/'data'/f'{name}.jsonl') for name in (['train','calibration','test'] if legacy else ['train','calibration'])}
    plan=None;protocol=None;roles=None;real_train_rows=None
    if not legacy:
        roles=json.loads(Path(roles_path).read_text())
        plan=build_grouped_plan(datasets['calibration'],roles)  # preflight BEFORE load_engine()/artifacts
        protocol=plan['protocol']
        real_train_rows=datasets['train'][:]  # captured before synthetic joins: priors must come from real unique train rows only
        # Prior preflight BEFORE any model/dir work (same truth as
        # scripts/validate_calibration.py): real unique train rows must
        # support every tier class, or the run fails before it starts.
        from calibration_groups import compute_prior
        prior=compute_prior(real_train_rows,'tier',grouped_schema())
        if args.structural_dry_run:
            from calibration_groups import anonymous_summary
            print('STRUCTURAL DRY RUN (no model, no artifacts, no test access)',json.dumps(anonymous_summary(plan),ensure_ascii=False),flush=True)
            return
        output.mkdir(parents=True,exist_ok=False)
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
    # Issue #2: this ROW bisection is the remaining leak — selection/fitting
    # stay group-correlated (see legacy_calibration_halves). Kept verbatim for
    # v3c reproduction under the explicit --legacy-calibration flag only.
    if legacy:
        selection_rows,fitting_rows=legacy_calibration_halves(datasets['calibration'],args.seed)
        print('calibration split: selection',len(selection_rows),'fitting',len(fitting_rows),flush=True)
        if args.smoke:
            datasets['train']=datasets['train'][:24];selection_rows=selection_rows[:6];fitting_rows=fitting_rows[:6];args.epochs=1
            print('SMOKE MODE: 24 train rows, 6+6 calibration rows, 1 epoch',flush=True)
    else:
        if args.smoke:
            # Grouped smoke (F7): whole-group calibration sub-pool under the
            # SAME policy + a 24-row train slice, so it is genuinely short
            # without relaxing any gate. See grouped_smoke_pool.
            plan,datasets['train']=grouped_smoke_pool(datasets,roles)
            args.epochs=1
            print('SMOKE MODE (grouped): 24 train rows, whole-group sub-pool',
                  len(plan['roles']['A'])+len(plan['roles']['B'])+len(plan['roles']['C']),
                  'groups, 1 epoch; support gates unchanged',flush=True)
        # Epoch/checkpoint selection reads ONLY role-A groups; temperatures
        # and tau are fitted ONLY on role-B groups later (group isolation).
        selection_rows=plan['rows_by_role']['A']
    engine=load_engine();baseline={}
    if not legacy:
        # Grouped dev runs make no baseline predictions: baseline evaluation
        # historically reads the test split, which grouped dev runs must not
        # touch (dev-no-test); no evaluation claim is being made here.
        print('grouped dev run: skipping baseline/test evaluation (dev-no-test; issue #2 scope)',flush=True)
    else:
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
    # Issue #14 single truth source: the oversampled per-epoch row lists are
    # built ONCE here from the fully assembled train pool (synth join and any
    # --smoke slice already applied; same helper and same RNG draw sequence as
    # the old inline per-epoch code). The training loop iterates these lists
    # and the scheduler counts its updates from them, so the LR plan can never
    # again be estimated from the raw pre-oversampling length.
    raw_train_rows=len(datasets['train'])
    epoch_train_rows=[oversample_epoch_rows(datasets['train']) for _ in range(args.epochs)]
    epoch_row_counts=[len(rows) for rows in epoch_train_rows]
    planned_updates=sum(optimizer_updates(n,args.accumulation) for n in epoch_row_counts)
    effective_lr=effective_initial_lr(args)
    # Official XTuner recipe (configs/training/qwen35.py): lr 2e-6, cosine decay
    # to 2e-7, warmup 3%, wd 0.01. Our fullft path follows it; LoRA paths keep lr 1e-4.
    optimizer=build_optimizer(getattr(args,'optimizer','adamw'),(p for p in model.parameters() if p.requires_grad),effective_lr)
    scheduler=None;scheduler_report={'type':'constant (no scheduler)','note':'LoRA paths keep the optimizer LR fixed'}
    if args.fullft:
        total_steps,warmup,scheduler_report=fullft_scheduler_plan(planned_updates,effective_lr)
        def _lr_lambda(step):return cosine_lr_multiplier(step,total_steps,warmup)
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
        # Oversampled per-epoch rows come from the single precomputed source
        # above (oversample_epoch_rows); the loop consumes, never rebuilds.
        rows=epoch_train_rows[epoch]
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
            # Model-soup guardrail: strict top-3 pool with eviction (issue #14).
            def _snapshot(epoch_id):
                return {k:v.detach().to('cpu',torch.float32).clone() for k,v in engine.backend.model.state_dict().items() if v.is_floating_point()}
            swa_pool=update_swa_pool(swa_pool,history,_snapshot)
            print('swa pool epochs:',[pair[0] for pair in swa_pool],flush=True)
    # Issue #14 invariant: the scheduler was built from planned_updates; if the
    # loop's real stepping ever diverges from the counting formula, fail loudly
    # instead of silently mis-scheduling every future run.
    if updates!=planned_updates:raise AssertionError(f'optimizer update drift: planned {planned_updates}, executed {updates}')
    # Model soup: average the pooled epoch states, then keep whichever wins on
    # the selection half (soup vs single best epoch). Same split, honest comparison.
    if args.swa and len(swa_pool)>=2:
        print('building soup from epochs',[pair[0] for pair in swa_pool],flush=True)
        avg=average_state_dicts([snap for _,snap in swa_pool])
        soup_state=soup_state_dict(engine.backend.model.state_dict(),avg)
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
    # Issue #14 audit snapshot, shared by both report paths: what actually ran
    # (effective LR, oversampled per-epoch update counts, scheduler shape,
    # final SWA pool members) — recorded separately from raw CLI input.
    effective=effective_config(args,raw_train_rows,epoch_row_counts,scheduler_report,swa_pool)
    if not legacy:
        # Grouped calibration tail (issue #2): score ONLY role-B rows with the
        # A-selected checkpoint, select tau/T out-of-fold over B's whole-group
        # folds, then refit once on full B — all inside calibration_groups.
        # The result binds plan/prior/schema/protocol/checkpoint hashes so
        # compared models provably shared one plan, folds, grid, and budget.
        # This is a development estimate, not an independent evaluation.
        from calibration_groups import checkpoint_digest
        scored_b=predict_rows(engine,plan['rows_by_role']['B'])
        write_predictions(output/'scored-B.jsonl',scored_b)
        # Prior was preflighted above from REAL unique train rows only
        # (synthetic augmentation and repeated ids excluded; missing tier
        # support is a hard error; the legacy path keeps its own prior).
        # F5: the calibration binds the SHA-256 of the ACTUAL checkpoint
        # weight files (adapter/ or full-checkpoint/, streamed, relative
        # filenames explicit) — not the best-epoch.json text, which is an
        # unvalidated selection event, not weight bytes.
        checkpoint_dir=output/('full-checkpoint' if args.fullft else 'adapter')
        checkpoint_ref=json.dumps(checkpoint_digest(checkpoint_dir),sort_keys=True)
        calibration=run_grouped_calibration(plan,scored_b,{'tier':prior['probabilities']},protocol,checkpoint_ref)
        servable=servable_calibration(calibration)
        if servable is not None:
            # Only a selected calibration is servable: temperatures fitted on
            # adjusted logits ship together with the tier_adjust the server
            # subtracts (same formula), keeping train/serve parity.
            (output/'calibration.json').write_text(json.dumps(servable,ensure_ascii=False,indent=2)+'\n')
        reasons=['Group-isolated development calibration only (issue #2): no test access; no independent evaluation has been run.',
                 'Training prerequisites under the pending evaluation protocol (approved development data/protocol) are NOT yet satisfied; this run is not training-ready or production-ready.']
        if calibration['status']!='selected':
            reasons.append('No feasible tau under the registered protocol: temperatures were not calibrated and NO calibration.json was written; the server refuses to load this run rather than serve unadjusted logits.')
        report={'run':args.run,'seed':args.seed,'mode':'grouped-dev','base_model':json.loads((ROOT/'model-manifest.json').read_text()),'training':{'epochs':args.epochs,'rank':args.rank,'lr':effective_lr,'accumulation':args.accumulation,'trainable_parameters':trainable,'optimizer_updates':updates,'seconds':time.perf_counter()-start,'history':history},'effective_config':effective,'evaluation_basis':'group-isolated development calibration (A: checkpoint selection, B: out-of-fold temperature/tau); NOT an independent evaluation','calibration':calibration,'tier_class_weights':weights,'vision_class_weights':vision_weights,'quality_gate':{'passed':False,'mode':'shadow','reasons':reasons},'split_report':json.loads((ROOT/'data/split-report.json').read_text())}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (output/'quality-gate.json').write_text(json.dumps(report['quality_gate'],ensure_ascii=False,indent=2)+'\n')
        print('DONE',output/'report.json',json.dumps({'mode':'grouped-dev','calibration_status':calibration['status'],'quality_gate':report['quality_gate']},ensure_ascii=False),flush=True)
        return
    tuned={}
    for name in ['calibration','test']:
        print('finetuned',name,flush=True);tuned[name]=predict_rows(engine,datasets[name]);write_predictions(output/f'finetuned-{name}.jsonl',tuned[name])
    # Red-team fix: temperatures and the tau operating point are fitted on the
    # fitting half only; the selection half chose the epoch and stays untouched.
    # Issue #2 (legacy only): this sweep still treats fitting-half ROWS as
    # independent observations; the grouped path replaces it with whole-group
    # out-of-fold selection in calibration_groups.py.
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
    report={'run':args.run,'seed':args.seed,'base_model':json.loads((ROOT/'model-manifest.json').read_text()),'training':{'epochs':args.epochs,'rank':args.rank,'lr':effective_lr,'accumulation':args.accumulation,'trainable_parameters':trainable,'optimizer_updates':updates,'seconds':time.perf_counter()-start,'history':history},'effective_config':effective,'evaluation_basis':'agreement with fresh expert prompt-only labels; NOT real generation task success','baseline':baseline_metrics,'finetuned':tuned_metrics,'baseline_temperatures':baseline_temps,'baseline_calibration_notes':baseline_temp_notes,'temperatures':temps,'calibration_notes':temp_notes,'logit_adjustment':{'sweep':sweep,'chosen':chosen},'tier_class_weights':weights,'vision_class_weights':vision_weights,'quality_gate':{'passed':False,'mode':'shadow','reasons':gate_reasons},'split_report':json.loads((ROOT/'data/split-report.json').read_text())}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'calibration.json').write_text(json.dumps({'temperatures':temps,'tier_adjust':tier_adjust,'tier_adjust_tau':chosen['tau'],'tier_prior':tier_prior,'notes':temp_notes},ensure_ascii=False,indent=2)+'\n')
    (output/'quality-gate.json').write_text(json.dumps(report['quality_gate'],ensure_ascii=False,indent=2)+'\n')
    print('DONE',output/'report.json',json.dumps({'baseline_accuracy':baseline_metrics['tier']['accuracy'],'finetuned_accuracy':tuned_metrics['tier']['accuracy'],'quality_gate':report['quality_gate']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='real-history-v1');p.add_argument('--epochs',type=int,default=6);p.add_argument('--rank',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4,help='LoRA initial learning rate; IGNORED under --fullft, which fixes the initial LR at 2e-6 (official recipe — effective_config records effective vs requested)');p.add_argument('--accumulation',type=int,default=8);p.add_argument('--seed',type=int,default=20261010);p.add_argument('--synth',action='store_true',help='merge data/synth/*.jsonl into train/calibration');p.add_argument('--beta',type=float,default=.999);p.add_argument('--mlp',action='store_true',help='extend LoRA to MLP and all attention output/input projections (deeper modification)');p.add_argument('--fullft',action='store_true',help='full fine-tune of the language backbone (deepest modification; vision tower stays frozen via LoRA-free path). LR schedule (issue #14): 3%% warmup + cosine over the OVERSAMPLED optimizer-update count (tail batch ceil), progress clamped to [0,1], decay floor at 10%% of the initial LR (2e-7; the archived v3c used a 1e-3 multiplier floor)');p.add_argument('--label-smoothing',type=float,default=.05);p.add_argument('--swa',action='store_true',help='average the 3 best epochs by selection macro-F1 (model soup guardrail; strict top-3 pool with eviction — issue #14)');p.add_argument('--smoke',action='store_true',help='short pipeline validation run: legacy = 24 train rows, 6+6 calibration rows, 1 epoch; grouped = 24 train rows + whole-group calibration sub-pool, 1 epoch (support gates never relaxed)')
    p.add_argument('--legacy-calibration',action='store_true',help='reproduce the archived v3c row-split calibration exactly (issue #2: explicit legacy; new runs default to group-isolated calibration)')
    p.add_argument('--optimizer',default='adamw',choices=['adamw','adamw8bit'],help='optimizer over trainable params: adamw (default; byte-identical to the archived v3c) or adamw8bit (bitsandbytes 8-bit m/v states — trades exact AdamW moments for ~2.8GiB VRAM on this 16GB GPU; requires the RUNTIME-side bitsandbytes install and exits loudly when missing, never silently falls back). LR semantics identical: fullft fixes the initial LR at 2e-6 under both')
    p.add_argument('--calibration-roles',default=None,help='JSON file with pre-registered A/B/C role groups (+ optional support_policy/protocol); required for new runs')
    p.add_argument('--structural-dry-run',action='store_true',help='grouped runs: run the CPU structural preflight (no model, no artifacts, no test access) and exit')
    args=p.parse_args()
    if not args.run.replace('-','').replace('_','').isalnum():raise SystemExit('Invalid run name')
    run(args)
