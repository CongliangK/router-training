"""Import candidate execution evidence against explicit routes; never derive labels from inherited routes.

Schema v2 (issue #7): routes are an explicit input (swap in another routes.json and
re-evaluate candidates with no code change), the selection gate is a one-sided exact
Clopper-Pearson lower bound instead of a success-rate point estimate, failures carry
typed semantics, and every trial keeps an audit envelope.

Privacy boundary: `trial_details` are execution evidence for LOCAL audit files only —
never paste them into issues, PRs, or any public artifact; public material may carry
only purely synthetic coarse aggregates (repository Hard rule 1).

Independence is a sourced assumption, not something a hash or a fresh id proves:
identical artifact bytes across distinct executions are legal (deterministic tasks can
make independent runs coincide) and only raise an audit flag — never a rejection or a
sample-size discount; a nonce or fresh execution_id must never mutate the artifact body
to fake distinct executions. Estimand: single-execution success rate of
harness-independent executions on the fixed evaluated task set; no extrapolation to new
task populations and no production-benefit claim.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re,sys
from pathlib import Path
from policy import DEFAULT_ROUTES,TIERS,validate_routes

FAILURE_TYPES=('timeout','acceptance','execution_error','other')
BOUND_METHOD='clopper_pearson_exact_one_sided_lower'

def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def routes_digest(routes):return hashlib.sha256(canonical(routes).encode()).hexdigest()

def route_fingerprint(route,version,effort):
    # Operating-point identity: the same provider/model under a different model_version
    # or reasoning_effort is a different candidate. [:16] is an audit id, not a hash chain.
    payload=json.dumps([route['provider'],route['model'],route['tier'],version,effort],separators=(',',':'),ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def _metric(record,key):
    value=record.get(key)
    # bool is an int subclass; resource metrics must reject it explicitly.
    if type(value) is bool or type(value) not in (float,int) or not math.isfinite(value) or value<0:raise ValueError('Invalid outcome resource metric')

def binomial_lower_bound(k,n,confidence=.95):
    """One-sided exact Clopper-Pearson lower bound: the L solving P(Bin(n,L)>=k)=alpha, alpha=1-confidence.

    L(0,n)=0 and n=0 -> 0.0 (no trials: no evidence; callers keep such candidates
    pending instead of crashing). L(n,n)=alpha**(1/n) closed form. Stable by
    construction: log-pmf through lgamma plus a tail summed by term-ratio recursion —
    the naive math.comb(n,k)*p**k*(1-p)**(n-k) tail overflows float at n>=1030
    (red-team probe: stable sf(515,1030,.5)=0.5124275650 while the naive sum raises
    OverflowError), and 200 bisection halves converge far below float resolution.
    Verified anchors at 95% one-sided: L(3,3)=0.3684031499, L(28,28)=0.8985342644,
    L(29,29)=0.9018553723, L(49,50)=0.9086018693, L(99,100)=0.9534401885.
    (Comparison note only, not implemented: a one-sided-95% Wilson bound for 3/3
    would be 0.5258044258; two-sided 0.4385029682.)
    """
    if type(k) is bool or type(n) is bool or type(confidence) is bool:raise ValueError('Bound inputs must not be booleans')
    if type(k) is not int or type(n) is not int:raise ValueError('Bound inputs must be integers')
    if type(confidence) not in (float,int) or not math.isfinite(confidence) or not 0<confidence<1:raise ValueError('Confidence must be within (0,1)')
    if n<0 or k<0 or k>n:raise ValueError('Bound requires 0<=k<=n')
    if n==0 or k==0:return 0.0
    alpha=1.0-confidence
    if k==n:return alpha**(1.0/n)
    def survival(p):  # P(Bin(n,p)>=k) for the fixed 0<k<n
        log_t=math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log1p(-p)
        t=math.exp(log_t);s=t
        for i in range(k+1,n+1):
            t*=(n-i+1.0)/i*(p/(1.0-p));s+=t
        return s
    lo,hi=0.0,1.0
    for _ in range(200):
        m=0.5*(lo+hi)
        if survival(m)>alpha:hi=m
        else:lo=m
    bound=0.5*(lo+hi)
    if not math.isfinite(bound):raise ValueError('Lower bound failed finiteness')
    return bound

def load_routes(path=None,repo_routes=None):
    """Resolve CLI --routes semantics (issue #7).

    An explicitly given path that does not exist fails fast (server.py reads the file
    and would crash; the point is: never silently substitute defaults for an explicit
    request). The default resolves to the repository routes.json; only when that file
    is absent does it fall back to the public DEFAULT_ROUTES and return a recorded
    warning so the substitution stays visible in the output.
    """
    if path is not None:
        file=Path(path)
        if not file.is_file():raise FileNotFoundError(f'Routes file not found: {file}')
        return validate_routes(json.loads(file.read_text())),[]
    default=Path(repo_routes) if repo_routes is not None else Path(__file__).resolve().parent/'routes.json'
    if default.is_file():return validate_routes(json.loads(default.read_text())),[]
    return [dict(route) for route in validate_routes(DEFAULT_ROUTES)],['routes-file-missing-fell-back-to-public-default-routes']

def import_outcomes(tasks,records,routes,min_trials=3,min_success_lb=.9,confidence=.95):
    if type(min_trials) is bool or type(min_trials) is not int or min_trials<1:raise ValueError('Invalid outcome acceptance thresholds')
    for value in (min_success_lb,confidence):
        if type(value) is bool or type(value) not in (float,int) or not math.isfinite(value) or not 0<value<1:raise ValueError('Invalid outcome acceptance thresholds')
    routes=validate_routes(routes)
    route_index={(route['provider'],route['model']):route for route in routes}
    ids={row['id'] for row in tasks}
    grouped=collections.defaultdict(list);seen=set();executions=set();contexts={}
    for record in records:
        if record['task_id'] not in ids:raise ValueError('Unknown task id')
        pair=(record['provider'],record['model'])
        if pair not in route_index:raise ValueError('Unknown candidate route')
        if type(record.get('trial')) is not int:raise ValueError('Trial index must be an integer')
        if type(record['passed']) is not bool:raise ValueError('Acceptance verdict must be boolean')
        failure_type=record.get('failure_type')
        if record['passed']:
            if failure_type is not None:raise ValueError('A passed trial cannot carry a failure type')
        elif failure_type not in FAILURE_TYPES:raise ValueError("Failed trials require a failure type in {'timeout','acceptance','execution_error','other'}")
        if type(record.get('timeout')) is not bool:raise ValueError('Timeout flag must be boolean')
        if record['timeout'] and record['passed']:raise ValueError('A timed-out trial is a failed trial')
        execution_id=record.get('execution_id')
        if not isinstance(execution_id,str) or not execution_id.strip() or len(execution_id)>200:raise ValueError('Execution evidence requires an execution id')
        if execution_id in executions:raise ValueError('Duplicate execution id; a single execution cannot be counted twice')
        executions.add(execution_id)
        seed=record.get('seed')
        if seed is not None and not((type(seed) is int) or (isinstance(seed,str) and seed.strip() and len(seed)<=200)):raise ValueError('Invalid execution seed')
        randomness_policy=record.get('randomness_policy')
        if randomness_policy is not None and (not isinstance(randomness_policy,str) or not randomness_policy.strip() or len(randomness_policy)>200):raise ValueError('Invalid randomness policy')
        version=record.get('model_version')
        if version is not None and (not isinstance(version,str) or not version.strip() or len(version)>200):raise ValueError('Invalid measured model version')
        effort=record.get('reasoning_effort')
        if effort is not None and (not isinstance(effort,str) or not effort.strip() or len(effort)>200):raise ValueError('Invalid measured reasoning effort')
        if record.get('evidence_kind')!='acceptance_tests':raise ValueError('Task-success evidence requires acceptance tests/rubric, not a model preference')
        if record.get('simulated',False):raise ValueError('Simulated results cannot produce measured labels')
        for key in ['acceptance_spec_sha256','artifact_sha256','context_sha256']:
            if not re.fullmatch('[a-f0-9]{64}',record.get(key,'')):raise ValueError('Evidence requires acceptance, artifact, and context hashes')
        for key in ['cost','latency_ms','input_tokens','output_tokens','wall_time']:
            _metric(record,key)
        if contexts.setdefault(record['task_id'],record['context_sha256'])!=record['context_sha256']:raise ValueError('Context snapshot changed within a task')
        identity=(record['task_id'],*pair,record['trial'])
        if identity in seen:raise ValueError('Duplicate execution trial')
        seen.add(identity);grouped[record['task_id'],pair].append(record)
    labels=[];pending=[];stats=[];details=[]
    for task_id in sorted(ids):
        candidates=[];complete=True
        for pair,route in route_index.items():
            attempts=grouped.get((task_id,pair),[]);count=len(attempts)
            if count<min_trials:complete=False
            specs={r['acceptance_spec_sha256'] for r in attempts}
            if len(specs)>1:raise ValueError('Acceptance criteria changed across trials')
            if not attempts:continue  # n=0 candidate: bound is defined as 0 and completeness keeps it pending; never a crash
            efforts={r.get('reasoning_effort') for r in attempts};versions={r.get('model_version') for r in attempts}
            if len(efforts)>1:raise ValueError('Reasoning effort changed across trials; import separate operating points')
            if len(versions)>1:raise ValueError('Model version changed across trials; import separate operating points')
            effort=next(iter(efforts));version=next(iter(versions))
            if route.get('reasoning_effort') is not None and effort!=route['reasoning_effort']:raise ValueError('Reasoning effort differs from the declared route; import separate operating points')
            if route.get('model_version') is not None and version!=route['model_version']:raise ValueError('Model version differs from the declared route; import separate operating points')
            fingerprint=route_fingerprint(route,version,effort)
            successes=sum(1 for r in attempts if r['passed']);failures=count-successes
            failure_types=collections.Counter(r['failure_type'] for r in attempts if not r['passed'])
            mean_cost=sum(r['cost'] for r in attempts)/count;mean_latency_ms=sum(r['latency_ms'] for r in attempts)/count
            if not(math.isfinite(mean_cost) and math.isfinite(mean_latency_ms)):raise ValueError('Aggregated resource metrics must stay finite')
            lb=binomial_lower_bound(successes,count,confidence)
            row={'task_id':task_id,'provider':pair[0],'model':pair[1],'tier':route['tier'],'reasoning_effort':effort,'model_version':version,'route_fingerprint':fingerprint,'trials':count,'successes':successes,'failures':failures,'failure_types':dict(sorted(failure_types.items())),'success_rate':successes/count,'success_lb':lb,'confidence':confidence,'bound_method':BOUND_METHOD,'mean_cost':mean_cost,'mean_latency_ms':mean_latency_ms,'acceptance_spec_sha256':next(iter(specs))}
            stats.append(row)
            # The gate is the conservative lower bound, never the point estimate;
            # min_trials only counts trial completeness and is not a reliability gate.
            if count>=min_trials and lb>=min_success_lb:candidates.append(row)
        all_specs={r['acceptance_spec_sha256'] for (tid,_),attempts in grouped.items() if tid==task_id for r in attempts}
        if len(all_specs)>1:raise ValueError('Candidates were judged with different criteria')
        if not complete or not candidates:
            pending.append({'id':task_id,'reason':'incomplete-candidate-trials' if not complete else 'no-candidate-passed'});continue
        # Preregistered deterministic tie-break on all-trials means among gate passers;
        # deliberately NOT the highest lower bound (that optimizes the statistic, not the route).
        selected=min(candidates,key=lambda r:(r['mean_cost'],r['mean_latency_ms'],TIERS.index(r['tier'])))
        labels.append({'id':task_id,'tier':selected['tier'],'provider':selected['provider'],'model':selected['model'],'reasoning_effort':selected['reasoning_effort'],'model_version':selected['model_version'],'route_fingerprint':selected['route_fingerprint'],'label_source':'measured_candidate_outcomes','evidence':'acceptance_tests','success_rate':selected['success_rate'],'success_lb':selected['success_lb'],'confidence':confidence,'bound_method':BOUND_METHOD,'trials':selected['trials'],'acceptance_spec_sha256':selected['acceptance_spec_sha256'],'selection_basis':'preregistered deterministic tie-break (mean_cost, mean_latency_ms, tier order) over all-trials means among lower-bound passers; not highest-lower-bound'})
    for (task_id,pair),attempts in sorted(grouped.items()):
        if not attempts:continue  # guard: a defaultdict read may have created an empty group
        route=route_index[pair];versions={r.get('model_version') for r in attempts};efforts={r.get('reasoning_effort') for r in attempts}
        counts=collections.Counter(r['artifact_sha256'] for r in attempts)
        fingerprint=route_fingerprint(route,next(iter(versions)),next(iter(efforts)))
        for r in sorted(attempts,key=lambda r:r['trial']):
            # Identical artifact bytes across distinct executions are an audit hint only:
            # flag them, never reject, never discount the sample size (adjudication).
            audit=['duplicate-artifact-hash'] if counts[r['artifact_sha256']]>1 else []
            details.append({'task_id':task_id,'route_fingerprint':fingerprint,'trial':r['trial'],'passed':r['passed'],'failure_type':r.get('failure_type'),'cost':r['cost'],'latency_ms':r['latency_ms'],'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens'],'artifact_sha256':r['artifact_sha256'],'execution_id':r['execution_id'],'context_sha256':r['context_sha256'],'wall_time':r['wall_time'],'timeout':r['timeout'],'seed':r.get('seed'),'randomness_policy':r.get('randomness_policy'),'audit_flags':audit})
    alpha=1.0-confidence
    result={'schema_version':2,'labels':labels,'pending':pending,'candidate_metrics':stats,'trial_details':details,
        'minimum_trials':min_trials,'minimum_success_lb':min_success_lb,'confidence':confidence,
        'bound':{'method':BOUND_METHOD,'confidence':confidence,'alpha':alpha,'side':'lower','threshold':min_success_lb,
            'formula':'Exact one-sided Clopper-Pearson lower bound: L solves P(Bin(n,L)>=k)=alpha=(1-confidence); L(0,n)=0, L(n,n)=alpha**(1/n); computed with lgamma log-pmfs, term-ratio tail recursion, and 200 bisection steps (naive math.comb tails overflow float at n>=1030).',
            'coverage':'per-candidate marginal, no simultaneous guarantee',
            'note':'min_trials counts trial completeness only and is not a reliability gate; the gate acts on this lower bound, never the point estimate. Passing it does not certify a candidate as cheapest or safest overall; there is no simultaneous or multiplicity-adjusted guarantee across candidates.'},
        'preregistration':{'routes_fingerprint':routes_digest(routes),'rule':{'min_trials':min_trials,'min_success_lb':min_success_lb,'confidence':confidence},
            'statement':'The candidate set (this routes fingerprint), the per-candidate trial count n, and this decision rule must be fixed before execution: after one failure the gate can still be cleared by continuing (L(99,100)=0.9534401885>=0.9 at 95% one-sided), so growing n until the bound clears is sequential peeking, not confirmation.'},
        'estimand':'single-execution success rate of harness-independent executions on the fixed evaluated task set; not an extrapolation to new task populations and not a production-benefit claim',
        'note':'Small trial counts are preliminary measurements, not calibrated production reliability. Independence is a sourced assumption; audit flags mark repeated artifact hashes without causal judgment.'}
    json.dumps(result,allow_nan=False)  # fail closed: no NaN/Inf may ever leave this function
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--outcomes',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--routes',type=Path);p.add_argument('--min-trials',type=int,default=3);p.add_argument('--min-success-lb',type=float,default=.9);p.add_argument('--confidence',type=float,default=.95)
    a=p.parse_args()
    # --min-success-lb replaces the old --min-success-rate: the threshold now applies
    # to the one-sided lower bound, not the point estimate (semantic change, see issue #7).
    if a.min_trials<1 or not 0<a.min_success_lb<1 or not 0<a.confidence<1:raise SystemExit('Invalid outcome acceptance thresholds')
    try:routes,warnings=load_routes(a.routes)
    except FileNotFoundError as error:raise SystemExit(str(error))
    tasks=rows(a.tasks)
    if a.outcomes:
        result=import_outcomes(tasks,rows(a.outcomes),routes,a.min_trials,a.min_success_lb,a.confidence)
    else:
        # A plan is not execution evidence. Tasks need snapshots and acceptance specs.
        result={'status':'planned-not-executed','task_corpus_sha256':hashlib.sha256(a.tasks.read_bytes()).hexdigest(),'routes_fingerprint':routes_digest(routes),'candidates':routes,'required_trials_per_candidate':a.min_trials,'required_success_lb':a.min_success_lb,'required_confidence':a.confidence,'tasks':[{'id':r['id'],'prompt':r['prompt'],'required_context_snapshot':True,'required_acceptance_spec':True} for r in tasks]}
    if warnings:result['route_warnings']=warnings
    for warning in warnings:print('warning:',warning,file=sys.stderr)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('outcome import' if a.outcomes else 'execution plan',a.output)
