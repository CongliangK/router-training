import json,math,re,subprocess,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
from extract import redact
from policy import select,DEFAULT_ROUTES,validate_routes
from outcomes import import_outcomes,binomial_lower_bound,load_routes
from synthetic_data import synthetic_splits, SPLIT_NAMES


def prediction(tier='fast',vision='no',confidence=.99,truncated=False):
    return {'answers':{'tier':{'choice':tier,'confidence':confidence},'vision':{'choice':vision,'confidence':.99}},'context_truncated':truncated}


def fixtures(trials=29):
    # 29 all-pass trials per candidate is the smallest n whose one-sided 95% exact
    # Clopper-Pearson lower bound reaches 0.9 (0.05**(1/29)=0.9018553723); the old
    # 3-trial fixture tops out at 0.3684031499 and must NOT produce a label.
    # Purely synthetic evidence; it mirrors no private distribution (Hard rule 1).
    records=[]
    for index,route in enumerate(DEFAULT_ROUTES):
        for trial in range(trials):
            records.append({'task_id':'task','provider':route['provider'],'model':route['model'],'trial':trial,'passed':True,'evidence_kind':'acceptance_tests','acceptance_spec_sha256':'a'*64,'artifact_sha256':f'{len(records):064x}','execution_id':f'exec-{index}-{trial}','context_sha256':'d'*64,'wall_time':1.,'timeout':False,'cost':index+1.,'latency_ms':10.,'input_tokens':20,'output_tokens':10})
    return records


class FrameworkTests(unittest.TestCase):
    def test_redaction_removes_credentials_and_coordination_suffix(self):
        text='token sk-abcdefghijk123456789\nYour parent agent id is "SECRET". Before you finish...'
        clean=redact(text);self.assertNotIn('abcdefghijk',clean);self.assertNotIn('SECRET',clean)
    def test_fast_text_recommends_fast(self):
        result=select(prediction(),DEFAULT_ROUTES);self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
    def test_actual_image_signal_cannot_be_overridden_by_classifier(self):
        result=select(prediction(),DEFAULT_ROUTES,has_image=True);self.assertTrue(result['needs_vision']);self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
    def test_hard_visual_uses_vision_capable_flagship(self):
        result=select(prediction('flagship','yes'),DEFAULT_ROUTES);self.assertEqual(result['recommendation']['model'],'gpt-6.1-sol')
    def test_absent_flagship_vision_still_refuses_unsafe_downgrade(self):
        routes=[{**route,'vision':False} if route['tier']=='flagship' else route for route in DEFAULT_ROUTES]
        result=select(prediction('flagship','yes'),routes);self.assertIsNone(result['recommendation'])
    def test_uncertain_visual_uses_vision_capable_route(self):
        result=select(prediction('fast','yes',confidence=.5),DEFAULT_ROUTES);self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
        routes=[{**route,'vision':False} if route['tier']=='fast' else route for route in DEFAULT_ROUTES]
        result=select(prediction('fast','yes',confidence=.5),routes);self.assertEqual(result['recommendation']['model'],'gpt-6.1-sol')
    def test_low_confidence_keeps_argmax_tier(self):
        result=select(prediction(confidence=.5),DEFAULT_ROUTES)
        self.assertEqual(result['required_tier'],'fast');self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
        self.assertIn('low-tier-confidence-effort-compensation-only',result['reasons'])
    def test_truncation_alone_escalates_to_flagship(self):
        result=select(prediction(truncated=True),DEFAULT_ROUTES);self.assertEqual(result['required_tier'],'flagship');self.assertEqual(result['recommendation']['model'],'gpt-6.1-sol')
        self.assertIn('context-truncated-flagship',result['reasons'])
    def test_measured_label_requires_complete_new_trials(self):
        result=import_outcomes([{'id':'task'}],fixtures(),DEFAULT_ROUTES);self.assertEqual(result['labels'][0]['tier'],'fast')
        result=import_outcomes([{'id':'task'}],fixtures(2),DEFAULT_ROUTES);self.assertEqual(result['labels'],[]);self.assertEqual(result['pending'][0]['reason'],'incomplete-candidate-trials')
    def test_three_all_pass_trials_do_not_clear_the_conservative_gate(self):
        result=import_outcomes([{'id':'task'}],fixtures(3),DEFAULT_ROUTES)
        self.assertEqual(result['labels'],[]);self.assertEqual(result['pending'][0]['reason'],'no-candidate-passed')
        for row in result['candidate_metrics']:self.assertAlmostEqual(row['success_lb'],0.3684031499,delta=1e-9)
    def test_twenty_eight_all_pass_trials_stay_pending(self):
        result=import_outcomes([{'id':'task'}],fixtures(28),DEFAULT_ROUTES)
        self.assertEqual(result['labels'],[]);self.assertEqual(result['pending'][0]['reason'],'no-candidate-passed')
        for row in result['candidate_metrics']:
            self.assertAlmostEqual(row['success_lb'],0.8985342644,delta=1e-9);self.assertLess(row['success_lb'],0.9)
    def test_forty_nine_of_fifty_with_one_timeout_clears_the_gate(self):
        records=fixtures(50);records[0].update(passed=False,failure_type='timeout',timeout=True)
        result=import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        self.assertEqual(result['labels'][0]['tier'],'fast')
        fast=next(row for row in result['candidate_metrics'] if row['tier']=='fast')
        self.assertEqual((fast['successes'],fast['failures']),(49,1));self.assertEqual(fast['failure_types'],{'timeout':1})
        self.assertAlmostEqual(fast['success_lb'],0.9086018693,delta=1e-9)
        failed=[detail for detail in result['trial_details'] if not detail['passed']]
        self.assertEqual(len(failed),1);self.assertEqual(failed[0]['failure_type'],'timeout');self.assertTrue(failed[0]['timeout'])
    def test_duplicate_trial_or_simulated_outcome_is_rejected(self):
        records=fixtures()
        with self.assertRaises(ValueError):import_outcomes([{'id':'task'}],records+[records[0]],DEFAULT_ROUTES)
        records[0]['simulated']=True
        with self.assertRaises(ValueError):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
    def test_duplicate_execution_id_is_rejected(self):
        records=fixtures();records[1]['execution_id']=records[0]['execution_id']
        with self.assertRaisesRegex(ValueError,'Duplicate execution id'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
    def test_failure_type_semantics_are_strict(self):
        records=fixtures();records[0]['passed']=False
        with self.assertRaisesRegex(ValueError,'failure type'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        records[0]['failure_type']='banana'
        with self.assertRaisesRegex(ValueError,'failure type'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        clean=fixtures();clean[0]['failure_type']='timeout'
        with self.assertRaisesRegex(ValueError,'passed trial'):import_outcomes([{'id':'task'}],clean,DEFAULT_ROUTES)
        clean=fixtures();del clean[0]['timeout']
        with self.assertRaisesRegex(ValueError,'Timeout flag'):import_outcomes([{'id':'task'}],clean,DEFAULT_ROUTES)
        clean=fixtures();clean[0]['timeout']=True
        with self.assertRaisesRegex(ValueError,'timed-out'):import_outcomes([{'id':'task'}],clean,DEFAULT_ROUTES)
    def test_measured_effort_is_preserved_and_mixed_trials_are_rejected(self):
        records=fixtures()
        for row in records:row['reasoning_effort']='low'
        result=import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        self.assertEqual(result['labels'][0]['reasoning_effort'],'low')
        self.assertTrue(all(row['reasoning_effort']=='low' for row in result['candidate_metrics']))
        records[1]['reasoning_effort']='high'
        with self.assertRaisesRegex(ValueError,'Reasoning effort changed'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        records[1]['reasoning_effort']=None
        with self.assertRaisesRegex(ValueError,'Reasoning effort changed'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        records[1]['reasoning_effort']=' '
        with self.assertRaisesRegex(ValueError,'Invalid measured'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
    def test_model_version_envelope_and_route_declarations(self):
        declared=[{**route,'model_version':'2026-10-01'} for route in DEFAULT_ROUTES]
        with self.assertRaisesRegex(ValueError,'differs from the declared route'):import_outcomes([{'id':'task'}],fixtures(),declared)
        records=fixtures()
        for row in records:row['model_version']='2026-10-01'
        result=import_outcomes([{'id':'task'}],records,declared)
        self.assertEqual(result['labels'][0]['model_version'],'2026-10-01')
        self.assertTrue(all(isinstance(row['route_fingerprint'],str) and re.fullmatch('[a-f0-9]{16}',row['route_fingerprint']) for row in result['candidate_metrics']))
        mixed=fixtures();mixed[0]['model_version']='v1';mixed[1]['model_version']='v2'
        with self.assertRaisesRegex(ValueError,'Model version changed'):import_outcomes([{'id':'task'}],mixed,DEFAULT_ROUTES)
        effort_declared=[{**route,'reasoning_effort':'high'} for route in DEFAULT_ROUTES]
        with self.assertRaisesRegex(ValueError,'Reasoning effort differs from the declared route'):import_outcomes([{'id':'task'}],fixtures(),effort_declared)
    def test_changed_acceptance_criteria_is_rejected(self):
        records=fixtures();records[1]['acceptance_spec_sha256']='c'*64
        with self.assertRaises(ValueError):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
    def test_context_snapshot_must_be_stable_within_a_task(self):
        records=fixtures();records[5]['context_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'Context snapshot changed'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
    def test_identical_artifact_hash_is_audited_not_rejected(self):
        records=fixtures();records[0]['artifact_sha256']=records[1]['artifact_sha256']='c'*64
        result=import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        self.assertEqual(result['labels'][0]['tier'],'fast')
        flagged={detail['trial'] for detail in result['trial_details'] if 'duplicate-artifact-hash' in detail['audit_flags']}
        self.assertEqual(flagged,{0,1})
    def test_dynamic_routes_reevaluation_without_code_change(self):
        routes=[{'tier':'fast','provider':'prov-a','model':'model-a','vision':True,'model_version':'vA','route_id':'a'},
                {'tier':'flagship','provider':'prov-b','model':'model-b','vision':True}]
        records=[]
        for index,route in enumerate(routes):
            for trial in range(29):
                records.append({'task_id':'task','provider':route['provider'],'model':route['model'],'model_version':route.get('model_version'),'trial':trial,'passed':True,'evidence_kind':'acceptance_tests','acceptance_spec_sha256':'a'*64,'artifact_sha256':f'{index*1000+trial:064x}','execution_id':f'e-{index}-{trial}','context_sha256':'d'*64,'wall_time':1.,'timeout':False,'cost':index+1.,'latency_ms':10.,'input_tokens':20,'output_tokens':10})
        result=import_outcomes([{'id':'task'}],records,routes)
        self.assertEqual(result['labels'][0]['provider'],'prov-a');self.assertEqual(result['labels'][0]['model_version'],'vA')
        self.assertEqual({row['provider'] for row in result['candidate_metrics']},{'prov-a','prov-b'})
        with self.assertRaisesRegex(ValueError,'Unknown candidate route'):import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
    def test_routes_loading_default_fallback_and_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            custom=[{'tier':'fast','provider':'p','model':'m','vision':False}]
            (root/'routes.json').write_text(json.dumps(custom))
            self.assertEqual(load_routes(root/'routes.json'),(custom,[]))
            self.assertEqual(load_routes(None,repo_routes=root/'routes.json'),(custom,[]))
            routes,warnings=load_routes(None,repo_routes=root/'missing.json')
            self.assertEqual(routes,DEFAULT_ROUTES);self.assertEqual(warnings,['routes-file-missing-fell-back-to-public-default-routes'])
            self.assertIsNot(routes,DEFAULT_ROUTES)
            with self.assertRaises(FileNotFoundError):load_routes(root/'missing.json')
            (root/'dup.json').write_text(json.dumps([{'tier':'fast','provider':'p','model':'m','vision':False},{'tier':'fast','provider':'p','model':'m','vision':True}]))
            with self.assertRaisesRegex(ValueError,'Duplicate route declaration'):load_routes(root/'dup.json')
    def test_validate_routes_optional_metadata(self):
        route={'tier':'fast','provider':'p','model':'m','vision':False,'model_version':'v1','reasoning_effort':'high','route_id':'r1','unknown_key':'tolerated'}
        self.assertEqual(validate_routes([route]),[route])
        for key in ['model_version','reasoning_effort','route_id']:
            for bad in ['',5,True,'x'*201]:
                with self.assertRaises(ValueError):validate_routes([{**route,key:bad}])
            self.assertEqual(validate_routes([{**route,key:None}]),[{**route,key:None}])
        with self.assertRaisesRegex(ValueError,'Duplicate route declaration'):validate_routes([route,{**route,'route_id':'r2','vision':True}])
        with self.assertRaisesRegex(ValueError,'Duplicate route id'):validate_routes([route,{**route,'provider':'q'}])
    def test_binomial_lower_bound_reference_table(self):
        self.assertEqual(binomial_lower_bound(0,0),0.0);self.assertEqual(binomial_lower_bound(0,29),0.0)
        self.assertAlmostEqual(binomial_lower_bound(3,3),0.3684031499,delta=1e-9)
        self.assertAlmostEqual(binomial_lower_bound(28,28),0.8985342644,delta=1e-9)
        self.assertAlmostEqual(binomial_lower_bound(29,29),0.9018553723,delta=1e-9)
        self.assertAlmostEqual(binomial_lower_bound(49,50),0.9086018693,delta=1e-9)
        self.assertAlmostEqual(binomial_lower_bound(99,100),0.9534401885,delta=1e-9)
        self.assertTrue(binomial_lower_bound(28,28)<0.9<=binomial_lower_bound(29,29))
        values=[binomial_lower_bound(k,10) for k in range(11)];self.assertEqual(values,sorted(values))
        for k,n in [(True,3),(3,True),(3.0,3),(3,3.0),(-1,3),(4,3)]:
            with self.assertRaises(ValueError):binomial_lower_bound(k,n)
        for confidence in [0.0,1.0,True]:
            with self.assertRaises(ValueError):binomial_lower_bound(3,3,confidence=confidence)
    def test_binomial_lower_bound_is_stable_at_large_n(self):
        # n>=1030 is exactly where float(math.comb(n,n//2)) overflows (red-team probe);
        # the lgamma+recursion implementation must stay finite and match the reference values.
        self.assertAlmostEqual(binomial_lower_bound(515,1030),0.4739127365,delta=1e-9)
        self.assertAlmostEqual(binomial_lower_bound(600,1200),0.4758609796,delta=1e-9)
        self.assertAlmostEqual(binomial_lower_bound(1030,1030),0.9970957476,delta=1e-9)
    def test_binomial_lower_bound_underflow_crash_samples(self):
        # Red-team round 1 crash probes: the old survival() seeded the tail recursion
        # at k with math.exp(log-pmf), which underflowed to 0.0 whenever |k-n*m|>>sqrt(n)
        # at a bisection midpoint (the first probe is m=0.5, so any k far from n/2 hit
        # it); every ratio-multiplied term stayed 0.0, the comparison took the wrong
        # branch, lo climbed to exactly 1.0 and math.log1p(-1.0) raised ValueError.
        # All ten must return the exact bound (L(18,1671)=0.006973702932 was the
        # red-team truth; clamping p to nextafter(1,0) instead returns ~1.0 here --
        # anti-conservative -- and is forbidden).
        expected={(1,5000):1.0258606257579484e-05,(10,100000):5.425502668687073e-05,
            (18,1671):0.006973702932312538,(122,1938):0.054114311616993926,
            (968,4525):0.20392346320147764,(1000,10000):0.09510201345758967,
            (5100,10000):0.5017259958859535,(11000,20000):0.5441836121635442,
            (100000,1000000):0.09950689981835192,(900000,1000000):0.8995051880047098}
        for (k,n),value in sorted(expected.items()):
            with self.subTest(k=k,n=n):self.assertAlmostEqual(binomial_lower_bound(k,n),value,delta=1e-9)
    def test_binomial_lower_bound_cross_domain_grid(self):
        # n from 1e3 to 1e6 across k/n from 1% to 99%: exercises both failure regimes
        # of the old seed-at-k recursion (k far below the n*0.5 probe mean, and k far
        # above it). Reference values were recomputed offline with an independent
        # Lentz continued-fraction implementation of the regularized incomplete beta
        # I_p(k,n-k+1) (Numerical Recipes 6.4 form; P(Bin(n,p)>=k)=I_p(k,n-k+1)),
        # itself cross-validated against exact integer bisection (worst |diff|
        # 2.2e-15 for n<=1030) and the closed forms alpha**(1/n) / 1-(1-alpha)**(1/n).
        grid=[(10,1000,0.005435140144707218),(50,1000,0.03916353592559227),(100,1000,0.0847847676684707),
            (200,1000,0.17936802936486268),(300,1000,0.27612277115901307),(400,1000,0.3742608818457236),
            (490,1000,0.4635447859344053),(500,1000,0.4735177312356933),(600,1000,0.5738234133353124),
            (800,1000,0.7780485668004171),(900,1000,0.8830084679036372),(990,1000,0.9830968248794381),
            (100,10000,0.008420198936744627),(500,10000,0.04646251581727112),(1000,10000,0.09510201345758967),
            (2000,10000,0.19343573137648307),(3000,10000,0.29245606166578675),(4000,10000,0.39191417088439406),
            (4900,10000,0.4817303023091203),(5000,10000,0.4917265044037602),(6000,10000,0.5918710999575174),
            (8000,10000,0.79330629306576),(9000,10000,0.894928883412784),(9900,10000,0.9882027660286605),
            (1000,100000,0.009487997257329153),(5000,100000,0.048871018372585645),(10000,100000,0.09844313756279766),
            (20000,100000,0.19792085021877326),(30000,100000,0.2976156858085911),(40000,100000,0.3974489691870199),
            (49000,100000,0.487395016670811),(50000,100000,0.4973942822602452),(60000,100000,0.5974446845651196),
            (80000,100000,0.7979079892027103),(90000,100000,0.8984259722276746),(99000,100000,0.9894667990993229),
            (10000,1000000,0.009836888398168375),(50000,1000000,0.04964197535683336),(100000,1000000,0.09950689981835192),
            (200000,1000000,0.19934220069112008),(300000,1000000,0.2992461617633737),(400000,1000000,0.39919390418324086),
            (490000,1000000,0.4891772598379899),(500000,1000000,0.4991770739496548),(600000,1000000,0.5991934764421705),
            (800000,1000000,0.7993409172422248),(900000,1000000,0.8995051880047098),(990000,1000000,0.9898347860179992)]
        for k,n,value in grid:
            with self.subTest(k=k,n=n):self.assertAlmostEqual(binomial_lower_bound(k,n),value,delta=1e-9)
    def test_binomial_lower_bound_matches_exact_integer_bisection(self):
        # Live independent cross-check inside the suite: dyadic bisection on
        # p=j/2**80 where every survival evaluation is a pure big-integer tail sum
        # (math.comb(n,i)*j**i*(2**80-j)**(n-i)) compared exactly against
        # 2**(80n)/20 -- no floating point until the final conversion. Any
        # divergence beyond bisection+float noise (>1e-9) means the lgamma/recursion
        # path changed semantics, not just numerics.
        for k,n in [(2,29),(9,43),(17,57)]:
            lo,hi=0,1<<80
            for _ in range(80):
                mid=(lo+hi)//2
                tail=sum(math.comb(n,i)*mid**i*((1<<80)-mid)**(n-i) for i in range(k,n+1))
                if 20*tail>=1<<(80*n):hi=mid
                else:lo=mid
            self.assertAlmostEqual(binomial_lower_bound(k,n),hi/2.0**80,delta=1e-9)
    def test_seed_and_randomness_policy_validation_and_envelope(self):
        # M9 red-team blind spot: deleting the seed/randomness_policy validation used
        # to leave the suite green. Valid values must pass through into the audit
        # envelope; invalid ones must fail closed.
        records=fixtures()
        for row in records:
            if row['trial']==0:row['seed']=42
            if row['trial']==1:row['randomness_policy']='provider-seeded'
            if row['trial']==2:row['seed']='run-2026'
        result=import_outcomes([{'id':'task'}],records,DEFAULT_ROUTES)
        seeded=[detail for detail in result['trial_details'] if detail['seed']==42]
        policy=[detail for detail in result['trial_details'] if detail['randomness_policy']=='provider-seeded']
        strseed=[detail for detail in result['trial_details'] if detail['seed']=='run-2026']
        unset=[detail for detail in result['trial_details'] if detail['trial']>2]
        self.assertEqual({detail['trial'] for detail in seeded},{0});self.assertEqual(len(seeded),3)
        self.assertEqual({detail['trial'] for detail in policy},{1});self.assertEqual(len(policy),3)
        self.assertEqual({detail['trial'] for detail in strseed},{2});self.assertEqual(len(strseed),3)
        self.assertTrue(all(detail['seed'] is None and detail['randomness_policy'] is None for detail in unset))
        for bad in [True,3.5,'','   ','x'*201]:
            bad_records=fixtures();bad_records[3]['seed']=bad
            with self.assertRaisesRegex(ValueError,'Invalid execution seed'):import_outcomes([{'id':'task'}],bad_records,DEFAULT_ROUTES)
        for bad in [5,True,'','   ','y'*201]:
            bad_records=fixtures();bad_records[3]['randomness_policy']=bad
            with self.assertRaisesRegex(ValueError,'Invalid randomness policy'):import_outcomes([{'id':'task'}],bad_records,DEFAULT_ROUTES)
    def test_zero_trial_candidate_is_pending_not_crashing(self):
        result=import_outcomes([{'id':'task'}],[],DEFAULT_ROUTES)
        self.assertEqual(result['labels'],[]);self.assertEqual(result['pending'],[{'id':'task','reason':'incomplete-candidate-trials'}])
    def test_output_schema_v2_serializes_without_nan(self):
        result=import_outcomes([{'id':'task'}],fixtures(),DEFAULT_ROUTES)
        json.dumps(result,allow_nan=False)
        self.assertEqual(result['schema_version'],2);self.assertEqual(len(result['trial_details']),87)
        self.assertEqual(result['bound']['method'],'clopper_pearson_exact_one_sided_lower')
        self.assertEqual(result['bound']['coverage'],'per-candidate marginal, no simultaneous guarantee')
        self.assertEqual(result['bound']['side'],'lower');self.assertEqual(result['bound']['threshold'],.9)
        # alpha is recorded (display-only) without the 1.0-0.95 float residue: exactly 0.05.
        self.assertEqual(result['bound']['alpha'],.05)
        self.assertEqual(result['preregistration']['rule'],{'min_trials':3,'min_success_lb':.9,'confidence':.95})
        self.assertTrue(re.fullmatch('[a-f0-9]{64}',result['preregistration']['routes_fingerprint']))
        self.assertIn('estimand',result)
        required={'task_id','route_fingerprint','trial','passed','failure_type','cost','latency_ms','input_tokens','output_tokens','artifact_sha256','execution_id','audit_flags'}
        self.assertTrue(required<=set(result['trial_details'][0]))
        self.assertEqual(len({detail['execution_id'] for detail in result['trial_details']}),87)
        self.assertIn('mean_cost',result['labels'][0]['selection_basis'])
    def test_generated_split_groups_and_ids_are_disjoint(self):
        tasks,splits,report=synthetic_splits()
        for a,b in [('train','calibration'),('train','test'),('calibration','test')]:
            self.assertFalse({r['id'] for r in splits[a]}&{r['id'] for r in splits[b]})
            self.assertFalse({r['split_group'] for r in splits[a]}&{r['split_group'] for r in splits[b]})
        self.assertEqual({name:len(rows) for name,rows in splits.items()}, {'train':80,'calibration':20,'test':20})
        self.assertEqual(sum(map(len,splits.values())),120)
        rows=[r for split in splits.values() for r in split]
        self.assertEqual({r['id'] for r in rows},{r['id'] for r in tasks})
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(report['independent_groups'],12)
        self.assertTrue(report['group_disjoint'])
        self.assertEqual(report['cross_root_near_duplicate_edges'],0)
        for name in SPLIT_NAMES:
            self.assertEqual(report['splits'][name]['rows'],len(splits[name]))
            self.assertEqual(set(report['splits'][name]['groups']),{r['split_group'] for r in splits[name]})
        self.assertTrue(all(r['synthetic'] for r in rows))

    @unittest.skipUnless(all((Path(__file__).resolve().parents[1]/'data'/f'{name}.jsonl').is_file() for name in SPLIT_NAMES),
                         'Optional private archive validation: real split files are not distributed')
    def test_actual_split_groups_and_ids_are_disjoint(self):
        root=Path(__file__).resolve().parents[1]/'data';splits={name:[json.loads(s) for s in (root/f'{name}.jsonl').read_text().splitlines()] for name in ['train','calibration','test']}
        for a,b in [('train','calibration'),('train','test'),('calibration','test')]:
            self.assertFalse({r['id'] for r in splits[a]}&{r['id'] for r in splits[b]})
            self.assertFalse({r['split_group'] for r in splits[a]}&{r['split_group'] for r in splits[b]})
        self.assertEqual({name:len(rows) for name,rows in splits.items()}, {'train':469,'calibration':95,'test':104})
        self.assertEqual(sum(map(len,splits.values())),668)
    def test_quality_gate_does_not_pass_from_expert_imitation(self):
        gate=json.loads((Path(__file__).resolve().parents[1]/'artifacts/real-history-v3c/quality-gate.json').read_text());self.assertFalse(gate['passed']);self.assertEqual(gate['mode'],'shadow')

    def test_cli_explicit_missing_routes_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tasks=root/'tasks.jsonl';tasks.write_text(json.dumps({'id':'t','prompt':'p'})+'\n')
            out=root/'out.json'
            process=subprocess.run([sys.executable,str(ROOT/'outcomes.py'),'--tasks',str(tasks),'--output',str(out),'--routes',str(root/'nope.json')],capture_output=True,text=True)
            self.assertNotEqual(process.returncode,0);self.assertIn('not found',process.stderr);self.assertFalse(out.exists())
    def test_cli_dynamic_routes_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'routes.json').write_text(json.dumps([{'tier':'fast','provider':'p1','model':'m1','vision':False}]))
            (root/'tasks.jsonl').write_text(json.dumps({'id':'t','prompt':'p'})+'\n')
            records=[{'task_id':'t','provider':'p1','model':'m1','trial':t,'passed':True,'evidence_kind':'acceptance_tests','acceptance_spec_sha256':'a'*64,'artifact_sha256':f'{t:064x}','execution_id':f'e-{t}','context_sha256':'d'*64,'wall_time':1.,'timeout':False,'cost':1.,'latency_ms':5.,'input_tokens':9,'output_tokens':9} for t in range(29)]
            (root/'outcomes.jsonl').write_text('\n'.join(json.dumps(record) for record in records)+'\n')
            out=root/'labels.json'
            process=subprocess.run([sys.executable,str(ROOT/'outcomes.py'),'--tasks',str(root/'tasks.jsonl'),'--outcomes',str(root/'outcomes.jsonl'),'--output',str(out),'--routes',str(root/'routes.json')],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            data=json.loads(out.read_text())
            self.assertEqual(data['labels'][0]['provider'],'p1');self.assertEqual(len(data['trial_details']),29)
            json.dumps(data,allow_nan=False)
    def test_cli_default_reads_repository_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'tasks.jsonl').write_text(json.dumps({'id':'task','prompt':'p'})+'\n')
            (root/'outcomes.jsonl').write_text('\n'.join(json.dumps(record) for record in fixtures())+'\n')
            out=root/'labels.json'
            process=subprocess.run([sys.executable,str(ROOT/'outcomes.py'),'--tasks',str(root/'tasks.jsonl'),'--outcomes',str(root/'outcomes.jsonl'),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            data=json.loads(out.read_text())
            self.assertEqual(data['labels'][0]['tier'],'fast');self.assertEqual(data['labels'][0]['model'],'glm-5.3-flash')

if __name__=='__main__':unittest.main()
