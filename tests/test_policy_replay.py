"""Pure synthetic policy/replay tests; no models, real data, or backend calls."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy import DEFAULT_ROUTES,select,validate_routes
from policy_replay import synthetic_rows,replay,study
from outcomes import route_fingerprint,import_outcomes


def prediction(p=.02,tier='balanced',confidence=.99,truncated=False):
    return {'answers':{'tier':{'choice':tier,'confidence':confidence},
        'vision':{'choice':'yes' if p>=.5 else 'no','confidence':max(p,1-p),'probabilities':{'yes':p,'no':1-p}}},'context_truncated':truncated}


def positive(pred,routes=DEFAULT_ROUTES,**kwargs):
    return select(pred,routes,vision_policy='positive-v2',vision_positive_threshold=.35,**kwargs)


class PolicyReplayTests(unittest.TestCase):
    def test_explicit_threshold_boundary_and_high_confidence_no(self):
        for p,expected in [(.02,False),(.349999,False),(.35,True),(.350001,True),(.99,True)]:
            d=positive(prediction(p));self.assertIs(d['needs_vision'],expected)
            self.assertEqual(d['selected_tier'],'flagship' if expected else 'balanced')
        for threshold in [0,.2,.8,1]:
            d=select(prediction(threshold),DEFAULT_ROUTES,vision_policy='positive-v2',vision_positive_threshold=threshold)
            self.assertTrue(d['needs_vision'])

    def test_default_conservative_behavior_and_version_migration(self):
        p=prediction(.2)
        self.assertTrue(select(p,DEFAULT_ROUTES)['needs_vision'])
        self.assertFalse(positive(p)['needs_vision'])
        self.assertFalse(select(prediction(.01),DEFAULT_ROUTES)['needs_vision'])
        for options in [{'vision_positive_threshold':.35},{'vision_policy':'positive-v2'},
                {'vision_policy':'v3'},{'minimum_confidence':True}]:
            with self.assertRaises(ValueError):select(p,DEFAULT_ROUTES,**options)
        for threshold in [True,None,-.01,1.01,float('nan'),float('inf'),'0.35']:
            with self.assertRaises(ValueError):select(p,DEFAULT_ROUTES,vision_policy='positive-v2',vision_positive_threshold=threshold)

    def test_images_force_vision_at_every_threshold(self):
        for threshold in [0,.35,1]:
            d=select(prediction(0),DEFAULT_ROUTES,True,vision_policy='positive-v2',vision_positive_threshold=threshold)
            self.assertTrue(d['needs_vision']);self.assertEqual(d['selected_tier'],'flagship')
        for invalid in [None,0,1,'false']:
            with self.assertRaises(ValueError):positive(prediction(),has_image=invalid)

    def test_missing_invalid_probability_always_requires_vision(self):
        for container in [None,{},[],{'yes':None},{'yes':True},{'yes':-.1},{'yes':1.1},
                {'yes':'0.1'},{'yes':float('nan')},{'yes':float('inf')},{'yes':10**1000}]:
            for version in ['legacy-v1','positive-v2']:
                p=prediction();p['answers']['vision']['probabilities']=container
                d=select(p,DEFAULT_ROUTES,vision_policy=version,vision_positive_threshold=.35 if version=='positive-v2' else None)
                self.assertTrue(d['needs_vision']);self.assertIsNone(d['vision_probability'])
                self.assertEqual(d['selected_tier'],'flagship');json.dumps(d,allow_nan=False)
        p=prediction();del p['answers']['vision']
        self.assertTrue(positive(p)['needs_vision'])

    def test_synthetic_balanced_vision_removes_only_capability_upgrade(self):
        routes=[{**r,'vision':True} if r['tier']=='balanced' else dict(r) for r in DEFAULT_ROUTES]
        before=positive(prediction(.9));after=positive(prediction(.9),routes)
        self.assertEqual(before['selected_tier'],'flagship');self.assertEqual(after['selected_tier'],'balanced')
        self.assertIn('vision_forced',[e['category'] for e in before['reason_details']])
        self.assertNotIn('vision_forced',[e['category'] for e in after['reason_details']])
        self.assertEqual(DEFAULT_ROUTES[1]['vision'],False)

    def test_multireasons_stable_tiers_refusal_and_truncation(self):
        p=prediction(.2,confidence=.4,truncated=True)
        d=select(p,DEFAULT_ROUTES)
        self.assertEqual((d['model_tier'],d['required_tier'],d['selected_tier']),('balanced','flagship','flagship'))
        self.assertEqual([e['category'] for e in d['reason_details']],['model_choice','context_truncation','risk_policy','risk_policy'])
        self.assertFalse(d['reason_details'][2]['compensation_applied'])
        self.assertEqual(d,select(p,DEFAULT_ROUTES))
        for routes in [[],[{'tier':'balanced','provider':'synthetic','model':'text','vision':False}]]:
            d=positive(prediction(.9),routes)
            self.assertIsNone(d['recommendation']);self.assertIsNone(d['selected_tier'])
            self.assertEqual(d['model_tier'],d['required_tier']);self.assertTrue(d['reason_details'])
        # A missing tier is not misreported as a vision-forced upgrade.
        d=positive(prediction(.9),[DEFAULT_ROUTES[2]])
        self.assertNotIn('vision_forced',[e['category'] for e in d['reason_details']])
        self.assertIn('authorized-tier-unavailable',[e['code'] for e in d['reason_details']])

    def test_unique_identity_and_declared_effort_only(self):
        route={'tier':'balanced','provider':'synthetic','model':'one','vision':True,
            'reasoning_effort':'not-verified','route_id':'r1','model_version':'v1'}
        d=positive(prediction(.9),[route]);r=d['recommendation']
        self.assertEqual(r,{k:route[k] for k in ['provider','model','reasoning_effort','route_id','model_version']})
        self.assertEqual(d['recommendation_contract_version'],1)
        self.assertEqual(route_fingerprint(route,route['model_version'],route['reasoning_effort']),
                         route_fingerprint({**route,'route_id':'alias'},r['model_version'],r['reasoning_effort']))
        with self.assertRaises(ValueError):validate_routes([route,{**route,'route_id':'r2','reasoning_effort':'max'}])
        with self.assertRaises(ValueError):select(prediction(),[route,dict(route)])
        self.assertEqual(import_outcomes([{'id':'synthetic'}],[],[route])['labels'],[])

    def test_replay_denominators_sensitivity_simulation_and_cost(self):
        rows=synthetic_rows();original=copy.deepcopy(DEFAULT_ROUTES)
        result=study(rows,DEFAULT_ROUTES,vision_positive_threshold=.35)
        self.assertEqual(DEFAULT_ROUTES,original);self.assertEqual(result['input_kind'],'local-unverified')
        self.assertEqual([r['minimum_confidence'] for r in result['legacy_confidence_sensitivity']],[.8,.85,.9,.95])
        self.assertGreater(result['simulated_upgrade_rate_reduction'],0)
        self.assertEqual(result['baseline']['vision_false_negatives'],0)
        self.assertGreater(result['positive_threshold_experiment']['vision_false_negatives'],0)
        self.assertIsNone(result['baseline']['estimated_total_cost'])
        costs=[{'provider':r['provider'],'model':r['model'],'cost':1} for r in DEFAULT_ROUTES]
        full=replay(rows,DEFAULT_ROUTES,costs=costs)
        self.assertEqual(full['estimated_total_cost'],len(rows))
        self.assertIsNone(replay(rows,DEFAULT_ROUTES,costs=costs[:1])['estimated_total_cost'])
        empty=replay([],DEFAULT_ROUTES);self.assertIsNone(empty['upgrade_rate']);self.assertIsNone(empty['vision_false_negative_rate'])
        refused=replay(rows,[]);self.assertEqual(refused['refusal_rate'],1);self.assertEqual(refused['vision_refusal_rate'],1)
        self.assertEqual(sum(refused['upgrade_reason_counts'].values()),0)
        with self.assertRaises(ValueError):replay([{'prediction':prediction()}],DEFAULT_ROUTES)

    def test_cli_synthetic_and_local_inputs_aggregate_only_no_overwrite(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'local.jsonl';source.write_text('\n'.join(json.dumps({**r,'prompt':'SYNTHETIC-SENTINEL'}) for r in synthetic_rows()))
            for args in [['--synthetic'],['--input',str(source)]]:
                output=Path(directory)/('synthetic.json' if args[0]=='--synthetic' else 'local.json')
                command=[sys.executable,str(root/'policy_replay.py'),*args,'--output',str(output),'--vision-positive-threshold','.35']
                run=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
                text=output.read_text();self.assertNotIn('SYNTHETIC-SENTINEL',text)
                data=json.loads(text);self.assertFalse(data['execution_performed'])
                self.assertEqual(data['input_kind'],'synthetic' if args[0]=='--synthetic' else 'local-unverified')
                self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
