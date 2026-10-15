"""Independent task-schema / vision-policy composition; invented CPU ASGI only."""
import argparse, ast, asyncio, copy, hashlib, json, sys, tempfile, types, unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch
import task_label_schema as schema
from outcomes import route_fingerprint
from test_server_contract import ROOT, server, request, prediction

POLICIES=({}, {'vision_policy':'positive-v2','vision_positive_threshold':.4})
ROUTES=[{'tier':'balanced','provider':'synthetic','model':'text','vision':False},
    {'tier':'flagship','provider':'synthetic','model':'vision','vision':True,
     'model_version':'invented-v1','reasoning_effort':'declared-only','route_id':'invented'}]


def exercise(selected,options,gate,mode='diagnostic',mutation=None):
    async def run():
        with tempfile.TemporaryDirectory(prefix='labels18-policy-') as tmp,ExitStack() as stack:
            root=Path(tmp);product=root/'artifacts'/'invented'
            (product/'full-checkpoint').mkdir(parents=True)  # EMPTY mock product, no weights
            calibration={'temperatures':{'tier':1.,'vision':1.,'task_type':1.}}
            meta=schema.metadata(selected)
            if selected==schema.V2:calibration['label_schema']=copy.deepcopy(meta)
            if mutation:mutation(calibration,meta)
            (product/'calibration.json').write_text(json.dumps(calibration))
            if selected==schema.V2:(product/'label-schema.json').write_text(json.dumps(meta))
            (product/'quality-gate.json').write_text(json.dumps(gate))
            stack.enter_context(patch.object(server,'ROOT',root))
            loader=stack.enter_context(patch.object(server,'load_engine',return_value=object()))
            pred=prediction('balanced');pred['answers']['vision'].update(confidence=.8,probabilities={'yes':.2})
            scorer=stack.enter_context(patch.object(server,'score',return_value=pred))
            observation=stack.enter_context(patch.object(server,'record_observation'))
            app=server.create_app('invented',ROUTES,mode=mode,label_schema=selected,**options)
            try:
                async with app.router.lifespan_context(app):
                    health=await request(app,'/health')
                    response=await request(app,'/route',{'prompt':'Invented CPU composition only.'})
                return health,response,loader,scorer,observation
            except ValueError:
                loader.assert_not_called();scorer.assert_not_called();observation.assert_not_called()
                raise
    return asyncio.run(run())


class CompositionTests(unittest.TestCase):
    def test_both_task_versions_both_vision_policies_and_origin_fingerprint(self):
        original=copy.deepcopy(ROUTES)
        for selected in schema.VERSIONS:
            for options in POLICIES:
                with self.subTest(selected=selected,options=options):
                    health,response,loader,scorer,observed=exercise(selected,options,{'passed':True,'mode':'qualified'})
                    self.assertEqual(response[0],200);d=response[1]
                    self.assertEqual(health[1]['task_label_schema_version'],selected)
                    self.assertEqual(d['task_label_schema_version'],selected)
                    self.assertEqual(d['vision_policy'],options.get('vision_policy','legacy-v1'))
                    self.assertIs(d['execution_allowed'],False);self.assertIs(d['advisory_only'],True)
                    chosen=ROUTES[0] if options else ROUTES[1];r=d['recommendation']
                    self.assertEqual(r['model'],chosen['model'])
                    for key in ('route_id','model_version','reasoning_effort'):
                        if key in chosen:self.assertEqual(r[key],chosen[key])
                        else:self.assertNotIn(key,r)
                    self.assertEqual(route_fingerprint(chosen,r.get('model_version'),r.get('reasoning_effort')),
                        route_fingerprint(chosen,chosen.get('model_version'),chosen.get('reasoning_effort')))
                    record=observed.call_args.args[0]
                    self.assertEqual(record['task_label_schema_version'],selected)
                    self.assertEqual(record['reason_details'],d['reason_details'])
                    self.assertEqual(record['recommendation'],r)
                    self.assertEqual(record['prompt_sha256'],hashlib.sha256(b'Invented CPU composition only.').hexdigest())
                    self.assertNotIn('prompt',record)
                    self.assertEqual(loader.call_args.kwargs['schema_version'],selected)
                    self.assertEqual(scorer.call_count,1)
        self.assertEqual(ROUTES,original)

    def test_execution_literal_gate_and_advisory_survive_all_four_combinations(self):
        for selected in schema.VERSIONS:
            for options in POLICIES:
                for passed in (True,False,1,'true'):
                    with self.subTest(selected=selected,options=options,passed=passed):
                        _,response,_,scorer,observed=exercise(selected,options,{'passed':passed,'mode':'qualified'},'execution')
                        self.assertEqual(response[0],200 if passed is True else 503)
                        d=response[1] if passed is True else response[1]['detail']
                        self.assertIs(d['execution_allowed'],passed is True)
                        self.assertIs(d['advisory_only'],passed is not True)
                        self.assertEqual(scorer.call_count,int(passed is True))
                        self.assertEqual(observed.call_count,int(passed is True))

    def test_v2_typed_metadata_conflicts_reject_before_backend_and_scorer(self):
        def mutate(kind):
            def apply(calibration,meta):
                if kind=='hash':meta['question_sha256']='0'*64
                elif kind=='order':meta['classes']['task_type'].reverse()
                elif kind=='integer':meta['task_label_schema_version']=1
                elif kind=='boolean':meta['task_label_schema_version']=True
                elif kind=='other-version':meta['task_label_schema_version']=schema.LEGACY
                elif kind=='bare-source':calibration['label_schema_version']=1
                else:calibration.pop('label_schema')
            return apply
        for options in POLICIES:
            for kind in ('hash','order','integer','boolean','other-version','bare-source','missing'):
                with self.subTest(options=options,kind=kind),self.assertRaises(ValueError):
                    exercise(schema.V2,options,{'passed':True,'mode':'qualified'},mutation=mutate(kind))

    def test_keyword_contract_defaults_and_typed_config_refusal(self):
        import inspect
        parameters=inspect.signature(server.create_app).parameters
        for name in ('label_schema','vision_policy','vision_positive_threshold'):
            self.assertEqual(parameters[name].kind,inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters['label_schema'].default,schema.LEGACY)
        self.assertEqual(parameters['vision_policy'].default,'legacy-v1')
        self.assertIsNone(parameters['vision_positive_threshold'].default)
        for options in ({'label_schema':1},{'label_schema':True},{'vision_policy':'positive-v2'},
                {'vision_positive_threshold':.4},{'label_schema':schema.V2,'vision_policy':'positive-v2','vision_positive_threshold':True}):
            with self.subTest(options=options),patch.object(server,'load_engine') as loader,patch.object(server,'score') as scorer:
                with self.assertRaises(ValueError):server.create_app(**options)
                loader.assert_not_called();scorer.assert_not_called()

    def test_actual_cli_body_propagates_all_independent_options_without_server_spawn(self):
        tree=ast.parse((ROOT/'server.py').read_text())
        body=tree.body[-1].body;code=compile(ast.Module(body=body,type_ignores=[]),'actual-server-cli','exec')
        for selected in schema.VERSIONS:
            for options in POLICIES:
                argv=['server.py','--run','invented','--label-schema',selected]
                if options:argv+=['--vision-policy','positive-v2','--vision-positive-threshold','.4']
                creator=Mock(return_value=object());uvicorn=types.SimpleNamespace(run=Mock())
                env={**vars(server),'create_app':creator,'uvicorn':uvicorn}
                with patch.object(sys,'argv',argv):exec(code,env)
                self.assertEqual(creator.call_args.kwargs,{'label_schema':selected,
                    'vision_policy':options.get('vision_policy','legacy-v1'),
                    'vision_positive_threshold':options.get('vision_positive_threshold')})
                self.assertEqual(creator.call_args.args,('invented',server.DEFAULT_ROUTES,.9,False,'diagnostic'))
                self.assertEqual(uvicorn.run.call_args.kwargs,{'host':'127.0.0.1','port':19948,'log_level':'info'})
        creator=Mock();env={**vars(server),'create_app':creator,'uvicorn':types.SimpleNamespace(run=Mock())}
        with patch.object(sys,'argv',['server.py']):exec(code,env)
        self.assertEqual(creator.call_args.kwargs,{'label_schema':schema.LEGACY,'vision_policy':'legacy-v1','vision_positive_threshold':None})


if __name__=='__main__':unittest.main()
