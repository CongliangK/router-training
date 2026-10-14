"""Exercise the real ASGI contract with synthetic files and a mocked model backend."""
import asyncio, importlib.util, json, sys, tempfile, types, unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Import server code without importing torch/peft or loading any model weights.
backend=types.ModuleType('model_utils')
backend.ROOT=ROOT;backend.load_engine=lambda **kwargs:None;backend.score=lambda *args:None
spec=importlib.util.spec_from_file_location('_server_contract_under_test',ROOT/'server.py')
server=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'model_utils':backend,spec.name:server}):spec.loader.exec_module(server)


def prediction(tier='fast',vision='no'):
    return {'answers':{'tier':{'choice':tier,'confidence':.99},
                      'vision':{'choice':vision,'confidence':.99,'probabilities':{'yes':.01}},
                      'task_type':{'choice':'implementation','confidence':.99}},
            'context_truncated':False}


async def request(app,path,body=None):
    sent=[];received=False
    async def receive():
        nonlocal received
        if received:return {'type':'http.disconnect'}
        received=True
        return {'type':'http.request','body':json.dumps(body).encode() if body is not None else b'','more_body':False}
    async def send(message):sent.append(message)
    scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','scheme':'http',
           'method':'POST' if body is not None else 'GET','path':path,'raw_path':path.encode(),
           'query_string':b'','headers':[(b'content-type',b'application/json')],
           'client':('127.0.0.1',1234),'server':('test',80),'root_path':''}
    await app(scope,receive,send)
    status=next(message['status'] for message in sent if message['type']=='http.response.start')
    payload=b''.join(message.get('body',b'') for message in sent if message['type']=='http.response.body')
    return status,json.loads(payload)


def exercise(gate,mode='diagnostic',pred=None,routes=None):
    async def run():
        with tempfile.TemporaryDirectory(prefix='router-contract-') as directory,ExitStack() as stack:
            root=Path(directory);artifacts=root/'artifacts'/'synthetic-run'
            (artifacts/'full-checkpoint').mkdir(parents=True)
            (artifacts/'calibration.json').write_text(json.dumps({'temperatures':{'tier':1,'vision':1,'task_type':1}}))
            if gate is not None:
                (artifacts/'quality-gate.json').write_text(gate if isinstance(gate,str) else json.dumps(gate))
            stack.enter_context(patch.object(server,'ROOT',root))
            loader=stack.enter_context(patch.object(server,'load_engine',return_value=object()))
            scorer=stack.enter_context(patch.object(server,'score',return_value=pred or prediction()))
            observation=stack.enter_context(patch.object(server,'record_observation'))
            app=server.create_app('synthetic-run',routes=routes,mode=mode)
            async with app.router.lifespan_context(app):
                health=await request(app,'/health')
                result=await request(app,'/route',{'prompt':'Synthetic contract exercise.'})
                health_after=await request(app,'/health')
            unloaded=await request(app,'/health')
            assert not (root/'observations').exists()
            return health,result,health_after,unloaded,scorer.call_count,observation.call_args_list,loader.call_count
    return asyncio.run(run())


class ServerContractTests(unittest.TestCase):
    def assert_permission(self,payload,allowed,mode):
        self.assertEqual(payload['schema_version'],2)
        self.assertEqual(payload['mode'],mode)
        self.assertIs(payload['execution_allowed'],allowed)
        self.assertIs(payload['advisory_only'],not allowed)

    def test_diagnostic_is_advisory_even_for_qualified_gate(self):
        for gate in [{'passed':False,'mode':'shadow'},{'passed':True,'mode':'qualified'}]:
            with self.subTest(gate=gate):
                health,result,after,unloaded,calls,records,loads=exercise(gate)
                self.assertEqual(health[0],200);self.assertTrue(health[1]['ready'])
                self.assertEqual(result[0],200);self.assert_permission(result[1],False,'diagnostic')
                self.assert_permission(health[1],False,'diagnostic')
                self.assertEqual(health[1],after[1]);self.assertFalse(unloaded[1]['ready'])
                self.assertIs(result[1]['recommendation']['advisory_only'],True)
                self.assertIs(result[1]['recommendation']['execution_allowed'],False)
                self.assertEqual(result[1]['quality_gate_passed'],gate['passed'])
                self.assertEqual((calls,len(records),loads),(1,1,1))
                self.assertNotIn('prompt',records[0].args[0])

    def test_execution_requires_qualified_gate_before_scoring(self):
        for gate in [{'passed':False,'mode':'shadow'},None,'{broken',[],
                     {'passed':'true','mode':'qualified'},{'passed':1,'mode':'qualified'},
                     {'passed':True,'mode':'shadow'},{'passed':True,'mode':'unexpected'},
                     {'passed':True},{'mode':'qualified'}]:
            with self.subTest(gate=gate):
                health,result,after,unloaded,calls,records,loads=exercise(gate,'execution')
                self.assertEqual(result[0],503)
                self.assertEqual(result[1]['detail']['reason'],'quality-gate-not-qualified')
                self.assert_permission(result[1]['detail'],False,'execution')
                self.assert_permission(health[1],False,'execution')
                self.assertFalse(health[1]['quality_gate_passed'])
                self.assertEqual(health[1],after[1]);self.assertEqual((calls,len(records)),(0,0))

    def test_execution_with_qualified_gate_and_route_is_allowed(self):
        health,result,after,unloaded,calls,records,loads=exercise({'passed':True,'mode':'qualified'},'execution')
        self.assertEqual(result[0],200)
        self.assert_permission(health[1],True,'execution');self.assert_permission(result[1],True,'execution')
        self.assertIs(result[1]['recommendation']['execution_allowed'],True)
        self.assertIs(result[1]['recommendation']['advisory_only'],False)
        self.assertEqual(health[1],after[1]);self.assert_permission(unloaded[1],False,'execution')
        self.assertEqual((calls,len(records),loads),(1,1,1))

    def test_diagnostic_missing_and_malformed_gate_remains_advisory(self):
        for gate,status in [(None,'missing'),('{broken','invalid'),([],'invalid'),
                            ({'passed':'true'},'invalid'),({'passed':True,'mode':'shadow'},'invalid')]:
            with self.subTest(gate=gate):
                health,result,after,unloaded,calls,records,loads=exercise(gate)
                self.assertEqual(result[0],200);self.assert_permission(result[1],False,'diagnostic')
                self.assertEqual(result[1]['quality_gate_status'],status)
                self.assertEqual(health[1]['quality_gate_status'],status)
                self.assertIs(result[1]['recommendation']['advisory_only'],True)

    def test_no_authorized_route_never_grants_execution(self):
        routes=[{'tier':'fast','provider':'synthetic','model':'text-only','vision':False}]
        for mode in ['diagnostic','execution']:
            with self.subTest(mode=mode):
                health,result,after,unloaded,calls,records,loads=exercise(
                    {'passed':True,'mode':'qualified'},mode,prediction('flagship','yes'),routes)
                if mode=='diagnostic':
                    self.assertEqual(result[0],200);self.assertIsNone(result[1]['recommendation'])
                    self.assert_permission(result[1],False,mode)
                else:
                    self.assertEqual(result[0],503)
                    self.assertEqual(result[1]['detail']['reason'],'no-authorized-recommendation')
                    self.assert_permission(result[1]['detail'],False,mode)
                    self.assertEqual(len(records),0)
                self.assertEqual(calls,1)

    def test_not_loaded_and_invalid_mode_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'Invalid service mode'):server.create_app(mode='qualified')
        app=server.create_app(mode='execution')
        status,payload=asyncio.run(request(app,'/route',{'prompt':'Synthetic request.'}))
        self.assertEqual(status,503);self.assert_permission(payload['detail'],False,'execution')
        self.assertEqual(payload['detail']['reason'],'model-not-loaded')

    def test_unreadable_gate_fails_closed(self):
        with patch.object(Path,'read_text',side_effect=PermissionError):gate=server.read_gate(Path('synthetic-gate'))
        self.assertIs(gate['passed'],False);self.assertEqual(gate['status'],'invalid')


if __name__=='__main__':unittest.main()
