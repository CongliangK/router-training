"""Loopback-only decision API. Run with the pinned framework Python environment."""
from __future__ import annotations
import argparse, hashlib, json, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field,ConfigDict
import uvicorn
from model_utils import ROOT,load_engine,score
from policy import select,DEFAULT_ROUTES,validate_routes,validate_policy
from extract import redact
from task_label_schema import LEGACY, VERSIONS, version, resolve_run_schema

class DecisionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    prompt:str=Field(min_length=1,max_length=50000)
    has_image:bool=False


def read_gate(path):
    # A missing or malformed gate must never become permission through truthiness.
    try:gate=json.loads(path.read_text())
    except FileNotFoundError:return {'passed':False,'mode':'shadow','status':'missing'}
    except (OSError,ValueError):return {'passed':False,'mode':'shadow','status':'invalid'}
    if not isinstance(gate,dict) or type(gate.get('passed')) is not bool:
        return {'passed':False,'mode':'shadow','status':'invalid'}
    if gate['passed'] and gate.get('mode')!='qualified':
        return {'passed':False,'mode':'shadow','status':'invalid'}
    return {**gate,'mode':'qualified' if gate['passed'] else 'shadow',
            'status':'passed' if gate['passed'] else 'not-passed'}


def record_observation(record):
    observations=ROOT/'observations';observations.mkdir(exist_ok=True)
    with (observations/'decisions.jsonl').open('a') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')


def create_app(run='real-history-v1',routes=None,minimum_confidence=.9,collect_prompts=False,mode='diagnostic',*,label_schema=LEGACY,vision_policy='legacy-v1',vision_positive_threshold=None):
    selected=version(label_schema)
    artifacts=ROOT/'artifacts'/run;state={};lock=threading.Lock()
    routes=validate_routes(routes if routes is not None else DEFAULT_ROUTES)
    validate_policy(minimum_confidence,vision_policy,vision_positive_threshold)
    if mode not in ('diagnostic','execution'):raise ValueError('Invalid service mode')
    def contract(recommendation_available=True):
        gate=state.get('gate',{'passed':False,'mode':'shadow','status':'not-loaded'})
        allowed='engine' in state and mode=='execution' and gate['passed'] is True and recommendation_available
        return {'schema_version':2,'task_label_schema_version':selected,'mode':mode,'quality_gate_passed':gate['passed'],
                'quality_gate_mode':gate['mode'],'quality_gate_status':gate['status'],
                'execution_allowed':allowed,'advisory_only':not allowed}
    @asynccontextmanager
    async def lifespan(app):
        # Schema mismatch fails BEFORE model loading, even in diagnostic mode.
        calibration=json.loads((artifacts/'calibration.json').read_text())
        schema_path=artifacts/'label-schema.json'
        run_schema=json.loads(schema_path.read_text()) if schema_path.exists() else None
        resolve_run_schema(calibration,run_schema,selected)
        full=artifacts/'full-checkpoint'
        adapter=artifacts/'adapter'
        if full.exists():state['engine']=load_engine(full_checkpoint=full,schema_version=selected)
        elif adapter.exists():state['engine']=load_engine(adapter=adapter,schema_version=selected)
        else:raise RuntimeError(f'No trained product found under {artifacts}')
        state['temperatures']=calibration['temperatures']
        state['adjustments']={'tier':calibration['tier_adjust']} if calibration.get('tier_adjust') else None
        state['gate']=read_gate(artifacts/'quality-gate.json')
        state['run']=run
        yield
        state.clear()
    app=FastAPI(title='User Mode local decision router',lifespan=lifespan)
    @app.get('/health')
    def health():return {'ready':'engine' in state,'run':run,**contract(),
                         'quality_gate':state.get('gate',{'passed':False,'mode':'shadow','status':'not-loaded'}),
                         'history_model_labels_used':False}
    @app.post('/route')
    def route(request:DecisionRequest):
        if 'engine' not in state:raise HTTPException(503,{'reason':'model-not-loaded',**contract(False)})
        if mode=='execution' and not contract()['execution_allowed']:
            raise HTTPException(503,{'reason':'quality-gate-not-qualified',**contract(False)})
        start=time.perf_counter()
        with lock:
            prediction=score(state['engine'],request.prompt,state['temperatures'],state.get('adjustments'))
            decision=select(prediction,routes,request.has_image,minimum_confidence,vision_policy=vision_policy,vision_positive_threshold=vision_positive_threshold)
            permission=contract(decision['recommendation'] is not None)
            if mode=='execution' and not permission['execution_allowed']:
                raise HTTPException(503,{'reason':'no-authorized-recommendation',**permission,**decision})
            if decision['recommendation'] is not None:
                decision['recommendation']={**decision['recommendation'],
                    'execution_allowed':permission['execution_allowed'],'advisory_only':permission['advisory_only']}
            result={**permission,'run':run,
                'tier':prediction['answers']['tier']['choice'],'tier_confidence':prediction['answers']['tier']['confidence'],
                'task_type':prediction['answers']['task_type']['choice'],
                'task_type_confidence':prediction['answers']['task_type']['confidence'],
                'context_truncated':prediction['context_truncated'],
                **decision,'latency_ms':round((time.perf_counter()-start)*1000,2)}
            record={'time':time.time(),'prompt_sha256':hashlib.sha256(request.prompt.encode()).hexdigest(),**result}
            if collect_prompts:record['prompt']=redact(request.prompt)
            record_observation(record)
            return result
    return app

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='real-history-v1');p.add_argument('--port',type=int,default=19948);p.add_argument('--collect-prompts',action='store_true');p.add_argument('--routes',type=Path);p.add_argument('--minimum-confidence',type=float,default=.9);p.add_argument('--mode',choices=['diagnostic','execution'],default='diagnostic')
    p.add_argument('--label-schema',choices=VERSIONS,default=LEGACY,help='Explicit checkpoint label/question schema; default preserves seven-task legacy')
    p.add_argument('--vision-policy',choices=['legacy-v1','positive-v2'],default='legacy-v1');p.add_argument('--vision-positive-threshold',type=float)
    args=p.parse_args()
    if not args.run.replace('-','').replace('_','').isalnum():raise SystemExit('Invalid run name')
    if not 0<=args.minimum_confidence<=1:raise SystemExit('Invalid confidence threshold')
    routes=json.loads(args.routes.read_text()) if args.routes else DEFAULT_ROUTES
    uvicorn.run(create_app(args.run,routes,args.minimum_confidence,args.collect_prompts,args.mode,label_schema=args.label_schema,vision_policy=args.vision_policy,vision_positive_threshold=args.vision_positive_threshold),host='127.0.0.1',port=args.port,log_level='info')
