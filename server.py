"""Loopback-only decision API. Run with the pinned framework Python environment."""
from __future__ import annotations
import argparse, hashlib, json, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field,ConfigDict
import uvicorn
from model_utils import ROOT,load_engine,score
from policy import select,DEFAULT_ROUTES,validate_routes
from extract import redact

class DecisionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    prompt:str=Field(min_length=1,max_length=50000)
    has_image:bool=False


def create_app(run='real-history-v1',routes=None,minimum_confidence=.9,collect_prompts=False):
    artifacts=ROOT/'artifacts'/run;state={};lock=threading.Lock()
    routes=validate_routes(routes if routes is not None else DEFAULT_ROUTES)
    if not 0<=minimum_confidence<=1:raise ValueError('Invalid minimum confidence')
    @asynccontextmanager
    async def lifespan(app):
        full=artifacts/'full-checkpoint'
        adapter=artifacts/'adapter'
        if full.exists():state['engine']=load_engine(full_checkpoint=full)
        elif adapter.exists():state['engine']=load_engine(adapter=adapter)
        else:raise RuntimeError(f'No trained product found under {artifacts}')
        calibration=json.loads((artifacts/'calibration.json').read_text())
        state['temperatures']=calibration['temperatures']
        state['adjustments']={'tier':calibration['tier_adjust']} if calibration.get('tier_adjust') else None
        state['gate']=json.loads((artifacts/'quality-gate.json').read_text())
        state['run']=run
        yield
        state.clear()
    app=FastAPI(title='User Mode local decision router',lifespan=lifespan)
    @app.get('/health')
    def health():return {'ready':'engine' in state,'run':run,'quality_gate':state.get('gate',{'passed':False}),'history_model_labels_used':False}
    @app.post('/route')
    def route(request:DecisionRequest):
        if 'engine' not in state:raise HTTPException(503,'Model not loaded')
        start=time.perf_counter()
        with lock:
            prediction=score(state['engine'],request.prompt,state['temperatures'],state.get('adjustments'))
            decision=select(prediction,routes or DEFAULT_ROUTES,request.has_image,minimum_confidence)
            result={'schema_version':1,'mode':'shadow' if not state['gate']['passed'] else 'qualified',
                'quality_gate_passed':state['gate']['passed'],'run':run,
                'tier':prediction['answers']['tier']['choice'],'tier_confidence':prediction['answers']['tier']['confidence'],
                'vision_probability':prediction['answers']['vision']['probabilities']['yes'],
                'task_type':prediction['answers']['task_type']['choice'],
                'task_type_confidence':prediction['answers']['task_type']['confidence'],
                'context_truncated':prediction['context_truncated'],
                **decision,'latency_ms':round((time.perf_counter()-start)*1000,2)}
            observations=ROOT/'observations';observations.mkdir(exist_ok=True)
            record={'time':time.time(),'prompt_sha256':hashlib.sha256(request.prompt.encode()).hexdigest(),**result}
            if collect_prompts:record['prompt']=redact(request.prompt)
            with (observations/'decisions.jsonl').open('a') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')
            return result
    return app

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='real-history-v1');p.add_argument('--port',type=int,default=19948);p.add_argument('--collect-prompts',action='store_true');p.add_argument('--routes',type=Path);p.add_argument('--minimum-confidence',type=float,default=.9)
    args=p.parse_args()
    if not args.run.replace('-','').replace('_','').isalnum():raise SystemExit('Invalid run name')
    if not 0<=args.minimum_confidence<=1:raise SystemExit('Invalid confidence threshold')
    routes=json.loads(args.routes.read_text()) if args.routes else DEFAULT_ROUTES
    uvicorn.run(create_app(args.run,routes,args.minimum_confidence,args.collect_prompts),host='127.0.0.1',port=args.port,log_level='info')
