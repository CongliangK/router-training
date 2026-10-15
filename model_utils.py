"""Fixed-schema decision scoring and LoRA loading; never generates task answers."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TIERS=['fast','balanced','flagship']
TYPES=['extraction','research','implementation','debugging','review','visual','creative']
QUESTIONS={
 'tier':{'type':'choice','instructions':'为这个实际子任务选择最低可合理胜任的能力档位。balanced 是默认档：边界明确的执行、实现、修改、常规审查都从 balanced 起步，只有满足下列 flagship 情形之一才升级，且不依据提示词长度判断。历史默认模型无参考价值。', 'criteria':{
  'fast':'机械提取、格式转换、枚举、简单摘要；低风险且几乎不需推理。',
  'balanced':'边界明确的实现、修改、局部排错、常规审查、研究综合、常规内容执行；称职工程师按步骤即可完成。',
  'flagship':'仅限：跨模块架构设计（多方案权衡）；无现成路径的创造性方案；症状与病因跨模块分离的复杂根因；安全/并发正确性存疑且难以靠常规测试发现；高影响产出的对抗性终审。'}},
 'vision':{'type':'choice','instructions':'完成该任务是否必须理解实际图片像素、截图、图表或视频画面？代码中提到图像/UI/视频不等于需要视觉。', 'criteria':{
  'no':'仅文本、代码、音频或配置即可完成；不需要实际像素理解。',
  'yes':'需要读取或比较实际图片、截图、图表、渲染画面或视频帧。'}},
 'task_type':{'type':'choice','instructions':'选择任务的主要操作类型。','criteria':dict(zip(TYPES,[
  '机械提取、统计、格式转换','调查资料、研究综合','编写或修改代码和测试','诊断错误、排查根因','审核代码、方案、规范或质量','直接分析图片或画面','音乐、视频、文案等创作或创作方案']))}
}


def reference_module():
    path=ROOT/'models/intern-decision-0.8b/inference.py'
    spec=importlib.util.spec_from_file_location('intern_decision_reference',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def load_engine(adapter=None,device='cuda',max_length=8192,full_checkpoint=None):
    # The base checkpoint natively supports 262144 positions; 8192 keeps every
    # historical task (max 3284 tokens incl. template) untruncated while the
    # head/tail scheme below remains the extreme-prompt safety net.
    reference=reference_module()
    base=ROOT/'models/intern-decision-0.8b'
    checkpoint=full_checkpoint if full_checkpoint else base
    engine=reference.DecisionEngine(checkpoint=checkpoint,device=device,max_length=max_length,temperature=1.0)
    if full_checkpoint:
        # Full fine-tune product: weights already merged; no adapter layer.
        engine.backend.model.eval()
        return engine
    if adapter:
        from peft import PeftModel
        engine.backend.model=PeftModel.from_pretrained(engine.backend.model,str(adapter),is_trainable=False)
    return engine


def request_for(engine,prompt,max_prompt_tokens=7700):
    # Explicit, deterministic head/tail representation shared by training and serving.
    # The reserved marker cannot be injected by an untrusted delegated task.
    prompt=prompt.replace('<decision>','[reserved-marker]')
    ids=engine.tokenizer.encode(prompt,add_special_tokens=False)
    truncated=len(ids)>max_prompt_tokens
    if truncated:
        head=max_prompt_tokens*3//4;tail=max_prompt_tokens-head
        prompt=engine.tokenizer.decode(ids[:head])+'\n[ROUTING_CONTEXT_TRUNCATED]\n'+engine.tokenizer.decode(ids[-tail:])
    return {'state':{'delegated_task':prompt},'questions':QUESTIONS},truncated


def field_logits(engine,request,train=False):
    compiled,batch,positions=engine.backend.encode(request)
    device=engine.backend.device;batch=batch.to(device)
    output=engine.backend.model(**batch,use_cache=False,logits_to_keep=positions.to(device)).logits[0]
    fields={}
    for i,field in enumerate(compiled.fields):
        symbols=compiled.symbols[field];encoded=[engine.tokenizer.encode(s,add_special_tokens=False) for s in symbols]
        if any(len(s)!=1 for s in encoded):raise ValueError('Decision symbols are not single tokens')
        fields[field]=output[i,[s[0] for s in encoded]].float()
    return fields,batch['input_ids'].shape[-1]


def score(engine,prompt,temperatures=None,adjustments=None):
    import torch
    request,truncated=request_for(engine,prompt);engine.backend.model.eval()
    with torch.inference_mode(): logits,length=field_logits(engine,request)
    result={}
    for field,values in logits.items():
        labels=list(QUESTIONS[field]['criteria']);temperature=(temperatures or {}).get(field,1.0)
        if adjustments and field in adjustments:
            # Decision-side logit adjustment (Menon et al. 2020): subtracting
            # tau*log(prior) moves the argmax boundary away from majority classes.
            values=values.clone()
            for index,label in enumerate(labels):
                if label in adjustments[field]:values[index]=values[index]-adjustments[field][label]
        probabilities=torch.softmax(values/temperature,dim=-1).cpu().tolist()
        best=max(range(len(labels)),key=lambda i:probabilities[i])
        result[field]={'choice':labels[best],'confidence':probabilities[best],'probabilities':dict(zip(labels,probabilities)),'logits':values.cpu().tolist()}
    return {'answers':result,'input_tokens':length,'context_truncated':truncated}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
