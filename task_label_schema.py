"""Explicit label/question versions (issue #18); stdlib only, no data inference.

Unversioned rows/checkpoints mean v1, NEVER v2. Migration is a reviewed operation,
not a string replacement: unresolved evidence stays outside supervised splits.
"""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path

LEGACY = 'router-label-v1'
V2 = 'router-label-v2'
VERSIONS = (LEGACY, V2)
SIX_TYPES = ('extraction','research','implementation','debugging','review','creative')


def version(value=None):
    if value is None:return LEGACY
    if not isinstance(value,str) or value not in VERSIONS:
        raise ValueError('unknown label schema version; explicit supported version required')
    return value


def questions_for(schema_version=LEGACY, legacy_questions=None):
    selected=version(schema_version)
    if legacy_questions is None:
        from model_utils import QUESTIONS
        legacy_questions=QUESTIONS
    questions=copy.deepcopy(legacy_questions)
    if selected==V2:
        questions['task_type']={'type':'choice',
            'instructions':'按主要交付动作选择六类；编写/修改可执行代码、脚本、样式表优先 implementation；非编程创作才是 creative。视觉是独立属性，读图按有证据的动作归入 extraction 或 review，信息不足须待裁定。',
            'criteria':dict(zip(SIX_TYPES,[
                '机械提取、统计、格式转换（含读图提取）','调查资料、研究综合',
                '主要交付为编写或修改可执行代码、脚本、样式表及测试',
                '诊断错误、排查根因','审核代码、方案、规范或画面质量',
                '非编程的文案、剧本、设计构思、音乐视频创作或鉴赏']))}
    return questions


def metadata(schema_version=LEGACY):
    selected=version(schema_version);questions=questions_for(selected)
    encoded=json.dumps(questions,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return {'task_label_schema_version':selected,'question_sha256':hashlib.sha256(encoded).hexdigest(),
            'classes':{head:list(spec['criteria']) for head,spec in questions.items()}}


def resolve_run_schema(calibration, run_metadata=None, requested=LEGACY):
    """Check schema identity BEFORE model load. Missing metadata is v1 only.

    New v2 products require BOTH label-schema.json and calibration.label_schema,
    identical to this code's question fingerprint and ordered classes. HTTP
    schema_version and outcome schema_version are unrelated namespaces.
    """
    selected=version(requested)
    if not isinstance(calibration,dict):raise ValueError('calibration must be an object')
    declared=calibration.get('label_schema')
    if 'task_label_schema_version' in calibration:
        raise ValueError('run schema must use complete label_schema metadata, not a bare version')
    if declared is None:
        if selected!=LEGACY or run_metadata is not None:
            raise ValueError('unversioned calibration is legacy only; v2 metadata cannot be inferred')
        return LEGACY
    if not isinstance(declared,dict):raise ValueError('invalid label schema metadata')
    actual=version(declared.get('task_label_schema_version'))
    if actual!=selected:raise ValueError('requested label schema differs from checkpoint schema')
    if declared!=metadata(actual):raise ValueError('checkpoint question fingerprint/classes mismatch')
    if actual==V2 and run_metadata is None:raise ValueError('v2 checkpoint needs label-schema.json')
    if run_metadata is not None and run_metadata!=declared:
        raise ValueError('run and calibration label schema metadata differ')
    return actual


def load_run_schema(directory, requested=LEGACY):
    directory=Path(directory)
    calibration=json.loads((directory/'calibration.json').read_text())
    path=directory/'label-schema.json'
    run_metadata=json.loads(path.read_text()) if path.exists() else None
    return resolve_run_schema(calibration,run_metadata,requested)


def check_row_versions(rows, schema_version=LEGACY):
    """Cheap boundary check also used by legacy metric readers."""
    selected=version(schema_version)
    for row in rows:
        if not isinstance(row,dict):raise ValueError('label rows must be objects')
        label=row.get('label') or {}
        declarations=[obj['task_label_schema_version'] for obj in (row,label)
                      if isinstance(obj,dict) and 'task_label_schema_version' in obj]
        if selected==V2 and row.get('task_label_schema_version')!=V2:
            raise ValueError('v2 rows require explicit top-level task_label_schema_version')
        if any(version(v)!=selected or v is None for v in declarations):
            raise ValueError('mixed row label schema versions are forbidden')


def validate_supervised_rows(rows, schema_version=LEGACY, allow_supplemental=False):
    selected=version(schema_version);check_row_versions(rows,selected)
    if selected==LEGACY:return  # legacy legality remains owned by its original readers
    for row in rows:
        label=row.get('label')
        if not isinstance(label,dict):raise ValueError('v2 supervised rows need labels')
        if label.get('label_status')!='resolved':
            raise ValueError('adjudication/excluded/unresolved labels cannot be supervised')
        if label.get('tier') not in ('fast','balanced','flagship') or \
                label.get('task_type') not in SIX_TYPES or label.get('needs_vision') not in ('yes','no'):
            raise ValueError('illegal v2 label; uncertain and visual require reviewed migration')
        if row.get('supplemental_only') and not allow_supplemental:
            raise ValueError('supplemental examples cannot supply real calibration/evaluation support')
        if row.get('synthetic') and row.get('supplemental_only') is not True:
            raise ValueError('v2 synthetic rows must be explicitly supplemental_only')


def validate_prediction_rows(rows, schema_version=LEGACY):
    """V2 cannot consume legacy seven-option logits/probabilities by accident."""
    validate_supervised_rows(rows,schema_version)
    if version(schema_version)!=V2:return
    classes=metadata(V2)['classes']
    for row in rows:
        answers=row.get('answers')
        if not isinstance(answers,dict) or set(answers)!=set(classes):
            raise ValueError('v2 prediction needs all schema heads')
        for head,labels in classes.items():
            answer=answers[head]
            if not isinstance(answer,dict) or answer.get('choice') not in labels or \
                    set(answer.get('probabilities',{}))!=set(labels):
                raise ValueError('prediction differs from selected label schema')
            if 'logits' in answer and (not isinstance(answer['logits'],list) or len(answer['logits'])!=len(labels)):
                raise ValueError('prediction logit width differs from selected label schema')


def reviewed_task_type(primary_action, evidence=None):
    """Executable rubric, driven by reviewer facts, NEVER prompt keyword guesses.

    Writing executable creative/visual deliverables wins over their subject.
    A visual-only legacy label supplies no action: explicit evidence is needed.
    """
    if primary_action=='write_executable':return 'implementation'
    if primary_action=='nonprogramming_creative':return 'creative'
    if primary_action in ('visual_extract','visual_review'):
        if not isinstance(evidence,str) or not evidence.strip():
            raise ValueError('visual mapping needs action evidence; otherwise adjudicate')
        return {'visual_extract':'extraction','visual_review':'review'}[primary_action]
    raise ValueError('ambiguous primary action requires adjudication, not a guessed task type')


def review_legacy_label(label, *, vision_decision=None, vision_evidence=None,
                        primary_action=None, action_evidence=None):
    """Return a NEW v2 label or an adjudication record; never mutate frozen-v1.

    Caller supplies independently reviewed evidence, not model predictions. This
    helper does not attest that the evidence is true or authorize real review.
    """
    result=copy.deepcopy(label);result['task_label_schema_version']=V2
    result['label_status']='resolved';pending=[];review={}
    if label.get('needs_vision')=='uncertain':
        if vision_decision not in ('yes','no') or not isinstance(vision_evidence,str) or not vision_evidence.strip():
            result['needs_vision']=None;pending.append('vision_evidence_required')
        else:
            result['needs_vision']=vision_decision;review['vision_evidence']=vision_evidence
    elif vision_decision is not None:
        raise ValueError('changing an existing binary label needs a separate reviewed label revision')
    if primary_action is not None:
        try:
            if label.get('task_type')=='visual' and (not isinstance(action_evidence,str) or not action_evidence.strip()):
                raise ValueError('legacy visual needs action evidence')
            result['task_type']=reviewed_task_type(primary_action,action_evidence)
        except ValueError:
            result['task_type']=None;pending.append('primary_action_evidence_required')
        review['primary_action']=primary_action
        if action_evidence:review['action_evidence']=action_evidence
    elif label.get('task_type')=='visual':
        result['task_type']=None;pending.append('visual_action_evidence_required')
    if pending:result['label_status']='adjudication';result['adjudication_reasons']=pending
    result['review_evidence']=review
    return result


def partition_labels(labels, schema_version=LEGACY):
    """Explicit resolved/pending partition for prepare; no binary coercions."""
    selected=version(schema_version);active=[];pending=[]
    for label in labels:
        row={'label':label}
        if 'task_label_schema_version' in label:row['task_label_schema_version']=label['task_label_schema_version']
        check_row_versions([row],selected)
        if selected==V2 and label.get('label_status') in ('adjudication','excluded'):
            if not label.get('adjudication_reasons'):
                raise ValueError('excluded/adjudication labels require reasons')
            pending.append(label);continue
        validate_supervised_rows([row],selected);active.append(label)
    return active,pending


def supplemental_rows(paths, schema_version):
    selected=version(schema_version)
    if selected!=V2:raise ValueError('v2 supplemental loader requires explicit router-label-v2')
    rows=[]
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    validate_supervised_rows(rows,selected,allow_supplemental=True)
    if any(row.get('synthetic') is not True or row.get('supplemental_only') is not True
           or 'split' in row or not row.get('id') or not row.get('prompt') for row in rows):
        raise ValueError('supplemental rows must be explicitly synthetic, split-independent examples')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate supplemental ids')
    return rows


def training_supplemental(paths, primary_rows, schema_version):
    """Shared structural/run join validation; never opens test or fits priors."""
    rows=supplemental_rows(paths,schema_version) if paths else []
    if {r['id'] for r in rows}&{r['id'] for r in primary_rows}:
        raise ValueError('supplemental ids overlap the primary splits')
    if {r['prompt'] for r in rows}&{r.get('prompt') for r in primary_rows}:
        raise ValueError('supplemental prompts overlap the primary splits')
    return rows
