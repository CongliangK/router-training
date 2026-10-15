"""Capability filtering and conservative recommendation; not a model-generated route."""
from __future__ import annotations
import math
TIERS=['fast','balanced','flagship']
DEFAULT_ROUTES=[
 {'tier':'fast','provider':'zai-coding-cn','model':'glm-5.3-flash','vision':True},
 {'tier':'balanced','provider':'zai-coding-cn','model':'glm-5.3','vision':False},
 {'tier':'flagship','provider':'openai-codex','model':'gpt-6.1-sol','vision':True},
]

def validate_routes(routes):
    if not isinstance(routes,list) or not routes:raise ValueError('Routes must be a nonempty list')
    seen=set();ids=set()
    for route in routes:
        if not isinstance(route,dict) or route.get('tier') not in TIERS or type(route.get('vision')) is not bool:raise ValueError('Invalid route capability declaration')
        if not all(isinstance(route.get(key),str) and route[key].strip() for key in ['provider','model']):raise ValueError('Invalid provider/model pair')
        pair=(route['provider'],route['model'])
        if pair in seen:raise ValueError('Duplicate route declaration')
        seen.add(pair)
        # Issue #7: optional execution metadata annotates the operating point only;
        # the (provider,model) uniqueness above stays the serving identity that
        # server.select() depends on. Unknown keys remain tolerated (existing behavior).
        for key in ['model_version','reasoning_effort','route_id']:
            value=route.get(key)
            if value is None:continue
            if not isinstance(value,str) or not value.strip() or len(value)>200:raise ValueError('Invalid route execution metadata')
        route_id=route.get('route_id')
        if route_id is not None:
            if route_id in ids:raise ValueError('Duplicate route id declaration')
            ids.add(route_id)
    return routes


def probability(value):
    return type(value) in (int,float) and 0<=value<=1 and math.isfinite(value)


def validate_policy(minimum_confidence=.9,vision_policy='legacy-v1',vision_positive_threshold=None):
    if not probability(minimum_confidence):raise ValueError('Invalid minimum confidence')
    if vision_policy not in ('legacy-v1','positive-v2'):raise ValueError('Invalid vision policy version')
    if vision_policy=='positive-v2':
        if not probability(vision_positive_threshold):raise ValueError('positive-v2 requires an explicit vision positive threshold')
    elif vision_positive_threshold is not None:raise ValueError('Vision positive threshold requires positive-v2')


def vision_probability(prediction):
    vision=prediction.get('answers',{}).get('vision',{})
    probabilities=vision.get('probabilities') if isinstance(vision,dict) else None
    value=probabilities.get('yes') if isinstance(probabilities,dict) else None
    return value if probability(value) else None


def select(prediction,routes,has_image=False,minimum_confidence=.9,*,vision_policy='legacy-v1',vision_positive_threshold=None):
    validate_policy(minimum_confidence,vision_policy,vision_positive_threshold)
    if type(has_image) is not bool:raise ValueError('has_image must be boolean')
    # Empty candidates are a valid refusal; malformed/ambiguous declarations are not.
    if routes!=[]:validate_routes(routes)
    tier=prediction['answers']['tier'];vision=prediction['answers'].get('vision') or {}
    if not isinstance(vision,dict):vision={}
    model_tier=tier['choice']
    if model_tier not in TIERS:raise ValueError('Invalid model tier')
    required=model_tier;reasons=[];events=[]
    def event(category,code,upgraded=False,**extra):
        events.append({'category':category,'code':code,'upgraded':upgraded,**extra})
    event('model_choice','classifier-tier',tier=model_tier)
    if prediction.get('context_truncated'):
        required='flagship';reasons.append('context-truncated-flagship')
        event('context_truncation','context-truncated-flagship',required!=model_tier)
    # Tier follows argmax: the highest-probability class is the best single
    # guess, so a low tier confidence alone no longer escalates the model.
    # Effort-side compensation for the same signal lives in the JS policy.
    # This service does not execute effort compensation. Keep the historical reason
    # for compatibility, but make its non-execution explicit in the versioned schema.
    if not probability(tier.get('confidence')) or tier['confidence']<minimum_confidence:
        reasons.append('low-tier-confidence-effort-compensation-only')
        event('risk_policy','low-tier-confidence-no-compensation',compensation_applied=False)
    p_yes=vision_probability(prediction)
    needs_vision=has_image
    if has_image:event('risk_policy','image-requires-vision')
    if p_yes is None:
        needs_vision=True;reasons.append('invalid-vision-probability-requires-compatible-route')
        event('risk_policy','invalid-vision-probability-requires-vision')
    elif vision_policy=='positive-v2':
        needs_vision=needs_vision or p_yes>=vision_positive_threshold
    else:
        needs_vision=needs_vision or vision.get('choice')=='yes'
        if vision.get('choice') not in ('yes','no') or not probability(vision.get('confidence')) or vision['confidence']<minimum_confidence:
            needs_vision=True;reasons.append('uncertain-modality-requires-compatible-route')
            event('risk_policy','uncertain-modality-requires-vision')
    candidates=[route for route in routes if TIERS.index(route['tier'])>=TIERS.index(required)]
    key=lambda route:(TIERS.index(route['tier']),route['provider'],route['model'])
    candidates.sort(key=key)
    eligible=[route for route in candidates if not needs_vision or route['vision']]
    route=eligible[0] if eligible else None
    selected=route['tier'] if route else None
    # Compare against the SAME required tier without the vision constraint. This
    # prevents attributing a missing intermediate tier or truncation to vision.
    if needs_vision and candidates and (not route or TIERS.index(selected)>TIERS.index(candidates[0]['tier'])):
        event('vision_forced','vision-capability-upgrade' if route else 'vision-capability-refusal',bool(route))
    elif route and selected!=required:
        event('risk_policy','authorized-tier-unavailable',True)
    if not route:
        reasons.append('no-authorized-route-meets-required-capability')
        event('risk_policy','no-authorized-route-meets-required-capability')
    order=['model_choice','vision_forced','context_truncation','risk_policy']
    events.sort(key=lambda item:order.index(item['category']))  # stable within category
    recommendation=None if route is None else {key:route[key] for key in ('provider','model','route_id','model_version','reasoning_effort') if route.get(key) is not None}
    return {'decision_schema_version':1,'recommendation_contract_version':1,
        'vision_policy':vision_policy,'vision_positive_threshold':vision_positive_threshold,
        'vision_probability':p_yes,'recommendation':recommendation,'needs_vision':needs_vision,
        'model_tier':model_tier,'required_tier':required,'selected_tier':selected,
        'reason_details':events,'reasons':reasons}
