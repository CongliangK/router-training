"""Capability filtering and conservative recommendation; not a model-generated route."""
from __future__ import annotations
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


def select(prediction,routes,has_image=False,minimum_confidence=.9):
    tier=prediction['answers']['tier'];vision=prediction['answers']['vision']
    needs_vision=bool(has_image or vision['choice']=='yes')
    reasons=[];required=tier['choice']
    # Tier follows argmax: the highest-probability class is the best single
    # guess, so a low tier confidence alone no longer escalates the model.
    # Effort-side compensation for the same signal lives in the JS policy.
    if prediction.get('context_truncated'):
        required='flagship';reasons.append('context-truncated-flagship')
    if tier['confidence']<minimum_confidence:
        reasons.append('low-tier-confidence-effort-compensation-only')
    if vision['confidence']<minimum_confidence:
        needs_vision=True;reasons.append('uncertain-modality-requires-compatible-route')
    eligible=[route for route in routes if TIERS.index(route['tier'])>=TIERS.index(required) and (not needs_vision or route['vision'])]
    eligible.sort(key=lambda route:(TIERS.index(route['tier']),route['provider'],route['model']))
    if not eligible:return {'recommendation':None,'needs_vision':needs_vision,'required_tier':required,'reasons':reasons+['no-authorized-route-meets-required-capability']}
    route=eligible[0]
    return {'recommendation':{'provider':route['provider'],'model':route['model']},'needs_vision':needs_vision,'required_tier':required,'reasons':reasons}
