"""Request-bound reach hints; these are IK seeds, never executable plans."""
import copy
import json
import math


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    if hasattr(value,'tolist'):return _plain(value.tolist())
    return value


def request_key(source,tcp,center,product,cfg):
    # Recognition embeddings/images do not describe the collision geometry.
    obj={k:v for k,v in product.items() if k not in ('features','examples')}
    return json.dumps(_plain([source,tcp,center,obj,cfg]),sort_keys=True,allow_nan=False)


def matching_hint(planner,source,tcp,center,product):
    hint=getattr(planner,'_reach_candidate_hint',None)
    if hint is None or hint['request_key']!=request_key(source,tcp,center,product,planner.cfg):return None
    return copy.deepcopy(hint)


def joint_transit(source,goal,names,max_step_deg=2.):
    """Same coarse joint transit used by reach screening and full planning."""
    steps=max(1,int(math.ceil(max(abs(goal[n]-source[n]) for n in names)/max_step_deg)))
    for index in range(steps+1):
        t=index/steps
        yield {**source,**{n:(source[n] if index==0 else goal[n] if index==steps
                             else source[n]+t*(goal[n]-source[n])) for n in names}}
