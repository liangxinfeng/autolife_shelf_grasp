"""Precheck hints cannot change task geometry or bypass transit validation."""
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from autolife_shelf_grasp_306.grasp_candidate_hint import matching_hint,joint_transit
from autolife_shelf_grasp_306 import reach_precheck as reach
from autolife_shelf_grasp_306 import planning as planning
from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner


def probe(collision=None):
    source={'j':0.};tcp=np.zeros(3);center=np.array([.4,0,0]);product={'height_m':.2,'diameter_m':.05}
    p=NS(cfg={'arm':'left'},names=['j'],_reach_quick_probe=True,
         context=NS(checkpoint=lambda:None),configure_source=lambda q:None,
         guard=NS(begin_fixed_neck=lambda q:None))
    st=[('pre',np.array([12.,0,0])),('contact',np.array([8.,0,0])),('lift',np.array([16.,0,0]))]
    geometry=(p.cfg,0.,.05,False,p.cfg,tcp,False,product,tcp,np.eye(3),np.array([1.,0,0]),0.,None,0.,False,st,True,None)
    p.route_geometry=lambda *a:geometry
    p.solves=[];p.checks=[]
    def solve(seed,tool,point,*a,**kw):
        p.solves.append((point[0],seed['j']))
        return {'j':float(point[0])}
    def check(q,*a):
        p.checks.append(q['j'])
        if collision is not None and abs(q['j']-collision)<1e-8:
            raise planning.CollisionFailure('body crossing')
    p.solve=solve;p.check=check
    return p,source,tcp,center,product


def test_precheck_rejects_collision_between_clear_endpoints():
    p,source,tcp,center,product=probe(6.)
    result=reach.check_key_poses(p,source,tcp,center,product)
    assert result['status']=='unresolved' and p._reach_candidate_hint is None
    assert 0. in p.checks and 12. in p.checks and 6. in p.checks


def test_precheck_rejects_colliding_source_before_solving():
    p,source,tcp,center,product=probe(0.)
    result=reach.check_key_poses(p,source,tcp,center,product)
    assert result['status']=='unresolved' and result['failure_stage']=='source'
    assert not p.solves


def test_precheck_keeps_exact_geometry_and_contact_branch():
    p,source,tcp,center,product=probe()
    result=reach.check_key_poses(p,source,tcp,center,product)
    assert result['status']=='key_poses_passed'
    assert p.solves==[(8.,0.),(12.,8.),(16.,8.)]
    assert result['checked_pregrasp_transit_states']==7
    hint=matching_hint(p,source,tcp,center,product)
    assert hint['key_poses'][0]=={'j':12.}
    assert np.array_equal(hint['geometry'][9],np.eye(3))
    hint['geometry'][9][0,0]=99.
    assert p._reach_candidate_hint['geometry'][9][0,0]==1.


@pytest.mark.parametrize('changed',['source','tcp','center','product','config'])
def test_hint_invalidated_by_any_task_change(changed):
    p,source,tcp,center,product=probe()
    reach.check_key_poses(p,source,tcp,center,product)
    if changed=='source':source['j']=.01
    elif changed=='tcp':tcp[0]=.001
    elif changed=='center':center[0]=.001
    elif changed=='product':product['diameter_m']=.051
    else:p.cfg['arm']='right'
    assert matching_hint(p,source,tcp,center,product) is None


def test_joint_transit_bounds_step_and_preserves_other_arm():
    rows=list(joint_transit({'j':-3.,'other':17.},{'j':8.,'other':-100.},['j']))
    assert rows[0]['j']==-3. and rows[-1]['j']==8.
    assert max(abs(b['j']-a['j']) for a,b in zip(rows,rows[1:]))<=2.
    assert all(q['other']==17. for q in rows)


def test_full_candidates_prioritize_hint_without_removing_fallbacks():
    p=NS(cfg={'arm':'left'},context=NS(set_stage=lambda *a,**k:None),
         _active_reach_candidate={'yaw_offset_deg':20.})
    result=list(VendorGraspPlanner.candidates(p,[0.,20.]))
    assert result[0]==(20.,False,'joint_prechecked')
    assert len(result)==13 and sum(row[2]=='joint_prechecked' for row in result)==1


def test_contact_collision_allows_another_attitude():
    p=NS(cfg={},context=NS(checkpoint=lambda:None),_attitude_cache={})
    p.solve=lambda *a:{'j':1.}
    def check(*args):raise planning.CollisionFailure('contact collision')
    p.check=check
    assert not VendorGraspPlanner.grasp_attitude_feasible(p,{},np.zeros(3),np.ones(3),np.eye(3),{})


@pytest.mark.parametrize('target_x',[.25,.85])
def test_full_prechecked_route_uses_saved_geometry_and_still_checks_transit(monkeypatch,target_x):
    p=planning.Planner.__new__(planning.Planner);p.names=list('abcdefg')
    p.cfg={'arm':'left','gripper_max_width_m':.15,'approach_yaw_deg':0.,'up_torso':[0,0,1],
           'T_wrist_tool_quat_xyzw':[0,0,0,1],'pregrasp_m':.1,'lift_m':.03,'withdraw_m':.15,
           'max_joint_speed_deg_s':120.}
    p.guard=NS(cache=set());p.fk=NS(sha='test')
    p.joint_limit_policy={'bounds_deg':{n:[-180,180] for n in p.names}}
    p.pose=lambda q,tcp:(np.array([q[n] for n in 'abc']),np.eye(3))
    seeds=[]
    def solve(seed,tool,point,*a,**kw):
        seeds.append(dict(seed));return {**seed,**dict(zip('abc',point))}
    p.solve=solve
    source={n:0. for n in p.names};tcp=np.zeros(3);center=np.array([target_x,0,0]);product={'diameter_m':.07}
    geometry=p.route_geometry(source,tcp,center,product,False)
    saved={**source,'a':float(geometry[15][0][1][0]),'g':4.}
    p._active_reach_candidate={'yaw_offset_deg':0.,'geometry':geometry,'key_poses':{0:saved}}
    p.route_geometry=lambda *a:pytest.fail('must retain the screened pitch/TCP/rotation')
    checked=[]
    def check(q,*a):
        checked.append(q['g'])
        if q['g']==2.:raise planning.CollisionFailure('transit collision')
    p.check=check
    with pytest.raises(planning.CollisionFailure,match='transit collision'):
        p._route(source,tcp,center,product,False,time.monotonic(),profile='joint_prechecked')
    assert seeds[0]==saved and checked==[0.,2.]
