import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import reach_precheck as r
from autolife_shelf_grasp_306.planning_control import PlanningControl,PlanningTimeout,PlanningCancelled


def test_child_cannot_extend_total_deadline():
    now=[0.];parent=PlanningControl(timeout_sec=15,clock=lambda:now[0])
    child=r.slice_control(parent,4,lambda s:None)
    now[0]=15
    with pytest.raises(PlanningTimeout):child.checkpoint()


def test_cancel_is_not_classified_as_unreachable():
    stop=[False];parent=PlanningControl(lambda:stop[0],timeout_sec=15)
    child=r.slice_control(parent,4,lambda s:None);stop[0]=True
    with pytest.raises(PlanningCancelled):child.checkpoint()


def test_precheck_failure_is_uncertain_not_a_motion_permission(monkeypatch):
    monkeypatch.setattr(r,'broad_margin',lambda *a:.1)
    monkeypatch.setattr(r,'model_probe',lambda p:p)
    def timeout(*args):raise PlanningTimeout('short budget')
    monkeypatch.setattr(r,'check_key_poses',timeout)
    p=NS(arm_config=lambda side,cfg:({'target_offset_torso_mm':[0,0,0]},np.zeros(3)),
         native_planner=lambda *a:NS(configure_source=lambda q:None))
    result=r.screen(p,{},np.zeros(3),{}, {},['left','right'],PlanningControl(timeout_sec=15),lambda s:None,None)
    assert len(result)==2 and all(x['status']=='unresolved' for x in result)


def test_pass_keeps_current_position_and_prioritizes_arm(monkeypatch):
    monkeypatch.setattr(r,'broad_margin',lambda *a:.1)
    monkeypatch.setattr(r,'model_probe',lambda p:p)
    calls=[]
    monkeypatch.setattr(r,'check_key_poses',lambda *a:dict(status='key_poses_passed',checked_key_poses=3))
    p=NS(arm_config=lambda side,cfg:(calls.append(side) or {},np.zeros(3)),native_planner=lambda *a:NS(configure_source=lambda q:None))
    result=r.screen(p,{},np.zeros(3),{}, {},['right','left'],PlanningControl(timeout_sec=15),lambda s:None,None)
    assert calls==['right'] and result[0]['status']=='key_poses_passed'


def test_definitely_outside_skips_native_planning(monkeypatch):
    monkeypatch.setattr(r,'broad_margin',lambda *a:-.1)
    p=NS(arm_config=lambda side,cfg:({},np.zeros(3)),native_planner=lambda *a:pytest.fail('不应进入规划'))
    result=r.screen(p,{},np.zeros(3),{}, {},['right','left'],PlanningControl(timeout_sec=15),lambda s:None,None)
    assert len(result)==2 and all(row['status']=='outside_workspace' for row in result)


def test_outside_pool_does_not_try_stationary_paths(monkeypatch):
    from autolife_shelf_grasp_306.arm_selection import ArmPlannerPool
    monkeypatch.setattr(r,'screen',lambda *a:[dict(arm='left',status='outside_workspace')])
    p=ArmPlannerPool({'arm':'left','auto_arm_selection':False,'grasp_backend':'vendor_ik_trajectory'},None,None,None)
    p.native_planner=lambda *a:pytest.fail('不应进入原地完整规划')
    messages=[]
    from autolife_shelf_grasp_306.auto_grasp import StationaryPathsExhausted
    with pytest.raises(StationaryPathsExhausted,match='工作包络'):p.plan({},np.zeros(3),{}, {},on_reach=messages.append)
    assert messages[0]['status']=='outside_workspace'


def test_numeric_probe_never_configures_vendor_process(monkeypatch):
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner
    actual=NS(context=None)
    probe=r.model_probe(actual)
    assert type(probe).configure_source is VendorGraspPlanner.configure_model
    assert type(probe).solve is not VendorGraspPlanner.solve


def test_screened_heading_changes_order_not_candidate_set():
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner
    p=NS(preferred_grasp_yaw=20.,cfg={'arm':'right'},context=NS(set_stage=lambda *a,**k:None))
    result=list(VendorGraspPlanner.candidates(p,[0.,-20.,20.]))
    assert result[0][0]==20. and {row[0] for row in result}=={0.,-20.,20.} and len(result)==18


def test_arm_slice_timeout_preserves_other_hand_opportunity(monkeypatch):
    from autolife_shelf_grasp_306.arm_selection import ArmPlannerPool
    import autolife_shelf_grasp_306.arm_selection as a
    monkeypatch.setattr(r,'screen',lambda *args:[])
    monkeypatch.setattr(a,'rank_arms',lambda *args:['left','right'])
    contexts=[]
    p=ArmPlannerPool({'arm':'left','grasp_backend':'vendor_ik_trajectory'},None,None,None)
    p.arm_config=lambda side,cfg:({'arm':side,'tool_geometry':{}},np.zeros(3))
    class Fake:
        candidate_failures=[]
        def __init__(self,ctx):self.ctx=ctx
        def plan(self,*args):raise PlanningTimeout('budget reached')
    p.native_planner=lambda cfg,ctx:(contexts.append(ctx) or Fake(ctx))
    with pytest.raises(PlanningTimeout):p.plan({},np.zeros(3),{}, {})
    assert len(contexts)==2 and contexts[0].timeout_sec<=r.STATIONARY_SECONDS/2
    assert contexts[1].timeout_sec==r.STATIONARY_SECONDS
    assert all(x['status']=='timeout' for x in p.attempts)


def test_workspace_margin_uses_union_not_big_symmetric_ball():
    from autolife_shelf_grasp_306.reach_envelope import margin
    positions=np.array([[0,0,0],[.1,0,0]])
    assert margin(positions,.5,[.59,0,0])>0
    assert margin(positions,.5,[-.59,0,0])<0


def test_fast_envelope_includes_native_ik_residual(monkeypatch):
    from autolife_shelf_grasp_306 import reach_envelope
    monkeypatch.setattr(reach_envelope,'current_margin',lambda *args:-.0025)
    assert r.broad_margin(NS(fk=None),{},np.zeros(3),np.zeros(3),{},dict(arm='left'))>0


def test_offset_hull_is_covered_even_when_trimmed_box_is_narrow(monkeypatch):
    from autolife_shelf_grasp_306 import reach_envelope
    monkeypatch.setattr(reach_envelope,'current_margin',lambda *args:-.12)
    product=dict(bbox_axes_torso=np.eye(3),bbox_extents_m=[.06,.04,.15],
                 measured_footprint_torso=[[-.02,-.02,0],[.18,-.02,0],[.18,.02,0]])
    assert r.broad_margin(NS(fk=None),{},np.zeros(3),np.zeros(3),product,dict(arm='left'))>0
