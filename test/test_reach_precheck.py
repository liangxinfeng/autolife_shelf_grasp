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
    with pytest.raises(ValueError,match='调整停靠位置'):p.plan({},np.zeros(3),{}, {},on_reach=messages.append)
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


def test_pool_timeout_budget_stays_with_selected_arm(monkeypatch):
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
    with pytest.raises(ValueError):p.plan({},np.zeros(3),{}, {})
    assert len(contexts)==1 and contexts[0].timeout_sec==r.STATIONARY_SECONDS
    assert all(x['status']=='timeout' for x in p.attempts)


def test_workspace_margin_uses_union_not_big_symmetric_ball():
    from autolife_shelf_grasp_306.reach_envelope import margin
    positions=np.array([[0,0,0],[.1,0,0]])
    assert margin(positions,.5,[.59,0,0])>0
    assert margin(positions,.5,[-.59,0,0])<0


def test_worker_timeout_enters_same_reposition_handler():
    import ast
    from pathlib import Path
    from autolife_shelf_grasp_306 import worker
    tree=ast.parse(Path(worker.__file__).read_text())
    handlers=[n for n in ast.walk(tree) if isinstance(n,ast.ExceptHandler) and n.type is not None
              and 'PlanningTimeout' in ast.unparse(n.type)]
    assert any('select_operation_dock()' in ast.unparse(n) for n in handlers)
    assert all('PlanningCancelled' not in ast.unparse(n.type) for n in handlers)
