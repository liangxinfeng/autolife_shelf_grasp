"""先选手、再完整规划：预算不平分，失败不在请求中换手。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import arm_selection as a,reach_precheck as r,planning_control as pc


def make_pool(monkeypatch,rows):
    now=[0.];calls=[];contexts=[];real=pc.PlanningControl
    monkeypatch.setattr(pc,'PlanningControl',lambda *args,**kw:real(*args,**kw,clock=lambda:now[0]))
    monkeypatch.setattr(a,'rank_arms',lambda *args:['left','right'])
    def screen(*args):now[0]=.2;return rows
    monkeypatch.setattr(r,'screen',screen)
    pool=a.ArmPlannerPool(dict(arm='left',grasp_backend='vendor_ik_trajectory'),None,None,None)
    pool.arm_config=lambda side,cfg:({'arm':side,'tool_geometry':{}},np.zeros(3))
    def planner(cfg,ctx):
        calls.append(cfg['arm']);contexts.append(ctx)
        def plan(*args):
            now[0]=6.;ctx.checkpoint() # 旧半份预算在这里必定失败。
            return {'segments':[]}
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    return pool,now,calls,contexts


@pytest.mark.parametrize('chosen',['left','right'])
def test_selected_hand_can_finish_after_half_budget(monkeypatch,chosen):
    rows=[dict(arm=chosen,status='key_poses_passed',yaw_offset_deg=75.)]
    pool,now,calls,contexts=make_pool(monkeypatch,rows);events=[]
    result=pool.plan({},np.zeros(3),{}, {},on_reach=events.append)
    assert result[0]['arm']==chosen and calls==[chosen]
    assert contexts[0].started==0 and contexts[0].deadline==10
    assert events[0]['status']=='key_poses_passed'
    assert result[1].preferred_grasp_yaw==75.


def test_real_deadline_does_not_get_extended_or_switch_arm(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[dict(arm='right',status='key_poses_passed',yaw_offset_deg=0.)])
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):now[0]=10.01;ctx.checkpoint()
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(pc.PlanningTimeout):pool.plan({},np.zeros(3),{}, {})
    assert calls==['right'] and len(pool.attempts)==1


def test_selected_hand_geometric_failure_returns_to_caller(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[dict(arm='right',status='key_poses_passed',yaw_offset_deg=0.)])
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):raise ValueError('blocked carry route')
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(ValueError,match='选定手臂路径未通过'):pool.plan({},np.zeros(3),{}, {})
    assert calls==['right']


def test_precheck_rejects_stale_docking_arm_hint(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[dict(arm='left',status='key_poses_passed',yaw_offset_deg=10.)])
    result=pool.plan({},np.zeros(3),{}, {},reach_hint={'arm':'right','yaw_offset_deg':75.})
    assert calls==['left'] and result[0]['arm']=='left'


def test_model_load_is_not_mistaken_for_ik_timeout(monkeypatch):
    clock=[0.];real=pc.PlanningControl
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(r,'PlanningControl',lambda *args,**kw:real(*args,**kw,clock=lambda:clock[0]))
    monkeypatch.setattr(r,'broad_margin',lambda *args:.1)
    monkeypatch.setattr(r,'model_probe',lambda p:p)
    def load(cfg,ctx):clock[0]+=1.;ctx.checkpoint();return NS(context=ctx,configure_source=lambda q:None)
    def check(p,*args):
        clock[0]+=.15;p.context.checkpoint()
        return dict(status='key_poses_passed',checked_key_poses=3,yaw_offset_deg=75.)
    monkeypatch.setattr(r,'check_key_poses',check)
    pool=NS(arm_config=lambda side,cfg:({},np.zeros(3)),native_planner=load)
    parent=real(timeout_sec=10,clock=lambda:clock[0])
    rows=r.screen(pool,{},np.zeros(3),{}, {},['right','left'],parent,lambda s:None,None)
    assert len(rows)==1 and rows[0]['status']=='key_poses_passed'
    assert parent.deadline==10 and clock[0]==1.15
