"""自动选手预留另一侧尝试时间；固定手独享总预算，取消不续期。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import arm_selection as a,reach_precheck as r,planning_control as pc


def make_pool(monkeypatch,rows,finish_time=1.):
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
            now[0]=finish_time;ctx.checkpoint()
            return {'segments':[]}
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    return pool,now,calls,contexts


@pytest.mark.parametrize('chosen',['left','right'])
def test_selected_hand_can_finish_after_half_budget(monkeypatch,chosen):
    rows=[dict(arm=chosen,status='key_poses_passed',yaw_offset_deg=75.)]
    pool,now,calls,contexts=make_pool(monkeypatch,rows,finish_time=6.);events=[]
    result=pool.plan({},np.zeros(3),{}, {'auto_arm_selection':False,'arm':chosen},on_reach=events.append)
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


def test_both_hands_geometric_failure_reports_both_without_returning_a_plan(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[dict(arm='right',status='key_poses_passed',yaw_offset_deg=0.)])
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):raise ValueError('blocked carry route')
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(ValueError,match='双臂路径均未通过'):pool.plan({},np.zeros(3),{}, {})
    assert calls==['right','left']
    assert [row['status'] for row in pool.attempts]==['unavailable','unavailable']


@pytest.mark.parametrize('left_fault',[pc.PlanningTimeout,ValueError,BrokenPipeError,a.StationaryPathsExhausted])
def test_every_allowed_hand_needs_evidence_before_proposing_a_new_stop(monkeypatch,left_fault):
    pool,now,calls,contexts=make_pool(monkeypatch,[])
    def planner(cfg,ctx):
        side=cfg['arm'];calls.append(side)
        def plan(*args):
            failure=left_fault if side=='left' else a.StationaryPathsExhausted
            raise failure(side)
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(ValueError) as outcome:pool.plan({},np.zeros(3),{}, {})
    assert calls==['left','right']
    assert isinstance(outcome.value,a.StationaryPathsExhausted)==(left_fault is a.StationaryPathsExhausted)


@pytest.mark.parametrize('outcome',['recovered','second_exit','deadline','cancel','exit_during_screen'])
def test_solver_exit_recovery_is_bounded_to_one_fresh_process(monkeypatch,outcome):
    from autolife_shelf_grasp_306.vendor_ik_process import VendorSolverExited
    fault=VendorSolverExited(20,-11,'ik',dict(pose=[.4,.1,.5,0.,0.,0.,1.]))
    screen=[dict(arm='left',status='unresolved',solver_exit=fault.details)] if outcome=='exit_during_screen' else []
    pool,now,_,_=make_pool(monkeypatch,screen)
    calls=[];closes=[];stopped=[False]
    def planner(cfg,ctx):
        def plan(*args):
            calls.append((id(args[0]),id(args[1]),id(args[2]),id(args[3]),ctx.deadline))
            if outcome=='recovered' and len(calls)==2:return {'segments':['checked full path']}
            if outcome=='deadline':now[0]=ctx.deadline
            if outcome=='cancel':stopped[0]=True
            raise fault
        return NS(plan=plan,close=lambda:closes.append(True))
    pool.native_planner=planner
    def run():return pool.plan({},np.zeros(3),{},dict(arm='left',auto_arm_selection=False),stopped=lambda:stopped[0])
    if outcome=='recovered':
        assert run()[0]['segments']==['checked full path']
    else:
        error={'second_exit':ValueError,'exit_during_screen':ValueError,
               'deadline':pc.PlanningTimeout,'cancel':pc.PlanningCancelled}[outcome]
        with pytest.raises(error) as raised:run()
        assert not isinstance(raised.value,a.StationaryPathsExhausted)
    assert len(calls)==(2 if outcome in ('recovered','second_exit') else 1)
    assert len(closes)==len(calls)-1 and len(set(calls))==1
    assert pool.solver_exits[-1]['returncode']==-11 and pool.solver_exits[-1]['inputs']==fault.details['inputs']


def test_automatic_collision_failure_runs_other_complete_path_with_same_deadline(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[dict(arm='left',status='key_poses_passed',yaw_offset_deg=20.)])
    observations=[];configs=[];messages=[]
    def config(side,current):
        configs.append(side)
        return dict(arm=side,tool_geometry={'side':side},target_offset_torso_mm=[0,0,0]),np.array([1. if side=='left' else 2.,0,0])
    pool.arm_config=config
    def observe(side,cfg):
        observations.append(side)
        return np.array([.4,-.2 if side=='left' else .2,.5]),{'observed_arm':side}
    def planner(cfg,ctx):
        side=cfg['arm'];calls.append(side);contexts.append(ctx)
        def plan(source,tcp,center,product):
            assert source=={'unchanged':1} and product['observed_arm']==side
            assert tcp[0]==(1. if side=='left' else 2.)
            assert center[1]==(-.2 if side=='left' else .2)
            if side=='left':
                now[0]=6.
                raise ValueError('左肘/膝自碰撞')
            now[0]=8.;ctx.checkpoint()
            return dict(segments=['right complete pregrasp, grasp, lift and return'])
        return NS(plan=plan,candidate_failures=[dict(reason='collision')] if side=='left' else [])
    pool.native_planner=planner
    result=pool.plan({'unchanged':1},np.zeros(3),{}, {},observation_for_arm=observe,on_status=messages.append)
    assert calls==configs==observations==['left','right']
    assert contexts[0] is not contexts[1] and contexts[0].deadline==pytest.approx(5.1)
    assert contexts[1].deadline==10 and contexts[1].started==0
    assert result[0]['arm']=='right' and result[0]['segments']==['right complete pregrasp, grasp, lift and return']
    assert [row['status'] for row in result[0]['arm_selection']]==['unavailable','planned']
    assert result[1].preferred_grasp_yaw is None
    assert any('验证右手' in message for message in messages)


@pytest.mark.parametrize('current,expected',[
    ({'auto_arm_selection':False,'arm':'left'},['left']),
    ({'auto_arm_selection':False,'arm':'right'},['right']),
])
def test_fixed_hand_never_switches_after_failure(monkeypatch,current,expected):
    pool,now,calls,contexts=make_pool(monkeypatch,[])
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):raise ValueError('blocked path')
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(ValueError,match='选定手臂路径未通过'):
        pool.plan({},np.zeros(3),{},current,reach_hint={'arm':'right' if expected==['left'] else 'left'})
    assert calls==expected


def test_reach_hint_is_a_preference_not_a_fixed_hand(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[])
    def planner(cfg,ctx):
        side=cfg['arm'];calls.append(side)
        def plan(*args):
            if side=='right':raise ValueError('hinted right path blocked')
            return {'segments':['left complete path']}
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    result=pool.plan({},np.zeros(3),{}, {},reach_hint={'arm':'right','yaw_offset_deg':10.})
    assert calls==['right','left'] and result[0]['arm']=='left'


def test_definite_workspace_exclusion_is_not_overridden_for_fallback(monkeypatch):
    rows=[dict(arm='left',status='outside_workspace'),dict(arm='right',status='key_poses_passed',yaw_offset_deg=0.)]
    pool,now,calls,contexts=make_pool(monkeypatch,rows)
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):raise ValueError('right path blocked')
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(ValueError,match='选定手臂路径未通过'):pool.plan({},np.zeros(3),{}, {})
    assert calls==['right']


def test_cancel_after_first_failure_does_not_try_other_hand(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[]);cancelled=[False]
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):
            cancelled[0]=True
            raise ValueError('left blocked')
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(pc.PlanningCancelled):
        pool.plan({},np.zeros(3),{}, {},stopped=lambda:cancelled[0])
    assert calls==['left']


def test_precheck_rejects_stale_docking_arm_hint(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[dict(arm='left',status='key_poses_passed',yaw_offset_deg=10.)])
    result=pool.plan({},np.zeros(3),{}, {},reach_hint={'arm':'right','yaw_offset_deg':75.})
    assert calls==['left'] and result[0]['arm']=='left'


def test_model_load_is_not_mistaken_for_ik_timeout(monkeypatch):
    clock=[0.];real=pc.PlanningControl
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(r,'PlanningControl',lambda *args,**kw:real(*args,**{**kw,'clock':lambda:clock[0]}))
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


def test_configured_budget_reserves_second_hand_after_first_slice_timeout(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[]);messages=[];reach=[]
    def planner(cfg,ctx):
        side=cfg['arm'];calls.append(side);contexts.append(ctx)
        def plan(*args):
            if side=='left':now[0]=ctx.deadline+.01;ctx.checkpoint()
            now[0]=24.;ctx.checkpoint()
            return {'segments':['complete right path']}
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    result=pool.plan({},np.zeros(3),{}, {'stationary_plan_budget_sec':25.},on_status=messages.append,on_reach=reach.append)
    assert calls==['left','right'] and result[0]['arm']=='right'
    assert contexts[0].deadline==pytest.approx(12.6) and contexts[1].deadline==25.
    assert [a['status'] for a in result[0]['arm_selection']]==['timeout','planned']
    assert reach[0]['stationary_budget_seconds']==25. and any('25秒预算' in s for s in messages)


def test_fixed_hand_receives_full_configured_budget_and_global_cancel_is_terminal(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[],finish_time=24.)
    result=pool.plan({},np.zeros(3),{}, {'stationary_plan_budget_sec':25.,'auto_arm_selection':False,'arm':'right'})
    assert result[0]['arm']=='right' and contexts[0].deadline==25. and calls==['right']
    now[0]=0.;calls.clear();cancelled=[False]
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):cancelled[0]=True;ctx.checkpoint()
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    with pytest.raises(pc.PlanningCancelled):
        pool.plan({},np.zeros(3),{}, {'stationary_plan_budget_sec':25.},stopped=lambda:cancelled[0])
    assert calls==['left']


@pytest.mark.parametrize('budget',[0.,-1.,float('inf'),float('nan'),True])
def test_invalid_total_budget_is_rejected_before_planning(monkeypatch,budget):
    pool,now,calls,contexts=make_pool(monkeypatch,[])
    with pytest.raises(ValueError):pool.plan({},np.zeros(3),{}, {'stationary_plan_budget_sec':budget})
    assert not calls


def test_child_status_reports_parent_budget_and_uses_same_clock():
    now=[0.];messages=[];parent=pc.PlanningControl(timeout_sec=25.,clock=lambda:now[0])
    now[0]=3.;child=r.slice_control(parent,4.,messages.append)
    now[0]=5.;child.checkpoint()
    assert child.started==3. and child.deadline==7.
    assert messages[-1].endswith('原地累计5.0/25秒')
    now[0]=7.
    with pytest.raises(pc.PlanningTimeout,match='本阶段规划达到4秒'):child.checkpoint()


def test_complete_validated_plan_is_not_discarded_for_crossing_only_hand_slice(monkeypatch):
    pool,now,calls,contexts=make_pool(monkeypatch,[])
    def planner(cfg,ctx):
        calls.append(cfg['arm'])
        def plan(*args):
            now[0]=ctx.deadline+.01
            return {'segments':['already fully validated path']}
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    result=pool.plan({},np.zeros(3),{}, {'stationary_plan_budget_sec':25.})
    assert calls==['left'] and result[0]['segments']==['already fully validated path']
