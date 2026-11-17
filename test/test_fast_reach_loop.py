import json,time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import operation_docking as dock,reach_precheck as reach
from autolife_shelf_grasp_306.mobile_reach import RepositionRequired


def test_unresolved_hands_off_before_full_stationary_plan(monkeypatch):
    from autolife_shelf_grasp_306.arm_selection import ArmPlannerPool
    monkeypatch.setattr(reach,'screen',lambda *a:[dict(arm='left',status='unresolved')])
    pool=ArmPlannerPool({'arm':'left','auto_arm_selection':False,'grasp_backend':'vendor_ik_trajectory'},None,None,None)
    pool.native_planner=lambda *a:pytest.fail('移位决定之前不应完整规划')
    def handoff(result):
        assert result['status']=='reposition_review'
        raise RepositionRequired({'reason':'better_pose'})
    with pytest.raises(RepositionRequired):pool.plan({},np.zeros(3),{}, {},on_reach=handoff)


def test_docking_only_solves_key_poses_not_full_trajectory(monkeypatch,tmp_path):
    ctx=dict(pose=[0,0,0],samples=[],chain={'name':'桌'},map='map',chain_sha256='a'*64)
    row=dict(candidate=dict(x=.2,y=0,yaw=0,index=1),arm='right',point=np.array([.5,0,.7]),T=np.eye(4),score=.1,margin_m=.2)
    monkeypatch.setattr(dock,'context',lambda *a:dict(ctx))
    monkeypatch.setattr(dock,'arm_envelope',lambda *a:(np.zeros(3),1.))
    monkeypatch.setattr(dock,'shortlist',lambda *a:[row])
    monkeypatch.setattr(dock,'navigation_filter',lambda r,c,rows,s:rows)
    monkeypatch.setattr(reach,'model_probe',lambda p:p)
    monkeypatch.setattr(reach,'native_key_poses',lambda *a:dict(status='key_poses_passed',checked_key_poses=3,yaw_offset_deg=10))
    planner=NS(plan=lambda *a:pytest.fail('停靠前不得计算完整抓取路径'))
    pool=NS(fk=NS(fk=lambda *a:np.eye(4)),arm_config=lambda *a:({'arm':'right'},np.zeros(3)),native_planner=lambda *a:planner)
    p=dock.grasp(tmp_path,{},pool,{},[.7,0,.7],{},dict(auto_arm_selection=False,arm='right'),[0,0,0],lambda:False,lambda _:None,None)
    assert p['reach_checked'] and not p['path_checked']
    dock.validate(p,p['token'])
    p['reach_evidence']['status']='unresolved'
    with pytest.raises(ValueError):dock.validate(p,p['token'])


def test_tried_points_cannot_cycle_back():
    candidate=dict(x=.3,y=0,yaw=0,index=1)
    ctx=dict(pose=[0,0,0],samples=[candidate])
    assert dock.shortlist(ctx,np.eye(4),{'right':[.8,0,.3]},{'right':(np.array([0,0,.3]),.8)})
    ctx['visited']=[[.3,0,0]]
    assert not dock.shortlist(ctx,np.eye(4),{'right':[.8,0,.3]},{'right':(np.array([0,0,.3]),.8)})


def test_second_reposition_can_start_and_preserves_manual_continue():
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.mobile_reach_ui import MobileReach
    app=Q.QApplication.instance() or Q.QApplication([]);o=Q.QWidget()
    o.automatic_grasp_enabled=lambda:False;o.demo_active=o.placement_active=False;o.workflow=NS(active=False)
    o.status=NS(setText=lambda _:None);o.navigation=NS(map=NS(update=lambda:None));o.controls=lambda:None
    m=MobileReach(o);m.stage='replan';m.execute_after=True;calls=[]
    m.start=lambda:calls.append(m.stage)
    m.offer({'proposal':dict(chain_name='桌',workspace_margin_m=.2,reach_checked=True)})
    app.processEvents()
    assert calls==[''] and m.execute_after
    m.cancel();o.deleteLater();app.processEvents()


def test_return_reobserves_and_carries_visited_history(tmp_path):
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.mobile_reach_ui import MobileReach
    app=Q.QApplication.instance() or Q.QApplication([]);o=Q.QWidget();o.run=tmp_path;o.ready=True
    o.parallel_prepare=NS(token='prepare');calls=[];o.send=lambda *a,**k:calls.append((a,k))
    m=MobileReach(o);m.preparation=o.parallel_prepare;m.stage='return';m.deadline=time.monotonic()+6
    m.pending=dict(token='current',source_map_pose=[0,0,0],goal_pose=[.2,0,0],resume=dict(action='plan',id=3,_operation_visited=[[-.2,0,0]]))
    (tmp_path/'parallel_prepare_state.json').write_text(json.dumps(dict(token='prepare',status='ready',stamp=time.monotonic())))
    m.tick()
    assert m.stage=='replan' and calls[0][1]['_mobile_done']=='current'
    assert calls[0][1]['_operation_visited']==[[-.2,0,0],[0,0,0],[.2,0,0]]
    m.timer.stop();o.deleteLater();app.processEvents()
