"""纠偏动作完成→释放底盘→下一航点，以及局部遮挡匹配的回归。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest


def test_completed_corrected_action_dispatches_next_waypoint_once(monkeypatch,tmp_path):
    import rclpy
    from PyQt5 import QtWidgets as Q,QtCore as C
    from autolife_shelf_grasp_306.action_studio import ActionExecutor
    from autolife_shelf_grasp_306.task_flow_ui import Workflow
    from autolife_shelf_grasp_306.task_flow import Cursor
    app=Q.QApplication.instance() or Q.QApplication([]);events=[]
    monkeypatch.setattr(rclpy,'spin_once',lambda *a,**kw:None)
    monkeypatch.setattr('time.monotonic',lambda:100.)
    monkeypatch.setattr(C.QTimer,'singleShot',lambda ms,fn:fn())
    monkeypatch.setattr('autolife_shelf_grasp_306.task_flow_ui.require_perception',lambda *a:None)
    class Flow:
        next=Workflow.next;advance=Workflow.advance;send_navigation=Workflow.send_navigation
        def __init__(self,owner):
            self.owner=owner;self.cursor=Cursor({'steps':[dict(kind='action',target='丢垃圾全程'),dict(kind='navigate',target='桌子1')]})
            self.phase='action';self.posture_resume=None;self.map='map'
        @property
        def active(self):return self.cursor.active
        def fail(self,text):pytest.fail(text)
    owner=NS(root=tmp_path,run=tmp_path,flow_label=NS(setText=lambda *a:None),controls=lambda:None,
        navigation=NS(proc=NS(state=lambda:C.QProcess.Running),command=lambda *a,**kw:events.append(('nav',kw))))
    flow=Flow(owner);e=ActionExecutor(owner);e.timer.stop()
    e.bridge=NS(_hardware_ready=True,_is_feedback_fresh=lambda:True,_current={},_action_state=None,
        _active_goals={},_hardware_prepare=None,_hardware_enabled=True)
    e.panel=NS(setEnabled=lambda *a:None);e.settled=lambda:True
    e.sync_session=NS(tick=lambda:None)
    # 独立采样器没有旧的sub成员，清理只能由guard自身负责。
    e.visual_base_guard=NS(last=True,check=lambda:None,close=lambda:events.append('base_released'))
    token=flow.cursor.token
    e.job=dict(token=token,phase='settle',started=99.,stable=99.,keep_session=True)
    e.completed.connect(lambda *a:Workflow.action_done(flow,*a))
    e.tick()
    assert e.job is None and e.manual_session and e.visual_base_guard is None
    assert events[0]=='base_released' and len(events)==2
    assert events[1][1]['name']=='桌子1' and flow.cursor.index==1
    Workflow.action_done(flow,token,True,'late duplicate')
    assert len(events)==2
    e._close_visual_base();assert len(events)==2
    e.bridge=None;e.sync_session=None;e.deleteLater()


def test_three_parts_match_ignores_one_occluder_but_not_two():
    from autolife_shelf_grasp_306.visual_object_memory import visible_part_scores
    ref=np.tile([1.,0.],(4,1));good=ref.copy();good[3]=[0.,1.]
    assert visible_part_scores([good],[ref],np.array([[.70]]),.72)[0,0]==1.
    good[2]=[0.,1.]
    assert visible_part_scores([good],[ref],np.array([[.70]]),.72)[0,0]==.70
    assert visible_part_scores([ref],[ref],np.array([[.1]]),.72)[0,0]==.1


def test_part_match_still_requires_unique_object():
    from autolife_shelf_grasp_306.visual_object_memory import ObjectAppearanceMemory,AmbiguousObject
    image=np.zeros((40,40,3),np.uint8);mask=np.ones((40,40),bool)
    p=NS(embed=lambda crops:np.tile([1.,0.],(len(crops),1)))
    memory=ObjectAppearanceMemory(p,{});views=[({},dict(image=image,mask=mask))]
    with pytest.raises(AmbiguousObject):
        memory.recognize_visible_parts(NS(view_ids=lambda _:['first']),'a',views,dict(image=image),
            [dict(mask=mask),dict(mask=mask)],np.array([[.7],[.7]]),.72,.06)


def test_visible_part_cache_tracks_view_deletion():
    from autolife_shelf_grasp_306.visual_object_memory import ObjectAppearanceMemory
    calls=[];image=np.zeros((40,40,3),np.uint8);mask=np.ones((40,40),bool)
    def embed(crops):calls.append(len(crops));return np.tile([1.,0.],(len(crops),1))
    memory=ObjectAppearanceMemory(NS(embed=embed),{});ids=['a','b']
    views=[({},dict(image=image,mask=mask))]*2
    store=NS(view_ids=lambda _:ids)
    memory.recognize_visible_parts(store,'a',views,dict(image=image),[dict(mask=mask)],np.array([[.7,.7]]),.72,.06)
    assert len(calls)==3
    ids=['a']
    memory.recognize_visible_parts(store,'a',views[:1],dict(image=image),[dict(mask=mask)],np.array([[.7]]),.72,.06)
    assert len(calls)==5


def test_foreground_and_reflection_do_not_outvote_distributed_surface():
    from test_visual_shape_position import scene,CFG
    from autolife_shelf_grasp_306.visual_shape_position import position_from_shape
    a=scene();b=scene(10);rng=np.random.default_rng(19)
    y,x=np.nonzero(b['mask']);bad=rng.random(len(y))<.40
    # 连续区域而非逐像素噪声，保留可用的深度邻域。
    b['depth'][70:100,140:160]=480.;b['depth'][140:175,170:195]=1100.
    d,s=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[.015,0,0],atol=.001)
    import json
    json.dumps(s)  # 诊断字段跨进程回到UI，不能含numpy布尔量。


def test_two_equally_plausible_depth_surfaces_not_arbitrarily_selected():
    from test_visual_shape_position import scene,CFG
    from autolife_shelf_grasp_306.visual_shape_position import position_from_shape
    a=scene();b=scene(10);b['depth'][120:180,140:200]=650.
    with pytest.raises(ValueError):position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
