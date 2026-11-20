"""调速贯穿保存、任务、镜像/纠偏和实际轨迹；不启动硬件。"""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from test_teaching import pose
from autolife_shelf_grasp_306.action_speed import compose,apply,editor_speed,percent
from autolife_shelf_grasp_306.pose_workspace_data import frame
from autolife_shelf_grasp_306.teaching_motion import compile_action
from autolife_shelf_grasp_306.teaching_trajectory import plan,evaluate
from autolife_shelf_grasp_306.teaching_library import TeachingLibrary
from autolife_shelf_grasp_306.task_flow import validate,Cursor


def rows():
    return [frame('前伸',pose(Joint_Right_Shoulder_Inner=v)) for v in (10,25,40)]


@pytest.mark.parametrize('speed',[25,50,70,100,120,150])
def test_editor_task_and_repeated_save_are_identical(tmp_path,speed):
    q=pose();d=compose('测试',rows(),speed);lib=TeachingLibrary(tmp_path)
    r=lib.save(d);times=[]
    for _ in range(3):
        loaded=lib.load('测试');assert editor_speed(loaded)==speed
        lib.save(compose('测试',loaded['frames'],editor_speed(loaded)))
        r=lib.records()['测试'];a,sa,_=plan(r,q,tmp_path)
        # 任务明确指定同一速度，不能再乘一次百分比。
        b,sb,_=plan(apply(r,speed),q,tmp_path)
        assert a==b
        assert np.allclose([s['seconds'] for s in sa],[s['seconds'] for s in sb])
        times.append(sum(s['seconds'] for s in sa))
    assert np.ptp(times)<1e-9


def test_first_move_and_all_body_segments_scale_once(tmp_path):
    r=compile_action(compose('测试',rows(),100));q=pose()
    fast,sf,_=plan(r,q,tmp_path);slow,ss,_=plan(apply(r,50),q,tmp_path)
    assert np.allclose([s['seconds'] for s in ss],np.array([s['seconds'] for s in sf])*2)
    for a,b in zip(sf,ss):
        for u in (0,.25,.5,.75,1):
            assert evaluate(a,u)==pytest.approx(evaluate(b,u))
    assert sf[0]['seconds']<ss[0]['seconds']


def test_above_100_really_shortens_path_without_unbounded_derivatives(tmp_path):
    from autolife_shelf_grasp_306.teaching_motion import limits
    from control_center_joint_control.joint_definitions import ALL_JOINTS
    r=compile_action(compose('提速',rows(),100));entry=pose();durations=[]
    for speed in (100,120,150):
        staged,segments,_=plan(apply(r,speed),entry,tmp_path)
        durations.append(sum(s['seconds'] for s in segments))
        for segment in segments:
            for u in np.linspace(0,1,61):
                for order in (1,2,3):
                    measured=evaluate(segment,u,order)
                    for n in ALL_JOINTS:
                        assert abs(measured[n])<=limits(n)[order-1]*speed/100.+1e-5
        assert segments[-1]['target']==r['keyframes'][-1]['joints_deg']
    assert durations[2]<durations[1]<durations[0]


def test_gripper_order_and_explicit_dwells_unchanged(tmp_path):
    a=frame('夹取',pose(Joint_Right_Shoulder_Inner=20,Joint_Right_Gripper=180));a['hold_sec']=1.5
    r=compile_action(compose('夹取',[a],100));original=deepcopy(r)
    fast,sf,pf=plan(r,pose(),tmp_path);slow,ss,ps=plan(apply(r,50),pose(),tmp_path)
    assert pf==ps and len(pf)==2
    assert [f['hold_sec'] for f in fast['keyframes']]==[f['hold_sec'] for f in slow['keyframes']]
    assert np.allclose([s['seconds'] for s in ss],np.array([s['seconds'] for s in sf])*2)
    assert r==original


def test_legacy_follow_unchanged_but_explicit_speed_removes_hidden_slowdown():
    from autolife_shelf_grasp_306.pose_workspace_data import compose as old_compose
    old=compile_action(old_compose('慢动作',rows(),50));original=deepcopy(old)
    assert apply(old,None)==old
    quick=apply(old,100)
    assert sum(r['transition_sec'] for r in quick['keyframes'])<sum(r['transition_sec'] for r in old['keyframes'])
    assert old==original


@pytest.mark.parametrize('value',[0,24,151,float('nan'),float('inf')])
def test_invalid_speed_cannot_enter_task(value):
    with pytest.raises(ValueError):validate(dict(steps=[dict(kind='action',target='x',action_speed_percent=value)]))


def test_task_save_retry_and_actual_dispatch(monkeypatch):
    from autolife_shelf_grasp_306.task_flow_ui import Workflow
    r=compile_action(compose('x',rows(),50));saved=deepcopy(r);sent=[]
    c=Cursor(dict(steps=[dict(kind='action',target='x',action_speed_percent=90,execution_arm='left')]))
    c.stop('test');c.retry()
    w=NS(owner=NS(studio=NS(execute=lambda *a,**k:sent.append((a,k)))),actions={'x':r},cursor=c,
        action_hold_grippers=True,mark_dispatch=c.dispatch)
    Workflow.start_fixed_action(w)
    out=sent[0][0][1]
    assert out['speed_percent']==90 and out['retarget_arm']=='left'
    assert w.actions['x']==saved


def test_continuous_recording_speed_preserves_samples_and_original_time(tmp_path):
    from autolife_shelf_grasp_306.teaching_track import save_raw,NAMES
    from autolife_shelf_grasp_306.teaching_replay import edited_samples
    t=np.linspace(0,2,101);q=pose();values=np.array([[q[n] for n in NAMES]]*len(t))
    values[:,NAMES.index('Joint_Right_Shoulder_Inner')]=np.sin(np.pi*t/2)*3
    sha=save_raw(tmp_path,t,values)
    # 引用两个端点，保留完整原始采样。
    r=compile_action(compose('x',[frame('a',q),frame('b',q)],100))
    r['trajectory']=dict(sha256=sha,indices=[0,100],speeds=[100,100],minimums=[.6,.2],timing='recorded')
    a,av,_=edited_samples(apply(r,100),tmp_path);b,bv,_=edited_samples(apply(r,50),tmp_path)
    assert a[-1]==pytest.approx(2.) and b[-1]==pytest.approx(4.)
    assert np.allclose(av,bv)


def test_qt_speed_persists_when_reordering_and_saving(tmp_path):
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.task_flow_ui import FlowDialog
    app=Q.QApplication.instance() or Q.QApplication([])
    (tmp_path/'data').mkdir();o=Q.QWidget();o.root=tmp_path;o.ready=True;o.controls=lambda:None
    o.flow_retry=Q.QPushButton();o.flow_label=Q.QLabel();o.navigation=NS(data=dict(names=['桌子'],ready=True))
    o.studio=NS(records=lambda:{'动作':compile_action(compose('动作',rows(),70))})
    d=FlowDialog(o);d.add('action','动作',action_speed_percent=85);d.add('navigate','桌子')
    d.table.selectRow(0);d.move(1)
    assert d.record()['steps'][1]['action_speed_percent']==85
    d.library={'测试':d.record()};d.load('测试')
    cell=d.table.cellWidget(1,6);assert cell.value()==85
    cell.follow.setChecked(True);assert 'action_speed_percent' not in d.record()['steps'][1]
    d.done(0);o.close()
