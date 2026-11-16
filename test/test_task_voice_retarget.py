from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from autolife_shelf_grasp_306.task_flow import validate,Cursor,full_record
from autolife_shelf_grasp_306.action_retarget import retarget
from autolife_shelf_grasp_306.pose_geometry import ModelGeometry
from autolife_shelf_grasp_306.pose_mirror import mirror_signs
from autolife_shelf_grasp_306.visual_reference import corrected_tcp
from autolife_shelf_grasp_306.task_tts_worker import cache_key
from control_center_joint_control.joint_definitions import ALL_JOINTS,JOINT_LIMITS,GROUP_ORDER
from autolife_handeye_calibration_306.core import ARMS,TORSO

def record():
    q={n:max(lo,min(hi,0.)) for n,(lo,hi) in JOINT_LIMITS.items()}
    q['Joint_Right_Elbow']=-30.;q['Joint_Left_Elbow']=-15.
    q['Joint_Right_Gripper']=180.;q['Joint_Left_Gripper']=10.
    return dict(groups=GROUP_ORDER,loop_count=1,speed_percent=70,keyframes=[dict(joints_deg=q,transition_sec=3.,hold_sec=0.,easing='smooth')])

def geometry():
    import yaml
    root=Path(__file__).parents[1];cfg=yaml.safe_load((root/'config/grasp.yaml').read_text())
    return ModelGeometry(cfg['urdf_path'])

def test_tts_and_arm_selection_survive_save_retry():
    r=validate(dict(steps=[dict(kind='tts',target='请稍等，正在取饮料'),dict(kind='action',target='取物',execution_arm='left')]))
    assert r['steps'][0]==dict(kind='tts',target='请稍等，正在取饮料')
    assert r['steps'][1]['execution_arm']=='left'
    c=Cursor(r);old=c.token;c.stop('test');c.retry();assert not c.complete(old)
    assert c.complete(c.token) and c.step['execution_arm']=='left'
    with pytest.raises(ValueError):validate(dict(steps=[dict(kind='tts',target='')]))
    with pytest.raises(ValueError):validate(dict(steps=[dict(kind='action',target='x',execution_arm='unknown')]))

def test_tts_cache_includes_speed_and_text():
    assert cache_key(' 你好 ')==cache_key('你好')
    assert cache_key('你好')!=cache_key('再见')
    assert cache_key('你好',1)!=cache_key('你好',1.2)
    with pytest.raises(ValueError):cache_key('')

@pytest.mark.parametrize('side',['left','right'])
def test_actual_urdf_mirror_and_inactive_arm_hold(tmp_path,side):
    r=record();original=deepcopy(r);current=deepcopy(r['keyframes'][0]['joints_deg']);g=geometry()
    origin='right' if side=='left' else 'left'
    out=retarget(r,side,current,g,tmp_path);q=out['keyframes'][0]['joints_deg']
    for a,b,sign in zip(ARMS[origin],ARMS[side],mirror_signs(g)):assert q[b]==pytest.approx(sign*current[a])
    for n in [*ARMS[origin],'Joint_'+origin.title()+'_Gripper']:assert q[n]==current[n]
    for n in set(q)-set(ARMS['left'])-set(ARMS['right'])-{'Joint_Left_Gripper','Joint_Right_Gripper'}:assert q[n]==current[n]
    assert r==original
    f=g.frames(current);new=g.frames(q);S=np.diag([1,-1,1,1]);torso=f[TORSO]
    src=np.linalg.inv(torso)@f['Link_'+origin.title()+'_Wrist_Lower_to_Gripper']
    dst=np.linalg.inv(torso)@new['Link_'+side.title()+'_Wrist_Lower_to_Gripper']
    assert np.allclose(dst[:3,3],(S@src)[:3,3],atol=.002)

def test_visual_binding_reflects_reference_then_uses_current_object(tmp_path):
    r=record();q=r['keyframes'][0]['joints_deg'];g=geometry();O=np.eye(4);O[:3,3]=[.4,-.2,.6]
    tool=g.frames(q)['Link_Right_Wrist_Lower_to_Gripper']@g.tool_transform('right')
    r['visual_bindings']={'0':dict(reference='a'*64,side='right',mode='position',taught_object_pose=O.tolist(),relative_tcp=(np.linalg.inv(O)@tool).tolist())}
    old=deepcopy(r);out=retarget(r,'left',q,g,tmp_path);b=out['visual_bindings']['0']
    mirrored=g.frames(out['keyframes'][0]['joints_deg'])['Link_Left_Wrist_Lower_to_Gripper']@g.tool_transform('left')
    reference=np.array(b['taught_object_pose']);new=reference.copy();new[:3,3]+=[.02,-.03,.01]
    actual=corrected_tcp(reference,new,b['relative_tcp'],b['mode'])
    assert b['side']=='left' and np.allclose(actual[:3,3],mirrored[:3,3]+[.02,-.03,.01])
    assert r==old

def test_continuous_mirror_keeps_original_times_and_library(tmp_path):
    from autolife_shelf_grasp_306.teaching_track import save_raw,load_raw,NAMES
    r=record();q=r['keyframes'][0]['joints_deg'];times=np.array([0.,.1,.2]);values=np.array([[q[n] for n in NAMES]]*3)
    values[1,NAMES.index('Joint_Right_Elbow')]+=1
    sha=save_raw(tmp_path,times,values)
    r['keyframes']=[deepcopy(r['keyframes'][0]),deepcopy(r['keyframes'][0])]
    r['trajectory']=dict(sha256=sha,indices=[0,2],speeds=[100,100],minimums=[.2,.2],timing='recorded')
    out=retarget(r,'left',q,geometry(),tmp_path);t,v=load_raw(tmp_path,out['trajectory']['sha256'])
    assert np.array_equal(t,times) and np.array_equal(load_raw(tmp_path,sha)[1],values)
    assert out['trajectory']['timing']=='recorded'
    assert np.ptp(v[:,NAMES.index('Joint_Left_Elbow')])==pytest.approx(1)
    assert np.ptp(v[:,NAMES.index('Joint_Right_Elbow')])==0


def test_task_editor_roundtrip_voice_and_arm_choice(tmp_path):
    from PyQt5 import QtWidgets as Q
    from types import SimpleNamespace as NS
    from autolife_shelf_grasp_306.task_flow_ui import FlowDialog
    app=Q.QApplication.instance() or Q.QApplication([])
    (tmp_path/'data').mkdir();o=Q.QWidget();o.root=tmp_path;o.ready=True;o.controls=lambda:None
    o.flow_retry=Q.QPushButton();o.flow_label=Q.QLabel();o.navigation=NS(data=dict(names=['桌子1'],ready=True))
    o.studio=NS(records=lambda:{'示教动作':record()})
    d=FlowDialog(o);d.add('tts','开始整理桌面');d.add('action','示教动作',execution_arm='left')
    rows=d.record()['steps'];assert rows[0]['target']=='开始整理桌面' and rows[1]['execution_arm']=='left'
    assert not d.table.cellWidget(0,2).isEnabled()
    d.table.selectRow(1);d.move(-1)
    assert d.record()['steps'][0]['execution_arm']=='left'
    d.done(0);o.close()


def test_wrong_source_side_does_not_silently_remove_object_binding(tmp_path):
    r=record();q=r['keyframes'][0]['joints_deg'];g=geometry();O=np.eye(4)
    r['visual_bindings']={'0':dict(reference='a'*64,side='left',mode='position',taught_object_pose=O.tolist(),relative_tcp=O.tolist())}
    with pytest.raises(ValueError,match='来源手臂'):
        retarget(r,'left',q,g,tmp_path)
