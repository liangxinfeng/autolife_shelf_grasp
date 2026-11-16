"""夹爪目标残差不影响时序；取消/通信错误仍先于任何下一条指令。"""
import json,ast
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306.grasp_gripper import execute


@pytest.mark.parametrize('arm',['left','right'])
@pytest.mark.parametrize('closing,residual',[(False,3.3),(False,50.),(True,-30.),(True,-120.)])
def test_position_residual_does_not_delay_or_abort(tmp_path,arm,closing,residual):
    p=tmp_path/'curve.json';p.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={arm:[[0,330],[45,180],[90,10]]})))
    cfg=dict(arm=arm,gripper_width_calibration=str(p),gripper_speed_scale=.5)
    jaw='Joint_'+arm.title()+'_Gripper';other='Joint_'+('Left' if arm=='right' else 'Right')+'_Gripper'
    goal=220. if closing else 10.;start=13.3 if closing else 250.
    q={jaw:start,other:77.};sent=[];reports=[];held=[];now=[0.]
    node=NS(_active_goals={},progress_watch={},_gripper_pub=object())
    def publish(pub,data):
        sent.append(data)
        q[jaw]=max(10.,min(330.,data[arm+'_gripper_target_joints_position'][0]+residual))
    node._publish_vendor=publish
    def checked():now[0]+=.02;return dict(q)
    result=execute(node,cfg,arm,dict(targets={jaw:goal,other:77.},closing=closing,tolerance=2.),
        checked,lambda:now[0],lambda *a:held.append(a),reports.append,.055)
    assert result=='command_completed_unverified'
    assert now[0]<=1.5*abs(goal-start)/120+.08
    assert sent[-1][arm+'_gripper_target_joints_position']==[goal]
    assert all(v[('left' if arm=='right' else 'right')+'_gripper_target_joints_position']==[77.] for v in sent)
    assert bool(held)!=closing
    assert reports[-1]['arrival_required'] is False and not reports[-1]['physical_grip_verified']


def test_completion_receipt_preserves_closing_intent_on_handoff():
    from autolife_shelf_grasp_306 import execution
    tree=ast.parse(Path(execution.__file__).read_text('utf8'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='retain_grip' for t in n.targets))
    env=dict(vendor_grasp=True,record={'status':'completed'},contact='command_completed_unverified')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual retention condition>','exec'),env)
    assert env['retain_grip']


def test_reset_readiness_ignores_only_jaw_position_residual():
    from autolife_shelf_grasp_306.prepare import ready_after_place
    cfg={'workspace_reset_joints':{'Joint_Right_Elbow':-90.,'Joint_Right_Gripper':10.}}
    assert ready_after_place(cfg,{'Joint_Right_Elbow':-90.,'Joint_Right_Gripper':13.3})
    assert not ready_after_place(cfg,{'Joint_Right_Elbow':-85.,'Joint_Right_Gripper':13.3})
    assert not ready_after_place(cfg,{'Joint_Right_Elbow':-90.,'Joint_Right_Gripper':float('nan')})


@pytest.mark.parametrize('fault',[None,'body_moved','invalid_jaw','stop'])
def test_end_handoff_does_not_reintroduce_jaw_arrival_wait(fault):
    from autolife_shelf_grasp_306.pose_handoff import await_loaded_pose
    now=[10.];source={'Joint_Right_Elbow':-90.,'Joint_Right_Gripper':220.}
    current={'Joint_Right_Elbow':-85. if fault=='body_moved' else -90.,
        'Joint_Right_Gripper':float('nan') if fault=='invalid_jaw' else 135.}
    def poll():
        now[0]+=.02
        if fault=='stop':raise ValueError('用户停止')
    args=(lambda:current,poll,lambda:now[0],source,'结束交接','place',lambda:now[0],lambda v:None)
    if fault:
        with pytest.raises(ValueError):await_loaded_pose(*args,arm='right',require_jaw_stable=False)
    else:
        assert await_loaded_pose(*args,arm='right',require_jaw_stable=False)==current
        assert now[0]==pytest.approx(10.02)
