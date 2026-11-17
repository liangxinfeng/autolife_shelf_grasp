import copy,math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_handeye_calibration_306.core import BODY,ARMS,NECK
from autolife_shelf_grasp_306 import placement_arm_reset as reset
from autolife_shelf_grasp_306 import docking_view as view
from autolife_shelf_grasp_306.vendor_grasp_trajectory import NativeTrajectoryAdapter


def pose():return {n:0. for n in BODY+ARMS['left']+ARMS['right']+NECK+['Joint_Left_Gripper','Joint_Right_Gripper']}


def test_reset_only_arms_uses_saved_current_reference():
    source=pose();source[BODY[0]]=-35;source[BODY[1]]=-70;source[NECK[1]]=30;source['Joint_Right_Gripper']=100
    saved={n:30 for n in source};saved[ARMS['right'][0]]=10
    cfg={'workspace_reset_joints':saved,'grasp_speed_percent':100}
    q=reset.goal(cfg,source)
    assert all(q[n]==source[n] for n in source if n not in reset.NAMES)
    assert all(q[n]==saved[n] for n in reset.NAMES)
    t=reset.trajectory(cfg,source)
    assert all(row[n]==source[n] for row in t['waypoints'] for n in source if n not in reset.NAMES)
    assert reset.trajectory(cfg,q)['duration_sec']==0


def test_reset_adapter_publishes_both_arms_only():
    q=pose();packets=[]
    p=dict(source_q=q,config={'arm':'right'},controlled_joints=reset.NAMES,arms_only_reset=True)
    node=NS(_whole_body_pub=None,_publish_vendor=lambda _,payload:packets.append(payload))
    a=NativeTrajectoryAdapter(node,p,lambda:q,lambda:0)
    left=[0.]*11;right=[0.]*11;left[4]=1.;right[4]=-1.
    a._set_target_body_joints_deg(left,right)
    from control_center_joint_control.joint_definitions import GROUPS
    assert set(packets[0])=={GROUPS[g]['command_key'] for g in ('left_arm','right_arm')}
    left[0]=1.;right[0]=1.
    with pytest.raises(ValueError,match='固定关节'):a._set_target_body_joints_deg(left,right)


def camera():
    T=np.eye(4);T[:3,:3]=np.array([[0,0,1],[-1,0,0],[0,-1,0]]);T[2,3]=1.
    return dict(camera_ground=T.tolist(),K=[[200,0,320],[0,200,240],[0,0,1]],D=[0]*5,width=640,height=480)


def test_facing_candidate_sees_target_when_old_heading_faces_away():
    product={'bbox_extents_m':[.04,.04,.04]}
    rows=view.candidate_options({'x':0,'y':-.2,'yaw':math.pi,'index':1},{'pose':[0,0,0]},np.eye(4),np.array([.7,0,.6]),product,camera())
    assert rows and all(abs(c['yaw']-math.pi)>1 for c,_,_ in rows)
    assert rows[0][2]['edge_margin_px']>38


def test_target_outside_vertical_fov_not_accepted():
    rows=view.candidate_options({'x':.6,'y':0,'yaw':0,'index':1},{'pose':[0,0,0]},np.eye(4),np.array([.7,0,.1]),{},camera())
    assert rows==[]


def test_budget_is_ten_seconds_including_precheck():
    from autolife_shelf_grasp_306.reach_precheck import STATIONARY_SECONDS,SCREEN_SECONDS
    assert STATIONARY_SECONDS==10 and SCREEN_SECONDS<=2


def test_localized_pose_reassociation_ignores_odom_drift(monkeypatch):
    from autolife_shelf_grasp_306 import mobile_reach as m,bottle
    monkeypatch.setattr(bottle,'object_points',lambda *a,**kw:(np.tile([.6,0,.7],(50,1)),None,None))
    original=dict(id=3,feature=[1,0],product=None,name='纸巾')
    ref=dict(target_base=[.7,0,.7],base_anchor=[0,0,0],source_map_pose=[10,2,0],arrival_map_pose=[10.1,2,0])
    value,diag=m.select_after_move(original,[dict(mask='valid',feature=[1,0])],{},np.eye(4),{},ref,[1,0,0])
    assert value['id']==3 and diag['position_error_m']<1e-8
    assert diag['method']=='localized_map_reprojection'
