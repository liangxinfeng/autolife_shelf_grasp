"""放置退离后的双臂复位：同一运动会话，腰腿、颈部与夹爪均不改目标。"""
import numpy as np
from autolife_handeye_calibration_306.core import ARMS,BODY,NECK
from autolife_handeye_calibration_306.verification_trajectory import JointPath
from .vendor_grasp_trajectory import timed_samples,NativeTrajectoryAdapter,validate_reset_reference
from .grasp_speed import checked_percent

NAMES=ARMS['left']+ARMS['right']


def goal(cfg,source):
    result=dict(source)
    saved=cfg.get('workspace_reset_joints')
    for side in ('left','right'):
        values=([saved[n] for n in ARMS[side]] if saved is not None else cfg['initial_pose_deg'][side+'_arm'])
        if len(values)!=7 or not np.isfinite(values).all():raise ValueError('已保存的双臂复位姿态不完整')
        result.update(zip(ARMS[side],map(float,values)))
    return result


def trajectory(cfg,source):
    target=goal(cfg,source)
    if max(abs(target[n]-source[n]) for n in NAMES)<1e-6:
        return dict(source_q=dict(source),target_q=target,duration_sec=0.,waypoints=[dict(source)])
    path=JointPath(NAMES,[source,target])
    scale=checked_percent(cfg.get('grasp_speed_percent',100))/100
    speed=float(cfg.get('vendor_grasp_arm_speed_deg_s',60))*scale
    values,duration=timed_samples(path,np.full(14,speed),np.full(14,240*scale**2))
    return dict(source_q=dict(source),target_q=target,duration_sec=float(duration),
                waypoints=[{**source,**dict(zip(NAMES,map(float,row)))} for row in values])


def attach(planner,plan,table_guard):
    cfg=planner.cfg;start=plan['segments'][-1]['waypoints'][-1]
    reset=trajectory(cfg,start);checks=0
    from .joint_limits import load_arm_policy,check_pose
    import xml.etree.ElementTree as ET
    tree=ET.parse(cfg['urdf_path']).getroot()
    limits={n:tuple(np.rad2deg(float(tree.find(f"joint[@name='{n}']/limit").get(k))) for k in ('lower','upper')) for n in NAMES}
    policies=[load_arm_policy({**cfg,'arm':side},ARMS[side],limits) for side in ('left','right')]
    for a,b in zip(reset['waypoints'],reset['waypoints'][1:]):
        count=max(2,int(np.ceil(max(abs(b[n]-a[n]) for n in NAMES)/.1))+1)
        for t in np.linspace(0,1,count):
            planner.context.checkpoint()
            q={**start,**{n:a[n]+float(t)*(b[n]-a[n]) for n in NAMES}}
            for policy in policies:check_pose(policy,q)
            planner.guard.check(q);table_guard.check(q,False);checks+=1
    planner.guard.check(reset['target_q']);table_guard.check(reset['target_q'],False);checks+=1
    reset['checked_states']=checks
    plan.update(placement_completion='withdraw_then_arms_reset',arms_reset=reset)
    return plan


def validate(plan):
    reset=plan['arms_reset'];cfg=plan['config'];validate_reset_reference(cfg)
    start=plan['segments'][-1]['waypoints'][-1]
    expected=trajectory(cfg,start)
    if any(reset.get(k)!=v for k,v in expected.items()) or not reset.get('checked_states'):
        raise ValueError('双臂复位轨迹与放置退离终点不一致，请重新规划')
    return reset


def execute(node,plan,checked,clock,report,trajectory_function):
    reset=validate(plan)
    if reset['duration_sec']==0:return
    command=dict(source_q=reset['source_q'],config=plan['config'],controlled_joints=NAMES,arms_only_reset=True)
    adapter=NativeTrajectoryAdapter(node,command,checked,clock)
    last_report=[-float('inf')]
    def progress(actual,target):
        now=clock()
        if now-last_report[0]<.2:return
        last_report[0]=now
        report(dict(stage='放置完成，双臂复位',measured_deg={n:actual[n] for n in NAMES},commanded_deg={n:target[n] for n in NAMES}))
    adapter.on_sample=progress
    trajectory_function(adapter,[[q[n] for n in BODY+NAMES] for q in reset['waypoints']],
                        duration=reset['duration_sec'],excluded_joints_name=BODY,rate_hz=60.)
    deadline=clock()+5.
    while True:
        actual=checked()
        if max(abs(actual[n]-reset['target_q'][n]) for n in NAMES)<=.7:break
        if clock()>deadline:raise ValueError('放置已完成，双臂复位尚未到位')
