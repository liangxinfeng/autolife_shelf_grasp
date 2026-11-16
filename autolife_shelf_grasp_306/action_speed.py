"""动作串统一速度：百分比只应用一次，任务覆盖不改写动作库。"""
from copy import deepcopy
import math


def percent(value):
    value=float(value)
    if not math.isfinite(value) or not 25<=value<=150:
        raise ValueError('动作整体速度应在25～150%之间')
    return value


def compose(name,rows,speed):
    from .teaching_motion import normalize_draft
    # 新的整体速度以标准轨迹为基准，不叠乘旧编辑器藏在每帧中的倍率。
    frames=[{**deepcopy(r),'speed_percent':100.,'minimum_sec':.6} for r in rows]
    return normalize_draft(dict(name=name,frames=frames,overall_speed_percent=percent(speed)))


def editor_speed(draft):
    if 'overall_speed_percent' in draft:return percent(draft['overall_speed_percent'])
    # 旧编辑器按每帧保存速度；首次载入显示其实际配置，不再一律显示100%。
    return min(150.,max(25.,max((f.get('speed_percent',100.) for f in draft['frames']),default=100.)))


def apply(record,speed):
    from .task_flow import full_record
    from .teaching_motion import duration
    from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
    if speed is None:return deepcopy(record)
    result=full_record(record)
    result['speed_percent']=percent(speed)
    if 'trajectory' in result:
        # 连续录制100%为原始时序，25%为四倍时长；不重采样或丢弃录制点。
        result['trajectory']['speeds']=[100.]*len(result['trajectory']['speeds'])
    else:
        previous=None
        for row in result['keyframes']:
            target=row['joints_deg']
            if previous is None:row['transition_sec']=.6
            else:
                body={**target,**{n:previous[n] for n in GRIPPER_JOINTS}}
                row['transition_sec']=duration(previous,body,100.,.6)
            previous=target
    return full_record(result)
