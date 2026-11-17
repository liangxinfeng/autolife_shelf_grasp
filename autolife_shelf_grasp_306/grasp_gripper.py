"""DM夹爪抓取：按时间输出有界位置目标，避免反馈推进停顿被误判为接触。

原厂整段抓取和放置共用。仍通过原厂夹爪话题；不改电流/力矩/驱动响应。
开合按原有受限速度发送至目标，位置残差只记录，不延长或中断动作。
进入物体/退出物体所需净开口仍按真实路径要求检查，不宣称夹牢或释放成功。
"""
from collections import deque
import math
import time
import numpy as np
from .arm_selection import gripper_name
from .grip_width import load_curve,opening_clears_path


class ClosingEvidence:
    def __init__(self,start,target,object_width_mm,widths,positions):
        self.start=float(start);self.target=float(target);self.object_width=float(object_width_mm)
        self.widths=widths;self.positions=positions;self.samples=deque();self.sent_at=None

    def update(self,now,feedback,command):
        if abs(command-self.target)>1e-6:
            self.samples.clear();self.sent_at=None;return None
        if self.sent_at is None:self.sent_at=now
        self.samples.append((now,float(feedback)))
        while len(self.samples)>1 and now-self.samples[1][0]>=.6:self.samples.popleft()
        if len(self.samples)<5 or now-self.samples[0][0]<.6:return None
        if np.ptp([v for _,v in self.samples])>.8:return None
        if abs(feedback-self.target)<=2.:return 'at_target'
        nominal=float(np.interp(feedback,self.positions[::-1],self.widths[::-1]))
        # 两指仍比物品宽，不能当成“已经接触”；曾经114/203便停的情况在此继续闭合。
        if (self.target>self.start and feedback-self.start>8. and
                nominal<=self.object_width+1.):
            return 'contact_unconfirmed'
        return None


def execute(node,cfg,arm,state,checked,clock,hold,report,object_width_m):
    targets=state['targets'];jaw=gripper_name(arm)
    start=checked();begin=clock();closing=state['closing'];goal=float(targets[jaw])
    lo=float(cfg.get('gripper_open_deg',10.));hi=float(cfg.get('gripper_closed_deg',330.))
    if not lo<=goal<=hi:raise ValueError('DM夹爪目标超出配置端点')
    speed=float(cfg.get('gripper_speed_scale',.5))
    if not math.isfinite(speed) or not .1<=speed<=1.:raise ValueError('DM夹爪速度配置无效')
    rate=240.*speed  # 原厂位置量/秒；50%时120，不是角速度，也不是夹紧力。
    duration=max(.15,1.5*abs(goal-start[jaw])/rate)
    widths,positions,_=load_curve(cfg)
    evidence=ClosingEvidence(start[jaw],goal,object_width_m*1000,widths,positions) if closing else None
    # 放置不等精确位置目标：开口覆盖物品及退出间隙即可衔接收回。
    release_width=state.get('release_width_mm')
    opening_width=state.get('required_opening_mm')
    if opening_width is not None:
        margin=float(cfg.get('grasp_opening_margin_mm',10.))
        if not math.isfinite(margin) or not 0<margin<=30:raise ValueError('抓取预张开余量应为0～30毫米（不含0）')
        expected=min(object_width_m*1000+margin,float(widths[-1]))
        if (closing or not math.isfinite(opening_width) or not math.isclose(opening_width,expected,abs_tol=1e-6)):
            raise ValueError('抓取预张开计划与物品宽度不一致')
    clear_width=release_width if release_width is not None else opening_width
    target_width=float(np.interp(goal,positions[::-1],widths[::-1]))
    if clear_width is not None and (closing or not math.isfinite(clear_width) or not 0<clear_width<=target_width):
        raise ValueError('物品释放所需开口超出夹爪范围，请重新规划')
    minimum=state.get('minimum_opening_mm')
    if minimum is None and clear_width is not None:
        minimum=object_width_m*1000 if object_width_m>0 else clear_width
    if minimum is not None and (not math.isfinite(minimum) or not 0<minimum<=clear_width or minimum+1e-6<object_width_m*1000):
        raise ValueError('路径所需夹爪净开口无效，请重新规划')
    effective_clear_width=minimum
    last_send=-float('inf');last_report=-float('inf');command=start[jaw]
    # 本执行器独占夹爪指令。其他手的夹爪保持本段开始时的实际开度。
    node._active_goals.pop('grippers',None);node.progress_watch.pop('grippers',None)
    while True:
        q=checked();now=clock()
        if not all(math.isfinite(float(q[n])) for n in targets):raise ValueError('夹爪反馈无效')
        if now-last_send>=1./60:
            t=min(1.,max(0.,(now-begin)/duration))
            command=start[jaw]+(goal-start[jaw])*(t*t*(3.-2.*t))
            values={n:(command if n==jaw else start[n]) for n in targets}
            node._publish_vendor(node._gripper_pub,{
                'left_gripper_target_joints_position':[values['Joint_Left_Gripper']],
                'right_gripper_target_joints_position':[values['Joint_Right_Gripper']]})
            node._last_operator_activity=time.monotonic();last_send=now
        result=evidence.update(now,q[jaw],command) if closing else None
        issued=abs(command-goal)<1e-6 and now-begin>=duration
        if now-last_report>=.2 or issued:
            report(dict(targets_deg=targets,commanded_deg=command,measured_deg={n:q[n] for n in targets},
                nominal_width_mm=float(np.interp(q[jaw],positions[::-1],widths[::-1])),
                release_width_mm=release_width,required_opening_mm=opening_width,
                minimum_opening_mm=minimum,effective_clear_width_mm=effective_clear_width,
                closing=closing,time=now,phase='command_complete' if issued else 'ramping',
                arrival_required=False,position_error=float(q[jaw]-goal),physical_grip_verified=False))
            last_report=now
        if issued:
            # 精确位置量与夹住/松开并不等价；不再等待目标残差归零。
            # 抓放路径有实际净开口要求时，只核对必要间隙，不要求全开90mm。
            actual_width=float(np.interp(q[jaw],positions[::-1],widths[::-1]))
            if not closing and minimum is not None and not opening_clears_path(actual_width,minimum):
                hold(node,['grippers'])
                raise ValueError(f'当前夹爪净开口{actual_width:.1f}mm小于路径所需{minimum:.1f}mm')
            if not closing:hold(node,['grippers'])
            if release_width is not None:return 'released'
            if opening_width is not None:return 'opening_clear'
            # 闭合后的原目标必须继续保持，不能用受物体阻挡的实际开口改写。
            return result or 'command_completed_unverified'
