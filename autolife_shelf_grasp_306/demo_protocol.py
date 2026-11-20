"""Prevent release from stale, unrelated or incomplete demo requests."""
from .runtime_identity import identity
import json
from .arm_selection import require_same_arm,gripper_name,plan_arm


def check_base_anchor(current,reference):
    import math
    from .demo_config import wrap
    if len(current)!=3 or len(reference)!=3 or not all(math.isfinite(v) for v in [*current,*reference]):
        raise ValueError('桌面观察的底盘位置无效')
    if math.hypot(current[0]-reference[0],current[1]-reference[1])>.01 or abs(wrap(current[2]-reference[2]))>.02:
        raise ValueError('观察桌面后底盘位置改变，请重新识别')


def validate_place(plan,run):
    arm=plan_arm(plan);jaw=gripper_name(arm)
    if plan.get('placement_mode')=='standalone':
        from .placement_request import standalone_source,check_held_gripper
        context,grasp,record=standalone_source(run)
        if plan.get('placement_request_id')!=context['run_id']:
            raise ValueError('放置请求已失效，请重新确认放置')
        if plan_arm(grasp)!=arm:raise ValueError('放置与抓取手臂不一致')
        check_held_gripper(plan['source_q'],record)
    else:
        from .held_receipt import current_payload
        record=current_payload(run)
        context=json.loads((run/'demo_run.json').read_text())
        base=json.loads((run/'demo_base_result.json').read_text())
        if base.get('status')!='completed' or (context.get('run_id') and (base.get('run_id')!=context['run_id'] or
                base.get('leg')!='to_front' or base.get('name')!=context['front'] or base.get('map')!=context['map'])):
            raise ValueError('尚未确认本轮导航回到所选前台航点')
    require_same_arm(plan,record)
    if (plan.get('place_protocol_version') not in (1,2,3,4) or not plan.get('table_height_checked') or
            record.get('status')!='completed' or
            record.get('plan_id')!=context['grasp_plan_id'] or plan.get('parent_plan_id')!=context['grasp_plan_id'] or
            plan['product']['id']!=context['product_id'] or plan.get('demo_settings')!=context['settings']):
        raise ValueError('放置来源与本轮演示不一致')
    if plan.get('kind')=='observe':
        import math
        attempt=json.loads((run/'observation_attempt.json').read_text())
        if attempt.get('plan_id')!=plan.get('plan_id') or attempt.get('run_id')!=context.get('run_id') or (run/'demo_observe_result.json').exists():
            raise ValueError('观察动作不属于当前请求或已经完成，禁止重复抬手')
        if plan.get('observation_protocol_version')!=1 or [s.get('phase') for s in plan['segments']]!=['observe_lift']:
            raise ValueError('观察路径无效，禁止夹爪操作')
        held=float(plan.get('held_gripper_deg',float('nan')))
        previous=float(record.get('q',{}).get(jaw,float('nan')))
        if not math.isfinite(held) or not math.isfinite(previous):
            raise ValueError('观察路径缺少有效夹爪反馈')
        if abs(float(plan['source_q'][jaw])-held)>1e-6:
            raise ValueError('观察路径夹爪起始状态不一致')
        return
    expected=(['place_approach','place_lower','place_return'] if plan.get('place_protocol_version') in (3,4)
              else ['place_approach','place_lower','place_release_retreat','place_return'])
    if [s.get('phase') for s in plan['segments']]!=expected:
        raise ValueError('放置阶段不完整，不能松爪')
    if plan.get('place_protocol_version')==4:
        from .vendor_place_trajectory import validate_blocks
        validate_blocks(plan)
    if plan.get('place_protocol_version') in (2,3):
        # Reject malformed/new-format speeds BEFORE unlocking or entering SYNC.
        from autolife_handeye_calibration_306.verification_trajectory import JointPath
        from autolife_handeye_calibration_306.core import ARMS
        from .motion import configure_grasp_path,PathSpeedProfile
        import math
        for segment in plan['segments']:
            path=configure_grasp_path(JointPath(ARMS[arm],segment['waypoints']),plan['config'])
            cap=float(segment.get('path_speed_cap',float('nan')))
            if not math.isfinite(cap) or not 0<cap<=path.max_speed+1e-6:
                raise ValueError('放置速度上限无效，请重新规划')
            path.max_speed=min(path.max_speed,cap)
            PathSpeedProfile(path,segment.get('path_speed_profile',[]))
            switch=float(segment.get('transfer_from_s',path.length))
            if not math.isfinite(switch) or not 0<=switch<=path.length:
                raise ValueError('退出阶段速度切换位置无效')


class StationaryBaseGuard:
    """抓取、放置与纠偏共用底盘租约；独立读取里程计，不发送速度。"""
    def __init__(self,node):
        import time,math
        from control_center_chassis_guard import ChassisControlLease
        from .base_feedback import BaseOdometryReader
        self.time=time;self.math=math;self.origin=None;self.reader=None
        self.lease=ChassisControlLease(domain_id=identity().domain_id,robot_id=identity().robot_id)
        if not self.lease.acquire(timeout_sec=0):
            self.lease.shutdown();raise ValueError('动作准备时底盘控制权不可用')
        try:self.reader=BaseOdometryReader(node,f'/topic_gv_wheel_odom_{identity().topic_suffix}')
        except BaseException:
            self.lease.release();self.lease.shutdown();raise

    @property
    def last(self):
        return self.reader.latest()

    def check(self):
        from .demo_config import wrap
        if not self.lease.held:raise ValueError('动作期间底盘控制权丢失')
        # 先直接取最新消息，再判断年龄。主节点处理慢不再等于底盘断流。
        sample=self.last
        if sample is None or self.time.monotonic()-sample[0]>.3:
            raise ValueError('动作期间底盘实时反馈中断')
        age=self.reader.source_age
        if age is None or not self.math.isfinite(age) or age<-.1 or age+self.time.monotonic()-sample[0]>.3:
            raise ValueError('底盘里程计时间无效或数据源未更新')
        if not all(self.math.isfinite(v) for v in sample):raise ValueError('底盘反馈无效')
        if sample[4]>.015 or sample[5]>.025:raise ValueError('动作期间底盘未停稳')
        if self.origin is None:self.origin=sample
        if self.math.hypot(sample[1]-self.origin[1],sample[2]-self.origin[2])>.01 or abs(wrap(sample[3]-self.origin[3]))>.02:
            raise ValueError('动作期间底盘位置改变')

    def close(self):
        try:
            if self.reader is not None:self.reader.close()
        finally:self.lease.release();self.lease.shutdown()
