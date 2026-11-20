"""自动任务的准备与感知并行；同进程直接交接给执行器，预览不调用。

抓取任务已确认双手空载：保持全身并张开双爪。放置只保持原夹持。
仅写原厂位置接口；任何停止、心跳失效或规划失败均退出会话。
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from .core import atomic_json, open_tolerance
from .motion_supervision import MotionSupervisor,ExecutionProgress,attempt_cleanup


def wait_body_ready(run,token,timeout=16.):
    """规划只等使能后的真实身体起点，不等夹爪张开。"""
    deadline=time.monotonic()+timeout
    while True:
        if (run/'stop').exists():raise ValueError('用户停止')
        path=run/'parallel_prepare_state.json'
        value=json.loads(path.read_text('utf8')) if path.exists() else {}
        if value.get('token')==token:
            if value.get('status')=='failed':raise ValueError(value['reason'])
            if value.get('status') in ('body_ready','ready') and 0<=time.monotonic()-value['stamp']<.5:return value
        if time.monotonic()>=deadline:raise ValueError('并行准备未就绪，未使用使能前旧姿态')
        time.sleep(.02)


def opening_ramp(cfg,start):
    """双爪同一时钟、独立距离；沿用原夹爪的有界速度和三次平滑。"""
    from .grip_width import load_curve
    goal=float(cfg.get('gripper_open_deg',10.));speed=float(cfg.get('gripper_speed_scale',.5))
    if not math.isfinite(speed) or not .1<=speed<=1.:raise ValueError('夹爪速度无效')
    duration={}
    for side in ('left','right'):
        name=f'Joint_{side.title()}_Gripper';widths,positions,_=load_curve(dict(cfg,arm=side))
        if not positions[-1]<=goal<=positions[0]:raise ValueError('张开目标超出本机标定范围')
        duration[side]=max(.15,1.5*abs(goal-start[name])/(240.*speed))
    def sample(elapsed,current):
        payload={};ready=True
        for side in ('left','right'):
            name=f'Joint_{side.title()}_Gripper';t=min(1.,max(0.,elapsed/duration[side]))
            payload[f'{side}_gripper_target_joints_position']=[start[name]+(goal-start[name])*t*t*(3.-2.*t)]
            if not math.isfinite(float(current[name])):raise ValueError('夹爪反馈无效')
            ready=ready and t>=1.
        return payload,ready
    return sample,max(duration.values())+4.


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--token',required=True);parser.add_argument('--kind',choices=('grasp','place'),required=True)
    args=parser.parse_args();run=Path(args.run);root=Path(args.root);node=session=base=preloader=None;handed=False
    failure=None;command={};preopen=False
    supervision=MotionSupervisor(run)
    execution_progress=ExecutionProgress(execution_id=args.token,
        on_update=lambda value:atomic_json(run/'execution_progress.json',value))
    from . import execution as executor
    import rclpy
    from .runtime_config import load_config
    status='starting';last_write=-float('inf');started=time.monotonic();paused_total=0.
    def motion_time():return time.monotonic()-paused_total
    def report(force=False):
        nonlocal last_write
        now=time.monotonic()
        if force or now-last_write>=.1:
            atomic_json(run/'parallel_prepare_state.json',dict(token=args.token,status=status,stamp=now,kind=args.kind,
                seconds=now-started,time=time.time()))
            last_write=now
    def pump():
        def callbacks():
            if node is not None:
                for _ in range(10):rclpy.spin_once(node,timeout_sec=0)
        supervision.poll(callbacks,session,base)
        if session is not None and session.active:node._last_operator_activity=time.monotonic()
        report();time.sleep(.005)
    def spin():
        nonlocal paused_total
        pump()
        if node is not None and (run/'task_pause').exists():
            begin=time.monotonic()
            executor.hold_groups(node,['left_arm','right_arm','neck','leg_waist']+(['grippers'] if args.kind=='grasp' else []))
            while (run/'task_pause').exists():pump()
            paused_total+=time.monotonic()-begin
    try:
        cfg=load_config(root)
        from .core import execution_blockers
        blockers=execution_blockers(cfg)
        if blockers:raise ValueError('；'.join(blockers))
        if cfg.get('grasp_backend')!='vendor_ik_trajectory':raise ValueError('当前运控不支持并行准备')
        from autolife_handeye_calibration_306.bridge import competing_controller
        if competing_controller():raise ValueError(competing_controller())
        from .held_receipt import has_held
        preopen=args.kind=='grasp' and not has_held(run)
        from .trajectory_preload import TrajectoryPreload
        from .vendor_grasp_trajectory import load_trajectory_function
        preloader=TrajectoryPreload(load_trajectory_function,cfg)
        # 首次原厂库导入可能长时间占用GIL；在取得运动会话前完成。
        # 感知工作进程仍并行运行，停止/心跳继续响应，不提前使能。
        function=preloader.result(spin);preloader.close();preloader=None
        rclpy.init(args=executor.bridge_ros_args(.3,30.,True,args.kind=='grasp',cfg))
        node=executor.CalibrationBridge();node.preserve_loaded_grippers=not preopen
        from .demo_protocol import StationaryBaseGuard
        deadline=time.monotonic()+1.
        while base is None:
            try:base=StationaryBaseGuard(node)
            except ValueError as exc:
                if '底盘控制权不可用' not in str(exc) or time.monotonic()>=deadline:raise
                spin()  # 只等导航释放租约，拿到控制权即继续，无固定延时。
        deadline=time.monotonic()+3.
        while not node._is_feedback_fresh() or base.last is None:
            spin()
            if time.monotonic()>deadline:raise ValueError('等待并行准备反馈超时')
        base.check()
        from .feedback_refresh import refresh,await_snapshot
        refreshed=refresh(node,spin);await_snapshot(node.snapshot,spin,refreshed)
        from .pose_manual_session import ManualHoldSession
        session=ManualHoldSession(node,preserve_gripper_targets=not preopen);session.requires_grasp_finish=True
        execution_progress.enter('preparing')
        execution_progress.dispatch('hold_command_started')
        session.enter_prepared(spin,refreshed)
        status='body_ready';report(True)
        if preopen:
            start=node.snapshot();ramp,timeout=opening_ramp(cfg,start);begin=motion_time();stable=None;last_send=-float('inf')
            while True:
                spin();q=node.snapshot();now=motion_time();payload,ready=ramp(now-begin,q)
                if now-last_send>=1./60:
                    execution_progress.enter('gripper_opening')
                    execution_progress.dispatch('opening_command_started')
                    node._publish_vendor(node._gripper_pub,payload);node._last_operator_activity=now;last_send=now
                if ready:break  # 目标按受限速度发送完成即交接，不等待精确位置。
                if now-begin>timeout:raise ValueError('并行张爪指令输出未完成')
            # 保留刚发送的张开目标；不能用稍滞后的反馈改写目标、提前锁住未全开的夹爪。
        status='ready';execution_progress.enter('prepared_waiting_plan');report(True)
        request=run/'parallel_prepare_execute.json'
        deadline=motion_time()+90.
        while True:
            spin()
            mobile=run/'mobile_reach_control.json'
            if mobile.exists():
                change=json.loads(mobile.read_text('utf8'))
                if change.get('prepare_token')==args.token:
                    if change.get('operation')=='release_base' and base is not None:
                        base.check();base.close();base=None
                        status='base_released';deadline=motion_time()+140.;report(True)
                    elif change.get('operation')=='reacquire_base' and base is None:
                        base=StationaryBaseGuard(node)
                        end=time.monotonic()+3.
                        while base.last is None:
                            spin()
                            if time.monotonic()>end:raise ValueError('补偿后等待底盘反馈超时')
                        base.check();status='ready';deadline=motion_time()+90.;report(True)
            if request.exists():
                if base is None:raise ValueError('底盘补偿尚未归还控制权，未执行手臂路径')
                command=json.loads(request.read_text('utf8'))
                if command.get('token')==args.token:break
            if motion_time()>deadline:raise ValueError('规划未提交执行，已退出并行准备')
        path=(run/command['filename']).resolve()
        if path.parent!=run.resolve():raise ValueError('执行路径不属于当前任务')
        plan=json.loads(path.read_text('utf8'))
        if plan.get('parallel_prepare_token')!=args.token:raise ValueError('规划不属于本次并行准备')
        if (args.kind=='place')!=(plan.get('kind')=='place'):raise ValueError('并行准备与动作类型不一致')
        # 同一进程、同一节点和同一会话；不会结束后再使能，不会同时存在两个写入者。
        prepared=dict(node=node,session=session,base_guard=base,trajectory_function=function,kind=args.kind,token=args.token,
            execution_progress=execution_progress,preopened=preopen)
        execution_args=['--plan',str(path),'--run',str(run),'--expected-plan-id',command['plan_id'],
            '--expected-product-id',command['product_id']]
        if command.get('execution_id'):
            execution_args.extend(['--execution-id',command['execution_id']])
        executor.main(execution_args,prepared=prepared)
        handed=True  # The executor completed its cleanup and terminal receipt.
    except Exception as exc:
        failure=exc
        print('并行准备停止：'+str(exc),flush=True)
    finally:
        cleanup_errors=[];diagnostic_errors=[]
        if preloader is not None:attempt_cleanup(cleanup_errors,'trajectory_preloader',preloader.close)
        if not handed:
            if node:
                if getattr(node,'_hardware_ready',False):
                    attempt_cleanup(diagnostic_errors,'hold_journal',lambda:execution_progress.dispatch('hold_command_started'))
                attempt_cleanup(cleanup_errors,'hold',lambda:executor.hold_groups(node,
                    ['left_arm','right_arm','neck','leg_waist']+(['grippers'] if args.kind=='grasp' else [])))
                if session:attempt_cleanup(cleanup_errors,'sync_close',lambda:session.close(preserve_grippers=not preopen))
                attempt_cleanup(cleanup_errors,'lock',lambda:node._lock('并行准备结束，保持当前位置'))
                attempt_cleanup(cleanup_errors,'destroy_node',node.destroy_node)
            if base:attempt_cleanup(cleanup_errors,'base_close',base.close)
            attempt_cleanup(cleanup_errors,'ros_shutdown',lambda:rclpy.shutdown() if rclpy.ok() else None)
        if failure is not None:
            record=dict(execution_progress.receipt('stopped',failure),status='stopped',reason=str(failure))
            if command.get('plan_id'):record['plan_id']=command['plan_id']
            if cleanup_errors:record['cleanup_errors']=cleanup_errors
            if diagnostic_errors:record['diagnostic_errors']=diagnostic_errors
            attempt_cleanup(diagnostic_errors,'prepare_state',lambda:atomic_json(run/'parallel_prepare_state.json',
                dict(record,token=args.token,status='failed',stamp=time.monotonic())))
            attempt_cleanup(diagnostic_errors,'execution_receipt',lambda:atomic_json(run/'execution_result.json',record))


if __name__=='__main__':main()
