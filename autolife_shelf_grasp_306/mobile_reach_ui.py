"""抓放共用补偿交接：准备保持→让出底盘→移动→重取景→原执行器。"""
import json
import time
from PyQt5 import QtCore as C
from .core import atomic_json
from .mobile_reach import direction_text


class MobileReach(C.QObject):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.pending=None;self.stage='';self.proc=None
        self.buffer=b'';self.result=None;self.execute_after=False
        self.timer=C.QTimer(self);self.timer.setInterval(30);self.timer.timeout.connect(self.tick)

    @property
    def active(self):return bool(self.stage)

    def offer(self,data):
        o=self.owner
        # 到位复查仍需移位时结束上一轮交接，新的提议才能再次启动。
        continuing=self.stage=='replan' and self.execute_after
        if self.stage=='replan':self.stage='';self.timer.stop()
        self.pending=data['proposal'];o.busy=False
        self.execute_after=bool(continuing or o.automatic_grasp_enabled() or o.demo_active or o.placement_active or o.workflow.active)
        proof='关键抓取位姿已通过，完整路径在到点后规划' if self.pending.get('reach_checked') else '完整操作路径已验证'
        o.status.setText(f'建议导航到“{self.pending["chain_name"]}”操作点；模型操作余量{self.pending["workspace_margin_m"]*100:.0f}厘米，{proof}。到点后重新取景。')
        o.navigation.map.operation_target=self.pending;o.navigation.map.update()
        o.controls()
        if self.execute_after:C.QTimer.singleShot(0,self.start)

    def controls(self):
        o=self.owner
        if self.pending and not self.active:
            o.execute.setText('导航到操作点并继续')
            o.execute.setEnabled(not o.cancel_pending and not o.motion and not o.navigation.inflight)

    def start(self):
        if self.active or not self.pending:return
        from .task_pause import defer_call
        if defer_call(self.owner,self.start):return
        o=self.owner
        if o.motion or o.cancel_pending or o.navigation.inflight:return
        try:
            from .operation_docking import validate
            validate(self.pending,self.pending['token'])
            self.execute_after=True
            o.workflow.mark_dispatch('mobile_prepare')
            if getattr(o,'parallel_prepare',None) is None:
                from .parallel_prepare_ui import start
                start(o,self.pending['kind'])
            self.preparation=o.parallel_prepare
            # 只交接底盘租约，原位置保持会话不退出，不重复使能/复位。
            atomic_json(o.run/'mobile_reach_request.json',self.pending)
            atomic_json(o.run/'mobile_reach_control.json',dict(prepare_token=self.preparation.token,operation='release_base'))
            self.stage='yield';self.last_tick=time.monotonic();self.deadline=self.last_tick+20;o.busy=True
            self.timer.start();o.controls()
        except Exception as exc:self.fail(str(exc))

    def tick(self):
        o=self.owner
        try:
            if (o.run/'stop').exists():self.cancel();return
            now=time.monotonic();elapsed=now-getattr(self,'last_tick',now);self.last_tick=now
            if (o.run/'task_pause').exists():self.deadline+=max(0.,elapsed);return
            if time.monotonic()>self.deadline:raise ValueError('底盘补偿交接超时')
            if self.stage in ('yield','return'):
                if getattr(o,'parallel_prepare',None) is not self.preparation:raise ValueError('补偿期间关节保持会话退出')
                path=o.run/'parallel_prepare_state.json'
                state=json.loads(path.read_text('utf8')) if path.exists() else {}
                if state.get('token')!=self.preparation.token:return
                if state.get('status')=='failed':raise ValueError(state['reason'])
                if time.monotonic()-state.get('stamp',0)>.5:return
                if self.stage=='yield' and state.get('status')=='base_released':self.move()
                elif self.stage=='return' and state.get('status')=='ready':
                    if not o.ready:raise ValueError('视觉后台不可用，补偿后未执行旧路径')
                    self.timer.stop();self.stage='replan'
                    resume=dict(self.pending['resume']);action=resume.pop('action')
                    resume['_mobile_done']=self.pending['token']
                    resume['_operation_visited']=list(resume.get('_operation_visited',[]))+[self.pending['source_map_pose'],self.pending['goal_pose']]
                    if self.pending.get('reach_checked'):
                        resume['_operation_reach_hint']={'arm':self.pending['arm'],'yaw_offset_deg':self.pending['reach_evidence']['yaw_offset_deg']}
                    o.send(action,**resume)
            elif self.stage=='moving' and not o.navigation.inflight:raise ValueError('操作停靠导航已退出，未使用旧路径')
        except Exception as exc:self.fail(str(exc))

    def move(self):
        o=self.owner;self.stage='moving';self.deadline=time.monotonic()+130
        request=self.pending
        o.workflow.mark_dispatch('operation_dock')
        o.navigation.command('navigate',name=request['chain_name'],map=request['map'],
            run_id=request['token'],leg='operation_dock',operation_token=request['token'],workflow=o.workflow.active)
        o.status.setText('正在导航到已验证的操作停靠点 · 保持双臂与夹持')

    def navigation_event(self,data):
        if data.get('leg')=='operation_dock' and not self.pending:
            return True
        if not self.pending or data.get('run_id')!=self.pending['token']:return False
        if data.get('event')=='nav_started':return True
        if self.stage!='moving':return True  # 迟到消息不能推进下一轮任务。
        if data.get('event') not in ('nav_done','nav_error'):return True
        self.owner.navigation.inflight=False
        if data.get('event')=='nav_error' or not data.get('success'):
            self.fail(data.get('text','操作停靠导航未完成'));return True
        atomic_json(self.owner.run/'mobile_reach_result.json',dict(token=self.pending['token'],status='completed'))
        atomic_json(self.owner.run/'mobile_reach_control.json',dict(prepare_token=self.preparation.token,operation='reacquire_base'))
        self.stage='return';self.deadline=time.monotonic()+6
        self.owner.status.setText('操作停靠已到位 · 更新物体坐标与抓放路径')
        return True

    def worker_event(self,data):
        if self.stage!='replan':return
        if data.get('event') in ('error','fatal','vision_unavailable'):
            self.fail(data.get('text','补偿后的重新识别未完成'));return
        if data.get('event') in ('planned','place_planned'):
            execute=data.get('event')=='planned' and self.execute_after and not self.owner.automatic_grasp_enabled()
            self.stage='';self.pending=None;self.timer.stop()
            self.owner.navigation.map.operation_target=None
            if execute:
                pid=data['plan_id']
                C.QTimer.singleShot(0,lambda:self.resume_grasp(pid))

    def resume_grasp(self,pid):
        o=self.owner
        if not o.busy and not o.cancel_pending and (o.pending_plan or {}).get('plan_id')==pid:
            o.start_execution(automatic=True)

    def cancel(self):
        previous=self.stage
        if previous=='moving':self.owner.navigation.command('stop')
        self.timer.stop();self.stage='';self.pending=None;self.execute_after=False
        if hasattr(self.owner,'navigation'):self.owner.navigation.map.operation_target=None
        if previous in ('yield','return','moving'):self.owner.busy=False
        if self.proc:(self.owner.run/'stop').touch()

    def fail(self,reason):
        self.cancel();self.owner.stop_motion(reason='底盘补偿：'+reason,source='mobile_reach')
        self.owner.busy=False;self.owner.controls()
