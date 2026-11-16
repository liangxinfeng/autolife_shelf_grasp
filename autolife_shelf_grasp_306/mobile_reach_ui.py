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
        o=self.owner;self.pending=data['proposal'];o.busy=False
        self.execute_after=bool(o.automatic_grasp_enabled() or o.demo_active or o.placement_active)
        o.status.setText(f'目标偏远：建议底盘{direction_text(self.pending)}（同步平移），再重新取景抓放；补偿通道已检查，到位后进入原规划。')
        o.controls()
        if self.execute_after:C.QTimer.singleShot(0,self.start)

    def controls(self):
        o=self.owner
        if self.pending and not self.active:
            o.execute.setText('补偿并抓取')
            o.execute.setEnabled(not o.cancel_pending and not o.motion and not o.navigation.inflight)

    def start(self):
        if self.active or not self.pending:return
        from .task_pause import defer_call
        if defer_call(self.owner,self.start):return
        o=self.owner
        if o.motion or o.cancel_pending or o.navigation.inflight:return
        try:
            from .mobile_reach import validate_request
            validate_request(self.pending,self.pending['token'])
            self.execute_after=True
            if getattr(o,'parallel_prepare',None) is None:
                from .parallel_prepare_ui import start
                start(o,self.pending['kind'])
            self.preparation=o.parallel_prepare
            # 只交接底盘租约，原位置保持会话不退出，不重复使能/复位。
            atomic_json(o.run/'mobile_reach_request.json',self.pending)
            atomic_json(o.run/'mobile_reach_control.json',dict(prepare_token=self.preparation.token,operation='release_base'))
            self.stage='yield';self.deadline=time.monotonic()+20;o.busy=True
            self.timer.start();o.controls()
        except Exception as exc:self.fail(str(exc))

    def tick(self):
        o=self.owner
        try:
            if (o.run/'stop').exists():self.cancel();return
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
                    o.send(action,**resume)
            elif self.stage=='moving' and self.proc is None:raise ValueError('底盘补偿进程已退出')
        except Exception as exc:self.fail(str(exc))

    def move(self):
        o=self.owner;self.stage='moving';self.deadline=time.monotonic()+30;self.result=None;self.buffer=b''
        p=C.QProcess(self);self.proc=p
        p.readyReadStandardOutput.connect(self.read);p.readyReadStandardError.connect(self.log)
        p.finished.connect(self.finished);p.errorOccurred.connect(self.error)
        p.start('/home/ubuntu/ros2_ws/venvs/s3_perception/bin/python',['-m','autolife_shelf_grasp_306.base_execution',
            '--run',str(o.run),'--token',self.pending['token'],'--approach'])
        o.status.setText(f'底盘同步补偿：{direction_text(self.pending)} · 保持双臂和夹持')

    def read(self):
        if self.proc is None:return
        self.buffer+=bytes(self.proc.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:d=json.loads(line)
            except ValueError:continue
            if not self.pending or d.get('token')!=self.pending['token']:continue
            if d.get('event')=='done':self.result=d
            elif d.get('message'):self.owner.status.setText(d['message'])

    def log(self):
        if self.proc:
            with (self.owner.run/'mobile_reach.log').open('ab') as f:f.write(bytes(self.proc.readAllStandardError()))

    def error(self,error):
        if error==C.QProcess.FailedToStart:self.finished(-1)

    def finished(self,*args):
        if self.proc is None:return
        self.read();self.log();p=self.proc;self.proc=None;p.deleteLater()
        if not self.active:
            self.owner.busy=False;self.owner.cancel_pending=False;self.owner.controls();return
        try:
            if not self.result or self.result.get('status')!='completed':raise ValueError((self.result or {}).get('reason','底盘补偿未完成'))
            atomic_json(self.owner.run/'mobile_reach_control.json',dict(prepare_token=self.preparation.token,operation='reacquire_base'))
            self.stage='return';self.deadline=time.monotonic()+6
            self.owner.status.setText('底盘已到位 · 保持原使能，更新物体坐标和抓放路径')
        except Exception as exc:self.fail(str(exc))

    def worker_event(self,data):
        if self.stage!='replan':return
        if data.get('event') in ('error','fatal','vision_unavailable'):
            self.fail(data.get('text','补偿后的重新识别未完成'));return
        if data.get('event') in ('planned','place_planned'):
            execute=data.get('event')=='planned' and self.execute_after and not self.owner.automatic_grasp_enabled()
            self.stage='';self.pending=None;self.timer.stop()
            if execute:
                pid=data['plan_id']
                C.QTimer.singleShot(0,lambda:self.resume_grasp(pid))

    def resume_grasp(self,pid):
        o=self.owner
        if not o.busy and not o.cancel_pending and (o.pending_plan or {}).get('plan_id')==pid:
            o.start_execution(automatic=True)

    def cancel(self):
        previous=self.stage
        self.timer.stop();self.stage='';self.pending=None;self.execute_after=False
        if previous in ('yield','return'):self.owner.busy=False
        if self.proc:(self.owner.run/'stop').touch()

    def fail(self,reason):
        self.cancel();self.owner.stop_motion(reason='底盘补偿：'+reason,source='mobile_reach')
        self.owner.busy=False;self.owner.controls()
