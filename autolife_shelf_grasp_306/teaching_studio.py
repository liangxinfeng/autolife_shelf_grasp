"""VR teaching UI facade; passive capture and exclusive task execution."""
import fcntl,json,os,time
from pathlib import Path
from .runtime_config import load_config
from PyQt5 import QtCore as C,QtWidgets as Q
from .teaching_feedback import FeedbackState,create_monitor
from .teaching_runtime import TeachingRuntime
from .teaching_library import TeachingLibrary


class TeachingStudio(Q.QWidget):
    completed=C.pyqtSignal(str,bool,str)
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.panel=self;self.monitor=None;self.lease=None
        self.cfg=load_config(owner.root);self.parallel_action=None
        self.robot_id=int(self.cfg['robot_id']);self.feedback=FeedbackState()
        self.library=TeachingLibrary(owner.root/'data/fixed_actions')
        from .visual_action_client import VisualClient
        self.visual=VisualClient(self)
        from .teaching_client import TeachingClient
        self.recorder=TeachingClient(self.robot_id,self.library.directory,owner.run,self)
        self.runtime=TeachingRuntime(self.robot_id,owner.run,self)
        from .action_studio import ActionExecutor
        self.engine=ActionExecutor(owner);self.engine.setParent(self);self.engine.hide()
        self.engine.completed.connect(self._completed)
        from .teaching_composer import TeachingComposer
        self.editor=TeachingComposer(self);layout=Q.QVBoxLayout(self);layout.setContentsMargins(8,6,8,6);layout.addWidget(self.editor)
        self.runtime.changed.connect(self.editor.update_runtime)
        self.recorder.changed.connect(self.editor.update_recorder)
        self.recorder.failed.connect(self.editor.recorder_failed)
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)
    def ensure(self):
        self.recorder.ensure()
        if self.monitor is None:self.monitor=create_monitor(self.robot_id,self.feedback)
        from ament_index_python.packages import get_package_share_directory
        model = Path(get_package_share_directory('control_center_joint_control'))/'urdf/robot_v3_1.urdf'
        self.editor.ensure_model(str(model))
    def records(self,names=None):return self.library.records(names)
    def moving(self):return self.parallel_action is not None or self.visual.busy or self.runtime.busy or self.engine.moving()
    def _acquire(self):
        if self.visual.busy:raise ValueError('物体参照操作正在进行，请完成或停止后再切换控制')
        if self.runtime.busy:raise ValueError('请先停止并关闭遥操，再执行动作串或抓取任务')
        if self.editor.remote_locked():raise ValueError('请先完成手柄示教和录制，再执行任务')
        if self.lease:return
        # Both participants must use the same controller lock on this host.
        lock_name=self.cfg.get('vr_arm_lock_name',f'vr{self.robot_id}-arm-control.lock')
        path=Path(f'/run/user/{os.getuid()}')/lock_name
        handle=path.open('a+')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close();raise ValueError('V5/VR导航正在控制关节，请先停用并关闭对应程序')
        self.lease=handle
    def _unlock(self):
        if self.lease:self.lease.close();self.lease=None
    def start_vr(self):
        if self.visual.busy or self.engine.moving():raise ValueError('动作或纠偏正在进行，请先停止并等待停稳')
        if not self.editor.begin_remote():return
        self.engine.release();self._unlock();self.runtime.start()
    def execute(self,token,record,**kwargs):
        self._acquire()
        try:
            if record.get('visual_bindings'):
                if self.engine.moving():raise ValueError('请等待当前动作结束')
                self.ensure()
                import uuid
                prepare_token=uuid.uuid4().hex
                self.parallel_action=dict(token=token,prepare_token=prepare_token,kwargs=kwargs,body_ready=False,started=False,result=None)
                def ready(data):
                    try:
                        pending=self.parallel_action
                        if pending is None or pending['prepare_token']!=prepare_token:return
                        if data.get('error'):raise ValueError(data['error'])
                        pending['result']=data['result'];self._launch_parallel_action()
                    except Exception as exc:
                        self.parallel_action=None;self.engine.stop();self._completed(token,False,'物体纠偏未执行：'+str(exc))
                self.visual.request('visual_correct',ready,record=record,prepare_token=prepare_token)
            else:self.engine.execute(token,record,**kwargs)
        except Exception:
            self.parallel_action=None
            if self.engine.bridge is None:self._unlock()
            raise
    def adjust_posture(self,token,setting,*,keep_session=False):
        self._acquire();self.engine.adjust_posture(token,setting,keep_session=keep_session)
    def prepare_posture(self,token):
        self._acquire();self.engine.prepare_posture(token)
    def _launch_parallel_action(self):
        pending=self.parallel_action
        if not pending or not pending['body_ready'] or pending['result'] is None:return
        self.parallel_action=None;result=pending['result']
        self.engine.execute(pending['token'],result['record'],visual_source=result['source'],
            visual_base_anchor=result['base_anchor'],**pending['kwargs'])

    def _poll_parallel_action(self):
        pending=self.parallel_action
        if not pending or pending['started'] or (self.owner.run/'task_pause').exists():return
        path=self.owner.run/'action_prepare_source.json'
        if not path.exists():return
        data=json.loads(path.read_text('utf8'))
        if data.get('token')!=pending['prepare_token']:return
        pending['started']=True
        # 过期的并行通知只触发正常重新读取，不用旧姿态使能。
        source=data['source'] if 0<=time.monotonic()-data['stamp']<.5 else None
        self.engine.prepare_posture('parallel:'+pending['prepare_token'],refreshed=source)

    def _completed(self,token,ok,text):
        pending=self.parallel_action
        if pending and not ok and token=='manual-pose-idle':
            self.parallel_action=None;self.visual.cancel();self.engine.stop();self._completed(pending['token'],False,text);return
        if pending and token=='parallel:'+pending['prepare_token']:
            if ok:
                pending['body_ready']=True
                try:self._launch_parallel_action()
                except Exception as exc:
                    self.parallel_action=None;self.engine.stop();self._completed(pending['token'],False,str(exc))
            else:
                self.parallel_action=None;self.visual.cancel();self.engine.stop();self._completed(pending['token'],False,text)
            return
        if self.engine.bridge is None:self._unlock()
        self.editor.message(text,not ok);self.completed.emit(token,ok,text)
    def release(self):
        if self.visual.busy:raise ValueError('正在计算物体纠偏，请先停止当前操作')
        if self.runtime.busy:raise ValueError('请先点击“停止并关闭”，结束遥操后再切换到抓取与任务流程')
        self.engine.release();self._unlock()
    def stop(self):
        self.parallel_action=None
        self.visual.cancel()
        self.engine.stop()
        if self.runtime.owned or self.runtime.start_pending:self.runtime.stop()
    def tick(self):
        try:self._poll_parallel_action()
        except Exception as exc:
            pending=self.parallel_action;self.stop()
            if pending:self._completed(pending['token'],False,str(exc))
        if self.monitor:
            import rclpy
            for _ in range(3):rclpy.spin_once(self.monitor,timeout_sec=0.)
        self.editor.tick()
    def shutdown(self):
        self.parallel_action=None
        self.visual.cancel();self.recorder.close()
        self.editor.persist_draft();self.timer.stop();self.runtime.close();self.engine.shutdown();self._unlock()
        if self.monitor:self.monitor.destroy_node();self.monitor=None
