"""仅任务执行入口创建准备进程；确认规划后接管为原 motion 进程。"""
import sys
import json
import uuid
from PyQt5 import QtCore as C
from .core import atomic_json


class ParallelPrepare(C.QObject):
    def __init__(self,owner,kind):
        super().__init__(owner);self.owner=owner;self.kind=kind;self.token=uuid.uuid4().hex;self.handed=False;self._finished=False
        self.proc=C.QProcess(self);self.proc.setProcessChannelMode(C.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.output);self.proc.finished.connect(self.finished)
        self.proc.errorOccurred.connect(self.error)
        for name in ('parallel_prepare_state.json','parallel_prepare_execute.json','execution_result.json','execution_progress.json'):
            (owner.run/name).unlink(missing_ok=True)
    def begin(self):
        self.proc.start(sys.executable,['-m','autolife_shelf_grasp_306.parallel_prepare',
            '--root',str(self.owner.root),'--run',str(self.owner.run),'--token',self.token,'--kind',self.kind])

    def output(self):
        raw=bytes(self.proc.readAllStandardOutput())
        if raw:
            with (self.owner.run/'execution.log').open('ab') as stream:stream.write(raw)
            if self.handed and not self._finished and getattr(self.owner,'motion',None) is self.proc:
                self.owner.status.setText(raw.decode(errors='replace').strip()[-250:])

    def error(self,error):
        if error==C.QProcess.FailedToStart:self.finished(-1,C.QProcess.CrashExit)

    def finished(self,*args):
        if self._finished:return
        self._finished=True;o=self.owner
        owned=(getattr(o,'motion',None) is self.proc if self.handed else getattr(o,'parallel_prepare',None) is self)
        try:
            if not owned:return  # A late callback cannot fail a newer attempt.
            try:self.output()
            except (OSError,RuntimeError) as exc:print('准备日志读取失败：'+str(exc),flush=True)
            if getattr(o,'parallel_prepare',None) is self:o.parallel_prepare=None
            if self.handed:o.motion_finished(*args)
            elif o.workflow.active:
                path=o.run/'execution_result.json'
                try:
                    record=json.loads(path.read_text('utf8'))
                    reason=record.get('reason','并行准备退出') if record.get('execution_id')==self.token else '并行准备回执缺失或不属于本次准备'
                except (ValueError,OSError):reason='并行准备退出'
                o.workflow.fail(reason)
            o.controls()
        finally:self.deleteLater()

    def submit(self,path,plan_id,product_id,*,execution_id=None):
        if self._finished or self.handed:raise ValueError('并行准备已经结束或提交')
        if self.proc.state()!=C.QProcess.Running:raise ValueError('并行准备进程已经退出')
        o=self.owner
        (o.run/'execution_result.json').unlink(missing_ok=True)
        atomic_json(o.run/'parallel_prepare_execute.json',dict(token=self.token,filename=path.name,
            plan_id=plan_id,product_id=product_id,execution_id=execution_id))
        self.handed=True;o.motion=self.proc;o.preparing=False;o.parallel_prepare=None
        o.execute.setEnabled(False);o.controls()


def start(owner,kind):
    if getattr(owner,'parallel_prepare',None) is not None:raise ValueError('上一并行准备尚未退出')
    # 已有任务授权；独立识别/规划预览绝不调用此函数。
    pending=ParallelPrepare(owner,kind);owner.parallel_prepare=pending
    try:pending.begin()
    except Exception:
        if owner.parallel_prepare is pending:owner.parallel_prepare=None
        pending.deleteLater()
        raise
