"""仅任务执行入口创建准备进程；确认规划后接管为原 motion 进程。"""
import json
import uuid
from PyQt5 import QtCore as C
from .core import atomic_json


class ParallelPrepare(C.QObject):
    def __init__(self,owner,kind):
        super().__init__(owner);self.owner=owner;self.kind=kind;self.token=uuid.uuid4().hex;self.handed=False
        self.proc=C.QProcess(self);self.proc.setProcessChannelMode(C.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.output);self.proc.finished.connect(self.finished)
        self.proc.errorOccurred.connect(self.error)
        for name in ('parallel_prepare_state.json','parallel_prepare_execute.json'):(owner.run/name).unlink(missing_ok=True)
        self.proc.start('/usr/bin/python3',['-m','autolife_shelf_grasp_306.parallel_prepare',
            '--root',str(owner.root),'--run',str(owner.run),'--token',self.token,'--kind',kind])

    def output(self):
        raw=bytes(self.proc.readAllStandardOutput())
        if raw:
            with (self.owner.run/'execution.log').open('ab') as stream:stream.write(raw)
            if self.handed:self.owner.status.setText(raw.decode(errors='replace').strip()[-250:])

    def error(self,error):
        if error==C.QProcess.FailedToStart:self.finished(-1,C.QProcess.CrashExit)

    def finished(self,*args):
        self.output();o=self.owner
        if getattr(o,'parallel_prepare',None) is self:o.parallel_prepare=None
        if self.handed:o.motion_finished(*args)
        elif o.workflow.active:
            path=o.run/'execution_result.json'
            try:reason=json.loads(path.read_text('utf8')).get('reason','并行准备退出')
            except (ValueError,OSError):reason='并行准备退出'
            o.workflow.fail(reason)
        o.controls();self.deleteLater()

    def submit(self,path,plan_id,product_id):
        if self.proc.state()!=C.QProcess.Running:raise ValueError('并行准备进程已经退出')
        o=self.owner;self.handed=True;o.motion=self.proc;o.preparing=False;o.parallel_prepare=None
        atomic_json(o.run/'parallel_prepare_execute.json',dict(token=self.token,filename=path.name,
            plan_id=plan_id,product_id=product_id))
        o.execute.setEnabled(False);o.controls()


def start(owner,kind):
    if getattr(owner,'parallel_prepare',None) is not None:raise ValueError('上一并行准备尚未退出')
    # 已有任务授权；独立识别/规划预览绝不调用此函数。
    owner.parallel_prepare=ParallelPrepare(owner,kind)
