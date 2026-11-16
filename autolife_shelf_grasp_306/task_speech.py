"""Qt异步语音步骤：不阻塞界面；完成才推进，停止取消本次音频。"""
import json,os,signal
from pathlib import Path
from PyQt5 import QtCore as C

class TaskSpeech(C.QObject):
    completed=C.pyqtSignal(str,bool,str)
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.proc=None;self.token=None
    def start(self,token,text):
        self.stop();self.token=token
        from .task_tts_worker import cache_key
        cache_key(text)
        p=C.QProcess(self);self.proc=p
        p.finished.connect(lambda code,status:self.finish(p,token,code))
        p.errorOccurred.connect(lambda error:self.finish(p,token,-1) if error==C.QProcess.FailedToStart else None)
        p.setProgram('/usr/bin/setsid');p.setArguments(['/usr/bin/python3',str(Path(__file__).with_name('task_tts_worker.py')),'--root',str(self.owner.root)])
        def started():
            if p is not self.proc:
                self.terminate(p);return
            p.write(json.dumps(dict(text=text),ensure_ascii=False).encode());p.closeWriteChannel()
        p.started.connect(started)
        p.start()
    def finish(self,p,token,code):
        if p is not self.proc:return
        self.proc=None;self.token=None
        error=bytes(p.readAllStandardError()).decode(errors='replace')[-600:]
        p.deleteLater();self.completed.emit(token,code==0,'播报完成' if code==0 else '语音播报失败：'+error)
    def stop(self):
        p=self.proc;self.proc=None;self.token=None
        if p is not None:
            self.terminate(p)
            p.finished.connect(p.deleteLater)
    @staticmethod
    def terminate(p):
        pid=int(p.processId())
        if pid>0:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:p.kill()  # setsid尚未建组时，只结束自己的子进程
