"""专用视觉请求通道，复用唯一视觉进程，不修改抓取计划/商品选择状态。"""
import json,time,uuid
from PyQt5 import QtCore as C


class VisualClient(C.QObject):
    REQUEST_TIMEOUT_SEC=120.
    def __init__(self,studio):
        super().__init__(studio);self.studio=studio;self.owner=studio.owner;self.pending=None
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(250)
    @property
    def busy(self):return self.pending is not None
    def request(self,action,callback,**values):
        o=self.owner
        if self.busy or o.busy or o.motion is not None:raise ValueError('当前视觉或动作还在运行，请完成后再操作')
        if self.studio.runtime.busy or self.studio.editor.remote_locked():raise ValueError('请先停止VR示教，再进行物体参照纠偏')
        if o.navigation.inflight:raise ValueError('请等待导航结束后定位物体参照')
        if not o.ready or not o.process:raise ValueError('视觉后台尚未就绪，普通姿态动作仍可使用')
        ident=uuid.uuid4().hex
        payload=dict(action=action,visual_id=ident,**values)
        encoded=(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n').encode()
        self.studio._acquire()
        self.pending=dict(id=ident,callback=callback,started=time.monotonic())
        try:
            if o.process.write(encoded)<0:raise ValueError('视觉请求发送失败')
        except Exception:
            self.pending=None
            if self.studio.engine.bridge is None:self.studio._unlock()
            raise
        o.status.setText('物体参照：'+{'visual_commit_view':'正在保存已确认的物体轮廓','visual_freeze':'正在取景','visual_remember':'正在记住整物体','visual_add_view':'正在关联补录视角','visual_bind':'正在关联姿态','visual_correct':'重新定位并计算纠偏轨迹'}.get(action,action))
        o.controls()
    def _finish(self,ident,data,*,cancel_backend=False):
        p=self.pending
        if not p or ident!=p['id']:return
        # Detach first: callback re-entry and late/duplicate results cannot
        # complete this request a second time or dispatch its old motion.
        self.pending=None
        marker=self.owner.run/('visual_cancel_'+ident)
        try:
            if cancel_backend:marker.touch()
            else:marker.unlink(missing_ok=True)
        except OSError:
            # The pending ID is already revoked even if the worker cannot be
            # signalled. Its eventual reply remains ineligible for execution.
            pass
        self.owner.controls()
        try:p['callback'](data)
        finally:
            if not self.busy and self.studio.engine.bridge is None:self.studio._unlock()
    def receive(self,data):
        self._finish(data.get('visual_id'),data)
    def cancel(self):
        if self.pending:
            ident=self.pending['id']
            self._finish(ident,dict(visual_id=ident,error='操作已取消，未执行纠偏动作',
                reason_code='OPERATOR_CANCELLED'),cancel_backend=True)
    def fail(self,text):
        if self.pending:
            ident=self.pending['id']
            self._finish(ident,dict(visual_id=ident,error=text,
                reason_code='VISION_UNAVAILABLE'),cancel_backend=True)
    def tick(self):
        if self.pending and time.monotonic()-self.pending['started']>self.REQUEST_TIMEOUT_SEC:
            ident=self.pending['id']
            text='物体参照计算超时，本次请求已结束；可重新观察，旧结果不会执行动作'
            self.owner.status.setText(text)
            self._finish(ident,dict(visual_id=ident,error=text,
                reason_code='VISION_TIMEOUT'),cancel_backend=True)
