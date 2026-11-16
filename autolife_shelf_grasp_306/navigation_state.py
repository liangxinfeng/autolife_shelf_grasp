"""导航连接、保护恢复和显示负载分离；本文件不发送运动指令。"""
import threading
from collections import OrderedDict


def guard_message(guard, age):
    if not guard or not 0 <= age < 2.:
        return '等待导航保护状态更新；后台已连接，无需重复启动'
    if guard.get('navigation_allowed') is True:
        return ''
    if guard.get('reason_code') == 'recovery_pending':
        return '导航已连接，正在确认心跳稳定、旧目标结束和底盘停稳；旧目标不会自动继续'
    return guard.get('detail') or '等待导航保护就绪'


def may_rearm(guard, age, session, idle, stationary):
    """仅为本窗口的闲置会话恢复接单权限，不能恢复旧目标或他人会话。"""
    return bool(0 <= age < 2. and guard.get('reason_code') == 'recovery_pending'
                and guard.get('owner_session') == session
                and guard.get('heartbeat_fresh') is True
                and guard.get('session_conflict') is False
                and idle and stationary)


class LatestImages:
    """重图像计算放到独立线程；每类只保留最新一帧，避免积压拖慢ROS回调。"""
    def __init__(self):
        self.pending = OrderedDict()
        self.cv = threading.Condition()
        self.closed = False
        self.error = ''
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, key, fn, message):
        with self.cv:
            if self.closed:return
            self.pending[key] = (fn, message)
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.closed or self.pending)
                if self.closed:return
                _, (fn, message) = self.pending.popitem(last=False)
            try:fn(message)
            except Exception as exc:self.error = str(exc)

    def close(self):
        with self.cv:
            self.closed = True
            self.pending.clear()
            self.cv.notify()
        self.thread.join(timeout=1.)
