"""保持式接管前的异步只读刷新；不使能、不发布关节目标、不跨线程spin。"""
from .runtime_identity import identity
import time
from .feedback_refresh import GROUPS,GroupRead


class PoseFeedbackRead:
    def __init__(self,node):
        from std_srvs.srv import Trigger
        self.node=node;self.request=Trigger.Request
        self.clients={g:node.create_client(Trigger,f'/refresh_grasp_feedback_{identity().topic_suffix}_{g}') for g in GROUPS}
        self.batch=GroupRead(self.clients,self.request)
        self.started=time.monotonic();self.values=None;self.received=None
    def poll(self):
        now=time.monotonic()
        if now-self.started>6:
            if self.values is None:raise self.batch.unavailable()
            raise ValueError('接管前关节反馈未跟上实测位置')
        if self.values is None:
            values=self.batch.poll(now)
            if values is None:return None
            from control_center_joint_control.joint_definitions import ALL_JOINTS
            if set(values)!=set(ALL_JOINTS):raise ValueError('真实位置刷新缺少完整23轴')
            self.values=values;self.received=now
            return None
        n=self.node
        if not n._is_feedback_fresh() or n._feedback_monotonic<=self.received:return None
        if any(abs(n._current.get(k,float('inf'))-v)>(2. if 'Gripper' in k else .5) for k,v in self.values.items()):return None
        return dict(n._current)
    def close(self):
        for c in self.clients.values():self.node.destroy_client(c)
        self.clients={}
