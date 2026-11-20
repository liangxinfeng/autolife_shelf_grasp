"""规划/使能前主动读取各组关节；新消息时间不能代替真实硬件刷新。"""
from .runtime_identity import identity
import json
import math
import time

GROUPS = ('neck','leg_waist','left_arm','right_arm','left_gripper','right_gripper')

class FeedbackReadUnavailable(ValueError):
    """Transport or sensor unavailable during a read-only operation."""
    code='FEEDBACK_UNAVAILABLE'


class GroupRead:
    def __init__(self,clients,request):
        self.clients=clients;self.request=request
        self.pending={};self.samples={};self.retry_at={};self.errors={}

    def poll(self,now):
        for group,client in self.clients.items():
            if group in self.samples:continue
            future=self.pending.get(group)
            if future is None:
                if now>=self.retry_at.get(group,0.) and client.service_is_ready():
                    self.pending[group]=client.call_async(self.request())
                continue
            if not future.done():continue
            del self.pending[group]
            try:reply=future.result()
            except Exception as exc:reply=None;self.errors[group]=str(exc)
            if reply is None or not reply.success:
                if reply is not None:self.errors[group]=str(reply.message)
                self.retry_at[group]=now+.20
                continue
            data=json.loads(reply.message);values=data.get('positions')
            if data.get('protocol')!=2 or data.get('group')!=group or not isinstance(values,dict) or not values:
                raise ValueError('关节刷新响应无效：'+group)
            values={k:float(v) for k,v in values.items()}
            if not all(math.isfinite(v) for v in values.values()):raise ValueError('关节刷新数据非有限值：'+group)
            self.samples[group]=(now,values)
            self.errors.pop(group,None)
        if len(self.samples)!=len(self.clients):return None
        # Re-read early groups after a delayed recovery. Never join an old arm
        # position with a new one to create a fictitious whole-body snapshot.
        oldest=min(t for t,_ in self.samples.values())
        if now-oldest>1.:
            self.samples={g:v for g,v in self.samples.items() if now-v[0]<=1.}
            return None
        positions={}
        for _,values in self.samples.values():
            if positions.keys() & values.keys():raise ValueError('关节刷新响应重复')
            positions.update(values)
        return positions

    def unavailable(self):
        groups=[g+' / '+self.errors.get(g,'等待只读服务响应') for g in self.clients if g not in self.samples]
        return FeedbackReadUnavailable('真实关节位置读取失败：'+'；'.join(groups))


def wait_for_measured(read,cancelled,on_wait):
    while True:
        if cancelled():raise ValueError('已取消当前反馈读取')
        try:positions=read()
        except FeedbackReadUnavailable as exc:
            on_wait(str(exc));continue
        if cancelled():raise ValueError('已取消当前反馈读取')
        return positions

def await_snapshot(snapshot, spin, positions, timeout=1.):
    """确认ROS反馈已跟上主动读数，避免随后仍锁存刷新前的消息。"""
    deadline = time.monotonic()+timeout
    while True:
        current = snapshot()
        if all(name in current and math.isfinite(float(current[name])) and
               abs(float(current[name])-value) <= (2. if 'Gripper' in name else .5)
               for name,value in positions.items()):
            return
        if time.monotonic() >= deadline:
            raise FeedbackReadUnavailable('主动读取与关节反馈尚未一致，未发送使能目标')
        spin()


def refresh(node, spin, suffix=None, timeout=5.):
    suffix = suffix or identity().topic_suffix
    from std_srvs.srv import Trigger
    clients = {g: node.create_client(Trigger, f'/refresh_grasp_feedback_{suffix}_{g}') for g in GROUPS}
    deadline = time.monotonic()+timeout
    try:
        batch=GroupRead(clients,Trigger.Request)
        while time.monotonic()<deadline:
            positions=batch.poll(time.monotonic())
            if positions is not None:
                from .feedback_state import NAMES
                if set(positions)!=set(NAMES):raise ValueError('真实位置刷新缺少完整23轴或含未知关节')
                return positions
            spin()
        raise batch.unavailable()
    finally:
        for client in clients.values():
            node.destroy_client(client)
