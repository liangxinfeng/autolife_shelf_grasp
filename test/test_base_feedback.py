"""底盘独立采样：主队列延迟不误停，真正旧帧/移动仍可检测。"""
import sys
from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306 import base_feedback as b, demo_protocol as d


def message(stamp=10., x=0., speed=0.):
    return NS(header=NS(stamp=NS(sec=int(stamp),nanosec=round((stamp-int(stamp))*1e9))),
        pose=NS(pose=NS(position=NS(x=x,y=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))),
        twist=NS(twist=NS(linear=NS(x=speed,y=0.),angular=NS(z=0.))))


@pytest.fixture
def guard(monkeypatch):
    now=[10.];queue=[];events=[]
    monkeypatch.setattr(b.time,'monotonic',lambda:now[0])
    class Reader(b.BaseOdometryReader):
        def __init__(self,*a):
            self.sample=None;self.stamp=None;self.source_age=None
            self.node=NS(get_clock=lambda:NS(now=lambda:NS(nanoseconds=round(now[0]*1e9))))
            self.executor=NS(spin_once=self.spin)
        def spin(self,**kw):
            if queue:self.receive(queue.pop())
        def close(self):events.append('reader_closed')
    class Lease:
        held=True
        def __init__(self,**kw):pass
        def acquire(self,**kw):return True
        def release(self):events.append('release')
        def shutdown(self):events.append('shutdown')
    monkeypatch.setattr(b,'BaseOdometryReader',Reader)
    monkeypatch.setitem(sys.modules,'control_center_chassis_guard',NS(ChassisControlLease=Lease))
    g=d.StationaryBaseGuard(NS());queue.append(message());g.check()
    yield g,now,queue,events
    g.close()


@pytest.mark.parametrize('delay',[.31,.46,1.2])
def test_delayed_owner_loop_pulls_fresh_frame_without_wait(guard,delay):
    g,now,queue,_=guard;origin=g.origin
    now[0]+=delay;queue.append(message(now[0]))
    g.check()
    assert g.last[0]==now[0] and g.origin==origin


def test_no_message_still_stops(guard):
    g,now,_,_=guard;now[0]+=.5
    with pytest.raises(ValueError,match='反馈中断'):g.check()


def test_repeated_old_stamp_is_not_fresh(guard):
    g,now,queue,_=guard;now[0]+=.5;queue.append(message(10.))
    with pytest.raises(ValueError,match='反馈中断'):g.check()


def test_queued_historical_message_is_not_fresh(guard):
    g,now,queue,_=guard;now[0]+=1.;queue.append(message(10.1))
    with pytest.raises(ValueError,match='数据源未更新'):g.check()


@pytest.mark.parametrize('field,value,error',[
    ('speed',.1,'未停稳'),('x',.02,'位置改变'),('x',float('nan'),'反馈无效')])
def test_real_movement_and_invalid_data_not_silenced(guard,field,value,error):
    g,now,queue,_=guard;now[0]+=.02;queue.append(message(now[0],**{field:value}))
    with pytest.raises(ValueError,match=error):g.check()


def test_lease_loss_still_stops(guard):
    g,_,_,_=guard;g.lease.held=False
    with pytest.raises(ValueError,match='控制权丢失'):g.check()


def test_reader_failure_releases_lease(monkeypatch):
    events=[]
    monkeypatch.setitem(sys.modules,'control_center_chassis_guard',NS(ChassisControlLease=lambda **kw:
        NS(acquire=lambda **kw:True,release=lambda:events.append('release'),shutdown=lambda:events.append('shutdown'))))
    def fail(*a):raise RuntimeError('reader failed')
    monkeypatch.setattr(b,'BaseOdometryReader',fail)
    with pytest.raises(RuntimeError):d.StationaryBaseGuard(NS())
    assert events==['release','shutdown']


def test_reader_is_readonly_separate_executor_latest_only(monkeypatch):
    import rclpy.node,rclpy.executors
    events=[];owner=NS(context=object())
    class Node:
        def __init__(self,name,**kw):
            assert kw['context'] is owner.context
            assert not kw['use_global_arguments']
        def create_subscription(self,cls,topic,callback,qos):
            assert qos.depth==1;events.append('subscription')
        def destroy_node(self):events.append('destroy')
    class Executor:
        def __init__(self,**kw):assert kw['context'] is owner.context
        def add_node(self,node):events.append('add')
        def spin_once(self,**kw):assert kw['timeout_sec']==0.;events.append('poll')
        def shutdown(self):events.append('shutdown')
    monkeypatch.setattr(rclpy.node,'Node',Node)
    monkeypatch.setattr(rclpy.executors,'SingleThreadedExecutor',Executor)
    reader=b.BaseOdometryReader(owner,'/odom');assert reader.latest() is None;reader.close()
    assert events==['subscription','add','poll','shutdown','destroy']
