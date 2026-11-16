"""原地腰臂可能可达就不抢先移动底盘；上界采用解析弦长，不采样判不可达。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306 import mobile_reach as m
from autolife_handeye_calibration_306.core import BODY,TORSO


class WaistFK:
    def __init__(self):
        offset=np.eye(4);offset[2,3]=.07
        self.joints={'pitch':(BODY[2],'revolute','Link_Zero_Point',np.eye(4)),
            'yaw':(BODY[3],'revolute','pitch',offset)}
    def fk(self,root,tip,q):
        if tip=='Link_Zero_Point':return np.eye(4)
        p=np.eye(4);p[:3,:3]=Rotation.from_euler('y',q[BODY[2]],degrees=True).as_matrix()
        if tip=='pitch':return p
        p=p@self.joints['yaw'][3]
        yaw=np.eye(4);yaw[:3,:3]=Rotation.from_euler('z',q[BODY[3]],degrees=True).as_matrix()
        return p@yaw


@pytest.mark.parametrize('side',[-1,1])
def test_analytic_bound_contains_all_sampled_waist_positions(side):
    fk=WaistFK();q={BODY[2]:-43.,BODY[3]:2.};shoulder=np.array([0.,side*.15,.275])
    cfg={'vendor_grasp_waist_range_deg':[12.,8.]};allow=m.waist_allowance(fk,q,shoulder,cfg)
    initial=(fk.fk('Link_Zero_Point',TORSO,q)@np.r_[shoulder,1])[:3]
    assert 0<allow<.2
    for pitch in np.linspace(-12,12,13):
        for yaw in np.linspace(-8,8,9):
            trial={BODY[2]:q[BODY[2]]+pitch,BODY[3]:q[BODY[3]]+yaw}
            actual=(fk.fk('Link_Zero_Point',TORSO,trial)@np.r_[shoulder,1])[:3]
            assert np.linalg.norm(actual-initial)<=allow+1e-10


def test_possible_waist_reach_never_builds_planner_or_base_proposal(monkeypatch):
    fk=WaistFK();q={BODY[2]:-43.,BODY[3]:2.};shoulder=np.array([0,-.15,.275])
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(shoulder,.77))
    cfg=dict(arm='right',auto_arm_selection=False,vendor_grasp_waist_range_deg=[12.,8.])
    pool=NS(fk=fk,arm_config=lambda *a:(cfg,[0,0,0]),native_planner=lambda *a:pytest.fail('应交回原地规划，不加载补偿规划器'))
    result=m.grasp_reposition(pool,q,shoulder+[.81,0,0],{},cfg,[0,0,0],None,lambda:False,lambda s:None)
    assert result is None


def test_clearly_outside_waist_upper_bound_remains_base_candidate():
    fk=WaistFK();q={BODY[2]:-43.,BODY[3]:2.};shoulder=np.array([0,-.15,.275]);cfg={}
    radius=.77;allow=m.waist_allowance(fk,q,shoulder,cfg)
    assert not m.needs_base(shoulder+[radius+allow-.001,0,0],shoulder,radius,fk,q,cfg)
    assert m.needs_base(shoulder+[radius+allow+.01,0,0],shoulder,radius,fk,q,cfg)


def test_placement_also_prefers_stationary_waist_reach(monkeypatch):
    fk=WaistFK();q={BODY[2]:-43.,BODY[3]:2.};shoulder=np.array([0,-.15,.275]);cfg={'arm':'right'}
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(shoulder,.77))
    monkeypatch.setattr(m,'collision_boxes',lambda *a:pytest.fail('原地可能可达时不应构造底盘扫掠'))
    ground=fk.fk('Link_Zero_Point',TORSO,q);desired=shoulder+[.81,0,0]
    point=(ground@np.r_[desired,1])[:3]-[0,0,.06]
    result=m.place_reposition(NS(fk=fk,cfg=cfg),q,[0,0,0],dict(product=dict(height_m=.1,diameter_m=.05)),
        np.array([point]),{},dict(place_clearance_m=.01),[0,0,0],None,lambda:False,lambda s:None)
    assert result is None
