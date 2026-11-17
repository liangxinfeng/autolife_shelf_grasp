"""Regression: waist allowance must be shared by classification and proposed displacement."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import mobile_reach as m
from autolife_shelf_grasp_306 import low_object_grasp as low

@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(np.array([0.,0.,.6]),.7))
    monkeypatch.setattr(m,'waist_allowance',lambda *a:.12)
    monkeypatch.setattr(m,'collision_boxes',lambda *a:[])
    monkeypatch.setattr(m,'sweep_clear',lambda *a:True)
    cfg=dict(arm='right',auto_arm_selection=False,up_torso=[0,0,1])
    fk=NS(fk=lambda *a:np.eye(4))
    pool=NS(fk=fk,arm_config=lambda *a:(cfg,[0,0,0]),native_planner=lambda *a:NS())
    return pool,cfg

@pytest.mark.parametrize('point',[[.9,0,.6],[0,.9,.6],[-.9,0,.6],[.7,-.6,.6]])
def test_extended_workspace_generates_short_omni_proposal(setup,point):
    pool,cfg=setup
    result=m.grasp_reposition(pool,{},point,{},cfg,[0,0,0],None,lambda:False,lambda s:None)
    assert result['workspace_radius_m']==pytest.approx(.82)
    assert 0<np.linalg.norm(result['translation_m'])<=.15
    assert np.linalg.norm(np.asarray(point)-np.r_[result['translation_m'],0]-[0,0,.6])<=.82
    assert not result['path_checked']

@pytest.mark.parametrize('point',[[1.3,0,.6],[0,0,1.6]])
def test_uncompensatable_does_not_silently_enter_original_plan(setup,point):
    pool,cfg=setup
    with pytest.raises(ValueError,match='单次15厘米'):
        m.grasp_reposition(pool,{},point,{},cfg,[0,0,0],None,lambda:False,lambda s:None)

def test_swept_obstacle_does_not_become_stationary_fallback(setup,monkeypatch):
    pool,cfg=setup;monkeypatch.setattr(m,'sweep_clear',lambda *a:False)
    with pytest.raises(ValueError,match='通道存在障碍'):
        m.grasp_reposition(pool,{},[.9,0,.6],{},cfg,[0,0,0],None,lambda:False,lambda s:None)

def test_within_waist_workspace_still_uses_original_planning(setup):
    pool,cfg=setup;pool.native_planner=lambda *a:pytest.fail('must not preload base collision model')
    assert m.grasp_reposition(pool,{},[.79,0,.6],{},cfg,[0,0,0],None,lambda:False,lambda s:None) is None

def test_place_uses_the_same_range(setup,monkeypatch):
    pool,cfg=setup
    monkeypatch.setattr(m,'sweep_clear',lambda *a:True)
    p=NS(fk=pool.fk,cfg=cfg)
    result=m.place_reposition(p,{},[0,0,0],dict(product=dict(height_m=.1,diameter_m=.05)),np.array([[.9,0,.54]]),{},dict(place_clearance_m=.01),[0,0,0],None,lambda:False,lambda s:None)
    assert 0<np.linalg.norm(result['translation_m'])<=.15


def test_reachable_attitude_failure_is_not_overwritten_by_next_contact(monkeypatch):
    from autolife_shelf_grasp_306 import gripper_contact_geometry as g
    from autolife_shelf_grasp_306.planning import IKFailure
    monkeypatch.setattr(g,'contact_tcp',lambda p,t,f:[f,0,0])
    def orientation(p,q,t,*a,**kw):
        if t[0]==.85:raise low.ContactReachFailure('离桌朝向逆解不可达')
        raise low.TableContactFailure('指尖触桌')
    monkeypatch.setattr(low,'orientation',orientation)
    with pytest.raises(IKFailure,match='逆解不可达'):
        low.contact_orientation(NS(cfg={}),{},[0,0,0],[0,0,0],[0,0,0],{},0)
