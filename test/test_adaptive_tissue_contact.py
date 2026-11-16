import copy
import numpy as np
import pytest
from test_complete_gripper import fixture
from autolife_shelf_grasp_306 import low_object_grasp as low
from autolife_shelf_grasp_306.gripper_contact_geometry import gripper_geometry,contact_tcp
from autolife_shelf_grasp_306.planning import IKFailure

@pytest.mark.parametrize('side',['left','right'])
def test_adaptive_contact_preserves_full_model_and_reference(tmp_path,monkeypatch,side):
    p,_=fixture(tmp_path,monkeypatch,side)
    p.cfg.update(up_torso=[0,0,1],low_object_table_clearance_m=.010,low_object_pitch_candidates_deg=[35,45,55,65,75])
    p.pose=lambda q,t:(np.zeros(3),np.eye(3))
    center=np.array([.4,0,.012]);product=low.support(dict(height_m=.024),center,[0,0,1])
    original=copy.deepcopy(product);cfg=copy.deepcopy(p.cfg);vertices=gripper_geometry(p)['vertices'].copy()
    with pytest.raises(low.TableContactFailure):low.orientation(p,{},contact_tcp(p,[.02,0,0]),center,product,0,True)
    physical,result,fraction=low.contact_orientation(p,{},[.02,0,0],[0,0,0],center,product,0)
    assert .8<fraction<=.9
    R,_,pitch,gap,shift=result
    assert 0<pitch<80 and gap>=.013-1e-9
    assert shift<=product['height_m']/2-.002
    assert np.min(((vertices-physical)@R.T+center+np.array([0,0,shift]))[:,2])>=.013-1e-9
    assert p.cfg==cfg and product==original
    assert np.array_equal(gripper_geometry(p)['vertices'],vertices)
    assert gripper_geometry(p)['finger_tip']-physical[0]>=.007-1e-9

def test_existing_contact_kept_when_it_fits(tmp_path,monkeypatch):
    p,_=fixture(tmp_path,monkeypatch);p.cfg.update(up_torso=[0,0,1],low_object_table_clearance_m=.010)
    p.pose=lambda q,t:(np.zeros(3),np.eye(3));center=np.array([.4,0,.04]);prod=low.support(dict(height_m=.08),center,[0,0,1])
    _,_,fraction=low.contact_orientation(p,{},[.02,0,0],[0,0,0],center,prod,0)
    assert fraction==.8

def test_no_contact_beyond_tip_for_unfeasible_thin_object(tmp_path,monkeypatch):
    p,_=fixture(tmp_path,monkeypatch);p.cfg.update(up_torso=[0,0,1],low_object_table_clearance_m=.010)
    p.pose=lambda q,t:(np.zeros(3),np.eye(3));center=np.array([.4,0,.002]);prod=low.support(dict(height_m=.004),center,[0,0,1])
    with pytest.raises(IKFailure,match='桌面'):low.contact_orientation(p,{},[.02,0,0],[0,0,0],center,prod,0)

@pytest.mark.parametrize('bad',[float('nan'),1.,.2])
def test_invalid_config_not_hidden_by_fallback(tmp_path,monkeypatch,bad):
    p,_=fixture(tmp_path,monkeypatch);p.cfg['low_object_finger_contact_fraction']=bad
    with pytest.raises(ValueError,match='指腹'):low.contact_orientation(p,{},[0,0,0],[0,0,0],[0,0,0],{},0)


def test_fine_pitch_fallback_respects_original_first(tmp_path,monkeypatch):
    p,_=fixture(tmp_path,monkeypatch)
    p.cfg.update(up_torso=[0,0,1],low_object_table_clearance_m=.010,low_object_pitch_candidates_deg=[35])
    p.pose=lambda q,t:(np.zeros(3),np.eye(3))
    center=np.array([.4,0,.01145]);prod=low.support(dict(height_m=.0229),center,[0,0,1])
    _,result,fraction=low.contact_orientation(p,{},[.02,0,0],[0,0,0],center,prod,0)
    assert fraction==.9 and result[2]==30
    assert p.cfg['low_object_pitch_candidates_deg']==[35]
