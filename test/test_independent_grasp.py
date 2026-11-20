"""Offline regression for selection, fusion and payload ownership."""
import json
import numpy as np
import pytest
from autolife_shelf_grasp_306.bottle import combine
from autolife_shelf_grasp_306.unregistered import temporary_product
from autolife_shelf_grasp_306.held_receipt import begin_grasp_opening,release_payload
from autolife_shelf_grasp_306.core import atomic_json
from autolife_shelf_grasp_306.demo_config import require_placement_bottom
from autolife_shelf_grasp_306.grasp_collision import CollisionGuard,GROUPS
from autolife_shelf_grasp_306.vendor_grasp_planning import WaistAwareCollisionGuard

@pytest.mark.parametrize('label',['white bag','cloth','towel','rope','crumpled tissue'])
def test_material_goes_to_geometry(label):
    item=dict(id=3,name=label,semantic_label=label)
    assert temporary_product(item,'run')['geometry']=='solid'

def test_sparse_frames_with_body_contact_can_plan_but_preserve_bottom_uncertainty():
    stats=dict(observed_height_mm=100,collision_height_m=.22,placement_bottom_observed=False)
    c,geometry=combine([(np.array([.5,0,.1]),stats)])
    assert geometry['frames']==1 and geometry['collision_height_m']>=.22
    with pytest.raises(ValueError,match='瓶底'):require_placement_bottom(dict(product={},geometry=geometry))
    with pytest.raises(ValueError):combine([])

def test_latest_grasp_replaces_only_after_handoff(tmp_path):
    path=tmp_path/'held_object.json'
    atomic_json(path,dict(status='completed',arm='left',plan_id='first'))
    begin_grasp_opening(tmp_path,'right')
    assert json.loads(path.read_text())['plan_id']=='first'
    atomic_json(path,dict(status='completed',arm='right',plan_id='second'))
    calls=[]
    with pytest.raises(ValueError):release_payload(tmp_path,dict(plan_id='place',parent_plan_id='first'),lambda:calls.append(1))
    assert not calls and json.loads(path.read_text())['plan_id']=='second'
    release_payload(tmp_path,dict(plan_id='place',parent_plan_id='second'),lambda:calls.append(1))
    assert calls==[1] and not path.exists()
    atomic_json(path,dict(status='completed',arm='left',plan_id='third'))
    begin_grasp_opening(tmp_path,'left')
    assert not path.exists() and len(list((tmp_path/'held_history').glob('*.json')))==1

def test_second_grasp_uses_normal_executor(tmp_path,monkeypatch):
    from test_execution import test_execution_finishes_without_stop
    path=tmp_path/'held_object.json';atomic_json(path,dict(status='completed',arm='left',plan_id='first'))
    test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp')
    assert json.loads(path.read_text())['plan_id']=='plan-token'
    assert json.loads(next((tmp_path/'held_history').iterdir()).read_text())['plan_id']=='first'

@pytest.mark.parametrize('known,initial_pitch,observed',[(True,35.62,True),(False,35.62,True),(True,0.,True),(True,35.62,False)])
def test_static_boot_overlap_is_scoped_to_initial_pair(known,initial_pitch,observed):
    from types import SimpleNamespace as NS
    from autolife_handeye_calibration_306.core import NECK
    head='Link_Waist_Yaw_to_Shoulder_Inner_0 <-> Link_Neck_Yaw_to_Head_0'
    arm='Link_Left_Wrist <-> Link_Neck_Yaw_to_Head_0'
    guard=WaistAwareCollisionGuard.__new__(WaistAwareCollisionGuard)
    guard.cache=set();guard.boot_head_model=known
    guard.relative_fk=NS(fk=lambda *a:np.eye(4))
    guard.kin=NS(q_from_feedback=lambda *a,**k:np.zeros(1),lower=np.array([-2]),upper=np.array([2]),
                 collision_pairs=lambda q:[head] if observed else [])
    q={n:0. for names in GROUPS.values() for n in names};q['Joint_Neck_Pitch']=initial_pitch
    guard.begin_fixed_neck(q);moved={**q,GROUPS['leg_waist'][-1]:12.}
    assert guard.blocking_pairs([head,arm],moved)==([arm] if known and initial_pitch>15 and observed else [head,arm])
    assert head in guard.blocking_pairs([head],{**moved,NECK[-1]:.01})
