"""Installed/share resource resolution, without ROS, planners or hardware."""
import hashlib
import json

from autolife_shelf_grasp_306.planner_cache import cache_key
from autolife_shelf_grasp_306.resources import collision_model_path
from autolife_shelf_grasp_306 import supported_neck_recovery as recovery


def resources(tmp_path):
    share=tmp_path/'install'/'share'/'grasp'/'urdf';share.mkdir(parents=True)
    model=share/'robot_v3_1.urdf';model.write_text('kinematic model')
    collision=share/'robot_v3_1_simplified.urdf';collision.write_text('collision model')
    srdf=share/'robot_v3_1.srdf';srdf.write_text('collision pairs')
    vendor=tmp_path/'vendor.json';vendor.write_text('{}')
    return dict(urdf_path=str(model),vendor_control_config=str(vendor)),collision,srdf


def test_cache_uses_installed_share_not_python_module_location(tmp_path):
    cfg,collision,srdf=resources(tmp_path)
    assert collision_model_path(cfg)==collision
    before=cache_key(cfg)
    stamped={entry[0] for entry in json.loads(before)[1]}
    assert str(collision) in stamped and str(srdf) in stamped
    srdf.write_text('changed collision pairs')
    assert cache_key(cfg)!=before
    assert cache_key(dict(cfg,up_torso=[0,0,1]))==cache_key(cfg)


def test_explicit_collision_model_and_recovery_use_the_same_resource(tmp_path,monkeypatch):
    cfg,_,_=resources(tmp_path)
    custom=tmp_path/'custom';custom.mkdir()
    collision=custom/'collision.urdf';collision.write_text('approved model')
    srdf=custom/'robot_v3_1.srdf';srdf.write_text('approved pairs')
    cfg['collision_urdf_path']=str(collision)
    monkeypatch.setattr(recovery,'MODEL_HASH',hashlib.sha256(collision.read_bytes()).hexdigest())
    monkeypatch.setattr(recovery,'SRDF_HASH',hashlib.sha256(srdf.read_bytes()).hexdigest())
    assert collision_model_path(cfg)==collision
    stamped={entry[0] for entry in json.loads(cache_key(cfg))[1]}
    assert str(collision) in stamped and str(srdf) in stamped
    recovery.verify_installed_model(cfg)
    collision.write_text('changed model')
    import pytest
    with pytest.raises(ValueError,match='模型已改变'):recovery.verify_installed_model(cfg)
