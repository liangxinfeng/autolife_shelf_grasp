"""Portable configuration resolution; no ROS initialization or hardware imports."""
from pathlib import Path
import sys
from types import ModuleType

import yaml
from autolife_shelf_grasp_306.runtime_config import load_config


def write_config(root,data):
    (root/'config').mkdir(parents=True)
    (root/'config'/'grasp.yaml').write_text(yaml.safe_dump(data))


def test_relative_resources_follow_configuration_root_not_cwd(tmp_path,monkeypatch):
    root=tmp_path/'deployment'
    write_config(root,dict(robot_id=367,urdf_path='urdf/model.urdf',
        gripper_width_calibration='config/jaw.json',calibration='calibration/left/session.json',
        right_calibration='calibration/right/session.json',grounding_dino_model='models/detector'))
    monkeypatch.chdir(tmp_path);monkeypatch.delenv('ROBOT_ID',raising=False)
    cfg=load_config(root)
    for key,relative in [('urdf_path','urdf/model.urdf'),('gripper_width_calibration','config/jaw.json'),
                         ('calibration','calibration/left/session.json'),('right_calibration','calibration/right/session.json')]:
        assert cfg[key]==str(root/relative)
    assert cfg['grounding_dino_model']=='models/detector'
    assert cfg['python']==sys.executable and cfg['robot_id']==367
    assert 'vendor_python_path' not in cfg


def test_environment_identity_and_explicit_absolute_paths(tmp_path,monkeypatch):
    model=tmp_path/'external'/'model.urdf'
    write_config(tmp_path,dict(robot_id=367,urdf_path=str(model),python='runtime/bin/python',
        vendor_python='vendor/bin/python',vendor_python_path='vendor/site-packages'))
    monkeypatch.setenv('ROBOT_ID','812')
    cfg=load_config(tmp_path)
    assert cfg['robot_id']==812 and cfg['urdf_path']==str(model)
    assert cfg['python']==str(tmp_path/'runtime/bin/python')
    assert cfg['vendor_python']==str(tmp_path/'vendor/bin/python')
    assert cfg['vendor_python_path']==str(tmp_path/'vendor/site-packages')


def test_data_only_root_uses_installed_configuration_assets(tmp_path,monkeypatch):
    share=tmp_path/'share';write_config(share,dict(robot_id=123,urdf_path='urdf/model.urdf'))
    package=ModuleType('ament_index_python.packages')
    package.get_package_share_directory=lambda name:str(share)
    monkeypatch.setitem(sys.modules,'ament_index_python',ModuleType('ament_index_python'))
    monkeypatch.setitem(sys.modules,'ament_index_python.packages',package)
    cfg=load_config(tmp_path/'data_only')
    assert cfg['urdf_path']==str(share/'urdf/model.urdf')
