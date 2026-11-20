"""Explicit transferable reference curves and exact plan-to-curve binding."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from autolife_shelf_grasp_306.grip_width import load_curve,closing_command,command_from_plan


def reference():
    return json.loads((Path(__file__).parents[1]/'config'/'gripper_width.json').read_text())


def configuration(tmp_path,data,arm='left',robot_id=367):
    path=tmp_path/'gripper.json';path.write_text(json.dumps(data))
    return dict(arm=arm,robot_id=robot_id,gripper_width_calibration=str(path))


@pytest.mark.parametrize('arm',['left','right'])
def test_reference_curve_retains_source_without_claiming_target_measurement(tmp_path,arm):
    data=reference()
    assert data['measured'] is False and data['source_robot_id']==345
    cfg=configuration(tmp_path,data,arm)
    widths,positions,digest=load_curve(cfg)
    assert widths.tolist()==[5,45,90] and positions.tolist()==[330,180,10]
    assert digest==hashlib.sha256(Path(cfg['gripper_width_calibration']).read_bytes()).hexdigest()
    for width,position in [(45,180),(90,10),(25,255)]:
        command=closing_command(width/1000,0,cfg)
        assert command['target_deg']==pytest.approx(position)
        assert command['mapping']=='dm_type1_reference_v1'


def test_reference_migrates_without_robot_id_lock(tmp_path):
    data=reference();data['robot_id']=800;data['source_robot_id']=700
    cfg=configuration(tmp_path,data,robot_id=900)
    assert closing_command(.045,0,cfg)['target_deg']==180
    cfg.pop('robot_id')
    assert closing_command(.045,0,cfg)['target_deg']==180


@pytest.mark.parametrize('change',[
    {'reference_accepted':False}, {'reference_accepted':'true'},
    {'source_robot_id':None}, {'measured':True},
    {'calibration_mode':'measured'}, {'hardware':'s2_95mm'},
    {'arms':{'left':[[5,330],[45,180],[90,180]]}},
])
def test_reference_requires_acceptance_compatible_hardware_and_valid_curve(tmp_path,change):
    data=reference();data.update(change)
    with pytest.raises(ValueError):load_curve(configuration(tmp_path,data))


def test_measured_curve_uses_generic_mapping(tmp_path):
    data=reference();data.update(calibration_mode='measured',measured=True)
    cfg=configuration(tmp_path,data,robot_id=999)
    assert closing_command(.045,0,cfg)['mapping']=='dm_type1_measured_v1'


def test_reference_cannot_be_applied_to_incompatible_gripper(tmp_path):
    cfg=configuration(tmp_path,reference());cfg['gripper_hardware']='s2_95mm'
    with pytest.raises(ValueError,match='DM Type1'):load_curve(cfg)


@pytest.mark.parametrize('change',['points','provenance','mode','plan'])
def test_curve_or_plan_change_requires_replanning(tmp_path,change):
    data=reference();cfg=configuration(tmp_path,data)
    command=closing_command(.045,4,cfg)
    plan=dict(config=cfg,grasp_width_m=.045,grip_command=copy.deepcopy(command))
    assert command_from_plan(plan)==pytest.approx(195)
    if change=='points':data['arms']['left'][1][1]=179
    elif change=='provenance':data['provenance']='different reference source'
    elif change=='mode':data.update(calibration_mode='measured',measured=True)
    else:plan['grip_command']['target_deg']+=1
    Path(cfg['gripper_width_calibration']).write_text(json.dumps(data))
    with pytest.raises(ValueError):command_from_plan(plan)
