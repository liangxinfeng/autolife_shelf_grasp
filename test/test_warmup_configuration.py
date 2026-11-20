"""Warmup must not fabricate or retain an old measured torso direction."""
from autolife_shelf_grasp_306 import arm_selection as arms


def test_model_warmup_needs_no_measured_up_direction(monkeypatch):
    monkeypatch.setattr(arms,'tool_geometry',lambda cfg,cal,fk,side:({**cfg,'arm':side},None))
    pool=arms.ArmPlannerPool(dict(arm='left',up_torso=[1,0,0]),{},None,None)
    config,_=pool.arm_config('right',{})
    assert config['arm']=='right' and 'up_torso' not in config
    current=dict(up_torso=[0,0,1])
    config,_=pool.arm_config('right',current)
    assert config['up_torso']==[0,0,1]
    current['up_torso'][0]=1
    assert config['up_torso']==[0,0,1]
    assert pool.base['up_torso']==[1,0,0]
