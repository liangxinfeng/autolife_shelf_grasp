"""放置最大张开目标不能冒充物品脱离宽度；实际回调和收回状态机验证。"""
import ast,inspect,json
from types import SimpleNamespace as NS
import pytest
from test_place_release import scenario
from test_vendor_place import fixture
from autolife_shelf_grasp_306.vendor_place_planning import VendorPlacePlanner


def attach_actual_metadata(plan,grasp):
    tree=ast.parse(inspect.getsource(__import__('autolife_shelf_grasp_306.vendor_place_planning',fromlist=['x'])))
    keys={'release_opening_m','grasp_width_m','release_minimum_opening_m'}
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='plan'
               and isinstance(t.slice,ast.Constant) and t.slice.value in keys for t in n.targets)]
    assert len(nodes)==3
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual placement width metadata>','exec'),
         dict(self=NS(cfg={'gripper_max_width_m':.09}),plan=plan,grasp=grasp))


def prepared(tmp_path,side='right',minimum=.08795048763775613):
    p=fixture(side);curve=tmp_path/'calibration.json'
    curve.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={a:[[5,330],[45,180],[90,10]] for a in ('left','right')})))
    p['config']['gripper_width_calibration']=str(curve)
    attach_actual_metadata(p,{'grasp_width_m':.08794836384784688,'minimum_opening_m':minimum})
    assert p['release_opening_m']==.09 and p['release_minimum_opening_m']==minimum
    return p


@pytest.mark.parametrize('side',['left','right'])
def test_actual_88mm_opening_completes_90mm_release_target(tmp_path,side):
    p=prepared(tmp_path,side)
    elapsed,actual=scenario(tmp_path,side,floor=16.71180248260498,plan_override=p)
    assert elapsed<5 and actual==pytest.approx(16.71180248260498)


def test_missing_legacy_minimum_uses_actual_grasp_width():
    p={};attach_actual_metadata(p,{'grasp_width_m':.055})
    assert p['release_minimum_opening_m']==.055 and p['release_opening_m']==.065


def test_real_blocked_jaw_is_not_reported_as_release(tmp_path):
    with pytest.raises(ValueError,match='净开口'):
        scenario(tmp_path,floor=100.,plan_override=prepared(tmp_path))


@pytest.mark.parametrize('fault',['用户停止','通信离线'])
def test_release_cancellation_still_prevents_return(tmp_path,fault):
    with pytest.raises(ValueError,match=fault):
        scenario(tmp_path,fault=fault,plan_override=prepared(tmp_path))
