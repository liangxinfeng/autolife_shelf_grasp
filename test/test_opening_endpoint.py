"""张开端点残差与真实开口能力分离；执行器真实回调，无硬件输出。"""
import ast,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import execution as executor
from autolife_shelf_grasp_306.arm_selection import gripper_targets,gripper_name
from autolife_shelf_grasp_306.core import open_tolerance
from autolife_shelf_grasp_306.motion import Arrival


@pytest.mark.parametrize('arm',['left','right'])
@pytest.mark.parametrize('fault',[None,'blocked','stale','stop','insufficient_path','legacy','measurement_rounding'])
def test_real_executor_preopening_endpoint(tmp_path,arm,fault):
    calibration=tmp_path/'jaw.json'
    calibration.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={arm:[[5,330],[45,180],[90,10]]})))
    cfg=dict(arm=arm,gripper_width_calibration=str(calibration),gripper_speed_scale=.5,
        gripper_open_tolerance_deg=4.,gripper_max_width_m=.09,grasp_opening_margin_mm=10.)
    plan=dict(config=cfg,grasp_width_m=.08399525854777365,minimum_opening_m=.08799525854777366)
    if fault=='insufficient_path':plan['minimum_opening_m']=.0898
    if fault=='measurement_rounding':plan['minimum_opening_m']=.08919730813966091
    if fault=='legacy':plan.pop('minimum_opening_m')
    jaw=gripper_name(arm);other=gripper_name('left' if arm=='right' else 'right')
    q={jaw:292.4,other:22.};now=[0.];sent=[];reports=[];holds=[]
    node=NS(_active_goals={},progress_watch={},_gripper_pub=object())
    def publish(pub,data):
        assert data[('left' if arm=='right' else 'right')+'_gripper_target_joints_position']==[22.]
        sent.append((now[0],data[arm+'_gripper_target_joints_position'][0]))
    node._publish_vendor=publish
    def checked():
        now[0]+=.02
        if now[0]>.3 and fault in ('stale','stop'):raise ValueError(fault)
        if sent and fault!='blocked':
            target=max(13.192890167236328,sent[-1][1])
            q[jaw]+=float(np.clip(target-q[jaw],-2.4,2.4))
        return dict(q)
    env=dict(prepared=None,__name__=executor.__name__,__package__=executor.__package__,observing=False,
        checked=checked,motion_now=lambda:now[0],arm=arm,cfg=cfg,gripper_targets=gripper_targets,
        node=node,native_motion=True,vendor_place=False,vendor_grasp=True,open_tolerance=open_tolerance,
        Arrival=Arrival,hold_groups=lambda *a:holds.append(a),atomic_json=lambda p,v:reports.append(v),run=tmp_path,plan=plan)
    tree=ast.parse(Path(executor.__file__).read_text(encoding='utf8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    funcs=[n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name in ('start_gripper','finish_gripper')]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<real opening callbacks>','exec'),env)
    state=env['start_gripper'](10.)
    assert state['minimum_opening_mm']==(None if fault=='legacy' else plan['minimum_opening_m']*1000)
    if fault and fault not in ('legacy','measurement_rounding'):
        with pytest.raises(ValueError):env['finish_gripper'](state)
    else:
        assert env['finish_gripper'](state)=='opening_clear'
        width=float(np.interp(q[jaw],[10,180,330],[90,45,5]))
        assert width<90 and width+.5>=plan.get('minimum_opening_m',plan['grasp_width_m'])*1000
        assert now[0]<4.2 and holds
        assert reports[-1]['effective_clear_width_mm'] is None or reports[-1]['effective_clear_width_mm']<90
    assert sent
    assert all(10<=v<=292.4 for _,v in sent)
    assert max(abs((b[1]-a[1])/(b[0]-a[0])) for a,b in zip(sent,sent[1:]))<=120.01
