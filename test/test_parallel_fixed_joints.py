"""执行器真实回调回归：双爪并行张开不应被算作另一手臂移动；全程无硬件输出。"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306 import execution as e


def callback(tmp_path, arm='left', prepared=None, backend=True, token='T'):
    other='right' if arm=='left' else 'left'
    source={n:0. for n in e.BODY+e.NECK+e.ARMS['left']+e.ARMS['right']}
    source.update(Joint_Left_Gripper=11.3566,Joint_Right_Gripper=270.38339614868164)
    current=dict(source);reports=[]
    tree=ast.parse(Path(e.__file__).read_text(encoding='utf8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignments={n.targets[0].id:n for n in ast.walk(main)
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('parallel_grasp_opening','compare_source')}
    fn=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='checked')
    node=NS(_hardware_ready=True,snapshot=lambda:dict(current))
    env=dict(prepared=prepared,vendor_grasp=backend,plan=dict(plan_id='replay',parallel_prepare_token=token),
        source=source,run=tmp_path,node=node,supervision=e.MotionSupervisor(tmp_path),base_guard=None,
        BODY=e.BODY,NECK=e.NECK,ARMS=e.ARMS,other=other,jaw=e.gripper_name(arm),
        gripper_name=e.gripper_name,workspace_reset=False,preparing=False,native_motion=backend,
        arm_reset_active=False,placing=False,observing=False,released=False,
        atomic_json=lambda path,value:reports.append((path.name,value)))
    env['spin']=lambda:env['base_guard'].check() if env['base_guard'] is not None else None
    exec(compile(ast.Module(body=[assignments['parallel_grasp_opening'],assignments['compare_source'],fn],
        type_ignores=[]),'<actual executor checked>','exec'),env)
    return env,current,reports


@pytest.mark.parametrize('arm',['left','right'])
def test_authorized_parallel_opening_is_not_fixed_body_motion(tmp_path,arm):
    env,q,reports=callback(tmp_path,arm,dict(kind='grasp',token='T'))
    original=dict(env['source'])
    for side in ('left','right'):q[e.gripper_name(side)]=13.19
    assert env['checked']()==q and not reports
    assert env['source']==original  # 不重新绑定身体/手臂参考。
    assert all('Gripper' not in n for n in env['compare_source'])


@pytest.mark.parametrize('prepared,backend,token',[
    (None,True,'T'),(dict(kind='place',token='T'),True,'T'),
    (dict(kind='grasp',token='wrong'),True,'T'),(dict(kind='grasp'),True,None),
    (dict(kind='grasp',token='T'),False,'T')])
def test_unrelated_or_unverified_preparation_does_not_exempt_fixed_jaw(tmp_path,prepared,backend,token):
    env,q,reports=callback(tmp_path,prepared=prepared,backend=backend,token=token)
    q['Joint_Right_Gripper']=13.19
    assert not env['parallel_grasp_opening'] and 'Joint_Right_Gripper' in env['compare_source']
    with pytest.raises(ValueError,match='Joint_Right_Gripper'):env['checked']()
    assert reports[0][1]['delta']==pytest.approx(257.19339614868164)


@pytest.mark.parametrize('arm',['left','right'])
@pytest.mark.parametrize('part,index',[('body',0),('body',1),('neck',0),('neck',1),
    *[('other_arm',i) for i in range(7)]])
def test_actual_uncommanded_joint_movement_still_reports_the_joint(tmp_path,arm,part,index):
    env,q,reports=callback(tmp_path,arm,dict(kind='grasp',token='T'))
    joint={'body':e.BODY,'neck':e.NECK,'other_arm':e.ARMS[env['other']]}[part][index]
    q[e.gripper_name(env['other'])]=13.19
    q[joint]=.71
    with pytest.raises(ValueError,match=joint):env['checked']()
    assert reports[0][0]=='fixed_joint_check.json'
    assert reports[0][1]['joint']==joint and reports[0][1]['actual']==.71


@pytest.mark.parametrize('arm',['left','right'])
def test_planned_arm_and_waist_motion_is_not_a_fixed_joint_violation(tmp_path,arm):
    env,q,reports=callback(tmp_path,arm,dict(kind='grasp',token='T'))
    for joint in e.ARMS[arm]+e.BODY[2:]:q[joint]=5.
    q[e.gripper_name(env['other'])]=13.19
    assert env['checked']()==q and not reports


def test_parallel_opening_does_not_bypass_hardware_or_base_failure(tmp_path):
    env,q,reports=callback(tmp_path,prepared=dict(kind='grasp',token='T'))
    env['node']._hardware_ready=False
    with pytest.raises(ValueError,match='控制权'):env['checked']()
    env['node']._hardware_ready=True
    def blocked():raise ValueError('base unavailable')
    env['base_guard']=NS(check=blocked)
    with pytest.raises(ValueError,match='base unavailable'):env['checked']()


def test_non_grasp_arms_reset_still_uses_its_checked_source(tmp_path):
    env,q,reports=callback(tmp_path)
    env['arm_reset_active']=True
    env['plan']['arms_reset']=dict(source_q={**env['source'],'Joint_Waist_Pitch':3.})
    q['Joint_Waist_Pitch']=3.
    assert env['checked']()==q
    q['Joint_Waist_Pitch']=4.
    with pytest.raises(ValueError,match='Joint_Waist_Pitch'):env['checked']()
