"""并行准备真实入口/状态机；用假硬件验证不复位、不重入、失败不运动。"""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import parallel_prepare as p


def config(tmp_path):
    curve=tmp_path/'curve.json'
    curve.write_text(json.dumps(dict(robot_id=345,hardware='dm_type1',measured=True,
        arms={s:[[5,330],[45,180],[90,10]] for s in ('left','right')})))
    return dict(arm='right',grasp_backend='vendor_ik_trajectory',supervised_experiment=True,hardware_execution=True,
                gripper_width_calibration=str(curve),gripper_speed_scale=.5,
                gripper_open_tolerance_deg=4.,gripper_open_deg=10.)


@pytest.mark.parametrize('kind',['grasp','place'])
@pytest.mark.parametrize('fault',[None,'stop','blocked','wrong_plan','pause','mobile'])
def test_preparation_process_single_lease_and_handoff(monkeypatch,tmp_path,kind,fault):
    import yaml,rclpy
    from autolife_shelf_grasp_306 import execution as e
    from autolife_shelf_grasp_306 import trajectory_preload,feedback_refresh,pose_manual_session,demo_protocol
    import autolife_handeye_calibration_306.bridge as bridge
    cfg=config(tmp_path);(tmp_path/'config').mkdir();(tmp_path/'config/grasp.yaml').write_text(yaml.safe_dump(cfg))
    (tmp_path/'heartbeat').touch();now=[100.];monkeypatch.setattr(time,'monotonic',lambda:now[0])
    monkeypatch.setattr(time,'time',lambda:now[0]);monkeypatch.setattr(Path,'stat',Path.stat) # 心跳用真实墙钟，下面每步刷新。
    original_time=__import__('os').path.getmtime
    monkeypatch.setattr(time,'time',lambda:original_time(tmp_path/'heartbeat'))
    paused=[False]
    def sleep(dt):
        now[0]+=dt
        if fault=='stop' and now[0]>100.4:(tmp_path/'stop').touch()
        if fault=='pause' and not paused[0] and now[0]>100.5:
            (tmp_path/'task_pause').touch();paused[0]=True
        if paused[0] and now[0]>100.9:(tmp_path/'task_pause').unlink(missing_ok=True)
    monkeypatch.setattr(time,'sleep',sleep)
    q={'Joint_Left_Gripper':300.,'Joint_Right_Gripper':220.};commands=[];events=[]
    node=NS(_current=q,_hardware_ready=True,_gripper_pub=object(),_is_feedback_fresh=lambda:True,
        snapshot=lambda:dict(q),_lock=lambda x:events.append('lock'),destroy_node=lambda:events.append('destroy'))
    def publish(pub,value):
        commands.append((now[0],value))
        if fault!='blocked':
            for side in ('left','right'):q[f'Joint_{side.title()}_Gripper']=max(13.19,float(value[f'{side}_gripper_target_joints_position'][0]))
    node._publish_vendor=publish
    class Session:
        def __init__(self,n,**kw):self.kw=kw;self.active=False;events.append(('session',kw))
        def enter_prepared(self,spin,refreshed):events.append('enable');self.active=True;spin()
        def tick(self):
            if fault!='mobile':return
            file=tmp_path/'parallel_prepare_state.json'
            state=json.loads(file.read_text()) if file.exists() else {}
            if state.get('status')=='ready' and 'released' not in events:
                (tmp_path/'mobile_reach_control.json').write_text(json.dumps(dict(prepare_token='T',operation='release_base')))
                events.append('released')
            elif state.get('status')=='base_released':
                (tmp_path/'mobile_reach_control.json').write_text(json.dumps(dict(prepare_token='T',operation='reacquire_base')))
                events.append('returned')
            elif state.get('status')=='ready' and 'returned' in events:
                (tmp_path/'parallel_prepare_execute.json').write_text(json.dumps(dict(token='T',filename='plan.json',plan_id='P',product_id='O')))
        def close(self,**kw):events.append(('close',kw))
    class Base:
        def __init__(self,n):self.last=[100,0,0,0,0,0]
        def check(self):pass
        def close(self):events.append('base_close')
    class Preload:
        def __init__(self,*a):events.append('preload')
        def result(self,spin):spin();return lambda:None
        def close(self):events.append('preload_close')
    monkeypatch.setattr(bridge,'competing_controller',lambda:None)
    monkeypatch.setattr(e,'CalibrationBridge',lambda:node)
    monkeypatch.setattr(e,'hold_groups',lambda n,g:events.append(('hold',g)))
    monkeypatch.setattr(rclpy,'init',lambda **kw:events.append('init'))
    monkeypatch.setattr(rclpy,'spin_once',lambda *a,**kw:None)
    monkeypatch.setattr(rclpy,'ok',lambda:True);monkeypatch.setattr(rclpy,'shutdown',lambda:events.append('shutdown'))
    monkeypatch.setattr(trajectory_preload,'TrajectoryPreload',Preload)
    monkeypatch.setattr(feedback_refresh,'refresh',lambda *a:dict(q));monkeypatch.setattr(feedback_refresh,'await_snapshot',lambda *a:None)
    monkeypatch.setattr(pose_manual_session,'ManualHoldSession',Session);monkeypatch.setattr(demo_protocol,'StationaryBaseGuard',Base)
    handoff=[]
    monkeypatch.setattr(e,'main',lambda argv,prepared:handoff.append((argv,prepared)))
    (tmp_path/'plan.json').write_text(json.dumps(dict(parallel_prepare_token='wrong' if fault=='wrong_plan' else 'T',kind='place' if kind=='place' else 'grasp')))
    (tmp_path/'parallel_prepare_execute.json').write_text(json.dumps(dict(token='T',filename='plan.json',plan_id='P',product_id='O')))
    monkeypatch.setattr(sys,'argv',['prepare','--root',str(tmp_path),'--run',str(tmp_path),'--token','T','--kind',kind])
    if fault=='mobile':(tmp_path/'parallel_prepare_execute.json').unlink()
    p.main()
    if fault=='mobile':assert events.count('base_close')==1 and 'returned' in events
    failed=fault in ('stop','wrong_plan')
    # 放置没有张爪时间，stop/pause注入需在交接前发生才适用。
    if kind=='place' and fault=='stop':failed=False
    assert bool(handoff)!=failed
    assert events.count('enable')==1 and events.count('init')==1
    if kind=='place':assert not commands and q=={'Joint_Left_Gripper':300.,'Joint_Right_Gripper':220.}
    if failed:assert 'destroy' in events and any(isinstance(x,tuple) and x[0]=='close' for x in events)
    else:
        assert handoff[0][1]['node'] is node and handoff[0][1]['session'].active
        assert 'destroy' not in events and not any(isinstance(x,tuple) and x[0]=='close' for x in events)
    if commands:
        assert all(10<=value[f'{side}_gripper_target_joints_position'][0]<=300 for _,value in commands for side in ('left','right'))
        for side in ('left','right'):
            speeds=[abs((b[1][f'{side}_gripper_target_joints_position'][0]-a[1][f'{side}_gripper_target_joints_position'][0])/(b[0]-a[0])) for a,b in zip(commands,commands[1:])]
            assert max(speeds)<=120.01


def test_actual_session_reuse_never_enables_again():
    import textwrap
    from autolife_shelf_grasp_306 import execution
    source=Path(execution.__file__).read_text();a=source.index("        mark('enable_requested')");b=source.index('        if native_motion:\n            response=node.enable_planned_waist',a)
    events=[];env=dict(prepared={'token':'T'},session=NS(tick=lambda:events.append('tick')),spin=lambda:events.append('spin'),mark=lambda x:events.append(x))
    exec(textwrap.dedent(source[a:b]),env)
    assert events==['enable_requested','spin','tick','enable_ready','sync_ready']


@pytest.mark.parametrize('order',['vision_first','enable_first'])
def test_visual_action_requires_both_branches_once(order):
    from autolife_shelf_grasp_306.teaching_studio import TeachingStudio
    launches=[]
    pnd=dict(token='a',prepare_token='p',body_ready=False,result=None,kwargs=dict(preserve_grippers_at_entry=True))
    fake=NS(parallel_action=pnd,engine=NS(execute=lambda *a,**kw:launches.append((a,kw))))
    result=dict(record={'test':1},source={'joint':12},base_anchor=[0,0,0])
    for branch in ([True,False] if order=='vision_first' else [False,True]):
        if branch:pnd['result']=result
        else:pnd['body_ready']=True
        TeachingStudio._launch_parallel_action(fake)
    TeachingStudio._launch_parallel_action(fake)
    assert len(launches)==1 and launches[0][1]['preserve_grippers_at_entry'] and fake.parallel_action is None


def test_prepared_opening_reuses_measured_clearance_without_wait(tmp_path):
    import ast
    from autolife_shelf_grasp_306 import execution
    cfg=config(tmp_path);jaw='Joint_Right_Gripper';state=dict(closing=False,targets={jaw:10.},tolerance=4.,minimum_opening_mm=88.,required_opening_mm=90.)
    ns=dict(__package__=execution.__package__,prepared={'token':'t'},vendor_grasp=True,cfg=cfg,jaw=jaw,np=np,
        checked=lambda:{jaw:13.19})
    main=next(n for n in ast.parse(Path(execution.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    fn=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='finish_gripper')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual prepared opening>','exec'),ns)
    assert ns['finish_gripper'](state)=='opening_clear'
    state['minimum_opening_mm']=89.9
    with pytest.raises(ValueError,match='不足'):ns['finish_gripper'](state)


@pytest.mark.parametrize('placing',[False,True])
def test_actual_ui_launch_reuses_preparation_process(tmp_path,placing):
    from autolife_shelf_grasp_306.app import Window
    submissions=[]
    data=dict(filename='plan.json',plan_id='P',target=dict(instance_id=1,product_id='O'))
    fake=NS(cancel_pending=False,run=tmp_path,root=tmp_path,demo_active=False,placement_active=placing,demo_phase='place',
        pending_plan=data,selected_id=lambda:1,parallel_prepare=NS(submit=lambda *a:submissions.append(a)))
    placement=dict(plan_id='P',product_id='O') if placing else None
    Window.launch_motion(fake,'plan.json',placement=placement)
    assert submissions==[(tmp_path/'plan.json','P','O')]


def test_late_cancelled_visual_reply_cannot_stop_new_action():
    from autolife_shelf_grasp_306.teaching_studio import TeachingStudio
    replies=[];stops=[]
    fake=NS(_acquire=lambda:None,ensure=lambda:None,_unlock=lambda:None,
        engine=NS(bridge=None,moving=lambda:False,stop=lambda:stops.append(1)),
        visual=NS(request=lambda action,callback,**kw:replies.append(callback)))
    TeachingStudio.execute(fake,'old',dict(visual_bindings={'0':{'reference':'r'}}))
    fake.parallel_action=dict(prepare_token='new')
    replies[0](dict(error='旧请求已取消'))
    assert not stops and fake.parallel_action==dict(prepare_token='new')


@pytest.mark.parametrize('fault',[None,'cancel','load_failure'])
def test_idle_warm_is_readonly_and_cancellable(monkeypatch,fault):
    from autolife_shelf_grasp_306.arm_selection import ArmPlannerPool
    from autolife_shelf_grasp_306 import vendor_ik_process
    cfg=dict(grasp_backend='vendor_ik_trajectory',arm='right',auto_arm_selection=False,urdf_path='model')
    pool=ArmPlannerPool(cfg,None,None,None);events=[]
    planner=NS(solver=None,close=lambda:events.append('close'))
    pool.arm_config=lambda side,cfg:(cfg,None);pool.native_planner=lambda *a:planner
    def solver(*a,**kw):
        kw['checkpoint']()
        if fault=='load_failure':raise ValueError('missing library')
        events.append('load');return NS()
    monkeypatch.setattr(vendor_ik_process,'VendorIKProcess',solver)
    result=pool.warm_next(cfg,stopped=lambda:fault=='cancel')
    if fault=='cancel':assert result is None and not events and not pool.warm_attempted
    elif fault=='load_failure':assert result['status']=='deferred' and events==['close']
    else:assert result['status']=='ready' and events==['load']
    if fault!='cancel':assert pool.warm_next(cfg) is None
