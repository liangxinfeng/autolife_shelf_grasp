"""Full reset state machine with fake time/ROS/motors; NEVER hardware."""
import json
import sys
import pytest
import numpy as np
from types import SimpleNamespace
from autolife_shelf_grasp_306 import execution as e
from autolife_handeye_calibration_306.core import ARMS,BODY,NECK


@pytest.mark.parametrize('mode',['hold_reset','hold_reset_failure','hold_reset_drift','hold_reset_stop','old_vendor_reset','unchecked_reset','reset','reset_recovery','reset_recovery_blocked','reset_ready','reset_parallel','reset_open_offset','reset_open_blocked','grasp','grasp_contact','grasp_blocked','grasp_sync_refused','grasp_fixed','grasp_place','grasp_observe','grasp_observe_slip'])
def test_execution_finishes_without_stop(tmp_path,monkeypatch,mode,handoff='unchanged',cleanup_fault=None,operator_confirmed_empty=False):
    names=BODY+NECK+ARMS['left']+ARMS['right']
    source={n:0. for n in names};source.update(Joint_Left_Gripper=180.,Joint_Right_Gripper=180.)
    if mode=='grasp_place_ready':source['Joint_Right_Gripper']=10.
    goal={**source,NECK[1]:-20.,ARMS['left'][3]:20.,ARMS['right'][3]:-20.}
    if mode.startswith('reset_recovery'):source[NECK[1]]=-46.13
    if mode=='reset_ready':source.update(goal);source.update(Joint_Left_Gripper=10.,Joint_Right_Gripper=10.)
    now=[100.];instances=[];grip_prompt=[False];arm_started=[];unlock_calls=[];jaw_events=[]
    class FakeBridge:
        tolerance=1.
        max_step=5.
        def get_parameter(self,n):return SimpleNamespace(value=self.tolerance if n=='goal_tolerance_deg' else self.max_step)
        def set_parameters(self,values):
            for v in values:
                if v.name=='goal_tolerance_deg':self.tolerance=v.value
                else:self.max_step=v.value
            return [SimpleNamespace(successful=True) for _ in values]
        def __init__(self):
            self.q=dict(source);self._hardware_ready=True;self._active_goals={};self._goal_sent_groups=set()
            self.progress_watch={};self._whole_body_pub='body';self._gripper_pub='jaw';instances.append(self)
        def _is_feedback_fresh(self):return True
        @property
        def _feedback_monotonic(self):return now[0]
        def snapshot(self):return dict(self.q)
        def create_client(self,service,name):
            groups=dict(neck=NECK,leg_waist=BODY,left_arm=ARMS['left'],right_arm=ARMS['right'],
                        left_gripper=['Joint_Left_Gripper'],right_gripper=['Joint_Right_Gripper'])
            group=next(g for g in groups if name.endswith('_'+g))
            def response():
                return SimpleNamespace(success=True,message=json.dumps(dict(protocol=2,group=group,
                    positions={n:self.q[n] for n in groups[group]})))
            return SimpleNamespace(service_is_ready=lambda:True,
                call_async=lambda request:SimpleNamespace(done=lambda:True,result=response))
        def destroy_client(self,client):pass
        def unlock(self):
            unlock_calls.append(True)
            if mode=='hold_reset_failure':return SimpleNamespace(success=False,message='使能拒绝')
            if mode=='hold_reset_drift':self.q[BODY[0]]=5.
            if mode=='hold_reset_stop':
                self._hardware_ready=False;self._hardware_enabled=False;self.last_lock_reason='硬件停止'
            return SimpleNamespace(success=True)
        def unlock_with_reset(self):raise AssertionError('禁止调用厂家系统复位')
        def apply(self,group,targets,speed_scale=None):
            assert .1<=speed_scale<=2.
            self._active_goals[group]={'targets':dict(targets),'speed_scale':speed_scale};self._goal_sent_groups.add(group)
            if group=='left_arm':arm_started.append(now[0])
            if group=='grippers':jaw_events.append((self.q[ARMS['left'][3]],targets['Joint_Left_Gripper']))
        def tick(self):
            for entry in self._active_goals.values():self.q.update(entry['targets'])
            if mode=='grasp_observe_slip' and arm_started:self.q['Joint_Left_Gripper']=200.
            if mode=='reset_recovery_blocked':self.q[NECK[1]]=-46.13
            if mode=='reset_parallel' and now[0]<102 and 'grippers' in self._active_goals:
                self.q['Joint_Left_Gripper']=180.;self.q['Joint_Right_Gripper']=180.
            if mode.startswith('reset_open_') and 'grippers' in self._active_goals:
                self.q['Joint_Left_Gripper']=11.627891
                self.q['Joint_Right_Gripper']=12.851880 if mode=='reset_open_offset' else 17.
            if mode in ('grasp_contact','grasp_blocked') and 'grippers' in self._active_goals:
                if self._active_goals['grippers']['targets']['Joint_Left_Gripper']==330.:
                    self.q['Joint_Left_Gripper']=150. if mode=='grasp_contact' else 10.
            # Deliberately keep stale bookkeeping to reproduce old completion
            # dependency: measured stable arrival must still finish autonomously.
        def _publish_vendor(self,*_):pass
        def _lock(self,*_):self._hardware_ready=False
        def destroy_node(self):pass
    monkeypatch.setattr(e,'CalibrationBridge',FakeBridge)
    sessions=[]
    class FakeSession:
        def __init__(self,node,preserve_gripper_targets=False):
            self.node=node;self.active=False;sessions.append(self);self.closed=False
        def enter(self,spin):
            if mode=='grasp_sync_refused':raise ValueError('SYNC rejected')
            self.active=True
            if handoff=='jaw_settle':self.node.q['Joint_Left_Gripper']-=.8524
            if handoff=='jaw_recorded':self.node.q['Joint_Left_Gripper']-=2.3386924574272427
            if handoff=='jaw_large_drift':self.node.q['Joint_Left_Gripper']-=8.1
            if handoff=='arm_drift':self.node.q[ARMS['left'][3]]+=.6
        def tick(self):pass
        def close(self,preserve_grippers=False):
            self.active=False;self.closed=True
            if handoff=='exit_settle':self.node.q['Joint_Left_Gripper']-=3.
            return True
    monkeypatch.setattr(e,'PositionSyncSession',FakeSession)
    monkeypatch.setattr(e,'competing_controller',lambda:'')
    monkeypatch.setattr(e,'validate_grasp_plan',lambda *args:{'bounds_deg':{n:[-180,180] for n in ARMS['left']}})
    monkeypatch.setattr(e.rclpy,'init',lambda **_:None)
    def feedback_tick(n,**kwargs):
        n.tick()
        now[0]+=.001 # Fresh feedback has elapsed time, including final read-only settling.
    monkeypatch.setattr(e.rclpy,'spin_once',feedback_tick)
    monkeypatch.setattr(e.rclpy,'ok',lambda:False)
    monkeypatch.setattr(e.time,'monotonic',lambda:now[0])
    def sleep(dt):
        now[0]+=dt
    monkeypatch.setattr(e.time,'sleep',sleep)
    def printed(*args,**kwargs):
        if any('GRIP_CHECK' in str(a) for a in args):grip_prompt[0]=True
    monkeypatch.setattr('builtins.print',printed)
    import autolife_handeye_calibration_306.core as core
    monkeypatch.setattr(core,'digest',lambda _:'hash')
    monkeypatch.setattr(core,'Kinematics',lambda _:SimpleNamespace(fk=lambda *args:np.eye(4)))
    from autolife_shelf_grasp_306.calibration_profiles import CalibrationProfiles
    # Isolate model/camera resources while retaining the executor's digest check.
    monkeypatch.setattr(CalibrationProfiles,'resolve',lambda self,cfg:({},'hash'))
    from autolife_shelf_grasp_306 import prepare
    monkeypatch.setattr(prepare,'digest',lambda _:'hash')
    if mode.startswith('grasp_place') or mode.startswith('grasp_observe'):
        from autolife_shelf_grasp_306 import demo_protocol
        monkeypatch.setattr(demo_protocol,'validate_place',lambda *args:None)
        monkeypatch.setattr(demo_protocol,'StationaryBaseGuard',lambda node:SimpleNamespace(last=True,check=lambda:None,close=lambda:None))
    cfg={'arm':'left','head_offsets_deg':[0.,0.,0.],'reset_hardware_enabled':True,'plan_ttl_sec':20,'urdf_path':'fake','reset_tolerance_deg':1.,
         'reset_joint_speed_deg_s':70.2,'reset_feedback_lead_deg':5.,'gripper_speed_scale':.5,
         'gripper_open_deg':10.,'gripper_open_tolerance_deg':4.,'gripper_closed_deg':330.,'max_joint_speed_deg_s':120.}
    plan={'kind':'prepare','config':cfg,'created_monotonic':100.,'source_q':source,'urdf_sha256':'hash',
          'self_collision_checked':True,'segments':[{'name':'reset','waypoints':[source,goal]}]}
    if mode.startswith('hold_reset') or mode=='unchecked_reset':
        plan.update(prepare_mode='hold_then_vr',reset_protocol_version=2)
    if mode=='unchecked_reset':plan['self_collision_checked']=False
    if mode=='old_vendor_reset':
        plan.update(prepare_mode='vendor_then_vr',reset_protocol_version=1,self_collision_checked=False,segments=[])
    if mode.startswith('reset_recovery'):
        inside={**source,NECK[1]:-43.}
        plan['segments']=[{'name':'inward','waypoints':[source,inside],
                          'neck_recovery':{NECK[1]:dict(source_deg=-46.13,target_deg=-43.,lower_deg=-45.,upper_deg=30.)}},
                          {'name':'reset','waypoints':[inside,goal]}]
    if mode.startswith('grasp'):
        cfg.update(hardware_execution=True,supervised_experiment=True,calibration='fake',tcp_offset_mm=15.)
        curve=tmp_path/'gripper_curve.json'
        curve.write_text(json.dumps(dict(hardware='dm_type1',measured=True,
            arms={'left':[[0.,330.],[45.,180.],[90.,10.]]})))
        cfg['gripper_width_calibration']=str(curve)
        from autolife_shelf_grasp_306.grip_width import closing_command
        plan.update(grasp_width_m=.05,grip_command=closing_command(.05,50.,cfg))
        plan.update(kind='grasp',geometry_version=2,tcp=[0,0,0],up_torso=[0,0,-1],calibration_sha256='hash',tcp_offset_mm=15.,
                    plan_id='plan-token',run_id=tmp_path.name,prepare_token='reset-token',
                    product={'id':'ice-tea','name':'冰红茶','diameter_m':.07},segments=[])
        previous=dict(source)
        for i,phase in enumerate(['pregrasp','grasp','lift','withdraw']+(['carry'] if mode=='grasp_fixed' else [])):
            end={**previous,ARMS['left'][3]:float((i+1)*5)}
            plan['segments'].append({'name':phase,'phase':phase,'waypoints':[previous,end]});previous=end
        if mode in ('grasp_fixed','grasp_continuous'):
            cfg.update(fixed_carry_enabled=True,carry_pose_deg=[previous[n] for n in ARMS['left']])
            plan['carry_protocol_version']=1
            if mode=='grasp_continuous':
                cfg.update(carry_return_mode='continuous_reset',initial_pose_deg={'left_arm':cfg['carry_pose_deg']},
                           grasp_transit_feedback_lead_deg=5.)
                plan['carry_protocol_version']=2
                segments=plan['segments']
                plan['segments']=segments[:2]+[{'name':'continuous return','phase':'carry',
                    'waypoints':[segments[1]['waypoints'][-1],segments[2]['waypoints'][-1],segments[-1]['waypoints'][-1]]}]
        if mode.startswith('grasp_place'):
            plan['kind']='place'
            for segment,phase in zip(plan['segments'],['place_approach','place_lower','place_release_retreat','place_return']):segment['phase']=phase
            if mode=='grasp_place_ready':
                final=plan['segments'][-1]['waypoints'][-1]
                cfg['initial_pose_deg']={group:[final[n] for n in ns]
                    for group,ns in [('left_arm',ARMS['left']),('right_arm',ARMS['right']),('neck',NECK)]}
            if mode=='grasp_place_continuous':
                plan['place_protocol_version']=3
                ret=plan['segments'].pop(2)
                plan['segments'][-1]['waypoints']=ret['waypoints']+plan['segments'][-1]['waypoints'][1:]
                for segment in plan['segments']:
                    length=sum(max(abs(b[n]-a[n]) for n in ARMS['left']) for a,b in zip(segment['waypoints'],segment['waypoints'][1:]))
                    segment['path_speed_cap']=30.
                    segment['path_speed_profile']=[[0.,30.],[length,30.]]
        if mode.startswith('grasp_observe'):
            plan.update(kind='observe',held_gripper_deg=180.)
            plan['segments']=plan['segments'][:1];plan['segments'][0]['phase']='observe_lift'
        (tmp_path/'prepare_result.json').write_text(json.dumps({'status':'prepared','token':'reset-token'}))
    if mode=='grasp_expired':plan['created_monotonic']=50.
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));(tmp_path/'heartbeat').touch()
    monkeypatch.setattr(sys,'argv',['executor','--plan',str(path),'--run',str(tmp_path),
                                  '--expected-plan-id','plan-token','--expected-product-id','ice-tea',
                                  '--execution-id','current-attempt']+(['--operator-confirmed-empty-grippers'] if operator_confirmed_empty else []))
    cleanup_attempts=[]
    if cleanup_fault:
        original=e.attempt_cleanup
        def attempt(errors,stage,operation):
            def injected():
                cleanup_attempts.append(stage)
                if stage==cleanup_fault:raise RuntimeError('injected '+stage)
                return operation()
            return original(errors,stage,injected)
        monkeypatch.setattr(e,'attempt_cleanup',attempt)
    e.main()
    result=json.loads((tmp_path/'execution_result.json').read_text())
    assert result['execution_id']=='current-attempt'
    if cleanup_fault:
        assert result['status']=='stopped' and not result['safe_to_retry'],result
        assert result['plan_id']=='plan-token'
        if instances:assert 'destroy_node' in cleanup_attempts
        assert 'ros_shutdown' in cleanup_attempts
        assert any(entry['stage']==cleanup_fault for entry in result.get('cleanup_errors',[])+result.get('receipt_errors',[]))
        if mode=='grasp' and cleanup_fault!='held_receipt':
            held=json.loads((tmp_path/'held_object.json').read_text())
            assert held['status']=='uncertain' and not held['grip_success_verified']
        return
    if mode=='grasp_expired':
        assert result['reason_code']=='PLAN_EXPIRED' and result['safe_to_retry']
        assert result['plan_id']=='plan-token' and not result['effects'] and not instances
        return
    if mode in ('old_vendor_reset','unchecked_reset'):
        assert result['status']=='stopped' and not instances and not unlock_calls
        return
    if mode in ('hold_reset_failure','hold_reset_drift','hold_reset_stop'):
        assert result['status']=='stopped' and not arm_started and unlock_calls==[True]
        assert instances[0].q['Joint_Left_Gripper']==180 and instances[0].q['Joint_Right_Gripper']==180
        assert result.get('token') is None
        return
    if mode=='hold_reset':assert unlock_calls==[True] and all(result['q'][n]==source[n] for n in BODY)
    if mode=='reset_recovery_blocked':
        assert result['status']=='stopped' and '颈部回退未到位' in result['reason']
        assert not arm_started and result.get('token') is None
        assert instances[0].q['Joint_Left_Gripper']==180 and instances[0].q['Joint_Right_Gripper']==180
        assert instances[0].tolerance==1.
        return
    if mode.startswith('grasp'):assert len(sessions)==1 and sessions[0].closed,result
    else:assert not sessions
    if handoff=='arm_drift':
        assert result['status']=='stopped' and 'SYNC切换期间' in result['reason']
        assert not arm_started and not jaw_events
        diagnostic=json.loads((tmp_path/'pose_handoff.json').read_text())
        assert not diagnostic['phases']['SYNC切换期间']['ok']
        return
    if mode.startswith('grasp_observe'):
        assert not jaw_events
        assert all(result.get('q',instances[0].q)[n]==source[n] for n in BODY+NECK+ARMS['right'])
        if mode=='grasp_observe':
            assert result['status']=='observed',result
            assert result['q']['Joint_Left_Gripper']==180-{'jaw_settle':.8524,'jaw_recorded':2.3386924574272427,'jaw_large_drift':8.1}.get(handoff,0.)
        else:
            assert result['status']=='observed' and result['q']['Joint_Left_Gripper']==200.
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']=='prepared'
        return
    if mode.startswith('grasp_place'):
        assert result['status']=='placed',result
        assert len(jaw_events)==1 and abs(jaw_events[0][0]-10.)<=.35 and jaw_events[0][1]==10.
        assert result['q']['Joint_Left_Gripper']==10 and result['q']['Joint_Right_Gripper']==source['Joint_Right_Gripper']
        if mode=='grasp_place_ready':
            assert not result['prepared_after_place'] and not result['auto_prepare_allowed']
            receipt=json.loads((tmp_path/'prepare_result.json').read_text())
            assert receipt['status']=='prepared' and receipt['token']=='reset-token'
        assert all(result['q'][n]==source[n] for n in BODY+NECK+ARMS['right'])
        return
    if mode=='grasp_sync_refused':
        assert result['status']=='stopped' and 'SYNC rejected' in result['reason']
        assert not arm_started and instances[0].q==source
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']=='prepared'
        return
    if mode=='reset_open_blocked':
        # Timed opening completion records a position residual without blocking
        # the reset; this is not physical evidence that the jaw fully opened.
        assert result['status']=='prepared',result
        opening=json.loads((tmp_path/'reset_gripper_result.json').read_text())
        assert not opening['arrival_required'] and not opening['physical_result_verified']
        assert opening['measured']['Joint_Right_Gripper']==17.
        return
    if mode.startswith('reset') or mode=='hold_reset':
        assert result['status']=='prepared',result
        assert abs(result['q']['Joint_Left_Gripper']-10)<=4 and abs(result['q']['Joint_Right_Gripper']-10)<=4
        assert abs(result['q'][NECK[1]]+20)<=1.
        assert result['token']
        if mode=='reset_parallel':assert min(arm_started)<102
    else:
        assert result['status']=='completed',result
        assert result['q']['Joint_Left_Gripper']==(150 if mode=='grasp_contact' else 10 if mode=='grasp_blocked' else 330)-(3 if handoff=='exit_settle' else 0)
        held=json.loads((tmp_path/'held_object.json').read_text())
        assert held['q']==result['q'] and held['hold_feedback_confirmed']
        if handoff=='exit_settle':assert held['pre_hold_q']['Joint_Left_Gripper']-held['q']['Joint_Left_Gripper']==3.
        assert result['q']['Joint_Right_Gripper']==180
        assert abs(result['q'][ARMS['left'][3]]-(25. if mode=='grasp_fixed' else 20.))<=.35
        if mode in ('grasp_fixed','grasp_continuous'):
            assert all(result['q'][n]==source[n] for n in BODY+NECK+ARMS['right'])
            closing=next(i for i,event in enumerate(jaw_events) if event[1]==330.)
            assert all(event[1]==330. for event in jaw_events[closing:])
        assert not grip_prompt[0] and not (tmp_path/'grip_confirmed').exists()
        assert result['grip_success_verified'] is False
        assert json.loads((tmp_path/'prepare_result.json').read_text())['status']=='consumed'
    assert not (tmp_path/'stop').exists() and now[0]-100<12.


def test_continuous_return_execution_keeps_closed_jaw(tmp_path,monkeypatch):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp_continuous')


def test_place_single_return_opens_once_after_lowering(tmp_path,monkeypatch):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp_place_continuous')
    assert not (tmp_path/'held_object.json').exists()


@pytest.mark.parametrize('mode',['grasp_place','grasp_observe'])
@pytest.mark.parametrize('handoff',['jaw_settle','jaw_recorded','jaw_large_drift','arm_drift'])
def test_loaded_sync_handoff_full_executor(tmp_path,monkeypatch,mode,handoff):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,mode,handoff)


def test_grasp_receipt_uses_post_sync_settled_feedback(tmp_path,monkeypatch):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp_contact','exit_settle')


def test_placed_return_preserves_preparation_receipt_without_extra_reset(tmp_path,monkeypatch):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp_place_ready')


@pytest.mark.parametrize('stage',['hold','sync_close','lock','destroy_node','base_close','ros_shutdown','held_receipt','prepare_receipt'])
def test_cleanup_fault_never_skips_terminal_receipt_or_remaining_release(tmp_path,monkeypatch,stage):
    mode='grasp_place' if stage=='base_close' else 'hold_reset' if stage=='prepare_receipt' else 'grasp'
    test_execution_finishes_without_stop(tmp_path,monkeypatch,mode,cleanup_fault=stage)


def test_expired_plan_receipt_can_request_replanning_before_any_dispatch(tmp_path,monkeypatch):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp_expired')


def test_cleanup_failure_revokes_otherwise_safe_preflight_retry(tmp_path,monkeypatch):
    test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp_expired',cleanup_fault='ros_shutdown')


def test_missing_execution_receipt_does_not_erase_confirmed_hold_evidence(tmp_path,monkeypatch):
    with pytest.raises(FileNotFoundError):
        test_execution_finishes_without_stop(tmp_path,monkeypatch,'grasp',cleanup_fault='execution_receipt')
    held=json.loads((tmp_path/'held_object.json').read_text())
    assert held['execution_id']=='current-attempt' and held['status']=='completed'
    assert held['hold_feedback_confirmed'] and not held['grip_success_verified']


@pytest.mark.parametrize('kind,confirmed',[('prepare',False)])
def test_existing_hold_rejects_new_grasp_or_unconfirmed_reset_before_hardware(tmp_path,monkeypatch,kind,confirmed):
    original=b'{"status":"uncertain","plan_id":"old-grasp","raw":"preserve exactly"}\n'
    (tmp_path/'held_object.json').write_bytes(original)
    path=tmp_path/'rejected_plan.json'
    path.write_text(json.dumps(dict(kind=kind,config={},created_monotonic=0)))
    def forbidden(*args,**kwargs):pytest.fail('Held-object rejection must precede ROS/controller creation')
    monkeypatch.setattr(e,'CalibrationBridge',forbidden)
    monkeypatch.setattr(e.rclpy,'init',forbidden)
    monkeypatch.setattr(e.rclpy,'ok',lambda:False)
    e.main(['--plan',str(path),'--run',str(tmp_path),'--execution-id','held-boundary']+
           (['--operator-confirmed-empty-grippers'] if confirmed else []))
    result=json.loads((tmp_path/'execution_result.json').read_text())
    assert result['status']=='stopped' and result['reason_code']=='HELD_OBJECT_PRESENT'
    assert result['effects']==[] and result['effect_certainty']=='not_dispatched' and not result['safe_to_retry']
    assert (tmp_path/'held_object.json').read_bytes()==original and not (tmp_path/'held_history').exists()


@pytest.mark.parametrize('mode',['hold_reset','unchecked_reset'])
def test_confirmed_manual_reset_archives_exact_evidence_only_after_preflight(tmp_path,monkeypatch,mode):
    original=b'{"status":"completed","plan_id":"previous","q":{"Joint_Left_Gripper":181}}\n'
    (tmp_path/'held_object.json').write_bytes(original)
    test_execution_finishes_without_stop(tmp_path,monkeypatch,mode,operator_confirmed_empty=True)
    result=json.loads((tmp_path/'execution_result.json').read_text())
    if mode=='unchecked_reset':
        assert (tmp_path/'held_object.json').read_bytes()==original
        assert not (tmp_path/'held_history').exists() and result['effects']==[]
    else:
        assert not (tmp_path/'held_object.json').exists()
        history=list((tmp_path/'held_history').iterdir())
        assert len(history)==1 and history[0].read_bytes()==original
        assert result['held_receipt_archived_after_empty_gripper_confirmation'] is True
