"""Fault injection for the shared executor contracts; no ROS or hardware."""
import os
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from autolife_shelf_grasp_306.motion_supervision import (
    ExecutionProgress, MotionFault, MotionSupervisor, attempt_cleanup)
from autolife_shelf_grasp_306.phase_arrival import (
    GRASP_ENDPOINT, PLACE_LANDING, wait_endpoint)


def supervisor(tmp_path):
    heartbeat=tmp_path/'heartbeat'
    heartbeat.touch()
    os.utime(heartbeat,(100.,100.))
    return MotionSupervisor(tmp_path,wall_clock=lambda:100.1)


def test_one_cycle_checks_each_source_once(tmp_path):
    events=[];guard=supervisor(tmp_path)
    session=SimpleNamespace(tick=lambda:events.append('session'))
    base=SimpleNamespace(last=(100.,),check=lambda:events.append('base'))
    node=SimpleNamespace(_hardware_ready=True,
        snapshot=lambda:events.append('snapshot') or {'joint':1.})
    guard.poll(lambda:events.append('pump'),session,base)
    assert guard.snapshot(node)=={'joint':1.}
    assert events==['pump','session','base','snapshot']


def test_lost_operator_prevents_session_or_motion_checks(tmp_path):
    events=[];guard=supervisor(tmp_path)
    (tmp_path/'heartbeat').unlink()
    with pytest.raises(MotionFault) as stopped:
        guard.poll(lambda:None,SimpleNamespace(tick=lambda:events.append('session')))
    assert stopped.value.code=='OPERATOR_HEARTBEAT_LOST'
    assert not events


def test_operator_stop_is_not_automatic_retry_permission(tmp_path):
    guard=supervisor(tmp_path);(tmp_path/'stop').touch()
    with pytest.raises(MotionFault) as stopped:guard.check_operator()
    receipt=ExecutionProgress().receipt('stopped',stopped.value)
    assert receipt['effect_certainty']=='not_dispatched'
    assert receipt['safe_to_retry'] is False


def test_dispatch_intent_survives_lost_motor_reply():
    persisted=[];state=ExecutionProgress(on_update=persisted.append)
    state.enter('gripper_releasing')
    def send():
        state.dispatch('release_command_started')
        raise ConnectionError('reply lost after send')
    with pytest.raises(ConnectionError):send()
    assert persisted[-1]['effects']==['release_command_started']
    receipt=state.receipt('stopped')
    assert receipt['effect_certainty']=='may_have_executed'
    assert receipt['safe_to_retry'] is False


def test_receipt_and_progress_bind_to_the_current_execution():
    persisted=[]
    state=ExecutionProgress(execution_id='prepare-token',on_update=persisted.append)
    state.dispatch('hold_command_started')
    assert persisted[-1]['execution_id']=='prepare-token'
    state.execution_id='motion-attempt'
    state.enter('moving')
    state.dispatch('motion_started')
    receipt=state.receipt('stopped')
    assert persisted[-1]['execution_id']==receipt['execution_id']=='motion-attempt'
    assert receipt['effects']==['hold_command_started','motion_started']
    assert not receipt['safe_to_retry']


def test_retreat_cannot_erase_earlier_gripper_effect():
    state=ExecutionProgress()
    state.dispatch('release_command_started');state.dispatch('motion_started')
    state.enter('retreat')
    receipt=state.receipt('stopped')
    assert 'release_command_started' in receipt['effects']
    assert receipt['effect_certainty']=='may_have_executed'
    complete=state.receipt('placed')
    assert complete['effect_certainty']=='command_completed'
    assert 'physical_placement_verified' not in complete


def test_preflight_rejection_has_no_motion_effect():
    receipt=ExecutionProgress().receipt('stopped',ValueError('bad plan'))
    assert receipt['effect_certainty']=='not_dispatched'
    assert receipt['effects']==[]
    assert receipt['safe_to_retry'] is False


@pytest.mark.parametrize('code,allowed',[
    ('PLAN_EXPIRED',True),('SOURCE_POSE_CHANGED',True),
    ('OPERATOR_STOPPED',False),('OPERATOR_HEARTBEAT_LOST',False),
    ('CONTROL_UNAVAILABLE',False),('INVALID_FEEDBACK',False),('UNKNOWN',False)])
def test_only_explicit_replanning_faults_allow_automatic_retry(code,allowed):
    state=ExecutionProgress()
    assert state.receipt('stopped',MotionFault(code,'injected'))['safe_to_retry'] is allowed
    state.dispatch('hold_command_started')
    assert not state.receipt('stopped',MotionFault(code,'injected'))['safe_to_retry']


def test_future_operator_heartbeat_is_not_a_valid_lease(tmp_path):
    guard=supervisor(tmp_path)
    os.utime(tmp_path/'heartbeat',(200.,200.))
    with pytest.raises(MotionFault) as result:guard.check_operator()
    assert result.value.code=='OPERATOR_HEARTBEAT_LOST'


def test_broken_ui_log_pipe_cannot_interrupt_remaining_cleanup(monkeypatch):
    errors=[];events=[]
    def broken_print(*args,**kwargs):raise BrokenPipeError('UI exited')
    def failed_close():raise RuntimeError('session failed')
    monkeypatch.setattr('builtins.print',broken_print)
    assert not attempt_cleanup(errors,'session',failed_close)
    assert attempt_cleanup(errors,'node',lambda:events.append('destroyed'))
    assert events==['destroyed'] and errors[0]['stage']=='session'


def test_enable_failure_after_hold_dispatch_is_not_retryable():
    state=ExecutionProgress()
    state.enter('preparing');state.dispatch('hold_command_started')
    receipt=state.receipt('stopped',ConnectionError('enable reply lost'))
    assert receipt['effect_certainty']=='may_have_executed'
    assert receipt['safe_to_retry'] is False


def test_legacy_open_completion_keeps_target_instead_of_lagging_feedback(tmp_path):
    path=Path(__file__).parents[1]/'autolife_shelf_grasp_306'/'execution.py'
    tree=ast.parse(path.read_text('utf8'))
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='poll_gripper')
    sent=[];holds=[]
    node=SimpleNamespace(_active_goals={'grippers':{}},progress_watch={'grippers':{}},_gripper_pub=None,
        _publish_vendor=lambda pub,payload:sent.append(payload))
    env=dict(node=node,run=tmp_path,atomic_json=lambda *args:None,hold_groups=lambda *args:holds.append(args))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual gripper completion>','exec'),env)
    targets=dict(Joint_Left_Gripper=10.,Joint_Right_Gripper=20.)
    state=dict(result=None,targets=targets,tolerance=4.,closing=False,last_log=0.,start=0.,command_duration=1.)
    result=env['poll_gripper'](state,dict(Joint_Left_Gripper=50.,Joint_Right_Gripper=20.),2.)
    assert result=='command_completed_unverified' and not holds
    assert sent[-1]['left_gripper_target_joints_position']==[10.]


def test_endpoint_resets_stability_after_a_bad_sample():
    values=iter([(0.,0.),(.10,0.),(.14,2.),(.20,0.),(.30,0.),(.36,0.)])
    now=[0.];seen=[]
    def sample():
        now[0],q=next(values);seen.append(now[0]);return {'joint':q}
    q=wait_endpoint(sample,lambda:now[0],{'joint':0.},['joint'],GRASP_ENDPOINT,label='抓点')
    assert q=={'joint':0.} and seen[-1]==.36


def test_landing_never_passes_only_because_joint_error_is_small():
    now=[0.];reports=[]
    def sample():now[0]+=.1;return {'joint':0.}
    with pytest.raises(MotionFault) as stopped:
        wait_endpoint(sample,lambda:now[0],{'joint':0.},['joint'],PLACE_LANDING,
            label='落点',geometry_ok=lambda q:False,report=reports.append)
    assert stopped.value.code=='ENDPOINT_NOT_REACHED'
    assert not any(r['endpoint_reached'] for r in reports)
    assert reports[-1]['remaining_sec']==0.


def test_cancellation_during_arrival_propagates_before_next_phase():
    def stopped():raise MotionFault('OPERATOR_STOPPED','用户停止')
    with pytest.raises(MotionFault) as result:
        wait_endpoint(stopped,lambda:0.,{'joint':0.},['joint'],GRASP_ENDPOINT,label='抓点')
    assert result.value.code=='OPERATOR_STOPPED'


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_invalid_endpoint_feedback_is_not_success(value):
    with pytest.raises(MotionFault) as result:
        wait_endpoint(lambda:{'joint':value},lambda:0.,{'joint':0.},['joint'],GRASP_ENDPOINT,label='抓点')
    assert result.value.code=='INVALID_FEEDBACK'
