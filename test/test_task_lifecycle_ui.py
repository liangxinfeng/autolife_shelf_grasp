"""Qt integration simulations; all actuator/worker dispatch is replaced."""
import json
from unittest.mock import patch
import pytest
from PyQt5 import QtWidgets as Q
from autolife_shelf_grasp_306.app import Window


@pytest.fixture(scope='session')
def qt():
    return Q.QApplication.instance() or Q.QApplication([])


@pytest.fixture
def window(qt,tmp_path):
    (tmp_path/'data').mkdir();run=tmp_path/'run';run.mkdir()
    flow={'steps':[{'kind':'grasp','target':'纸巾'},{'kind':'tts','target':'完成'}]}
    (tmp_path/'data/task_flows.json').write_text(json.dumps({'任务':flow}),encoding='utf8')
    w=Window(tmp_path,run,start_worker=False)
    for timer in (w.timer,w.automatic.timer,w.navigation.product_timer,w.studio.timer,w.workflow.recovery_timer):
        timer.stop()
    w.ready=True;w.busy=False;w.blockers=[]
    w.workflow.next=lambda:None
    commands=[]
    def send(action,**values):
        w.request+=1;w.workflow.note_worker_request(w.request);commands.append((action,values))
    w.send=send;w.commands=commands
    w.launch_motion=lambda *a,**k:pytest.fail('No physical motion allowed in simulation')
    w.workflow.base.start=lambda *a,**k:pytest.fail('No chassis motion allowed in simulation')
    yield w
    w.automatic.abort('cleanup');w.workflow.cancel('cleanup')
    w.automatic.shutdown();w.workflow.recovery_timer.stop();w.hide();w.deleteLater();qt.processEvents()


def start(w):
    w.automatic.start('loop',confirmed=True)
    assert w.automatic.active and w.workflow.active
    w.workflow.phase='scan'


def test_read_only_retry_preserves_loop_and_never_repeats_preparation(window):
    w=window;start(w);f=w.workflow
    task,step,attempt=f.cursor.task_id,f.cursor.step_id,f.cursor.token
    f.fail('无可用识别目标',retryable=True)
    assert f.waiting and w.automatic.active and w.automatic.completed==0
    f.cursor.retry_at=0
    with patch.object(Q.QMessageBox,'question',side_effect=AssertionError('Safe retry has no dialog')):
        f.recover_tick()
    assert f.active and f.cursor.task_id==task and f.cursor.step_id==step and f.cursor.token!=attempt
    assert [a for a,_ in w.commands]==['scan_task']
    assert getattr(w,'parallel_prepare',None) is None


def test_unknown_grasp_does_not_auto_retry_or_exit_loop(window):
    w=window;start(w);f=w.workflow;f.phase='grasp';f.plan_id='current'
    f.mark_dispatch('grasp')
    f.motion_done(dict(status='stopped',plan_id='current',reason='回执丢失',
                       safe_to_retry=False,effect_certainty='may_have_executed'))
    assert f.waiting and f.cursor.resume=='operator' and w.automatic.active
    f.cursor.retry_at=0;f.recover_tick()
    assert not w.commands and f.cursor.index==0


def test_readonly_failure_does_not_stop_idle_channels_and_modal_cancel_wins(window):
    w=window;start(w);f=w.workflow
    with patch.object(w,'stop_motion') as stop:
        f.fail('遮挡',retryable=True)
        stop.assert_not_called()
    f.cursor.retry_at=0
    def while_open(*args,**kwargs):
        f.recover_tick()
        assert f.waiting and not w.commands
        f.cancel('用户取消')
        return Q.QMessageBox.Yes
    with patch.object(Q.QMessageBox,'question',side_effect=while_open):
        f.retry_current()
    assert f.cursor.state=='cancelled' and not w.commands


def test_unknown_relative_move_can_only_reconcile_without_dispatch(window):
    w=window;f=w.workflow
    f.start({'steps':[{'kind':'base','base':{'enabled':True,'moves':[
        {'kind':'forward','amount':30,'speed':10}]}}]},confirmed=True)
    f.phase='base';f.mark_dispatch('base');f.fail('里程计丢失')
    with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.No):
        f.retry_current()
    assert f.waiting and f.cursor.index==0
    with patch.object(Q.QMessageBox,'question',return_value=Q.QMessageBox.Yes):
        f.retry_current()
    assert f.cursor.state=='completed' and '现场已确认' in f.cursor.warnings[-1]['reason']


def test_current_execution_with_wrong_plan_cannot_retry_or_leave_running(window):
    w=window;start(w);f=w.workflow;f.phase='grasp';f.plan_id='current';f.mark_dispatch('grasp')
    f.motion_done(dict(status='stopped',plan_id='old',reason='stale',
                       safe_to_retry=True,effect_certainty='not_dispatched'))
    assert f.waiting and f.cursor.effect=='unknown' and not w.commands
    assert f.cursor.resume=='operator' and '回执与本次规划不一致' in f.cursor.reason


@pytest.mark.parametrize('required',[False,True])
def test_optional_speech_failure_continues_required_speech_waits(window,required):
    w=window;f=w.workflow
    f.start({'steps':[{'kind':'tts','target':'done','required':required},
                     {'kind':'base','base':{'enabled':False,'moves':[]}}]},confirmed=True)
    f.phase='tts';f.speech_done(f.cursor.token,False,'声卡不可用')
    assert f.cursor.index==(0 if required else 1)
    assert f.waiting is required
    if not required:assert f.cursor.warnings[-1]['reason']=='声卡不可用'


def test_duplicate_action_receipt_does_not_skip_next_step(window):
    w=window;f=w.workflow;start(w)
    f.phase='action';token=f.cursor.token
    f.action_done(token,True,'done');assert f.cursor.index==1
    f.action_done(token,True,'late duplicate');assert f.cursor.index==1


def test_cancelled_safe_retry_never_restarts_on_timer(window):
    w=window;start(w);f=w.workflow
    f.fail('depth',retryable=True);f.cursor.retry_at=0
    w.stop_motion(source='operator')
    f.recover_tick()
    assert not f.pending and not w.automatic.active and not w.commands


def test_restart_restores_waiting_without_worker_or_actuator_dispatch(window):
    w=window;start(w);f=w.workflow;f.phase='grasp';f.mark_dispatch('grasp')
    task_id=f.cursor.task_id
    f.cursor=None;w.automatic.active=False
    f.restore_pending();f.recover_tick()
    assert f.waiting and f.cursor.task_id==task_id and f.cursor.resume=='operator'
    assert not w.commands
    w.automatic.update_view();assert '任务已保留' in w.automatic.state.text()


def test_waiting_ignores_late_worker_completion(window):
    w=window;start(w);f=w.workflow;f.note_worker_request(123)
    f.fail('camera unavailable',retryable=True)
    f.worker_event({'event':'scene','request_id':123,'items':[]})
    assert f.waiting and f.cursor.index==0 and not w.commands


def test_loop_interval_reboot_snapshot_points_to_next_cycle(window):
    w=window;start(w);f=w.workflow
    f.advance();f.advance()
    assert w.automatic.completed==1 and w.automatic.active
    saved=f.store.read()
    assert saved['state']=='waiting' and saved['index']==0 and saved['cycle_index']==1
    assert saved['metadata']['automatic']['completed']==1


def test_optional_speech_startup_failure_also_continues(window):
    from autolife_shelf_grasp_306.task_flow_ui import Workflow
    w=window;f=w.workflow
    f.next=lambda:Workflow.next(f)
    def failed_start(*args):raise OSError('no audio output')
    f.speech.start=failed_start
    f.start({'steps':[{'kind':'tts','target':'done'}]},confirmed=True)
    assert f.cursor.state=='completed' and f.cursor.warnings[-1]['reason']=='no audio output'


def test_completion_disk_failure_retains_step_and_does_not_dispatch_next(window):
    w=window;start(w);f=w.workflow;f.phase='grasp';f.mark_dispatch('grasp');f.plan_id='current'
    class BrokenStore:
        def save(self,value):raise OSError('disk full')
    f.cursor.store=BrokenStore()
    f.motion_done(dict(status='completed',plan_id='current'))
    assert f.waiting and f.cursor.index==0 and f.cursor.effect=='unknown'
    assert not w.commands and f.cursor.resume=='operator'


def test_retry_waits_for_shared_visual_channel_before_starting(window):
    w=window;start(w);f=w.workflow;f.fail('no target',retryable=True);f.cursor.retry_at=0
    with patch.object(w.studio.visual,'pending',{'fixture':True}):
        f.recover_tick()
        assert f.waiting and not w.commands
    f.recover_tick()
    assert f.active and [name for name,_ in w.commands]==['scan_task']


def test_duplicate_grasp_receipt_before_qt_turn_does_not_skip_step(window):
    w=window;start(w);f=w.workflow;f.phase='grasp';f.plan_id='current';f.mark_dispatch('grasp')
    result=dict(status='completed',plan_id='current')
    assert f.motion_done(result)
    assert f.cursor.index==1
    assert not f.motion_done(result)
    assert f.cursor.index==1 and f.cursor.state=='running'


def test_restart_between_steps_can_resume_the_next_step(window):
    w=window;start(w);f=w.workflow;f.advance()
    assert f.cursor.index==1 and f.cursor.phase==''
    f.cursor=None;w.automatic.active=False
    f.restore_pending()
    assert f.waiting and f.retry_available and f.cursor.index==1
    assert f.cursor.metadata['session_name']==w.run.name
