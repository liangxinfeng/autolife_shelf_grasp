"""No ROS or hardware: crash, retry and duplicate receipt simulations."""
from copy import deepcopy
import json
import pytest
from autolife_shelf_grasp_306.task_runtime import TaskRuntime, TaskStore


def record():
    return {'steps': [{'kind': 'grasp', 'target': 'cup'}, {'kind': 'tts', 'target': 'done'}]}


def test_transient_retry_keeps_task_and_step_but_fences_old_result(tmp_path):
    task=TaskRuntime(record(), store=TaskStore(tmp_path/'task.json'))
    task_id,step_id,old=task.task_id,task.step_id,task.token
    task.wait('depth transient',retryable=True,now=10.)
    assert task.pending and not task.active and task.resume=='automatic'
    task.retry()
    assert task.task_id==task_id and task.step_id==step_id and task.token!=old
    assert not task.complete(old)
    assert task.complete(task.token)
    assert task.index==1


def test_retry_budget_ends_in_waiting_not_failure_or_busy_loop():
    task=TaskRuntime(record())
    for _ in range(2):
        task.wait('no candidate',retryable=True)
        assert task.resume=='automatic'
        task.retry()
    task.wait('still no candidate',retryable=True)
    assert task.state=='waiting' and task.resume=='operator' and task.index==0
    with pytest.raises(ValueError):task.retry()
    task.retry(confirmed=True)
    assert task.active and task.retry_count==0


@pytest.mark.parametrize('phase',['navigate','base','action','grasp','place','prepare','posture'])
def test_unknown_physical_outcome_never_auto_repeats(phase,tmp_path):
    task=TaskRuntime(record(),store=TaskStore(tmp_path/'task.json'))
    task.dispatch(phase)
    assert json.loads((tmp_path/'task.json').read_text())['effect']=='unknown'
    task.wait('lost acknowledgment',retryable=True,await_input=True)
    assert task.resume=='operator'
    with pytest.raises(ValueError):task.retry()
    assert task.index==0


def test_restore_after_dispatch_requires_review_and_does_not_advance(tmp_path):
    store=TaskStore(tmp_path/'task.json')
    task=TaskRuntime(record(),store=store)
    task.dispatch('grasp')
    recovered=TaskRuntime.restore(store)
    assert recovered.task_id==task.task_id and recovered.index==0
    assert recovered.state=='waiting' and recovered.resume=='operator'
    assert recovered.effect=='unknown'
    with pytest.raises(ValueError):recovered.retry()
    assert not recovered.complete(task.token)


def test_restart_during_safe_retry_does_not_automatically_start(tmp_path):
    store=TaskStore(tmp_path/'task.json');task=TaskRuntime(record(),store=store)
    task.wait('camera',retryable=True)
    recovered=TaskRuntime.restore(store)
    assert recovered.resume=='operator' and not recovered.active


def test_duplicate_completion_and_cancelled_attempt_cannot_advance():
    task=TaskRuntime(record());first=task.token
    assert task.complete(first)
    assert not task.complete(first)
    second=task.token;task.cancel('operator stop')
    assert not task.complete(second) and task.index==1 and not task.pending
    with pytest.raises(ValueError):task.retry(confirmed=True)


def test_pause_cannot_dispatch_and_does_not_create_new_attempt():
    task=TaskRuntime(record());token=task.token
    task.pause()
    with pytest.raises(ValueError):task.dispatch('grasp')
    assert not task.complete(token)
    task.unpause();assert task.token==token and task.complete(token)


def test_success_and_cancel_are_terminal_on_restore(tmp_path):
    store=TaskStore(tmp_path/'task.json');task=TaskRuntime(record(),store=store)
    task.complete(task.token);task.complete(task.token)
    assert TaskRuntime.restore(store) is None
    task=TaskRuntime(record(),store=store);task.cancel('operator')
    assert TaskRuntime.restore(store) is None


def test_durable_intent_failure_prevents_caller_dispatch():
    class Store:
        def save(self,data):
            if data['effect']=='unknown':raise OSError('disk full')
    task=TaskRuntime(record(),store=Store())
    emitted=[]
    with pytest.raises(OSError):
        task.dispatch('base')
        emitted.append('motor-command')
    assert emitted==[]


def test_next_cycle_snapshot_recovers_next_round_not_completed_one(tmp_path):
    store=TaskStore(tmp_path/'task.json');task=TaskRuntime(record(),store=store)
    task.complete(task.token);task.complete(task.token)
    saved=task.snapshot();saved.update(state='waiting',index=0,cycle_index=1,effect='none')
    store.save(saved);recovered=TaskRuntime.restore(store)
    assert recovered.cycle_index==1 and recovered.index==0 and recovered.resume=='operator'


def test_snapshot_is_owned_and_warning_is_diagnostic():
    source=record();task=TaskRuntime(source);source['steps'][0]['target']='different'
    task.warn('optional speech unavailable')
    assert task.step['target']=='cup' and task.index==0 and task.active


def test_completion_save_failure_does_not_advance_or_hide_physical_uncertainty():
    class Store:
        def save(self,data):
            if data['index']>0:raise OSError('disk full')
    task=TaskRuntime(record(),store=Store());task.dispatch('grasp');token=task.token
    with pytest.raises(OSError):task.complete(token)
    assert task.index==0 and task.token==token and task.effect=='unknown' and task.active


def test_resume_save_failure_keeps_pause():
    task=TaskRuntime(record());task.pause()
    class Store:
        def save(self,data):raise OSError('disk full')
    task.store=Store()
    with pytest.raises(OSError):task.unpause()
    assert task.state=='paused'


@pytest.mark.parametrize('limit',[0,2])
def test_restart_between_final_receipt_and_next_cycle_snapshot(limit,tmp_path):
    store=TaskStore(tmp_path/'task.json')
    task=TaskRuntime(record(),store=store,metadata={'automatic':{'limit':limit,'completed':0}})
    task.complete(task.token);task.complete(task.token)
    assert store.read()['state']=='completed'
    resumed=TaskRuntime.restore(store)
    assert resumed.state=='waiting' and resumed.resume=='operator'
    assert resumed.index==0 and resumed.cycle_index==1
    assert resumed.metadata['automatic']['completed']==1


def test_completed_last_cycle_never_recovers_an_extra_round(tmp_path):
    store=TaskStore(tmp_path/'task.json')
    task=TaskRuntime(record(),store=store,cycle_index=1,metadata={'automatic':{'limit':2,'completed':1}})
    task.complete(task.token);task.complete(task.token)
    assert TaskRuntime.restore(store) is None
