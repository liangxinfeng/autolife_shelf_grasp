"""QProcess callback and handoff fault injection; no Qt process or robot node."""
import json
from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306 import parallel_prepare_ui as module


class Pending:
    finished=module.ParallelPrepare.finished
    error=module.ParallelPrepare.error
    submit=module.ParallelPrepare.submit


@pytest.fixture
def pending(tmp_path):
    events=[];value=Pending()
    value._finished=False;value.handed=False;value.token='prepare-token'
    value.proc=NS(state=lambda:module.C.QProcess.Running)
    value.output=lambda:events.append('output')
    value.deleteLater=lambda:events.append('delete')
    value.owner=NS(run=tmp_path,parallel_prepare=value,motion=None,preparing=True,
        workflow=NS(active=True,fail=lambda reason:events.append(('failed',reason))),
        motion_finished=lambda *args:events.append('motion_done'),
        controls=lambda:events.append('controls'),execute=NS(setEnabled=lambda flag:events.append(('enabled',flag))))
    value.events=events
    return value


def test_failed_start_and_finished_complete_only_once(pending):
    pending.error(module.C.QProcess.FailedToStart)
    pending.finished(-1,module.C.QProcess.CrashExit)
    assert sum(isinstance(event,tuple) and event[0]=='failed' for event in pending.events)==1
    assert pending.events.count('delete')==1 and pending.events.count('controls')==1
    assert pending.owner.parallel_prepare is None


def test_handed_process_duplicate_finished_does_not_complete_next_step(pending):
    pending.handed=True;pending.owner.parallel_prepare=None;pending.owner.motion=pending.proc
    pending.finished(0);pending.finished(0)
    assert pending.events.count('motion_done')==1


@pytest.mark.parametrize('handed',[False,True])
def test_late_old_process_cannot_fail_a_new_attempt(pending,handed):
    pending.handed=handed
    newer=object();pending.owner.parallel_prepare=newer;pending.owner.motion=newer
    pending.finished(0)
    assert pending.events==['delete']
    assert pending.owner.parallel_prepare is newer and pending.owner.motion is newer


def test_late_process_output_cannot_replace_current_status(pending):
    pending.handed=True;pending.owner.motion=object()
    pending.proc.readAllStandardOutput=lambda:b'old process output'
    pending.owner.status=NS(setText=lambda text:pending.events.append(text))
    module.ParallelPrepare.output(pending)
    assert not pending.events


def test_submit_write_failure_keeps_preparation_ownership(pending,monkeypatch):
    def fail(*args):raise OSError('disk full')
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(OSError):
        pending.submit(pending.owner.run/'plan.json','P','O',execution_id='attempt')
    assert not pending.handed and pending.owner.parallel_prepare is pending
    assert pending.owner.motion is None and not pending.events


def test_submit_is_committed_once_and_keeps_attempt_id(pending):
    path=pending.owner.run/'plan.json'
    pending.submit(path,'P','O',execution_id='attempt')
    saved=json.loads((pending.owner.run/'parallel_prepare_execute.json').read_text())
    assert saved['execution_id']=='attempt' and saved['token']=='prepare-token'
    assert pending.handed and pending.owner.motion is pending.proc
    with pytest.raises(ValueError):pending.submit(path,'P','O',execution_id='other')
    assert json.loads((pending.owner.run/'parallel_prepare_execute.json').read_text())==saved


def test_preparation_error_ignores_previous_receipt_reason(pending):
    (pending.owner.run/'execution_result.json').write_text(json.dumps(
        dict(execution_id='old',reason='stale failure')))
    pending.finished(1)
    failures=[event[1] for event in pending.events if isinstance(event,tuple) and event[0]=='failed']
    assert len(failures)==1 and 'stale failure' not in failures[0]


def test_owner_is_set_before_a_synchronous_start_failure(monkeypatch):
    owner=NS(parallel_prepare=None)
    class Starts:
        def __init__(self,owner,kind):self.owner=owner
        def begin(self):
            assert self.owner.parallel_prepare is self
            self.owner.parallel_prepare=None
        def deleteLater(self):pass
    monkeypatch.setattr(module,'ParallelPrepare',Starts)
    module.start(owner,'grasp')
    assert owner.parallel_prepare is None
