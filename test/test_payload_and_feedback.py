import json
from concurrent.futures import Future
from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306.feedback_refresh import GroupRead,FeedbackReadUnavailable,wait_for_measured
from autolife_shelf_grasp_306.held_receipt import has_held,release_payload


def test_release_pending_success_and_retreat_failure(tmp_path):
    held=tmp_path/'held_object.json';held.write_text('{"status":"completed"}')
    plan={'plan_id':'place-1','arm':'right'}
    def fail_open():
        assert json.loads(held.read_text())['status']=='uncertain'
        raise RuntimeError('no opening feedback')
    with pytest.raises(RuntimeError):release_payload(tmp_path,plan,fail_open)
    assert has_held(tmp_path)
    release_payload(tmp_path,plan,lambda:None)
    assert not held.exists()


def test_legacy_uncertain_only_cleared_by_matching_completed_place(tmp_path):
    (tmp_path/'held_object.json').write_text('{"status":"uncertain","plan_id":"p"}')
    for plan,status,expected in [('q','placed',True),('p','stopped',True),('p','placed',False)]:
        (tmp_path/'execution_result.json').write_text(json.dumps(dict(plan_id=plan,status=status,effects=['release_command_started'])))
        assert has_held(tmp_path)==expected


class Client:
    def __init__(self,name,outputs):
        self.name=name;self.outputs=iter(outputs);self.calls=0
    def service_is_ready(self):return True
    def call_async(self,request):
        self.calls+=1;value=next(self.outputs)
        response=json.dumps(dict(protocol=2,group=self.name,positions={self.name:value})) if value is not None else 'missing axis'
        future=Future();future.set_result(NS(success=value is not None,message=response))
        return future


@pytest.mark.parametrize('slow',[False,True])
def test_retry_group_and_refresh_older_groups_when_recovered(slow):
    good=Client('right',[1,3]);bad=Client('left',[None,2])
    batch=GroupRead({'right':good,'left':bad},lambda:None)
    assert batch.poll(0) is None and batch.poll(.01) is None
    t=2 if slow else .25
    assert batch.poll(t) is None
    result=batch.poll(t+.01)
    if slow:
        assert result is None and batch.poll(t+.02) is None
        result=batch.poll(t+.03)
    assert result=={'left':2,'right':3 if slow else 1}
    assert good.calls==(2 if slow else 1) and bad.calls==2


def test_persistent_loss_cancel_and_eventual_recovery():
    calls=[]
    def read():
        if len(calls)<2:raise FeedbackReadUnavailable('left shoulder')
        return {'fresh':1}
    assert wait_for_measured(read,lambda:False,calls.append)=={'fresh':1}
    calls.clear()
    with pytest.raises(ValueError,match='取消'):
        wait_for_measured(read,lambda:bool(calls),calls.append)
    assert calls==['left shoulder']


def test_stale_pre_enable_snapshot_has_recoverable_feedback_code():
    from autolife_shelf_grasp_306.feedback_refresh import await_snapshot
    with pytest.raises(FeedbackReadUnavailable) as missing:
        await_snapshot(lambda:{'shoulder':0.},lambda:None,{'shoulder':4.},timeout=0.)
    assert missing.value.code=='FEEDBACK_UNAVAILABLE'
