"""Fake UI/process requests: late perception must never dispatch old motion."""
import json
from types import SimpleNamespace as NS
import pytest
from autolife_shelf_grasp_306 import visual_action_client as module


class ClientHarness:
    """Exercise the real methods without a Qt event loop or worker process."""
    REQUEST_TIMEOUT_SEC=module.VisualClient.REQUEST_TIMEOUT_SEC
    busy=module.VisualClient.busy
    request=module.VisualClient.request
    _finish=module.VisualClient._finish
    receive=module.VisualClient.receive
    cancel=module.VisualClient.cancel
    fail=module.VisualClient.fail
    tick=module.VisualClient.tick


@pytest.fixture
def client(tmp_path,monkeypatch):
    now=[0.];calls=[];writes=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    value=ClientHarness();value.pending=None
    value.owner=NS(run=tmp_path,busy=False,motion=None,ready=True,
        navigation=NS(inflight=False),
        process=NS(write=lambda data:writes.append(json.loads(data)) or len(data)),
        status=NS(setText=lambda text:calls.append(('status',text))),
        controls=lambda:calls.append(('controls',)))
    value.studio=NS(runtime=NS(busy=False),editor=NS(remote_locked=lambda:False),
        engine=NS(bridge=None),_acquire=lambda:calls.append(('lock',)),
        _unlock=lambda:calls.append(('unlock',)))
    value.now=now;value.calls=calls;value.writes=writes
    return value


def test_timeout_finishes_without_waiting_for_missing_worker(client):
    replies=[];client.request('visual_correct',replies.append,record={})
    ident=client.pending['id'];client.now[0]=121.;client.tick();client.tick()
    assert not client.busy
    assert len(replies)==1 and replies[0]['reason_code']=='VISION_TIMEOUT'
    assert (client.owner.run/('visual_cancel_'+ident)).exists()
    client.receive(dict(visual_id=ident,result={'record':'never execute'}))
    assert len(replies)==1
    assert client.calls.count(('unlock',))==1


def test_late_old_reply_cannot_complete_a_new_request(client):
    old=[];new=[]
    client.request('visual_correct',old.append,record={});old_id=client.pending['id']
    client.now[0]=121.;client.tick()
    client.request('visual_correct',new.append,record={});new_id=client.pending['id']
    client.receive(dict(visual_id=old_id,result={'record':'old'}))
    assert client.busy and not new
    client.receive(dict(visual_id=new_id,result={'record':'new'}))
    client.receive(dict(visual_id=new_id,result={'record':'duplicate'}))
    assert len(old)==1 and len(new)==1 and new[0]['result']['record']=='new'


@pytest.mark.parametrize('finish,code',[('cancel','OPERATOR_CANCELLED'),('fail','VISION_UNAVAILABLE')])
def test_cancel_and_worker_exit_revoke_motion_callback(client,finish,code):
    events=[]
    def ready(data):events.append('motion' if not data.get('error') else data['reason_code'])
    client.request('visual_correct',ready,record={});ident=client.pending['id']
    if finish=='cancel':client.cancel()
    else:client.fail('worker exited')
    assert not client.busy
    client.receive(dict(visual_id=ident,result={'record':'late'}))
    assert events==[code]


def test_callback_reentry_keeps_new_request_lease(client):
    replies=[]
    def first(data):client.request('visual_freeze',replies.append)
    client.request('visual_correct',first,record={})
    client.cancel()
    assert client.busy and client.calls.count(('unlock',))==0
    ident=client.pending['id'];client.receive(dict(visual_id=ident,result={}))
    assert len(replies)==1 and client.calls.count(('unlock',))==1


def test_invalid_request_payload_never_leaves_pending_or_lease(client):
    with pytest.raises(ValueError):
        client.request('visual_correct',lambda data:None,value=float('nan'))
    assert not client.busy and not client.writes
    assert ('lock',) not in client.calls


@pytest.mark.parametrize('raises',[False,True])
def test_write_failure_releases_pending(client,raises):
    def write(data):
        if raises:raise RuntimeError('process closed')
        return -1
    client.owner.process.write=write
    with pytest.raises((ValueError,RuntimeError)):
        client.request('visual_correct',lambda data:None,record={})
    assert not client.busy and client.calls.count(('unlock',))==1
