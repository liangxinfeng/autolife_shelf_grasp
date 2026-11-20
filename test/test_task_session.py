"""Restart simulations: no ROS, process launch, or motor commands."""
import json
import os
import pytest
from autolife_shelf_grasp_306.core import atomic_json
from autolife_shelf_grasp_306.task_session import open_session
from autolife_shelf_grasp_306.task_runtime import TaskRuntime,TaskStore
from autolife_shelf_grasp_306.placement_request import held_grasp,new_standalone_request,standalone_source


def save(root,name,state='running',kind='place',**options):
    task=TaskRuntime({'steps':[{'kind':kind,'target':'target'}]},
        store=TaskStore(root/'data/task_runtime.json'),metadata={'session_name':name,**options})
    task.state=state;task.persist();return task


def evidence(session,arm='left'):
    pid='a'*32
    plan=dict(plan_id=pid,run_id=session.name,arm=arm,config={'arm':arm},
        product=dict(id='water',geometry='cylinder',diameter_m=.06,height_m=.2),tcp=[0,0,.1],
        segments=[dict(phase='grasp',waypoints=[{'joint':1}])])
    receipt=dict(status='completed',plan_id=pid,arm=arm,q={'Joint_'+arm.title()+'_Gripper':12.})
    atomic_json(session/('plan_'+pid+'.json'),plan);atomic_json(session/'held_object.json',receipt)
    return plan,receipt


@pytest.mark.parametrize('arm',['left','right'])
def test_restart_copies_verified_holding_but_never_renews_old_control_session(tmp_path,arm):
    old,_=open_session(tmp_path);plan,receipt=evidence(old,arm);save(tmp_path,old.name)
    heartbeat=old/'heartbeat';heartbeat.touch();os.utime(heartbeat,(10,10))
    controls=('parallel_prepare_execute.json','mobile_reach_control.json','mobile_reach_request.json',
        'prepare_result.json','execution_result.json','standalone_place.json','observation_attempt.json',
        'demo_observe_result.json','navigation_state.json','task_flow.json','task_pause')
    for name in controls:(old/name).write_text('stale control')
    fresh,warning=open_session(tmp_path)
    assert fresh!=old and not warning
    assert heartbeat.stat().st_mtime==10 and (old/'stop').exists()
    assert set(p.name for p in fresh.iterdir())=={'held_object.json','plan_'+plan['plan_id']+'.json'}
    copied,held=held_grasp(fresh)
    assert copied['run_id']==fresh.name and copied['plan_id']==plan['plan_id'] and held['q']==receipt['q']
    assert copied['segments']==plan['segments'] and copied['recovered_from_session']==old.name
    assert json.loads((old/('plan_'+plan['plan_id']+'.json')).read_text())==plan
    state=TaskStore(tmp_path/'data/task_runtime.json').read()
    assert state['metadata']['session_name']==fresh.name and state['state']=='waiting' and state['resume']=='operator'
    context=new_standalone_request(fresh,{})
    assert standalone_source(fresh)[0]==context and context['session_id']==fresh.name


def test_second_restart_uses_another_control_session_and_keeps_original_evidence(tmp_path):
    first,_=open_session(tmp_path);plan,_=evidence(first);save(tmp_path,first.name)
    second,_=open_session(tmp_path);third,warning=open_session(tmp_path)
    assert len({first,second,third})==3 and not warning
    assert (first/'stop').exists() and (second/'stop').exists()
    grasp,held=held_grasp(third)
    assert grasp['plan_id']==plan['plan_id'] and grasp['recovered_from_session']==first.name
    assert TaskStore(tmp_path/'data/task_runtime.json').read()['metadata']['session_name']==third.name


@pytest.mark.parametrize('fault',['arm','run','missing_plan','bad_json','missing_held'])
def test_incomplete_holding_never_commits_a_recovery_pointer(tmp_path,fault):
    old,_=open_session(tmp_path);plan,receipt=evidence(old);save(tmp_path,old.name)
    path=old/('plan_'+plan['plan_id']+'.json')
    if fault=='arm':receipt['arm']='right';atomic_json(old/'held_object.json',receipt)
    elif fault=='run':plan['run_id']='other-session';atomic_json(path,plan)
    elif fault=='missing_plan':path.unlink()
    elif fault=='bad_json':(old/'held_object.json').write_text('{')
    else:(old/'held_object.json').unlink()
    fresh,warning=open_session(tmp_path)
    assert warning and fresh!=old and (old/'stop').exists()
    assert not (fresh/'held_object.json').exists()
    assert TaskStore(tmp_path/'data/task_runtime.json').read()['metadata']['session_name']==old.name


def test_pending_nonholding_task_gets_fresh_empty_control_session(tmp_path):
    old,_=open_session(tmp_path);save(tmp_path,old.name,kind='navigate')
    (old/'task_flow.json').write_text('old command context')
    fresh,warning=open_session(tmp_path)
    assert fresh!=old and not warning and not list(fresh.iterdir())
    assert (old/'stop').exists()
    assert TaskStore(tmp_path/'data/task_runtime.json').read()['metadata']['session_name']==fresh.name


def test_completed_task_gets_new_session(tmp_path):
    first,_=open_session(tmp_path);save(tmp_path,first.name,'completed')
    second,_=open_session(tmp_path);assert second!=first


def test_missing_and_invalid_session_are_not_reused(tmp_path):
    for name in ('missing','../outside','/tmp/another','..'):
        (tmp_path/'data').mkdir(exist_ok=True);save(tmp_path,name,'waiting')
        session,warning=open_session(tmp_path)
        assert warning and session.parent==tmp_path/'data/sessions'


def test_cycle_completion_crash_uses_new_control_session_without_replaying_round(tmp_path):
    old,_=open_session(tmp_path)
    task=save(tmp_path,old.name,kind='tts',automatic={'limit':0,'completed':0})
    task.complete(task.token)
    fresh,warning=open_session(tmp_path)
    assert fresh!=old and not warning and (old/'stop').exists()
    restored=TaskRuntime.restore(TaskStore(tmp_path/'data/task_runtime.json'))
    assert restored.cycle_index==1 and restored.index==0 and restored.resume=='operator'


def test_grasp_evidence_is_required_during_intermediate_navigation_too(tmp_path):
    old,_=open_session(tmp_path)
    task=TaskRuntime({'steps':[{'kind':'grasp','target':'water'},{'kind':'navigate','target':'table'},
                              {'kind':'place','target':''}]},
        store=TaskStore(tmp_path/'data/task_runtime.json'),metadata={'session_name':old.name})
    task.complete(task.token)
    fresh,warning=open_session(tmp_path)
    assert warning and fresh!=old
    assert TaskStore(tmp_path/'data/task_runtime.json').read()['metadata']['session_name']==old.name
