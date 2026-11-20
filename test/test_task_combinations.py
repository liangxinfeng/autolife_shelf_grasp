"""Real UI/event orchestration with fake navigation, worker and actuator edges.

These tests simulate receipts and do not evaluate physical reachability or
publish any hardware command. Motion/model tests cover those separately.
"""
import json
import time
import uuid
from types import SimpleNamespace as NS
import pytest
from PyQt5 import QtWidgets as Q, QtCore as C
from autolife_shelf_grasp_306.app import Window


@pytest.fixture(scope='session')
def qt():
    return Q.QApplication.instance() or Q.QApplication([])


def save(path,value):
    path.write_text(json.dumps(value),encoding='utf8')


@pytest.fixture
def window(qt,tmp_path,monkeypatch):
    (tmp_path/'data').mkdir();(tmp_path/'config').mkdir();run=tmp_path/'run';run.mkdir()
    (tmp_path/'config/grasp.yaml').write_text('require_reset_before_grasp: false\n')
    w=Window(tmp_path,run,start_worker=False)
    for timer in (w.timer,w.automatic.timer,w.navigation.product_timer,w.studio.timer,w.workflow.recovery_timer):timer.stop()
    w.ready=True;w.busy=False;w.blockers=[]
    class Worker:
        def __init__(self):self.commands=[]
        def write(self,data):self.commands.append(json.loads(data))
    w.process=Worker();w.calls=[]
    w.navigation.proc=NS(state=lambda:C.QProcess.Running)
    w.navigation.data=dict(snapshot_time=time.time(),ready=True,pose=[0,0,0],active_map='map',names=['货架','桌前'])
    def navigate(action,**values):
        w.calls.append((action,values))
        w.navigation.inflight=action=='navigate'
    w.navigation.command=navigate
    def launch(filename,preparing=False,placement=None):
        w.workflow.mark_dispatch('prepare' if preparing else w.workflow.phase)
        w.motion_execution_id=uuid.uuid4().hex
        w.calls.append(('motion',dict(filename=filename,placement=placement,
            arm=(placement or w.pending_plan or {}).get('arm'))))
        w.motion=NS();w.parallel_prepare=None
    w.launch_motion=launch
    w.workflow.base.start=lambda token:w.calls.append(('base',dict(token=token)))
    w.studio.execute=lambda token,record,**kw:w.calls.append(('action',dict(token=token,record=record,**kw)))
    w.workflow.speech.start=lambda token,text:w.calls.append(('speech',dict(token=token,text=text)))
    monkeypatch.setattr(Q.QMessageBox,'question',lambda *a,**kw:Q.QMessageBox.Yes)
    monkeypatch.setattr(Q.QMessageBox,'warning',lambda *a,**kw:pytest.fail('Unexpected recovery dialog'))
    yield w
    w.motion=None;w.parallel_prepare=None;w.navigation.inflight=False;w.navigation.proc=None;w.process=None
    w.mobile_reach.cancel();w.workflow.cancel('test cleanup');w.automatic.abort('test cleanup');w.automatic.shutdown()
    w.hide();w.deleteLater();qt.processEvents()


def action_record():
    from control_center_joint_control.joint_definitions import ALL_JOINTS,GROUP_ORDER,JOINT_LIMITS
    q={n:sum(JOINT_LIMITS[n])/2 for n in ALL_JOINTS}
    return dict(groups=GROUP_ORDER,loop_count=1,speed_percent=100,
        keyframes=[dict(joints_deg=q,transition_sec=3.,hold_sec=0.,easing='smooth')])


def scene(w):
    w.receive_event(dict(event='scene',request_id=w.request,
        items=[dict(id=1,name='水瓶',product_id='water',score=.95),
               dict(id=2,name='水瓶',product_id='water',score=.9)],
        shape=[720,1280,3],elapsed_ms=1))


@pytest.mark.parametrize('eligible_ids',[[],[1],[2],[1,2]])
def test_all_targets_are_checked_here_before_any_dock_request(window,eligible_ids):
    w=window
    w.mobile_reach_enabled.setChecked(True)
    w.workflow.start(dict(steps=[dict(kind='grasp',target='水瓶')]),confirmed=True)
    scene(w)
    for ident in (1,2):
        request=w.process.commands[-1]
        assert request['id']==ident and request['stationary_only']
        w.receive_event(dict(event='error',request_id=w.request,text='路径未通过',
                             candidate_unavailable=True,dock_eligible=ident in eligible_ids))
    plans=[r for r in w.process.commands if r['action']=='plan']
    if eligible_ids:
        assert [(r['id'],r['stationary_only']) for r in plans]==[(1,True),(2,True),(eligible_ids[0],False)]
        assert w.workflow.candidates==eligible_ids
        assert plans[-1]['request_id']!=plans[0]['request_id']
    else:
        assert len(plans)==2 and w.workflow.waiting
    assert not w.navigation.inflight and not w.mobile_reach.pending


def planned(w,arm='left',pid='a'*32):
    event=dict(event='planned',request_id=w.request,plan_id=pid,filename='plan_'+pid+'.json',
        requires_reset=False,target=dict(instance_id=w.selected_id(),product_id='water',name='水瓶'),
        blockers=[],shape=[720,1280,3],pixels=[],seconds=1,arm=arm)
    w.receive_event(event)
    return event


def test_stationary_identity_loss_tries_next_target_without_docking(window):
    from autolife_shelf_grasp_306.selection import resolve_selected,require_selected
    from autolife_shelf_grasp_306.auto_grasp import CandidateUnavailable,StationaryPathsExhausted
    w=window;w.mobile_reach_enabled.setChecked(True)
    w.workflow.start(dict(steps=[dict(kind='grasp',target='水瓶')]),confirmed=True)
    scene(w)
    selected,identity=resolve_selected({'id':1},[])
    with pytest.raises(CandidateUnavailable) as failure:
        require_selected(selected,identity,task_scan=True,after_move=False)
    w.receive_event(dict(event='error',request_id=w.request,text=str(failure.value),
        candidate_unavailable=True,dock_eligible=isinstance(failure.value,StationaryPathsExhausted)))
    commands=[r for r in w.process.commands if r['action']=='plan']
    assert [(r['id'],r['stationary_only']) for r in commands]==[(1,True),(2,True)]
    assert w.workflow.active and not w.workflow.dock_candidates
    assert not w.navigation.inflight and not w.mobile_reach.pending


def finish_motion(w,result,*,correlate=True):
    if correlate:result={**result,'execution_id':w.motion_execution_id}
    save(w.run/'execution_result.json',result);w.motion_finished()


def test_occlusion_recovers_after_six_empty_scans_then_completes(window,qt):
    w=window;f=w.workflow
    f.start(dict(steps=[dict(kind='grasp',target='水瓶'),dict(kind='tts',target='完成')]),confirmed=True)
    step=f.cursor.step_id
    for _ in range(6):
        w.receive_event(dict(event='scene',request_id=w.request,items=[],shape=[720,1280,3],elapsed_ms=1))
        assert f.waiting and f.cursor.resume=='automatic' and f.cursor.step_id==step
        assert not w.calls
        f.cursor.retry_at=0
        f.recover_tick()
        assert f.phase=='scan' and w.process.commands[-1]['action']=='scan_task'
    w.receive_event(dict(event='scene',request_id=w.request,
        items=[dict(id=3,name='水瓶',product_id='water',score=.09)],shape=[720,1280,3],elapsed_ms=1))
    assert w.process.commands[-1]['id']==3
    event=planned(w,'right');qt.processEvents()
    assert f.phase=='grasp'
    held(w,event['plan_id'],'right')
    finish_motion(w,dict(status='completed',plan_id=event['plan_id'],arm='right'));qt.processEvents()
    assert f.phase=='tts'
    f.speech_done(f.cursor.token,True,'完成')
    assert f.cursor.state=='completed' and [k for k,_ in w.calls]==['motion','speech']


def held(w,pid,arm):
    product=dict(id='water',name='水瓶',geometry='cylinder',diameter_m=.06,height_m=.2)
    save(w.run/('plan_'+pid+'.json'),dict(run_id=w.run.name,plan_id=pid,arm=arm,config=dict(arm=arm),product=product))
    save(w.run/'held_object.json',dict(status='completed',plan_id=pid,arm=arm,
        q={'Joint_'+arm.title()+'_Gripper':12.}))


@pytest.mark.parametrize('certainty',['not_dispatched','may_have_executed'])
def test_place_feedback_loss_after_multiple_retries_preserves_payload(window,qt,certainty):
    w=window;f=w.workflow
    held(w,'a'*32,'right')
    f.start(dict(steps=[dict(kind='place')]),confirmed=True)
    pid='b'*32
    w.receive_event(dict(event='place_planned',request_id=w.request,plan_id=pid,
        filename='place.json',product_id='water',shape=[720,1280,3],pixels=[],arm='right'))
    f.cursor.retry_count=5
    finish_motion(w,dict(status='stopped',plan_id=pid,reason='等待右臂反馈',
        reason_code='FEEDBACK_UNAVAILABLE',effect_certainty=certainty,safe_to_retry=certainty=='not_dispatched'))
    assert f.waiting and (w.run/'held_object.json').is_file()
    assert f.cursor.resume==('automatic' if certainty=='not_dispatched' else 'operator')
    f.cursor.retry_at=0
    before=len(w.process.commands)
    f.recover_tick();qt.processEvents()
    if certainty=='not_dispatched':
        assert f.phase=='place' and len(w.process.commands)==before+1
        assert w.process.commands[-1]['action']=='standalone_place'
    else:assert len(w.process.commands)==before and f.waiting


@pytest.mark.parametrize('arm',['left','right'])
def test_navigation_candidates_grasp_base_observe_place_action_speech(window,qt,arm):
    w=window;f=w.workflow;record=action_record();w.studio.records=lambda names=None:{'收尾':record}
    f.start(dict(steps=[dict(kind='navigate',target='货架'),dict(kind='grasp',target='水瓶'),
        dict(kind='base',base=dict(enabled=True,moves=[dict(kind='backward',amount=5.,speed=5.)])),
        dict(kind='place'),dict(kind='action',target='收尾',execution_arm=arm),
        dict(kind='tts',target='已完成')]),confirmed=True)
    identity=f.cursor.task_id
    assert f.phase=='navigate' and w.calls[-1][0]=='navigate'
    leg=f.cursor.token;w.navigation.inflight=False
    f.navigation_done(dict(run_id=f.cursor.run_id,leg=leg,success=True,map='map'));qt.processEvents()
    assert f.phase=='scan' and w.process.commands[-1]['action']=='scan_task'
    scene(w);assert w.process.commands[-1]['id']==1
    w.receive_event(dict(event='error',request_id=w.request,text='第一候选无可用路径',candidate_unavailable=True))
    assert f.phase=='plan' and w.process.commands[-1]['id']==2
    event=planned(w,arm);qt.processEvents()
    assert f.phase=='grasp' and w.calls[-1][1]['arm']==arm
    held(w,event['plan_id'],arm)
    finish_motion(w,dict(status='completed',plan_id=event['plan_id'],arm=arm));qt.processEvents()
    assert f.phase=='base' and f.cursor.index==2
    base_token=f.cursor.token;f.base_done(base_token,True,'到位');qt.processEvents()
    assert f.phase=='place' and w.process.commands[-1]['action']=='standalone_place'
    observation='b'*32
    w.receive_event(dict(event='observe_planned',request_id=w.request,plan_id=observation,
        filename='observe.json',product_id='water',arm=arm))
    assert w.demo_phase=='observe' and f.cursor.index==3
    finish_motion(w,dict(status='observed',plan_id=observation,arm=arm))
    assert w.demo_phase=='planning' and w.process.commands[-1]['action']=='standalone_place'
    placement='c'*32
    w.receive_event(dict(event='place_planned',request_id=w.request,plan_id=placement,
        filename='place.json',product_id='water',shape=[720,1280,3],pixels=[],arm=arm))
    assert w.demo_phase=='place'
    (w.run/'held_object.json').unlink()
    finish_motion(w,dict(status='placed',plan_id=placement,arm=arm));qt.processEvents()
    assert f.phase=='action' and w.calls[-1][0]=='action'
    assert w.calls[-1][1]['record']['retarget_arm']==arm
    action_token=f.cursor.token;f.action_done(action_token,True,'完成');qt.processEvents()
    assert f.phase=='tts'
    f.action_done(action_token,True,'重复回执');f.base_done(base_token,True,'重复回执')
    assert f.cursor.index==5
    f.speech_done(f.cursor.token,False,'声卡不可用')
    assert f.cursor.state=='completed' and f.cursor.task_id==identity
    assert f.cursor.warnings[-1]['reason']=='声卡不可用'
    assert [kind for kind,_ in w.calls]==['navigate','motion','base','motion','motion','action','speech']


def test_operation_docking_reobserves_then_uses_new_plan(window,qt,monkeypatch):
    from autolife_shelf_grasp_306 import parallel_prepare_ui
    w=window;f=w.workflow
    f.start(dict(steps=[dict(kind='grasp',target='水瓶')]),confirmed=True);scene(w)
    request_before=w.request
    proposal=dict(route_type='parking_chain',kind='grasp',token='d'*32,created=time.monotonic(),
        map='map',chain_name='货架工作区',chain_sha256='e'*64,goal_pose=[.2,0,0],source_map_pose=[0,0,0],
        workspace_margin_m=.1,path_checked=True,checked_states=100,arm='right',resume=dict(action='plan',id=1))
    def prepare(owner,kind):owner.parallel_prepare=NS(token='f'*32)
    monkeypatch.setattr(parallel_prepare_ui,'start',prepare)
    w.receive_event(dict(event='base_reposition',request_id=w.request,proposal=proposal));qt.processEvents()
    mover=w.mobile_reach;mover.timer.stop()
    assert mover.stage=='yield' and f.cursor.effect=='unknown'
    save(w.run/'parallel_prepare_state.json',dict(token='f'*32,status='base_released',stamp=time.monotonic()))
    mover.tick();assert mover.stage=='moving' and w.calls[-1][1]['leg']=='operation_dock'
    assert f.cursor.index==0
    mover.navigation_event(dict(event='nav_done',run_id=proposal['token'],leg='operation_dock',success=True))
    save(w.run/'parallel_prepare_state.json',dict(token='f'*32,status='ready',stamp=time.monotonic()))
    mover.tick();assert mover.stage=='replan' and w.request>request_before
    assert w.process.commands[-1]['_mobile_done']==proposal['token']
    assert len(w.process.commands[-1]['_operation_visited'])==2
    stale=dict(event='error',request_id=request_before,text='旧规划失败',candidate_unavailable=True)
    w.receive_event(stale);assert mover.stage=='replan' and f.active
    event=planned(w,'right');qt.processEvents()
    assert not mover.active and f.phase=='grasp'
    assert [kind for kind,_ in w.calls]==['navigate','motion']
    finish_motion(w,dict(status='completed',plan_id=event['plan_id'],arm='right'))
    assert f.cursor.state=='completed'


def test_paused_completion_is_latched_once_and_keeps_same_attempt(window,qt):
    w=window;f=w.workflow;p=w.automatic
    p.library={'任务':dict(steps=[dict(kind='tts',target='first'),dict(kind='tts',target='second')])}
    p.names.addItem('任务');p.names.setCurrentText('任务');p.show_display=lambda:None
    p.start('single',confirmed=True)
    token=f.cursor.token;p.toggle_pause()
    f.speech_done(token,True,'done');f.speech_done(token,True,'duplicate')
    assert f.cursor.state=='paused' and f.cursor.index==0 and len(p.deferred)==1
    p.toggle_pause();qt.processEvents()
    assert f.cursor.index==1 and f.cursor.run_id==token.split(':')[0]
    f.speech_done(token,True,'late');assert f.cursor.index==1
    f.speech_done(f.cursor.token,True,'done')
    assert not p.active and p.completed==1 and f.cursor.state=='completed'


def test_action_entry_grip_policy_survives_restart_after_receipt_consumed(window):
    w=window;f=w.workflow;w.studio.records=lambda names=None:{'持物动作':action_record()}
    save(w.run/'held_object.json',dict(status='completed',plan_id='a'*32))
    f.start(dict(steps=[dict(kind='action',target='持物动作')]),confirmed=True)
    assert w.calls[-1][1]['preserve_grippers_at_entry'] is True
    assert not (w.run/'held_object.json').exists()
    f.cursor=None;f.restore_pending();assert f.waiting
    f.retry_current()
    assert f.active and w.calls[-1][1]['preserve_grippers_at_entry'] is True
    assert f.cursor.metadata['action_entry_step']==f.cursor.step_id


def test_new_cycle_recovery_enters_action_before_consuming_held_record(window):
    from autolife_shelf_grasp_306.task_flow import Cursor
    w=window;f=w.workflow;record=dict(steps=[dict(kind='action',target='持物动作')])
    actions={'持物动作':action_record()}
    c=Cursor(record,metadata=dict(actions=actions,map='map',session_name=w.run.name),store=f.store)
    c.phase='cycle_interval';c.wait('等待下一轮')
    save(w.run/'held_object.json',dict(status='completed',plan_id='a'*32))
    f.restore_pending();f.retry_current()
    assert f.active and w.calls[-1][1]['preserve_grippers_at_entry'] is True
    assert not (w.run/'held_object.json').exists()


@pytest.mark.parametrize('receipt',['missing','wrong'])
def test_missing_or_wrong_execution_receipt_keeps_unknown_step_waiting(window,qt,receipt):
    w=window;f=w.workflow
    f.start(dict(steps=[dict(kind='grasp',target='水瓶'),dict(kind='tts',target='done')]),confirmed=True)
    scene(w);event=planned(w);qt.processEvents()
    assert f.phase=='grasp' and w.motion_execution_id
    result=dict(status='completed',plan_id=event['plan_id'])
    if receipt=='wrong':result['execution_id']='not-this-execution'
    finish_motion(w,result,correlate=False)
    assert f.waiting and f.cursor.index==0 and f.cursor.effect=='unknown'
    assert f.cursor.resume=='operator' and '回执' in f.cursor.reason
    assert w.motion_execution_id is None
    assert [kind for kind,_ in w.calls].count('motion')==1
