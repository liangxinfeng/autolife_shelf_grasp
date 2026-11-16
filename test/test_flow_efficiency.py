"""无ROS节点、无真实运动：验证立即衔接及当前阶段的依赖边界。"""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from autolife_shelf_grasp_306.navigation_logic import StationaryHistory
from autolife_shelf_grasp_306.task_capabilities import worker_required_now
from test_cuda_optional import qt, window, nav
from test_transition_session import session_fixture
from autolife_shelf_grasp_306.task_flow_ui import FlowDialog


def test_arrival_reuses_observed_stop_without_another_timer():
    h=StationaryHistory()
    for i in range(12):h.add(10+i*.05,0,0)
    assert h.ready(10.56)  # 到点状态此时才到；无需重新等到11.06。


@pytest.mark.parametrize('fault',['moving','angular','gap','reversed','nan','reset','stale'])
def test_stationary_history_does_not_reuse_invalid_interval(fault):
    h=StationaryHistory()
    for i in range(12):h.add(10+i*.05,0,0)
    now=10.6
    if fault=='moving':h.add(now,.02,0)
    elif fault=='angular':h.add(now,0,.03)
    elif fault=='gap':now=11.;h.add(now,0,0)
    elif fault=='reversed':now=9.;h.add(now,0,0)
    elif fault=='nan':h.add(now,float('nan'),0)
    elif fault=='reset':h.reset()
    elif fault=='stale':now=11.
    assert not h.ready(now)


def test_elapsed_clock_and_duplicate_samples_do_not_prove_stationary():
    h=StationaryHistory();h.add(10.,0,0)
    for _ in range(20):h.add(10.,0,0)
    assert not h.ready(10.3) and not h.ready(10.6)


@pytest.mark.parametrize('phase,needed',[
    ('navigate',False),('posture',False),('prepare_navigation',False),('base',False),
    ('action',False),('grasp',False),('prepare',True),('scan',True),('plan',True),('confirm',True)])
def test_dependency_follows_current_phase(phase,needed):
    o=NS(workflow=NS(active=True,phase=phase),studio=NS(visual=NS(busy=False)),demo_phase=None)
    assert worker_required_now(o)==needed
    o.studio.visual.busy=True
    assert worker_required_now(o)


@pytest.mark.parametrize('phase,needed',[('planning',True),('place',False),('retreat',False),('observe',False)])
def test_placement_only_requires_worker_when_planning(phase,needed):
    o=NS(workflow=NS(active=True,phase='place'),demo_phase=phase)
    assert worker_required_now(o)==needed


@pytest.mark.parametrize('event',['fatal','vision_unavailable','error','exit'])
def test_visual_failure_does_not_cancel_mixed_task_navigation(window,qt,event):
    w=window;w.receive_event(dict(event='ready',blockers=[]))
    w.workflow.start(dict(steps=[nav(),dict(kind='grasp',target='水瓶')]),confirmed=True)
    w.stop_motion=Mock();w.send=Mock()
    if event=='exit':w.finished()
    else:w.receive_event(dict(event=event,text='视觉暂不可用'))
    assert w.workflow.active and not w.stop_motion.called
    if event!='error':
        assert not w.ready
        f=w.workflow
        f.navigation_done(dict(run_id=f.cursor.run_id,leg=f.cursor.token,success=True,map='map'))
        qt.processEvents()
        assert w.stop_motion.called and not w.send.called  # 不以失效视觉继续抓取。


def test_active_visual_failure_still_reports_and_stops(window):
    w=window;w.receive_event(dict(event='ready',blockers=[]))
    w.workflow.start(dict(steps=[nav(),dict(kind='grasp',target='水瓶')]),confirmed=True)
    w.workflow.phase='scan';w.stop_motion=Mock()
    w.receive_event(dict(event='fatal',text='视觉退出'))
    assert w.stop_motion.called


def test_task_grasp_is_authorized_without_repeated_confirmation(window):
    w=window;w.auto_grasp.setChecked(False)
    assert not w.automatic_grasp_enabled()
    w.workflow.start(dict(steps=[nav()]),confirmed=True)
    assert w.automatic_grasp_enabled()
    w.workflow.stopped('用户停止')
    assert not w.automatic_grasp_enabled()


def test_manual_continue_keeps_completed_steps_and_remaining_task(window,qt):
    w=window;f=w.workflow
    f.start(dict(steps=[nav(),dict(kind='navigate',target='货架'),nav()]),confirmed=True)
    f.navigation_done(dict(run_id=f.cursor.run_id,leg=f.cursor.token,success=True,map='map'))
    qt.processEvents();assert f.cursor.index==1
    f.stopped('临时错误');w.controls()
    assert w.flow_retry.isEnabled() and w.flow_retry.text()=='手动继续'
    assert '第2步' in w.flow_retry.toolTip()
    d=FlowDialog(w);assert d.continue_button.isEnabled()
    w.navigation.command.reset_mock();d.continue_task()
    assert f.active and f.cursor.index==1
    assert w.navigation.command.call_args.kwargs['name']=='货架'
    f.navigation_done(dict(run_id=f.cursor.run_id,leg=f.cursor.token,success=True,map='map'))
    qt.processEvents();assert f.cursor.index==2
    assert w.navigation.command.call_args.kwargs['name']=='前台'
    f.navigation_done(dict(run_id=f.cursor.run_id,leg=f.cursor.token,success=True,map='map'))
    w.controls();assert not f.active and not w.flow_retry.isEnabled()
    d.deleteLater()


def test_continue_from_automatic_page_does_not_restart_loop(window):
    w=window;w.workflow.start(dict(steps=[nav()]),confirmed=True)
    w.workflow.stopped('已停止');w.module_tabs.setCurrentWidget(w.automatic)
    w.automatic.update_view();assert w.automatic.continue_button.isEnabled()
    w.automatic.continue_task()
    assert w.workflow.active and not w.automatic.active
    assert w.module_tabs.currentWidget() is w.grasp_page


def test_retry_nonvisual_step_not_blocked_by_later_visual_step(window):
    w=window;w.receive_event(dict(event='ready',blockers=[]))
    w.workflow.start(dict(steps=[nav(),dict(kind='grasp',target='水瓶')]),confirmed=True)
    w.workflow.stopped('导航已停止');w.ready=False
    w.workflow.retry_current()
    assert w.workflow.active and w.workflow.phase=='navigate'


def test_pending_enable_can_recover_feedback_without_second_enable(monkeypatch):
    n,s,clock,calls,sent,spin=session_fixture(monkeypatch)
    done=[False];client=s.client;original=client.call_async
    def request(r):
        original(r)
        return NS(done=lambda:done[0],result=lambda:NS(success=True,message='held'))
    client.call_async=request
    s.begin(refreshed=dict(n._current));s.on_state(NS(data='{"state":0}'))
    assert not s.poll() and len(calls)==1
    n._is_feedback_fresh=lambda:False
    for _ in range(10):spin();assert not s.poll()
    assert not s.verified and not s.active and not sent
    n._is_feedback_fresh=lambda:True;done[0]=True
    for _ in range(12):
        spin()
        if s.poll():break
    assert s.verified and len(calls)==1


@pytest.mark.parametrize('fault',['timeout','disabled','accepted_stale'])
def test_enable_wait_does_not_mask_real_faults(monkeypatch,fault):
    n,s,clock,calls,sent,spin=session_fixture(monkeypatch)
    s.begin(refreshed=dict(n._current));s.on_state(NS(data='{"state":0}'));s.poll()
    s.future=NS(done=lambda:fault=='accepted_stale',result=lambda:NS(success=True))
    n._is_feedback_fresh=lambda:False
    if fault=='timeout':clock[0]+=13.
    if fault=='disabled':n._hardware_enabled=False
    with pytest.raises(ValueError):s.poll()
    assert not s.verified
