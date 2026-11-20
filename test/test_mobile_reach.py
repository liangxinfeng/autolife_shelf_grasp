import copy
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import mobile_reach as m
from test_auto_grasp import window,qt,scene,plan


def box(center=(0,0,.5),half=(.1,.1,.1)):
    return dict(rotation=np.eye(3).tolist(),translation=list(center),low=(-np.array(half)).tolist(),high=list(half))


def distant():
    return np.random.default_rng(8).uniform([1.,-.5,.2],[1.2,.5,1.5],(500,3))


def test_world_point_roundtrip_and_turn():
    for a,b in [([0,0,0],[.1,0,0]),([2,3,1.],[2.1,2.9,.9])]:
        p=np.array([.7,-.3,.6]);q=m.map_point(p,a,b)
        assert np.allclose(m.map_point(q,b,a),p)
    assert np.allclose(m.map_point([1,0,1],[0,0,0],[0,0,np.pi/2]),[0,-1,1])


def test_shift_support_plane_not_dimensions_or_source():
    p=dict(height_m=.03,object_center_offset_wrist=[.01,0,0],low_object_support=dict(normal=[0,0,1],offset=.6,center=[.7,0,.615]))
    old=copy.deepcopy(p);delta=np.array([.1,0,.02]);q=m.shifted_product(p,delta)
    assert p==old and q['height_m']==.03 and q['object_center_offset_wrist']==[.01,0,0]
    assert q['low_object_support']['offset']==pytest.approx(.58)


def test_table_support_grid_and_obstacles_translate_together():
    from autolife_shelf_grasp_306.placement_support import pack_support,unpack_support
    table=dict(minimum=[.5,-.3,.6],maximum=[1,.3,.6],coef=[.1,0,.55],
        obstacles=[dict(minimum=[.7,0,.7],maximum=[.8,.1,.8])],footprint_support=pack_support(np.ones((3,3),bool),[.6,.1],.005))
    t=m.shifted_table(table,[.1,0,0]);grid,origin,res=unpack_support(t['footprint_support'])
    assert grid.all() and np.allclose(origin,[.5,.1])
    assert t['coef'][2]==pytest.approx(.56)
    assert t['obstacles'][0]['minimum'][0]==pytest.approx(.6)
    assert table['minimum'][0]==.5


@pytest.mark.parametrize('x',[.12,.149,.20,.249])
def test_full_swept_volume_rejects_obstacle_between_endpoints(x):
    p=np.vstack([distant(),[x,0,.5]])
    assert not m.sweep_clear(p,[box()],.15)


def test_sweep_retains_overhead_clearance_and_held_object():
    p=np.vstack([distant(),[.18,0,.9]])
    assert m.sweep_clear(p,[box()],.15)
    assert not m.sweep_clear(p,[box()],.15,box((0,0,.9)))
    assert m.sweep_clear(np.vstack([distant(),[0,0,.5]]),[box()],.15)


@pytest.mark.parametrize('change',[dict(forward_m=.151),dict(forward_m=-.1),dict(forward_m=float('nan')),
    dict(reach_checked=False),dict(sweep_checked=False),dict(created=1),dict(token='x')])
def test_compensation_protocol_does_not_authorize_unbounded_or_stale_request(change):
    d=dict(token='a'*32,forward_m=.1,reach_checked=True,decision_stage="recognition_workspace",sweep_checked=True,created=100)
    d.update(change)
    with pytest.raises(ValueError):m.validate_request(d,'a'*32,100)


def test_valid_request_uses_existing_bounded_controller():
    d=dict(token='a'*32,forward_m=.1,reach_checked=True,decision_stage="recognition_workspace",sweep_checked=True,created=100)
    assert m.validate_request(d,'a'*32,100)==dict(enabled=True,moves=[dict(kind='translate',x_cm=10.,y_cm=0.,speed=12.)])


def test_current_reachable_never_constructs_planner_or_runs_ik(monkeypatch):
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(np.zeros(3),1.))
    pool=NS(fk=NS(fk=lambda *a:np.eye(4)),arm_config=lambda *a:({},[0,0,0]),
        native_planner=lambda *a:pytest.fail('范围内不得试算IK或完整路径'))
    result=m.grasp_reposition(pool,{},[.5,0,.6],dict(height_m=.1),dict(auto_arm_selection=False,arm='right',up_torso=[0,0,1]),[0,0,0],distant(),lambda:False,lambda s:None)
    assert result is None


def test_outside_range_uses_only_base_sweep_not_full_grasp_plan(monkeypatch):
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(np.array([0,0,.6]),.7))
    monkeypatch.setattr(m,'collision_boxes',lambda *a:[box()])
    p=NS(plan=lambda *a:pytest.fail('补偿前不得计算完整抓取路径'))
    pool=NS(fk=NS(fk=lambda *a:np.eye(4)),arm_config=lambda *a:({},[0,0,0]),native_planner=lambda *a:p)
    result=m.grasp_reposition(pool,{},[.74,0,.6],dict(height_m=.1),dict(auto_arm_selection=False,arm='right',up_torso=[0,0,1]),[0,0,0],distant(),lambda:False,lambda s:None)
    assert result and .05<=result['translation_m'][0]<=.06 and not result['path_checked']
    assert result['reach_checked'] and result['sweep_checked'] and result['checked_states']==0
    assert m.validate_request(result,result['token'])['enabled']
    # 同一个距离提议遇到机身前方真实点，不能启动补偿。
    with pytest.raises(ValueError,match='补偿通道存在障碍'):
        m.grasp_reposition(pool,{},[.74,0,.6],{},dict(auto_arm_selection=False,arm='right',up_torso=[0,0,1]),[0,0,0],np.vstack([distant(),[.13,0,.5]]),lambda:False,lambda s:None)


def test_preview_offer_never_starts_base_or_enables(monkeypatch):
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.mobile_reach_ui import MobileReach
    app=Q.QApplication.instance() or Q.QApplication([])
    owner=Q.QWidget();owner.controls=lambda:None;owner.automatic_grasp_enabled=lambda:False
    owner.demo_active=owner.placement_active=False;owner.status=NS(setText=lambda s:None)
    owner.workflow=NS(active=False);owner.navigation=NS(map=NS(operation_target=None,update=lambda:None))
    reach=MobileReach(owner);calls=[];reach.start=lambda:calls.append('move')
    reach.offer(dict(proposal={'chain_name':'货架工作区','goal_pose':[.1,0,0],
                              'workspace_margin_m':.1,'path_checked':True}));app.processEvents()
    assert not calls and reach.pending and not reach.active


def test_ui_late_worker_result_cannot_restart_stopped_compensation():
    from PyQt5 import QtWidgets as Q
    from autolife_shelf_grasp_306.mobile_reach_ui import MobileReach
    app=Q.QApplication.instance() or Q.QApplication([]);owner=Q.QWidget();mover=MobileReach(owner)
    mover.stage='replan';mover.cancel();mover.worker_event(dict(event='planned',plan_id='late'))
    app.processEvents();assert not mover.active and mover.pending is None


def test_real_window_compensation_then_fresh_plan_single_execution(window,qt,monkeypatch):
    import time
    from autolife_shelf_grasp_306 import parallel_prepare_ui
    w=window;scene(w);w.send('plan',id=1);controller=w.mobile_reach
    record=dict(route_type='parking_chain',kind='grasp',token='d'*32,created=time.monotonic(),
        map='map',chain_name='货架工作区',chain_sha256='e'*64,goal_pose=[.1,0,0],source_map_pose=[0,0,0],
        workspace_margin_m=.1,path_checked=True,checked_states=120,arm='right')
    record['resume']=dict(action='plan',id=1)
    record['request_id']=w.request
    w.receive_event(dict(event='base_reposition',request_id=w.request,proposal=record))
    qt.processEvents()
    assert not controller.active and w.execute.isEnabled() and '操作点' in w.execute.text()
    calls=[]
    def prepare(owner,kind):
        calls.append(('prepare',kind));owner.parallel_prepare=NS(token='P'*32)
    monkeypatch.setattr(parallel_prepare_ui,'start',prepare)
    def move():controller.stage='moving';calls.append(('base',.1))
    monkeypatch.setattr(controller,'move',move)
    w.execute.click();assert calls==[('prepare','grasp')] and controller.stage=='yield'
    state=w.run/'parallel_prepare_state.json'
    state.write_text(json.dumps(dict(token='P'*32,status='base_released',stamp=time.monotonic())))
    controller.tick();assert calls[-1]==('base',.1)
    controller.navigation_event(dict(event='nav_done',run_id=record['token'],leg='operation_dock',success=True))
    state.write_text(json.dumps(dict(token='P'*32,status='ready',stamp=time.monotonic())))
    controller.tick();assert controller.stage=='replan'
    command=w.process.commands[-1]
    assert command['_mobile_done']==record['token'] and command['parallel_prepare_token']=='P'*32
    assert all(c['action']=='plan' for c in w.process.commands)
    moves=[];w.launch_motion=lambda filename,**kw:moves.append(filename)
    event=plan(w);event.update(requires_reset=False)
    w.receive_event(event);qt.processEvents()
    assert moves==[event['filename']] and not controller.active
    w.parallel_prepare=None


def test_stop_during_base_releases_ui_after_receipt(window):
    w=window;c=w.mobile_reach;c.stage='moving';w.busy=True
    w.stop_motion();assert (w.run/'stop').exists()
    c.navigation_event(dict(event='nav_done',leg='operation_dock',success=True))
    assert not w.busy and not w.cancel_pending and not c.active


def test_visual_failure_during_manual_compensation_stops_base(window):
    w=window;w.mobile_reach.stage='moving';w.finished()
    assert (w.run/'stop').exists() and not w.mobile_reach.active


def test_replan_failure_stops_prepared_session_and_never_runs_old_plan(window):
    w=window;w.mobile_reach.stage='replan';w.busy=True
    w.mobile_reach.worker_event(dict(event='error',text='当前目标深度缺失'))
    assert (w.run/'stop').exists() and not w.mobile_reach.active and not w.busy
    assert w.pending_plan is None


def test_target_reassociation_skips_bad_depth_candidate(monkeypatch):
    from autolife_shelf_grasp_306 import bottle
    def points(frame,mask,T,cfg,**kwargs):
        if mask=='bad':raise ValueError('no depth')
        return np.tile([.6,0,.7],(50,1)),None,None
    monkeypatch.setattr(bottle,'object_points',points)
    chosen=dict(id=3,feature=[1,0],product=None,name='纸巾')
    fresh=[dict(mask='bad',feature=[1,0]),dict(mask='good',feature=[1,0])]
    value,diag=m.select_after_move(chosen,fresh,{},np.eye(4),{},dict(target_base=[.7,0,.7],base_anchor=[0,0,0]),[.1,0,0])
    assert value['mask']=='good' and value['id']==3 and diag['position_error_m']<1e-8


def test_place_reachable_preserves_original_context_and_no_base(monkeypatch):
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(np.zeros(3),1.))
    context=object()
    p=NS(cfg={'arm':'right'},context=context,configure_source=lambda q:None,fk=NS(fk=lambda *a:np.eye(4)),
        placement_grasp_rotation=lambda g:np.eye(3),solve=lambda *a:None)
    g=dict(product=dict(height_m=.1,diameter_m=.05))
    result=m.place_reposition(p,{},[0,0,0],g,np.array([[.5,0,.6]]),{},dict(place_clearance_m=.01),[0,0,0],distant(),lambda:False,lambda s:None)
    assert result is None and p.context is context


def test_arm_specific_calibration_controls_post_move_association(monkeypatch):
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(np.array([0,0,.6]),.7))
    monkeypatch.setattr(m,'collision_boxes',lambda *a:[box()])
    p=NS(fk=NS(fk=lambda *a:np.eye(4)),plan=lambda *a:dict(checked_states=10))
    pool=NS(fk=NS(fk=lambda *a:np.eye(4)),arm_config=lambda *a:({},[0,0,0]),native_planner=lambda *a:p)
    cfg=dict(auto_arm_selection=False,arm='right',up_torso=[0,0,1])
    result=m.grasp_reposition(pool,{},[.8,0,.6],dict(height_m=.1),cfg,[0,0,0],distant(),lambda:False,lambda s:None,
        observation=lambda *a:([.72,0,.6],dict(height_m=.1)))
    assert np.allclose(result['target_base'],[.72,0,.6])


def test_explicit_new_preview_clears_finished_task_stop(window):
    w=window;scene(w);(w.run/'stop').touch()
    w.send('plan',id=1)
    assert not (w.run/'stop').exists() and w.process.commands[-1]['action']=='plan'
    w.stop_motion();assert (w.run/'stop').exists()  # 新请求仍可正常取消。


@pytest.mark.parametrize('field,value',[('busy',True),('cancel_pending',True),('motion',object()),('parallel_prepare',object())])
def test_new_preview_never_clears_stop_of_live_executor(window,field,value):
    w=window;scene(w);(w.run/'stop').touch();old=getattr(w,field,None);setattr(w,field,value)
    count=len(w.process.commands);w.send('plan',id=1)
    assert (w.run/'stop').exists() and len(w.process.commands)==count
    setattr(w,field,old)


def test_compensation_resume_cannot_clear_operator_stop(window):
    w=window;scene(w);(w.run/'stop').touch();w.send('plan',id=1,_mobile_done='a'*32)
    assert (w.run/'stop').exists() and not w.process.commands


@pytest.mark.parametrize('point,expected',[([.6,0,0],0.),([.75,0,0],.06),([.8,0,0],.11),([1.,0,0],None),([0,.8,0],.11),([-.8,0,0],.11)])
def test_workspace_distance_is_bounded_geometric_calculation(point,expected):
    delta=m.workspace_translation(point,[0,0,0],.7,np.eye(3))
    result=None if delta is None else float(np.linalg.norm(delta))
    assert result is None if expected is None else result==pytest.approx(expected,abs=.01001)


def test_urdf_envelope_sums_links_and_tcp_without_ik():
    def transform(x):
        t=np.eye(4);t[0,3]=x;return t
    fk=NS(joints={'Link_Right_Wrist_Lower_to_Gripper':('a','r','elbow',transform(.3)),
        'elbow':('b','r','shoulder',transform(.3)),
        'shoulder':('c','r','Link_Waist',transform(.1))},fk=lambda *a:transform(.1))
    from autolife_handeye_calibration_306.core import TORSO
    fk.joints['shoulder']=('c','r',TORSO,transform(.1))
    shoulder,radius=m.arm_envelope(fk,{},[.1,0,0],'right')
    assert np.allclose(shoulder,[.1,0,0]) and radius==pytest.approx(.7)


@pytest.fixture(autouse=True)
def fixed_waist_for_direction_tests(monkeypatch):
    monkeypatch.setattr(m,"waist_allowance",lambda *a:0.)
