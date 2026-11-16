"""全向可达判断、斜向扫掠、里程计执行与现有界面的无动作回归。"""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import mobile_reach as m
from autolife_shelf_grasp_306.base_sequence import Controller,clearance,normalize,ObstacleHold
from autolife_shelf_grasp_306.chassis import InputWatchdog


def request(delta):
    return m.proposal('grasp',delta,{},[0,0,0],[.7,.5,.6],'right',0,.01)


def box(center=(0,0,.5)):
    return dict(rotation=np.eye(3).tolist(),translation=list(center),low=[-.1]*3,high=[.1]*3)


def background():return np.random.default_rng(12).uniform([2.,2.,.2],[3.,3.,2.],(300,3))


@pytest.mark.parametrize('angle',range(0,360,30))
def test_workspace_shortest_planar_shift_all_directions(angle):
    u=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
    point=np.r_[u*.78,.15]
    delta=m.workspace_translation(point,[0,0,0],.7,np.eye(3))
    assert delta is not None and np.linalg.norm(delta)<=.15
    assert np.allclose(delta/np.linalg.norm(delta),u)
    after=point-np.r_[delta,0]
    assert np.linalg.norm(after)<=.69+1e-8
    # 距离以厘米向上取整，额外位移不足1厘米。
    exact=.78-math.sqrt(.69**2-.15**2)
    assert 0<=np.linalg.norm(delta)-exact<.0100001


def test_torso_pitch_does_not_turn_horizontal_compensation_into_height():
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('zyx',[30,20,0],degrees=True).as_matrix()
    shoulder=np.array([0,.2,.05]);world_relative=np.array([.7,-.4,.08])
    d=m.workspace_translation(shoulder+R.T@world_relative,shoulder,.7,R)
    assert d[0]>0 and d[1]<0
    assert np.linalg.norm(world_relative-np.r_[d,0])<=.69+1e-8


@pytest.mark.parametrize('point',[[0,0,.75],[.8,0,.7],[1.3,0,0]])
def test_unreachable_height_and_excessive_distance_do_not_move(point):
    assert m.workspace_translation(point,[0,0,0],.7,np.eye(3)) is None


@pytest.mark.parametrize('delta',[[.08,.06],[-.08,.06],[.08,-.06],[-.08,-.06],[0,.1],[0,-.1]])
def test_vector_protocol_single_move_and_correct_ui_direction(delta):
    r=request(delta);c=m.validate_request(r,r['token'])
    assert len(c['moves'])==1
    assert c['moves'][0]==dict(kind='translate',x_cm=delta[0]*100,y_cm=delta[1]*100,speed=12.)
    assert normalize(c)==c
    label=m.direction_text(r)
    if delta[0]:assert ('前移' if delta[0]>0 else '后移') in label
    if delta[1]:assert ('左移' if delta[1]>0 else '右移') in label


@pytest.mark.parametrize('delta',[[.15,.15],[0,0],[float('nan'),.1],[.1,.1,.1],[.001,0]])
def test_total_distance_not_per_axis_distance(delta):
    r=dict(token='a'*32,translation_m=delta,created=100,decision_stage='recognition_workspace',reach_checked=True,sweep_checked=True)
    with pytest.raises(ValueError):m.validate_request(r,'a'*32,100)


@pytest.mark.parametrize('degrees',range(0,360,15))
def test_one_centimeter_boundary_roundoff_does_not_reject_diagonals(degrees):
    a=math.radians(degrees);r=request([.01*math.cos(a),.01*math.sin(a)])
    c=m.validate_request(r,r['token']);assert normalize(c)==c


@pytest.mark.parametrize('delta',[[.1,.1],[-.1,.1],[.1,-.1],[-.1,-.1],[0,.15],[0,-.15]])
def test_swept_body_and_carried_object_are_checked_for_lateral_motion(delta):
    u=np.array(delta)/np.linalg.norm(delta)
    # 当前包围盒之外，但落在平移扫掠中；不能只检查前向X。
    corner=np.sign(u)*.099+np.array(delta)*.65
    p=np.vstack([background(),[*corner,.5]])
    assert not m.sweep_clear(p,[box()],delta)
    assert m.sweep_clear(background(),[box()],delta)
    p=np.vstack([background(),[*corner,.9]])
    assert m.sweep_clear(p,[box()],delta)
    assert not m.sweep_clear(p,[box()],delta,box((0,0,.9)))


@pytest.mark.parametrize('delta',[[.08,.06],[-.08,.06],[.08,-.06],[-.08,-.06],[0,.1],[0,-.1]])
def test_guard_tracks_diagonal_corridor_in_robot_frame(delta):
    r=request(delta);here=np.r_[np.array(delta)*.4,0]
    remaining=m.corridor_remaining(r,here,0)
    assert np.allclose(remaining,np.array(delta)*.6)
    with pytest.raises(ValueError):m.corridor_remaining(r,here,0,starting=True)
    u=np.array(delta)/np.linalg.norm(delta);cross=np.array([-u[1],u[0]])
    with pytest.raises(ValueError):m.corridor_remaining(r,here[:2]+cross*.021,0)
    with pytest.raises(ValueError):m.corridor_remaining(r,[0,0,0],.03)
    assert np.allclose(m.corridor_remaining(r,delta,0),0)


@pytest.mark.parametrize('delta',[[.1,0],[-.1,0],[0,.1],[0,-.1],[.08,.06],[-.08,.06],[.08,-.06],[-.08,-.06],[.006,.008]])
@pytest.mark.parametrize('yaw',[0.,1.2,-2.4])
def test_odometry_controller_all_directions_simultaneous_and_bounded(delta,yaw):
    r=request(delta);config=m.validate_request(r,r['token']);ctrl=Controller(config)
    pose=np.array([2.,-1.,yaw]);origin=pose.copy();v=np.zeros(3);old=np.zeros(3);dt=.02;commands=[]
    watch=InputWatchdog(fixed_sequence=True)
    for step in range(700):
        now=10+step*dt;v=np.array(ctrl.update(pose,v,now));commands.append(v.copy())
        watch.update(dict(velocity=v.tolist(),stamp=now,sequence=step),now)
        assert np.linalg.norm(v[:2])<=.120001
        if np.linalg.norm(v[:2])>0:assert np.linalg.norm(v[:2]-old[:2])<=.25*dt+1e-8
        if delta[0] and delta[1] and np.linalg.norm(v[:2])>0:
            assert v[0]*delta[0]>0 and v[1]*delta[1]>0  # 同一控制周期两轴都有输出。
            assert v[0]/v[1]==pytest.approx(delta[0]/delta[1],abs=1e-7)
        c,s=math.cos(pose[2]),math.sin(pose[2]);pose[:2]+=np.array([[c,-s],[s,c]])@v[:2]*dt
        pose[2]+=v[2]*dt;old=v.copy()
        if ctrl.done:break
    assert ctrl.done and ctrl.index==1
    c,s=math.cos(yaw),math.sin(yaw);actual=np.array([[c,s],[-s,c]])@(pose[:2]-origin[:2])
    assert np.linalg.norm(actual-delta)<=min(np.linalg.norm(delta)*.2,.008)+1e-7
    assert pose[2]==pytest.approx(yaw)


def test_pause_holds_and_resumes_same_vector_without_timeout():
    ctrl=Controller(m.validate_request(r:=request([.08,.06]),r['token']))
    ctrl.update([0,0,0],[0,0,0],1)
    for i in range(1,1001):assert ctrl.update([0,0,0],[0,0,0],1+i*.02,paused=True)==[0.,0.,0.]
    v=ctrl.update([0,0,0],[0,0,0],21.02)
    assert v[0]>0 and v[1]>0


@pytest.mark.parametrize('delta',[[.08,.06],[-.08,.06],[.08,-.06],[-.08,-.06],[0,.1],[0,-.1]])
def test_external_laser_obstacle_blocks_each_motion_direction(delta):
    u=np.array(delta)/np.linalg.norm(delta)
    p=(.34+.03)*u/max(abs(u))
    v=np.r_[u*.12,0]
    assert not clearance([p],v)
    hold=ObstacleHold();assert hold.update([p],v,[0,0,0],1)==[0,0,0]
    assert hold.active


def test_grasp_left_and_diagonal_out_of_range_use_same_early_precheck(monkeypatch):
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(np.array([0,0,.6]),.7))
    monkeypatch.setattr(m,'collision_boxes',lambda *a:[box()])
    planner=NS(plan=lambda *a:pytest.fail('补偿前不走完整路径规划'))
    pool=NS(fk=NS(fk=lambda *a:np.eye(4)),arm_config=lambda *a:({},[0,0,0]),native_planner=lambda *a:planner)
    for xy in ([0,.78],[.55,.55],[-.55,.55]):
        value=m.grasp_reposition(pool,{},[*xy,.6],{},dict(auto_arm_selection=False,arm='left'),[0,0,0],background(),lambda:False,lambda s:None)
        assert value is not None and value['translation_m'][1]>0 and value['arm']=='left'
        assert not value['path_checked'] and value['checked_states']==0


def test_place_lateral_candidate_has_vector_and_held_object(monkeypatch):
    monkeypatch.setattr(m,'arm_envelope',lambda *a:(np.array([0,0,.6]),.7))
    monkeypatch.setattr(m,'collision_boxes',lambda *a:[box()])
    from autolife_handeye_calibration_306 import verification_core
    monkeypatch.setattr(verification_core,'wrist_frame',lambda cfg:'wrist')
    planner=NS(cfg={'arm':'right'},fk=NS(fk=lambda *a:np.eye(4)))
    result=m.place_reposition(planner,{},[0,0,0],dict(product=dict(height_m=.1,diameter_m=.05)),
        np.array([[0,-.78,.54]]),{},dict(place_clearance_m=.01),[0,0,0],background(),lambda:False,lambda s:None)
    assert result and result['translation_m'][1]<0 and result['held_box']


@pytest.fixture(autouse=True)
def fixed_waist_for_direction_tests(monkeypatch):
    monkeypatch.setattr(m,"waist_allowance",lambda *a:0.)
