"""Numerical policy tests; no ROS node or hardware commands."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from autolife_shelf_grasp_306.planning import (Planner,IKFailure,CollisionFailure,
    approach_candidates,reference_position,distance_to_polyline)


def candidate_planner():
    # These tests isolate candidate ordering; full geometry has separate tests.
    p=Planner.__new__(Planner);p.cfg={}
    p.guard=SimpleNamespace(begin_fixed_neck=lambda source:None)
    return p


def coupled_fixture():
    # Analytic kinematic fixture: translating X necessarily changes roll.
    # Proves soft orientation works without accepting position error/jumps.
    p=Planner.__new__(Planner);p.cfg={'orientation_relax_deg':30.}
    p.names=list('abcdefg');p.bounds=np.array([[-1.]*7,[1.]*7])
    def pose(q,tcp):
        x=np.deg2rad([q[n] for n in p.names])
        return x[:3],Rotation.from_rotvec([5*x[0],0,0]).as_matrix()
    p.pose=pose
    return p,{n:0. for n in p.names}


def test_flexible_solves_position_strict_cannot():
    p,q=coupled_fixture();goal=np.array([.05,0,0])
    with pytest.raises(IKFailure):p.solve(q,np.zeros(3),goal,np.eye(3))
    result=p.solve(q,np.zeros(3),goal,np.eye(3),True)
    position,rotation=p.pose(result,None)
    assert np.linalg.norm(position-goal)<.004
    assert 3.<np.rad2deg(Rotation.from_matrix(rotation).magnitude())<30.
    assert max(abs(result[n]-q[n]) for n in q)<=8.00001


def test_flexible_not_unbounded():
    p,q=coupled_fixture()
    with pytest.raises(IKFailure):p.solve(q,None,[.13,0,0],np.eye(3),True)
    with pytest.raises(IKFailure):p.solve(q,None,[.05,0,0],np.eye(3),True,limit_deg=5.)


def test_route_retries_with_full_checks():
    p=candidate_planner();calls=[]
    def route(*args):
        calls.append(args[4])
        if not args[4]:raise IKFailure('strict unreachable')
        return {'orientation_mode':'flexible'}
    p._route=route
    assert p.plan({},None,None,{})['orientation_mode']=='flexible'
    assert calls==[False]*8+[True]
    attempted=[]
    def new_approach(*args):
        attempted.append(args[6])
        if args[6]==0:raise CollisionFailure('self collision')
        return {'orientation_mode':'horizontal'}
    p._route=new_approach
    assert p.plan({},None,None,{})['orientation_mode']=='horizontal'
    assert attempted==[0]*16+[-5]
    def collision(*args):raise ValueError('self collision')
    p._route=collision
    with pytest.raises(ValueError,match='self collision'):p.plan({},None,None,{})


def test_small_angles_and_forward_alternatives_first():
    candidates=list(approach_candidates())
    angles=[abs(c[0]) for c in candidates]
    assert angles==sorted(angles)
    assert sorted(set(angles))==list(range(0,61,5))
    assert {c[2] for c in candidates if c[0]==0}=={'direct','elbow_positive','elbow_negative','outward_4','outward_8',
        'joint_direct','joint_elbow_positive','joint_elbow_negative'}
    p=candidate_planner();attempted=[]
    def route(*args):
        attempted.append((args[6],args[7]))
        if args[7]!='outward_4':raise CollisionFailure('direct blocked')
        return {'approach_yaw_deg':args[6]}
    p._route=route
    assert p.plan({},None,None,{})['approach_yaw_deg']==0
    assert all(yaw==0 for yaw,profile in attempted)


def test_detour_preserves_endpoints_and_checks_curve_not_shortcut():
    a=np.array([0.,0.,0.]);b=np.array([.2,0.,0.]);offset=np.array([0.,-.08,.02])
    points=np.array([reference_position(a,b,t,offset) for t in np.linspace(0,1,80)])
    assert np.allclose(points[0],a) and np.allclose(points[-1],b)
    assert distance_to_polyline(reference_position(a,b,.5,offset),points)<.0001
    assert distance_to_polyline((a+b)/2,points)>.01


def test_posture_bias_retains_pose_and_step_limits():
    p,q=coupled_fixture();target=np.zeros(7);target[4]=.4
    result=p.solve(q,None,[0,0,0],np.eye(3),posture_target=target)
    assert 0<result['e']<=8.00001
    position,rotation=p.pose(result,None)
    assert np.linalg.norm(position)<.004


def test_late_failure_does_not_repeat_pregrasp_only_variants():
    p=candidate_planner();calls=[]
    def route(*args):
        calls.append((args[6],args[4],args[7]))
        if args[6]==0:
            p._candidate_stage=3
            raise IKFailure('carry endpoint infeasible')
        return {}
    p._route=route
    p.plan({},None,None,{})
    assert calls==[(0.,flex,profile) for flex in [False,True] for profile in
                  ['direct','joint_direct','joint_elbow_positive','joint_elbow_negative']]+[(-5.,False,'direct')]


def test_late_failure_keeps_different_pregrasp_for_final_stationary_pass():
    p=candidate_planner();calls=[]
    p.candidates=lambda yaws:iter([(0.,False,'direct'),(0.,False,'outward_4'),(10.,False,'direct')])
    def route(*args):
        calls.append((args[6],args[7]))
        if args[7]=='outward_4':return {'usable':True}
        p._candidate_stage=3;raise CollisionFailure('blocked carry')
    p._route=route
    assert p.plan({},None,None,{})['usable']
    assert calls==[(0.,'direct'),(10.,'direct'),(0.,'outward_4')]


def test_native_interpolation_rejection_tries_next_stationary_approach(monkeypatch):
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner
    from autolife_shelf_grasp_306 import vendor_grasp_trajectory
    from autolife_shelf_grasp_306.planning_control import PlanningControl
    p=VendorGraspPlanner.__new__(VendorGraspPlanner)
    p.cfg={'arm':'left','up_torso':[0.,0.,1.]};p.guard=SimpleNamespace(begin_fixed_neck=lambda q:None)
    p.configure_source=lambda q:None;p.context=PlanningControl();p.close=lambda:None
    p.candidates=lambda yaws:iter([(0.,False,'direct'),(25.,False,'direct')])
    p.names=['j'];p.bounds=np.zeros((2,2));p.anchor=np.eye(4)
    p.pose=lambda q,tcp:(np.zeros(3),np.eye(3));p.check=lambda *args:None
    product={'height_m':.18,'diameter_m':.05}
    def route(self,source,tcp,center,obj,flexible,started,yaw,profile):
        return dict(segments=[{'phase':'grasp','waypoints':[source]}],tcp=tcp,product=obj,yaw=yaw)
    monkeypatch.setattr(Planner,'_route',route)
    attempts=[]
    def native(plan,cfg):
        attempts.append(plan['yaw'])
        if plan['yaw']==0.:raise CollisionFailure('native interpolation collision')
        return [dict(waypoints=[{'j':0.},{'j':.05}],carrying=False)]
    monkeypatch.setattr(vendor_grasp_trajectory,'build_blocks',native)
    result=p.plan({'j':0.},np.zeros(3),np.zeros(3),product)
    assert attempts==[0.,25.] and result['yaw']==25. and result['native_checked_states']>=2


def test_eccentric_contact_keeps_object_centre_through_native_interpolation(monkeypatch):
    from autolife_shelf_grasp_306 import local_contact,low_object_grasp,vendor_grasp_trajectory
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner
    from autolife_shelf_grasp_306.planning_control import PlanningControl
    p=VendorGraspPlanner.__new__(VendorGraspPlanner)
    p.cfg={'arm':'left','up_torso':[0.,0.,1.],'approach_yaw_deg':0.,'T_wrist_tool_quat_xyzw':[0.,0.,0.,1.]}
    p.names=['j'];p.bounds=np.zeros((2,2));p.anchor=np.eye(4);p.context=PlanningControl()
    p.bounded_seed_recoveries=0;p.candidates=lambda yaws:iter((yaw,False,'direct') for yaw in yaws)
    physical=np.array([.13,-.008,.02]);contact=np.array([-.023,.010,0.]);centre=np.array([.4,.1,.2])
    rotation=Rotation.from_euler('z',35.,degrees=True).as_matrix()
    offset=rotation.T@(-contact)
    virtual=physical+offset
    product={'height_m':.15,'diameter_m':.08,'depth_m':.07}
    source={'j':0.}
    monkeypatch.setattr(local_contact,'candidates',lambda *args:[dict(yaw=0.,contact=contact,required_opening_m=.05)])
    monkeypatch.setattr(low_object_grasp,'orientation',lambda *a,**kw:(rotation,))
    def route(self,q,tcp,target,obj,*unused):
        np.testing.assert_allclose(tcp,virtual)
        return dict(segments=[{'phase':'grasp','waypoints':[q]}],tcp=tcp,product=obj)
    monkeypatch.setattr(Planner,'_route',route)
    blocks=[dict(carrying=holding,waypoints=[source,{'j':.05}]) for holding in (False,True)]
    monkeypatch.setattr(vendor_grasp_trajectory,'build_blocks',lambda *unused:blocks)
    p.pose=lambda q,tcp:(centre+rotation@(tcp-virtual),rotation)
    samples=[]
    def check(q,tcp,carrying,obj):
        samples.append(carrying)
        pos,R=p.pose(q,tcp)
        np.testing.assert_allclose(pos,centre)
        np.testing.assert_allclose(pos-R@obj['object_center_offset_wrist'],centre+contact)
    p.check=check
    plan=p._local_contact_plan(source,physical,centre,product,0.)
    assert samples==[False,False,True,True] and plan['native_checked_states']==4
    assert plan['grasp_contact_mode']=='front_local'
    np.testing.assert_allclose(plan['physical_tcp'],physical)
    np.testing.assert_allclose(plan['tcp'],virtual)
    np.testing.assert_allclose(plan['contact_center'],centre+contact)


def test_different_joint_branch_precedes_larger_grasp_angle():
    p=candidate_planner();calls=[]
    def route(*args):
        profile=args[7]
        calls.append((args[6],profile));p._candidate_stage=3
        if profile!='joint_direct':raise IKFailure('wrist limit on withdrawal')
        return {'approach_yaw_deg':args[6]}
    p._route=route
    result=p.plan({},None,None,{})
    assert result['approach_yaw_deg']==0.
    assert calls==[(0.,'direct'),(0.,'joint_direct')]


def test_no_tail_retry_for_grasp_failure_or_collision():
    for stage,error in [(1,IKFailure),(3,CollisionFailure)]:
        p=candidate_planner();calls=[]
        def route(*args):
            assert len(args)==8
            calls.append(args[6]);p._candidate_stage=stage
            if args[6]==0:raise error('blocked')
            return {}
        p._route=route;p.plan({},None,None,{})
        assert calls==[0.]*8+[-5.]


def test_withdraw_freedom_preserves_initial_clearance_and_is_bounded():
    from autolife_shelf_grasp_306.planning import withdrawal_freedom
    for d in [-.01,0.,.02,.06]:assert withdrawal_freedom(d)==0.
    assert withdrawal_freedom(.08)==pytest.approx(.5)
    assert withdrawal_freedom(.1)==pytest.approx(1.)
    assert withdrawal_freedom(.2)==1.


def test_retreat_does_not_relax_grasp_or_position_accuracy():
    p,q=coupled_fixture();p.cfg['up_torso']=[1.,0.,0.]
    # This fixture must rotate about UP to translate. Heading can adapt after
    # clearance; strict grasp still rejects the exact same request.
    result=p.solve(q,None,[.05,0,0],np.eye(3),retreat_freedom=1.)
    assert np.linalg.norm(p.pose(result,None)[0]-[.05,0,0])<.004
    assert max(abs(result[n]-q[n]) for n in q)<=8.00001
    with pytest.raises(IKFailure):p.solve(q,None,[.05,0,0],np.eye(3))
    with pytest.raises(IKFailure):p.solve(q,None,[1.,0,0],np.eye(3),retreat_freedom=1.)
    p.cfg['up_torso']=[0.,0.,1.]
    with pytest.raises(IKFailure):p.solve(q,None,[.13,0,0],np.eye(3),retreat_freedom=1.)
