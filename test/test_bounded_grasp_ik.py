"""Offline numerical recovery tests; never create ROS nodes or motor commands."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_fast_planning import arm
from autolife_shelf_grasp_306.bounded_grasp_ik import BoundedPoseResidual, recover_seed
from autolife_shelf_grasp_306.planning import IKFailure
from autolife_shelf_grasp_306.planning_control import PlanningCancelled


def fixture(tmp_path):
    p, fk = arm(tmp_path)
    p.full_pose_indices = list(zip(p.names, p.pose_indices))
    p.anchor = np.eye(4)
    p.anchor[:3, :3] = Rotation.from_euler('xyz', [.13, -.23, .31]).as_matrix()
    p.anchor[:3, 3] = [.1, -.08, .25]
    p.context = SimpleNamespace(checkpoint=lambda: None)
    return p


def actual(p, q, tcp):
    residual = BoundedPoseResidual(p, q, tcp, np.zeros(3), np.eye(3))
    x = np.deg2rad([q[n] for n in p.names])
    residual.fun(x)
    return residual.actual_point.copy(), residual.actual_rotation.copy()


def test_tcp_jacobian_matches_central_differences(tmp_path):
    p = fixture(tmp_path)
    rng = np.random.default_rng(345)
    tcp = np.array([.12, .017, -.032])
    for _ in range(10):
        x = rng.uniform(-.7, .7, len(p.names))
        q = dict(zip(p.names, np.rad2deg(x)))
        point, rotation = actual(p, q, tcp)
        rotation = rotation @ Rotation.from_rotvec([.2, -.1, .15]).as_matrix()
        residual = BoundedPoseResidual(p, q, tcp, point + .01, rotation)
        analytic = residual.jac(x)
        numeric = np.empty_like(analytic)
        for j in range(len(x)):
            delta = np.zeros_like(x); delta[j] = 1e-6
            numeric[:, j] = (residual.fun(x+delta)-residual.fun(x-delta))/2e-6
        assert np.allclose(analytic, numeric, atol=1e-7, rtol=0)


def test_recovery_preserves_bounds_and_exact_tcp(tmp_path):
    p = fixture(tmp_path)
    n = len(p.names)
    source = dict(zip(p.names, np.rad2deg(np.linspace(-.3, .3, n))))
    goal = dict(zip(p.names, np.rad2deg(np.linspace(-.27, .33, n))))
    tcp = np.array([.12, .02, -.03])
    point, rotation = actual(p, goal, tcp)
    seed = np.deg2rad([source[k] for k in p.names])
    lower, upper = np.full(n, -.6), np.full(n, .6)
    original = (lower.copy(), upper.copy(), dict(source))
    result = recover_seed(p, source, tcp, point, rotation, seed, lower, upper)
    residual = BoundedPoseResidual(p, source, tcp, point, rotation).fun(result)
    assert np.linalg.norm(residual[:3]) <= 1e-5
    assert np.linalg.norm(residual[3:]) <= 1e-5
    assert np.all(result >= lower) and np.all(result <= upper)
    assert np.array_equal(lower, original[0]) and np.array_equal(upper, original[1])
    assert source == original[2]


def test_unreachable_target_is_not_clipped_or_accepted(tmp_path):
    p = fixture(tmp_path); n = len(p.names)
    source = dict.fromkeys(p.names, 0.)
    with pytest.raises(IKFailure):
        recover_seed(p, source, np.zeros(3), [10,10,10], np.eye(3),
            np.zeros(n), np.full(n,-.05), np.full(n,.05))


@pytest.mark.parametrize('distance,angle,vendor_reply,requests,accepted', [
    (.002, 1.5, 'seed', 2, True),
    (.002, 1.5, 'no_ik', 2, False),
    (.002, 1.5, 'bad_fk', 2, False),
    (.004, 1.5, 'seed', 1, False),
    (.002, 2.5, 'seed', 1, False),
])
def test_near_boundary_seed_must_pass_vendor_and_final_fk(
        tmp_path, distance, angle, vendor_reply, requests, accepted):
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner
    from autolife_shelf_grasp_306.vendor_ik_process import NoIKSolution

    planner = VendorGraspPlanner.__new__(VendorGraspPlanner)
    vars(planner).update(vars(fixture(tmp_path)))
    planner.cfg = {'arm': 'right'}
    planner.context.set_stage = lambda *args, **kwargs: None
    n = len(planner.names)
    planner.bounds = (np.full(n, -1e-6), np.full(n, 1e-6))
    source = dict.fromkeys(planner.names, 0.)
    tcp = np.array([.12, .02, -.03])
    start, orientation = planner.pose(source, tcp)
    target = start + [distance, 0., 0.]
    rotation = orientation @ Rotation.from_euler('z', angle, degrees=True).as_matrix()
    calls = []

    def request(kind, **inputs):
        calls.append(inputs)
        if len(calls) == 1 or vendor_reply == 'no_ik':
            raise NoIKSolution('no vendor solution')
        if vendor_reply == 'bad_fk':
            planner.pose = lambda q, tcp: (target + [.006, 0., 0.], rotation)
        return inputs['seed']

    planner.solver = SimpleNamespace(request=request)
    if accepted:
        result = planner.solve(source, tcp, target, rotation)
        pose, final_rotation = planner.pose(result, tcp)
        assert .001 < np.linalg.norm(pose - target) < .003
        assert 1. < Rotation.from_matrix(rotation.T @ final_rotation).magnitude()*180/np.pi < 2.
    else:
        with pytest.raises(IKFailure):
            planner.solve(source, tcp, target, rotation)
    assert len(calls) == requests
    if requests == 2:
        assert calls[0]['pose'] == calls[1]['pose']
        assert calls[0]['lower'] == calls[1]['lower'] and calls[0]['upper'] == calls[1]['upper']


def test_cancel_interrupts_optimization(tmp_path):
    p = fixture(tmp_path); n = len(p.names)
    def cancel(): raise PlanningCancelled('cancelled')
    p.context.checkpoint = cancel
    with pytest.raises(PlanningCancelled):
        recover_seed(p, dict.fromkeys(p.names,0.), np.zeros(3), [0,0,0], np.eye(3),
            np.zeros(n), np.full(n,-1.), np.full(n,1.))


def test_ik_cache_keeps_cancel_and_does_not_alias_source():
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner
    from types import SimpleNamespace
    p=VendorGraspPlanner.__new__(VendorGraspPlanner);calls=[];checks=[]
    p.context=SimpleNamespace(checkpoint=lambda:checks.append(True))
    def solve(source,*args):calls.append(source.copy());return dict(source,a=1.)
    p._solve_uncached=solve
    a=p.solve({'a':0.},np.zeros(3),np.ones(3),np.eye(3));a['a']=99
    assert p.solve({'a':0.},np.zeros(3),np.ones(3),np.eye(3))['a']==1.
    assert len(calls)==1 and len(checks)==2
    p.solve({'a':.1},np.zeros(3),np.ones(3),np.eye(3));assert len(calls)==2
    def cancel():raise PlanningCancelled('cancel')
    p.context.checkpoint=cancel
    with pytest.raises(PlanningCancelled):p.solve({'a':0.},np.zeros(3),np.ones(3),np.eye(3))


def test_reverse_contact_seed_follows_same_approach_and_checks_each_step():
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner
    p=VendorGraspPlanner.__new__(VendorGraspPlanner);targets=[];checked=[]
    def solve(source,tcp,point,rotation):
        targets.append(np.asarray(point).copy());return dict(source,x=float(point[0]))
    p.solve=solve;p.check=lambda q,tcp,carrying,product:checked.append((q.copy(),carrying))
    q=p.pregrasp_from_contact({'x':0.},np.zeros(3),np.array([.5,0,0]),np.array([.4,0,0]),np.eye(3),{})
    assert q['x']==pytest.approx(.4) and np.allclose(targets[0],[.5,0,0])
    assert len(checked)==len(targets)-1 and all(not carrying for _,carrying in checked)
    assert max(np.linalg.norm(b-a) for a,b in zip(targets,targets[1:]))<=.008
