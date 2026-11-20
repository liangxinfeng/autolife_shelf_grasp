from types import SimpleNamespace
import numpy as np
import pytest
from autolife_shelf_grasp_306 import low_object_grasp as grasp
from autolife_shelf_grasp_306.planning_control import PlanningControl,PlanningTimeout


@pytest.mark.parametrize('source,expected',[('local_rgbd_plane',15.),('object_bottom_estimate',0.)])
def test_midbody_pitch_is_available_only_with_observed_support(monkeypatch,source,expected):
    planner=SimpleNamespace(cfg=dict(up_torso=[0.,0.,1.],T_wrist_tool_quat_xyzw=[0.,0.,0.,1.]),
        pose=lambda *_:(np.zeros(3),np.eye(3)),
        grasp_attitude_feasible=lambda q,t,c,R,p:R[2,0]<-.15)
    monkeypatch.setattr(grasp,'gripper_vertices',lambda p:np.array([[.01,0.,-.005]]))
    product=dict(height_m=.16,geometry='solid',low_object_support=dict(normal=[0,0,1],offset=0.,source=source))
    _,axis,pitch,_,_=grasp.orientation(planner,{},np.zeros(3),np.array([.5,0.,.08]),product,0.)
    assert pitch==expected and axis[2]==pytest.approx(-np.sin(np.deg2rad(expected)))


def test_finishing_contact_uses_original_deadline_only():
    now=[0.];ctx=PlanningControl(clock=lambda:now[0],timeout_sec=8.)
    ctx.finish_budget=ctx.timeout_sec;ctx.limit_total(3.)
    now[0]=2.9;ctx.finish_validated_approach()
    assert ctx.deadline==8. and ctx.finish_budget is None
    now[0]=8.
    with pytest.raises(PlanningTimeout):ctx.checkpoint()
