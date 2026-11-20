import numpy as np
import pytest
from test_selected_arm_budget import make_pool
from autolife_shelf_grasp_306 import planning_control as pc
from autolife_shelf_grasp_306.worker import plan_before_docking


@pytest.mark.parametrize('chosen,rows,limit',[
    ('left',[dict(arm='left',status='key_poses_passed',yaw_offset_deg=0.)],8.),
    ('right',[dict(arm='left',status='outside_workspace'),dict(arm='right',status='key_poses_passed',yaw_offset_deg=0.)],8.),
    ('right',[dict(arm='left',status='unresolved',key_pose_miss=True),dict(arm='right',status='unresolved',key_pose_budget_used=True)],4.5),
    ('left',[dict(arm=side,status='unresolved',key_pose_miss=True) for side in ('left','right')],3.)])
def test_interactive_uses_one_full_arm_and_a_shared_deadline(monkeypatch,chosen,rows,limit):
    pool,now,calls,contexts=make_pool(monkeypatch,rows)
    events=[]
    result=pool.plan({},np.zeros(3),{},dict(stationary_plan_budget_sec=25.),interactive=True,on_reach=events.append)
    assert calls==[chosen] and result[0]['arm']==chosen
    assert contexts[0].deadline==limit and events[0]['stationary_budget_seconds']==limit


@pytest.mark.parametrize('status',['reposition_review','key_poses_passed','unresolved'])
def test_dock_on_timeout_requires_prior_reach_evidence(status):
    def stationary(on_reach):
        on_reach(dict(status=status))
        raise pc.PlanningTimeout('budget')
    calls=[]
    def dock():calls.append(1);return 'validated dock'
    if status=='reposition_review':
        assert plan_before_docking(stationary,dock,on_reach=lambda _:None)=='validated dock'
        assert calls==[1]
    else:
        with pytest.raises(pc.PlanningTimeout):plan_before_docking(stationary,dock,on_reach=lambda _:None)
        assert calls==[]
