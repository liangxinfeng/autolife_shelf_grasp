import numpy as np
import pytest
from test_selected_arm_budget import make_pool
from autolife_shelf_grasp_306 import reach_precheck


def test_cold_model_leaves_one_local_attempt_inside_hard_limit(monkeypatch):
    rows=[dict(arm=s,status='unresolved',key_pose_miss=True) for s in ('left','right')]
    pool,now,calls,contexts=make_pool(monkeypatch,rows,finish_time=4.3)
    def screen(*args):
        now[0]=3.15
        return rows
    monkeypatch.setattr(reach_precheck,'screen',screen)
    result=pool.plan({},np.zeros(3),{},{'stationary_plan_budget_sec':25.},interactive=True)
    assert calls==['left'] and result[0]['arm']=='left'
    assert contexts[0].deadline==pytest.approx(4.65)
