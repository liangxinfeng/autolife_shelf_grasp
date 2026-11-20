from types import SimpleNamespace as NS
import numpy as np
import pytest
from test_selected_arm_budget import make_pool

def test_only_alternative_with_key_pose_miss_keeps_a_quarter_of_budget(monkeypatch):
    rows=[dict(arm='left',status='unresolved',key_pose_budget_used=True),dict(arm='right',status='unresolved',key_pose_miss=True)]
    pool,now,calls,contexts=make_pool(monkeypatch,rows)
    def planner(cfg,ctx):
        calls.append(cfg['arm']);contexts.append(ctx)
        def plan(*args):
            now[0]=17.;ctx.checkpoint()
            return dict(segments=['complete path'])
        return NS(plan=plan,candidate_failures=[])
    pool.native_planner=planner
    result=pool.plan({},np.zeros(3),{},dict(stationary_plan_budget_sec=25.))
    assert result[0]['arm']=='left' and calls==['left']
    assert contexts[0].deadline==pytest.approx(.2+.75*24.8)
