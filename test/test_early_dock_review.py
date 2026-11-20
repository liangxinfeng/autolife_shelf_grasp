import pytest
from autolife_shelf_grasp_306.auto_grasp import StationaryPathsExhausted
from autolife_shelf_grasp_306.mobile_reach import RepositionRequired
from autolife_shelf_grasp_306.worker import plan_before_docking


@pytest.mark.parametrize('outcome',['planned','exhausted','sensor_failure'])
@pytest.mark.parametrize('alternate',[True,False])
def test_uncertain_screen_never_displaces_a_valid_stationary_path(outcome,alternate):
    events=[]
    def stationary(on_reach):
        on_reach(dict(status='reposition_review'))
        events.append('full')
        if outcome=='exhausted':raise StationaryPathsExhausted('no local path')
        if outcome=='sensor_failure':raise ValueError('sensor failure')
        return 'stationary plan'
    def dock():
        events.append('review')
        if alternate:raise RepositionRequired({'goal_pose':[0,0,0]})
        raise ValueError('no clear alternative')
    def run():return plan_before_docking(stationary,dock,on_reach=lambda r:events.append('ui'))
    if outcome=='planned':assert run()=='stationary plan'
    elif outcome=='exhausted' and alternate:
        with pytest.raises(RepositionRequired):run()
    else:
        with pytest.raises(ValueError,match='sensor failure' if outcome=='sensor_failure' else 'no clear alternative'):run()
    assert events==(['ui','full','review'] if outcome=='exhausted' else ['ui','full'])
