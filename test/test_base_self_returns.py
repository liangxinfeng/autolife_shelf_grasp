import math
import numpy as np
import pytest
from autolife_shelf_grasp_306.base_sequence import scan_points,clearance


def observation(point,front):
    x,y=point;lx,ly=(x-.2634,y) if front else (x+.215,-y)
    ranges=np.full(360,np.inf);ranges[0]=math.hypot(lx,ly)
    return scan_points(ranges,math.atan2(ly,lx),math.pi/180,.01,25,front)


@pytest.mark.parametrize('front',[True,False])
@pytest.mark.parametrize('point',[[-.233,-.097],[-.232,.124],[-.228,-.085]])
def test_recorded_chassis_returns_removed(point,front):
    assert len(observation(point,front))==0


@pytest.mark.parametrize('front',[True,False])
@pytest.mark.parametrize('point',[[-.27,-.10],[.31,.05],[0,.18],[0,-.18],[.35,0]])
def test_external_returns_not_deleted(point,front):
    out=observation(point,front)
    assert len(out)==1 and np.allclose(out[0],point)


def test_external_front_object_still_blocks_motion():
    assert not clearance(observation([.35,0],True),[.12,0,0])
