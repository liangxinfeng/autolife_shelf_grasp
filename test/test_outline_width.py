import numpy as np
import pytest
from autolife_shelf_grasp_306.object_extent import projected_width


def test_diagonal_jaw_width_uses_observed_outline_and_preserves_carried_box():
    angle=np.arange(32)*np.pi/16
    points=np.stack([.035*np.cos(angle),.035*np.sin(angle),np.zeros(32)],axis=1)
    p=dict(measured_footprint_torso=points.tolist(),bbox_axes_torso=np.eye(3).tolist(),
           bbox_extents_m=[.074,.074,.10],grasp_outline_measured=True)
    direction=np.array([1.,1.,0.])/np.sqrt(2)
    assert projected_width(p,direction)==pytest.approx(.074)
    assert projected_width(p,[0,0,1])==pytest.approx(.104)
    assert projected_width(dict(p,grasp_outline_measured=False),direction)>.10
    assert p['bbox_extents_m']==[.074,.074,.10]
