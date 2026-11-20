import numpy as np
import pytest
from autolife_shelf_grasp_306.object_geometry import measured_relief
from autolife_shelf_grasp_306.unregistered import visible_surface
from autolife_shelf_grasp_306 import bottle

def test_noisy_tilted_background_is_not_object_relief():
    rng=np.random.default_rng(171)
    x,z=rng.uniform(-.04,.04,(2,4000))
    y=.65+.25*x-.37*z+rng.normal(0,.002,len(x))
    assert not measured_relief(np.c_[x,y,z])

def test_measured_opposing_sides_without_visible_top():
    y,z=np.meshgrid(np.linspace(-.02,.02,34),np.linspace(0,.064,40))
    sides=np.concatenate([np.c_[np.full(y.size,x),y.ravel(),z.ravel()] for x in (-.014,.014)])
    p=np.vstack([sides,np.tile([0.,0.,.10],(50,1))])
    evidence=visible_surface(p,p.min(axis=0),p.max(axis=0),np.eye(3),np.array([0,.6,0]))
    assert evidence['surface_evidence']=='opposed_side_patches'

def test_normal_estimate_is_independent_of_circle_solver(monkeypatch):
    from test_bottle import cropped_frame
    f,m,cfg=cropped_frame(monkeypatch)
    def forbidden(*a,**k):pytest.fail('online grasp called the circle solver')
    monkeypatch.setattr(bottle,'center_from_points',forbidden)
    c,stats=bottle.estimate(f,m,np.eye(4),[0,-1,0],dict(diameter_m=.07,height_m=.22),cfg)
    assert np.isfinite(c).all() and stats['center_source']=='front_depth_and_cross_section'
    f['depth'][:]=600
    with pytest.raises(bottle.DepthQualityError):bottle.estimate(f,m,np.eye(4),[0,-1,0],dict(diameter_m=.07,height_m=.22),cfg)
