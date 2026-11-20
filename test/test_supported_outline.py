import cv2
import numpy as np
import pytest
from autolife_shelf_grasp_306 import supported_outline as outline
from autolife_shelf_grasp_306.bottle import DepthQualityError


def scene(monkeypatch):
    camera=np.array([0.,-.45,.32]);forward=np.array([0.,0.,.04])-camera
    forward/=np.linalg.norm(forward)
    x=np.array([1.,0.,0.]);T=np.eye(4);T[:3,:3]=np.column_stack([x,np.cross(forward,x),forward]);T[:3,3]=camera
    frame=dict(K=np.array([[550.,0.,160.],[0.,550.,120.],[0.,0.,1.]]),D=np.zeros(5))
    def project(p):
        return np.rint(cv2.projectPoints((p-camera)@T[:3,:3],np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0][:,0]).astype('int32')
    corners=np.array([[x,y,z] for x in [-.034,.034] for y in [-.026,.026] for z in [0.,.08]])
    mask=np.zeros((240,320),'uint8');cv2.fillConvexPoly(mask,cv2.convexHull(project(corners)),1)
    x,z=np.meshgrid(np.linspace(-.034,.034,40),np.linspace(0,.08,35))
    face=np.c_[x.ravel(),np.full(x.size,-.026),z.ravel()]
    x,y=np.meshgrid(np.linspace(-.034,.034,34),np.linspace(-.026,.026,34))
    top=np.c_[x.ravel(),y.ravel(),np.full(x.size,.08)]
    points=np.r_[face,top]
    plane=dict(normal=[0.,0.,1.],offset=0.,source='local_rgbd_plane',fit_residual_mm=1.)
    monkeypatch.setattr(outline,'observe_support',lambda *a:dict(low_object_support=plane))
    return frame,mask.astype(bool),T,points,project,plane


@pytest.mark.parametrize('name',['cup','box','unlabelled'])
def test_supported_outline_recovers_measured_body_without_class_rule(monkeypatch,name):
    frame,mask,T,p,project,_=scene(monkeypatch)
    center,stats,*_=outline.estimate_outline(frame,mask,T,[0,0,1],dict(name=name),{},p,project(p))
    np.testing.assert_allclose(center,[0,0,.04],atol=.003)
    assert stats['silhouette_iou']>.95
    assert min(stats['temporary_product']['bbox_extents_m'][:2])>=.052


@pytest.mark.parametrize('fault',['table','no_plane','separate_masks','separate_sides','occluder'])
def test_background_and_mixed_targets_are_rejected(monkeypatch,fault):
    frame,mask,T,p,project,plane=scene(monkeypatch)
    if fault=='table':p[:,2]=0.
    elif fault=='no_plane':plane['source']='estimated'
    elif fault=='separate_masks':mask[15:65,20:70]=True
    elif fault=='separate_sides':p=p[np.abs(p[:,0])>.022]
    else:p=np.r_[p,p[:1000]+[0.,-.08,.02]]
    with pytest.raises(DepthQualityError):outline.estimate_outline(frame,mask,T,[0,0,1],{}, {},p,project(p))


@pytest.mark.parametrize('sides,upper_radius',[(48,.032),(4,.040),(7,.030)])
def test_midsection_uses_geometry_and_bounds_complete_taper(monkeypatch,sides,upper_radius):
    frame,_,T,_,project,plane=scene(monkeypatch)
    phi=np.arange(sides)*2*np.pi/sides
    corners=np.c_[np.cos(phi),np.sin(phi)]
    xy=np.concatenate([a+(b-a)*np.linspace(0.,1.,max(3,120//sides),endpoint=False)[:,None]
                       for a,b in zip(corners,np.roll(corners,-1,axis=0))])
    cloud=np.vstack([np.c_[xy*(.022+(upper_radius-.022)*z/.08),np.full(len(xy),z)]
                     for z in np.linspace(0,.08,50)])
    pix=project(cloud);mask=np.zeros((240,320),'uint8')
    cv2.fillConvexPoly(mask,cv2.convexHull(pix),1)
    center,stats,*_=outline.measured_section(frame,mask.astype(bool),T,{},cloud,pix,plane,np.eye(3)[:,:2],.08)
    assert np.linalg.norm(center-[0,0,.04])<.003
    product=stats['temporary_product']
    footprint=cv2.convexHull(np.asarray(product['footprint_vertices_torso'],np.float32)[:,:2].copy())
    assert all(cv2.pointPolygonTest(footprint,tuple(point),False)>=0 for point in xy*upper_radius-center[:2])


def test_planar_face_alone_cannot_supply_hidden_extent(monkeypatch):
    frame,_,T,p,project,plane=scene(monkeypatch)
    face=p[:1400];pix=project(face)
    mask=np.zeros((240,320),'uint8');cv2.fillConvexPoly(mask,cv2.convexHull(pix),1)
    with pytest.raises(DepthQualityError):
        outline.measured_section(frame,mask.astype(bool),T,{},face,pix,plane,np.eye(3)[:,:2],.08)
