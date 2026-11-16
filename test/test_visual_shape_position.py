"""已识别物体的深度位置纠偏：真实投影、错误输入及调用模式隔离。"""
from copy import deepcopy
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_shape_position import position_from_shape

CFG=dict(depth_scale_m=.001,depth_range_m=[.1,2.])


def scene(dx=0,dy=0,z=600.):
    mask=np.zeros((240,320),bool);mask[60+dy:180+dy,130+dx:190+dx]=True
    depth=np.full(mask.shape,900.,float);depth[mask]=z
    return dict(mask=mask,depth=depth,K=np.array([[400.,0,160],[0,400.,120],[0,0,1]]),D=np.zeros(5))


@pytest.mark.parametrize('dx,dy',[(0,0),(15,-10),(-12,10)])
def test_shape_translation_with_no_descriptors(dx,dy):
    a=scene();b=scene(dx,dy);d,s=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[dx*.6/400,dy*.6/400,0],atol=1e-8)
    assert np.array_equal(d[:3,:3],np.eye(3)) and not s['orientation_observed']
    assert s['angle_resolution_deg']==180 and s['inliers']>100


def test_uses_camera_transform_not_camera_pixel_difference():
    a=scene();T=np.eye(4);T[:3,3]=[.03,-.02,.01]
    d,_=position_from_shape(a,np.eye(4),a,T,CFG)
    assert np.allclose(d[:3,3],T[:3,3])


def test_partial_transparent_background_does_not_bias_translation():
    a=scene();b=scene(10);b['depth'][65:95,145:175]=1300
    d,s=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[.015,0,0],atol=.0001) and s['inliers']<s['matches']


def test_partial_object_or_wrong_shape_rejected():
    a=scene();b=scene();b['mask'][120:]=False
    with pytest.raises(ValueError,match='深度'):position_from_shape(a,np.eye(4),b,np.eye(4),CFG)


def test_small_label_patch_depth_not_enough_for_whole_object():
    a=scene();b=scene();b['depth'][:]=0;b['depth'][105:130,140:180]=600
    with pytest.raises(ValueError,match='集中|不足'):position_from_shape(a,np.eye(4),b,np.eye(4),CFG)


def test_bad_camera_direction_or_offscreen_cannot_fake_translation():
    from scipy.spatial.transform import Rotation
    a=scene();T=np.eye(4);T[:3,:3]=Rotation.from_euler('y',25,degrees=True).as_matrix()
    with pytest.raises(ValueError,match='观察方向'):position_from_shape(a,np.eye(4),a,T,CFG)
    a['mask'][:65,130:190]=True
    with pytest.raises(ValueError,match='缺少完整边界'):position_from_shape(a,np.eye(4),a,np.eye(4),CFG)


def clipped_scene(dx,dy):
    # 裁切真正平移后的物体，不使用 Python 负下标回绕。
    b=scene();b['mask'][:]=False
    b['mask'][max(0,60+dy):min(240,180+dy),max(0,130+dx):min(320,190+dx)]=True
    b['depth'][:]=900.;b['depth'][b['mask']]=600.
    return b


@pytest.mark.parametrize('dx,dy',[(-132,0),(132,0),(0,-65),(0,65),(-132,-65),(-130,0)])
def test_small_clipping_uses_complete_extent_and_unbiased_translation(dx,dy):
    d,s=position_from_shape(scene(),np.eye(4),clipped_scene(dx,dy),np.eye(4),CFG)
    assert np.allclose(d[:3,3],[dx*.6/400,dy*.6/400,0],atol=.0001)
    assert s['partial_outline'] and s['inliers']>=40


@pytest.mark.parametrize('dx,dy',[(-150,0),(150,0),(0,-90),(0,90),(-165,0),(165,0),(-150,-90)])
def test_large_clipping_uses_visible_support_without_bias(dx,dy):
    d,s=position_from_shape(scene(),np.eye(4),clipped_scene(dx,dy),np.eye(4),CFG)
    assert np.allclose(d[:3,3],[dx*.6/400,dy*.6/400,0],atol=.0001)
    assert s['partial_outline'] and s['inliers']>=40


def test_cropped_object_with_internal_missing_depth():
    a=scene();b=clipped_scene(-150,0)
    b['depth'][95:120,15:30]=0
    d,s=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[-.225,0,0],atol=.0001)
    assert s['inliers']>=40


def test_reflections_do_not_set_projected_extent_from_bad_median():
    a=scene();b=clipped_scene(-150,0)
    # 远处反光值总数较多，但分散；真实表面仍是最密集的连贯深度簇。
    y,x=np.nonzero(b['mask']);rng=np.random.default_rng(4)
    bad=rng.random(len(y))<.52
    b['depth'][y[bad],x[bad]]=rng.uniform(950,1700,bad.sum())
    from autolife_shelf_grasp_306.visual_shape_position import _outline,_partial_outline
    ob,partial=_partial_outline(a,b,_outline(a),_outline(b),CFG)
    assert partial and np.allclose(ob[1],[-20,60]) and np.allclose(ob[2],[39,179])


def test_tiny_sliver_or_missing_depth_cannot_determine_position():
    for b in (clipped_scene(-183,0),clipped_scene(-150,0)):
        if b['mask'].sum()>1000:b['depth'][:]=0
        with pytest.raises(ValueError):position_from_shape(scene(),np.eye(4),b,np.eye(4),CFG)


def test_both_ends_missing_is_ambiguous_without_texture():
    b=scene();b['mask'][:]=False;b['mask'][60:180,:]=True;b['depth'][b['mask']]=600
    with pytest.raises(ValueError,match='两端'):
        position_from_shape(scene(),np.eye(4),b,np.eye(4),CFG)


def test_clipping_does_not_accept_wrong_background_depth():
    b=clipped_scene(-132,0);b['depth'][b['mask']]=1200
    with pytest.raises(ValueError):position_from_shape(scene(),np.eye(4),b,np.eye(4),CFG)


def test_appearance_supplement_can_win_over_weak_original(monkeypatch):
    from autolife_shelf_grasp_306.visual_multiview import seed_match
    a=(dict(metric_registered=True),{'tag':'old'})
    b=(dict(metric_registered=False),{'tag':'new'})
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.register_views',lambda *args:(np.eye(4),dict(inliers=8,rms_mm=3),*a))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_registration.register',lambda *args:(np.eye(4),dict(inliers=40,rms_mm=1)))
    _,_,meta,arrays=seed_match([a,b],{})
    assert arrays['tag']=='new' and not meta['metric_registered']


def test_observe_fallback_uses_initial_not_supplement_position(monkeypatch):
    from types import SimpleNamespace as NS
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    from autolife_shelf_grasp_306.visual_reference import matrix
    a=scene();a.update(image=np.zeros((240,320,3),np.uint8),points=np.zeros((10,3)),pixels=np.zeros((10,2)),descriptors=np.zeros((10,128)),cloud=np.zeros((40,3)))
    b=deepcopy(a);b['mask']=scene(10)['mask']
    meta=dict(camera_pose=np.eye(4).tolist(),object_pose=np.eye(4).tolist(),created=1)
    supplemental=dict(meta,metric_registered=False,created=2,object_pose=(np.eye(4)+np.diag([0,0,0,0])).tolist())
    supplemental['object_pose'][0][3]=.7
    v=VisualActions.__new__(VisualActions);v.cfg=CFG;v.perception=None;v.reference_cache={}
    v.store=NS(views=lambda _: [(meta,a),(supplemental,a)],view_ids=lambda _:['first','supplement'])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.resolve_views',lambda x:x)
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_registration.image_features',lambda *a,**k:(_ for _ in ()).throw(ValueError('use stored')))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_multiview.seed_match',lambda *args:(np.eye(4),dict(inliers=20),supplemental,a))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.projected_box',lambda *a:[.1,.1,.9,.9])
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_box_selection.segment_box',lambda *a:dict(mask=b['mask']))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.features',lambda *a,**k:{})
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.register_views',lambda *a:(_ for _ in ()).throw(ValueError('different surface')))
    monkeypatch.setattr('autolife_shelf_grasp_306.bottle.object_points',lambda *a:(np.zeros((40,3)),None,None))
    monkeypatch.setattr('autolife_shelf_grasp_306.visual_action_worker.validate_surface',lambda *a:1.)
    d,s,*_=v.observe('id',b,np.eye(4),position_only=True)
    assert np.allclose(d[:3,3],[.015,0,0],atol=1e-8) and s['method']=='shape_position'
    with pytest.raises(ValueError,match='朝向'):v.observe('id',b,np.eye(4),position_only=False)


@pytest.mark.parametrize('dx,dy',[(10,0),(-10,7)])
def test_distributed_depth_band_not_rejected_by_relative_width(dx,dy):
    a=scene();b=scene(dx,dy)
    # 真实深度跨上下与左右中心，但只剩物体宽度40%的连续条带。
    b['depth'][:]=0;b['depth'][60+dy:180+dy,148+dx:172+dx]=600
    d,s=position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[dx*.6/400,dy*.6/400,0],atol=.0001)
    assert s['physical_support_mm']>=8 and s['inliers']>=40


def test_line_depth_is_not_two_dimensional_support():
    a=scene();b=scene(10);b['depth'][:]=0;b['depth'][60:180,168:172]=600
    with pytest.raises(ValueError,match='集中|不足|不一致'):
        position_from_shape(a,np.eye(4),b,np.eye(4),CFG)
