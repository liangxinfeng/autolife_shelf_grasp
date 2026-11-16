"""绑定编号、局部遮挡、亮度变化、真实深度、歧义及不可变参照。"""
from types import SimpleNamespace as NS
from copy import deepcopy
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_bound_registration import reference_surface,visible_support,BoundReferenceLocator
from autolife_shelf_grasp_306.visual_object_memory import AmbiguousObject

CFG=dict(depth_scale_m=.001,depth_range_m=[.1,2.])


def scene():
    image=np.zeros((160,200,3),np.uint8);mask=np.zeros((160,200),bool);mask[30:130,40:160]=True
    image[mask]=[80,65,40];depth=np.zeros(mask.shape,np.uint16);depth[mask]=600
    f=dict(image=image,mask=mask,depth=depth,K=np.array([[300.,0,100],[0,300.,80],[0,0,1]]),D=np.zeros(5))
    meta=dict(camera_pose=np.eye(4).tolist(),object_pose=np.eye(4).tolist(),name='垃圾盖')
    return meta,f


@pytest.mark.parametrize('mode',['unchanged','brighter','dimmer','near_occluder','same_depth_occluder','holes'])
def test_visible_reference_survives_local_changes_without_template_update(mode):
    meta,ref=scene();before=deepcopy(ref);surface=reference_surface(meta,ref,CFG);f=deepcopy(ref)
    if mode=='brighter':f['image']=np.clip(f['image']*1.5,0,255).astype(np.uint8)
    if mode=='dimmer':f['image']=(f['image']*.65).astype(np.uint8)
    if mode in ('near_occluder','same_depth_occluder'):
        f['image'][90:130,120:160]=[220,220,220]
        if mode=='near_occluder':f['depth'][90:130,120:160]=560
    if mode=='holes':f['depth'][::3,::3]=0
    stats=visible_support(surface,f,np.eye(4),np.eye(4),f['mask'],CFG)
    assert stats['bound_surface_points']>80 and not stats['whole_appearance_gate']
    if mode=='near_occluder':assert stats['excluded_foreground_points']>100
    if mode=='same_depth_occluder':assert stats['excluded_surface_color_points']>100
    assert all(np.array_equal(ref[k],before[k]) for k in ref)


@pytest.mark.parametrize('mode',['empty_depth','wrong_depth','wrong_color','line'])
def test_nonexistent_or_foreign_surface_is_not_fake_correction(mode):
    meta,ref=scene();surface=reference_surface(meta,ref,CFG);f=deepcopy(ref)
    if mode=='empty_depth':f['depth'][:]=0
    if mode=='wrong_depth':f['depth'][f['mask']]=900
    if mode=='wrong_color':f['image'][f['mask']]=[130,130,120]
    if mode=='line':f['depth'][:]=0;f['depth'][30:130,99:102]=600
    with pytest.raises(ValueError):visible_support(surface,f,np.eye(4),np.eye(4),f['mask'],CFG)


def test_ref_projection_accounts_for_camera_pose_and_true_translation():
    meta,ref=scene();surface=reference_surface(meta,ref,CFG)
    t=np.eye(4);t[:3,3]=[.15,-.1,.03]
    stats=visible_support(surface,ref,t,t,ref['mask'],CFG)
    assert stats['bound_surface_points']>1000


def test_cached_binding_never_uses_supplement_position_or_overwrites_original():
    meta,ref=scene();store=NS(view_ids=lambda _:['original','supplement'])
    views=[(meta,ref),(dict(camera_pose='not a coordinate source',metric_registered=False),{})]
    locator=BoundReferenceLocator(None,CFG);a=locator.surface(store,'a',views)
    assert a is locator.surface(store,'a',views)
    assert np.allclose(a['points'][:,2],.6)


def test_binding_locator_never_calls_embed_and_does_not_choose_between_instances(monkeypatch):
    import autolife_shelf_grasp_306.visual_bound_registration as module
    meta,ref=scene();item=dict(mask=ref['mask'],box=[40,30,160,130]);items=[item]
    p=NS(segmenter=NS(proposals=lambda *a,**k:items),embed=lambda *a:pytest.fail('appearance gate'))
    store=NS(view_ids=lambda _:['original']);locator=BoundReferenceLocator(p,CFG)
    monkeypatch.setattr(module,'visible_support',lambda *a:dict(bound_surface_points=100))
    def geometry(*args):return np.eye(4),dict(inliers=200),args[4],{},np.zeros((100,3))
    result=locator.locate(store,'id',[(meta,ref)],ref,np.eye(4),True,geometry)
    assert result[1]['reference_id']=='id' and result[1]['recognition_method']=='bound_visible_rgbd'
    items.append(dict(item,box=[5,5,30,30]))
    with pytest.raises(AmbiguousObject):locator.locate(store,'id',[(meta,ref)],ref,np.eye(4),True,geometry)


def test_geometry_linked_supplement_uses_registered_camera_not_capture_position():
    meta,ref=scene();linked=deepcopy(meta);t=np.eye(4);t[0,3]=.03;linked.update(camera_pose=t.tolist(),metric_registered=True)
    store=NS(view_ids=lambda _:['initial','linked','old-no-depth'])
    locator=BoundReferenceLocator(None,CFG)
    surfaces=locator.surfaces(store,'id',[(meta,ref),(linked,ref),(dict(metric_registered=True),{})])
    assert len(surfaces)==2 and np.allclose(surfaces[1][1]['points']-surfaces[0][1]['points'],[.03,0,0])
