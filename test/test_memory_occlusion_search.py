"""指定物体检索、光照与遮挡恢复；身份歧义和不可变参照回归。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_object_memory import ObjectAppearanceMemory,AmbiguousObject,lighting_crop


def setup(name='垃圾盖'):
    image=np.full((40,60,3),60,np.uint8);mask=np.ones((40,60),bool)
    views=[(dict(name=name),dict(image=image,mask=mask))]
    store=NS(view_ids=lambda _:['first']);item=dict(mask=mask,box=[0,0,60,40])
    return image,views,store,item


def test_specific_search_precedes_unrelated_categories_and_stops_on_match():
    image,views,store,item=setup();calls=[]
    p=NS(embed=lambda crops:np.tile([1.,0],(len(crops),1)),segmenter=NS(proposals=lambda *a,**k:calls.append(k['prompt']) or [item]))
    out=ObjectAppearanceMemory(p,{}).recognize(store,'id',views,dict(image=image))
    assert calls==['trash bin lid.']
    assert out[1]['recognition_prompt']=='trash bin lid.'


def test_missing_named_proposal_uses_generic_search_without_requiring_new_record():
    image,views,store,item=setup();calls=[]
    def proposals(*a,**kw):
        calls.append(kw['prompt']);return [] if len(calls)==1 else [item]
    p=NS(embed=lambda crops:np.tile([1.,0],(len(crops),1)),segmenter=NS(proposals=proposals))
    assert ObjectAppearanceMemory(p,{}).recognize(store,'id',views,dict(image=image)) is not None
    assert len(calls)==2 and 'bottle' in calls[1]


def test_ambiguous_specific_search_not_overridden_by_generic_one():
    image,views,store,item=setup();calls=[]
    p=NS(embed=lambda crops:np.tile([1.,0],(len(crops),1)),segmenter=NS(proposals=lambda *a,**k:calls.append(k['prompt']) or [item,item]))
    with pytest.raises(AmbiguousObject):ObjectAppearanceMemory(p,{}).recognize(store,'id',views,dict(image=image))
    assert len(calls)==1


def test_lighting_crop_preserves_source_and_normalizes_uniform_exposure():
    image,views,store,item=setup();original=image.copy();mask=item['mask']
    assert np.array_equal(lighting_crop(image,mask),lighting_crop(image*2,mask))
    assert np.array_equal(image,original)
    with pytest.raises(ValueError):lighting_crop(image,np.zeros_like(mask))


def test_lighting_match_requires_original_appearance_and_invalidates_deleted_view():
    image,views,store,item=setup();calls=[];ids=['first'];store.view_ids=lambda _:ids
    def embed(crops):calls.append(len(crops));return np.tile([1.,0],(len(crops),1))
    m=ObjectAppearanceMemory(NS(embed=embed),{});m.last_diagnostics=[{}]
    assert m.recognize_lighting(store,'id',views,dict(image=image),[item],np.array([[.70]]),.72,.06)==(0,0,1.)
    assert m.recognize_lighting(store,'id',views,dict(image=image),[item],np.array([[.1]]),.72,.06) is None
    ids.append('supplement');m.recognize_lighting(store,'id',views*2,dict(image=image),[item],np.array([[.70,.70]]),.72,.06)
    assert calls==[1,1,1,2,1]


def test_lighting_does_not_hide_two_similar_entities():
    image,views,store,item=setup();m=ObjectAppearanceMemory(NS(embed=lambda crops:np.tile([1.,0],(len(crops),1))),{})
    m.last_diagnostics=[{}]
    with pytest.raises(AmbiguousObject):
        m.recognize_lighting(store,'id',views,dict(image=image),[item,item],np.array([[.7],[.7]]),.72,.06)


def test_part_recovery_cannot_win_by_comparing_different_scoring_methods():
    image,views,store,item=setup()
    def embed(crops):return np.tile([.70,np.sqrt(1-.70**2)],(len(crops),1))
    m=ObjectAppearanceMemory(NS(embed=embed),{})
    m.recognize_visible_parts=lambda *a:(0,0,.81)
    with pytest.raises(AmbiguousObject):
        m.match_items(store,'id',views,dict(image=image),[item,item],np.array([[1.,0.]]),'trash bin lid.')


def test_small_hole_appearance_does_not_fill_occlusion_or_change_input():
    from autolife_shelf_grasp_306.visual_object_memory import appearance_mask,appearance_crop
    m=np.zeros((100,140),bool);m[10:90,10:130]=True
    m[30:33,30:33]=False  # SAM内部碎孔
    m[40:60,55:75]=False  # 真实大孔应保持
    m[70:90,100:130]=False  # 边界遮挡不能补画
    old=m.copy();a=appearance_mask(m)
    assert a[30:33,30:33].all()
    assert not a[40:60,55:75].any() and not a[70:90,100:130].any()
    assert np.array_equal(m,old)
    image=np.full((100,140,3),51,np.uint8);before=image.copy()
    assert appearance_crop(image,m).ndim==3 and np.array_equal(image,before)


def test_appearance_mask_keeps_empty_and_separate_components():
    from autolife_shelf_grasp_306.visual_object_memory import appearance_mask
    m=np.zeros((40,80),bool)
    assert not appearance_mask(m).any()
    m[5:35,5:20]=True;m[5:35,55:70]=True
    assert np.array_equal(appearance_mask(m),m)


def test_rival_recovery_cannot_silently_change_identity():
    image,views,store,item=setup()
    values=np.array([[.76,np.sqrt(1-.76**2)],[.67,np.sqrt(1-.67**2)]])
    m=ObjectAppearanceMemory(NS(embed=lambda crops:values),{})
    m.recognize_visible_parts=lambda *a:(1,0,.85)
    m.recognize_lighting=lambda *a:None
    with pytest.raises(AmbiguousObject):
        m.match_items(store,'id',views,dict(image=image),[item,item],np.array([[1.,0.]]),'trash bin lid.')
