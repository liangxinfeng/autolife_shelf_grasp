"""反光盖子的实测深度整理：遮挡、顺序不变性和错误位姿回归。"""
import numpy as np
import pytest
from test_visual_shape_position import scene,CFG
from autolife_shelf_grasp_306.visual_depth_surface import coherent_frame,projected_surface,surface_pair,dominant_plane
from autolife_shelf_grasp_306.visual_shape_position import position_from_shape
from autolife_shelf_grasp_306.visual_registration import validate_surface


def test_coherence_does_not_fill_holes_or_mutate_input():
    f=scene();f['depth'][80:100,150:160]=0;f['depth'][110,155]=1100
    original=f['depth'].copy();clean,stats=coherent_frame(f,f['mask'],CFG)
    assert np.array_equal(f['depth'],original)
    assert np.array_equal(clean['mask'],f['mask'])
    assert clean['depth'][110,155]==0
    assert not clean['depth'][80:100,150:160].any()
    keep=clean['depth']!=0
    assert np.array_equal(clean['depth'][keep],original[keep])
    assert stats['coherent_valid']<stats['raw_valid']


@pytest.mark.parametrize('fault',['holes','spikes','occluder','background','sparse'])
def test_planar_translation_survives_partial_bad_depth(fault):
    a=scene();b=scene(10)
    if fault=='holes':b['depth'][90:120,150:175]=0
    elif fault=='spikes':
        y,x=np.nonzero(b['mask']);rng=np.random.default_rng(7);bad=rng.random(len(y))<.15
        b['depth'][y[bad],x[bad]]=rng.uniform(1000,1500,bad.sum())
    elif fault=='occluder':b['depth'][85:125,150:180]=450
    elif fault=='background':b['depth'][85:125,150:180]=1100
    else:
        y,x=np.indices(b['mask'].shape);b['depth'][(x//12+y//12)%3!=0]=0
    r,c,report=surface_pair(a,np.eye(4),b,np.eye(4),CFG)
    d,stats=position_from_shape(r,np.eye(4),c,np.eye(4),CFG)
    assert np.allclose(d[:3,3],[.015,0,0],atol=.002)
    assert validate_surface(r['cloud'],c['cloud'],d,True)>.9
    assert report['depth_filter']=='observed_dominant_plane'
    keep=c['depth']!=0
    assert np.array_equal(c['depth'][keep],b['depth'][keep])


def test_validation_uses_geometry_not_row_sampling_order():
    # 每隔3点抽稀会只保留一个窄条，恰好丢掉另一个有效观测条带。
    x=np.linspace(0,.6,3000)
    a=np.column_stack([np.repeat(x,3),np.tile([0,.05,.1],3000),np.ones(9000)])
    b=a[1::3]
    assert validate_surface(a,b,np.eye(4),True)==1
    shuffled=a[np.random.default_rng(9).permutation(len(a))]
    assert validate_surface(shuffled,b,np.eye(4),True)==1
    with pytest.raises(ValueError,match='深度不一致'):
        validate_surface(a,b+[0,0,.1],np.eye(4),True)


def test_curved_surface_is_not_flattened():
    f=scene();y,x=np.indices(f['depth'].shape)
    f['depth'][f['mask']]=600+200*((x[f['mask']]-160)/30)**2
    r,c,report=surface_pair(f,np.eye(4),f,np.eye(4),CFG)
    assert report['depth_filter']=='local_continuity'
    keep=c['depth']!=0
    assert np.array_equal(c['depth'][keep],f['depth'][keep])


@pytest.mark.parametrize('mode',['empty','line','nonfinite'])
def test_insufficient_geometry_does_not_invent_plane(mode):
    p=np.empty((0,3)) if mode=='empty' else np.column_stack([np.linspace(0,1,200),np.zeros((200,2))])
    if mode=='nonfinite':p[:]=np.nan
    assert dominant_plane(p) is None


def test_two_surfaces_cannot_fabricate_a_translation():
    a=scene();b=scene(10);b['depth'][60:120,140:200]=900
    # 两个等大的水平带没有一个覆盖完整目标，不先删除其中一层来制造唯一解。
    r,c,report=surface_pair(a,np.eye(4),b,np.eye(4),CFG)
    assert report['depth_filter']=='local_continuity'
    with pytest.raises(ValueError):position_from_shape(r,np.eye(4),c,np.eye(4),CFG)
