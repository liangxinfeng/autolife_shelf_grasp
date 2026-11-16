import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.unregistered import solid_observation,estimate_temporary,combine_product
from autolife_shelf_grasp_306.bottle import planning_product,combine,DepthQualityError
from autolife_shelf_grasp_306.demo_config import require_placeable

def fixture(monkeypatch,height,angle=0.,residual_mm=1.):
    x,y=np.meshgrid(np.linspace(-.03,.03,35),np.linspace(-.025,.025,35))
    z=height+.0003*np.sin(x*300)*np.cos(y*200)
    rotation=Rotation.from_euler('y',angle,degrees=True).as_matrix()
    points=np.c_[x.ravel(),y.ravel(),z.ravel()]@rotation.T+[.5,.1,.2]
    up=rotation[:,2];offset=float(up@np.array([.5,.1,.2]))
    pixels=np.c_[np.zeros(len(points)),np.linspace(10,110,len(points))]
    surface=dict(normal=up.tolist(),offset=offset,source='local_rgbd_plane',plane_points=1200,
                 fit_residual_mm=residual_mm,center=np.mean(points,axis=0).tolist(),radius_m=.23)
    monkeypatch.setattr('autolife_shelf_grasp_306.low_object_grasp.observe_support',lambda *a:{'low_object_support':surface})
    T=np.eye(4);T[:3,3]=[.45,.1,.6]
    mask=np.zeros((130,130),bool);mask[10:120,10:120]=True
    template=dict(id='paper',geometry='solid',temporary=True,dimension_source='multi_frame_rgbd')
    return points,pixels,up,T,mask,template,offset

@pytest.mark.parametrize('height',[.008,.012,.020,.024,.026,.04])
@pytest.mark.parametrize('angle',[0.,25.,-35.])
def test_surface_ripple_is_not_object_height(monkeypatch,height,angle):
    points,pixels,up,T,mask,template,offset=fixture(monkeypatch,height,angle)
    center,stats,_,_=solid_observation({'depth':np.ones((130,130))},mask,T,up,template,{},points,pixels)
    assert stats['visible_surface_span_mm']<1
    assert stats['height_source']=='measured_support_to_top'
    assert stats['observed_height_mm']==pytest.approx(height*1000,abs=.4)
    assert center@up-offset==pytest.approx(height/2,abs=.0003)
    product=stats['temporary_product']
    samples=[(center,stats)]*5;p=combine_product(template,samples);c,g=combine(samples)
    final=planning_product(p,g);require_placeable(final)
    assert final['height_m']==pytest.approx(height,abs=.0004)

@pytest.mark.parametrize('height',[0.,.001,.002])
def test_table_depth_is_not_fabricated_as_graspable(monkeypatch,height):
    points,pixels,up,T,mask,template,_=fixture(monkeypatch,height)
    with pytest.raises(DepthQualityError,match='桌面无法区分'):
        solid_observation({'depth':np.ones((130,130))},mask,T,up,template,{},points,pixels)

def test_points_below_table_are_not_fixed_by_relaxing_limits(monkeypatch):
    points,pixels,up,T,mask,template,_=fixture(monkeypatch,.02)
    points[:200]-=up*.04
    # 两个分离实体在几何聚类阶段或穿桌证据阶段被拒绝。
    with pytest.raises(DepthQualityError):solid_observation({'depth':np.ones((130,130))},mask,T,up,template,{},points,pixels)

def test_registered_and_direct_recognition_share_short_geometry(monkeypatch):
    from autolife_shelf_grasp_306.object_geometry import estimate_object
    points,pixels,up,T,mask,template,_=fixture(monkeypatch,.02)
    monkeypatch.setattr('autolife_shelf_grasp_306.unregistered.object_points',lambda *a:(points,pixels,(10,119)))
    monkeypatch.setattr('autolife_shelf_grasp_306.object_geometry.object_points',lambda *a:(points,pixels,(10,119)))
    frame={'depth':np.ones((130,130))}
    a,ga=estimate_temporary(frame,mask,T,up,template,{})
    b,gb=estimate_object(frame,mask,T,up,{**template,'diameter_m':.06,'depth_m':.05,'height_m':.02},{})
    np.testing.assert_allclose(a,b);assert ga['observed_height_mm']==gb['observed_height_mm']

def test_nonfinite_height_still_invalid():
    with pytest.raises(ValueError):planning_product(dict(geometry='solid',height_m=.01),dict(observed_height_mm=float('nan')))


@pytest.mark.parametrize('angle',[0.,25.,-35.])
def test_missing_segmentation_edges_do_not_reject_measured_solid(monkeypatch,angle):
    points,pixels,up,T,mask,template,offset=fixture(monkeypatch,.02,angle)
    # 只有内部有有效深度，但它仍覆盖物体实际三维上表面；上下薄边缺失。
    pixels[:,1]=np.linspace(40,80,len(pixels))
    monkeypatch.setattr('autolife_shelf_grasp_306.unregistered.object_points',lambda *a:(points,pixels,(10,119)))
    center,stats=estimate_temporary({'depth':np.ones((130,130))},mask,T,up,template,{})
    assert center@up-offset==pytest.approx(.01,abs=.0003)
    assert stats['height_source']=='measured_support_to_top'
    assert stats['image_vertical_depth_coverage'][0]>.12
    assert stats['image_vertical_depth_coverage'][1]<.88
    # 保存的高度和夹取宽度仍来自真实几何，没有用类别或默认尺寸替代。
    assert stats['observed_height_mm']==pytest.approx(20.,abs=.4)
    assert stats['temporary_product']['diameter_m']>0
