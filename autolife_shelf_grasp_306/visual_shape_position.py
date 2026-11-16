"""已确认物体身份后，用可见轮廓与分布式深度对应估计平移。

不把补录时的摆放位置当作初始位置，也不拿部分点云的质心差当运动。
本方法只服务“只纠正位置”：轮廓相似且视角接近时，在归一化轮廓内
取分散的深度对应，求一致的平移。无法观测的绕轴角不伪装成已测量。
"""
import cv2
import numpy as np
from .visual_reference import matrix


def _outline(arrays):
    mask=np.asarray(arrays['mask'],bool)
    if mask.ndim!=2 or mask.shape!=np.asarray(arrays['depth']).shape:
        raise ValueError('物体轮廓与深度尺寸不一致，请重新取景')
    from .visual_box_selection import dominant_component
    mask=dominant_component(mask)
    y,x=np.nonzero(mask)
    if len(x)<100:raise ValueError('请框选完整物体，当前轮廓太小')
    h,w=mask.shape
    lo=np.array([x.min(),y.min()]);hi=np.array([x.max(),y.max()])
    if np.min(hi-lo)<8:raise ValueError('物体轮廓太窄，请靠近观察')
    crop=mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1].astype(np.uint8)
    shape=cv2.resize(crop,(64,96),interpolation=cv2.INTER_NEAREST)>0
    return mask,lo,hi,shape


def _edge_flags(outline):
    mask,lo,hi,_=outline[:4]
    h,w=mask.shape
    # 贴边一像素并不意味着物体已不可识别。
    return lo<=1,hi>=np.array([w-2,h-2])


def _partial_outline(reference,current,oa,ob,cfg):
    """裁边后从参考尺寸与连贯表面深度恢复投影尺寸，只采样真实像素。

    不能把裁短的包围框拉伸成完整物体，也不能用反光区域的深度中位数
    决定整物体大小。用最密集的连贯表面深度计算投影比例，在仍可见的
    对边锚定缺失端。最终由分布式三维对应判断，不设缺失比例上限。
    """
    left,right=_edge_flags(ob)
    if not (left.any() or right.any()):return ob,False
    if any(x.any() for x in _edge_flags(oa)):
        raise ValueError('参考物体缺少完整边界；需要纹理对应来确定位置')
    if np.any(left&right):
        raise ValueError('物体同一方向两端均不可见，需通过表面纹理确定该方向位置')
    extent=(oa[2]-oa[1]).astype(float)
    lo=ob[1].astype(float).copy();hi=ob[2].astype(float).copy()
    focal=np.diag(current['K'])[:2]/np.diag(reference['K'])[:2]
    def surface_depth(arr,mask):
        d=np.asarray(arr['depth'],float)*float(cfg['depth_scale_m'])
        low,high=cfg['depth_range_m'];v=np.sort(d[mask&np.isfinite(d)&(d>=low)&(d<=high)])
        if len(v)<100:raise ValueError('当前可见部分有效深度不足，无法确定投影尺寸')
        # 反光可能产生大量远处假深度。取最密集的连续表面深度簇，
        # 只用于投影尺寸；最终平移仍由真实分散对应点估计。
        ends=np.searchsorted(v,v+.04,side='right');i=int(np.argmax(ends-np.arange(len(v))))
        return float(np.median(v[i:ends[i]]))
    scale=surface_depth(reference,oa[0])/surface_depth(current,ob[0])
    expected=extent*focal*scale
    for axis in (0,1):
        if left[axis]:lo[axis]=hi[axis]-expected[axis]
        elif right[axis]:hi[axis]=lo[axis]+expected[axis]
    return (ob[0],lo,hi,ob[3]),True


def _oriented_outline(arrays):
    """按完整轮廓主方向展开，避免横向包围框把透视倾斜当作形状变化。"""
    mask,lo,hi,shape=_outline(arrays)
    y,x=np.nonzero(mask);xy=np.column_stack((x,y)).astype(float);center=xy.mean(0)
    values,vectors=np.linalg.eigh(np.cov((xy-center).T))
    if values[1]<1.35*max(values[0],1.):return mask,lo,hi,shape
    along=vectors[:,1]
    if along[1]<0:along=-along
    across=np.array([along[1],-along[0]]);basis=np.column_stack((across,along))
    uv=(xy-center)@basis;lo=uv.min(0);hi=uv.max(0)
    gx,gy=np.meshgrid(np.linspace(0,1,64),np.linspace(0,1,96))
    pixels=(lo+np.column_stack((gx.ravel(),gy.ravel()))*(hi-lo))@basis.T+center
    px,py=np.rint(pixels).astype(int).T;h,w=mask.shape
    valid=(px>=0)&(px<w)&(py>=0)&(py<h);normalized=np.zeros(len(px),bool)
    normalized[valid]=mask[py[valid],px[valid]]
    return mask,lo,hi,normalized.reshape(96,64),basis,center


def _samples(arrays,T,cfg,grid,outline):
    mask,lo,hi,_=outline[:4]
    pixels=lo+grid*(hi-lo)
    if len(outline)==6:pixels=pixels@outline[4].T+outline[5]
    x,y=np.rint(pixels).astype(int).T;h,w=mask.shape
    inside=(x>=1)&(x<w-1)&(y>=1)&(y<h-1);x=x.clip(1,w-2);y=y.clip(1,h-2)
    depth=np.asarray(arrays['depth'],float)*float(cfg['depth_scale_m'])
    z=depth[y,x];low,high=cfg['depth_range_m']
    valid=inside&mask[y,x]&np.isfinite(z)&(z>=low)&(z<=high)
    neighbors=np.zeros(len(z),np.uint8)
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        near=depth[y+dy,x+dx]
        neighbors+=mask[y+dy,x+dx]&np.isfinite(near)&(near>=low)&(near<=high)&(np.abs(near-z)<.015)
    # 反光空洞不是相反证据；保留有两个邻点支持的真实深度，而不是要求四邻域全满。
    # 后面仍需分散对应点的一致平移、歧义排除及完整表面复核，不插值填深度。
    valid&=neighbors>=2
    rays=cv2.undistortPoints(pixels[:,None,:].astype(float),arrays['K'],arrays['D']).reshape(-1,2)
    points=np.column_stack((rays*z[:,None],z))@T[:3,:3].T+T[:3,3]
    return points,pixels,valid


def _estimate(reference,reference_camera,current,current_camera,cfg,oriented=False):
    """调用前必须已通过某个保存视角认出物体；返回初始帧到当前帧的平移。"""
    aT=matrix(reference_camera);bT=matrix(current_camera)
    cosine=np.clip((np.trace(aT[:3,:3].T@bT[:3,:3])-1)/2,-1,1)
    if np.degrees(np.arccos(cosine))>12:
        raise ValueError('物体已认出，但观察方向变化较大；请从接近初始观察方向预览位置纠偏')
    oa=_outline(reference);ob=_outline(current)
    ob,partial=_partial_outline(reference,current,oa,ob,cfg)
    if not partial and any(x.any() for x in _edge_flags(oa)):
        raise ValueError('参考物体已超出画面，请补录完整参考视角')
    if oriented:
        if partial:raise ValueError('裁边使用原始投影方向，不从残缺轮廓推测旋转')
        oa=_oriented_outline(reference);ob=_oriented_outline(current)
    sa,sb=oa[3],ob[3];iou=float(np.sum(sa&sb)/max(1,np.sum(sa|sb)))
    aspect=(oa[2]-oa[1])[0]/(oa[2]-oa[1])[1]/((ob[2]-ob[1])[0]/(ob[2]-ob[1])[1])
    # 二维形状仅作诊断评分；是否可纠偏由分布式深度对应和三维重叠决定。
    grid=np.array([(x,y) for y in np.linspace(.04,.96,40) for x in np.linspace(.08,.92,24)])
    ap,_,av=_samples(reference,aT,cfg,grid,oa)
    bp,pixels,bv=_samples(current,bT,cfg,grid,ob)
    valid=av&bv
    diff=(bp-ap)[valid];uv=grid[valid];pixels=pixels[valid]
    if len(diff)<40:raise ValueError('物体已认出，但有效深度覆盖不足；请调整相机视角再预览')
    # 共识平移剔除透明表面透出的背景深度；每个格子一票，不重复计数。
    best=None
    for shift in diff[::max(1,len(diff)//120)]:
        good=np.linalg.norm(diff-shift,axis=1)<.012
        if best is None or good.sum()>best.sum():best=good
    shift=np.median(diff[best],axis=0)
    good=np.linalg.norm(diff-shift,axis=1)<.012
    if good.sum()<40:
        raise ValueError('物体已认出，但深度对应不一致；请让物体完整可见后预览')
    # 遮挡物、反光深度不属于定位参照，不能强求它们也服从同一平移。
    # 少于60%时要求存在唯一主导簇，后面仍检查空间分布、残差和表面。
    occluded=bool(good.mean()<.60);runner=0
    if occluded:
        for candidate in diff[::max(1,len(diff)//120)]:
            if np.linalg.norm(candidate-shift)>.024:
                runner=max(runner,int((np.linalg.norm(diff-candidate,axis=1)<.012).sum()))
        if runner>=.65*int(good.sum()):
            raise ValueError('可见物体深度存在多个位置解释，暂不能确定纠偏方向')
    shift=np.median(diff[good],axis=0)
    rms=float(np.sqrt(np.mean(np.sum((diff[good]-shift)**2,axis=1))))
    spread=np.ptp(uv[good],axis=0)
    # 裁边时检查真实可见区域的覆盖，而不是要求画面外也有深度。
    # 同时保留三维分布，不能仅凭一条线或极小标签推断整体平移。
    support=ap[valid][good]
    physical_span=np.linalg.svd(support-support.mean(0),compute_uv=False)[1]/np.sqrt(len(support))
    if partial:
        visible_span=np.ptp(uv,axis=0)
        coverage=spread/np.maximum(visible_span,1e-9)
        distributed=min(coverage)>=.65 and physical_span>=.004
    else:
        quadrants={(int(x>=.5),int(y>=.5)) for x,y in uv[good]}
        distributed=min(spread)>=.55 and len(quadrants)>=(3 if occluded else 4)
        # 完整外轮廓可见但深度缺失时，不能强求两轴都覆盖固定百分比。
        # 允许有真实二维面积、跨至少三象限的窄带支持平移；不由一条线或小标签外推。
        distributed=distributed or (max(spread)>=.55 and np.prod(spread)>=.25 and physical_span>=.008 and len(quadrants)>=3)
    if rms>.008 or not distributed:
        raise ValueError('当前可见部分的深度对应过于集中，尚不能确定物体位置')
    delta=np.eye(4);delta[:3,3]=shift
    return delta,dict(method='shape_position',method_label='可见轮廓＋深度位置纠偏' if partial else ('方向对齐＋深度位置纠偏' if oriented else '整体轮廓＋深度位置纠偏'),
        silhouette_alignment='principal_axes' if oriented else 'image_axes',
        orientation_observed=False,angle_resolution_deg=180.,partial_outline=partial,
        occlusion_consensus=occluded,alternative_inliers=runner,
        position_resolution_mm=max(2.,2*rms*1000),rms_mm=rms*1000,
        physical_support_mm=float(physical_span*1000),inliers=int(good.sum()),matches=len(diff),silhouette_overlap=iou,silhouette_aspect_ratio=float(aspect),
        pixels=pixels[good].tolist())


def position_from_shape(reference,reference_camera,current,current_camera,cfg):
    """保持原估计优先；形状/深度不符时先纠正轮廓方向，再核验同样的深度证据。"""
    try:return _estimate(reference,reference_camera,current,current_camera,cfg)
    except ValueError as original:
        try:return _estimate(reference,reference_camera,current,current_camera,cfg,oriented=True)
        except ValueError:
            raise original
