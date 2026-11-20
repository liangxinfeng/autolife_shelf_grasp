"""Estimate an upright contact region from an RGB outline and a measured support plane."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .bottle import DepthQualityError,horizontal_basis
from .low_object_grasp import observe_support
from .object_extent import coherent_component,oriented_bounds


def measured_section(frame,mask,T,template,points,pixels,plane,basis,height):
    """Measure a contact section and enclose the whole observed object independently."""
    keep=coherent_component(points)
    points,pixels=points[keep],pixels[keep]
    normal=np.asarray(plane['normal'])
    z=points@normal-plane['offset']
    noise=max(.001,plane['fit_residual_mm']/1000)
    if np.percentile(z,1)<-3*noise:
        raise DepthQualityError('截面点云穿过实测支撑面')
    band=np.abs(z-height/2)<=height*.10
    if np.count_nonzero(band)<24:
        raise DepthQualityError('指尖接触高度的实测截面不足')
    xy=points@basis
    section=xy[band]
    rect=cv2.minAreaRect(section.astype('float32'))
    # The last few edge pixels often have no depth. Extend only those rays,
    # using the range of the nearest measured depths at each individual edge.
    boundary=mask & ~cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
    yy,xx=np.where(boundary)
    edge=np.column_stack([xx,yy])
    distances,neighbors=cKDTree(pixels).query(edge,k=4)
    optical_z=(points-T[:3,3])@T[:3,2]
    error=float(distances[:,-1].max()*optical_z.max()/min(frame['K'][0,0],frame['K'][1,1]))
    if error>max(3*noise,.20*min(rect[1])):
        raise DepthQualityError('实测深度未覆盖物体外轮廓，无法约束夹取宽度')
    uv=cv2.undistortPoints(edge.astype(float),frame['K'],frame['D']).reshape(-1,2)
    rays=np.c_[uv,np.ones(len(uv))]@T[:3,:3].T
    ranges=optical_z[neighbors]
    footprint=np.vstack([xy,*[(T[:3,3]+rays*r[:,None])@basis for r in (ranges.min(1),ranges.max(1))]])
    middle=np.asarray(cv2.minAreaRect(footprint.astype('float32'))[0])
    contact_hull=cv2.convexHull(section.astype('float32'))
    clearance=cv2.pointPolygonTest(contact_hull,tuple(middle),True)
    # An oblique flat face can project a band even though its thickness is
    # unknown. Require real 3-D extent beyond the measured plane noise.
    thickness=np.sqrt(max(0.,np.linalg.eigvalsh(np.cov(points.T))[0]))
    if clearance<=3*noise or thickness<=3*noise:
        raise DepthQualityError('夹取截面只有单侧深度，无法确认中心')
    hull=cv2.convexHull((footprint-middle).astype('float32'))[:,0]
    margin=3*noise
    directions=np.arange(8)*np.pi/4
    pad=np.c_[np.cos(directions),np.sin(directions)]*margin/np.cos(np.pi/8)
    hull=cv2.convexHull((hull[:,None,:]+pad).reshape(-1,2).astype('float32'))[:,0]
    vertices=hull@basis.T
    _,_,_,axes=oriented_bounds(points,normal)
    span=2*np.max(np.abs(vertices@axes[:,:2]),axis=0)
    center=basis@middle+normal*(plane['offset']+height/2)
    product=dict(template,diameter_m=float(span[0]),depth_m=float(span[1]),height_m=height,
                 bbox_axes_torso=axes.tolist(),bbox_extents_m=[*span,height],footprint_vertices_torso=vertices.tolist())
    stats=dict(temporary_product=product,geometry='solid',observed_height_mm=height*1000,
               height_source='measured_support_to_top',image_cropped=False,points=int(band.sum()),
               center_source='measured_midsection',surface_evidence='measured_midsection',
               contact_section_width_mm=float(min(rect[1])*1000),edge_depth_gap_mm=error*1000,
               outline_margin_mm=margin*1000,measured_support=plane)
    return center,stats,points[band],pixels[band]


def estimate_outline(frame,mask,T,up,template,cfg,points,pixels):
    """Intersect the silhouette at two heights and verify the reconstructed sides against depth."""
    local,low,high,axes=oriented_bounds(points,up)
    origin=axes@((low+high)/2)
    plane=observe_support(frame,mask,T,dict(template,height_m=high[2]-low[2]),origin,up,cfg)['low_object_support']
    if plane['source']!='local_rgbd_plane':
        raise DepthQualityError('未测到轮廓附近的支撑面')
    normal=np.asarray(plane['normal'])
    z=points@normal-plane['offset']
    height=float(np.percentile(z,97))
    noise=max(.001,plane['fit_residual_mm']/1000)
    if height<max(.012,6*noise) or height>.40:
        raise DepthQualityError('轮廓内未测到突出支撑面的物体')
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<.88*np.count_nonzero(mask):
        raise DepthQualityError('图像轮廓包含多个物体')
    camera=T[:3,3]
    rays=cv2.undistortPoints(contour.astype(float),frame['K'],frame['D']).reshape(-1,2)
    directions=np.c_[rays,np.ones(len(rays))]@T[:3,:3].T
    slope=directions@normal
    if np.any(slope>=-.05):raise DepthQualityError('当前视角无法用支撑面测量轮廓')
    u,v,axis=horizontal_basis(normal)
    basis=np.column_stack([u,v])
    def hull_at(h):
        t=(plane['offset']+h-camera@normal)/slope
        return cv2.convexHull(((camera+t[:,None]*directions)@basis).astype('float32'))
    area,polygon=cv2.intersectConvexConvex(hull_at(0),hull_at(height))
    if area<=0 or polygon is None:
        return measured_section(frame,mask,T,template,points,pixels,plane,basis,height)
    xy=polygon.reshape(-1,2)
    bottom=xy@basis.T+normal*plane['offset']
    corners=np.vstack([bottom,bottom+height*normal])
    optical=(corners-camera)@T[:3,:3]
    projected=cv2.projectPoints(optical,np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0]
    silhouette=np.zeros(mask.shape,'uint8')
    cv2.fillConvexPoly(silhouette,cv2.convexHull(np.rint(projected).astype('int32')),1)
    overlap=np.logical_and(mask,silhouette).sum()/max(1,np.logical_or(mask,silhouette).sum())
    if overlap<.80:
        return measured_section(frame,mask,T,template,points,pixels,plane,basis,height)
    rect=cv2.minAreaRect(xy);theta=np.deg2rad(rect[2])
    first=basis@np.array([np.cos(theta),np.sin(theta)])
    if rect[1][0]>rect[1][1]:first=np.cross(axis,first)
    if first@u<0:first=-first
    axes=np.column_stack([first,np.cross(axis,first),axis])
    center=basis@rect[0]+normal*(plane['offset']+height/2)
    # Agreeing depth must cover both sides of the mid-body in the camera view.
    side=np.cross(axis,center-camera);side/=max(np.linalg.norm(side),1e-9)
    lateral=(bottom-center)@side
    signed=np.array([cv2.pointPolygonTest(polygon,(float(x),float(y)),True) for x,y in points@basis])
    tol=max(.005,3*noise)
    body=(z>.20*height)&(z<.80*height)&(np.abs(signed)<2*tol)
    if body.sum()<24:raise DepthQualityError('未测到与轮廓吻合的夹取侧面')
    contact_axis=(points[body]-center)@side
    a,b=np.percentile(contact_axis,[3,97])
    bins,_=np.histogram(contact_axis,bins=6,range=(a,b)) if b>a else (np.zeros(6),None)
    if np.min(bins)<2:raise DepthQualityError('夹取带的中间存在深度断层')
    if a>lateral.min()+.28*np.ptp(lateral) or b<lateral.max()-.28*np.ptp(lateral):
        raise DepthQualityError('实测夹取侧面未覆盖轮廓两侧')
    # Reject a nearer object intruding in front of the visual hull.
    front=center-camera;front-=axis*(front@axis);front/=max(np.linalg.norm(front),1e-9)
    intrusion=(points-center)@front<np.min((bottom-center)@front)-2*tol
    if np.mean(intrusion&(z>.15*height))>.05:raise DepthQualityError('轮廓前有另一片更近的深度')
    # Bound taper and silhouette reprojection error in the carried footprint.
    missing=cv2.distanceTransform((silhouette==0).astype('uint8'),cv2.DIST_L2,5)
    err=float(missing[mask].max())*np.max(optical[:,2])/min(frame['K'][0,0],frame['K'][1,1])
    margin=max(tol,err)
    shifted=np.vstack([xy+[x,y] for x in (-margin,margin) for y in (-margin,margin)])
    hull=cv2.convexHull(shifted.astype('float32')).reshape(-1,2)@basis.T
    vertices=hull-(center-axis*(center@axis))
    span=2*np.max(np.abs(vertices@axes[:,:2]),axis=0)
    product=dict(template,diameter_m=float(span[0]),depth_m=float(span[1]),height_m=height,
        bbox_axes_torso=axes.tolist(),bbox_extents_m=[*span,height],footprint_vertices_torso=vertices.tolist())
    stats=dict(temporary_product=product,geometry='solid',observed_height_mm=height*1000,
        height_source='measured_support_to_top',image_cropped=False,points=int(body.sum()),
        center_source='supported_outline',surface_evidence='measured_midbody_and_silhouette',
        silhouette_iou=float(overlap),outline_margin_mm=margin*1000,measured_support=plane)
    return center,stats,points[body],pixels[body]
