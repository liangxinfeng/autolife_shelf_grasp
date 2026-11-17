"""停靠前预测目标在头部相机中的完整位置；不移动相机或底盘。"""
import math,itertools
import numpy as np
import cv2


def observation(ground,T,frame):
    h,w=frame['image'].shape[:2]
    return dict(camera_ground=(ground@T).tolist(),K=np.asarray(frame['K']).tolist(),D=np.asarray(frame['D']).tolist(),width=w,height=h)


def quality(point,product,T,view):
    """目标中心和外接角点都保留在图像内部；边缘位置降权。"""
    from .operation_docking import se2
    extent=np.asarray(product.get('bbox_extents_m',[product.get('diameter_m',.06),product.get('depth_m',.06),product.get('height_m',.06)]))
    axes=np.asarray(product.get('bbox_axes_torso',np.eye(3)))
    offsets=np.array(list(itertools.product((-1.,1.),repeat=3)))*extent/2
    corners=np.vstack([point,point+offsets@axes.T])
    cam=corners@T[:3,:3].T+T[:3,3]
    if np.any(cam[:,2]<.20) or np.any(cam[:,2]>1.8):return None
    uv=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),np.asarray(view['K']),np.asarray(view['D']))[0][:,0,:]
    w,h=view['width'],view['height'];edge=.08
    if np.any(uv[:,0]<edge*w) or np.any(uv[:,0]>(1-edge)*w) or np.any(uv[:,1]<edge*h) or np.any(uv[:,1]>(1-edge)*h):return None
    norm=(uv[0]-np.array([w/2,h/2]))/np.array([w/2,h/2])
    return dict(score=float(norm@norm),pixel=uv[0].tolist(),edge_margin_px=float(min(uv[:,0].min(),w-uv[:,0].max(),uv[:,1].min(),h-uv[:,1].max())))


def candidate_options(candidate,ctx,ground,point,product,view):
    from .operation_docking import se2
    if view is None:return [(dict(candidate),0.,None)]
    before=se2(ctx['pose']);target_world=(before@ground@np.r_[point,1.])[:3]
    camera=np.asarray(view['camera_ground']);axis=camera[:3,2]
    # 在原链上的位置停靠，朝向允许面向本次目标；导航端必须使用同一已检朝向。
    facing=math.atan2(target_world[1]-candidate['y'],target_world[0]-candidate['x'])-math.atan2(axis[1],axis[0])
    options=[]
    for yaw in dict.fromkeys((float(candidate['yaw']),math.atan2(math.sin(facing),math.cos(facing)))):
        c={**candidate,'yaw':yaw};B=np.linalg.inv(se2([c['x'],c['y'],yaw]))@before
        T=np.linalg.inv(camera)@B@ground
        value=quality(np.asarray(point),product,T,view)
        if value is not None:options.append((c,value['score'],value))
    return options
