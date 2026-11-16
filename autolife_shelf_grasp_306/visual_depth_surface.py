"""物体纠偏的同帧深度整理；只剔除不连续回波，不补造缺失深度。"""
import cv2
import numpy as np


def coherent_frame(frame,mask,cfg):
    z=np.asarray(frame['depth'],float)*float(cfg['depth_scale_m'])
    lo,hi=cfg['depth_range_m'];valid=np.isfinite(z)&(z>=lo)&(z<=hi)&np.asarray(mask,bool)
    count=np.zeros(z.shape,np.uint8)
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        neighbor=np.roll(z,(dy,dx),(0,1));good=np.roll(valid,(dy,dx),(0,1))
        if dy<0:good[-1,:]=False
        elif dy>0:good[0,:]=False
        if dx<0:good[:,-1]=False
        elif dx>0:good[:,0]=False
        count+=good&(np.abs(neighbor-z)<.015)
    keep=valid&(count>=2)
    depth=np.array(frame['depth'],copy=True);depth[~keep]=0
    return dict(frame,depth=depth,mask=np.asarray(mask,bool)),dict(raw_valid=int(valid.sum()),coherent_valid=int(keep.sum()))


def projected_surface(frame,T,cfg):
    z=np.asarray(frame['depth'],float)*float(cfg['depth_scale_m']);lo,hi=cfg['depth_range_m']
    safe=cv2.erode(np.asarray(frame['mask'],np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    safe[:2,:]=False;safe[-2:,:]=False;safe[:,:2]=False;safe[:,-2:]=False
    good=safe&np.isfinite(z)&(z>=lo)&(z<=hi)
    y,x=np.nonzero(good);pixels=np.c_[x,y]
    rays=cv2.undistortPoints(pixels.astype(float)[:,None,:],frame['K'],frame['D']).reshape(-1,2) if len(x) else np.empty((0,2))
    points=np.c_[rays*z[y,x,None],z[y,x]]
    T=np.asarray(T,float);return points@T[:3,:3].T+T[:3,3],pixels


def dominant_plane(points):
    """仅从已观测表面拟合主平面；不将桌面/类别先验强加给未知物品。"""
    points=np.asarray(points,float);points=points[np.isfinite(points).all(1)]
    if len(points)<100:return None
    rng=np.random.default_rng(381);p=points[rng.choice(len(points),min(2400,len(points)),False)]
    best=None
    for _ in range(160):
        q=p[rng.choice(len(p),3,False)];n=np.cross(q[1]-q[0],q[2]-q[0]);length=np.linalg.norm(n)
        if length<1e-7:continue
        n/=length;d=float(q[0]@n);inside=np.abs(p@n-d)<.006
        if best is None or inside.sum()>best.sum():best=inside
    if best is None or best.sum()<80:return None
    center=p[best].mean(0);_,singular,v=np.linalg.svd(p[best]-center,full_matrices=False);n=v[-1];d=float(center@n)
    inside=np.abs(p@n-d)<.009
    if singular[1]/np.sqrt(best.sum())<.008:return None
    return n,d,float(inside.mean())


def surface_pair(reference,reference_camera,current,current_camera,cfg):
    """轮廓保留完整RGB分割；定位与表面复核共用整理后的真实深度。"""
    ref,rs=coherent_frame(reference,reference['mask'],cfg)
    cur,cs=coherent_frame(current,current['mask'],cfg)
    rp,_=projected_surface(ref,reference_camera,cfg);cp,uv=projected_surface(cur,current_camera,cfg)
    plane=dominant_plane(rp);mode='local_continuity'
    if plane is not None and plane[2]>=.80:
        candidate=dominant_plane(cp)
        if candidate is not None and candidate[2]>=.35:
            n,d,_=candidate;inside=np.abs(cp@n-d)<.009
            # 主表面必须横跨可见区域，局部遮挡物不能代替整个物体的深度。
            span=np.ptp(uv,axis=0);coverage=np.ptp(uv[inside],axis=0)/np.maximum(span,1)
            if np.min(coverage)>=.65:
                rn,rd,_=plane
                for frame,T,normal,offset in ((ref,reference_camera,rn,rd),(cur,current_camera,n,d)):
                    p,pix=projected_surface(frame,T,cfg);keep=np.abs(p@normal-offset)<.009
                    depth=np.zeros_like(frame['depth']);xy=pix[keep];depth[xy[:,1],xy[:,0]]=frame['depth'][xy[:,1],xy[:,0]];frame['depth']=depth
                mode='observed_dominant_plane'
    for frame,T in ((ref,reference_camera),(cur,current_camera)):
        frame['cloud'],_=projected_surface(frame,T,cfg)
    return ref,cur,dict(depth_filter=mode,reference_depth=rs,current_depth=cs,
                        reference_surface_points=len(ref['cloud']),current_surface_points=len(cur['cloud']))
