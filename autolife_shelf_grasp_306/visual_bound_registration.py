"""绑定物体的RGB-D定位：候选搜索→暂定配准→可见表面验证；不重新做商品身份评分。

原始参照以不可变编号缓存。遮挡只删去不相符观测，不更新参照、不写姿态、
不发布运动。补录外观不能改变最初物体坐标。只有一致的当前观测才返回纠偏。
"""
import cv2
import numpy as np
from .visual_reference import matrix
from .visual_depth_surface import coherent_frame,projected_surface
from .visual_object_memory import AmbiguousObject,memory_prompt


def reference_surface(meta,arrays,cfg):
    clean,_=coherent_frame(arrays,arrays['mask'],cfg)
    points,uv=projected_surface(clean,matrix(meta['camera_pose']),cfg)
    return dict(points=points,colors=arrays['image'][uv[:,1],uv[:,0]])


def visible_support(surface,frame,camera,delta,mask,cfg):
    """在同一当前像素核对预测表面、实测深度与局部色度，剔除前景遮挡。

    色度只用于排除落在同一深度层的异物，不计算整物体外观嵌入分数。
    不填补深度，不把缺失像素或无观察区域当成已定位表面。
    """
    points=surface['points'];colors=surface['colors']
    moved=points@delta[:3,:3].T+delta[:3,3]
    inv=np.linalg.inv(matrix(camera));cam=moved@inv[:3,:3].T+inv[:3,3]
    valid=np.isfinite(cam).all(1)&(cam[:,2]>.05);cam=cam[valid];points=points[valid];colors=colors[valid]
    if not len(cam):raise ValueError('绑定参照未投影到当前视野')
    xy=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),frame['K'],frame['D'])[0].reshape(-1,2)
    xy=np.rint(xy).astype(int);h,w=frame['depth'].shape
    valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    cam=cam[valid];points=points[valid];colors=colors[valid];xy=xy[valid]
    # 多个参考像素投影到同一像素只算一次，保留最近的可见表面。
    order=np.argsort(cam[:,2],kind='stable');_,idx=np.unique((xy[order,1]*w+xy[order,0]),return_index=True)
    idx=order[idx];cam=cam[idx];points=points[idx];colors=colors[idx];xy=xy[idx]
    z=np.asarray(frame['depth'],float)[xy[:,1],xy[:,0]]*float(cfg['depth_scale_m'])
    lo,hi=cfg['depth_range_m'];measured=np.isfinite(z)&(z>=lo)&(z<=hi)
    on_surface=measured&(np.abs(z-cam[:,2])<.015)&np.asarray(mask,bool)[xy[:,1],xy[:,0]]
    observed=frame['image'][xy[:,1],xy[:,0]].astype(float);reference=colors.astype(float)
    # 归一化每个像素的亮度；同一蓝/灰表面变亮不会被整体外观评分挡住。
    chroma=np.linalg.norm(reference/np.maximum(reference.sum(1,keepdims=True),1)-observed/np.maximum(observed.sum(1,keepdims=True),1),axis=1)
    supported=on_surface&(chroma<.10)
    if on_surface.sum()<80 or supported.sum()<80:
        raise ValueError('绑定物体当前可见RGB-D对应不足，未生成虚构纠偏')
    if supported.sum()<.55*on_surface.sum():
        raise ValueError('候选表面的局部RGB-D与绑定参照不一致，已排除其他物体')
    samples=points[supported];span=np.linalg.svd(samples-samples.mean(0),compute_uv=False)[1]/np.sqrt(len(samples))
    if span<.008:raise ValueError('绑定参照只剩近共线局部，当前不能确定整体位置')
    return dict(bound_surface_points=int(supported.sum()),bound_depth_points=int(on_surface.sum()),
                excluded_foreground_points=int((measured&(z<cam[:,2]-.015)).sum()),
                excluded_surface_color_points=int((on_surface&~supported).sum()),
                bound_support_mm=float(span*1000),whole_appearance_gate=False,
                binding_policy='immutable_reference_visible_rgbd')


class BoundReferenceLocator:
    def __init__(self,perception,cfg):
        self.perception=perception;self.cfg=cfg;self.cache={};self.last_diagnostics=[]

    def surfaces(self,store,ident,views):
        signature=tuple(store.view_ids(ident));cached=self.cache.get(ident)
        if cached is None or cached[0]!=signature:
            # 初始帧是唯一公制基准；补录不同位置不改写此缓存的坐标。
            cached=(signature,[(i,reference_surface(meta,arrays,self.cfg)) for i,(meta,arrays) in enumerate(views) if (i==0 or meta.get('metric_registered',True)) and all(k in arrays for k in ('depth','image','mask','K','D'))]);self.cache[ident]=cached
            while len(self.cache)>32:self.cache.pop(next(iter(self.cache)))
        return cached[1]

    def surface(self,store,ident,views):
        return self.surfaces(store,ident,views)[0][1]

    def locate(self,store,ident,views,frame,camera,position_only,locate_mask):
        from .grounded_sam2 import normalize_prompt
        from .visual_box_selection import segment_box
        try:target=normalize_prompt(views[0][0].get('name',''))
        except ValueError:target=''
        prompts=list(dict.fromkeys(p for p in (target,memory_prompt(self.cfg,views)) if p))
        self.last_diagnostics=[];seen=[];failures=[]
        surfaces=self.surfaces(store,ident,views)
        for prompt in prompts:
            items=self.perception.segmenter.proposals(frame,'all',prompt=prompt)
            accepted=[]
            for item in items:
                box=np.asarray(item['box'],float)
                if any(np.max(np.abs(box-b))<3 for b in seen):continue
                seen.append(box)
                diag=dict(prompt=prompt,box=box.tolist(),label=item.get('semantic_label',''),stage='geometry')
                self.last_diagnostics.append(diag)
                alternatives=[item]
                for attempt in range(2):
                    local=[]
                    for candidate in alternatives:
                        try:
                            result=locate_mask(views,views,frame,camera,candidate,position_only,{})
                            verified=[];surface_error=None
                            for view,surface in surfaces:
                                try:verified.append((visible_support(surface,frame,camera,result[0],candidate['mask'],self.cfg),view))
                                except ValueError as exc:surface_error=exc
                            if not verified:raise surface_error
                            support,view=max(verified,key=lambda x:x[0]['bound_surface_points'])
                            result[1].update(support,recognition_method='bound_visible_rgbd',recognition_view='初始帧' if view==0 else f'已关联视角{view}',recognition_prompt=prompt,reference_id=ident)
                            local.append(result)
                        except ValueError as exc:diag['reason']=str(exc);failures.append(str(exc))
                    if local:
                        anchor=matrix(views[0][0]['object_pose'])
                        positions=[(x[0]@anchor)[:3,3] for x in local]
                        if any(np.linalg.norm(a-b)>.015 for a in positions for b in positions):
                            raise AmbiguousObject('绑定物体的分割产生不同位置解释，未选择任一运动目标')
                        best=max(local,key=lambda x:x[1]['bound_surface_points']);accepted.append(best)
                        diag.update(stage='accepted',**{k:v for k,v in best[1].items() if k.startswith('bound_') or k.startswith('excluded_')})
                        break
                    if attempt==0:
                        try:
                            h,w=frame['image'].shape[:2];main=segment_box(self.perception,frame,box/[w,h,w,h])
                            alternatives=[main]+main.get('_alternatives',[])
                        except ValueError as exc:diag['reason']=str(exc);break
            if len(accepted)>1:
                raise AmbiguousObject('当前有多个实体符合绑定参照，请确认本次交互对象；不自动更换绑定')
            if accepted:return accepted[0]
        if failures:raise ValueError('绑定物体定位未完成：'+failures[0])
        return None
