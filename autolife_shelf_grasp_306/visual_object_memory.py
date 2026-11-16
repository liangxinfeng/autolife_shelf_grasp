"""动作参照复用抓取库的DINOv2整物体特征；不读取硬件、不生成运动。

外观与几何分离：补录的任意位置只提供外观，初始帧仍是唯一位姿基准。
"""
import numpy as np
from .core import crop_object


class AmbiguousObject(ValueError):
    """多个实体外观近似时，不能用路径是否可达来替用户挑对象。"""


def normalized(values):
    a=np.asarray(values,dtype=np.float32)
    if a.ndim!=2 or not len(a) or not np.isfinite(a).all():raise ValueError('物体外观特征无效')
    norms=np.linalg.norm(a,axis=1,keepdims=True)
    if np.any(norms<1e-8):raise ValueError('物体外观特征为空')
    return a/norms


def select_match(query,reference,threshold=.72,margin=.06):
    """取最相近已录视角，避免正面/背面平均反而惩罚已录的背面。

margin比较当前画面中不同实体，而不是同一物体的两个补录视角。
返回相似度而非置信概率；几何可靠性另行判断。
"""
    scores=normalized(query)@normalized(reference).T
    best=scores.max(axis=1);order=np.argsort(-best,kind='stable');index=int(order[0])
    if best[index]<threshold:return None
    if len(order)>1 and best[index]-best[order[1]]<margin:
        raise AmbiguousObject('画面中有多个外观近似的已记忆物体，请只保留本次交互对象可见；无需重新录入')
    return index,int(scores[index].argmax()),float(best[index])


def memory_prompt(cfg,views):
    """物体记忆检索独立于抓取页的临时文字筛选；补录不改变初始参照。"""
    from .grounded_sam2 import normalize_prompt
    # 自定义名字不一定是类别；不能解释的名字仍使用原有广泛候选检索。
    name=str(views[0][0].get('name','')) if views else ''
    try:prompt=normalize_prompt(name)
    except ValueError:prompt=''
    default=cfg.get('grounded_sam2_prompts',{}).get('all','bottle. cup. box. bag. tissue. trash bin lid. cabinet door.')
    # 保留通用候选，避免用户自定义英文编号变成唯一检测条件。
    phrases=(prompt+normalize_prompt(default)).split('.')
    unique=list(dict.fromkeys(x.strip() for x in phrases if x.strip()))[:16]
    return normalize_prompt('. '.join(unique))


def appearance_mask(mask):
    """只恢复外观裁图中封闭的小碎孔；不扩张外轮廓，不修改定位用掩码/深度。

    SAM会将同一表面的反光斑点切成孔，灰底裁图会把这些孔误当成外观变化。
    仅取孔内实际拍到的RGB；大孔、与外边界相连的缺口和遮挡仍保持原状。
    """
    import cv2
    mask=np.asarray(mask,bool)
    outer=np.zeros(mask.shape,np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outer,contours,-1,1,cv2.FILLED)
    n,labels,stats,_=cv2.connectedComponentsWithStats((outer.astype(bool)&~mask).astype(np.uint8))
    small=(np.arange(n)>0)&(stats[:,cv2.CC_STAT_AREA]<=mask.sum()*.01)
    return mask|small[labels]


def appearance_crop(image,mask):
    return crop_object(image,appearance_mask(mask))


def object_parts(image,mask):
    """同一物体的四块外观；只裁实际可见像素，不补画遮挡区域。"""
    y,x=np.nonzero(mask)
    if len(x)<320:return None
    x0,x1=int(x.min()),int(x.max()+1);y0,y1=int(y.min()),int(y.max()+1)
    xm,ym=(x0+x1)//2,(y0+y1)//2
    regions=((y0,ym,x0,xm),(y0,ym,xm,x1),(ym,y1,x0,xm),(ym,y1,xm,x1))
    if any(np.count_nonzero(mask[a:b,c:d])<80 for a,b,c,d in regions):return None
    return [crop_object(image[a:b,c:d],mask[a:b,c:d]) for a,b,c,d in regions]


def lighting_crop(image,mask):
    """仅归一化实际物体亮度；不改颜色比例、不填补遮挡、不改原图或深度。"""
    import cv2
    mask=appearance_mask(mask)
    pixels=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)[mask]
    if not len(pixels):raise ValueError('物体分割为空')
    gain=np.clip(112./max(1.,float(np.median(pixels))),.5,2.)
    adjusted=np.clip(np.asarray(image,dtype=float)*gain,0,255).astype(np.uint8)
    return crop_object(adjusted,mask)


def visible_part_scores(query,reference,whole,threshold):
    """至少三块同位置外观一致，并有整物体佐证；单块标签不代表整物体。"""
    scores=np.asarray(whole).copy()
    for i,q in enumerate(query):
        if q is None:continue
        for j,r in enumerate(reference):
            if r is None or whole[i,j]<threshold-.15:continue
            local=np.sort(np.sum(q*r,axis=1))[-3:]
            if local.min()>=threshold:scores[i,j]=max(scores[i,j],float(local.mean()))
    return scores


class ObjectAppearanceMemory:
    def __init__(self,perception,cfg):
        self.perception=perception;self.cfg=cfg;self.cache={};self.part_cache={};self.lighting_cache={};self.last_diagnostics=[]

    def part_features(self,images):
        values=[]
        for image,mask in images:
            parts=object_parts(image,appearance_mask(mask))
            values.append(None if parts is None else normalized(self.perception.embed(parts)))
        return values

    def recognize_visible_parts(self,store,ident,views,frame,items,whole,threshold,margin):
        signature=tuple(store.view_ids(ident));cached=self.part_cache.get(ident)
        if cached is None or cached[0]!=signature:
            cached=(signature,self.part_features([(a['image'],a['mask']) for _,a in views]))
            self.part_cache[ident]=cached
            while len(self.part_cache)>32:self.part_cache.pop(next(iter(self.part_cache)))
        query=self.part_features([(frame['image'],x['mask']) for x in items])
        scores=visible_part_scores(query,cached[1],whole,threshold)
        best=scores.max(axis=1);order=np.argsort(-best,kind='stable');index=int(order[0])
        if best[index]<threshold:return None
        if len(order)>1 and best[index]-best[order[1]]<margin:
            raise AmbiguousObject('可见部分对应多个外观近似物体，请确认本次交互对象；无需重新录入')
        return index,int(scores[index].argmax()),float(best[index])

    def reference_features(self,store,ident,views):
        signature=tuple(store.view_ids(ident))
        cached=self.cache.get(ident)
        if cached is None or cached[0]!=signature:
            crops=[appearance_crop(a['image'],a['mask']) for _,a in views]
            # 分批限制显存；模型复用已有抓取进程，不重复加载。
            values=np.concatenate([self.perception.embed(crops[i:i+8]) for i in range(0,len(crops),8)])
            self.cache[ident]=(signature,normalized(values))
            while len(self.cache)>32:self.cache.pop(next(iter(self.cache)))
        return self.cache[ident][1]

    def recognize_lighting(self,store,ident,views,frame,items,whole,threshold,margin):
        signature=tuple(store.view_ids(ident));cached=self.lighting_cache.get(ident)
        if cached is None or cached[0]!=signature:
            crops=[lighting_crop(a['image'],a['mask']) for _,a in views]
            values=np.concatenate([self.perception.embed(crops[i:i+8]) for i in range(0,len(crops),8)])
            cached=(signature,normalized(values));self.lighting_cache[ident]=cached
            while len(self.lighting_cache)>32:self.lighting_cache.pop(next(iter(self.lighting_cache)))
        query=normalized(self.perception.embed([lighting_crop(frame['image'],x['mask']) for x in items]))
        scores=query@cached[1].T
        # 原始外观也须相近；不能靠拉亮一件无关物体来制造相同身份。
        scores=np.where(whole>=threshold-.15,scores,-1.)
        self.last_diagnostics[-1]['lighting_similarity']=scores.max(axis=1).astype(float).tolist()
        best=scores.max(axis=1);order=np.argsort(-best,kind='stable');index=int(order[0])
        if best[index]<threshold:return None
        if len(order)>1 and best[index]-best[order[1]]<margin:
            raise AmbiguousObject('光照归一化后仍有多个近似物体，请确认本次交互对象')
        return index,int(scores[index].argmax()),float(best[index])

    def recognize(self,store,ident,views,frame):
        """先查指定记忆物体，再查通用候选；不让无关类别改变目标分割。"""
        reference=self.reference_features(store,ident,views)
        segmenter=getattr(self.perception,'segmenter',None)
        self.last_diagnostics=[]
        if segmenter is None:
            return self.match_items(store,ident,views,frame,self.perception._legacy_items(frame,'all'),reference,'legacy')
        from .grounded_sam2 import normalize_prompt
        try:target=normalize_prompt(views[0][0].get('name','')) if views else ''
        except ValueError:target=''
        prompts=list(dict.fromkeys(p for p in (target,memory_prompt(self.cfg,views)) if p))
        for prompt in prompts:
            items=segmenter.proposals(frame,'all',prompt=prompt)
            match=self.match_items(store,ident,views,frame,items,reference,prompt)
            if match is not None:return match
        return None

    def match_items(self,store,ident,views,frame,items,reference,prompt):
        if not items:
            self.last_diagnostics.append(dict(prompt=prompt,candidates=[]))
            return None
        # 不复用含背景的矩形crop，登记和查询都使用同一分割灰底预处理。
        query=self.perception.embed([appearance_crop(frame['image'],x['mask']) for x in items])
        threshold=float(self.cfg.get('match_threshold',.72));margin=float(self.cfg.get('match_margin',.06))
        whole=normalized(query)@reference.T
        self.last_diagnostics.append(dict(prompt=prompt,candidates=[dict(box=x.get('box'),label=x.get('semantic_label',''),similarity=float(whole[i].max())) for i,x in enumerate(items)]))
        match=select_match(query,reference,threshold,margin);method='dinov2_object'
        if match is None:
            whole=normalized(query)@reference.T
            # 正常整物体识别不增加计算；局部遮挡才比较三块可见外观。
            if whole.max()>=threshold-.15:
                match=self.recognize_visible_parts(store,ident,views,frame,items,whole,threshold,margin)
                method='dinov2_visible_parts'
                if match is None:
                    match=self.recognize_lighting(store,ident,views,frame,items,whole,threshold,margin)
                    method='dinov2_lighting'
        if match is None:return None
        index,view,score=match
        # 多实体时，对接近阈值的竞争者也使用相同恢复证据，避免只修好赢家的分割。
        raw=whole.max(axis=1)
        if method=='dinov2_object' and any(raw[i]>=threshold-margin for i in range(len(items)) if i!=index):
            for recovered in (
                self.recognize_visible_parts(store,ident,views,frame,items,whole,threshold,margin),
                self.recognize_lighting(store,ident,views,frame,items,whole,threshold,margin),
            ):
                if recovered is not None and recovered[0]!=index:
                    raise AmbiguousObject('多个物体的整物体与可见部分匹配指向不同目标，请确认本次交互对象')
        if method!='dinov2_object':
            # 局部/光照分数不能与另一实体的原始分数直接比高低。
            # 两个原始外观均很接近时，恢复方法不能凭分数增益换选对象。
            raw=whole.max(axis=1)
            rivals=[i for i in range(len(items)) if i!=index and raw[i]>=threshold-margin and abs(raw[index]-raw[i])<margin]
            if rivals:raise AmbiguousObject('当前可见部分对应多个近似物体，请确认本次交互对象；无需重新录入')
        return items[index],dict(recognition_method=method,recognition_view='初始帧' if view==0 else f'补录视角{view}',
                                recognition_similarity=score,recognition_candidates=len(items),recognition_views=len(views),recognition_prompt=prompt)
