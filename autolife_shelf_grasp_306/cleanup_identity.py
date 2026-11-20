"""Disposal decisions use operator names; detector words describe one observation."""
import base64
import cv2
import numpy as np

from .core import unit


WASTE_FAMILIES = {
    'discarded wrapper': '废弃包装', 'torn packaging': '废弃包装',
    'crumpled paper': '废纸团', 'food scraps': '食物残渣', 'broken fragments': '破损碎片',
}
KEEP = ('sponge', 'cleaning cloth', 'phone', 'wallet', 'keys', 'plant', 'reusable mug')
STATE_PROMPT = '. '.join((*WASTE_FAMILIES, *KEEP)) + '.'


def local_parts(perception,crop):
    """Foreground DINO patch descriptors sampled across the segmented object."""
    h,w=crop.shape[:2]; side=max(h,w)
    canvas=np.full((side,side,3),127,np.uint8)
    canvas[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=crop
    image=cv2.resize(canvas,(224,224))
    coverage=cv2.resize(np.any(image!=127,axis=2).astype('float32'),(16,16),interpolation=cv2.INTER_AREA)
    indices=np.flatnonzero(coverage.ravel()>.8)
    if not len(indices):return None
    indices=indices[np.linspace(0,len(indices)-1,min(64,len(indices)),dtype=int)]
    torch=perception.torch
    x=torch.from_numpy(image[:,:,::-1].copy()).permute(2,0,1)[None].to(perception.device)/255.
    with torch.inference_mode():
        patches=perception.dino.forward_features((x-perception.mean)/perception.std)['x_norm_patchtokens'][0]
        return torch.nn.functional.normalize(patches,dim=1).cpu().numpy()[indices]


def visual_profile(perception,item,labels):
    labels=sorted({v.strip() for v in labels if v.strip()})
    if not labels:return None
    parts=local_parts(perception,item['crop'])
    if parts is None:return None
    return dict(labels=labels,width=parts.shape[1],parts=base64.b64encode(parts.astype('<f2').tobytes()).decode())


def waste_family(labels):
    if any(' '+word+' ' in label for label in labels for word in KEEP):
        return ''
    for phrase, family in WASTE_FAMILIES.items():
        if any(' '+phrase+' ' in label for label in labels):
            return family
    return ''


class TaughtCategories:
    def __init__(self, samples):
        self.names = []
        self.profiles = []
        vectors = []
        for name, examples in samples.items():
            for entry in examples:
                profile=entry.get('profile')
                if profile:
                    parts=np.frombuffer(base64.b64decode(profile['parts']),dtype='<f2').reshape(-1,profile['width'])
                    views=np.asarray([unit(v) for v in (entry['feature'],*entry.get('views',[]))])
                    self.profiles.append((name,set(profile['labels']),views,parts.astype('float32')))
                for vector in (entry['feature'], *entry.get('views', [])):
                    vectors.append(unit(vector))
                    self.names.append(name)
        self.vectors = np.asarray(vectors)
        self.groups = {name: np.flatnonzero(np.asarray(self.names) == name)
                       for name in dict.fromkeys(self.names)}

    def lookup(self, feature, cfg):
        if not self.groups:
            return {}
        similarities = self.vectors @ unit(feature)
        ranked = sorted(((float(similarities[indices].max()), name)
                         for name, indices in self.groups.items()), reverse=True)
        score, name = ranked[0]
        gap = score - (ranked[1][0] if len(ranked) > 1 else -1.)
        accepted = score >= max(.86, cfg['match_threshold']) and gap >= cfg['match_margin']
        return dict(id=name if accepted else None, score=score, margin=gap)

    def alternate_view(self,perception,item,labels,global_match):
        if global_match.get('id') or not self.profiles:return None
        words={label.strip() for label in labels}
        if any(word in label.split() for label in words for word in ('phone','wallet','keys','plant')):
            return None
        query=unit(item['feature'])
        minimum=max(.40,global_match['score']-perception.cfg['match_margin'])
        choices=[(name,parts) for name,tags,views,parts in self.profiles
                 if tags & words and (views@query).max()>=minimum]
        if not choices:return None
        current=local_parts(perception,item['crop'])
        if current is None:return None
        scores={}
        for name,parts in choices:
            pairs=current@parts.T
            similarity=float(min(pairs.max(axis=0).mean(),pairs.max(axis=1).mean()))
            scores[name]=max(scores.get(name,0.),similarity)
        ranked=sorted(scores,key=scores.get,reverse=True)
        best=ranked[0];gap=scores[best]-(scores[ranked[1]] if len(ranked)>1 else 0.)
        if scores[best]>=.42 and gap>=perception.cfg['match_margin']:
            return dict(id=best,parts=scores[best],score=global_match['score'],margin=gap,method='parts_and_label')
