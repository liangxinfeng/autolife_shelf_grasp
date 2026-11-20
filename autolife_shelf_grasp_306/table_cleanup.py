"""Recognize disposable items on the visible tabletop; this module never moves the robot."""
from .cleanup_catalog import checked_types,library
import random
import time

import numpy as np

from .cleanup_identity import STATE_PROMPT, TaughtCategories, waste_family
from .perception import duplicate_mask
from .cleanup_surface import overlaps_table


CONTRAST = ('sponge', 'cleaning cloth', 'phone', 'wallet', 'keys')

def comparison_prompts(kinds):
    """Each detector query offers both disposal and retention alternatives."""
    protected = [word for word in CONTRAST if word not in kinds]
    group = []
    for phrase in kinds:
        words = [*group, phrase, *protected]
        if group and (len(words) > 16 or len('. '.join(words)) + 1 > 300):
            yield '. '.join([*group, *protected]) + '.'
            group = []
        group.append(phrase)
    if group:
        yield '. '.join([*group, *protected]) + '.'


def same_instance(first, second):
    largest = max(np.count_nonzero(first), np.count_nonzero(second))
    return largest > 0 and 2 * np.count_nonzero(first & second) >= largest


def disposal_names(labels,kinds):
    return [name.partition('=')[0].strip() for phrase,name in kinds.items()
            if any(' '+phrase+' ' in label for label in labels)]


def inspect_scene(perception, frame, store, request):
    started = time.monotonic()
    active=request['types']
    samples={k:v for k,v in library(store.root.parent)['samples'].items() if k in active}
    kinds = checked_types(active,samples)
    segment = perception.segmenter.proposals
    tables = segment(frame,'all',prompt='table. tabletop.')
    request['table_confirmed'] = bool(tables)
    if not tables:
        perception.previous=[]
        return [],round((time.monotonic()-started)*1000,1)
    items, _ = perception.infer(frame, store, 'all', prompt='discarded trash. litter.')
    generic_masks=[item['mask'] for item in items]
    references=TaughtCategories(samples)
    states=segment(frame,'all',prompt=STATE_PROMPT) if request.get('open_waste') or references.profiles else []
    hints=list(dict.fromkeys(e['hint'] for examples in samples.values() for e in examples))
    hinted=[]
    for i in range(0,len(hints),5):
        hinted.extend(segment(frame,'all',prompt='. '.join(hints[i:i+5])+'.'))
    evidence = []
    extras = []
    for prompt in comparison_prompts(kinds):
        for proposal in segment(frame,'all',prompt=prompt):
            evidence.append(proposal)
            label=' '+proposal['semantic_label'].lower().strip()+' '
            if (disposal_names([label],kinds) and overlaps_table(proposal,tables)
                    and not duplicate_mask(proposal['mask'],proposal['box'],items)
                    and not duplicate_mask(proposal['mask'],proposal['box'],extras)):
                extras.append(proposal)
    semantic_count=len(extras)
    if references.groups:
        for v in [*hinted,*states]:
            if (overlaps_table(v,tables) and not duplicate_mask(v['mask'],v['box'],items)
                    and not duplicate_mask(v['mask'],v['box'],extras)):
                extras.append(v)
    if extras:
        for index,(proposal,feature) in enumerate(zip(extras,perception.embed([v['crop'] for v in extras]))):
            if index>=semantic_count and not references.profiles and not references.lookup(feature,perception.cfg).get('id'):
                continue
            perception.counter+=1
            items.append(dict(proposal,id=perception.counter,feature=feature,name=proposal['semantic_label'],
                              product=None,match=dict(score=0),appearance_only=index>=semantic_count))
    if items:
        for item in items:
            mask = item['mask']
            labels = {' '+v['semantic_label'].lower().strip()+' ' for v in evidence
                      if same_instance(mask,v['mask'])}
            matched = disposal_names(labels,kinds)
            state_labels={' '+v['semantic_label'].lower().strip()+' ' for v in states if same_instance(mask,v['mask'])}
            sample=references.lookup(item['feature'],perception.cfg)
            alternate=references.alternate_view(perception,item,state_labels,sample)
            if alternate:sample=alternate
            example=sample.get('id')
            taught_hints={e['hint'] for e in samples.get(example,[])}
            localized=any(same_instance(mask,v['mask']) and any(
                set(h.split()) <= set(v['semantic_label'].lower().split()) for h in taught_hints if h!='object')
                for v in hinted) if example else False
            retained=[phrase for phrase in CONTRAST if phrase not in kinds
                      and any(' '+phrase+' ' in label for label in labels)]
            novel=waste_family(state_labels) if request.get('open_waste') and any(same_instance(mask,m) for m in generic_masks) else ''
            kind=example.partition('=')[0].strip() if example else ' / '.join(matched) or novel
            support = '' if overlaps_table(item,tables) else '图像中与桌面没有重叠'
            reason = support or (
                '识别到保留类别（'+' / '.join(retained)+'）' if retained and not localized and
                (not alternate or any(k in retained for k in ('phone','wallet','keys')))
                else '种类不明确或未列入清单' if not kind else '')
            if kind and not reason:
                item['name']=kind
            item['cleanup'] = dict(kind=kind or '疑似垃圾', reason=reason, on_table=not support,
                source='sample' if example else 'catalog' if matched else 'waste_state' if novel else '',
                state_labels=sorted(label.strip() for label in state_labels),
                labels=sorted(label.strip() for label in labels),retained=retained,sample=sample,
                sample_localized=localized)
    items=[v for v in items if not v.get('appearance_only') or v['cleanup']['sample'].get('id')]
    return items, round((time.monotonic()-started)*1000, 1)


def skip_reason(item):
    info=item.get('cleanup') or {}
    if not info.get('on_table'):
        return info.get('reason') or '未确认物品与桌面重叠'
    return info.get('reason','未确认是垃圾')


def candidates(items):
    ids = [v['id'] for v in items if not skip_reason(v)]
    random.shuffle(ids)
    return ids


def reobserve_target(perception,frame,store,chosen,scene_items):
    """Use the scan as an RGB prompt. Only a fresh, matched mask can supply grasp geometry."""
    from .point_selection import NoPointMask,reobserve
    from .selection import resolve_selected

    detail=dict(method='local_sam2',selected_id=chosen['id'],frame_id=frame['frame_id'])
    if chosen['mask'].shape!=frame['image'].shape[:2]:
        return None,dict(detail,reason='复核画面尺寸已变化，无法定位原目标')
    try:
        fresh=reobserve(perception,frame,store,chosen,box=chosen['box'])
    except NoPointMask:
        return None,dict(detail,reason='原目标位置未分出有效轮廓',candidates=[])
    selected,identity=resolve_selected(chosen,fresh)
    detail.update(identity)
    if selected is None:return None,detail

    # If two original objects overlap this new mask, check which one it belongs to.
    nearby=[item for item in scene_items if np.any(item['mask'] & selected['mask'])]
    if len(nearby)>1:
        origin,reverse=resolve_selected(fresh[0],nearby)
        detail['nearby']=reverse['candidates']
        if origin is None or reverse['detector_id']!=chosen['id']:
            return None,dict(detail,reason='局部轮廓与邻近物体重叠，无法唯一确认原目标')
    selected.update(cleanup=chosen['cleanup'],semantic_label=chosen.get('semantic_label',''))
    return selected,detail


def grasp_shape(perception,frame,selected):
    """Trash is a disposal decision. Recover the physical shape of this specific mask for planning."""
    if not selected.get('cleanup'):return ''
    mask=selected['mask']
    for proposal in perception.segmenter.proposals(frame,'all',prompt='bottle. beverage can.'):
        other=proposal['mask']
        words=set(proposal['semantic_label'].lower().split())
        if (words & {'bottle','can'} and
                np.count_nonzero(mask & other)>.5*max(np.count_nonzero(mask),np.count_nonzero(other))):
            return 'cylinder'
    return ''


def report(items, failures, picked):
    entries = []
    for item in items:
        ident = item['id']
        info = item.get('cleanup') or {}
        reason = failures.get(ident) or skip_reason(item)
        entries.append(dict(id=ident, name=info.get('kind','疑似垃圾'), reason=reason,on_table=bool(info.get('on_table')),
                            state='picked' if ident==picked else 'skipped' if reason else 'remaining'))
    left = [e for e in entries if e['state']!='picked' and e['on_table']]
    outside=sum(not e['on_table'] for e in entries)
    details = '；'.join(f"#{e['id']} {e['name']}"+(f"（{e['reason']}）" if e['reason'] else '') for e in left)
    outcome = f'已完成抓取 #{picked}' if picked is not None else '本次未抓取'
    text = f'桌面清理：{outcome}。本轮画面剩余 {len(left)} 项'+('：'+details if details else '。')
    if outside:text+=f'；另有 {outside} 项与桌面图像无重叠，已排除。'
    return dict(picked=picked, remaining=len(left),items=entries, text=text)
