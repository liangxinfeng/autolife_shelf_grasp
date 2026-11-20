from types import SimpleNamespace

import numpy as np
import pytest
from autolife_shelf_grasp_306.cleanup_catalog import enroll, save_types
from autolife_shelf_grasp_306.cleanup_identity import STATE_PROMPT, TaughtCategories
from autolife_shelf_grasp_306.table_cleanup import inspect_scene, candidates, report


def test_distinct_views_keep_one_name_and_compete_against_other_categories():
    cfg=dict(match_threshold=.72,match_margin=.06)
    samples={'蓝包装':[dict(feature=[0,1,0],views=[[1,0,0]])], '白杯':[dict(feature=[0,0,1])]}
    assert TaughtCategories(samples).lookup([1,0,0],cfg)['id']=='蓝包装'
    samples['相似包装']=[dict(feature=[.999,.01,0])]
    assert TaughtCategories(samples).lookup([1,0,0],cfg)['id'] is None


@pytest.mark.parametrize('label,overlap,expected',[
    ('torn packaging',True,'废弃包装'), ('food scraps',True,'食物残渣'),
    ('broken fragments',True,'破损碎片'), ('crumpled paper sponge',True,''),
    ('plant',True,''), ('phone',True,''), ('food scraps',False,''), ('litter',True,'')])
def test_unregistered_object_needs_agreeing_regions(tmp_path,label,overlap,expected):
    mask=np.eye(6,dtype=bool)
    item=dict(id=1,mask=mask,box=[0,0,5,5],feature=[1,0],name='litter')
    def proposals(frame,mode,prompt):
        if prompt=='table. tabletop.':return [dict(box=item['box'])]
        assert prompt==STATE_PROMPT
        return [dict(mask=mask if overlap else ~mask,semantic_label=label)]
    p=SimpleNamespace(cfg={},infer=lambda *a,**k:([item],0),segmenter=SimpleNamespace(proposals=proposals))
    items,_=inspect_scene(p,{},SimpleNamespace(root=tmp_path/'data'),dict(types=[],open_waste=True))
    assert bool(candidates(items))==bool(expected)
    if expected:assert items[0]['name']==expected and items[0]['cleanup']['source']=='waste_state'


def test_learned_name_survives_detector_renaming(tmp_path):
    enroll(tmp_path,'白色试样',dict(feature=[1,0],suggested_name='cup'))
    mask=np.eye(6,dtype=bool)
    item=dict(id=9,mask=mask,box=[0,0,5,5],feature=[1,0],name='container',semantic_label='container')
    def proposals(frame,mode,prompt):
        return [dict(box=item['box'])] if prompt=='table. tabletop.' else []
    p=SimpleNamespace(cfg=dict(match_threshold=.72,match_margin=.06),
                      infer=lambda *a,**k:([item],0),segmenter=SimpleNamespace(proposals=proposals))
    items,_=inspect_scene(p,{},SimpleNamespace(root=tmp_path/'data'),dict(types=['白色试样'],open_waste=True))
    assert candidates(items)==[9] and items[0]['name']=='白色试样'
    assert report(items,{},9)['items'][0]['name']=='白色试样'


def test_changed_view_requires_local_appearance_and_competes_with_similar_classes(monkeypatch):
    import base64
    from autolife_shelf_grasp_306 import cleanup_identity as identity
    parts=np.eye(2,dtype='<f2')
    profile=dict(labels=['reusable mug'],width=2,parts=base64.b64encode(parts.tobytes()).decode())
    example=dict(feature=[0,1],views=[[.6,.8]],profile=profile)
    samples={'纸杯':[example]}
    monkeypatch.setattr(identity,'local_parts',lambda *args:parts)
    p=SimpleNamespace(cfg=dict(match_threshold=.72,match_margin=.06))
    def lookup(labels):
        refs=TaughtCategories(samples)
        item=dict(feature=[1,0],crop=None)
        return refs.alternate_view(p,item,labels,refs.lookup(item['feature'],p.cfg))
    assert lookup({' reusable mug '})['id']=='纸杯'
    assert lookup({' phone ',' reusable mug '}) is None
    assert lookup({' wrapper '}) is None
    samples['其他杯']=[example]
    assert lookup({' reusable mug '}) is None
