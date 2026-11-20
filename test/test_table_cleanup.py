import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from test_auto_grasp import qt,window
from autolife_shelf_grasp_306 import table_cleanup as cleanup
from autolife_shelf_grasp_306.cleanup_surface import overlaps_table
from autolife_shelf_grasp_306.cleanup_catalog import load_types,save_types,enroll,library
from autolife_shelf_grasp_306.task_flow_ui import FlowDialog


@pytest.mark.parametrize('box,expected',[
    ([0,0,11,11],True),([15,15,18,18],True),([5,5,35,35],True),
    ([30,10,40,15],False),([31,10,40,15],False),([56,56,58,58],True)])
def test_rgb_region_including_one_pixel_and_fully_occluded_support(box,expected):
    tables=[{'box':[10,10,30,30]},{'box':[55,55,60,60]}]
    assert overlaps_table({'box':box},tables)==expected


def test_scan_without_depth_or_joint_feedback(tmp_path):
    mask=np.ones((5,5),bool)
    item={'id':1,'mask':mask,'box':[2,2,7,7],'feature':[1,0]}
    def proposals(frame,mode,prompt):
        if prompt=='table. tabletop.':return [{'box':[6,6,10,10]}]
        return [dict(mask=mask,box=item['box'],semantic_label='crumpled paper')]
    p=SimpleNamespace(segmenter=SimpleNamespace(proposals=proposals),
                      cfg={},infer=lambda *a,**kw:([item],1))
    request=dict(types=['纸团'])
    items,_=cleanup.inspect_scene(p,{},SimpleNamespace(root=tmp_path/'data'),request)
    assert request['table_confirmed'] and not items[0]['cleanup']['reason']


@pytest.mark.parametrize('label,same,shape',[('water bottle',True,'cylinder'),('bottle',False,''),('bag',True,'')])
def test_shape_is_specific_to_selected_mask(label,same,shape):
    mask=np.eye(6,dtype=bool)
    p=SimpleNamespace(segmenter=SimpleNamespace(proposals=lambda *a,**kw:[dict(mask=mask if same else ~mask,semantic_label=label)]))
    assert cleanup.grasp_shape(p,{},dict(mask=mask,cleanup={'kind':'塑料包装'}))==shape


def test_no_table_does_not_run_garbage_detector(tmp_path):
    perception=SimpleNamespace(segmenter=SimpleNamespace(proposals=Mock(return_value=[])),infer=Mock(),cfg={})
    request=dict(types=['纸团'])
    store=SimpleNamespace(root=tmp_path/'data')
    items,_=cleanup.inspect_scene(perception,{},store,request)
    assert items==[] and not request['table_confirmed']
    perception.infer.assert_not_called()


def test_catalog_growth_never_creates_protected_only_query():
    from autolife_shelf_grasp_306.grounded_sam2 import normalize_prompt
    names=['sample '+str(i)+'x'*44 for i in range(17)]
    queries=list(cleanup.comparison_prompts(names))
    assert len(queries)>1
    for query in queries:
        phrases=normalize_prompt(query).rstrip('.').split('. ')
        assert set(phrases)&set(names) and set(cleanup.CONTRAST)<=set(phrases)
    assert all(sum(n in q.split('. ') for q in queries)==1 for n in names)


@pytest.mark.parametrize('feature,hint_position,protected,accepted',[
    ([1.,0.],'same','keys',True), ([1.,0.],'other','keys',False),
    ([0.,1.],'same','keys',False), ([0.,1.],'same','crumpled paper',True),
    ([0.,1.],'same','unknown',False)])
def test_joint_labels_and_taught_instance(tmp_path,feature,hint_position,protected,accepted):
    enroll(tmp_path,'我的纸团',dict(feature=[1.,0.],suggested_name='crumpled paper'))
    mask=np.zeros((10,10),bool);mask[:5,:5]=True
    item=dict(id=1,box=[0,0,5,5],mask=mask,feature=feature)
    other=dict(id=2,box=[5,5,10,10],mask=~mask,feature=[0.,1.])
    def proposals(frame,mode,prompt):
        if prompt=='table. tabletop.':return [dict(box=[0,0,10,10])]
        if prompt=='crumpled paper.':
            return [dict(item if hint_position=='same' else other,semantic_label='crumpled paper')]
        assert 'crumpled paper.' in prompt and 'keys.' in prompt
        return [dict(item,semantic_label=protected)]
    p=SimpleNamespace(segmenter=SimpleNamespace(proposals=proposals),
        cfg=dict(match_threshold=.72,match_margin=.06),infer=lambda *a,**kw:([item,other],1))
    items,_=cleanup.inspect_scene(p,{},SimpleNamespace(root=tmp_path/'data'),dict(types=['纸团','我的纸团']))
    assert (items[0]['id'] in cleanup.candidates(items))==accepted
    assert items[0]['cleanup']['sample_localized']==(feature[0]==1 and hint_position=='same')


def test_sample_add_append_delete_and_reload(tmp_path):
    item=dict(feature=[1.,0.],suggested_name='纸团')
    enroll(tmp_path,'自定义碎片',item);enroll(tmp_path,'自定义碎片',item)
    assert len(library(tmp_path)['samples']['自定义碎片'])==2
    save_types(tmp_path,['自定义碎片','烟头 = cigarette butt'])
    assert load_types(tmp_path)==['自定义碎片','烟头 = cigarette butt']
    save_types(tmp_path,[])
    assert not load_types(tmp_path) and not library(tmp_path)['samples']


def start(w,steps=None):
    w.receive_event(dict(event='ready',blockers=[]))
    w.workflow.start(dict(steps=steps or [dict(kind='cleanup')]),confirmed=True)


def seen(w,reasons,table=True):
    items=[dict(id=i+1,name='候选',box=[10*i,1,10*i+8,10],
                cleanup=dict(kind='纸团',on_table=True,reason=reason)) for i,reason in enumerate(reasons)]
    w.receive_event(dict(event='scene',request_id=w.request,shape=[200,400,3],items=items,
                         elapsed_ms=1,task_recognition=dict(cleanup=True,table_confirmed=table)))


@pytest.mark.parametrize('reasons,table,remaining', [([],False,None),([],True,0),(['外观不清'],True,1)])
def test_empty_or_uncertain_ends_without_motion(window,reasons,table,remaining):
    selection=window.run/'task_candidate_selection.json'
    selection.write_text('{"token":"previous","selected":42}')
    start(window);seen(window,reasons,table)
    result=window.workflow.cursor.metadata['cleanup_result']
    assert window.workflow.cursor.state=='completed' and result['remaining']==remaining
    assert [c['action'] for c in window.process.commands]==['scan_task']
    receipt=json.loads(selection.read_text())
    assert receipt['token']!='previous' and receipt['selected'] is None


def test_empty_catalog_does_not_place_previous_grasp(window,qt):
    save_types(window.root,[])
    held=window.run/'held_object.json';held.write_text('old receipt')
    start(window,[dict(kind='cleanup'),dict(kind='place')])
    qt.processEvents()
    assert window.workflow.cursor.state=='completed' and not window.process.commands
    assert held.read_text()=='old receipt'


def test_unreachable_candidates_tried_once(window):
    start(window);seen(window,['','','种类不清'])
    for _ in range(2):
        window.receive_event(dict(event='error',request_id=window.request,candidate_unavailable=True,dock_eligible=True,text='原地不可达'))
    ids=[c['id'] for c in window.process.commands if c['action']=='plan']
    assert set(ids)=={1,2} and len(ids)==2
    result=window.workflow.cursor.metadata['cleanup_result']
    assert result['remaining']==3 and not result['picked'] and window.workflow.cursor.state=='completed'
    assert all(c['stationary_only'] for c in window.process.commands if c['action']=='plan')


def test_pick_one_and_ignore_duplicate_receipt(window):
    start(window);seen(window,['',''])
    f=window.workflow
    f.phase='grasp';f.plan_id='first'
    selected=window.process.commands[-1]['id']
    assert f.motion_done(dict(status='completed',plan_id='first'))
    assert not f.motion_done(dict(status='completed',plan_id='first'))
    result=f.cursor.metadata['cleanup_result']
    assert result['remaining']==1 and result['picked']==selected


def test_unconfirmed_motion_never_reissues_grasp(window):
    start(window);seen(window,['',''])
    f=window.workflow;f.phase='grasp';f.plan_id='first';f.mark_dispatch()
    f.motion_done(dict(status='stopped',plan_id='first',effect_certainty='may_have_executed'))
    assert f.waiting and len([c for c in window.process.commands if c['action']=='plan'])==1


def test_editor_no_target_needed(window):
    d=FlowDialog(window);d.add('cleanup','')
    assert d.record()['steps'][0]['target']=='当前桌面'
    assert not d.table.cellWidget(0,1).isEditable()
    d.close()


def test_point_enrollment_rejects_background_expiry_and_wrong_instance(monkeypatch,tmp_path):
    import time
    from autolife_shelf_grasp_306 import cleanup_enrollment as ce
    session=ce.CleanupEnrollment()
    session.token='view';session.created=time.monotonic();session.frame={}
    session.tables=[{'box':[10,10,20,20]}]
    obj=dict(id=7,feature=[1.,0.],suggested_name='纸团',box=[0,0,10,10],mask=np.ones((10,10),bool))
    monkeypatch.setattr(ce,'segment',lambda *args:obj)
    def perform(action,**kw):
        return session.handle(dict(action=action,token='view',**kw),tmp_path,tmp_path,SimpleNamespace(cfg={}),None,None)
    with pytest.raises(ValueError,match='重叠'):perform('cleanup_point',x=.5,y=.5)
    assert session.item is None
    session.tables=[{'box':[9,9,19,19]}]
    perform('cleanup_point',x=.5,y=.5)
    with pytest.raises(ValueError):perform('cleanup_save',id=8,name='自定义碎片')
    assert '自定义碎片' not in load_types(tmp_path)
    session.created-=121
    with pytest.raises(ValueError,match='过期'):perform('cleanup_save',id=7,name='自定义碎片')


def test_random_order_is_used_only_for_confirmed_candidates(monkeypatch):
    monkeypatch.setattr(cleanup.random,'shuffle',lambda seq:seq.reverse())
    items=[dict(id=i,cleanup=dict(on_table=True,reason=r)) for i,r in enumerate(['','桌后',''])]
    assert cleanup.candidates(items)==[2,0]


@pytest.mark.parametrize('initial',['uncertain','empty'])
def test_other_catalog_object_joins_queue_even_if_generic_search_missed_it(tmp_path,initial):
    def proposal(left,label):
        mask=np.zeros((8,40),bool);mask[1:7,left:left+6]=True
        return dict(mask=mask,box=[left,1,left+6,7],crop=np.zeros((6,6,3)),semantic_label=label)
    unknown=dict(proposal(0,'litter'),id=1,feature=[1,0])
    paper=proposal(8,'crumpled paper')
    conflict=proposal(16,'crumpled paper keys')
    off_table=proposal(32,'crumpled paper')
    def proposals(frame,mode,prompt):
        if prompt=='table. tabletop.':return [dict(box=[0,0,24,8])]
        return [paper,dict(paper),conflict,off_table]
    perception=SimpleNamespace(segmenter=SimpleNamespace(proposals=proposals),counter=1,cfg={},
        embed=lambda crops:[[0,1] for crop in crops],
        infer=lambda *a,**kw:([unknown] if initial=='uncertain' else [],0))
    items,_=cleanup.inspect_scene(perception,{},SimpleNamespace(root=tmp_path/'data'),dict(types=['纸团']))
    ids=cleanup.candidates(items)
    assert len(ids)==1 and next(v['box'] for v in items if v['id']==ids[0])==paper['box']
    assert len(items)==2+int(initial=='uncertain')
    assert 'keys' in next(v['cleanup']['reason'] for v in items if v['box']==conflict['box'])


def test_uncertain_and_disappeared_targets_skip_to_next_once(window,monkeypatch):
    monkeypatch.setattr(cleanup.random,'shuffle',lambda seq:None)
    start(window);seen(window,['未确认是垃圾','','',''])
    assert window.process.commands[-1]['id']==2
    window.items=[v for v in window.items if v['id']!=3]
    request=window.request
    failure=dict(event='error',request_id=request,candidate_unavailable=True,text='局部轮廓未确认')
    window.receive_event(failure);window.workflow.worker_event(failure)
    assert [c['id'] for c in window.process.commands if c['action']=='plan']==[2,4]
    f=window.workflow;f.phase='grasp';f.plan_id='last'
    assert f.motion_done(dict(status='completed',plan_id='last'))
    result=f.cursor.metadata['cleanup_result']
    assert result['picked']==4 and [v['state'] for v in result['items']]==['skipped']*3+['picked']


def test_all_uncertain_advances_to_following_step(window,qt):
    start(window,[dict(kind='cleanup'),dict(kind='place'),dict(kind='cleanup')])
    seen(window,['未确认','识别到保留类别'])
    qt.processEvents();qt.processEvents()
    assert [c['action'] for c in window.process.commands]==['scan_task','scan_task']
    seen(window,[])
    assert window.workflow.cursor.state=='completed'


def test_incomplete_item_decision_skips_only_that_item():
    items=[dict(id=1),dict(id=2,cleanup=dict(on_table=True,kind='未知')),
           dict(id=3,cleanup=dict(on_table=True,reason='',kind='纸团'))]
    assert cleanup.candidates(items)==[3]
    receipt=cleanup.report(items,{},3)
    assert [v['state'] for v in receipt['items']]==['skipped','skipped','picked']
