from types import SimpleNamespace

import numpy as np
import pytest

from autolife_shelf_grasp_306 import point_selection, table_cleanup


def item(ident, left, feature=(1., 0.)):
    mask=np.zeros((32,48),bool)
    mask[8:24,left:left+16]=True
    return dict(id=ident,name='杯子',cleanup=dict(kind='杯子'),semantic_label='cup',product=None,
                mask=mask,box=[left,8,left+16,24],feature=feature)


@pytest.mark.parametrize('left,feature,neighbor,expected',[
    (9,(1.,0.),None,True), (30,(1.,0.),None,False),
    (8,(0.,1.),None,False), (9,(1.,0.),9,False), (8,(1.,0.),28,True)])
def test_reobserve_only_selected_object(monkeypatch,left,feature,neighbor,expected):
    chosen=item(1,8)
    fresh=item(50,left,feature)
    fresh.pop('cleanup')
    frame=dict(frame_id=12,image=np.zeros((32,48,3),np.uint8))
    def reobserve(perception,seen,store,target,box):
        assert seen is frame and box==chosen['box'] and target is chosen
        return [fresh]
    monkeypatch.setattr(point_selection,'reobserve',reobserve)
    initial=[chosen] if neighbor is None else [chosen,item(2,neighbor)]
    selected,detail=table_cleanup.reobserve_target(None,frame,None,chosen,initial)
    assert detail['selected_id']==1 and detail['frame_id']==12
    assert (selected is not None)==expected
    if selected:
        assert selected['id']==1 and selected['mask'] is fresh['mask'] and selected['cleanup']==chosen['cleanup']
    else:assert detail['reason']


def test_missing_contour_is_skippable_but_backend_failure_surfaces(monkeypatch):
    chosen=item(1,8)
    def observe(*args,**kwargs):raise point_selection.NoPointMask('empty')
    monkeypatch.setattr(point_selection,'reobserve',observe)
    frame=dict(frame_id=3,image=np.zeros((32,48,3)))
    selected,detail=table_cleanup.reobserve_target(None,frame,None,chosen,[chosen])
    assert selected is None and detail['candidates']==[] and detail['reason']
    def broken(*args,**kwargs):raise RuntimeError('backend failed')
    monkeypatch.setattr(point_selection,'reobserve',broken)
    with pytest.raises(RuntimeError,match='backend failed'):
        table_cleanup.reobserve_target(None,frame,None,chosen,[chosen])
