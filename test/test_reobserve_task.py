import numpy as np
import pytest
from types import SimpleNamespace as NS
from autolife_shelf_grasp_306 import mobile_reach as m,bottle,reach_precheck as reach


def identify(monkeypatch,items,request=None):
    def points(frame,mask,T,cfg,**kw):
        assert kw['dense'] is False
        return np.tile(mask,(100,1)),None,None
    monkeypatch.setattr(bottle,'object_points',points)
    chosen=dict(id=1,product=None,feature=[1.,0.],semantic_label='tissue paper',name='纸巾')
    return m.select_after_move(chosen,items,{},np.eye(4),{},
        dict(target_base=[.6,0,.7],base_anchor=[0,0,0]),[0,0,0],request=request)


def test_view_change_uses_semantics_and_new_depth(monkeypatch):
    v,d=identify(monkeypatch,[dict(id=8,mask=[.68,0,.7],feature=[0,1],semantic_label='white crumpled tissue')])
    assert v['id']==1 and d['position_error_m']==pytest.approx(.08)
    assert d['candidates'][0]['similarity']==0


def test_task_scoped_detection_tolerates_label_variants(monkeypatch):
    v,d=identify(monkeypatch,[dict(id=8,mask=[.7,0,.7],feature=[0,1])],{'semantic':True})
    assert v is not None


def test_unrelated_box_not_substituted(monkeypatch):
    v,d=identify(monkeypatch,[dict(mask=[.61,0,.7],feature=[0,1],semantic_label='tissue box')])
    assert v is None and d['candidates'] and d['reason']


def test_two_indistinguishable_objects_not_arbitrarily_chosen(monkeypatch):
    v,d=identify(monkeypatch,[dict(mask=[.61,0,.7],feature=[0,1]),dict(mask=[.615,0,.7],feature=[0,1])],{'semantic':True})
    assert v is None and '多个' in d['reason']


def test_gross_position_conflict_does_not_use_old_geometry(monkeypatch):
    v,d=identify(monkeypatch,[dict(mask=[1.4,0,.7],feature=[1,0])],{'semantic':True})
    assert v is None and d['candidates'][0]['position_error_m']>.7


@pytest.mark.parametrize('fails',[False,True])
def test_native_precheck_restores_full_planner_mode(monkeypatch,fails):
    planner=NS(_reach_quick_probe=False)
    def check(p,*args):
        assert p is planner and p._reach_quick_probe
        if fails:raise ValueError('no native solution')
        return {'status':'key_poses_passed'}
    monkeypatch.setattr(reach,'check_key_poses',check)
    if fails:
        with pytest.raises(ValueError):reach.native_key_poses(planner,{},[],[],{})
    else:assert reach.native_key_poses(planner,{},[],[],{})['status']=='key_poses_passed'
    assert planner._reach_quick_probe is False
