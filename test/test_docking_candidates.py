"""候选覆盖和拒绝证据；只模拟导航净空回复，不发布任何运动命令。"""
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import operation_docking as dock


def row(index,arm='left',yaw=0.):
    return dict(candidate=dict(index=index,x=float(index+1),y=0.,yaw=yaw),arm=arm,
                point=np.array([.5,0.,.7]),T=np.eye(4),score=float(index),margin_m=.1)


def context():
    return dict(pose=[0.,0.,0.],map='map',chain={'name':'table','clearance':.05},
                chain_sha256='a'*64,samples=[])


def replies(monkeypatch,accept):
    from autolife_shelf_grasp_306 import core
    original=core.atomic_json;queries=[]
    def write(path,value):
        original(path,value)
        if path.name=='operation_screen_request.json':
            queries.append(value)
            allowed=[c['index'] for c in value['candidates'] if accept(c)]
            original(path.with_name('operation_screen_result.json'),dict(token=value['token'],allowed=allowed,
                rejected=[dict(index=c['index'],reason='occupied') for c in value['candidates'] if c['index'] not in allowed]))
    monkeypatch.setattr(core,'atomic_json',write)
    return queries


def test_geometric_shortlist_does_not_hide_candidates_after_first_six():
    ctx=context()
    ctx['samples']=[dict(index=i,x=.1*(i+1),y=0.,yaw=0.) for i in range(10)]
    rows=dock.shortlist(ctx,np.eye(4),{'left':np.array([1.,0.,.3])},
                        {'left':(np.array([0.,0.,.3]),1.2)})
    assert len(rows)==10
    assert ctx['diagnostics']['geometry']['retained_rows']==10


def test_same_position_different_heading_survives_geometric_deduplication():
    ctx=context()
    ctx['samples']=[dict(index=i,x=.3,y=0.,yaw=yaw) for i,yaw in enumerate((0.,.5))]
    rows=dock.shortlist(ctx,np.eye(4),{'left':np.array([.8,0.,.3])},
                        {'left':(np.array([0.,0.,.3]),1.)})
    assert {r['candidate']['yaw'] for r in rows}=={0.,.5}


def test_current_xy_can_turn_in_place_but_unchanged_pose_is_skipped():
    ctx=context()
    ctx['samples']=[dict(index=i,x=0.,y=0.,yaw=yaw) for i,yaw in enumerate((0.,.5))]
    rows=dock.shortlist(ctx,np.eye(4),{'left':np.array([.6,0.,.3])},
                        {'left':(np.array([0.,0.,.3]),1.)})
    assert len(rows)==1 and rows[0]['candidate']['yaw']==.5
    assert ctx['diagnostics']['geometry']['current_position']==1


def test_navigation_replenishes_after_first_six_rejected(tmp_path,monkeypatch):
    queries=replies(monkeypatch,lambda c:c['x']>=7.)
    rows=[row(i) for i in range(9)];ctx=context()
    result=dock.navigation_filter(tmp_path,ctx,rows,lambda:False)
    assert result==rows[6:]
    assert [len(q['candidates']) for q in queries]==[6,3]
    diag=ctx['diagnostics']['navigation']
    assert diag['checked_poses']==9 and diag['complete']
    assert diag['batches'][0]['rejected'][0]['reason']=='occupied'


def test_both_arms_share_clearance_query_and_yaw_is_kept(tmp_path,monkeypatch):
    queries=replies(monkeypatch,lambda c:c['yaw']==.5)
    rows=[row(0,'left'),row(0,'right'),row(0,'left',.5),row(0,'right',.5)]
    result=dock.navigation_filter(tmp_path,context(),rows,lambda:False)
    assert result==rows[2:]
    assert len(queries)==1 and len(queries[0]['candidates'])==2


def test_navigation_timeout_does_not_claim_exhaustive_rejection(tmp_path,monkeypatch):
    ticks=iter(np.arange(0.,10.,.2))
    monkeypatch.setattr(dock.time,'monotonic',lambda:float(next(ticks)))
    monkeypatch.setattr(dock.time,'sleep',lambda _:None)
    ctx=context()
    with pytest.raises(ValueError,match='尚不能判断其余候选'):
        dock.navigation_filter(tmp_path,ctx,[row(i) for i in range(9)],lambda:False)
    diag=ctx['diagnostics']['navigation']
    assert not diag['complete'] and diag['status']=='budget_exhausted'
    assert diag['checked_poses']==0


def test_navigation_timeout_retains_already_allowed_candidates(tmp_path,monkeypatch):
    from autolife_shelf_grasp_306 import core
    original=core.atomic_json;queries=[];now=[0.]
    monkeypatch.setattr(dock.time,'monotonic',lambda:now[0])
    def sleep(seconds):now[0]+=.2
    monkeypatch.setattr(dock.time,'sleep',sleep)
    def write(path,value):
        original(path,value)
        if path.name=='operation_screen_request.json':
            queries.append(value)
            if len(queries)==1:
                original(path.with_name('operation_screen_result.json'),dict(token=value['token'],allowed=[0]))
    monkeypatch.setattr(core,'atomic_json',write)
    ctx=context();rows=[row(i) for i in range(9)]
    result=dock.navigation_filter(tmp_path,ctx,rows,lambda:False)
    assert result==rows[:1] and len(queries)==2
    diag=ctx['diagnostics']['navigation']
    assert diag['status']=='budget_exhausted' and not diag['complete']
    assert diag['checked_poses']==6 and diag['selected_rows']==1


def test_all_clearance_rejections_report_navigation_stage_not_ik(tmp_path,monkeypatch):
    replies(monkeypatch,lambda c:False)
    monkeypatch.setattr(dock,'context',lambda *a:context())
    monkeypatch.setattr(dock,'shortlist',lambda *a:[row(0)])
    monkeypatch.setattr(dock,'arm_envelope',lambda *a:(np.zeros(3),1.))
    pool=NS(fk=NS(fk=lambda *a:np.eye(4)),
            arm_config=lambda *a:({'arm':'left'},np.zeros(3)),
            native_planner=lambda *a:pytest.fail('导航拒绝的候选不能进入IK'))
    with pytest.raises(ValueError,match='未进行关键抓取位姿检查'):
        dock.grasp(tmp_path,{},pool,{},np.array([.7,0.,.7]),{},
                   {'auto_arm_selection':False,'arm':'left'},[0,0,0],lambda:False,lambda _:None,None)
    diag=json.loads((tmp_path/'operation_diagnostics.json').read_text())
    assert diag['stage']=='navigation' and diag['key_pose_checks']==[]
    assert diag['navigation']['status']=='all_rejected'


def test_key_pose_failure_is_saved_with_candidate_arm_and_reason(tmp_path,monkeypatch):
    from autolife_shelf_grasp_306 import reach_precheck
    replies(monkeypatch,lambda c:True)
    monkeypatch.setattr(dock,'context',lambda *a:context())
    monkeypatch.setattr(dock,'shortlist',lambda *a:[row(0)])
    monkeypatch.setattr(dock,'arm_envelope',lambda *a:(np.zeros(3),1.))
    monkeypatch.setattr(reach_precheck,'native_key_poses',lambda *a:dict(status='unresolved',reason='self collision'))
    pool=NS(fk=NS(fk=lambda *a:np.eye(4)),
            arm_config=lambda *a:({'arm':'left'},np.zeros(3)),native_planner=lambda *a:NS())
    with pytest.raises(ValueError,match='self collision'):
        dock.grasp(tmp_path,{},pool,{},np.array([.7,0.,.7]),{},
                   {'auto_arm_selection':False,'arm':'left'},[0,0,0],lambda:False,lambda _:None,None)
    diag=json.loads((tmp_path/'operation_diagnostics.json').read_text())
    assert diag['stage']=='key_poses'
    assert diag['key_pose_checks'][0]['arm']=='left'
    assert diag['key_pose_checks'][0]['reason']=='self collision'
