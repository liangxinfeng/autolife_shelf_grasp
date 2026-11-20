import copy
import math
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306 import operation_docking as d


def test_se2_reprojection_roundtrip():
    before=[1.,2.,.8];after=[.2,3.,-.4];point=np.array([.8,-.2,.6])
    T=np.linalg.inv(d.se2(after))@d.se2(before)
    moved=(T@np.r_[point,1.])[:3]
    assert np.allclose(moved,d.map_point(point,before,after))
    assert np.allclose((np.linalg.inv(T)@np.r_[moved,1.])[:3],point)


def test_product_plane_and_axes_are_transformed_not_dimensions():
    obj=dict(height_m=.02,bbox_axes_torso=np.eye(3).tolist(),
        low_object_support=dict(normal=[0,0,1],offset=.3,center=[.6,0,.3]))
    T=d.se2([.2,.1,.8]);T[2,3]=.1
    moved=d.transform_product(obj,T)
    assert moved['height_m']==.02 and obj['low_object_support']['offset']==.3
    assert np.allclose(moved['bbox_axes_torso'],T[:3,:3])
    p=moved['low_object_support'];assert np.dot(p['center'],p['normal'])==pytest.approx(p['offset'])


def test_shortlist_uses_actual_height_and_avoids_boundary():
    ctx={'pose':[0,0,0],'samples':[dict(x=x,y=0.,yaw=0.,index=i) for i,x in enumerate([0,.04,.20,.30])]}
    a=d.shortlist(ctx,np.eye(4),{'right':[.9,0,.3]},{'right':(np.array([0,0,.3]),.8)})
    assert a and all(r['margin_m']>=.06 for r in a)
    assert a[0]['candidate']['x']==.30
    ground=np.eye(4);ground[2,3]=-.6
    b=d.shortlist(ctx,ground,{'right':[.9,0,1.3]},{'right':(np.array([0,0,.3]),.8)})
    assert b==[]  # 水平导航无法修正垂直不可达。


def test_rotated_table_support_keeps_holes():
    from autolife_shelf_grasp_306.placement_support import pack_support,unpack_support
    grid=np.ones((40,40),np.uint8);grid[15:25,15:25]=0
    table=dict(minimum=[0,0,.3],maximum=[.2,.2,.3],coef=[0,0,.3],
               footprint_support=pack_support(grid,[0,0],.005))
    transformed=d.transform_table(table,d.se2([.1,-.1,.3]))
    new,origin,res=unpack_support(transformed['footprint_support'])
    center=(d.se2([.1,-.1,.3])@np.array([.10,.10,.3,1]))[:2]
    x,y=np.floor((center-origin)/res).astype(int)
    assert new[y,x]==0
    assert transformed['coef'][2]==pytest.approx(.3)


def test_protocol_rejects_old_translation_and_expired_goal():
    with pytest.raises(ValueError):d.validate({'route_type':'translate'},'x'*32)
    row={'candidate':{'x':1.,'y':2.,'yaw':0.,'index':4},'arm':'right','score':1.,'margin_m':.1}
    ctx={'map':'map','chain':{'name':'桌子'},'chain_sha256':'a'*64,'pose':[0,0,0]}
    p=d.package('grasp',ctx,row,{},[0,0,0],[.8,0,.7],.5,13)
    assert d.validate(p,p['token']) is p
    with pytest.raises(ValueError):d.validate(p,p['token'],now=p['created']+91)


def test_navigation_result_does_not_advance_outer_task(tmp_path):
    from autolife_shelf_grasp_306.mobile_reach_ui import MobileReach
    state=NS(pending={'token':'a'},stage='moving',preparation=NS(token='prep'),
             owner=NS(run=tmp_path,navigation=NS(inflight=True),status=NS(setText=lambda s:None)))
    assert MobileReach.navigation_event(state,{'event':'nav_done','success':True,'run_id':'a','leg':'operation_dock'})
    assert state.stage=='return' and not state.owner.navigation.inflight
    assert (tmp_path/'mobile_reach_result.json').exists()
    assert not MobileReach.navigation_event(state,{'event':'nav_done','run_id':'different'})


def test_placement_completion_does_not_send_prepare():
    from autolife_shelf_grasp_306.app import Window
    events=[]
    owner=NS(live=lambda:events.append('live'),workflow=NS(active=False,pending=False),
             status=NS(setText=lambda s:events.append(s)),controls=lambda:None,
             send=lambda *a,**k:pytest.fail('放置后不应发送复位'))
    Window.finish_placement(owner,{'status':'placed'})
    assert '不自动复位' in events[-1]


def test_withdraw_completion_allows_waist_to_hold_landing_branch():
    from test_vendor_place import fixture
    from autolife_shelf_grasp_306.vendor_place_trajectory import build_blocks,validate_blocks
    from autolife_handeye_calibration_306.core import BODY
    p=fixture();p['placement_completion']='withdraw_hold'
    p['segments'][-1]['waypoints'][-1][BODY[2]]=.2
    p['vendor_blocks']=build_blocks(p,p['config']);validate_blocks(p)
    p['segments'][-1]['waypoints'][-1][BODY[0]]=1
    p['vendor_blocks']=build_blocks(p,p['config'])
    with pytest.raises(ValueError):validate_blocks(p)


def test_navigation_screen_does_not_turn_unknown_into_free(tmp_path,monkeypatch):
    import threading,json
    from autolife_shelf_grasp_306.core import atomic_json
    def reply():
        while not (tmp_path/'operation_screen_request.json').exists():time.sleep(.005)
        query=json.loads((tmp_path/'operation_screen_request.json').read_text())
        atomic_json(tmp_path/'operation_screen_result.json',{'token':query['token'],'allowed':[1]})
    thread=threading.Thread(target=reply);thread.start()
    rows=[{'candidate':{'index':i}} for i in (1,2)]
    selected=d.navigation_filter(tmp_path,{'map':'map','chain':{'clearance':.05}},rows,lambda:False)
    thread.join();assert selected==[rows[1]]
    assert not (tmp_path/'mobile_reach_request.json').exists()
