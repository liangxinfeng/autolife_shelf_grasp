"""执行正式 worker 的移位生产分支及异常出口；不启动 ROS，不输出运动。"""
import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace as NS

import numpy as np
import pytest
from autolife_shelf_grasp_306 import worker,operation_docking
from autolife_shelf_grasp_306.mobile_reach import RepositionRequired


def request_handler(namespace):
    tree=ast.parse(inspect.getsource(worker.main))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Try) and
              any(isinstance(h.type,ast.Name) and h.type.id=='RepositionRequired' for h in n.handlers))
    node=copy.deepcopy(node)
    node.body=ast.parse('trigger()').body
    body=ast.unparse(node)
    exec('def request():\n'+textwrap.indent('plan="old"\n'+body+'\nreturn plan','    '),namespace)
    return namespace['request']


def environment(tmp_path):
    events=[]
    env=dict(__name__=worker.__name__,__package__=worker.__package__,
             RepositionRequired=RepositionRequired,CandidateUnavailable=worker.CandidateUnavailable,
             StationaryPathsExhausted=worker.StationaryPathsExhausted,
             run=tmp_path,ident=12,cmd={'action':'plan','request_id':12},time=worker.time,
             sys=worker.sys,atomic_json=worker.atomic_json,
             emit=lambda kind,**kw:events.append((kind,kw)))
    return env,events


@pytest.mark.parametrize('stale_ground',[False,True])
def test_first_grasp_real_selector_and_handler_keep_worker_alive(monkeypatch,tmp_path,stale_ground):
    # 真实 select_operation_dock：当前下蹲/颈部坐标不是单位阵，也不预先运行放置。
    G=np.eye(4);G[:3,3]=[.1,0,.7]
    T=np.eye(4);T[:3,3]=[.2,.03,.4]
    env,events=environment(tmp_path)
    env.update(q={'waist':20},T=T,fk=NS(fk=lambda *a:G),TORSO='torso',
        frame=dict(image=np.zeros((20,30,3)),K=np.eye(3),D=np.zeros(5)),
        cfg={'nested':{'value':1}},arm_pool=object(),center=[.7,0,.4],planned_product={},
        robot=NS(base_sample=lambda:[0,0,0]),observation_for_arm=None,
        reposition_proposal=worker.reposition_proposal,operation_docking=operation_docking)
    if stale_ground:env['ground']=np.eye(4)*99
    monkeypatch.setattr(operation_docking,'grasp',lambda *a,**kw:{'token':'test'})
    tree=ast.parse(inspect.getsource(worker))
    selector=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='select_operation_dock')
    source=ast.unparse(selector)+'\nselect_operation_dock()'
    exec('def trigger():\n'+textwrap.indent(source,'    '),env)
    request=request_handler(env)
    assert request() is None
    assert events[-1][0]=='base_reposition',events
    p=events[-1][1]['proposal']
    np.testing.assert_allclose(p['camera_to_base'],G@T)
    assert p['request_id']==12
    assert (tmp_path/'mobile_reach_proposal.json').exists()
    # 下一请求仍可处理；没有 fatal，也没有关闭相机/退出主循环。
    env['trigger']=lambda:env['emit']('next_request_ok')
    request()
    assert events[-1][0]=='next_request_ok'
    assert not any(kind=='fatal' for kind,_ in events)


@pytest.mark.parametrize('action',['plan','place','standalone_place'])
def test_request_metadata_is_owned_and_resume_discards_old_session(action):
    cmd=dict(action=action,request_id=4,parallel_prepare_token='p',_mobile_done='old',
             _operation_visited=[[1,2,3]],mobile_reach=True)
    cfg={'nested':[1]};source={'token':'new'};T=np.eye(4)
    p=worker.reposition_proposal(source,cfg,cmd,4,T)
    cfg['nested'][0]=9;cmd['_operation_visited'][0][0]=9;T[0,3]=9
    assert source=={'token':'new'} and p['camera_cfg']['nested']==[1]
    assert p['resume']['_operation_visited']==[[1,2,3]] and p['camera_to_base'][0][3]==0
    assert not {'request_id','parallel_prepare_token','_mobile_done'} & p['resume'].keys()


@pytest.mark.parametrize('bad',[None,[[1,2]],np.full((4,4),np.nan)])
def test_invalid_handoff_is_request_error_not_worker_exit(tmp_path,bad):
    env,events=environment(tmp_path)
    env['trigger']=lambda:worker.reposition_proposal({}, {}, {}, 12,bad)
    request_handler(env)()
    assert events[-1][0]=='error'
    assert not (tmp_path/'mobile_reach_proposal.json').exists()


def test_proposal_write_failure_does_not_close_vision(tmp_path):
    env,events=environment(tmp_path)
    def trigger():raise RepositionRequired({'token':'new'})
    def fail(*a):raise OSError('test disk error')
    env.update(trigger=trigger,atomic_json=fail)
    request=request_handler(env)
    assert request() is None
    assert events[-1][0]=='error' and '移位交接失败' in events[-1][1]['text']
    env['trigger']=lambda:env['emit']('next_request_ok')
    request()
    assert events[-1][0]=='next_request_ok'


def test_place_and_grasp_producers_both_pack_frame_before_raise():
    tree=ast.parse(inspect.getsource(worker.main))
    raises=[n.exc for n in ast.walk(tree) if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call)
            and isinstance(n.exc.func,ast.Name) and n.exc.func.id=='RepositionRequired']
    assert len(raises)==2
    assert all(isinstance(r.args[0],ast.Call) and r.args[0].func.id=='reposition_proposal' for r in raises)
