"""反光缺深度只表示未观测，不以缺失比例代替物体定位与路径验证。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from autolife_shelf_grasp_306.bottle import object_points,DepthQualityError
from autolife_shelf_grasp_306.visual_registration import validate_surface


def sparse_frame():
    h,w=240,320;mask=np.zeros((h,w),bool);mask[30:210,60:260]=True
    yy,xx=np.indices((h,w));visible=mask&((xx//16+yy//16)%3==0)
    depth=np.zeros((h,w),np.uint16);depth[visible]=600
    K=np.array([[400.,0,160],[0,400.,120],[0,0,1]])
    intr=dict(fx=400.,fy=400.,ppx=160.,ppy=120.,model_id=0,coeffs=[0.]*5)
    frame=dict(image=np.zeros((h,w,3),np.uint8),depth=depth,K=K,D=np.zeros(5),
               frame_id=1,depth_id=1,intrinsics=intr,depth_intrinsics=dict(intr))
    return frame,mask,dict(depth_scale_m=.001,depth_range_m=[.2,1.8])


def test_sparse_registration_retains_only_real_depth_but_grasp_stays_dense():
    f,m,c=sparse_frame()
    with pytest.raises(DepthQualityError):object_points(f,m,np.eye(4),c)
    points,pixels,_=object_points(f,m,np.eye(4),c,False)
    assert len(points)>80 and np.allclose(points[:,2],.6)
    assert np.all(f['depth'][pixels[:,1],pixels[:,0]]==600)


@pytest.mark.parametrize('fault',['empty','misaligned','nan_intrinsics'])
def test_invalid_data_cannot_become_a_fake_correction(fault):
    f,m,c=sparse_frame()
    if fault=='empty':f['depth'][:]=0
    elif fault=='misaligned':f['depth_id']=2
    else:f['depth_intrinsics']['fx']+=10  # 不一致的相机内参仍拒绝。
    with pytest.raises(ValueError):object_points(f,m,np.eye(4),c,False)


def surface():
    x,y=np.meshgrid(np.linspace(-.5,.5,100),np.linspace(-.5,.5,100))
    return np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])


def test_visible_surface_verification_does_not_penalize_missing_depth():
    a=surface();sparse=a[(np.abs(a[:,0])>.45)&(np.abs(a[:,1])>.45)]
    d=np.eye(4);d[:3,3]=[.3,-.02,.01];b=sparse+d[:3,3]
    with pytest.raises(ValueError):validate_surface(a,b,d)
    assert validate_surface(a,b,d,True)==1.


def test_observed_wrong_background_still_fails_surface_verification():
    a=surface();b=a[::10]+[0,0,.12]
    with pytest.raises(ValueError):validate_surface(a,b,np.eye(4),True)


@pytest.mark.parametrize('shift,angle,fail',[(.35,0,None),(.03,45,None),(.35,0,'ik'),(.35,0,'path')])
def test_large_correction_requires_ik_and_complete_path(monkeypatch,tmp_path,shift,angle,fail):
    from scipy.spatial.transform import Rotation
    from autolife_shelf_grasp_306.visual_action_worker import VisualActions
    from autolife_shelf_grasp_306.pose_workspace_data import frame,compose
    from autolife_shelf_grasp_306.teaching_motion import compile_action
    from control_center_joint_control.joint_definitions import JOINT_LIMITS
    q={n:max(lo,min(hi,0.)) for n,(lo,hi) in JOINT_LIMITS.items()}
    ident='a'*64;f=frame('纠偏',q)
    f['visual_binding']=dict(reference=ident,side='right',mode='rigid',taught_object_pose=np.eye(4).tolist(),relative_tcp=np.eye(4).tolist())
    record=compile_action(compose('大偏移',[f],100),q)
    meta=dict(name='参照',object_pose=np.eye(4).tolist(),model_sha256='same',frame='root')
    d=np.eye(4);d[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix();d[0,3]=shift
    calls=[]
    class Guard:
        kin=NS(joint_q=q)
        def begin(self,values):pass
        def check(self,values):
            calls.append('path')
            if fail=='path':raise ValueError('path collision')
    class Solver:
        def __init__(self,*args):self.planner=NS(guard=Guard())
        def calculate(self,req):
            calls.append('ik');assert np.allclose(req['target'],d)
            if fail=='ik':raise ValueError('IK不可达')
            return dict(joints={},checked_states=1)
    monkeypatch.setattr('autolife_shelf_grasp_306.pose_ik_worker.PoseSolver',Solver)
    v=VisualActions.__new__(VisualActions);v.cfg={};v.run=tmp_path;v.solvers={}
    v.geometry=NS(root='root');v.model_hash=lambda:'same';v.check_mount=lambda *a:None
    v.robot=NS(refresh_measured=lambda:None,base_sample=lambda:[0,0,0]);v.stable=lambda *a:None
    v.sample=lambda:({'image':np.zeros((60,80,3),np.uint8)},dict(q),np.eye(4))
    v.store=NS(load=lambda _: (meta,{}),directory=tmp_path/'refs')
    v.observe=lambda *a,**kw:(d,dict(angle_resolution_deg=1.),None,None,None)
    cmd=dict(action='visual_correct',visual_id='test',record=record)
    if fail:
        with pytest.raises(ValueError,match='IK不可达|path collision'):v.handle(cmd)
        assert not (tmp_path/'visual_action_last_preview.json').exists()
    else:
        result=v.handle(cmd)
        assert calls[0]=='ik' and 'path' in calls
        assert result['diagnostics'][0]['range_policy']=='ik_and_full_path'
        assert result['checked_states']>1
