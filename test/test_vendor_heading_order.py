"""Use the stationary planning budget on distinct allowed headings early."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from autolife_shelf_grasp_306.object_extent import coarse_yaws,directional_yaws
from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner


def routes(yaws=None, preferred=None, hint=None):
    calls=[]
    planner=SimpleNamespace(cfg={'arm':'right'},context=SimpleNamespace(set_stage=lambda *a,**k:calls.append(k)),
                            preferred_grasp_yaw=preferred,_active_reach_candidate=hint)
    result=list(VendorGraspPlanner.candidates(planner,yaws))
    assert [r['candidate'] for r in calls]==list(range(1,len(result)+1))
    assert all(r['total_candidates']==len(result) for r in calls)
    return result


def test_broad_pass_checks_distinct_headings_without_losing_paths():
    result=routes()
    assert result[0]==(0.,False,'joint_direct')
    assert {-60.,-35.,0.,35.,60.}<=set(row[0] for row in result[:6])
    assert all(r[2]=='joint_direct' for r in result[:6])
    admitted={0.,-10.,10.,-20.,20.,-35.,35.,-50.,50.,-60.,60.}
    profiles={'joint_direct','joint_from_grasp','joint_elbow_positive','joint_elbow_negative','direct','outward_8'}
    assert set(result)=={(yaw,False,p) for yaw in admitted for p in profiles}
    assert len(result)==len(set(result))


def test_verified_route_leads_without_admitting_an_unmeasured_heading():
    admitted=[5.,8.,12.]
    assert routes(admitted,8.,{'yaw_offset_deg':12.})[0]==(12.,False,'joint_prechecked')
    result=routes(admitted,8.,{'yaw_offset_deg':-35.})
    assert result[0]==(8.,False,'joint_direct')
    assert {v[0] for v in result}==set(admitted)
    assert len(result)==6*len(admitted)
    assert routes([])==[]


def test_measured_long_axis_gets_a_complete_route_early():
    r=Rotation.from_euler('z',42,degrees=True).as_matrix()
    axes=r@np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    obj=dict(bbox_axes_torso=axes.tolist(),bbox_extents_m=[.03,.065,.08])
    cfg=dict(up_torso=[0.,0.,1.],approach_yaw_deg=0.,gripper_max_width_m=.09)
    admitted=list(directional_yaws(obj,cfg))
    assert np.allclose(admitted[:2],[0.,42.])
    first=routes(admitted)[:2]
    assert np.allclose([r[0] for r in first],[0.,42.])


def test_nearly_square_tissue_keeps_measured_yaw_for_fine_search():
    # The recorded tissue returned 0 and -0.0045549 degrees as its first
    # full IK attempts on both arms. Include the reflected case as well.
    for measured in (-.00455490766114508,.00455490766114508):
        admitted=[0.,measured,-25.,25.]
        assert coarse_yaws(admitted)==[0.,-25.,25.]
        path=routes(admitted)
        assert [p[0] for p in path[:3]]==[0.,-25.,25.]
        assert sum(p[0]==measured for p in path)==6
        assert routes(admitted,hint=dict(yaw_offset_deg=measured))[0]==(measured,False,'joint_prechecked')
        assert coarse_yaws([measured])==[measured]


def test_antiparallel_jaws_keep_both_directed_approaches_in_coarse_search():
    admitted=[0.,-89.9,90.]
    assert coarse_yaws(admitted)==[0.,-89.9,90.]
    assert coarse_yaws(admitted,(90.,))==[90.,0.,-89.9]
    assert [p[0] for p in routes(admitted)[:3]]==[0.,-89.9,90.]
    assert {yaw for yaw,_,_ in routes(admitted)}==set(admitted)
