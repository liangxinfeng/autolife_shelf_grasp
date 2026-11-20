"""Check geometric exclusions against actual robot FK; no ROS or motor I/O."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autolife_handeye_calibration_306.core import ARMS, BODY, TORSO, Kinematics
from autolife_handeye_calibration_306.verification_core import wrist_frame
from autolife_shelf_grasp_306.reach_envelope import WristEnvelope
from autolife_shelf_grasp_306.runtime_config import load_config


@pytest.fixture(params=['left','right'])
def model(request):
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root);side=request.param;cfg['arm']=side
    fk=Kinematics(cfg['urdf_path'])
    source=json.loads((root/'data/fixed_actions/manual_reset_pose.json').read_text())['joints_deg']
    source.update(zip(BODY,[-27.15,-55.35,-28.49,.30]))
    import xml.etree.ElementTree as ET
    xml=ET.parse(cfg['urdf_path'])
    names=BODY[2:]+ARMS[side]
    bounds=[]
    for name in names:
        limit=xml.find(f".//joint[@name='{name}']/limit")
        bounds.append([float(limit.get('lower')),float(limit.get('upper'))])
    planner=SimpleNamespace(fk=fk,cfg=cfg,names=names,bounds=np.array(bounds).T)
    return planner,source,WristEnvelope(planner,source)


def test_fk_poses_inside_waist_cells_survive_screen(model):
    planner,source,envelope=model
    cfg=planner.cfg;fk=planner.fk
    reference=np.linalg.inv(fk.fk('Link_Zero_Point',TORSO,source))
    rng=np.random.default_rng(367)
    for _ in range(160):
        q=dict(source)
        q.update(zip(planner.names,np.rad2deg(rng.uniform(*planner.bounds))))
        for name,width in zip(BODY[2:],cfg.get('vendor_grasp_waist_range_deg',[12.,8.])):
            q[name]=source[name]+rng.uniform(-width,width)
        T=reference@fk.fk('Link_Zero_Point',wrist_frame(cfg),q)
        tcp=rng.uniform(-.15,.15,size=3)
        point=T[:3,:3]@tcp+T[:3,3]
        # Independent FK supplies a witness; include the accepted solver errors.
        assert envelope.allows(point+[.0029,0.,0.],T[:3,:3]@Rotation.from_euler('x',1.9,degrees=True).as_matrix(),tcp)


def test_recorded_outside_wrist_avoids_seed_search(model):
    planner,_,envelope=model
    if planner.cfg['arm']=='right':
        R=np.array([[.5780410784,.0871557353,-.8113398730],
                    [-.0502329576,.9961946124,.0712246037],
                    [.8144600430,-.0004147453,.5802194984]])
        assert not envelope.allows([.68838336,-.33657432,-.27058734],R,[-.00197084,0.,-.22908317])
    else:
        assert not envelope.allows([2.,0.,0.],np.eye(3),[0.,0.,-.2])
