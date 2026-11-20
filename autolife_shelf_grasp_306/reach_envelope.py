"""带当前腰部限位的工作空间外包络。只可证明范围外，不证明路径可执行。"""
import itertools
import json
import xml.etree.ElementTree as ET
import numpy as np
from .mobile_reach import arm_envelope
from .resources import vendor_control_path


def shoulder_envelope(fk,q,tcp,side,cfg,step_deg=4.):
    """用实际可转腰角的肩部球并集代替向所有方向扩大的单个大球。

    网格之间加上由URDF链长推导的旋转弦长上界，避免漏掉格间可达区。
    膝踝保持当前值，反向下蹲/身体高度自然进入坐标变换。
    """
    from autolife_handeye_calibration_306.core import BODY,TORSO
    _,radius=arm_envelope(fk,q,tcp,side)
    child='Link_'+side.title()+'_Wrist_Lower_to_Gripper'
    chain=[]
    while fk.joints[child][2]!=TORSO:
        chain.append(child);child=fk.joints[child][2]
    shoulder_child=child
    # The next joint's displacement along the shoulder axis cannot swing.
    # Move the sphere centre onto that axis; only the transverse component
    # belongs in its radius. This also works for non-collinear robot models.
    joint=fk.joints[child]
    pivot=np.zeros(3)
    if chain and joint[1] in ('revolute','continuous'):
        axis=joint[4]/np.linalg.norm(joint[4])
        link=fk.joints[chain[-1]][3][:3,3]
        pivot=axis*np.dot(axis,link)
        radius+=float(np.linalg.norm(link-pivot)-np.linalg.norm(link))
    tree=ET.parse(cfg['urdf_path']).getroot()
    path=vendor_control_path(cfg)
    dead=json.loads(path.read_text())['MOTOR_DEAD_ZONE']
    axes=[];padding=0.
    for i,name in enumerate(BODY[2:],2):
        limit=tree.find(f"joint[@name='{name}']/limit")
        reach=float(cfg.get('vendor_grasp_waist_range_deg',[12.,8.])[i-2])
        if not 0<reach<=20:raise ValueError('抓取腰部参与范围必须在0～20度内')
        lo=max(np.rad2deg(float(limit.get('lower')))+dead['LEG_WAIST_LOWER_DEAD_ZONE'][i]+.5,q[name]-reach)
        hi=min(np.rad2deg(float(limit.get('upper')))-dead['LEG_WAIST_UPPER_DEAD_ZONE'][i]-.5,q[name]+reach)
        if not lo<=q[name]<=hi:raise ValueError('当前腰部位于原厂限位保护区，请先调整姿态')
        values=np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)/step_deg))+1));axes.append(values)
        # 从该旋转轴到肩部的链长始终不小于实际旋转半径。
        child=shoulder_child;lever=float(np.linalg.norm(pivot))
        while fk.joints[child][0]!=name:
            lever+=float(np.linalg.norm(fk.joints[child][3][:3,3]));child=fk.joints[child][2]
        padding+=2*lever*np.sin(np.deg2rad(values[1]-values[0])/4)
    inverse=np.linalg.inv(fk.fk('Link_Zero_Point',TORSO,q));positions=[]
    for pitch,yaw in itertools.product(*axes):
        moved={**q,BODY[2]:float(pitch),BODY[3]:float(yaw)}
        pose=inverse@fk.fk('Link_Zero_Point',shoulder_child,moved)
        positions.append(pose[:3,3]+pose[:3,:3]@pivot)
    return np.array(positions),float(radius+padding)


def margin(positions,radius,point):
    return float(radius-np.linalg.norm(positions-np.asarray(point),axis=1).min())


class WristEnvelope:
    """Necessary reach conditions including the last two directed wrist links."""
    def __init__(self,planner,source):
        from scipy.spatial.transform import Rotation
        from autolife_handeye_calibration_306.verification_core import wrist_frame
        # Reuse a fine waist grid for all candidate wrists in this pose. The
        # chord allowance still covers every angle between grid points.
        self.shoulders,self.radius=shoulder_envelope(
            planner.fk,source,np.zeros(3),planner.cfg['arm'],planner.cfg,step_deg=1.)
        self.middle=self.shoulders.mean(axis=0)
        child=wrist_frame(planner.cfg);rot=np.eye(3)[None];points=np.zeros((1,3))
        length=0.;angles=[]
        for _ in range(2):
            name,kind,parent,origin,axis=planner.fk.joints[child]
            if kind!='revolute' or name not in planner.names:break
            lo,hi=planner.bounds[:,planner.names.index(name)]
            grid=np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)/np.deg2rad(5.)))+1))
            angles.append(grid[1]-grid[0]);length+=np.linalg.norm(origin[:3,3])
            back=Rotation.from_rotvec(-grid[:,None]*axis/np.linalg.norm(axis)).as_matrix()@origin[:3,:3].T
            rot=rot[:,None]@back
            points=(points[:,None,:]-(rot@origin[:3,3])).reshape(-1,3)
            rot=rot.reshape(-1,3,3);child=parent
        # The next axial component is invariant under its own rotation.
        # Project it exactly, leaving its transverse component in the sphere.
        name,kind,_,origin,axis=planner.fk.joints[child]
        reduction=0.
        if kind in ('revolute','continuous') and name in planner.names:
            vec=origin[:3,:3].T@origin[:3,3];axis=axis/np.linalg.norm(axis)
            axial=axis*(axis@vec)
            points=points-rot@axial
            reduction=np.linalg.norm(vec)-np.linalg.norm(vec-axial)
            axial_len=np.linalg.norm(axial)
        else:axial_len=0.
        self.offsets=points
        # A grid cell's largest deviation occurs at most half a step from a
        # sample. Each angular perturbation can swing the whole distal length.
        self.length=length+axial_len
        self.proximal_radius=self.radius-length-reduction+sum(2*self.length*np.sin(step/4) for step in angles)

    def allows(self,center,rotation,tcp):
        from .planning_control import VENDOR_IK_POSITION_TOLERANCE_M
        wrist=np.asarray(center)-rotation@tcp
        angular=2*np.sin(np.deg2rad(1.))
        tolerance=VENDOR_IK_POSITION_TOLERANCE_M+angular*np.linalg.norm(tcp)
        if margin(self.shoulders,self.radius,wrist)<-tolerance:return False
        direction=wrist-self.middle;distance=np.linalg.norm(direction)
        if distance<1e-9:return True
        direction/=distance
        # A separating plane outside every shoulder sphere excludes the pose
        # for ALL permitted distal angles, including positions between samples.
        nearest=wrist@direction+np.min(self.offsets@(rotation.T@direction))
        return nearest <= np.max(self.shoulders@direction)+self.proximal_radius+tolerance+angular*self.length+1e-9


def current_margin(fk,q,tcp,side,cfg,point):
    """大余量目标只做粗格；临界目标细分，避免格间外扩把不可达点误包进去。"""
    value=None
    for step in (4.,1.,.5):
        positions,radius=shoulder_envelope(fk,q,tcp,side,cfg,step)
        value=margin(positions,radius,point)
        if value<0 or value>.03:break
    return value
