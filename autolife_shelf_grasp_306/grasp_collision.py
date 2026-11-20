"""普通抓取使用URDF/SRDF实体碰撞；不附加标定实验的标定板球。

沿用已验证模型、限位及碰撞对（包括双爪之间）；不删任何实体碰撞对。
手眼标定工具本身仍使用其原有包络，本模块只供普通抓取/复位使用。
"""
import numpy as np
from .resources import collision_model_path
from .reset_collision_scope import BOOT_HEAD_PAIR,validated_body_model
from autolife_handeye_calibration_306.auto_plan_worker import CollisionGuard as CalibrationGuard,GROUPS


# 起点颈胸重叠只在相对姿态静止时登记，头壳另按已核实模型与开机姿态识别。
STATIC_NECK_PAIR=frozenset(('Link_Waist_Yaw_to_Shoulder_Inner_0','Link_Neck_Pitch_to_Neck_Yaw_0'))
FIXED_JOINTS=tuple(GROUPS['leg_waist'])+tuple(GROUPS['neck'])
# 仅覆盖记录与反馈量化的微小差异；不能把有意的颈部/腰腿运动归为静止。
FIXED_EPS_DEG=.05

class CollisionGuard(CalibrationGuard):
    def __init__(self,cfg):
        # 使用抓取包自己的完整/碰撞模型；不修改原厂或VR模型。
        import tempfile,xml.etree.ElementTree as ET
        from pathlib import Path
        from openarmx_teleop_vr_306_v4.kinematics import IndependentArmKinematics
        model=collision_model_path(cfg)
        self.boot_head_model=validated_body_model(model)
        srdf=ET.parse(model.with_name('robot_v3_1.srdf'))
        for el in list(srdf.getroot().findall('disable_collisions')):
            if all('Wrist_Lower_to_Gripper' in el.get(k,'') for k in ('link1','link2')):srdf.getroot().remove(el)
        with tempfile.TemporaryDirectory(prefix='grasp_collision_') as tmp:
            path=Path(tmp)/'guard.srdf';srdf.write(path)
            self.kin=IndependentArmKinematics(str(model),str(path),collision_check=True,limit_margin_rad=0)
        if not self.kin.collision_available:raise ValueError('碰撞模型加载失败：'+self.kin.collision_error)
        self.side=cfg['arm'].title()
        self.frame=self.kin.model.getFrameId(f'Link_{self.side}_Wrist_Lower_to_Gripper')
        self.skip=(f'Link_{self.side}_Wrist_Lower_to_Gripper',f'Link_{self.side}_Wrist_Upper_to_Wrist_Lower')
        self.boxes=[];self.cache=set();self.exact_distance_calls=0
        for obj in self.kin.geom_model.geometryObjects:
            obj.geometry.computeLocalAABB();box=obj.geometry.aabb_local
            self.boxes.append((np.array(box.min_),np.array(box.max_)))

    def begin_fixed_neck(self,source,goal=None):
        """每条新路径重新绑定。涉及腰腿/颈部运动的复位仍使用完整碰撞检查。"""
        self.cache.clear();self.fixed_neck_source=None;self.static_neck_pairs=[]
        if goal is not None and any(abs(goal[n]-source[n])>FIXED_EPS_DEG for n in FIXED_JOINTS):return
        q=self.kin.q_from_feedback({g:[source[n] for n in names] for g,names in GROUPS.items()},clip=False)
        if not np.isfinite(q).all() or np.any(q<self.kin.lower-1e-6) or np.any(q>self.kin.upper+1e-6):return
        pairs=self.kin.collision_pairs(q)
        allowed={STATIC_NECK_PAIR}
        if self.boot_head_model and 15.<source['Joint_Neck_Pitch']<=40. and abs(source['Joint_Neck_Roll'])<=2. and abs(source['Joint_Neck_Yaw'])<=10.:
            allowed.add(BOOT_HEAD_PAIR)
        self.static_neck_pairs=[p for p in pairs if frozenset(p.split(' <-> ')) in allowed]
        self.boot_head_notes={p for p in self.static_neck_pairs if frozenset(p.split(' <-> '))==BOOT_HEAD_PAIR}
        if self.static_neck_pairs:self.fixed_neck_source={n:source[n] for n in FIXED_JOINTS}

    def blocking_pairs(self,pairs,values):
        source=getattr(self,'fixed_neck_source',None)
        if source is None or any(abs(values[n]-v)>FIXED_EPS_DEG for n,v in source.items()):return pairs
        # 不删URDF/SRDF碰撞对。只将本次路径中不动部件的既有重叠记为静态告警。
        return self.stationary_overlaps(pairs,values)

    def stationary_overlaps(self,pairs,values):
        # A common waist motion leaves the head and shoulder housing in the
        # same relative pose. Any commanded neck motion restores the head pair.
        same_neck=all(values[n]==self.fixed_neck_source[n] for n in GROUPS['neck'])
        return [p for p in pairs if p not in self.static_neck_pairs or
                (not same_neck and p in getattr(self,'boot_head_notes',()))]

    def check(self,values):
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('关节数值不是有限数')
        key=tuple(sorted(values.items()))
        if key in self.cache:return
        groups={g:[values[n] for n in names] for g,names in GROUPS.items()}
        q=self.kin.q_from_feedback(groups,clip=False)
        if not np.all(np.isfinite(q)) or np.any(q<self.kin.lower-1e-6) or np.any(q>self.kin.upper+1e-6):
            raise ValueError('姿态超出URDF限位')
        pairs=self.blocking_pairs(self.kin.collision_pairs(q),values)
        if pairs:raise ValueError('机器人自碰撞：'+'; '.join(pairs[:3]))
        self.cache.add(key)
