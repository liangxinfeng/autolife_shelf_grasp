"""Arm ownership and provisional tool geometry; no hardware operations."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def arm_name(value='left'):
    if value not in ('left','right'):raise ValueError('无效抓取手臂：'+str(value))
    return value


def gripper_name(arm='left'):
    return 'Joint_'+arm_name(arm).title()+'_Gripper'


def plan_arm(plan):
    arm=arm_name(plan.get('config',{}).get('arm','left'))
    if plan.get('arm',arm)!=arm:raise ValueError('路径与手臂归属不一致')
    return arm


def require_same_arm(plan,record):
    arm=plan_arm(plan)
    if arm_name(record.get('arm','left'))!=arm:raise ValueError('持物记录与路径手臂不一致')
    return arm


def gripper_targets(arm,angle,current,both=False):
    """Gripper driver accepts both fields. Inactive jaw holds measured position."""
    selected=gripper_name(arm)
    return {gripper_name(s):float(angle if both or gripper_name(s)==selected else current[gripper_name(s)])
            for s in ('left','right')}


def tool_geometry(cfg,calibration,fk,arm):
    """Share camera extrinsics, NOT a torso-space mirror of the left arm.

    For identical factory grippers only, transfer the measured local TCP through
    their URDF mounting frames. This is explicitly provisional, not a measured
    right-hand calibration. Robot and URDF identity are still validated.
    """
    from .grasp_geometry import load_geometry,is_nominal
    from autolife_handeye_calibration_306.verification_core import wrist_frame
    if is_nominal(calibration):
        from .pose_geometry import ModelGeometry
        arm=arm_name(arm);result=copy.deepcopy(cfg);result['arm']=arm
        tool=ModelGeometry(cfg['urdf_path']).tool_transform(arm)
        result['T_wrist_tool_quat_xyzw']=Rotation.from_matrix(tool[:3,:3]).as_quat().tolist()
        result['tool_geometry']={'arm':arm,'source_arm':arm,'provisional':True,
            'method':'urdf_nominal_tcp','camera_source':'urdf',
            'tcp_m':tool[:3,3].tolist(),'independent_tool_calibration':False}
        return result,tool[:3,3].copy()
    arm=arm_name(arm);source=arm_name(calibration.get('config',{}).get('arm','left'))
    tcp=np.asarray(load_geometry(calibration,fk,cfg['tcp_offset_mm'],robot_id=cfg.get('robot_id')),float)
    result=copy.deepcopy(cfg);result['arm']=arm
    provisional=arm!=source
    if provisional:
        zero={j[0]:0. for j in fk.joints.values()}
        def mount(side):return fk.fk(wrist_frame({'arm':side}),'Link_'+side.title()+'_Gripper',zero)
        transfer=mount(arm)@np.linalg.inv(mount(source))
        # This fallback is only for homologous factory grippers, not arbitrary tools.
        if (np.linalg.norm(transfer[:3,3])>.005 or
                Rotation.from_matrix(transfer[:3,:3]).magnitude()>np.deg2rad(3)):
            raise ValueError('两侧夹爪安装几何不一致，无法复用夹取中心')
        tcp=(transfer@np.r_[tcp,1.])[:3]
        tool=transfer[:3,:3]@Rotation.from_quat(cfg['T_wrist_tool_quat_xyzw']).as_matrix()
        result['T_wrist_tool_quat_xyzw']=Rotation.from_matrix(tool).as_quat().tolist()
        # The historical 15mm torso-Y trim was measured on the left hand only.
        # It is not a camera extrinsic and must not silently bias the right hand.
        result['target_offset_torso_mm']=list(cfg.get(arm+'_target_offset_torso_mm',[0.,0.,0.]))
    result['tool_geometry']={'arm':arm,'source_arm':source,'provisional':provisional,
        'method':'urdf_gripper_mount_transfer' if provisional else 'handeye_measured_tcp',
        'camera_source':'shared_head_calibration','tcp_m':tcp.tolist(),
        'independent_tool_calibration':not provisional}
    return result,tcp


def rank_arms(fk,point,preferred='left'):
    from autolife_handeye_calibration_306.core import TORSO
    point=np.asarray(point,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('物体三维位置无效')
    distances={}
    for side in ('left','right'):
        child='Link_'+side.title()+'_Wrist_Lower_to_Gripper'
        while fk.joints[child][2]!=TORSO:child=fk.joints[child][2]
        shoulder=fk.joints[child][3][:3,3]
        distances[side]=float(np.linalg.norm(point-shoulder))
    return sorted(distances,key=lambda s:(round(distances[s],4),s!=preferred))


class ArmPlannerPool:
    def __init__(self,cfg,calibration,fk,factory,calibration_profiles=None):
        self.base=copy.deepcopy(cfg);self.calibration=calibration;self.fk=fk;self.factory=factory
        self.planners={};self.attempts=[];self.diagnostics=[];self.native_planners={};self.place_planners={}
        self.calibration_profiles=calibration_profiles
        self.warm_attempted=set();self.reach_diagnostics=[]

    def close(self):
        for _,planner in self.native_planners.values():planner.close()
        self.native_planners.clear()
        for _,planner in self.place_planners.values():planner.close()
        self.place_planners.clear()

    def for_placement(self,cfg,stopped,on_status):
        from .vendor_place_planning import VendorPlacePlanner
        from .planner_cache import cache_key
        from .planning_control import PlanningControl
        side=arm_name(cfg['arm']);key=cache_key(cfg);cached=self.place_planners.get(side)
        context=PlanningControl(stopped,on_status)
        if cached is None or cached[0]!=key:
            if cached:cached[1].close()
            planner=VendorPlacePlanner(copy.deepcopy(cfg),context)
            self.place_planners[side]=(key,planner)
        else:planner=cached[1];planner.cfg=copy.deepcopy(cfg);planner.context=context
        from .planner_cache import transfer_idle_solver
        transfer_idle_solver(planner,self.native_planners.get(side),key)
        return planner

    def for_config(self,cfg):
        side=arm_name(cfg['arm'])
        if side not in self.planners:self.planners[side]=self.factory(copy.deepcopy(cfg))
        planner=self.planners[side];planner.cfg=copy.deepcopy(cfg)
        return planner

    def arm_config(self,side,current_cfg):
        """预加载与正式规划共用配置；动态复位/补偿始终取本次值。"""
        from .target_offsets import checked_profiles
        profiles=(checked_profiles(current_cfg['target_offsets_by_arm_torso_mm'])
                  if 'target_offsets_by_arm_torso_mm' in current_cfg else None)
        profile_cfg,profile_data=(self.base,self.calibration)
        if self.calibration_profiles is not None:
            profile_cfg,profile_data,_=self.calibration_profiles.for_arm(side)
        # 标定快照只提供几何信息，不能把启动时的旧复位目标带回来。
        # 同时删除已取消的引用，避免沿用上一次规划的保存姿态。
        profile_cfg=copy.deepcopy(profile_cfg)
        for key in ('initial_pose_deg','workspace_reset_joints','workspace_reset_path',
                    'workspace_reset_sha256','workspace_pose_speed_percent','grasp_speed_percent'):
            profile_cfg.pop(key,None)
            if key in current_cfg:profile_cfg[key]=copy.deepcopy(current_cfg[key])
        config,tcp=tool_geometry(profile_cfg,profile_data,self.fk,side)
        if self.calibration_profiles is not None:
            config['tool_geometry'].update(camera_source=config['tool_geometry'].get('camera_source','arm_calibration_profile'),
                calibration_profile=config['calibration_profile'])
        if profiles is not None:
            config['target_offset_torso_mm']=list(profiles[side])
            config['target_offsets_by_arm_torso_mm']=copy.deepcopy(profiles)
        config['up_torso']=list(current_cfg['up_torso'])
        if 'semantic_prompt_overrides' in current_cfg:
            config['semantic_prompt_overrides']=copy.deepcopy(current_cfg['semantic_prompt_overrides'])
        return config,tcp

    def native_planner(self,config,context):
        side=arm_name(config['arm'])
        from .vendor_grasp_planning import VendorGraspPlanner
        from .planner_cache import cache_key
        key=cache_key(config);cached=self.native_planners.get(side)
        if cached is None or cached[0]!=key:
            if cached:cached[1].close()
            planner=VendorGraspPlanner(config,context=context)
            self.native_planners[side]=(key,planner)
        else:
            planner=cached[1];planner.cfg=copy.deepcopy(config);planner.context=context
        from .planner_cache import transfer_idle_solver
        transfer_idle_solver(planner,self.place_planners.get(side),key)
        planner.preferred_grasp_yaw=None  # 预判方向只属于本次姿态与物体，换请求不沿用。
        return planner

    def warm_next(self,current_cfg,stopped=lambda:False):
        """空闲时每次只预加载一侧纯IK；有新请求立即取消，不访问硬件。

        不生成路径、不保存关节起点。正式规划仍重新读取姿态、配置固定
        关节并核对FK；预加载失败不阻止任务，正式请求会正常重新加载。
        """
        if self.base.get('grasp_backend')!='vendor_ik_trajectory':return None
        order=(self.base.get('arm','left'),)
        if self.base.get('auto_arm_selection',True):order+=('right' if order[0]=='left' else 'left',)
        side=next((v for v in order if v not in self.warm_attempted),None)
        if side is None:return None
        from .planning_control import PlanningControl,PlanningCancelled
        from .vendor_ik_process import VendorIKProcess
        from .vendor_grasp_planning import BASE
        from autolife_handeye_calibration_306.core import BODY
        from autolife_handeye_calibration_306.verification_core import wrist_frame
        import time
        start=time.monotonic();planner=None
        try:
            context=PlanningControl(stopped,timeout_sec=20.)
            context.checkpoint()
            config,_=self.arm_config(side,current_cfg)
            planner=self.native_planner(config,context)
            context.checkpoint()
            if planner.solver is None:
                planner.solver=VendorIKProcess(config,config['urdf_path'],BASE,wrist_frame(config),
                    {n:0. for n in BODY[:2]},checkpoint=context.checkpoint)
            self.warm_attempted.add(side)
            return dict(arm=side,status='ready',seconds=time.monotonic()-start)
        except PlanningCancelled:
            if planner is not None:planner.close()
            return None
        except Exception as exc:
            if planner is not None:planner.close()
            self.warm_attempted.add(side)
            return dict(arm=side,status='deferred',reason=str(exc),seconds=time.monotonic()-start)

    def plan(self,source,point,product,current_cfg,stopped=lambda:False,on_status=lambda text:None,observation_for_arm=None,on_reach=lambda result:None,reach_hint=None):
        from .core import corrected_target
        from .planning_control import PlanningControl,PlanningTimeout,PlanningCancelled
        from .reach_precheck import screen,STATIONARY_SECONDS
        import time
        self.attempts=[];self.diagnostics=[];self.reach_diagnostics=[]
        native=self.base.get('grasp_backend')=='vendor_ik_trajectory'
        context=PlanningControl(stopped,timeout_sec=STATIONARY_SECONDS) if native else None
        order=rank_arms(self.fk,point,self.base.get('arm','left')) if self.base.get('auto_arm_selection',True) else [self.base['arm']]
        if reach_hint:order.sort(key=lambda side:side!=reach_hint.get('arm'))
        if native:
            on_status('先检查当前高度、腰臂姿态与夹取方向的可抓范围…')
            self.reach_diagnostics=screen(self,source,point,product,current_cfg,order,context,on_status,observation_for_arm,reach_hint)
            preferred={r['arm'] for r in self.reach_diagnostics if r['status']=='key_poses_passed'}
            outside=len(self.reach_diagnostics)==len(order) and all(r['status']=='outside_workspace' for r in self.reach_diagnostics)
            on_reach(dict(status='outside_workspace' if outside else 'key_poses_passed' if preferred else 'reposition_review',arms=self.reach_diagnostics,
                          elapsed_seconds=round(context.clock()-context.started,3),stationary_budget_seconds=STATIONARY_SECONDS))
            if outside:raise ValueError('快速可达预判：当前身体高度及夹取方向下，双臂均超出工作范围，需要调整停靠位置')
            order=sorted(order,key=lambda side:side not in preferred)
            # 先选手、再规划：不能把已经选中的手臂在收回阶段按半份预算截断。
            # 快速预判可比较双臂，但完整路径只尝试本次选中的一只手。
            order=order[:1]
        for index,side in enumerate(order):
            if stopped():raise PlanningCancelled('用户已停止规划，未生成可执行路径')
            label='左手' if side=='left' else '右手'
            on_status('已选择'+label+'，规划完整抓取与收回路径（原地总预算10秒，不中途换手）…' if native else '自动选手：检查'+label+'抓取与收回路径…')
            config,tcp=self.arm_config(side,current_cfg)
            if native:
                context.checkpoint()
                # 直接使用整次请求的同一截止时间，预判耗时也在其中；不重置或延长10秒。
                context.on_status=on_status
                planner=self.native_planner(config,context)
                planner.preferred_grasp_yaw=next((r['yaw_offset_deg'] for r in self.reach_diagnostics
                    if r['arm']==side and r['status']=='key_poses_passed'),None)
            else:planner=self.for_config(config)
            began=time.monotonic()
            try:
                arm_point,arm_product=(observation_for_arm(side,config) if observation_for_arm else (point,product))
                center,correction=corrected_target(arm_point,config.get('target_offset_torso_mm',[0.,0.,0.]))
                plan=planner.plan(source,tcp,center,arm_product)
                if context:context.checkpoint()
            except (ValueError,PlanningTimeout) as exc:
                self.attempts.append({'arm':side,'status':'timeout' if isinstance(exc,PlanningTimeout) else 'unavailable','reason':str(exc),'seconds':round(time.monotonic()-began,3)})
                self.diagnostics.append({'arm':side,'failures':list(getattr(planner,'candidate_failures',[]))})
                if context:context.checkpoint()
                continue
            if stopped():raise PlanningCancelled('用户已停止规划，未生成可执行路径')
            self.attempts.append({'arm':side,'status':'planned','seconds':round(time.monotonic()-began,3)})
            plan.update(arm=side,arm_selection=copy.deepcopy(self.attempts),tool_geometry=config['tool_geometry'],reach_precheck=copy.deepcopy(self.reach_diagnostics))
            return plan,planner,config,tcp,center,correction
        raise ValueError(('选定手臂路径未通过：' if native else '双臂路径均未通过：')+'；'.join(('左手' if a['arm']=='left' else '右手')+'：'+a['reason'] for a in self.attempts))
