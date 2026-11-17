"""当前全身姿态下的抓取关键位姿预判；仅计算，不发布控制命令。

与正式规划共用指尖、倾角、桌面和腰臂限位。关键点通过不等于整条路径
通过；求解超时也不等于物理不可达。所有尝试计入同一个10秒预算。
"""
import time
import numpy as np
from .planning_control import PlanningControl,PlanningTimeout
from .low_object_grasp import grasp_scope,enabled,support


STATIONARY_SECONDS=10.
SCREEN_SECONDS=1.5


def slice_control(parent,seconds,on_status):
    """子阶段预算不续期；任何子阶段都先检查整个请求和用户取消。"""
    parent.checkpoint()
    def stopped():
        parent.checkpoint()
        return False
    def status(text):
        # 界面显示整次原地预算的累计时间，不在换手时看起来重新计时。
        on_status(text.split(' · 已用')[0]+f' · 原地累计{parent.clock()-parent.started:.1f}/10秒')
    return PlanningControl(stopped,status,timeout_sec=max(.001,min(seconds,parent.deadline-parent.clock())))


def model_probe(planner):
    """仅用已加载模型作快速数值预判，不启动/重配原厂IK子进程。

    预判解不会进入执行器；正式规划仍使用原厂IK及完整路径验证。
    """
    from .vendor_grasp_planning import VendorGraspPlanner
    from .bounded_grasp_ik import BoundedPoseResidual
    from .planning import IKFailure
    from scipy.optimize import least_squares
    class Probe(VendorGraspPlanner):
        configure_source=VendorGraspPlanner.configure_model
        def solve(self,source,tcp,point,rotation,*args,**kwargs):
            self.context.checkpoint()
            key=(tuple(np.round(tcp,8)),tuple(np.round(point,8)),tuple(np.round(rotation.ravel(),8)))
            if key in self._solve_cache:
                result=self._solve_cache[key]
                if isinstance(result,str):raise IKFailure(result)
                return dict(result)
            residual=BoundedPoseResidual(self,source,tcp,point,rotation)
            lo,hi=self.bounds;seed=np.clip(np.deg2rad([source[n] for n in self.names]),lo,hi)
            result=least_squares(residual.fun,seed,jac=residual.jac,bounds=(lo,hi),max_nfev=35,
                                 ftol=1e-5,xtol=1e-5,gtol=1e-5)
            error=residual.fun(result.x)
            if np.linalg.norm(error[:3])>.03 or np.linalg.norm(error[3:])>np.deg2rad(2.):
                reason='快速关键位姿逆解未收敛（不等于绝对不可达）'
                self._solve_cache[key]=reason;raise IKFailure(reason)
            q={**source,**dict(zip(self.names,np.rad2deg(result.x).tolist()))}
            self._solve_cache[key]=dict(q);return q
    probe=object.__new__(Probe);probe.__dict__=dict(planner.__dict__)
    probe._reach_quick_probe=True
    return probe


def native_key_poses(planner,source,tcp,center,product):
    """停靠和原地预判使用正式原厂逆解，避免数值预判可达、正式逆解却失败。"""
    old=getattr(planner,'_reach_quick_probe',False)
    planner._reach_quick_probe=True
    try:return check_key_poses(planner,source,tcp,center,product)
    finally:planner._reach_quick_probe=old


def broad_margin(pool,source,tcp,center,product,config):
    """明确范围外的目标不加载规划器；使用最大实际指腹接触长度。"""
    from .reach_envelope import current_margin
    from .gripper_contact_geometry import contact_tcp
    from types import SimpleNamespace
    longest=np.asarray(tcp)
    extra=0.
    if enabled(product,config):
        longest=contact_tcp(SimpleNamespace(cfg=config),tcp,.90)
        extra+=float(product['height_m'])/2  # 允许物体内上移接触点。
    if 'measured_footprint_torso' in product:
        extra+=float(np.linalg.norm(product['bbox_extents_m'])/2) if 'bbox_extents_m' in product else max(float(product.get(k,0)) for k in ('diameter_m','height_m','depth_m'))
    return current_margin(pool.fk,source,longest,config['arm'],config,center)+extra


def orientation_envelope(planner,source,tcp,center,product,yaws):
    """抓取方向约束下的腕部外包络，包含局部接触可能产生的位置偏移。

    不能把完整指长加到任意方向：斜抓时手指向下，水平够取距离变短。
    这里只排除所有允许朝向都在包络外的目标，不生成可执行逆解。
    """
    from .reach_envelope import shoulder_envelope,margin
    from .planning import IKFailure
    from .vendor_grasp_planning import VendorGraspPlanner
    if not enabled(product,planner.cfg):return None
    positions,radius=shoulder_envelope(planner.fk,source,np.zeros(3),planner.cfg['arm'],planner.cfg,1.)
    # 前端局部夹取可移到物体内部任意点；用物体外接球覆盖，不能漏算。
    if 'measured_footprint_torso' in product:
        radius+=float(np.linalg.norm(np.asarray(product['bbox_extents_m']))/2) if 'bbox_extents_m' in product else max(float(product.get(k,0)) for k in ('diameter_m','height_m','depth_m'))
        product={**product,'local_required_opening_m':min(float(product['diameter_m']),float(planner.cfg['gripper_max_width_m']))}
        yaws=sorted(set(yaws)|set(map(float,range(-90,91,5))),key=lambda x:(abs(x),x))
    tested=[]
    class Envelope(VendorGraspPlanner):
        def grasp_attitude_feasible(self,q,tool,point,R,obj):
            value=margin(positions,radius,np.asarray(point)-R@tool);tested.append(value)
            return value>=-1e-9
    probe=object.__new__(Envelope);probe.__dict__=dict(planner.__dict__)
    for yaw in yaws:
        planner.context.checkpoint()
        try:
            g=probe.route_geometry(source,tcp,center,product,False,yaw)
            # 高物体orientation不调用grasp_attitude_feasible，补做同一判断。
            if probe.grasp_attitude_feasible(source,g[8],center,g[9],g[7]):return None
        except IKFailure:continue
    if tested and max(tested)<-1e-9:
        return dict(status='outside_workspace',reason='所有允许夹取朝向的腕部位置均超出当前腰臂工作包络',
                    best_margin_m=round(max(tested),6),orientation_checks=len(tested))
    return None


@grasp_scope
def check_key_poses(planner,source,tcp,center,product):
    import copy
    from .object_extent import directional_yaws
    from .planning import IKFailure,CollisionFailure
    from .grasp_candidate_hint import request_key,joint_transit
    planner._reach_candidate_hint=None
    planner.bounded_seed_recoveries=0;planner._attitude_cache={}
    planner.configure_source(source)
    planner.guard.begin_fixed_neck(source)
    if enabled(product,planner.cfg):product=support(product,center,planner.cfg['up_torso'])
    # A reachable endpoint cannot repair an already colliding crouched source.
    try:planner.check(source,tcp,False,product)
    except (IKFailure,CollisionFailure) as exc:
        return dict(status='unresolved',reason=str(exc),failure_stage='source')
    key=request_key(source,tcp,center,product,planner.cfg)
    yaws=list(directional_yaws(product,planner.cfg)) if 'bbox_axes_torso' in product else [0.,-10.,10.,-20.,20.,-35.,35.,-50.,50.,-60.,60.]
    preferred=getattr(planner,'preferred_grasp_yaw',None)
    if preferred is not None and np.isfinite(preferred) and -90<=preferred<=90:
        yaws=[float(preferred)]+[y for y in yaws if abs(y-preferred)>1e-6]
    # 初筛不展开上百个朝向/细网格；未收敛只表示需要更好的操作点。
    if not getattr(planner,'_reach_quick_probe',False):
        outside=orientation_envelope(planner,source,tcp,center,product,yaws)
        if outside is not None:return outside
    failures=[]
    for yaw in yaws:
        planner.context.checkpoint()
        try:
            geometry=planner.route_geometry(source,tcp,center,product,False,yaw)
            obj,virtual,rotation,st=geometry[7],geometry[8],geometry[9],geometry[15]
            # 先看最苛刻的接触点，再检查预抓和抬离；不构建密集轨迹。
            solutions={};transit_checks=0
            for index in (1,0,2):
                if index>=len(st):continue
                planner._candidate_stage=index
                # Solve the pregrasp and lift from the same contact branch.
                seed=source if index==1 else solutions[1]
                q=planner.solve(seed,virtual,st[index][1],rotation,global_goal=True)
                planner.check(q,virtual,index>=2,obj)
                solutions[index]=q
                if index==0:
                    # Reject a blocked source->pregrasp branch before spending
                    # time solving its lift. The full planner still checks the
                    # Cartesian paths and dense native interpolation.
                    for transit_q in joint_transit(source,q,planner.names):
                        planner.check(transit_q,virtual,False,obj);transit_checks+=1
            planner._reach_candidate_hint=copy.deepcopy(dict(request_key=key,
                yaw_offset_deg=float(yaw),geometry=geometry,key_poses=solutions))
            return dict(status='key_poses_passed',yaw_offset_deg=float(yaw),checked_key_poses=len(solutions),
                checked_pregrasp_transit_states=transit_checks,approach_pitch_deg=float(geometry[11]),
                low_object_finger_contact_fraction=None if geometry[17] is None else float(geometry[17]),
                grasp_rotation=rotation.tolist(),virtual_tcp=virtual.tolist())
        except (IKFailure,CollisionFailure) as exc:
            failures.append(str(exc))
    # 局部接触路径仍可能可行，不能把中心点失败当成绝对不可达。
    return dict(status='unresolved',reason=failures[-1] if failures else '需检查局部夹取区域',failures=failures)


def screen(pool,source,point,product,cfg,order,parent,on_status,observation,hint=None):
    from .core import corrected_target
    rows=[];screen_used=0.
    for i,side in enumerate(order):
        remaining=SCREEN_SECONDS-screen_used
        if remaining<=0:break
        planner=None;probe_started=None;begin=time.monotonic()
        try:
            config,tcp=pool.arm_config(side,cfg)
            p,obj=observation(side,config) if observation else (point,product)
            center,_=corrected_target(p,config.get('target_offset_torso_mm',[0.,0.,0.]))
            distance_margin=broad_margin(pool,source,tcp,center,obj,config)
            if distance_margin<0:
                row=dict(status='outside_workspace',reason='当前腰臂完整外包络仍无法覆盖物体接触区域',best_margin_m=round(distance_margin,6))
            else:
                # 配置、包络与模型冷加载计入整次10秒；短时逆解额度只累计实际预判，
                # 避免首次加载吞掉额度而把同一可达位置错误地推荐再次移位。
                planner=pool.native_planner(config,parent)
                planner._reach_candidate_hint=None
                planner.configure_source(source)
                parent.checkpoint()
                remaining=SCREEN_SECONDS-screen_used
                child=slice_control(parent,max(.001,remaining),on_status)
                planner.context=child
                child.set_stage('可抓范围：接触、预抓和抬离位姿',arm=side)
                if hint and hint.get('arm')==side:planner.preferred_grasp_yaw=hint.get('yaw_offset_deg')
                probe_started=time.monotonic()
                row=native_key_poses(planner,source,tcp,center,obj)
        except PlanningTimeout as exc:
            parent.checkpoint()  # 总预算耗尽时退出，不把它变成可继续的子超时。
            row=dict(status='unresolved',reason=str(exc))
        except ValueError as exc:
            row=dict(status='unresolved',reason=str(exc))
        finally:
            if probe_started is not None:screen_used+=time.monotonic()-probe_started
        rows.append(dict(arm=side,seconds=round(time.monotonic()-begin,3),**row))
        if row['status']=='key_poses_passed':break
    return rows
