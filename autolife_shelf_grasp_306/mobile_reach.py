"""短距离底盘补偿的纯计算/协议；不导入ROS、不发送运动。

平面全向最多15cm、单次直线平移补偿，不旋转身体。URDF链长只判断距离，不冒充完整IK/路径可行性。
移动前检查身体扫掠；移动后重新取景并进入原有完整规划。
"""
import copy
import math
import time
import uuid
import numpy as np

MAX_TRANSLATION = .15


class RepositionRequired(RuntimeError):
    def __init__(self, proposal):
        super().__init__('目标超出当前工作范围，需要底盘小幅平移')
        self.proposal = proposal


def shifted_product(product, delta):
    p = copy.deepcopy(product)
    if 'low_object_support' in p:
        s = p['low_object_support']
        s['offset'] -= float(np.dot(s['normal'], delta))
        s['center'] = (np.asarray(s['center']) - delta).tolist()
    # 物体自身尺寸、主轴和抓点偏移是向量，不能再减去底盘位移。
    return p


def shifted_table(table, delta):
    t = copy.deepcopy(table);d = np.asarray(delta)
    for k in ('minimum','maximum'):
        t[k] = (np.asarray(t[k])-d).tolist()
    a,b,c = t['coef'];t['coef'] = [a,b,c+a*d[0]+b*d[1]-d[2]]
    # 平面平移时，支撑点与障碍数据逐项换坐标。
    for obstacle in t.get('obstacles',[]):
        for k in ('minimum','maximum','center'):
            if k in obstacle:obstacle[k] = (np.asarray(obstacle[k])-d).tolist()
    if 'footprint_support' in t:
        from .placement_support import unpack_support
        unpack_support(t['footprint_support'])
        t['footprint_support']['origin']=(np.asarray(t['footprint_support']['origin'])-d[:2]).tolist()
    return t


def map_point(point, before, after):
    """静止世界中的点，从移动前底座系转换到移动后底座系。"""
    p=np.asarray(point,float);a=np.asarray(before,float);b=np.asarray(after,float)
    if p.shape!=(3,) or a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(np.r_[p,a,b]).all():
        raise ValueError('底盘补偿坐标无效')
    def r(y):return np.array([[math.cos(y),-math.sin(y)],[math.sin(y),math.cos(y)]])
    result=p.copy();result[:2]=r(b[2]).T@(r(a[2])@p[:2]+a[:2]-b[:2]);return result


def translation(value):
    """底座系位移：X向前、Y向左；兼容旧前移记录，不改变旧记录含义。"""
    if isinstance(value,dict):
        raw=value.get('translation_m',[value.get('forward_m',float('nan')),0.])
    else:raw=value
    d=np.asarray(raw,float)
    if d.ndim==0:d=np.array([float(d),0.])
    if d.shape!=(2,) or not np.isfinite(d).all():raise ValueError('底盘补偿平移向量无效')
    return d


def direction_text(value):
    d=translation(value);parts=[]
    for v,positive,negative in zip(d,('前移','左移'),('后移','右移')):
        if abs(v)>=.0005:parts.append(f'{positive if v>0 else negative}{abs(v)*100:.1f}厘米')
    return '、'.join(parts) or '无需移动'


def validate_request(value, token, now=None):
    now=time.monotonic() if now is None else now
    if value.get('token')!=token or not isinstance(token,str) or len(token)!=32:
        raise ValueError('底盘补偿请求已失效')
    delta=translation(value);distance=float(np.linalg.norm(delta))
    if not .01-1e-9<=distance<=MAX_TRANSLATION+1e-9:
        raise ValueError('底盘单次平移补偿总距离为1～15厘米')
    if 'translation_m' not in value and delta[0]<0:raise ValueError('旧补偿请求方向无效')
    if not 0<=now-float(value.get('created',-1e9))<=30:
        raise ValueError('补偿预览已过期，请重新规划')
    if value.get('decision_stage')!='recognition_workspace' or not value.get('reach_checked') or not value.get('sweep_checked'):
        raise ValueError('补偿工作范围或身体扫掠未检查')
    return dict(enabled=True,moves=[dict(kind='translate',x_cm=float(delta[0]*100),y_cm=float(delta[1]*100),speed=12.)])


def proposal(kind, delta, q, anchor, target, side, checked, elapsed):
    delta=translation(delta)
    return dict(kind=kind,token=uuid.uuid4().hex,translation_m=delta.tolist(),source_q=dict(q),
        base_anchor=list(anchor),target_base=list(target),arm=side,path_checked=False,reach_checked=True,decision_stage='recognition_workspace',
        sweep_checked=True,checked_states=int(checked),created=time.monotonic(),
        preflight_seconds=round(elapsed,3))


def corridor_remaining(request,here,yaw,starting=False):
    """在初始底座系核对斜向走廊，返回当前底座系剩余位移。"""
    delta=translation(request);length=float(np.linalg.norm(delta));direction=delta/length
    position=np.asarray(here,float)[:2];along=float(position@direction)
    cross=float(np.linalg.norm(position-along*direction))
    if cross>.02 or abs(yaw)>.025 or along<-.01 or along>length+.02:
        raise ValueError('底盘偏离已检查的短距离补偿走廊')
    if starting and np.linalg.norm(position)>.01:raise ValueError('规划后底盘位置改变，请重新规划')
    # 到达/略过目标时不生成反向补偿；横向偏差已由走廊判断约束。
    remaining=delta-position if along<length else np.zeros(2)
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,s],[-s,c]])@remaining


def collision_boxes(planner,q):
    """当前姿态的URDF局部包围盒，统一到Link_Zero_Point坐标。"""
    import pinocchio as pin
    from autolife_handeye_calibration_306.core import BODY,NECK,ARMS
    k=planner.guard.kin
    raw=k.q_from_feedback({'leg_waist':[q[n] for n in BODY],'neck':[q[n] for n in NECK],
        'left_arm':[q[n] for n in ARMS['left']],'right_arm':[q[n] for n in ARMS['right']]},clip=False)
    pin.forwardKinematics(k.model,k.data,raw);pin.updateFramePlacements(k.model,k.data)
    pin.updateGeometryPlacements(k.model,k.data,k.geom_model,k.geom_data,raw)
    root=k.data.oMf[k.model.getFrameId('Link_Zero_Point')]
    records=[]
    for transform,(low,high) in zip(k.geom_data.oMg,planner.guard.boxes):
        records.append(dict(rotation=(root.rotation.T@transform.rotation).tolist(),
            translation=(root.rotation.T@(transform.translation-root.translation)).tolist(),
            low=np.asarray(low).tolist(),high=np.asarray(high).tolist()))
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from .pose_geometry import ModelGeometry
    from .gripper_contact_geometry import gripper_geometry
    model=ModelGeometry(planner.cfg['urdf_path'])
    for side in ('left','right'):
        cfg={**planner.cfg,'arm':side,'T_wrist_tool_quat_xyzw':Rotation.from_matrix(model.tool_transform(side)[:3,:3]).as_quat().tolist()}
        vertices=gripper_geometry(SimpleNamespace(cfg=cfg))['vertices']
        wrist=planner.fk.fk('Link_Zero_Point','Link_'+side.title()+'_Wrist_Lower_to_Gripper',q)
        records.append(dict(rotation=wrist[:3,:3].tolist(),translation=wrist[:3,3].tolist(),
            low=vertices.min(axis=0).tolist(),high=vertices.max(axis=0).tolist()))
    return records


def outside(points,boxes,padding=0.):
    points=np.asarray(points,float).reshape(-1,3);keep=np.ones(len(points),bool)
    for b in boxes:
        local=(points-np.asarray(b['translation']))@np.asarray(b['rotation'])
        keep &= np.any((local<np.asarray(b['low'])-padding)|(local>np.asarray(b['high'])+padding),axis=1)
    return keep


def sweep_clear(points,boxes,distance,held_box=None):
    """检查平移整段扫掠，不将当前机身之外的真实点清除。"""
    p=np.asarray(points,float).reshape(-1,3)
    p=p[np.isfinite(p).all(axis=1)]
    if len(p)<100:raise ValueError('底盘补偿缺少当前场景深度')
    all_boxes=boxes+([held_box] if held_box else [])
    environment=p[outside(p,all_boxes,.005)]
    # 点相对移动包围盒的线段与AABB相交；解析计算避免逐厘米重复遍历点云。
    for box in all_boxes:
        R=np.asarray(box['rotation']);local=(environment-np.asarray(box['translation']))@R
        direction=-np.r_[translation(distance),0.]@R
        enter=np.zeros(len(local));leave=np.ones(len(local))
        for axis in range(3):
            low=box['low'][axis]-.004;high=box['high'][axis]+.004
            if abs(direction[axis])<1e-12:
                leave[(local[:,axis]<low)|(local[:,axis]>high)]=-1
            else:
                a=(low-local[:,axis])/direction[axis];b=(high-local[:,axis])/direction[axis]
                enter=np.maximum(enter,np.minimum(a,b));leave=np.minimum(leave,np.maximum(a,b))
        if np.any(enter<=leave):return False
    return True


def arm_envelope(fk,q,tcp,side):
    """URDF肩部到TCP的链长上界。只做距离判断，不调用逆解或路径规划。"""
    from autolife_handeye_calibration_306.core import TORSO
    child='Link_'+side.title()+'_Wrist_Lower_to_Gripper';radius=float(np.linalg.norm(tcp))
    while fk.joints[child][2]!=TORSO:
        radius+=float(np.linalg.norm(fk.joints[child][3][:3,3]));child=fk.joints[child][2]
    shoulder=fk.fk(TORSO,child,q)[:3,3]
    if not .3<radius<1.5:raise ValueError('URDF手臂工作范围无效')
    return shoulder,radius


def waist_allowance(fk,q,shoulder,cfg):
    """原地腰臂可达范围的保守上界；在此范围内一律交给原规划，不先移动。

    腰部各转轴使肩部移动的弦长之和是肩部总位移上界。使用已有腰部
    参与角度，不放宽IK限位；忽略死区只会让更多目标优先原地规划。
    无需IK迭代或采样，避免新增长时间的预检查。
    """
    from autolife_handeye_calibration_306.core import BODY,TORSO
    ranges=np.asarray(cfg.get('vendor_grasp_waist_range_deg',[12.,8.]),float)
    if ranges.shape!=(2,) or not np.isfinite(ranges).all() or np.any((ranges<0)|(ranges>20)):
        raise ValueError('腰臂参与范围配置无效')
    ground=fk.fk('Link_Zero_Point',TORSO,q)
    point=(ground@np.r_[shoulder,1.])[:3];bound=0.
    for name,degrees in zip(BODY[2:],ranges):
        joint=next((j for j in fk.joints.values() if j[0]==name),None)
        if joint is None:raise ValueError('工作范围模型缺少腰关节：'+name)
        origin=fk.fk('Link_Zero_Point',joint[2],q)@joint[3]
        # 用到关节原点的距离替代轴线半径，得到向外包含的上界。
        bound+=2.*float(np.linalg.norm(point-origin[:3,3]))*math.sin(math.radians(degrees)/2.)
    return bound


def needs_base(point,shoulder,radius,fk,q,cfg):
    # 只在连腰臂上界都超出时考虑底盘；边界、腰臂可达区都保留原地规划。
    return np.linalg.norm(np.asarray(point)-shoulder)>radius+waist_allowance(fk,q,shoulder,cfg)+1e-9


def workspace_translation(point,shoulder,radius,torso_to_base_rotation):
    """距离可达预判：投影到水平移动平面，求最短二维补偿，不做IK。"""
    relative=np.asarray(torso_to_base_rotation)@(np.asarray(point)-np.asarray(shoulder))
    if np.linalg.norm(relative)<=radius:return np.zeros(2)
    usable=radius-.01;vertical=float(relative[2]);horizontal=float(np.linalg.norm(relative[:2]))
    if abs(vertical)>=usable or horizontal<1e-9:return None  # 高度不能由水平移动修复。
    reach=math.sqrt(usable*usable-vertical*vertical)
    distance=math.ceil(max(.01,horizontal-reach)*100-1e-10)/100
    return relative[:2]*(distance/horizontal) if distance<=MAX_TRANSLATION+1e-9 else None


def grasp_reposition(pool,q,point,product,cfg,anchor,cloud,stopped,on_status,observation=None):
    """识别阶段的工作范围判断；范围内零IK，范围外仅检查底盘扫掠。"""
    from .arm_selection import rank_arms
    from .planning_control import PlanningControl
    from autolife_handeye_calibration_306.core import TORSO
    started=time.monotonic();context=PlanningControl(stopped,timeout_sec=5.)
    context.checkpoint()
    ground=pool.fk.fk('Link_Zero_Point',TORSO,q)
    order=rank_arms(pool.fk,point,cfg.get('arm','left')) if cfg.get('auto_arm_selection',True) else [cfg['arm']]
    choices=[]
    for side in order:
        c,tcp=pool.arm_config(side,cfg)
        arm_point,_=observation(side,c) if observation else (point,product)
        center=np.asarray(arm_point)+np.asarray(c.get('target_offset_torso_mm',[0,0,0]))/1000
        shoulder,radius=arm_envelope(pool.fk,q,tcp,side)
        if not needs_base(center,shoulder,radius,pool.fk,q,c):return None
        delta=workspace_translation(center,shoulder,radius,ground[:3,:3])
        if delta is not None and not np.any(delta):return None  # 范围内保留原规划。
        if delta is not None:choices.append((float(np.linalg.norm(delta)),delta,side,c,arm_point,radius))
    for distance,delta,side,c,arm_point,radius in sorted(choices,key=lambda v:v[0]):
        context.checkpoint();on_status(f'识别可达判断：建议{direction_text(delta)}，正在核对底盘通道…')
        p=pool.native_planner(c,context);boxes=collision_boxes(p,q)
        if not sweep_clear(cloud,boxes,delta):continue
        target=(ground@np.r_[arm_point,1])[:3]
        result=proposal('grasp',delta,q,anchor,target,side,0,time.monotonic()-started)
        result.update(body_boxes=boxes,decision_stage='recognition_workspace',reach_checked=True,
            path_checked=False,workspace_radius_m=radius)
        return result
    return None


def place_reposition(planner,q,tcp,grasp,points,table,settings,anchor,cloud,stopped,on_status):
    """桌面识别后检查候选落点工作范围，补偿到位后才调用原放置规划。"""
    from .planning_control import PlanningControl
    from .demo_config import carried_height,footprint_radius
    from autolife_handeye_calibration_306.core import TORSO
    started=time.monotonic();context=PlanningControl(stopped,timeout_sec=5.)
    context.checkpoint()
    ground=planner.fk.fk('Link_Zero_Point',TORSO,q);inv=np.linalg.inv(ground)
    points=np.asarray(points);up=ground[:3,:3].T@np.array([0.,0.,1.])
    centers=points@inv[:3,:3].T+inv[:3,3]+up*(carried_height(grasp['product'])/2+settings['place_clearance_m'])
    shoulder,radius=arm_envelope(planner.fk,q,tcp,planner.cfg['arm'])
    choices=[]
    offset=np.asarray(planner.cfg.get('target_offset_torso_mm',[0,0,0]))/1000
    for index,center in enumerate(centers):
        if not needs_base(center+offset,shoulder,radius,planner.fk,q,planner.cfg):return None
        delta=workspace_translation(center+offset,shoulder,radius,ground[:3,:3])
        if delta is not None and not np.any(delta):return None
        if delta is not None:choices.append((float(np.linalg.norm(delta)),index,delta))
    if not choices:return None
    boxes=collision_boxes(planner,q)
    from autolife_handeye_calibration_306.verification_core import wrist_frame
    wrist=planner.fk.fk('Link_Zero_Point',wrist_frame(planner.cfg),q)
    held_center=(wrist@np.r_[tcp,1])[:3]
    radius=math.hypot(carried_height(grasp['product'])/2,footprint_radius(grasp['product']))+.015
    held=dict(rotation=np.eye(3).tolist(),translation=held_center.tolist(),low=[-radius]*3,high=[radius]*3)
    seen=set()
    for distance,index,delta in sorted(choices,key=lambda v:v[0]):
        key=tuple(np.round(delta,6))
        if key in seen:continue
        seen.add(key);context.checkpoint()
        if not sweep_clear(cloud,boxes,delta,held):continue
        result=proposal('place',delta,q,anchor,points[index],planner.cfg['arm'],0,time.monotonic()-started)
        result.update(body_boxes=boxes,held_box=held,decision_stage='recognition_workspace',reach_checked=True,path_checked=False)
        return result
    return None


def select_after_move(chosen,fresh,frame,T_ground,cfg,reference,anchor):
    """按实测底盘位移预测物体位置，避免用移动前2D重叠率误拒绝。"""
    from .bottle import object_points
    expected=map_point(reference['target_base'],reference['base_anchor'],anchor)
    candidates=[]
    for item in fresh:
        if chosen.get('product') and item.get('product') and chosen['product']['id']!=item['product']['id']:continue
        try:points,_,_=object_points(frame,item['mask'],T_ground,cfg)
        except (ValueError,KeyError):continue  # 一个候选没有深度，不应阻止其他有效候选。
        if len(points)<20:continue
        center=np.median(points,axis=0);error=float(np.linalg.norm(center-expected))
        a=np.asarray(chosen['feature']);b=np.asarray(item['feature'])
        similarity=float(a@b/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b))) if a.shape==b.shape else 0.
        if error<.06 and similarity>.65:candidates.append((error,item))
    candidates.sort(key=lambda v:v[0])
    if not candidates or len(candidates)>1 and candidates[1][0]-candidates[0][0]<.02:
        raise ValueError('底盘到位后目标关联不唯一，请重新选择物品；未执行旧轨迹')
    error,winner=candidates[0]
    return {**winner,'id':chosen['id'],'product':chosen.get('product'),'name':chosen.get('name',winner.get('name',''))},dict(
        method='odometry_reprojection',selected_id=chosen['id'],position_error_m=error)
