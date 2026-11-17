"""抓放停靠选点：纯模型计算，无底盘输出；执行交给现有停靠链/Nav2。

原地规划成功直接复用。需要换点时，从链中选择有链长余量的候选，
用真实当前腰腿姿态验证完整操作路径。到达后重新观察，旧路径不执行。
"""
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
from .mobile_reach import arm_envelope,waist_allowance,map_point,RepositionRequired


def chain_api():
    # 复用导航自身的采样/指纹定义，不维护第二套线条插值规则。
    from .navigation_logic import NAV
    spec=importlib.util.spec_from_file_location('_operation_parking_chain',NAV/'scripts/parking_chain.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def se2(pose):
    x,y,yaw=map(float,pose);c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,0],[0,0,0,1.]])


def transform_product(product,T):
    result=copy.deepcopy(product);R=T[:3,:3];t=T[:3,3]
    if 'bbox_axes_torso' in result:result['bbox_axes_torso']=(R@np.asarray(result['bbox_axes_torso'])).tolist()
    if 'local_contact_torso' in result:result['local_contact_torso']=(R@np.asarray(result['local_contact_torso'])).tolist()
    for key in ('measured_footprint_torso','footprint_vertices_torso'):
        if key in result:result[key]=(np.asarray(result[key])@R.T).tolist()
    if 'low_object_support' in result:
        s=result['low_object_support'];normal=R@np.asarray(s['normal'])
        s['normal']=normal.tolist();s['offset']=float(s['offset']+normal@t)
        s['center']=(R@np.asarray(s['center'])+t).tolist()
    return result


def transform_table(table,T):
    """将观察时的桌面变到候选底座系；未知支撑仍然未知。"""
    result=copy.deepcopy(table);R=T[:3,:3];t=T[:3,3]
    def box(item):
        corners=np.array(list(itertools.product(*zip(item['minimum'],item['maximum']))))@R.T+t
        item['minimum']=corners.min(axis=0).tolist();item['maximum']=corners.max(axis=0).tolist()
        if 'center' in item:item['center']=(R@np.asarray(item['center'])+t).tolist()
    box(result)
    for item in result.get('obstacles',[]):box(item)
    a,b,c=table['coef'];normal=R@np.array([-a,-b,1.]);offset=c+normal@t
    if normal[2]<.5:raise ValueError('桌面变换不支持倾覆坐标')
    result['coef']=[-normal[0]/normal[2],-normal[1]/normal[2],offset/normal[2]]
    if 'footprint_support' in result:
        import cv2
        from .placement_support import unpack_support,pack_support
        grid,origin,res=unpack_support(table['footprint_support'])
        corners=np.array([[0,0],[grid.shape[1],0],[0,grid.shape[0]],list(grid.shape[::-1])])*res+origin
        newcorners=corners@R[:2,:2].T+t[:2];neworigin=np.floor(newcorners.min(axis=0)/res)*res
        shape=np.ceil((newcorners.max(axis=0)-neworigin)/res).astype(int)+1
        if np.any(shape>900):raise ValueError('桌面支撑范围过大')
        shift=(R[:2,:2]@(origin+res/2)+t[:2]-neworigin)/res-.5
        matrix=np.column_stack([R[:2,:2],shift])
        warped=cv2.warpAffine(grid.astype(np.uint8),matrix,tuple(map(int,shape)),flags=cv2.INTER_NEAREST,borderValue=0)
        # 旋转后边缘采样不把半个未知格充当完整支撑。
        if abs(R[0,1])>1e-6:warped=cv2.erode(warped,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        result['footprint_support']=pack_support(warped,neworigin,res)
    return result


def context(run,cmd):
    path=Path(run)/'navigation_state.json'
    state=json.loads(path.read_text('utf8')) if path.exists() else {}
    if not state.get('ready') or state.get('pose') is None or not 0<=time.time()-state.get('snapshot_time',0)<3:
        raise ValueError('需要调整停靠位置，请先启动导航并确认定位')
    name=cmd.get('operation_chain_name','')
    chain=state.get('parking_chains',{}).get(name)
    if not chain:raise ValueError('需要调整停靠位置：请选择本区域的停靠链，或先执行该停靠链导航')
    api=chain_api();chain=api.validate_chain(chain)
    return dict(chain=chain,map=state['active_map'],pose=state['pose'],chain_sha256=api.chain_fingerprint(chain),
                samples=api.sample_chain(chain))


def shortlist(ctx,ground,targets,shoulders,view=None,products=None):
    """当前位置不挪；候选至少留6cm链长余量，优先中间工作区而非最小位移。"""
    rows=[];inverse=np.linalg.inv(ground);before=se2(ctx['pose'])
    counts=dict(samples=len(ctx['samples']),current_position=0,not_visible=0,visited=0,
                insufficient_margin=0,geometric_rows=0,duplicate_rows=0)
    for candidate in ctx['samples']:
        distance=math.dist([candidate['x'],candidate['y']],ctx['pose'][:2])
        for side,point in targets.items():
            from .docking_view import candidate_options
            options=candidate_options(candidate,ctx,ground,point,(products or {}).get(side,{}),view)
            if not options:counts['not_visible']+=1
            for chosen,view_score,visibility in options:
                # 同一位置可换朝向；只有位姿都未改变时才不值得再次停靠。
                yaw_change=abs(math.atan2(math.sin(chosen['yaw']-ctx['pose'][2]),math.cos(chosen['yaw']-ctx['pose'][2])))
                if distance<.025 and yaw_change<.06:counts['current_position']+=1;continue
                if any(math.hypot(chosen['x']-p[0],chosen['y']-p[1])<.08 and abs(math.atan2(math.sin(chosen['yaw']-p[2]),math.cos(chosen['yaw']-p[2])))<.35 for p in ctx.get('visited',[])):
                    counts['visited']+=1;continue
                pose=[chosen[k] for k in ('x','y','yaw')];B=np.linalg.inv(se2(pose))@before;T=inverse@B@ground
                moved=(T@np.r_[point,1.])[:3];shoulder,radius=shoulders[side]
                reach=float(np.linalg.norm(moved-shoulder));margin=radius-reach
                if margin<.06:counts['insufficient_margin']+=1;continue
                score=4.*abs(reach/radius-.72)+.8*distance+.6*abs(math.atan2(math.sin(pose[2]-ctx['pose'][2]),math.cos(pose[2]-ctx['pose'][2])))+1.5*view_score
                rows.append(dict(candidate=chosen,arm=side,point=moved,T=T,B=B,score=score,margin_m=margin,visibility=visibility))
    rows.sort(key=lambda row:row['score'])
    counts['geometric_rows']=len(rows)
    # 保留不同朝向；同点另一朝向可能才有底盘净空。导航筛选前不能截断，
    # 否则最优六点被占据会遮住链上其余可用位置。
    selected=[]
    for row in rows:
        c=row['candidate']
        if any(row['arm']==r['arm'] and math.hypot(c['x']-r['candidate']['x'],c['y']-r['candidate']['y'])<.07 and
               abs(math.atan2(math.sin(c['yaw']-r['candidate']['yaw']),math.cos(c['yaw']-r['candidate']['yaw'])))<.15 for r in selected):
            counts['duplicate_rows']+=1;continue
        selected.append(row)
    counts['retained_rows']=len(selected)
    ctx.setdefault('diagnostics',{})['geometry']=counts
    return selected


def package(kind,ctx,row,q,anchor,target,elapsed,checks):
    c=row['candidate'];pose=[c[k] for k in ('x','y','yaw')]
    return dict(route_type='parking_chain',kind=kind,token=uuid.uuid4().hex,created=time.monotonic(),
        map=ctx['map'],chain_name=ctx['chain']['name'],chain_sha256=ctx['chain_sha256'],
        candidates=[dict(index=c['index'],score=row['score'],yaw=c['yaw'])],goal_pose=pose,source_map_pose=ctx['pose'],
        source_q=dict(q),base_anchor=list(anchor),target_base=list(target),arm=row['arm'],
        workspace_margin_m=row['margin_m'],path_checked=True,checked_states=int(checks),
        preflight_seconds=round(elapsed,3),decision_stage='operation_chain',visibility=row.get('visibility'))


def navigation_filter(run,ctx,rows,stopped):
    """分批查询最新代价地图；优先点被占据时继续覆盖全链，不发送导航目标。"""
    from .core import atomic_json
    run=Path(run);start=time.monotonic();groups={}
    # 左右臂共享同一个底盘位姿的净空结果，不占用两次查询名额。
    for row in rows:
        c=row['candidate']
        key=tuple(round(float(c[k]),7) for k in ('x','y','yaw')) if all(k in c for k in ('x','y','yaw')) else ('index',c['index'])
        groups.setdefault(key,[]).append(row)
    groups=list(groups.values());accepted=[]
    diag=dict(input_rows=len(rows),unique_poses=len(groups),checked_poses=0,allowed_rows=0,
              complete=False,budget_seconds=1.,batches=[])
    ctx.setdefault('diagnostics',{})['navigation']=diag
    try:
        for begin in range(0,len(groups),6):
            token=uuid.uuid4().hex;batch=groups[begin:begin+6]
            atomic_json(run/'operation_screen_request.json',dict(token=token,created=time.monotonic(),map=ctx['map'],
                clearance=ctx['chain']['clearance'],candidates=[{**g[0]['candidate'],'index':i} for i,g in enumerate(batch)]))
            record=dict(token=token,candidates=[dict(index=i,chain_index=g[0]['candidate']['index'],
                pose=[g[0]['candidate'].get(k) for k in ('x','y','yaw')],arms=[r.get('arm') for r in g]) for i,g in enumerate(batch)])
            diag['batches'].append(record)
            while time.monotonic()-start<1.:
                if stopped():
                    from .planning_control import PlanningCancelled
                    raise PlanningCancelled('用户已停止规划')
                path=run/'operation_screen_result.json'
                result=json.loads(path.read_text('utf8')) if path.exists() else {}
                if result.get('token')==token:
                    if result.get('error'):
                        record['error']=result['error'];raise ValueError(result['error'])
                    allowed=set(result['allowed']);record['allowed']=sorted(allowed)
                    record['rejected']=result.get('rejected',[dict(index=i,reason='navigation_not_clear') for i in range(len(batch)) if i not in allowed])
                    diag['checked_poses']+=len(batch)
                    for i,g in enumerate(batch):
                        if i in allowed:accepted.extend(g)
                    break
                time.sleep(.02)
            else:
                diag['status']='budget_exhausted';record['status']='no_response_before_deadline'
                if accepted:break
                raise ValueError('操作停靠导航预筛在1秒内未完成，尚不能判断其余候选；请确认导航后台状态')
            if len(accepted)>=6:break
        diag['complete']=diag['checked_poses']==len(groups)
        diag.setdefault('status','passed' if accepted else 'all_rejected' if groups else 'no_geometric_candidates')
        # 重IK预算仍只分给最优六条；导航拒绝不消耗这六个名额。
        selected={id(row) for row in accepted}
        result=[row for row in rows if id(row) in selected][:6]
        diag['allowed_rows']=len(accepted);diag['selected_rows']=len(result)
        return result
    finally:diag['seconds']=round(time.monotonic()-start,3)


def validate(value,token,now=None):
    now=time.monotonic() if now is None else now
    if value.get('route_type')!='parking_chain' or value.get('token')!=token or len(token)!=32:raise ValueError('操作停靠请求无效')
    if not 0<=now-value.get('created',-1e9)<=90:raise ValueError('操作停靠预览已过期，请重新识别')
    key_checked=(value.get('reach_checked') is True and value.get('decision_stage')=='key_pose_docking' and
                 value.get('reach_evidence',{}).get('status')=='key_poses_passed' and value.get('reach_evidence',{}).get('checked_key_poses')==3)
    if not value.get('path_checked') and not key_checked:raise ValueError('操作点尚未通过可达验证')
    if not value.get('checked_states'):raise ValueError('操作点没有模型检查记录')
    if not value.get('chain_name') or len(value.get('chain_sha256',''))!=64:raise ValueError('操作停靠链缺失')
    if not np.isfinite(value['goal_pose']).all() or len(value['goal_pose'])!=3:raise ValueError('操作停靠位姿无效')
    return value


def grasp_outside(pool,q,point,cfg):
    """只在连腰臂上界都超出时直接选点；范围内不改变原规划。"""
    sides=('left','right') if cfg.get('auto_arm_selection',True) else (cfg['arm'],)
    for side in sides:
        c,tcp=pool.arm_config(side,cfg);shoulder,radius=arm_envelope(pool.fk,q,tcp,side)
        center=np.asarray(point)+np.asarray(c.get('target_offset_torso_mm',[0,0,0]))/1000
        if np.linalg.norm(center-shoulder)<=radius+waist_allowance(pool.fk,q,shoulder,c):return False
    return True


def grasp(run,cmd,pool,q,point,product,cfg,anchor,stopped,on_status,observation,view=None):
    from autolife_handeye_calibration_306.core import TORSO
    from .planning_control import PlanningControl,PlanningTimeout
    from .core import atomic_json
    ctx=context(run,cmd);ctx['visited']=cmd.get('_operation_visited',[]);ground=pool.fk.fk('Link_Zero_Point',TORSO,q)
    sides=('left','right') if cfg.get('auto_arm_selection',True) else (cfg['arm'],)
    targets={};shoulders={};configs={};products={}
    for side in sides:
        c,tcp=pool.arm_config(side,cfg);p,obj=observation(side,c) if observation else (point,product)
        targets[side]=p;products[side]=obj;configs[side]=(c,tcp)
        shoulders[side]=arm_envelope(pool.fk,q,tcp,side)
    from .reach_precheck import native_key_poses
    control=PlanningControl(stopped,on_status,timeout_sec=4.);failures=[]
    diag=dict(status='checking',stage='geometry',chain_name=ctx['chain']['name'],source_map_pose=ctx['pose'],
              visited=ctx['visited'],key_pose_checks=[],request_id=cmd.get('request_id',cmd.get('id')))
    ctx['diagnostics']=diag
    try:
        candidates=shortlist(ctx,ground,targets,shoulders,view,products)
        diag['stage']='navigation'
        rows=navigation_filter(run,ctx,candidates,stopped)
        if not candidates:
            diag['stage']='geometry'
            raise ValueError('本停靠链没有新的可见且具有抓取余量的候选；未进行关键位姿检查，请查看停靠诊断中的视野、已访问位置与余量筛选原因。')
        if not rows:
            raise ValueError('本停靠链的可见候选均未通过导航净空检查（障碍、未知区域或地图边界）；未进行关键抓取位姿检查，请查看停靠诊断。')
        diag['stage']='key_poses'
        for row in rows:
            control.checkpoint();side=row['arm'];c,tcp=configs[side]
            control.set_stage('停靠候选的关键抓取位姿',arm=side)
            record=dict(arm=side,chain_index=row['candidate']['index'],
                        pose=[row['candidate'][k] for k in ('x','y','yaw')],status='checking')
            diag['key_pose_checks'].append(record);began=time.monotonic()
            # 一个难解候选不能占满整次停靠筛选，给另一只手/下一位置保留机会。
            def stopped_candidate():
                control.checkpoint()
                return False
            child=PlanningControl(stopped_candidate,on_status,timeout_sec=min(1.2,control.deadline-control.clock()))
            center=row['point']+np.asarray(c.get('target_offset_torso_mm',[0,0,0]))/1000
            obj=transform_product(products[side],row['T'])
            try:
                planner=pool.native_planner(c,child)
                evidence=native_key_poses(planner,q,tcp,center,obj)
                record.update(evidence)
                if evidence['status']!='key_poses_passed':failures.append(evidence.get('reason','关键位姿未通过'));continue
            except (ValueError,PlanningTimeout) as exc:
                record.update(status='timeout' if isinstance(exc,PlanningTimeout) else 'unresolved',reason=str(exc))
                failures.append(str(exc));control.checkpoint();continue
            finally:record['seconds']=round(time.monotonic()-began,3)
            control.checkpoint()
            target=(ground@np.r_[targets[side],1.])[:3]
            result=package('grasp',ctx,row,q,anchor,target,time.monotonic()-control.started,3)
            result.update(path_checked=False,reach_checked=True,decision_stage='key_pose_docking',reach_evidence=evidence)
            diag.update(status='key_poses_passed',selected_arm=side,selected_pose=result['goal_pose'])
            return result
        raise ValueError('已检查的停靠候选关键抓取位姿均未确认可达；请查看停靠诊断。'+(failures[-1] if failures else ''))
    except Exception as exc:
        diag.update(status='failed',reason=str(exc));raise
    finally:
        diag['seconds']=round(time.monotonic()-control.started,3)
        atomic_json(Path(run)/'operation_diagnostics.json',diag)


def place(run,cmd,planner,q,tcp,grasp_plan,points,table,settings,anchor,stopped,on_status):
    from autolife_handeye_calibration_306.core import TORSO
    from .demo_config import carried_height
    from .planning_control import PlanningControl
    ctx=context(run,cmd);ground=planner.fk.fk('Link_Zero_Point',TORSO,q);inv=np.linalg.inv(ground)
    side=planner.cfg['arm'];shoulder,radius=arm_envelope(planner.fk,q,tcp,side)
    # 以可放区域中值筛选停靠点；最终仍调用完整桌面候选搜索，不固定这个落点。
    point=(inv@np.r_[np.median(points,axis=0),1.])[:3]
    point+=ground[:3,:3].T@np.array([0,0,carried_height(grasp_plan['product'])/2+settings['place_clearance_m']])
    control=PlanningControl(stopped,on_status,timeout_sec=12.);planner.context=control;failures=[]
    rows=navigation_filter(run,ctx,shortlist(ctx,ground,{side:point},{side:(shoulder,radius)}),stopped)
    for row in rows:
        control.checkpoint();on_status('检查操作停靠点的完整放置与退离路径…')
        B=row['B'];moved=np.asarray(points)@B[:3,:3].T+B[:3,3]
        try:plan=planner.plan_place(q,tcp,grasp_plan,moved,transform_table(table,B),settings)
        except ValueError as exc:failures.append(str(exc));continue
        return package('place',ctx,row,q,anchor,np.median(points,axis=0),time.monotonic()-control.started,plan.get('checked_states',1))
    raise ValueError('本停靠链尚无具有操作余量且放置路径通过的位置。'+(failures[-1] if failures else ''))
