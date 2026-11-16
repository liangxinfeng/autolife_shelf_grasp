"""任务调用时生成另一手的动作副本；原姿态、轨迹和物体记忆不变。"""
from copy import deepcopy
import numpy as np
from autolife_handeye_calibration_306.core import ARMS,TORSO
from .pose_mirror import mirror_signs
from .task_flow import full_record
from .visual_reference import matrix

def retarget(record,target_side,current,geometry,directory):
    result=full_record(record);result.pop('retarget_arm',None)
    if target_side not in ARMS:raise ValueError('请选择左手或右手执行')
    origin='right' if target_side=='left' else 'left'
    signs=mirror_signs(geometry);source_names=ARMS[origin];target_names=ARMS[target_side]
    src_jaw='Joint_'+origin.title()+'_Gripper';dst_jaw='Joint_'+target_side.title()+'_Gripper'
    def convert(q):
        out=dict(q)
        for src,dst,sign in zip(source_names,target_names,signs):out[dst]=float(q[src])*sign
        out[dst_jaw]=float(q[src_jaw])
        for name in [*source_names,src_jaw]:out[name]=float(current[name])
        return out
    original=deepcopy(result['keyframes'])
    for row in result['keyframes']:row['joints_deg']=convert(row['joints_deg'])
    bindings={};S=np.diag([1.,-1.,1.,1.])
    for index,bound in result.get('visual_bindings',{}).items():
        if bound['side']!=origin:continue
        old_q=original[int(index)]['joints_deg'];new_q=result['keyframes'][int(index)]['joints_deg']
        torso=geometry.frames(old_q)[TORSO]
        # 镜像物体基准和工具：当前物体重新定位后，仍落在实际物体而非身体对侧的空位。
        reference=torso@S@np.linalg.inv(torso)@matrix(bound['taught_object_pose'])@S
        tcp=geometry.frames(new_q)['Link_'+target_side.title()+'_Wrist_Lower_to_Gripper']@geometry.tool_transform(target_side)
        bindings[index]={**bound,'side':target_side,'taught_object_pose':reference.tolist(),
                         'relative_tcp':(np.linalg.inv(reference)@tcp).tolist()}
    if result.get('visual_bindings') and not bindings:
        raise ValueError('所选镜像来源手臂与动作的物体纠偏手臂不一致，请核对执行手臂选项')
    if bindings:result['visual_bindings']=bindings
    else:result.pop('visual_bindings',None)
    if 'trajectory' in result:
        from .teaching_track import load_raw,save_raw,NAMES,load_events
        meta=result['trajectory'];times,values=load_raw(directory,meta['sha256'])
        events=load_events(directory,meta['sha256'],len(times))
        mirrored=np.array([[convert(dict(zip(NAMES,row)))[n] for n in NAMES] for row in values])
        meta['sha256']=save_raw(directory,times,mirrored,events)
        if meta.get('entry_gripper_hold'):meta['entry_gripper_hold']={dst_jaw:meta['entry_gripper_hold'].get(src_jaw,current[dst_jaw]),src_jaw:current[src_jaw]}
    return full_record(result)

def prepare_local(record,current,cfg,directory):
    """无视觉动作的异步模型预检；输出仍交给原执行器和原时序。"""
    from .pose_geometry import ModelGeometry
    from .pose_ik_worker import PoseSolver
    from .teaching_trajectory import plan,evaluate
    from .visual_path_sampling import sample_count
    side=record['retarget_arm'];geometry=ModelGeometry(cfg['urdf_path'])
    result=retarget(record,side,current,geometry,directory)
    staged,segments,phases=plan(result,current,directory)
    solver=PoseSolver({**cfg,'arm':side});guard=solver.planner.guard
    model_names=set(guard.kin.joint_q)
    for seg in segments:
        for t in np.linspace(0,1,sample_count(seg,model_names)+1):
            q=evaluate(seg,float(t));guard.begin(q);guard.check(q)
    return result


def prepare(record,current,cfg,directory):
    """沿用现有末端IK的独立Python环境；界面进程不加载另一套numpy/Pinocchio。"""
    import json,os,subprocess
    from pathlib import Path
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    env['LD_LIBRARY_PATH']=str(Path(cfg['python']).parent.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env['OPENBLAS_NUM_THREADS']='1'
    payload=dict(record=record,current=current,cfg=cfg,directory=str(directory))
    done=subprocess.run([cfg['python'],'-m',__name__],input=json.dumps(payload),text=True,
                        capture_output=True,env=env,timeout=50)
    lines=[line[len('ACTION_MIRROR='):] for line in done.stdout.splitlines() if line.startswith('ACTION_MIRROR=')]
    if not lines:raise ValueError('镜像模型计算未完成：'+done.stderr[-500:])
    reply=json.loads(lines[-1])
    if done.returncode or 'error' in reply:raise ValueError(reply.get('error','镜像模型计算失败'))
    return full_record(reply['record'])


if __name__=='__main__':
    import json,sys
    try:
        request=json.load(sys.stdin)
        result=prepare_local(request['record'],request['current'],request['cfg'],request['directory'])
        print('ACTION_MIRROR='+json.dumps(dict(record=result),allow_nan=False))
    except Exception as exc:
        print('ACTION_MIRROR='+json.dumps(dict(error=str(exc))))
        sys.exit(1)
