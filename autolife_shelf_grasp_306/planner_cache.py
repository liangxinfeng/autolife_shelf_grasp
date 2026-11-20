"""只复用模型和纯计算进程，绝不复用路径、关节起点或碰撞结果。"""
import json
from pathlib import Path
from .resources import collision_model_path,vendor_control_path

DYNAMIC={'up_torso','target_offset_torso_mm','target_offsets_by_arm_torso_mm',
    'initial_pose_deg','workspace_reset_joints','workspace_reset_path','workspace_reset_sha256',
    'workspace_pose_speed_percent','grasp_speed_percent','semantic_prompt_overrides'}

def cache_key(cfg):
    static={k:v for k,v in cfg.items() if k not in DYNAMIC}
    vendor=vendor_control_path(cfg)
    collision=collision_model_path(cfg)
    files=[Path(cfg['urdf_path']),vendor,collision,collision.with_name('robot_v3_1.srdf')]
    if cfg.get('gripper_width_calibration'):files.append(Path(cfg['gripper_width_calibration']))
    stamp=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    return json.dumps([static,stamp],sort_keys=True,allow_nan=False)


def transfer_idle_solver(receiver,donor_entry,key):
    """单线程工作器内转移同模型纯IK进程的所有权，不共享路径/起点/碰撞缓存。

    接收方下一步仍执行 configure_source，更新固定膝踝和坐标核验。
    双方不能同时持有进程，避免释放时互相关闭。
    """
    if receiver.solver is not None or donor_entry is None or donor_entry[0]!=key:return False
    donor=donor_entry[1];solver=donor.solver
    if solver is None or getattr(solver,'_closed',True) or solver.process.poll() is not None:return False
    receiver.solver=solver;donor.solver=None
    return True
