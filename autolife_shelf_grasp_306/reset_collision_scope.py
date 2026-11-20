"""已确认的颈胸模型误差只在保存姿态复位路径内兼容。

初始头壳模型重叠可随复位退出，不用网格接触深度的微小非单调变化拦截。
终点须无头壳重叠、不得新引入碰撞；其他部件和关节限位仍完整检查。
"""
import hashlib
from contextlib import contextmanager
from pathlib import Path

MODEL_SHA256='118ba602bd1b0a0964d6d149d095f0825231613218faf7b38f08b35742e72e17'
BOOT_HEAD_PAIR=frozenset(('Link_Waist_Yaw_to_Shoulder_Inner_0','Link_Neck_Yaw_to_Head_0'))

def validated_body_model(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() in {
        MODEL_SHA256,'18bc869a98ede3e8cf9a5f8f26f3db11a861f2e86d35c7a31a249df5d47a364e'}


@contextmanager
def reset_path_scope(guard,source,goal):
    from .grasp_collision import STATIC_NECK_PAIR,GROUPS
    original=guard.blocking_pairs;notes=[];cleared=[False]
    kin=guard.kin
    # 367本机模型：现场确认初始低头仍有实物间隙；仅沿用下述有界退出路径兼容。
    enabled=validated_body_model(kin.urdf_path)
    def model_pairs(values):
        q=kin.q_from_feedback({g:[values[n] for n in names] for g,names in GROUPS.items()},clip=False)
        return {frozenset(p.split(' <-> ')) for p in kin.collision_pairs(q)}
    try:
        if enabled:
            final=model_pairs(goal);initial=model_pairs(source)
            enabled=not bool(final-{STATIC_NECK_PAIR}) and not bool(initial-{STATIC_NECK_PAIR,BOOT_HEAD_PAIR})
            cleared[0]=BOOT_HEAD_PAIR not in initial
        def blocking(pairs,values):
            remaining=original(pairs,values)
            if not enabled:return remaining
            inside=all(min(source[n],goal[n])-1e-6<=values[n]<=max(source[n],goal[n])+1e-6 for n in source)
            if not inside:return remaining
            keys={frozenset(p.split(' <-> ')) for p in remaining}
            if BOOT_HEAD_PAIR not in keys:cleared[0]=True
            at_goal=all(abs(values[n]-goal[n])<=1e-6 for n in goal)
            result=[]
            for pair in remaining:
                key=frozenset(pair.split(' <-> '))
                if key==STATIC_NECK_PAIR or (key==BOOT_HEAD_PAIR and not cleared[0] and not at_goal):
                    if pair not in notes:notes.append(pair)
                else:result.append(pair)
            return result
        guard.cache.clear();guard.blocking_pairs=blocking
        yield notes
    finally:
        guard.blocking_pairs=original;guard.cache.clear()
