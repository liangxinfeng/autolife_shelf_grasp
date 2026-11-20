"""One endpoint wait, with explicit policies for grasp and placement phases."""
from dataclasses import dataclass
import math
from .motion_supervision import MotionFault


@dataclass(frozen=True)
class ArrivalPolicy:
    name: str
    joint_tolerance_deg: float
    stable_sec: float = .15
    timeout_sec: float = 5.
    position_tolerance_m: float = .008
    orientation_tolerance_deg: float = 2.


GRASP_ENDPOINT = ArrivalPolicy('grasp_endpoint', .7)
PLACE_TRANSFER = ArrivalPolicy('place_transfer', .7)
PLACE_LANDING = ArrivalPolicy('place_landing', .35, position_tolerance_m=.004)


def wait_endpoint(checked, clock, endpoint, names, policy, *, label,
                  geometry_ok=None, report=lambda value: None):
    """Wait only at a physical phase boundary; never skip a failed endpoint."""
    deadline = clock() + policy.timeout_sec
    stable = None
    last_report = -float('inf')
    while True:
        q = checked()
        now = clock()
        errors = {n: abs(float(q[n]) - float(endpoint[n])) for n in names}
        if not errors or not all(math.isfinite(v) for v in errors.values()):
            raise MotionFault('INVALID_FEEDBACK', '终点关节反馈无效：' + label)
        worst = max(errors, key=errors.get)
        error = errors[worst]
        joint_ok = error <= policy.joint_tolerance_deg
        spatial_ok = bool(geometry_ok(q)) if joint_ok and geometry_ok is not None else None
        valid = joint_ok and (geometry_ok is None or spatial_ok)
        if valid:
            if stable is None:
                stable = now
        else:
            stable = None
        reached = stable is not None and now - stable >= policy.stable_sec
        if reached or now - last_report >= .2 or now > deadline:
            report(dict(stage=label, arrival_policy=policy.name,
                max_error_deg=error, worst_joint=worst,
                joint_tolerance_deg=policy.joint_tolerance_deg,
                geometry_required=geometry_ok is not None, geometry_ok=spatial_ok,
                remaining_sec=max(0., deadline-now), endpoint_reached=reached))
            last_report = now
        if reached:
            return q
        if now > deadline:
            detail = (f'阶段终点尚未到位：{label}，{worst} 偏差{error:.2f}°；'
                      f'允许{policy.joint_tolerance_deg:g}°，')
            if joint_ok and geometry_ok is not None and not spatial_ok:
                detail += '末端位置/朝向尚未通过；'
            detail += f'等待{policy.timeout_sec:g}秒，未进入下一阶段'
            raise MotionFault('ENDPOINT_NOT_REACHED', detail)
