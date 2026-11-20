"""DM Type1 width mapping, with explicit measured or authorized reference provenance."""
import math
import hashlib
import json
from pathlib import Path
import numpy as np


def _curve_mapping(data,hardware):
    if hardware!='dm_type1' or data.get('hardware')!=hardware:
        raise ValueError('夹爪曲线必须适用于DM Type1夹爪')
    mode=data.get('calibration_mode','measured')
    if mode=='measured' and data.get('measured') is True:
        return 'dm_type1_measured_v1'
    # Robot IDs describe the data source, not a permission lock. Reference use
    # is explicit, and does not claim that the target robot was measured.
    if (mode=='reference' and data.get('measured') is False
            and data.get('reference_accepted') is True
            and data.get('source_robot_id') is not None):
        return 'dm_type1_reference_v1'
    raise ValueError('夹爪曲线需要实测数据，或已接受并注明来源的参考数据；参考数据不能标为本机实测')


def _load_curve(cfg):
    configured=cfg.get('gripper_width_calibration')
    if not configured or not Path(configured).is_file():
        raise ValueError('缺少本机DM夹爪宽度曲线；请配置本机实测或明确授权的参考数据')
    path=Path(configured)
    raw=path.read_bytes();data=json.loads(raw)
    mapping=_curve_mapping(data,cfg.get('gripper_hardware','dm_type1'))
    points=np.array(data.get('arms',{}).get(cfg['arm'],[]),dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.all(np.isfinite(points)):
        raise ValueError('每侧夹爪至少需要3组[开口毫米,原厂位置量]有效数据')
    widths,positions=points.T
    if (np.any(widths<0) or np.any(positions<10) or np.any(positions>360)
            or not np.all(np.diff(widths)>0) or not np.all(np.diff(positions)<0)):
        raise ValueError('夹爪数据应按开口递增、位置量递减排序，位置量范围10～360')
    return widths,positions,hashlib.sha256(raw).hexdigest(),mapping


def load_curve(cfg):
    return _load_curve(cfg)[:3]


def calibrated_max_width(cfg):
    widths,_,_=load_curve(cfg)
    return float(widths[-1])/1000.

def checked_margin(value):
    if isinstance(value,bool):raise ValueError('抓取收紧量应为0～50毫米')
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=50:raise ValueError('抓取收紧量应为0～50毫米')
    return value

def closing_command(width_m,margin_mm,cfg):
    widths,positions,sha,mapping=_load_curve(cfg)
    width=float(width_m);margin=checked_margin(margin_mm)
    if not math.isfinite(width) or not widths[0]<width*1000<=widths[-1]:
        raise ValueError(f'夹取部位应大于{widths[0]:g}毫米且不超过{widths[-1]:g}毫米；不能用最小开口夹紧更薄物品')
    requested=max(float(widths[0]),width*1000-margin)
    maximum=float(cfg.get('gripper_closed_deg',330.));minimum=float(cfg.get('gripper_open_deg',10.))
    if not 10<=minimum<=maximum<=360:raise ValueError('DM夹爪位置端点无效')
    angle=max(minimum,min(maximum,float(np.interp(requested,widths,positions))))
    actual=float(np.interp(angle,positions[::-1],widths[::-1]))
    return dict(estimated_width_mm=width*1000,tightening_mm=margin,
                requested_width_mm=requested,nominal_target_width_mm=actual,
                target_deg=angle,mapping=mapping,calibration_sha256=sha)

def command_from_plan(plan):
    command=plan.get('grip_command')
    if command is None:raise ValueError('旧抓取计划缺少本机DM夹爪曲线信息，请重新规划')
    expected=closing_command(plan['grasp_width_m'],command['tightening_mm'],plan['config'])
    for key in ('mapping','calibration_sha256'):
        if command.get(key)!=expected[key]:raise ValueError('夹爪标定已变化，请重新规划')
    for key in ('estimated_width_mm','requested_width_mm','nominal_target_width_mm','target_deg'):
        if not math.isclose(float(command[key]),expected[key],abs_tol=1e-6):raise ValueError('夹爪宽度计划不一致，请重新规划')
    return expected['target_deg']


def opening_clears_path(actual_mm,required_mm):
    """毫米实测标定只记录到整毫米；半毫米内差异不作为路径不通。

    仅兼容测量/换算分辨率，不改变90毫米能力范围，也不把明显受阻当成张开。
    """
    actual=float(actual_mm);required=float(required_mm)
    return math.isfinite(actual) and math.isfinite(required) and actual>=0 and required>0 and actual+.5>=required
