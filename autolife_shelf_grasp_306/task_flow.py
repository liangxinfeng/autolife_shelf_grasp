"""Small serial workflow protocol. A failed step never advances the cursor."""
from copy import deepcopy
import uuid
import math
from .task_runtime import TaskRuntime

KINDS = {'navigate': '导航', 'grasp': '抓取', 'cleanup': '自动清理桌面', 'place': '放置', 'action': '示教动作', 'base':'底盘定量', 'tts':'语音播报'}


def full_record(record):
    from control_center_joint_control.action_library import normalize_action_record
    from control_center_joint_control.joint_definitions import ALL_JOINTS, GROUP_ORDER, JOINT_LIMITS
    if set(record.get('groups', [])) != set(GROUP_ORDER):
        raise ValueError('动作必须记录全部关节，请在示教动作页重新记录')
    for frame in record.get('keyframes', []):
        values = frame.get('joints_deg', {})
        if set(values) != set(ALL_JOINTS): raise ValueError('关键帧缺少全身关节，未自动补零')
        for name, value in values.items():
            lo, hi = JOINT_LIMITS[name]
            if not math.isfinite(float(value)) or not lo <= float(value) <= hi:
                raise ValueError('关键帧关节超限：' + name)
    result=normalize_action_record(record)
    if 'trajectory' in record:
        from .teaching_track import check_hash
        meta=record['trajectory'];indices=meta['indices'];speeds=meta['speeds']
        if len(indices)!=len(result['keyframes']) or len(speeds)!=len(indices) or len(indices)<2:
            raise ValueError('连续轨迹的编辑节点与录制引用不匹配')
        if indices!=sorted(set(indices)) or indices[0]!=0 or any(type(i) is not int or not 0<=i<=15000 for i in indices):
            raise ValueError('连续轨迹节点顺序不正确')
        if any(not math.isfinite(float(v)) or not 25<=float(v)<=150 for v in speeds):raise ValueError('连续轨迹速度无效')
        if result['loop_count']!=1:raise ValueError('连续轨迹需要在任务流程中重复调用，不能改为内部循环')
        minimums=meta.get('minimums',[.2]*len(indices))
        if len(minimums)!=len(indices) or any(not math.isfinite(float(v)) or not .2<=float(v)<=30 for v in minimums):raise ValueError('轨迹最短时间无效')
        result['trajectory']=dict(sha256=check_hash(meta['sha256']),indices=list(indices),speeds=list(map(float,speeds)),minimums=list(map(float,minimums)))
        if 'timing' in meta:
            if meta['timing'] not in ('recorded','interaction'):raise ValueError('连续轨迹时序模式无效')
            result['trajectory']['timing']=meta['timing']
        if 'entry_gripper_hold' in meta:
            from control_center_joint_control.joint_definitions import GRIPPER_JOINTS
            hold=meta['entry_gripper_hold']
            if any(n not in GRIPPER_JOINTS or not math.isfinite(float(v)) or not JOINT_LIMITS[n][0]<=float(v)<=JOINT_LIMITS[n][1] for n,v in hold.items()):raise ValueError('持物开度无效')
            result['trajectory']['entry_gripper_hold']=dict(hold)
    if record.get('visual_bindings'):
        from .visual_reference import record_bindings
        result['visual_bindings']=record_bindings(record)
    if record.get('retarget_arm'):
        if record['retarget_arm'] not in ('left','right'):raise ValueError('镜像手臂无效')
        result['retarget_arm']=record['retarget_arm']
    return result


def validate(record):
    rows = record.get('steps', [])
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError('流程需要1～100个步骤')
    result = []
    for row in rows:
        kind = row.get('kind'); target = str(row.get('target', '')).strip()
        if kind not in KINDS: raise ValueError('不支持的任务类型')
        if kind=='cleanup':target='当前桌面'
        if kind not in ('place','base') and not target: raise ValueError('请填写航点、物品或动作名称')
        if len(target) > (500 if kind=='tts' else 160): raise ValueError('名称过长')
        from .step_posture import normalize
        item=dict(kind=kind,target=target);posture=normalize(row.get('posture'))
        if kind=='tts':
            posture=None
            required=row.get('required', False)
            if type(required) is not bool:raise ValueError('播报必需标记无效')
            item['required']=required
        if kind=='action' and row.get('action_speed_percent') is not None:
            from .action_speed import percent
            item['action_speed_percent']=percent(row['action_speed_percent'])
        if kind=='action' and row.get('execution_arm','original')!='original':
            if row['execution_arm'] not in ('left','right'):raise ValueError('执行手臂无效')
            item['execution_arm']=row['execution_arm']
        if posture and posture['body_mode']=='recorded' and kind!='action':
            raise ValueError('沿用动作记录高度仅用于示教动作步骤')
        if kind=='base':
            from .base_sequence import normalize as base_normalize
            item['base']=base_normalize(row.get('base'));item['target']='底盘定量动作'
        if posture is not None:item['posture']=posture
        result.append(item)
    return dict(version=1, steps=result)


class Cursor(TaskRuntime):
    def __init__(self, record, **kwargs):
        super().__init__(validate(record), **kwargs)

    def stop(self, reason):
        self.wait(reason)

    def retry(self, *, confirmed=True):
        # Explicit legacy caller retries remain confirmed; automatic callers
        # must opt in to the no-side-effect policy with confirmed=False.
        super().retry(confirmed=confirmed)


def select_instance(items, target, products):
    return select_candidates(items, target, products)[0]


class TargetNotVisible(ValueError):
    """A fresh task scan can recover without repeating any physical step."""


def select_candidates(items, target, products, semantic=False):
    from .auto_grasp import rank_candidates
    product_ids = {p['id'] for p in products if p['name'] == target}
    matches = items if semantic else [v for v in items if v.get('product_id') in product_ids or v.get('name') == target]
    if not matches: raise TargetNotVisible('未识别到任务目标：' + target)
    return rank_candidates(matches)


class NavigationSettling(ValueError):
    """Fresh stationary feedback is not ready yet; retry only before dispatch."""


def navigation_pose(command, context, samples, now):
    """A generic workflow snapshots fresh stationary joints, not a fake grasp receipt."""
    run_id = context.get('run_id', '')
    try:
        prefix, index = command['leg'].split(':'); index = int(index)
        rows = validate(context['record'])['steps']
        if not 0 <= index < len(rows): raise ValueError()
        row = rows[index]
        if (prefix != run_id or command.get('run_id') != run_id or
                command.get('map') != context.get('map') or row['kind'] != 'navigate' or
                row['target'] != command.get('name')): raise ValueError()
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('导航请求与本轮任务快照不一致') from exc
    recent = [(t,q) for t,q in samples if 0 <= now-t <= .8]
    if len(recent)<3 or now-recent[-1][0]>.3 or recent[-1][0]-recent[0][0]<.4:
        raise NavigationSettling('等待关节连续反馈与停稳后再开始任务')
    q = recent[-1][1]
    from control_center_joint_control.joint_definitions import BODY_JOINTS
    if not set(BODY_JOINTS).issubset(q): raise ValueError('全身关节反馈不完整')
    for _, sample in recent:
        if set(sample)!=set(q): raise ValueError('关节反馈不完整')
        for n,v in sample.items():
            if not math.isfinite(v) or abs(v-q[n])>(3. if 'Gripper' in n else 1.):
                if not math.isfinite(v):raise ValueError('关节反馈含无效数值')
                raise NavigationSettling('身体或手臂尚未停稳，未开始导航')
    return dict(q)
