"""Perception is optional for saved actions, posture and waypoint tasks."""
VISUAL_KINDS = frozenset(('grasp', 'place', 'cleanup'))


def uses_perception(record):
    return any(row['kind'] in VISUAL_KINDS for row in (record or {}).get('steps', []))


def require_perception(owner, record):
    if not uses_perception(record) or owner.ready:
        return
    labels = {'grasp': '视觉抓取', 'place': '视觉放置', 'cleanup':'自动清理桌面'}
    steps = '、'.join(f"第{i+1}步{labels[row['kind']]}" for i, row in enumerate(record['steps'])
                     if row['kind'] in VISUAL_KINDS)
    reason = getattr(owner, 'vision_error', '') or '识别后台尚未就绪'
    raise ValueError(f'{steps}需要视觉能力：{reason}。预录制动作、航点导航可独立使用。')


def starts_with_grasp(record):
    """新任务首个操作是抓取时，从空手开始；放置/动作优先则保留持物来源。"""
    for row in (record or {}).get('steps',[]):
        if row['kind'] in ('grasp','cleanup','place','action'):return row['kind'] in ('grasp','cleanup')
    return False


def needs_grasp_preparation(record):
    return any(row['kind'] == 'grasp' for row in (record or {}).get('steps', []))


def worker_required_now(owner):
    """只按当前阶段判断视觉依赖，不用后续抓取步骤取消正在进行的导航。"""
    visual=getattr(getattr(owner,'studio',None),'visual',None)
    if visual is not None and visual.busy:return True
    automatic=getattr(owner,'automatic',None)
    if automatic is not None and automatic.active and automatic.phase=='preparing':return True
    flow=getattr(owner,'workflow',None)
    if flow is not None and flow.active:
        if flow.phase in ('prepare','scan','plan','confirm'):return True
        return flow.phase=='place' and owner.demo_phase=='planning'
    # 独立抓放演示仍保留原有退出处理；这里只缩小编排任务的依赖范围。
    return bool(getattr(owner,'demo_active',False) or getattr(owner,'placement_active',False))
