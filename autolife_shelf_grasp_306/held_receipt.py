"""Session-local payload evidence. A confirmed release survives a later retreat fault."""
import json
from .core import atomic_json


def has_held(run):
    path=run/'held_object.json'
    if not path.is_file():return False
    # Older executors wrote an uncertain receipt AFTER reporting a completed
    # placement. Only that exact operation proves the old receipt obsolete.
    try:
        held=json.loads(path.read_text())
        if held.get('status')=='uncertain' and held.get('plan_id'):
            result=json.loads((run/'execution_result.json').read_text())
            if (result.get('status')=='placed' and result.get('plan_id')==held['plan_id']
                    and 'release_command_started' in result.get('effects',[])):
                return False
    except (OSError,ValueError,AttributeError):
        pass
    return True


def current_payload(run):
    try:return json.loads((run/'held_object.json').read_text())
    except (OSError,ValueError) as exc:raise ValueError('当前没有有效的抓取记录') from exc


def begin_grasp_opening(run,arm):
    if not has_held(run):return
    held=current_payload(run)
    if held.get('arm','left')==arm:
        from .task_pause import archive_held
        archive_held(run)


def release_payload(run,plan,open_gripper):
    path=run/'held_object.json'
    parent=plan.get('parent_plan_id') or plan.get('grasp_plan_id')
    if parent and (not path.exists() or json.loads(path.read_text()).get('plan_id')!=parent):
        raise ValueError('放置引用的抓取已更新，请使用最近一次抓取重新规划放置')
    atomic_json(path,dict(status='uncertain',plan_id=plan['plan_id'],arm=plan.get('arm'),
                          grasp_plan_id=plan.get('grasp_plan_id'),reason='松爪结果尚未确认'))
    open_gripper()
    path.unlink()
