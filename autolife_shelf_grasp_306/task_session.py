"""Recover evidence into a NEW control session; never renew an orphan's lease."""
from copy import deepcopy
from pathlib import Path
import tempfile
from .core import atomic_json
from .task_runtime import SESSION_PREFIX, TaskRuntime, TaskStore


def copy_held_evidence(source, destination):
    """Only the confirmed grasp and its plan are placement evidence.

    The full grasp plan is needed for placement orientation/carry geometry.
    No prepared receipt, observation, motion request or heartbeat is imported.
    The archived source is untouched; the copy is scoped to the new session.
    """
    from .placement_request import held_grasp
    from .held_receipt import has_held
    if not has_held(source):return False
    if (source/'held_object.json').resolve().parent!=source:
        raise ValueError('持物记录不在原会话目录中')
    grasp,receipt=held_grasp(source)
    name='plan_'+grasp['plan_id']+'.json'
    if (source/name).resolve().parent!=source:
        raise ValueError('抓取计划不在原会话目录中')
    grasp=deepcopy(grasp);receipt=deepcopy(receipt)
    origin=grasp.get('recovered_from_session',source.name)
    grasp.update(run_id=destination.name,recovered_from_session=origin)
    receipt['recovered_from_session']=origin
    atomic_json(destination/name,grasp)
    atomic_json(destination/'held_object.json',receipt)
    held_grasp(destination)  # Validate the pair before committing the new pointer.
    return True


def expects_held_evidence(task):
    last=task.metadata.get('cleanup_result',{})
    if task.step['kind']=='place':
        return last.get('index')!=task.index-1 or last.get('picked') is not None
    if task.step['kind']=='action' and task.metadata.get('action_entry_step')==task.step_id:
        return False  # Its entry policy was durably captured before consuming the receipt.
    previous=task.record['steps'][:task.index]
    if not previous and task.cycle_index>0:previous=task.record['steps']
    operations=[row['kind'] for row in previous if row['kind'] in ('grasp','cleanup','place','action')]
    if operations and operations[-1]=='cleanup':return last.get('picked') is not None
    return bool(operations and operations[-1]=='grasp')


def open_session(root):
    directory=(Path(root)/'data/sessions').resolve()
    directory.mkdir(parents=True,exist_ok=True)
    state=Path(root)/'data/task_runtime.json'
    # A fast restart cannot refresh the heartbeat read by an orphaned process.
    fresh=Path(tempfile.mkdtemp(prefix=SESSION_PREFIX,dir=directory))
    warning=''
    if state.exists():
        try:
            pending=TaskRuntime.restore(TaskStore(state))
            if pending is not None:
                name=pending.metadata.get('session_name','')
                if name and name not in ('.','..') and '/' not in name and '\\' not in name:
                    previous=(directory/name).resolve()
                    if previous.parent==directory and previous.is_dir():
                        (previous/'stop').touch()
                        copied=copy_held_evidence(previous,fresh)
                        if expects_held_evidence(pending) and not copied:
                            raise ValueError('缺少已完成抓取的持物来源，未恢复放置凭据')
                        pending.metadata['session_name']=fresh.name
                        pending.persist()  # Still operator waiting; no action is queued.
                        return fresh,''
                warning='任务已保留，但原会话记录缺失；请核对持物和动作结果后处理，未自动执行。'
        except (OSError,ValueError,TypeError,KeyError) as exc:
            warning='任务恢复资料不可用，未自动执行：'+str(exc)
    return fresh,warning
