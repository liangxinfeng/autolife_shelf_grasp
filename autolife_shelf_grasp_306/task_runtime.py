"""Durable task progress, independent of Qt, ROS and motor execution.

An attempt identity fences late results. A task identity survives retries.
Only an explicitly confirmed result advances a step; an unknown physical
outcome remains waiting. Reopening a process never dispatches an action.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import uuid

SESSION_PREFIX = 'session_'


class TaskStore:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def read(self):
        return json.loads(self.path.read_text('utf-8')) if self.path.exists() else None


class TaskRuntime:
    RETRIES = 2

    def __init__(self, record, *, task_id=None, cycle_index=0, store=None, metadata=None):
        self.record = deepcopy(record)
        self.task_id = task_id or uuid.uuid4().hex
        self.cycle_index = cycle_index
        self.run_id = uuid.uuid4().hex  # Existing transport token remains run:index.
        self.attempt_id = 0
        self.index = 0
        self.state = 'running'
        self.reason = ''
        self.resume = 'operator'
        self.retry_count = 0
        self.retry_at = 0.
        self.effect = 'none'
        self.phase = ''
        self.metadata = deepcopy(metadata or {})
        self.warnings = []
        self.store = store
        self.persist()

    @property
    def active(self):
        return self.state == 'running'

    @property
    def pending(self):
        return self.state in ('running', 'waiting', 'paused')

    @property
    def token(self):
        return f'{self.run_id}:{self.index}'

    @property
    def step_id(self):
        return f'{self.task_id}:{self.cycle_index}:{self.index}'

    @property
    def step(self):
        return self.record['steps'][self.index]

    def snapshot(self):
        return deepcopy(dict(version=1, record=self.record, task_id=self.task_id,
                    cycle_index=self.cycle_index, run_id=self.run_id,
                    attempt_id=self.attempt_id, index=self.index, state=self.state,
                    reason=self.reason, resume=self.resume, retry_count=self.retry_count,
                    effect=self.effect, phase=self.phase, metadata=self.metadata,
                    warnings=self.warnings, updated_at=time.time()))

    def persist(self):
        if self.store is not None:
            self.store.save(self.snapshot())

    def dispatch(self, phase):
        if not self.active:
            raise ValueError('任务未运行，禁止下发运动')
        self.phase = phase
        self.effect = 'unknown'
        self.persist()  # Intent must be durable BEFORE sending a physical command.

    def complete(self, token):
        if not self.active or token != self.token:
            return False
        previous = (self.index, self.retry_count, self.effect, self.phase, self.reason, self.state)
        self.index += 1
        self.retry_count = 0
        self.effect = 'none'
        self.phase = ''
        self.reason = ''
        self.state = 'completed' if self.index == len(self.record['steps']) else 'running'
        try:
            self.persist()
        except OSError:
            self.index, self.retry_count, self.effect, self.phase, self.reason, self.state = previous
            raise
        return True

    def wait(self, reason, *, retryable=False, await_input=False, now=None):
        if not self.pending:
            return False
        now = time.monotonic() if now is None else now
        self.state = 'waiting'
        self.reason = str(reason)
        self.resume = ('automatic' if self.effect == 'none' and
                       (await_input or (retryable and self.retry_count < self.RETRIES)) else 'operator')
        self.retry_at = now + 2. ** min(2, self.retry_count)
        self.persist()
        return True

    def retry(self, *, confirmed=False):
        if self.state != 'waiting':
            raise ValueError('没有等待恢复的步骤')
        if not confirmed and (self.resume != 'automatic' or self.effect != 'none'):
            raise ValueError('本次执行结果需要确认，禁止自动重复动作')
        self.attempt_id += 1
        self.run_id = uuid.uuid4().hex
        self.retry_count = 0 if confirmed else self.retry_count + 1
        self.state = 'running'
        self.reason = ''
        self.effect = 'none'
        self.persist()

    def cancel(self, reason):
        if self.pending:
            self.state = 'cancelled'
            self.reason = str(reason)
            self.resume = 'operator'
            self.persist()

    def pause(self):
        if self.state == 'running':
            self.state = 'paused'
            self.persist()

    def unpause(self):
        if self.state == 'paused':
            self.state = 'running'
            try:
                self.persist()
            except OSError:
                self.state = 'paused'
                raise

    def warn(self, reason):
        self.warnings.append(dict(step=self.index, reason=str(reason)))
        self.warnings = self.warnings[-100:]
        # Optional diagnostics must not gate the next step. The completion
        # transition immediately persists the warning alongside progress.

    @classmethod
    def restore(cls, store):
        data = store.read()
        if not data or data.get('state') == 'cancelled':
            return None
        if data.get('state') == 'completed':
            automatic = data.get('metadata', {}).get('automatic')
            next_cycle = data['cycle_index'] + 1
            if not automatic or (automatic['limit'] and next_cycle >= automatic['limit']):
                return None
            # Completion and the next interval are two UI callbacks. Recover
            # their crash window without replaying the completed round.
            data.update(state='waiting', index=0, cycle_index=next_cycle,
                        effect='none', phase='cycle_interval', retry_count=0)
            automatic['completed'] = next_cycle
        if data.get('version') != 1 or not 0 <= data['index'] < len(data['record']['steps']):
            raise ValueError('保存的任务状态无效，未自动启动')
        obj = cls.__new__(cls)
        for key in ('record', 'task_id', 'cycle_index', 'run_id', 'attempt_id', 'index',
                    'retry_count', 'effect', 'phase', 'metadata', 'warnings'):
            setattr(obj, key, deepcopy(data[key]))
        obj.state = 'waiting'
        obj.reason = '程序重新启动；请确认现场及上一动作结果后恢复任务'
        obj.resume = 'operator'
        obj.retry_at = 0.
        obj.store = store
        obj.persist()
        return obj
