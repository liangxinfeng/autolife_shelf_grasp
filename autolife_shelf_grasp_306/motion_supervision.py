"""Shared execution supervision and honest dispatch receipts; no ROS imports.

Each executor pumps callbacks, then checks its operator lease, control session
and base once. Physical entry/arrival checks still run at their own boundaries.
"""
from dataclasses import dataclass, field
from pathlib import Path
import time
import math


class MotionFault(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class MotionSupervisor:
    def __init__(self, run, wall_clock=None):
        self.run = Path(run)
        self.wall_clock = wall_clock or time.time

    def check_operator(self):
        if (self.run / 'stop').exists():
            raise MotionFault('OPERATOR_STOPPED', '用户停止')
        try:
            age = self.wall_clock() - (self.run / 'heartbeat').stat().st_mtime
        except OSError:
            age = float('inf')
        if not math.isfinite(age) or age < -1. or age > 1.:
            raise MotionFault('OPERATOR_HEARTBEAT_LOST', '界面心跳中断')

    def poll(self, pump, session=None, base=None):
        pump()
        self.check_operator()
        if session is not None:
            session.tick()
        if base is not None and base.last is not None:
            base.check()

    @staticmethod
    def snapshot(node):
        q = node.snapshot()
        if not node._hardware_ready:
            raise MotionFault('CONTROL_UNAVAILABLE',
                '控制权/硬件状态失效：' + getattr(node, 'last_lock_reason', '请检查驱动状态'))
        return q


@dataclass
class ExecutionProgress:
    """Record before dispatch: a thrown/lost reply never means no movement.

Command completion is deliberately not a claim of physical grip/release.
The list retains earlier gripper effects while the arm subsequently retreats.
"""
    phase: str = 'preflight'
    effects: list = field(default_factory=list)
    execution_id: object = None
    on_update: object = field(default=None, repr=False)

    def _notify(self):
        if self.on_update is not None:
            self.on_update(self.receipt('running'))

    def enter(self, phase):
        if self.phase != phase:
            self.phase = phase
            self._notify()

    def dispatch(self, effect):
        if effect not in self.effects:
            self.effects.append(effect)
            self._notify()

    def receipt(self, status, fault=None):
        complete = status in ('prepared', 'completed', 'placed', 'observed')
        certainty = ('command_completed' if complete and self.effects else
                     'may_have_executed' if self.effects else 'not_dispatched')
        code = getattr(fault, 'code', None) if fault is not None else None
        return dict(execution_id=self.execution_id,
            execution_phase='completed' if complete else self.phase, effects=list(self.effects),
            effect_state=self.effects[-1] if self.effects else 'none',
            effect_certainty=certainty,
            safe_to_retry=not complete and not self.effects and code in (
                'PLAN_EXPIRED', 'SOURCE_POSE_CHANGED','FEEDBACK_UNAVAILABLE'),
            reason_code=code)


def attempt_cleanup(errors, stage, operation):
    """One failed release must not skip the remaining releases or the receipt."""
    try:
        if operation() is False:
            raise ValueError('操作未确认')
        return True
    except Exception as exc:
        errors.append(dict(stage=stage, error=str(exc)))
        try:print('退出清理提示：'+stage+'：'+str(exc), flush=True)
        except OSError:pass  # A closed UI log pipe must not interrupt cleanup.
        return False
