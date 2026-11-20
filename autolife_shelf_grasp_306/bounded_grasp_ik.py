"""Bounded numerical seed recovery for KDL; no robot I/O or limit relaxation."""
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .planning_control import VENDOR_IK_POSITION_TOLERANCE_M


POSITION_WEIGHT = 10.


class BoundedPoseResidual:
    """Six-dimensional TCP residual and analytic Jacobian in the fixed task frame."""
    def __init__(self, planner, source, tcp, point, rotation):
        self.planner = planner
        self.model = planner.pose_model
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        for name, index in planner.full_pose_indices:
            self.q[index] = np.deg2rad(source.get(name, 0.))
        joints = [self.model.joints[self.model.getJointId(n)] for n in planner.names]
        self.q_indices = [j.idx_q for j in joints]
        self.v_indices = [j.idx_v for j in joints]
        self.tcp = np.asarray(tcp, float)
        self.point = np.asarray(point, float)
        self.rotation = np.asarray(rotation, float)
        self.world_to_task = planner.anchor[:3, :3].T
        self.origin = planner.anchor[:3, 3]
        self.last = None

    def _evaluate(self, x):
        self.planner.context.checkpoint()
        if self.last is not None and np.array_equal(x, self.last):
            return
        self.q[self.q_indices] = x
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        wrist = self.data.oMf[self.planner.wrist_frame]
        self.world_rotation = wrist.rotation.copy()
        self.actual_rotation = self.world_to_task @ self.world_rotation
        self.actual_point = self.world_to_task @ (
            wrist.translation + self.world_rotation @ self.tcp - self.origin)
        self.error_rotation = self.rotation.T @ self.actual_rotation
        self.last = np.array(x, copy=True)

    def fun(self, x):
        self._evaluate(x)
        return np.r_[POSITION_WEIGHT * (self.actual_point - self.point),
                     Rotation.from_matrix(self.error_rotation).as_rotvec()]

    def jac(self, x):
        self._evaluate(x)
        j = pin.computeFrameJacobian(self.model, self.data, self.q,
            self.planner.wrist_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, self.v_indices]
        lever = self.world_rotation @ self.tcp
        linear = self.world_to_task @ (j[:3] - pin.skew(lever) @ j[3:])
        angular = (pin.Jlog3(self.error_rotation) @ self.actual_rotation.T
                   @ self.world_to_task @ j[3:])
        return np.vstack((POSITION_WEIGHT * linear, angular))


def recover_seed(planner, source, tcp, point, rotation, seed, lower, upper):
    """Find a bounded seed; the caller must re-solve and verify it with vendor KDL."""
    from .planning import IKFailure
    residual = BoundedPoseResidual(planner, source, tcp, point, rotation)
    def errors(x):
        if not np.isfinite(x).all() or np.any(x < lower) or np.any(x > upper):
            raise IKFailure('有界初值含无效或超限的关节')
        error = residual.fun(x)
        if not np.isfinite(error).all():
            raise IKFailure('有界初值的位姿残差无效')
        return np.linalg.norm(error[:3]) / POSITION_WEIGHT, np.linalg.norm(error[3:])

    initial = np.clip(np.asarray(seed, float), lower, upper)
    weight = .002
    fun = lambda x: np.r_[residual.fun(x), weight * (x - initial)]
    jac = lambda x: np.vstack((residual.jac(x), weight * np.eye(len(x))))
    biased = least_squares(fun, initial, jac=jac, bounds=(lower, upper),
        max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    position, angle = errors(biased.x)
    if position <= 1e-6 and angle <= 1e-5:
        return biased.x
    result = least_squares(residual.fun, biased.x, jac=residual.jac,
        bounds=(lower, upper), max_nfev=50, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    position, angle = errors(result.x)
    # Only a seed for a second KDL solve. Match the caller's final TCP FK envelope.
    if position > VENDOR_IK_POSITION_TOLERANCE_M or angle > np.deg2rad(2.):
        raise IKFailure('有界初值搜索未找到满足位置、姿态和原限位的解')
    return result.x
