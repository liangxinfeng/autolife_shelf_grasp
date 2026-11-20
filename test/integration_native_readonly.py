"""Real model acceptance: saved pose -> FK -> vendor IK, without robot I/O.

Run with the deployed computation environment:
  python test/integration_native_readonly.py --root /path/to/package

Only the FK/IK worker subprocess is created. No ROS node, controller, trajectory
executor, or motion publisher is constructed. stdout is one auditable JSON.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def inspect_model(root, report):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from autolife_handeye_calibration_306.core import ARMS, BODY, NECK, Kinematics
    from autolife_shelf_grasp_306.runtime_config import load_config
    from autolife_shelf_grasp_306.calibration_profiles import CalibrationProfiles
    from autolife_shelf_grasp_306.arm_selection import tool_geometry
    from autolife_shelf_grasp_306.grasp_collision import GROUPS
    from autolife_shelf_grasp_306.planning_control import PlanningControl
    from autolife_shelf_grasp_306.vendor_grasp_planning import VendorGraspPlanner

    cfg = load_config(root)
    pose_path = root / 'data' / 'fixed_actions' / 'manual_reset_pose.json'
    saved = json.loads(pose_path.read_text('utf8'))
    source = {name: float(value) for name, value in saved['joints_deg'].items()}
    required = BODY + NECK + ARMS['left'] + ARMS['right'] + [
        'Joint_Left_Gripper', 'Joint_Right_Gripper']
    missing = sorted(set(required) - source.keys())
    if missing or not all(np.isfinite(value) for value in source.values()):
        raise ValueError('Saved pose has missing or non-finite joints: ' + str(missing))
    model = Path(cfg['urdf_path'])
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    if saved.get('urdf_sha256') != model_sha:
        raise ValueError('Saved pose and configured URDF checksums differ')
    report.update(model_path=str(model), model_sha256=model_sha,
        pose_path=str(pose_path), pose_sha256=hashlib.sha256(pose_path.read_bytes()).hexdigest(),
        configured_robot_id=cfg.get('robot_id'), pose_source_robot_id=saved.get('robot_id'),
        source_joints_deg=source)
    # The pose's robot number is provenance, not an identity permission check.
    fk = Kinematics(str(model))
    profiles = CalibrationProfiles(cfg, fk)
    for side in ('left', 'right'):
        started = time.monotonic()
        item = dict(arm=side, status='error', stage='geometry')
        report['arms'].append(item)
        planner = None
        try:
            side_cfg, calibration, calibration_sha = profiles.for_arm(side)
            side_cfg, tcp = tool_geometry(side_cfg, calibration, fk, side)
            item.update(calibration_sha256=calibration_sha,
                geometry_source=calibration.get('geometry_source', 'calibrated'),
                tcp_m=np.asarray(tcp, float).tolist())
            item['stage'] = 'load_collision_and_vendor_ik'
            planner = VendorGraspPlanner(side_cfg, context=PlanningControl(timeout_sec=60.))
            planner.configure_source(source)  # Includes independent FK/chain-order check.
            item['controlled_joints'] = list(planner.names)
            item['solver'] = 'autolife_robot_sdk.KdlSolver.nr_jl'

            item['stage'] = 'source_collision'
            guard = planner.guard
            guard.begin_fixed_neck(source)
            collision_q = guard.kin.q_from_feedback(
                {group: [source[name] for name in names] for group, names in GROUPS.items()},
                clip=False)
            item['raw_collision_pairs'] = sorted(map(str, guard.kin.collision_pairs(collision_q)))
            item['registered_static_neck_pairs'] = sorted(map(str, guard.static_neck_pairs))
            item['collision_geometry_count'] = len(guard.kin.geom_model.geometryObjects)
            guard.check(source)

            item['stage'] = 'same_pose_fk_ik'
            point, rotation = planner.pose(source, tcp)
            solved = planner.solve(source, tcp, point, rotation)
            actual_point, actual_rotation = planner.pose(solved, tcp)
            position_error = float(np.linalg.norm(actual_point - point))
            orientation_error = float(np.rad2deg(Rotation.from_matrix(
                rotation.T @ actual_rotation).magnitude()))
            if not np.isfinite([position_error, orientation_error]).all():
                raise ValueError('FK/IK residual is non-finite')
            # These are the planner's existing acceptance limits, not a relaxed
            # integration-test tolerance. Exact residuals are reported below.
            if position_error > .003 or orientation_error > 2.:
                raise ValueError('FK/IK residual exceeds the planner acceptance limits')
            changed_fixed = [name for name in source if name not in planner.names
                             and solved[name] != source[name]]
            if changed_fixed:
                raise ValueError('IK changed fixed joints: ' + str(changed_fixed))
            guard.check(solved)
            item.update(status='ok', stage='complete',
                source_tcp_m=np.asarray(point, float).tolist(),
                position_error_m=position_error, orientation_error_deg=orientation_error,
                max_controlled_joint_change_deg=max(abs(solved[n] - source[n]) for n in planner.names),
                solved_joints_deg=solved, fixed_joints_unchanged=True,
                source_and_solution_collision_checked=True)
        except Exception as exc:
            item.update(error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc(file=sys.stderr)
        finally:
            if planner is not None:
                planner.close()
            item['elapsed_sec'] = round(time.monotonic() - started, 4)
    report['ok'] = len(report['arms']) == 2 and all(item['status'] == 'ok' for item in report['arms'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='Optional duplicate JSON report file')
    args = parser.parse_args()
    started = time.monotonic()
    report = dict(schema_version=1, check='native_model_readonly', ok=False,
        root=str(args.root.resolve()), hardware_execution=False,
        scope='saved-pose geometry, collision, and FK/IK only; no motion path or physical validation',
        arms=[])
    try:
        # Native libraries may print load diagnostics. Keep stdout parseable.
        with contextlib.redirect_stdout(sys.stderr):
            inspect_model(args.root.resolve(), report)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc(file=sys.stderr)
    report['elapsed_sec'] = round(time.monotonic() - started, 4)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(encoded + '\n', encoding='utf8')
    print(encoded)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
