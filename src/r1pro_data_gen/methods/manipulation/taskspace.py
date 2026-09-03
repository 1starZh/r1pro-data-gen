"""Task-space trajectory planning (paths via iterative IK).

Task-space planning is the reliable way to move the arm through a cluttered
scene (joint-space interpolation swings links through obstacles). A path is a
sequence of joint configurations where the end-effector follows a straight
line in position while the orientation interpolates (slerp) toward the target;
each step is one DLS IK solve seeded from the previous step.

This module is pure logic (pinocchio only) -- a *method* that skills call
internally; planners never invoke it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# A chained DLS step that stays on one redundant branch moves far less than
# this between 2 cm Cartesian samples. A jump above the gate is an IK-seed
# branch change, which is exactly the high-torque wrist snap we must reject.
_MAX_CONSECUTIVE_JOINT_STEP_RAD = 0.50
_CARTESIAN_STEP_M = 0.02
_VIA_LIFT_M = 0.12
_VIA_XY_M = 0.05


@dataclass(frozen=True, slots=True)
class TaskPath:
    """A planned task-space path as a list of joint configurations."""

    waypoints: tuple[np.ndarray, ...]
    success: bool
    final_position_error: float
    final_rotation_error: float
    notes: str = field(default="", kw_only=True)


def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two quaternions (w, x, y, z)."""
    rots = Rotation.from_quat(
        [[q1[1], q1[2], q1[3], q1[0]], [q2[1], q2[2], q2[3], q2[0]]]
    )
    q = Slerp([0, 1], rots)(t).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _skew(omega: np.ndarray) -> np.ndarray:
    x, y, z = (float(omega[0]), float(omega[1]), float(omega[2]))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    matrix[:3, 3] = np.asarray(pos, dtype=float)
    return matrix


def _pose_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return matrix[:3, 3].copy(), np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _se3_log(matrix: np.ndarray) -> np.ndarray:
    rotvec = Rotation.from_matrix(matrix[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(rotvec))
    translation = np.asarray(matrix[:3, 3], dtype=float)
    twist = np.zeros(6, dtype=float)
    if theta < 1e-9:
        twist[:3] = translation
        return twist
    skew = _skew(rotvec)
    theta2 = theta * theta
    inverse = (
        np.eye(3)
        - 0.5 * skew
        + (1.0 / theta2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
        * (skew @ skew)
    )
    twist[:3] = inverse @ translation
    twist[3:] = rotvec
    return twist


def _se3_exp(twist: np.ndarray) -> np.ndarray:
    linear = np.asarray(twist[:3], dtype=float)
    rotvec = np.asarray(twist[3:], dtype=float)
    theta = float(np.linalg.norm(rotvec))
    matrix = np.eye(4)
    if theta < 1e-9:
        matrix[:3, 3] = linear
        return matrix
    matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    skew = _skew(rotvec)
    theta2 = theta * theta
    coupling = (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta2 * skew
        + (theta - np.sin(theta)) / (theta2 * theta) * (skew @ skew)
    )
    matrix[:3, 3] = coupling @ linear
    return matrix


def _screw_pose(pos0, quat0, pos1, quat1, t: float) -> tuple[np.ndarray, np.ndarray]:
    start = _pose_matrix(pos0, quat0)
    goal = _pose_matrix(pos1, quat1)
    interpolated = start @ _se3_exp(float(t) * _se3_log(np.linalg.inv(start) @ goal))
    return _pose_from_matrix(interpolated)


def _task_path_steps(start_pos: np.ndarray, target_pos: np.ndarray, n: int | None) -> int:
    if n is not None:
        return max(2, int(n))
    distance = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - np.asarray(start_pos, dtype=float)))
    if not np.isfinite(distance):
        return 8
    return min(80, max(8, int(np.ceil(distance / _CARTESIAN_STEP_M))))


def _solve_continuous_ik(kin: Any, pos_i: np.ndarray, quat_i: np.ndarray, q_cur: np.ndarray):
    """Stay on the live branch: one DLS step, then public IK if that stalls."""
    sol = None
    if hasattr(kin, "_ik_once"):
        sol = kin._ik_once(pos_i, quat_i, q_cur)
        if (
            sol is not None
            and getattr(sol, "success", False)
            and sol.q_arm is not None
            and float(np.max(np.abs(np.asarray(sol.q_arm, dtype=float) - q_cur)))
            <= _MAX_CONSECUTIVE_JOINT_STEP_RAD
        ):
            return sol
    if not hasattr(kin, "ik"):
        return sol
    robust = kin.ik(pos_i, quat_i, q_init=q_cur)
    if robust is None or robust.q_arm is None:
        return sol if sol is not None else robust
    if (
        sol is not None
        and getattr(sol, "success", False)
        and sol.q_arm is not None
        and float(np.max(np.abs(np.asarray(robust.q_arm, dtype=float) - q_cur)))
        > float(np.max(np.abs(np.asarray(sol.q_arm, dtype=float) - q_cur)))
    ):
        return sol
    return robust


def plan_task_path(
    kin,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None,
    q_start: np.ndarray,
    n: int | None = None,
    pos_tol: float = 0.03,
    rot_tol: float = 0.1,
    soft_pos: float = 0.08,
    soft_rot: float = 0.3,
    max_joint_step_rad: float = _MAX_CONSECUTIVE_JOINT_STEP_RAD,
    interp: str = "screw",
) -> TaskPath:
    """Plan a task-space path from ``q_start`` to the target pose.

    Default interpolation is an SE(3) screw (mplib ``plan_screw`` / ScLERP):
    the end-effector follows the relative twist instead of independently
    lerping translation and slerping orientation. ``interp="cartesian"`` keeps
    the older decoupled line for via segments. Each sample is IK-seeded from
    the previous joint configuration so the path stays on one redundant branch.

    ``target_quat`` may be omitted to hold the start orientation. A consecutive
    joint jump above ``max_joint_step_rad`` is treated as an IK-branch change
    and aborts the interpolant so a later planner can take over.
    """
    del pos_tol, rot_tol
    if interp not in {"screw", "cartesian"}:
        raise ValueError("plan_task_path interp must be 'screw' or 'cartesian'")
    pos0, quat0 = kin.fk(q_start)
    pos0 = np.asarray(pos0, dtype=float)
    quat0 = np.asarray(quat0, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    if target_quat is None:
        target_quat = np.asarray(quat0, dtype=float)
    else:
        target_quat = np.asarray(target_quat, dtype=float)
        target_quat = target_quat / max(float(np.linalg.norm(target_quat)), 1e-12)
    steps = _task_path_steps(pos0, target_pos, n)
    waypoints = [np.asarray(q_start, dtype=float).copy()]
    q_cur = np.asarray(q_start, dtype=float).copy()
    for i in range(1, steps + 1):
        t = i / steps
        if interp == "screw":
            pos_i, quat_i = _screw_pose(pos0, quat0, target_pos, target_quat, t)
        else:
            pos_i = pos0 + (target_pos - pos0) * t
            quat_i = _slerp(quat0, target_quat, t)
        sol = _solve_continuous_ik(kin, pos_i, quat_i, q_cur)
        if sol is None or sol.q_arm is None:
            return TaskPath(
                waypoints=tuple(waypoints),
                success=False,
                final_position_error=float("inf"),
                final_rotation_error=float("inf"),
                notes=f"IK failed at step {i}/{steps}",
            )
        if not sol.success:
            if sol.position_error > soft_pos or sol.rotation_error > soft_rot:
                return TaskPath(
                    waypoints=tuple(waypoints),
                    success=False,
                    final_position_error=sol.position_error,
                    final_rotation_error=sol.rotation_error,
                    notes=f"IK failed at step {i}/{steps}",
                )
        q_next = np.asarray(sol.q_arm, dtype=float)
        if float(np.max(np.abs(q_next - q_cur))) > float(max_joint_step_rad):
            return TaskPath(
                waypoints=tuple(waypoints),
                success=False,
                final_position_error=float(getattr(sol, "position_error", 0.0)),
                final_rotation_error=float(getattr(sol, "rotation_error", 0.0)),
                notes=f"IK branch jump at step {i}/{steps}",
            )
        q_cur = q_next
        waypoints.append(q_cur.copy())

    pos, quat = kin.fk(q_cur)
    return TaskPath(
        waypoints=tuple(waypoints),
        success=True,
        final_position_error=float(np.linalg.norm(target_pos - pos)),
        final_rotation_error=float(
            np.linalg.norm(_slerp_angle(target_quat, quat))
        ),
    )


def _stack_waypoints(path: TaskPath) -> np.ndarray | None:
    if len(path.waypoints) < 2:
        return None
    stacked = np.asarray(path.waypoints, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(stacked, axis=0), axis=1) > 1e-8]
    stacked = stacked[keep]
    if len(stacked) < 2:
        return None
    return stacked


def _plan_straight_waypoints(
    kin: Any,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None,
    q_start: np.ndarray,
) -> np.ndarray | None:
    planned = plan_task_path(kin, target_pos, target_quat, q_start)
    if not planned.success:
        return None
    return _stack_waypoints(planned)


def _plan_via_waypoints(
    kin: Any,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None,
    q_start: np.ndarray,
    *,
    lift_m: float = _VIA_LIFT_M,
) -> np.ndarray | None:
    """One geometric path: hold height to the goal XY, then descend.

    The via is not a second executed skill. Both segments are concatenated
    before retiming so the arm still follows a single C2 reference.
    """
    pos0, quat0 = kin.fk(q_start)
    target_pos = np.asarray(target_pos, dtype=float)
    via_z = max(float(pos0[2]), float(target_pos[2]) + float(lift_m))
    via_pos = np.asarray([target_pos[0], target_pos[1], via_z], dtype=float)
    xy = float(np.linalg.norm(via_pos[:2] - np.asarray(pos0[:2], dtype=float)))
    dz = abs(via_z - float(pos0[2]))
    if xy < _VIA_XY_M and dz < 0.02:
        return None
    hold_quat = np.asarray(quat0, dtype=float)
    to_via = plan_task_path(kin, via_pos, hold_quat, q_start)
    if not to_via.success:
        return None
    via_q = np.asarray(to_via.waypoints[-1], dtype=float)
    to_goal = plan_task_path(kin, target_pos, target_quat, via_q)
    if not to_goal.success:
        return None
    stacked = _stack_waypoints(
        TaskPath(
            waypoints=tuple(to_via.waypoints) + tuple(to_goal.waypoints[1:]),
            success=True,
            final_position_error=to_goal.final_position_error,
            final_rotation_error=to_goal.final_rotation_error,
        )
    )
    return stacked


def plan_certified_task_path(
    planner: Any,
    kin: Any,
    q_current: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None,
    scene: Any,
    *,
    base_xy: tuple[float, float],
    base_yaw: float,
    full_q_current: np.ndarray,
    speed_scale: float,
    side: str,
) -> dict[str, Any]:
    """Collision-check and retime a chained-IK Cartesian interpolant.

    Follows mplib's recommended pick-place order: try an SE(3) screw first
    (straighter than joint-space OMPL, no IK-branch snap), then one raised-via
    polyline if the screw collides. Both attempts remain one geometric path
    and one retimed trajectory.
    """
    from r1pro_data_gen.methods.manipulation.mplib_path import retime_and_validate_path

    if not hasattr(kin, "fk") or not hasattr(kin, "ik"):
        return {
            "success": False,
            "status": "TaskSpaceUnavailable",
            "reason": "kinematics backend cannot chain task-space IK",
            "failure_stage": "task_space",
        }
    q_current = np.asarray(q_current, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    builders = (
        lambda: _plan_straight_waypoints(kin, target_pos, target_quat, q_current),
        lambda: _plan_via_waypoints(kin, target_pos, target_quat, q_current),
    )
    last_failure = {
        "success": False,
        "status": "TaskSpaceFailed",
        "reason": "no continuous Cartesian interpolant reached the goal",
        "failure_stage": "task_space",
    }
    for builder in builders:
        geometric = builder()
        if geometric is None:
            continue
        certified = retime_and_validate_path(
            planner,
            geometric,
            scene,
            base_xy=base_xy,
            base_yaw=base_yaw,
            kin=kin,
            speed_scale=float(speed_scale),
            side=side,
            full_q_current=full_q_current,
        )
        if certified.get("success"):
            certified = dict(certified)
            certified["status"] = "TaskSpaceVerified"
            certified["reason"] = None
            return certified
        last_failure = {
            "success": False,
            "status": str(certified.get("status", "TaskSpaceFailed")),
            "reason": str(certified.get("reason") or "Cartesian interpolant rejected"),
            "failure_stage": str(certified.get("failure_stage", "task_space")),
        }
    return last_failure


def _slerp_angle(q_ref: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Rotation vector from q_cur to q_ref (small helper, keeps module self-contained)."""
    from r1pro_data_gen.robot.kinematics import _quat_error_rotation_vector

    return _quat_error_rotation_vector(q_ref, q_cur)


__all__ = ["TaskPath", "plan_certified_task_path", "plan_task_path"]
