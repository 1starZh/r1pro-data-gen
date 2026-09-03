"""Base motion skills: move the 3-wheel omni chassis.

The R1Pro base is a 3-wheel steer-and-drive chassis: ``base_move_to`` drives
straight to any world pose (x, y, yaw) with closed-loop P control;
``base_rotate_to`` turns in place; ``base_navigate_to`` plans a collision-free
2D path (grid A*) around static scene obstacles and follows it waypoint by
waypoint; ``base_lock_wheels`` / ``base_unlock_wheels`` freeze or free the
wheel drives (used around arm-manipulation phases).

All targets are world coordinates; the skills are task-agnostic -- any reachable
pose can be passed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from r1pro_data_gen.robot import shortest_steer_command, wheel_commands
from r1pro_data_gen.robot.robot_config import (
    R1PRO_BASE_ANG_ACCEL_MAX,
    R1PRO_BASE_LIN_ACCEL_MAX,
    R1PRO_BASE_OMEGA_MAX,
    R1PRO_BASE_V_MAX,
)
from r1pro_data_gen.domain import object_xy_half_extents_m
from r1pro_data_gen.planning.navigation.contract import NAVIGATION_INFLATION_CLEARANCE_M

from ..core.base import ParamSpec, SkillResult, release_skill_wheel_lock

# Indoor human-like chassis speeds for WBC training trajectories.
DEFAULT_V_MAX = R1PRO_BASE_V_MAX
DEFAULT_OMEGA_MAX = R1PRO_BASE_OMEGA_MAX
_DRIVE_DT_S = 1.0 / 60.0
# Drive at full wheel speed once every steer module is within this of the
# commanded angle. Larger errors fade wheel speed to zero by 90 deg so a
# 3-wheel reconfiguration does not scrape a wheel sideways off the floor.
_STEER_ALIGN_RAD = 0.40
# Follow a point this far ahead on the polyline so a 90 deg doorway corner
# is taken as an arc instead of chasing the residual of the current vertex.
_PATH_LOOKAHEAD_M = 0.50
# Stop and reverse only when the carrot is almost behind the chassis.
# 90 deg (1.57 rad) must keep some forward speed.
_TURN_IN_PLACE_HEADING_RAD = 2.0


def _slew(current: float, desired: float, max_accel: float, dt: float = _DRIVE_DT_S) -> float:
    """Limit one-step command change so the chassis does not skip a wheel."""
    delta = float(desired) - float(current)
    max_step = abs(float(max_accel)) * float(dt)
    if abs(delta) <= max_step:
        return float(desired)
    return float(current) + math.copysign(max_step, delta)


def _current_steer_angles(
    adapter: Any, steer_joints: tuple[str, ...]
) -> dict[str, float]:
    """Measured steer angles, or empty when the adapter has no joint state."""
    try:
        observation = adapter.read_observation(0.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    positions = getattr(observation, "joint_positions", {}) or {}
    return {
        name: float(positions[name])
        for name in steer_joints
        if name in positions
    }


def _steer_drive_scale(
    current_angles: Mapping[str, float],
    cmds: Mapping[str, Any],
    steer_joints: tuple[str, ...],
) -> float:
    """Scale wheel speed by how far the steer modules still have to turn."""
    errors: list[float] = []
    for name in steer_joints:
        if name not in current_angles:
            return 1.0
        errors.append(
            abs(_wrap_pi(float(cmds[name].steer_angle) - float(current_angles[name])))
        )
    max_error = max(errors) if errors else 0.0
    if max_error <= _STEER_ALIGN_RAD:
        return 1.0
    fade = (math.pi / 2.0) - _STEER_ALIGN_RAD
    if fade <= 1.0e-6:
        return 0.0
    return max(0.0, 1.0 - (max_error - _STEER_ALIGN_RAD) / fade)


def _stop_wheels_hold_steer(
    adapter: Any,
    steer_joints: tuple[str, ...],
    wheel_joints: tuple[str, ...],
    aux_position_fn: Callable[[], dict[str, float]] | None = None,
) -> None:
    """Zero wheel speed while holding the live steer angles.

    Snapping steer to zero at every stop reconfigures all three modules and
    is what lifted a wheel for ~0.27 s after an otherwise settled drive.
    """
    current = _current_steer_angles(adapter, steer_joints)
    position = {name: current.get(name, 0.0) for name in steer_joints}
    if aux_position_fn is not None:
        position.update(aux_position_fn() or {})
    adapter.set_targets(
        position=position,
        velocity={name: 0.0 for name in wheel_joints},
    )


def _initial_body_twist(adapter: Any) -> tuple[float, float, float]:
    """Start slew from the live body-frame twist, not from a zero command."""
    velocity = _read_base_velocity(adapter)
    if velocity is None:
        return (0.0, 0.0, 0.0)
    try:
        _, _, yaw = _read_base(adapter)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return (0.0, 0.0, float(velocity[2]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    vx_body = float(velocity[0]) * cosine + float(velocity[1]) * sine
    vy_body = -float(velocity[0]) * sine + float(velocity[1]) * cosine
    return (vx_body, vy_body, float(velocity[2]))


DEFAULT_ARRIVE_TOL = 0.02  # m / rad
DEFAULT_MAX_STEPS = 600

# The omni chassis settles heading only to a few degrees and drifts slightly
# during the parking hold; a navigation goal does not need sub-degree yaw.  This
# is the final-heading the base can physically hold, still below the room
# evaluator's 0.12 rad navigate margin so a held heading never sits on the edge.
_NAV_YAW_ACCEPT_RAD = 0.10
# Finish heading here when cheap; never grind toward 0.01 rad.
_NAV_YAW_FINISH_RAD = 0.05
# 1 cm / 0.01 rad final closure made the last metre look finished while the
# skill kept stepping in place for tens of seconds (yaw P-control cannot hold
# 0.01 rad). That pause is unusable as whole-body motion-control data.
_FINAL_XY_TOL_M = 0.04
_FINAL_CLOSURE_MAX_STEPS = 180
_ARM_READY_STEP_RAD = 0.03
# After the last drive command the chassis still has momentum. Returning
# immediately left |wz| ≈ 0.16 rad/s, which rotated the grasp base-frame
# target during approach. Brake until physically slow, never grind for tens
# of seconds.
_BRAKE_MAX_STEPS = 60
_BRAKE_LIN_MPS = 0.03
_BRAKE_ANG_RADPS = 0.05
# A blocked or already-near goal must not keep stepping in place for the
# full waypoint budget: those idle physics frames are recorded as video.
_STALL_WINDOW_STEPS = 90
_STALL_PROGRESS_M = 0.02


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class _ProgressWatchdog:
    """Abort closed-loop driving when the pose error stops shrinking."""

    def __init__(self) -> None:
        self.best: float | None = None
        self.stale = 0

    def stalled(self, error: float) -> bool:
        current = abs(float(error))
        if self.best is None or current < self.best - _STALL_PROGRESS_M:
            self.best = current
            self.stale = 0
            return False
        self.stale += 1
        return self.stale >= _STALL_WINDOW_STEPS


def _read_base_velocity(adapter: Any) -> tuple[float, float, float] | None:
    try:
        observation = adapter.read_observation(0.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    velocity = getattr(observation, "base_velocity", None)
    if velocity is None or len(velocity) < 3:
        return None
    return (float(velocity[0]), float(velocity[1]), float(velocity[2]))


def _brake_until_stopped(
    adapter: Any,
    max_steps: int = _BRAKE_MAX_STEPS,
    step_hook: Callable[[], None] | None = None,
) -> float:
    """Zero wheel commands until linear/yaw speed are small, or ``max_steps``."""
    if not hasattr(adapter, "set_targets") or not hasattr(adapter, "step"):
        return 0.0
    velocity = _read_base_velocity(adapter)
    if velocity is None:
        return 0.0
    steer_joints = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
    wheel_joints = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
    _stop_wheels_hold_steer(adapter, steer_joints, wheel_joints)
    steps = 0
    while steps < max(0, int(max_steps)):
        vx, vy, yaw_rate = velocity
        if math.hypot(vx, vy) < _BRAKE_LIN_MPS and abs(yaw_rate) < _BRAKE_ANG_RADPS:
            break
        _stop_wheels_hold_steer(adapter, steer_joints, wheel_joints)
        adapter.step()
        if step_hook is not None:
            step_hook()
        steps += 1
        velocity = _read_base_velocity(adapter) or (0.0, 0.0, 0.0)
    return float(steps)


def _read_base(adapter: Any) -> tuple[float, float, float]:
    """World (x, y, yaw) of the base root."""
    pos = adapter.robot.data.root_pos_w[0].detach().cpu().numpy()
    quat = adapter.robot.data.root_quat_w[0].detach().cpu().numpy()  # w, x, y, z
    w, x, y, z = quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(pos[0]), float(pos[1]), yaw


def _footprint_radius(adapter: Any, scene: Any = None) -> float:
    """Return the execution-calibrated radius without requiring planner input.

    A scene may provide a calibrated planning footprint. It is an execution
    fact, so the skill consumes it directly rather than requiring an LLM plan to
    copy the value into every navigation call. Adapter/chassis geometry remains
    the fallback for scenes without an authored calibration.
    """
    robot = getattr(scene, "robot", None)
    configured = getattr(robot, "navigation_footprint_radius_m", None)
    if configured is not None:
        try:
            value = float(configured)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return value
    if hasattr(adapter, "base_footprint"):
        try:
            return float(adapter.base_footprint()["circumscribed_radius_m"])
        except (KeyError, TypeError, RuntimeError):
            pass
    from r1pro_data_gen.robot.chassis import default_footprint_radius_m

    return default_footprint_radius_m()


def _arm_ready_targets(adapter: Any) -> dict[str, float]:
    """One bounded step of both arms toward the manipulation rest pose."""
    from r1pro_data_gen.robot.robot_config import R1PRO_ARM_READY_Q_BY_SIDE

    from ..manipulation.arm import ARM_JOINTS_BY_SIDE
    from ..manipulation.gripper import GRIPPER_OPEN

    try:
        observation = adapter.read_observation(0.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    positions = getattr(observation, "joint_positions", {}) or {}
    extra: dict[str, float] = {}
    for side in ("left", "right"):
        joints = ARM_JOINTS_BY_SIDE[side]
        if not any(name in positions for name in joints):
            continue
        ready = R1PRO_ARM_READY_Q_BY_SIDE[side]
        for index, name in enumerate(joints):
            current = float(positions.get(name, 0.0))
            delta = float(ready[index]) - current
            extra[name] = current + max(-_ARM_READY_STEP_RAD, min(_ARM_READY_STEP_RAD, delta))
        extra[f"{side}_gripper_finger_joint1"] = GRIPPER_OPEN
        extra[f"{side}_gripper_finger_joint2"] = -GRIPPER_OPEN
    return extra


def _set_drive_targets(
    adapter: Any,
    cmds: Mapping[str, Any],
    steer_joints: tuple[str, ...],
    wheel_joints: tuple[str, ...],
    aux_position_fn: Callable[[], dict[str, float]] | None,
) -> None:
    current = _current_steer_angles(adapter, steer_joints)
    adjusted = {}
    for name in steer_joints:
        raw = cmds[name]
        if name in current:
            adjusted[name] = shortest_steer_command(
                current[name], raw.steer_angle, raw.wheel_speed
            )
        else:
            adjusted[name] = raw
    position = {name: adjusted[name].steer_angle for name in steer_joints}
    if aux_position_fn is not None:
        position.update(aux_position_fn() or {})
    scale = _steer_drive_scale(current, adjusted, steer_joints)
    adapter.set_targets(
        position=position,
        velocity={
            wheel: adjusted[steer_joints[index]].wheel_speed * scale
            for index, wheel in enumerate(wheel_joints)
        },
    )


def _drive_to(
    adapter: Any,
    target: tuple[float, float, float],
    v_max: float,
    omega_max: float,
    arrive_tol: float,
    max_steps: int,
    hold_steps: int,
    step_hook: Callable[[], None] | None,
    *,
    position_only: bool = False,
    aux_position_fn: Callable[[], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Closed-loop P drive to (x, y, yaw); returns arrival metrics.

    The position error is world-frame, but ``wheel_commands`` expects
    base_link-frame velocity. When the base has turned (yaw != 0) the two
    disagree -- e.g. after rotating 180 degrees, world +x is base -x and the
    naive command drives away from the target. Rotate the error into the
    base_link frame before commanding.
    """
    steer_joints = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
    wheel_joints = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
    steps = 0
    vx_cmd, vy_cmd, omega_cmd = _initial_body_twist(adapter)
    watchdog = _ProgressWatchdog()
    while steps < max_steps:
        bx, by, byaw = _read_base(adapter)
        dx = target[0] - bx
        dy = target[1] - by
        dyaw = _wrap_pi(target[2] - byaw)
        if position_only:
            arrived = abs(dx) < arrive_tol and abs(dy) < arrive_tol
        else:
            arrived = abs(dx) < arrive_tol and abs(dy) < arrive_tol and abs(dyaw) < arrive_tol
        if arrived:
            break
        if watchdog.stalled(math.hypot(dx, dy)):
            break
        # Rotate world-frame error into the base_link frame.
        dx_b = dx * math.cos(byaw) + dy * math.sin(byaw)
        dy_b = -dx * math.sin(byaw) + dy * math.cos(byaw)
        vx_des = max(-v_max, min(v_max, 2.0 * dx_b))
        vy_des = max(-v_max, min(v_max, 2.0 * dy_b))
        omega_des = max(-omega_max, min(omega_max, 2.0 * dyaw))
        vx_cmd = _slew(vx_cmd, vx_des, R1PRO_BASE_LIN_ACCEL_MAX)
        vy_cmd = _slew(vy_cmd, vy_des, R1PRO_BASE_LIN_ACCEL_MAX)
        omega_cmd = _slew(omega_cmd, omega_des, R1PRO_BASE_ANG_ACCEL_MAX)
        cmds = wheel_commands(vx=vx_cmd, vy=vy_cmd, omega=omega_cmd)
        _set_drive_targets(adapter, cmds, steer_joints, wheel_joints, aux_position_fn)
        adapter.step()
        if step_hook is not None:
            step_hook()
        steps += 1
    # Stop and hold the live steer angles. Forcing steer to 0 here used to
    # reconfigure the modules and drop a wheel after an otherwise settled drive.
    _stop_wheels_hold_steer(adapter, steer_joints, wheel_joints, aux_position_fn)
    for _ in range(hold_steps):
        adapter.step()
        if step_hook is not None:
            step_hook()
    bx, by, byaw = _read_base(adapter)
    if position_only:
        success = abs(target[0] - bx) < arrive_tol and abs(target[1] - by) < arrive_tol
    else:
        success = (
            abs(target[0] - bx) < arrive_tol
            and abs(target[1] - by) < arrive_tol
            and abs(_wrap_pi(target[2] - byaw)) < arrive_tol
        )
    return {
        "success": float(success),
        "steps": float(steps),
        "final_x": bx,
        "final_y": by,
        "final_yaw": byaw,
        "arrival_error_m": max(abs(target[0] - bx), abs(target[1] - by)),
        "yaw_error_rad": abs(_wrap_pi(target[2] - byaw)),
    }


def _forward_tracking_command(
    dx_world: float,
    dy_world: float,
    current_yaw: float,
    v_max: float,
    omega_max: float,
) -> tuple[float, float, float, float]:
    """Pure-pursuit style command with no body-frame lateral translation."""
    distance = math.hypot(dx_world, dy_world)
    desired_yaw = math.atan2(dy_world, dx_world)
    heading_error = _wrap_pi(desired_yaw - current_yaw)
    omega = max(-omega_max, min(omega_max, 2.2 * heading_error))
    # Linear fade keeps some forward speed through a 90 deg doorway corner.
    # Only stop-and-reverse when the carrot is almost behind the chassis.
    if abs(heading_error) >= _TURN_IN_PLACE_HEADING_RAD:
        alignment = 0.0
    else:
        alignment = 1.0 - abs(heading_error) / _TURN_IN_PLACE_HEADING_RAD
    vx = min(v_max, 1.6 * distance) * alignment
    return vx, 0.0, omega, heading_error


def _drive_forward_to(
    adapter: Any,
    target_xy: tuple[float, float],
    v_max: float,
    omega_max: float,
    arrive_tol: float,
    max_steps: int,
    step_hook: Callable[[], None] | None,
    aux_position_fn: Callable[[], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Track a world-space point while keeping the chassis facing motion."""
    steer_joints = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
    wheel_joints = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
    steps = 0
    max_lateral_command = 0.0
    vx_cmd, vy_cmd, omega_cmd = _initial_body_twist(adapter)
    watchdog = _ProgressWatchdog()
    while steps < max_steps:
        bx, by, byaw = _read_base(adapter)
        dx, dy = float(target_xy[0] - bx), float(target_xy[1] - by)
        distance = math.hypot(dx, dy)
        if distance < arrive_tol:
            break
        if watchdog.stalled(distance):
            break
        vx_des, vy_des, omega_des, _ = _forward_tracking_command(dx, dy, byaw, v_max, omega_max)
        vx_cmd = _slew(vx_cmd, vx_des, R1PRO_BASE_LIN_ACCEL_MAX)
        vy_cmd = _slew(vy_cmd, vy_des, R1PRO_BASE_LIN_ACCEL_MAX)
        omega_cmd = _slew(omega_cmd, omega_des, R1PRO_BASE_ANG_ACCEL_MAX)
        max_lateral_command = max(max_lateral_command, abs(vy_cmd))
        cmds = wheel_commands(vx=vx_cmd, vy=vy_cmd, omega=omega_cmd)
        _set_drive_targets(adapter, cmds, steer_joints, wheel_joints, aux_position_fn)
        adapter.step()
        if step_hook is not None:
            step_hook()
        steps += 1
    bx, by, byaw = _read_base(adapter)
    error = math.hypot(float(target_xy[0] - bx), float(target_xy[1] - by))
    return {
        "success": float(error < arrive_tol),
        "steps": float(steps),
        "final_x": bx,
        "final_y": by,
        "final_yaw": byaw,
        "arrival_error_m": error,
        "max_lateral_command_mps": max_lateral_command,
    }


def _polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index in range(len(points) - 1):
        total += math.hypot(
            float(points[index + 1][0]) - float(points[index][0]),
            float(points[index + 1][1]) - float(points[index][1]),
        )
    return total


def _point_on_polyline(points: list[tuple[float, float]], distance: float) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1 or distance <= 0.0:
        return (float(points[0][0]), float(points[0][1]))
    remaining = float(distance)
    for index in range(len(points) - 1):
        x0, y0 = float(points[index][0]), float(points[index][1])
        x1, y1 = float(points[index + 1][0]), float(points[index + 1][1])
        segment = math.hypot(x1 - x0, y1 - y0)
        if segment <= 1.0e-9:
            continue
        if remaining <= segment:
            ratio = remaining / segment
            return (x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0))
        remaining -= segment
    return (float(points[-1][0]), float(points[-1][1]))


def _project_on_polyline(points: list[tuple[float, float]], x: float, y: float) -> float:
    """Arc length of the closest point on the polyline to (x, y)."""
    if len(points) < 2:
        return 0.0
    best_distance = float("inf")
    best_s = 0.0
    accumulated = 0.0
    for index in range(len(points) - 1):
        x0, y0 = float(points[index][0]), float(points[index][1])
        x1, y1 = float(points[index + 1][0]), float(points[index + 1][1])
        dx, dy = x1 - x0, y1 - y0
        segment = math.hypot(dx, dy)
        if segment <= 1.0e-9:
            continue
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (segment * segment)))
        px, py = x0 + t * dx, y0 + t * dy
        distance = math.hypot(x - px, y - py)
        if distance < best_distance:
            best_distance = distance
            best_s = accumulated + t * segment
        accumulated += segment
    return best_s


def _lookahead_point(
    points: list[tuple[float, float]],
    x: float,
    y: float,
    lookahead_m: float = _PATH_LOOKAHEAD_M,
) -> tuple[float, float]:
    """Carrot on the polyline: never aim at the residual of the nearest vertex."""
    if not points:
        return (float(x), float(y))
    if len(points) == 1:
        return (float(points[0][0]), float(points[0][1]))
    along = _project_on_polyline(points, x, y) + max(0.05, float(lookahead_m))
    return _point_on_polyline(points, along)


def _drive_path(
    adapter: Any,
    waypoints: list[tuple[float, float]] | list[list[float]],
    v_max: float,
    omega_max: float,
    arrive_tol: float,
    max_steps: int,
    step_hook: Callable[[], None] | None,
    aux_position_fn: Callable[[], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Follow a polyline with lookahead instead of stopping at every vertex."""
    points = [(float(point[0]), float(point[1])) for point in waypoints]
    if not points:
        bx, by, byaw = _read_base(adapter)
        return {
            "success": 1.0,
            "steps": 0.0,
            "final_x": bx,
            "final_y": by,
            "final_yaw": byaw,
            "arrival_error_m": 0.0,
            "max_lateral_command_mps": 0.0,
        }
    goal = points[-1]
    steer_joints = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
    wheel_joints = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
    steps = 0
    max_lateral_command = 0.0
    vx_cmd, vy_cmd, omega_cmd = _initial_body_twist(adapter)
    watchdog = _ProgressWatchdog()
    while steps < max_steps:
        bx, by, byaw = _read_base(adapter)
        error = math.hypot(goal[0] - bx, goal[1] - by)
        if error < arrive_tol:
            break
        if watchdog.stalled(error):
            break
        carrot = _lookahead_point(points, bx, by)
        dx, dy = carrot[0] - bx, carrot[1] - by
        vx_des, vy_des, omega_des, _ = _forward_tracking_command(dx, dy, byaw, v_max, omega_max)
        vx_cmd = _slew(vx_cmd, vx_des, R1PRO_BASE_LIN_ACCEL_MAX)
        vy_cmd = _slew(vy_cmd, vy_des, R1PRO_BASE_LIN_ACCEL_MAX)
        omega_cmd = _slew(omega_cmd, omega_des, R1PRO_BASE_ANG_ACCEL_MAX)
        max_lateral_command = max(max_lateral_command, abs(vy_cmd))
        cmds = wheel_commands(vx=vx_cmd, vy=vy_cmd, omega=omega_cmd)
        _set_drive_targets(adapter, cmds, steer_joints, wheel_joints, aux_position_fn)
        adapter.step()
        if step_hook is not None:
            step_hook()
        steps += 1
    bx, by, byaw = _read_base(adapter)
    error = math.hypot(goal[0] - bx, goal[1] - by)
    return {
        "success": float(error < arrive_tol),
        "steps": float(steps),
        "final_x": bx,
        "final_y": by,
        "final_yaw": byaw,
        "arrival_error_m": error,
        "max_lateral_command_mps": max_lateral_command,
    }


def _rotate_in_place(
    adapter: Any,
    target_yaw: float,
    omega_max: float,
    arrive_tol: float,
    max_steps: int,
    hold_steps: int,
    step_hook: Callable[[], None] | None,
    aux_position_fn: Callable[[], dict[str, float]] | None = None,
) -> dict[str, float]:
    steer_joints = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
    wheel_joints = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
    steps = 0
    _, _, omega_cmd = _initial_body_twist(adapter)
    while steps < max_steps:
        _, _, yaw = _read_base(adapter)
        error = _wrap_pi(float(target_yaw) - yaw)
        if abs(error) < arrive_tol:
            break
        omega_des = max(-omega_max, min(omega_max, 2.0 * error))
        omega_cmd = _slew(omega_cmd, omega_des, R1PRO_BASE_ANG_ACCEL_MAX)
        cmds = wheel_commands(vx=0.0, vy=0.0, omega=omega_cmd)
        _set_drive_targets(adapter, cmds, steer_joints, wheel_joints, aux_position_fn)
        adapter.step()
        if step_hook is not None:
            step_hook()
        steps += 1
    _stop_wheels_hold_steer(adapter, steer_joints, wheel_joints, aux_position_fn)
    for _ in range(hold_steps):
        adapter.step()
        if step_hook is not None:
            step_hook()
    _, _, final_yaw = _read_base(adapter)
    final_error = abs(_wrap_pi(float(target_yaw) - final_yaw))
    return {
        "success": float(final_error < arrive_tol),
        "steps": float(steps),
        "final_yaw": final_yaw,
        "yaw_error_rad": final_error,
    }


class BaseMoveTo:
    """Drive the base straight to a world pose (x, y, yaw)."""

    name = "base_move_to"
    description = "Drive the base in a straight line to any world pose (x, y, yaw)."
    parameters: dict[str, ParamSpec] = {
        "target": ParamSpec("array", "Target world pose (x, y, yaw)", required=True),
        "v_max": ParamSpec("number", "Max linear speed (m/s)", default=DEFAULT_V_MAX),
        "omega_max": ParamSpec("number", "Max yaw rate (rad/s)", default=DEFAULT_OMEGA_MAX),
        "arrive_tol": ParamSpec("number", "Arrival tolerance (m / rad)", default=DEFAULT_ARRIVE_TOL),
    }

    def __init__(
        self,
        v_max: float = DEFAULT_V_MAX,
        omega_max: float = DEFAULT_OMEGA_MAX,
        arrive_tol: float = DEFAULT_ARRIVE_TOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        hold_steps: int = 30,
    ) -> None:
        self.v_max = v_max
        self.omega_max = omega_max
        self.arrive_tol = arrive_tol
        self.max_steps = max_steps
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target: tuple[float, float, float] | list[float] = None,
        v_max: float | None = None,
        omega_max: float | None = None,
        arrive_tol: float | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        release_skill_wheel_lock(adapter)
        if target is None:
            raise ValueError("base_move_to requires target (x, y, yaw)")
        v_max = self.v_max if v_max is None else v_max
        omega_max = self.omega_max if omega_max is None else omega_max
        arrive_tol = self.arrive_tol if arrive_tol is None else arrive_tol

        # Translating and rotating a steer-and-drive base in one coupled
        # command can make the wheel steering fight the yaw controller at a
        # large heading error.  Close XY with the current heading first, then
        # rotate in place.  Both phases remain bounded by the same skill
        # budget and the final metrics preserve the public contract.
        _, _, current_yaw = _read_base(adapter)
        position = _drive_to(
            adapter,
            (float(target[0]), float(target[1]), current_yaw),
            v_max,
            omega_max,
            arrive_tol,
            self.max_steps,
            0,
            step_hook,
            position_only=True,
        )
        rotation = _rotate_in_place(
            adapter,
            float(target[2]),
            omega_max,
            arrive_tol,
            self.max_steps,
            self.hold_steps,
            step_hook,
        )
        metrics = {
            "success": float(bool(position["success"]) and bool(rotation["success"])),
            "steps": float(position["steps"] + rotation["steps"]),
            "final_x": position["final_x"],
            "final_y": position["final_y"],
            "final_yaw": float(rotation.get("final_yaw", 0.0)),
            "arrival_error_m": position["arrival_error_m"],
            "yaw_error_rad": rotation["yaw_error_rad"],
        }
        return SkillResult(
            success=bool(metrics["success"]),
            skill=self.name,
            metrics={
                "steps": metrics["steps"],
                "arrival_error_m": metrics["arrival_error_m"],
                "yaw_error_rad": metrics["yaw_error_rad"],
            },
            details={"final": {"x": metrics["final_x"], "y": metrics["final_y"], "yaw": metrics["final_yaw"]}},
        )


class BaseRotateTo:
    """Rotate the base in place to a target yaw."""

    name = "base_rotate_to"
    description = "Rotate the base in place to a target yaw (rad)."
    parameters: dict[str, ParamSpec] = {
        "target_yaw": ParamSpec("number", "Target yaw (rad, world frame)", required=True),
        "omega_max": ParamSpec("number", "Max yaw rate (rad/s)", default=DEFAULT_OMEGA_MAX),
        "arrive_tol": ParamSpec("number", "Arrival tolerance (rad)", default=DEFAULT_ARRIVE_TOL),
    }

    def __init__(
        self,
        omega_max: float = DEFAULT_OMEGA_MAX,
        arrive_tol: float = DEFAULT_ARRIVE_TOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        hold_steps: int = 30,
    ) -> None:
        self.omega_max = omega_max
        self.arrive_tol = arrive_tol
        self.max_steps = max_steps
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target_yaw: float = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        release_skill_wheel_lock(adapter)
        if target_yaw is None:
            raise ValueError("base_rotate_to requires target_yaw")
        metrics = _rotate_in_place(
            adapter,
            float(target_yaw),
            self.omega_max,
            self.arrive_tol,
            self.max_steps,
            self.hold_steps,
            step_hook,
        )
        return SkillResult(
            success=bool(metrics["success"]),
            skill=self.name,
            metrics={
                "steps": float(metrics["steps"]),
                "yaw_error_rad": float(metrics["yaw_error_rad"]),
            },
        )


# Final position closure tolerance for base_navigate_to.  The intermediate A*
# waypoints keep a wider 0.06 m tolerance (path constraint, not goal), but the
# final pose must settle tightly: a few-centimetre residual at a narrow
# operation stance can push the arm's IK target outside the workspace, and the
# subsequent arm_move_to has no way to recover by itself.
DEFAULT_FINAL_ARRIVE_TOL = 0.01  # m / rad


class BaseNavigateTo:
    """Plan a collision-free 2D path (grid A*) and follow it waypoint by waypoint."""

    name = "base_navigate_to"
    description = (
        "Navigate the base to a collision-free world pose. Prefer a semantic "
        "target_ref (scene://object) with a purpose such as pregrasp; the "
        "runtime derives a safe approach pose from scene geometry. A literal "
        "target pose remains supported as a legacy preferred/executable pose."
    )
    parameters: dict[str, ParamSpec] = {
        # Semantic navigation resolves the concrete target at execution time.
        # Literal targets remain valid for legacy plans and exact-pose tasks.
        "target": ParamSpec("array", "Optional preferred/exact world pose (x, y, yaw)", default=None, shape=(3,)),
        "target_ref": ParamSpec("string", "Semantic scene reference such as scene://object", default=None),
        "purpose": ParamSpec("string", "Why the robot is approaching the target", default="navigation", enum=("navigation", "pregrasp", "dropoff", "staging", "observe", "park")),
        "preferred_pose": ParamSpec("array", "Optional preferred pose; resolver may choose a safer candidate", default=None, shape=(3,)),
        "approach_side": ParamSpec("string", "Optional semantic approach side", default=None, enum=("west", "east", "south", "north")),
        "resolution": ParamSpec("number", "Grid cell size (m)", default=0.05),
        "footprint_radius": ParamSpec("number", "Optional override; otherwise derived from the R1Pro chassis footprint", default=None, minimum=0.05),
        "v_max": ParamSpec("number", "Max linear speed (m/s)", default=DEFAULT_V_MAX),
        "omega_max": ParamSpec("number", "Max yaw rate (rad/s)", default=DEFAULT_OMEGA_MAX),
        "arrive_tol": ParamSpec("number", "Arrival tolerance (m / rad)", default=DEFAULT_ARRIVE_TOL),
        "final_arrive_tol": ParamSpec("number", "Final pose arrival tolerance (m / rad); tighter than arrive_tol so an operation stance settles accurately", default=DEFAULT_FINAL_ARRIVE_TOL),
        "max_steps_per_waypoint": ParamSpec("integer", "Maximum physics steps per A* waypoint", default=600, minimum=30),
        "motion_mode": ParamSpec("string", "Path tracking style: forward faces the route; holonomic allows lateral motion", default="forward", enum=("forward", "holonomic")),
        "clearance_margin": ParamSpec("number", "Soft preferred clearance beyond hard footprint inflation (m)", default=0.25, minimum=0.0),
        "clearance_weight": ParamSpec("number", "Strength of the soft wall-clearance cost", default=3.0, minimum=0.0),
    }

    def __init__(
        self,
        kinematics: Any = None,
        v_max: float = DEFAULT_V_MAX,
        omega_max: float = DEFAULT_OMEGA_MAX,
        arrive_tol: float = DEFAULT_ARRIVE_TOL,
        max_steps: int = 2000,
        hold_steps: int = 30,
    ) -> None:
        self.kinematics = kinematics
        self.v_max = v_max
        self.omega_max = omega_max
        self.arrive_tol = arrive_tol
        self.max_steps = max_steps
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target: tuple[float, float, float] | list[float] = None,
        target_ref: str | None = None,
        purpose: str = "navigation",
        preferred_pose: tuple[float, float, float] | list[float] | None = None,
        approach_side: str | None = None,
        resolution: float = 0.05,
        footprint_radius: float | None = None,
        v_max: float | None = None,
        omega_max: float | None = None,
        arrive_tol: float | None = None,
        final_arrive_tol: float | None = None,
        max_steps_per_waypoint: int | None = None,
        motion_mode: str = "forward",
        clearance_margin: float = 0.25,
        clearance_weight: float = 3.0,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        resolution_record = None
        if target is None and target_ref is not None:
            from r1pro_data_gen.planning.navigation.targets import (
                NavigationTargetError,
                resolve_navigation_target,
            )

            try:
                resolution_record = resolve_navigation_target(
                    scene,
                    target_ref,
                    purpose=purpose,
                    preferred_pose=preferred_pose,
                    approach_side=approach_side,
                    kinematics=self.kinematics,
                )
            except NavigationTargetError as exc:
                return SkillResult(
                    success=False,
                    skill=self.name,
                    details={
                        "reason": str(exc),
                        "error_code": exc.code,
                        "failure_code": exc.code,
                        "target_ref": target_ref,
                        "purpose": purpose,
                        **exc.details,
                    },
                )
            target = list(resolution_record.resolved_pose)
        elif target is None:
            raise ValueError("base_navigate_to requires target or target_ref")
        release_skill_wheel_lock(adapter)
        target_evidence = resolution_record.to_details() if resolution_record is not None else {}
        from r1pro_data_gen.methods import astar_path, clearance_cost_grid, occupancy_from_boxes, path_to_world_waypoints, simplify_grid_path
        from r1pro_data_gen.skills.planning import runtime_scene_snapshot

        bx, by, byaw = _read_base(adapter)
        tx, ty, tyaw = float(target[0]), float(target[1]), float(target[2])
        footprint_radius = _footprint_radius(adapter, scene) if footprint_radius is None else float(footprint_radius)
        xy_error = math.hypot(tx - bx, ty - by)
        yaw_error = abs(_wrap_pi(tyaw - byaw))
        if xy_error <= _FINAL_XY_TOL_M and yaw_error <= _NAV_YAW_ACCEPT_RAD:
            return SkillResult(
                success=True,
                skill=self.name,
                metrics={
                    "steps": 0.0,
                    "waypoints": 0.0,
                    "arrival_error_m": xy_error,
                    "yaw_error_rad": yaw_error,
                    "max_lateral_command_mps": 0.0,
                },
                details={
                    "path": [],
                    "target": [tx, ty, tyaw],
                    "reason": "already at navigation target",
                    "footprint_radius_m": footprint_radius,
                    **target_evidence,
                },
            )

        # Bounding box around start and goal, padded so the path has room.
        pad = 1.5
        xmin = min(bx, tx) - pad
        xmax = max(bx, tx) + pad
        ymin = min(by, ty) - pad
        ymax = max(by, ty) + pad
        rows = max(2, int(math.ceil((ymax - ymin) / resolution)))
        cols = max(2, int(math.ceil((xmax - xmin) / resolution)))

        live_scene = runtime_scene_snapshot(scene, adapter)
        # Inflated 2D obstacle boxes from the scene objects (world xy extents).
        boxes: list[tuple[float, float, float, float]] = []
        if live_scene is not None:
            for obj in live_scene.objects:
                if not obj.physics.collision_enabled:
                    continue
                hx, hy = object_xy_half_extents_m(obj)
                inflate = footprint_radius + NAVIGATION_INFLATION_CLEARANCE_M
                boxes.append(
                    (obj.pos[0] - hx - inflate, obj.pos[1] - hy - inflate,
                     obj.pos[0] + hx + inflate, obj.pos[1] + hy + inflate)
                )
        grid = occupancy_from_boxes(boxes, xmin, ymin, resolution, (rows, cols))

        # Start/goal cells (cell = row from +y, col from +x).
        start = (int((by - ymin) / resolution), int((bx - xmin) / resolution))
        goal = (int((ty - ymin) / resolution), int((tx - xmin) / resolution))
        if grid[start]:
            return SkillResult(
                success=False,
                skill=self.name,
                details={
                    "reason": "start cell is inside an obstacle",
                    "target": [tx, ty, tyaw],
                    "footprint_radius_m": footprint_radius,
                    **target_evidence,
                },
            )
        if grid[goal]:
            return SkillResult(
                success=False,
                skill=self.name,
                details={
                    "reason": "goal cell is inside an obstacle",
                    "target": [tx, ty, tyaw],
                    "footprint_radius_m": footprint_radius,
                    **target_evidence,
                },
            )
        if motion_mode not in {"forward", "holonomic"}:
            raise ValueError("base_navigate_to motion_mode must be 'forward' or 'holonomic'")
        soft_cost = clearance_cost_grid(
            grid,
            clearance_cells=float(clearance_margin) / float(resolution),
            weight=float(clearance_weight),
        )
        path = astar_path(grid, start, goal, allow_diagonal=True, traversal_cost=soft_cost)
        if path is None:
            return SkillResult(
                success=False,
                skill=self.name,
                details={
                    "reason": "no collision-free 2D path to target",
                    "target": [tx, ty, tyaw],
                    **target_evidence,
                },
            )
        path = simplify_grid_path(path, grid, traversal_cost=soft_cost)
        waypoints = path_to_world_waypoints(path, xmin, ymin, resolution)

        v_max = self.v_max if v_max is None else v_max
        omega_max = self.omega_max if omega_max is None else omega_max
        arrive_tol = self.arrive_tol if arrive_tol is None else arrive_tol
        final_arrive_tol = (
            DEFAULT_FINAL_ARRIVE_TOL if final_arrive_tol is None else float(final_arrive_tol)
        )
        waypoint_max_steps = self.max_steps if max_steps_per_waypoint is None else int(max_steps_per_waypoint)
        final_xy_tol = max(float(final_arrive_tol), _FINAL_XY_TOL_M)
        final_closure_steps = min(int(waypoint_max_steps), _FINAL_CLOSURE_MAX_STEPS)
        total_steps = 0.0
        max_err = 0.0
        max_lateral_command = 0.0
        waypoint_tol = max(float(arrive_tol), 0.06)
        if motion_mode == "forward":
            path_budget = int(waypoint_max_steps) * max(1, len(waypoints))
            metrics = _drive_path(
                adapter,
                [(float(wx), float(wy)) for wx, wy in waypoints],
                v_max,
                omega_max,
                waypoint_tol,
                path_budget,
                step_hook,
            )
            total_steps += metrics["steps"]
            max_err = max(max_err, metrics["arrival_error_m"])
            max_lateral_command = max(
                max_lateral_command, metrics["max_lateral_command_mps"]
            )
        else:
            for i, (wx, wy) in enumerate(waypoints):
                current_x, current_y, _ = _read_base(adapter)
                if i == 0 and math.hypot(float(wx) - current_x, float(wy) - current_y) <= waypoint_tol:
                    continue
                if i + 1 < len(waypoints):
                    target_wp_yaw = math.atan2(waypoints[i + 1][1] - wy, waypoints[i + 1][0] - wx)
                else:
                    target_wp_yaw = tyaw
                metrics = _drive_to(
                    adapter, (wx, wy, target_wp_yaw), v_max, omega_max,
                    waypoint_tol, waypoint_max_steps, 0, step_hook,
                    position_only=True,
                )
                total_steps += metrics["steps"]
                max_err = max(max_err, metrics["arrival_error_m"])
                if not bool(metrics["success"]) and i + 1 < len(waypoints):
                    _, _, current_yaw = _read_base(adapter)
                    return SkillResult(
                        success=False,
                        skill=self.name,
                        metrics={
                            "steps": total_steps,
                            "arrival_error_m": max_err,
                            "yaw_error_rad": abs(_wrap_pi(target_wp_yaw - current_yaw)),
                        },
                        details={
                            "reason": "lost waypoint tracking",
                            "target": [tx, ty, tyaw],
                            **target_evidence,
                        },
                    )
        if motion_mode == "forward":
            final_position = _drive_forward_to(
                adapter, (tx, ty), v_max, omega_max, final_xy_tol,
                final_closure_steps, step_hook,
            )
            if not bool(final_position["success"]):
                # Pure-forward tracking turns first and can stall a few cm off
                # the target when the residual heading sits just outside its
                # field of view. A holonomic position closure drives the world
                # error directly and reliably settles the last residual.
                _, _, current_yaw = _read_base(adapter)
                lateral_residual = float(final_position.get("max_lateral_command_mps", 0.0))
                final_position = _drive_to(
                    adapter, (tx, ty, current_yaw), v_max, omega_max, final_xy_tol,
                    final_closure_steps, 0, step_hook, position_only=True,
                )
                final_position["max_lateral_command_mps"] = lateral_residual
        else:
            # Separate final translation from yaw closure.  The omni chassis can
            # settle the requested XY pose without fighting a large heading
            # error; closing both errors in one wheel command made the final
            # steering state stall near corners and report a false miss.
            _, _, current_yaw = _read_base(adapter)
            final_position = _drive_to(
                adapter, (tx, ty, current_yaw), v_max, omega_max, final_xy_tol,
                final_closure_steps, 0, step_hook, position_only=True,
            )
        _, _, current_yaw = _read_base(adapter)
        yaw_error = abs(_wrap_pi(tyaw - current_yaw))
        if yaw_error <= _NAV_YAW_FINISH_RAD:
            final_rotation = {
                "success": 1.0,
                "steps": 0.0,
                "final_yaw": current_yaw,
                "yaw_error_rad": yaw_error,
            }
        else:
            final_rotation = _rotate_in_place(
                adapter, tyaw, omega_max, _NAV_YAW_FINISH_RAD,
                final_closure_steps, 0, step_hook,
            )
            yaw_error = float(final_rotation["yaw_error_rad"])
        final_error = float(final_position["arrival_error_m"])
        final_success = bool(final_position["success"]) and (
            bool(final_rotation["success"]) or yaw_error <= _NAV_YAW_ACCEPT_RAD
        )
        total_steps += final_position["steps"] + final_rotation["steps"]
        brake_steps = _brake_until_stopped(adapter, _BRAKE_MAX_STEPS, step_hook)
        total_steps += brake_steps
        max_lateral_command = max(
            max_lateral_command, float(final_position.get("max_lateral_command_mps", 0.0))
        )
        return SkillResult(
            success=final_success,
            skill=self.name,
            metrics={
                "steps": total_steps,
                "waypoints": float(len(waypoints)),
                "arrival_error_m": final_error,
                "yaw_error_rad": yaw_error,
                "max_lateral_command_mps": max_lateral_command,
            },
            details={
                "path": [[round(float(x), 4), round(float(y), 4)] for x, y in waypoints],
                "target": [tx, ty, tyaw],
                "footprint_radius_m": footprint_radius,
                "max_waypoint_error_m": max_err,
                "motion_mode": motion_mode,
                "clearance_margin_m": float(clearance_margin),
                **target_evidence,
            },
        )


class BaseLockWheels:
    """Freeze the wheel drives (position hold) so arm motion does not roll the base."""

    name = "base_lock_wheels"
    tier = "backend"
    exposed = False
    description = "Freeze the wheel drives so arm motion does not roll the base."
    parameters: dict[str, ParamSpec] = {
        "settle_steps": ParamSpec("integer", "Physics steps used to settle the upright parking brake", default=30, minimum=0),
    }

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        settle_steps: int = 30,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        adapter.lock_wheels()
        for _ in range(max(0, int(settle_steps))):
            adapter.step()
            if step_hook is not None:
                step_hook()
        return SkillResult(success=True, skill=self.name, metrics={"settle_steps": float(max(0, int(settle_steps)))})


class BaseUnlockWheels:
    """Release the wheel drives (restore velocity control) after manipulation."""

    name = "base_unlock_wheels"
    tier = "backend"
    exposed = False
    description = "Release the wheel drives (restore velocity control) after an arm-manipulation phase."
    parameters: dict[str, ParamSpec] = {}

    def execute(self, adapter: Any, scene: Any = None, **_: Any) -> SkillResult:
        del scene
        adapter.unlock_wheels()
        return SkillResult(success=True, skill=self.name)


def _clamp_velocity(vx: float, vy: float, omega: float, v_max: float, omega_max: float) -> tuple[float, float, float]:
    """Clamp base velocity commands to safe limits (velocity_set safety net)."""
    return (
        max(-v_max, min(v_max, vx)),
        max(-v_max, min(v_max, vy)),
        max(-omega_max, min(omega_max, omega)),
    )


class BaseFollowPath:
    """Follow a given waypoint path (world x, y) then hold the target yaw."""

    name = "base_follow_path"
    tier = "backend"
    exposed = False
    description = "Follow a given waypoint path (world x, y pairs) then rotate to the target yaw at the end."
    parameters: dict[str, ParamSpec] = {
        "path": ParamSpec("array", "List of (x, y) waypoints in world frame", required=True),
        "target_yaw": ParamSpec("number", "Final yaw (rad)", default=0.0),
        "v_max": ParamSpec("number", "Max linear speed (m/s)", default=DEFAULT_V_MAX),
        "omega_max": ParamSpec("number", "Max yaw rate (rad/s)", default=DEFAULT_OMEGA_MAX),
        "arrive_tol": ParamSpec("number", "Waypoint arrival tolerance (m)", default=DEFAULT_ARRIVE_TOL),
    }

    def __init__(
        self,
        v_max: float = DEFAULT_V_MAX,
        omega_max: float = DEFAULT_OMEGA_MAX,
        arrive_tol: float = DEFAULT_ARRIVE_TOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        hold_steps: int = 30,
    ) -> None:
        self.v_max = v_max
        self.omega_max = omega_max
        self.arrive_tol = arrive_tol
        self.max_steps = max_steps
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        path: list[list[float]] = None,
        target_yaw: float = 0.0,
        v_max: float | None = None,
        omega_max: float | None = None,
        arrive_tol: float | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        if path is None or len(path) < 1:
            raise ValueError("base_follow_path requires a non-empty path")
        v_max = self.v_max if v_max is None else v_max
        omega_max = self.omega_max if omega_max is None else omega_max
        arrive_tol = self.arrive_tol if arrive_tol is None else arrive_tol
        total_steps = 0.0
        for i, (wx, wy) in enumerate(path):
            bx, by, _ = _read_base(adapter)
            # Planned paths commonly include the current pose as point zero.
            # Re-commanding the current-point waypoint with a tangent would
            # create an unnecessary in-place turn, so skip it when already
            # inside tolerance. Intermediate points constrain only position:
            # an omni base can translate through a corner without stopping to
            # rotate, while the final point is the only pose that must satisfy
            # the requested yaw.
            is_last = i == len(path) - 1
            if not is_last and math.hypot(float(wx) - bx, float(wy) - by) < arrive_tol:
                continue
            waypoint_yaw = target_yaw if is_last else _read_base(adapter)[2]
            metrics = _drive_to(
                adapter, (float(wx), float(wy), float(waypoint_yaw)),
                v_max, omega_max, arrive_tol, self.max_steps, 0, step_hook,
            )
            total_steps += metrics["steps"]
            if not bool(metrics["success"]):
                return SkillResult(
                    False,
                    self.name,
                    metrics={
                        "steps": total_steps,
                        "waypoints": float(len(path)),
                        "arrival_error_m": metrics["arrival_error_m"],
                        "yaw_error_rad": metrics["yaw_error_rad"],
                    },
                    details={"reason": "waypoint tracking failed", "failed_waypoint": i},
                )
        # Final hold at the last waypoint with the target yaw.
        final = _drive_to(
            adapter, (float(path[-1][0]), float(path[-1][1]), float(target_yaw)),
            v_max, omega_max, arrive_tol, self.max_steps, self.hold_steps, step_hook,
        )
        return SkillResult(
            success=bool(final["success"]),
            skill=self.name,
            metrics={
                "steps": total_steps + final["steps"],
                "waypoints": float(len(path)),
                "arrival_error_m": final["arrival_error_m"],
                "yaw_error_rad": final["yaw_error_rad"],
            },
        )


class BaseVelocitySet:
    """Directly command raw base velocity (low-level, dangerous)."""

    name = "base_velocity_set"
    tier = "backend"
    exposed = False
    description = (
        "Directly command raw base velocity (vx, vy, omega). WARNING: bypasses path "
        "planning and obstacle avoidance -- the base will drive unguarded, risking "
        "collision, overshoot and instability. Use only for dynamic behaviors "
        "(chasing, drifting) where a target pose is not meaningful. Prefer "
        "base_move_to / base_navigate_to for all positioning. Commands are clamped "
        "to safe limits."
    )
    parameters: dict[str, ParamSpec] = {
        "vx": ParamSpec("number", "Forward velocity (m/s)", default=0.0),
        "vy": ParamSpec("number", "Lateral velocity (m/s)", default=0.0),
        "omega": ParamSpec("number", "Yaw rate (rad/s)", default=0.0),
        "duration": ParamSpec("number", "Seconds to hold the command", default=1.0),
    }

    def __init__(self, v_max: float = DEFAULT_V_MAX, omega_max: float = DEFAULT_OMEGA_MAX, dt: float = 1.0 / 60.0):
        self.v_max = v_max
        self.omega_max = omega_max
        self.dt = dt

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        vx: float = 0.0,
        vy: float = 0.0,
        omega: float = 0.0,
        duration: float = 1.0,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        vx, vy, omega = _clamp_velocity(float(vx), float(vy), float(omega), self.v_max, self.omega_max)
        cmds = wheel_commands(vx=vx, vy=vy, omega=omega)
        steer_joints = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
        wheel_joints = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
        steps = max(1, int(round(float(duration) / self.dt)))
        for _ in range(steps):
            _set_drive_targets(adapter, cmds, steer_joints, wheel_joints, None)
            adapter.step()
            if step_hook is not None:
                step_hook()
        _stop_wheels_hold_steer(adapter, steer_joints, wheel_joints)
        return SkillResult(success=True, skill=self.name, metrics={"steps": float(steps)})


__all__ = [
    "BaseFollowPath",
    "BaseLockWheels",
    "BaseMoveTo",
    "BaseNavigateTo",
    "BaseRotateTo",
    "BaseUnlockWheels",
    "BaseVelocitySet",
]
