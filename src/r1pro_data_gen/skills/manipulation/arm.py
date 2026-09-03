"""Arm execution skills: follow a joint trajectory or move to a joint config.

Trajectory *planning* (IK, MPlib collision avoidance, TOPP smoothing) lives in
``methods`` and the solve/plan skills (``skills.planning``); the skills here
only *execute* a given joint trajectory against the simulation with
speed-limited interpolation (position + velocity reference per step) so the arm
moves within its real joint limits.

- ``arm_joint_to``: move to a target joint configuration (single segment).
- ``arm_trajectory_follow`` (in ``arm_manip``): follow a planned joint trajectory.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.base import ParamSpec, SkillResult
from ..core.sides import for_side, require_side

ARM_JOINTS_BY_SIDE = {
    "left": tuple(f"left_arm_joint{i}" for i in range(1, 8)),
    "right": tuple(f"right_arm_joint{i}" for i in range(1, 8)),
}

_FINAL_ERROR_TOL = 0.08  # rad

# Fraction of a segment's time spent accelerating / decelerating (each end).
_RAMP_FRAC = 0.25


def trapezoid_scale(u: float, u1: float = _RAMP_FRAC) -> float:
    """Velocity scale of a trapezoidal profile at normalized time ``u`` in
    [0, 1]: ramps linearly 0 -> 1 over [0, u1], holds 1, ramps back to 0 over
    [1-u1, 1]. Start/stop velocity is exactly zero -- no step in the reference.
    """
    if u < u1:
        return u / u1
    if u > 1.0 - u1:
        return (1.0 - u) / u1
    return 1.0


def trapezoid_progress(u: float, u1: float = _RAMP_FRAC) -> float:
    """Normalized position progress (integral of the trapezoid scale) at ``u``:
    monotone 0 -> 1, consistent with ``trapezoid_scale`` so the position
    reference advances at the scaled velocity."""
    if u < u1:
        a = u * u / (2.0 * u1)
    elif u < 1.0 - u1:
        a = u1 / 2.0 + (u - u1)
    else:
        w = u - (1.0 - u1)
        a = u1 / 2.0 + (1.0 - 2.0 * u1) + w - w * w / (2.0 * u1)
    return a / (1.0 - u1)


class ArmSegmentExecutor:
    """Speed-limited joint-space segment execution (shared by arm skills).

    The reference follows a trapezoidal velocity profile (accel -> cruise ->
    decel) with the position target integrated from the same profile, so the
    PD drive never sees a step: a constant per-step position jump with an
    instant 0 -> v velocity reference makes the damping term saturate at the
    segment start/end and the arm overshoots then oscillates (measured: the
    wrist rebounded +1.7 rad/s after a 45 deg rotation before settling).
    """

    def __init__(self, kin: Any, vel_limits: np.ndarray, speed_scale: float, hold_steps: int):
        self.kin = kin
        self.vel_limits = vel_limits
        self.speed_scale = speed_scale
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        side: str,
        q_from: np.ndarray,
        q_to: np.ndarray,
        step_hook: Callable[[], None] | None,
    ) -> float:
        """Interpolate q_from -> q_to; returns the final max per-joint error."""
        side = require_side(side)
        joints = ARM_JOINTS_BY_SIDE[side]
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        # Cruise-time steps at the planned (speed-limited) velocity. The
        # trapezoid stretches the segment so the peak speed stays at the
        # cruise speed: cruise occupies (1 - 2*u1) of the total time.
        steps = self.kin.plan_segment_steps(
            q_from, q_to, self.vel_limits, speed_scale=self.speed_scale
        )
        steps = max(3, int(np.ceil(steps / (1.0 - 2.0 * _RAMP_FRAC))))
        disp = q_to - q_from
        dt = 1.0 / 60.0
        for step in range(1, steps + 1):
            u = step / steps
            q_cur = q_from + disp * trapezoid_progress(u)
            vel = disp * trapezoid_scale(u) / (steps * dt)
            adapter.set_targets(
                position={j: float(q_cur[i]) for i, j in enumerate(joints)},
                velocity={j: float(vel[i]) for i, j in enumerate(joints)},
            )
            adapter.step()
            if step_hook is not None:
                step_hook()
        adapter.set_targets(
            position={j: float(q_to[i]) for i, j in enumerate(joints)},
            velocity={j: 0.0 for i, j in enumerate(joints)},
        )
        for _ in range(self.hold_steps):
            adapter.step()
            if step_hook is not None:
                step_hook()
        obs = adapter.read_observation(0.0)
        return max(abs(obs.joint_positions[j] - q_to[i]) for i, j in enumerate(joints))


class ArmJointTo:
    """Move the arm to a target joint configuration (exact, no IK)."""

    name = "arm_joint_to"
    tier = "backend"
    exposed = False
    description = "Move one arm to a target joint configuration (7 values), useful for home poses, diagnostics and replays."
    parameters: dict[str, ParamSpec] = {
        "target_q": ParamSpec("array", "Target joint positions (7, rad)", required=True),
        "side": ParamSpec("string", "Which arm ('left' or 'right')", default="left", enum=("left", "right")),
        "speed_scale": ParamSpec("number", "Fraction of the joint velocity limits", default=0.3),
    }

    def __init__(self, kin: Any, vel_limits: np.ndarray, speed_scale: float = 0.3):
        self.kin = kin
        self.vel_limits = vel_limits
        self.speed_scale = speed_scale

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target_q: tuple[float, ...] | list[float] = None,
        side: str = "left",
        speed_scale: float = 0.3,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        if target_q is None:
            raise ValueError("arm_joint_to requires target_q")
        side = require_side(side)
        obs = adapter.read_observation(0.0)
        q_cur = np.array([obs.joint_positions[j] for j in ARM_JOINTS_BY_SIDE[side]])
        q_to = np.asarray(target_q, dtype=float)
        if q_to.shape != (7,):
            raise ValueError(f"arm_joint_to target_q must be 7 values, got {q_to.shape}")
        segment = ArmSegmentExecutor(for_side(self.kin, side), for_side(self.vel_limits, side), speed_scale, hold_steps=6)
        final_err = segment.execute(adapter, side, q_cur, q_to, step_hook)
        return SkillResult(
            success=bool(final_err < _FINAL_ERROR_TOL),
            skill=self.name,
            metrics={"final_error_rad": float(final_err)},
        )


def quat_from_z_axis(z_axis: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating the gripper z-axis onto ``z_axis``.

    Uses scipy's align_vectors: the manual cross-product rotation has the
    wrong handedness for e.g. (1, 0, 0).
    """
    from scipy.spatial.transform import Rotation

    z_axis = np.asarray(z_axis, dtype=float)
    norm = np.linalg.norm(z_axis)
    if z_axis.shape != (3,) or norm < 1e-9:
        raise ValueError("z_axis must be a non-zero 3-vector")
    z = z_axis / norm
    # align_vectors(a, b) returns R with R @ b ~ a, so pass (target, source).
    rot = Rotation.align_vectors(np.array([z]), np.array([[0.0, 0.0, 1.0]]))[0]
    q = rot.as_quat()  # x, y, z, w
    return np.array([q[3], q[0], q[1], q[2]])


__all__ = ["ARM_JOINTS_BY_SIDE", "ArmJointTo", "ArmSegmentExecutor", "quat_from_z_axis"]
