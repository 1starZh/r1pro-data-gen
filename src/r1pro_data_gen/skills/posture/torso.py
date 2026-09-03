"""Torso skill: move the 4-DOF torso to a target joint configuration.

The torso adjusts the upper body height/pitch, which changes the arm's base
reachability. It uses position control with speed-limited interpolation (the
torso moves under its own conservative velocity limit -- the reference project
reports steady-state pitch error at 2000/500 gains, so segments hold briefly).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.base import ParamSpec, SkillResult
from ..manipulation.arm import trapezoid_progress, trapezoid_scale

TORSO_JOINTS = tuple(f"torso_joint{i}" for i in range(1, 5))
# Conservative per-joint velocity limit (rad/s); no authored limit in the
# USDA, so keep the torso slow and smooth.
TORSO_VEL_LIMIT = np.array([0.5, 0.5, 0.5, 0.5])

_FINAL_ERROR_TOL = 0.10  # rad; gravity-loaded pitch of ~1.2 rad settles ~0.05 rad off target
_MIN_SETTLE_STEPS = 48  # 0.8 s at the 60 Hz physics cadence


class TorsoMoveTo:
    """Move the torso to a target joint configuration (4 values)."""

    name = "torso_move_to"
    tier = "backend"
    exposed = False
    description = "Move the torso to a target joint configuration (4 values, rad)."
    parameters: dict[str, ParamSpec] = {
        "target_q": ParamSpec("array", "Target torso joint positions (4, rad)", required=True),
        "speed_scale": ParamSpec("number", "Fraction of the torso velocity limits", default=0.7),
    }

    def __init__(self, vel_limits: np.ndarray = TORSO_VEL_LIMIT, hold_steps: int = 48):
        self.vel_limits = np.asarray(vel_limits, dtype=float)
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target_q: tuple[float, ...] | list[float] = None,
        speed_scale: float = 0.7,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        if target_q is None:
            raise ValueError("torso_move_to requires target_q (4 values)")
        obs = adapter.read_observation(0.0)
        q_from = np.array([obs.joint_positions[j] for j in TORSO_JOINTS], dtype=float)
        q_to = np.asarray(target_q, dtype=float)
        if q_to.shape != (4,):
            raise ValueError(f"torso_move_to target_q must be 4 values, got {q_to.shape}")

        displacement = np.abs(q_to - q_from)
        allowed = np.maximum(np.abs(self.vel_limits) * speed_scale, 1e-6)
        dt = 1.0 / 60.0
        cruise_steps = max(30, int(np.ceil(float(np.max(displacement / allowed)) / dt)) + 1)
        # Same trapezoid the arm uses: a constant-velocity linear segment
        # commanded a non-zero velocity on the first step and saturated
        # torso_joint1 against the 100 N·m clamp.
        steps = max(3, int(np.ceil(cruise_steps / (1.0 - 2.0 * 0.25))))
        disp = q_to - q_from
        for step in range(1, steps + 1):
            u = step / steps
            q_cur = q_from + disp * trapezoid_progress(u)
            vel = disp * trapezoid_scale(u) / (steps * dt)
            adapter.set_targets(
                position={j: float(q_cur[i]) for i, j in enumerate(TORSO_JOINTS)},
                velocity={j: float(vel[i]) for i, j in enumerate(TORSO_JOINTS)},
            )
            adapter.step()
            if step_hook is not None:
                step_hook()
        adapter.set_targets(
            position={j: float(q_to[i]) for i, j in enumerate(TORSO_JOINTS)},
            velocity={j: 0.0 for i, j in enumerate(TORSO_JOINTS)},
        )
        # A moving target trajectory can finish before the articulated torso
        # has caught up under gravity/load.  The old 30-step tail (0.5 s)
        # reported a false planning failure for a valid low-workspace posture,
        # and the caller then locked the torso at that transient state.  Keep
        # this convergence window robot-level and bounded: it is independent
        # of object names, task coordinates, and scene layout.  Exit early as
        # soon as the measured target error is inside the contract tolerance.
        settle_steps = max(int(self.hold_steps), _MIN_SETTLE_STEPS)
        for _ in range(settle_steps):
            adapter.set_targets(
                position={j: float(q_to[i]) for i, j in enumerate(TORSO_JOINTS)},
                velocity={j: 0.0 for j in TORSO_JOINTS},
            )
            adapter.step()
            if step_hook is not None:
                step_hook()
            measured = adapter.read_observation(0.0)
            current_error = max(
                abs(measured.joint_positions[j] - q_to[i])
                for i, j in enumerate(TORSO_JOINTS)
            )
            if current_error < _FINAL_ERROR_TOL:
                break
        obs = adapter.read_observation(0.0)
        final_err = max(abs(obs.joint_positions[j] - q_to[i]) for i, j in enumerate(TORSO_JOINTS))
        return SkillResult(
            success=bool(final_err < _FINAL_ERROR_TOL),
            skill=self.name,
            metrics={"final_error_rad": float(final_err), "steps": float(steps)},
        )


__all__ = ["TORSO_JOINTS", "TORSO_VEL_LIMIT", "TorsoMoveTo"]
