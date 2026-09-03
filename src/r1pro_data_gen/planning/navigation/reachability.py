"""Robot-level interaction reachability contracts.

This module combines a world-space base pose with a world-space interaction
point and asks the selected arm kinematics backend for an actual IK branch.
It deliberately contains no scene names, task ordering, or task-specific pose
recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class InteractionReachabilityReport:
    """Evidence for accepting one navigation/interaction candidate."""

    navigation_free: bool
    target_reachable: bool
    accepted: bool
    reason: str
    target_position_base: tuple[float, float, float]
    ik_candidates: int = 0


def assess_interaction_target(
    *,
    candidate_pose_world: tuple[float, float, float] | list[float],
    target_position_world: tuple[float, float, float] | list[float],
    target_quaternion: tuple[float, float, float, float] | np.ndarray | None,
    target_frame: str = "ee",
    kinematics: Any,
    navigation_free: bool = True,
    q_current: np.ndarray | None = None,
) -> InteractionReachabilityReport:
    """Check whether a free-space base candidate supports an IK-reachable target.

    ``candidate_pose_world`` is the base pose ``(x, y, yaw)``.  The target
    position is converted into that base frame before asking the robot model
    for IK.  ``navigation_free`` is supplied by the occupancy planner; this
    function does not duplicate rasterization or silently snap an occupied
    candidate.
    """
    if target_frame not in {"ee", "grasp_center"}:
        raise ValueError("target_frame must be 'ee' or 'grasp_center'")
    candidate = np.asarray(candidate_pose_world, dtype=float)
    target = np.asarray(target_position_world, dtype=float)
    if candidate.shape != (3,):
        raise ValueError("candidate_pose_world must have shape (3,)")
    if target.shape != (3,):
        raise ValueError("target_position_world must have shape (3,)")
    if not np.isfinite(candidate).all() or not np.isfinite(target).all():
        raise ValueError("candidate and target positions must be finite")
    if not navigation_free:
        return InteractionReachabilityReport(
            navigation_free=False,
            target_reachable=False,
            accepted=False,
            reason="navigation candidate is occupied",
            target_position_base=(0.0, 0.0, 0.0),
        )

    x, y, yaw = (float(value) for value in candidate)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    dx, dy = float(target[0] - x), float(target[1] - y)
    target_base = np.array(
        [c * dx + s * dy, -s * dx + c * dy, float(target[2])],
        dtype=float,
    )
    quat = None
    if target_quaternion is not None:
        quat = np.asarray(target_quaternion, dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-9:
            raise ValueError("target_quaternion must be a finite non-zero quaternion")
        quat = quat / np.linalg.norm(quat)

    ik_target = target_base
    if target_frame == "grasp_center":
        if quat is None:
            raise ValueError("grasp_center target requires target_quaternion")
        ik_target = np.asarray(
            kinematics.ee_target_from_grasp_center(target_base, quat),
            dtype=float,
        )
    q_seed = np.zeros(7, dtype=float) if q_current is None else np.asarray(q_current, dtype=float)
    if q_seed.shape != (7,):
        raise ValueError("q_current must have shape (7,)")

    if not hasattr(kinematics, "ik_candidates"):
        return InteractionReachabilityReport(
            navigation_free=True,
            target_reachable=False,
            accepted=False,
            reason="robot kinematics does not expose IK candidates",
            target_position_base=tuple(float(value) for value in target_base),
        )
    candidates = kinematics.ik_candidates(
        ik_target,
        quat,
        q_seed,
        max_candidates=1,
    )
    reachable = bool(candidates)
    return InteractionReachabilityReport(
        navigation_free=True,
        target_reachable=reachable,
        accepted=reachable,
        reason="reachable" if reachable else "interaction target is not reachable",
        target_position_base=tuple(float(value) for value in target_base),
        ik_candidates=len(candidates),
    )


__all__ = ["InteractionReachabilityReport", "assess_interaction_target"]
