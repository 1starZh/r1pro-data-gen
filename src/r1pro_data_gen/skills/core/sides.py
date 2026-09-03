"""Shared left/right dependency selection for side-aware atomic skills."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ARM_SIDES = ("left", "right")
ARM_SIDE_REQUESTS = ("auto", *ARM_SIDES)


def require_side(side: str, *, allow_auto: bool = False) -> str:
    """Validate an arm side, optionally accepting semantic ``auto``."""
    allowed = ARM_SIDE_REQUESTS if allow_auto else ARM_SIDES
    if side not in allowed:
        raise ValueError(f"side must be one of {allowed}, got {side!r}")
    return side


def rank_arm_sides(
    adapter: Any,
    *,
    object_name: str | None = None,
) -> tuple[str, ...]:
    """Rank available arms from live geometry with a deterministic fallback.

    This is intentionally a small shared selector, not a task policy.  It
    uses the current end-effector/object distance only to choose which of the
    two equivalent semantic backends should own the transaction.  IK,
    collision and stability certification remain the responsibility of the
    selected semantic skill.
    """
    scores: list[tuple[float, int, str]] = []
    object_position = None
    if object_name and hasattr(adapter, "object_position"):
        try:
            object_position = tuple(float(value) for value in adapter.object_position(object_name))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            object_position = None
    poses = {}
    if hasattr(adapter, "end_effector_poses"):
        try:
            poses = adapter.end_effector_poses() or {}
        except (AttributeError, RuntimeError, TypeError, ValueError):
            poses = {}
    for tie_break, side in enumerate(ARM_SIDES):
        score = float("inf")
        pose = poses.get(f"{side}_ee") or poses.get(f"{side}_gripper_finger_midpoint")
        if object_position is not None and pose is not None and len(pose) >= 3:
            try:
                score = sum(
                    (float(pose[index]) - object_position[index]) ** 2
                    for index in range(3)
                ) ** 0.5
            except (TypeError, ValueError):
                score = float("inf")
        scores.append((score, tie_break, side))
    scores.sort()
    return tuple(item[2] for item in scores)


def resolve_side(
    side: str,
    adapter: Any,
    *,
    object_name: str | None = None,
) -> str:
    """Resolve a semantic side request to one concrete arm side."""
    requested = require_side(side, allow_auto=True)
    if requested != "auto":
        return requested
    if object_name and hasattr(adapter, "attachment_state"):
        try:
            effector = (adapter.attachment_state() or {}).get(object_name, "")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            effector = ""
        for candidate in ARM_SIDES:
            if candidate in str(effector).lower():
                return candidate
    return rank_arm_sides(adapter, object_name=object_name)[0]


def for_side(value: Any, side: str) -> Any:
    """Select a side-specific dependency while preserving legacy singletons."""
    require_side(side)
    if isinstance(value, Mapping):
        if side not in value:
            raise RuntimeError(f"backend for side={side!r} is unavailable")
        return value[side]
    return value


__all__ = [
    "ARM_SIDES",
    "ARM_SIDE_REQUESTS",
    "for_side",
    "rank_arm_sides",
    "require_side",
    "resolve_side",
]
