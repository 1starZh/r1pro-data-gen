"""Named workspace postures for the closed-loop agent.

The agent names a profile, not four torso joint angles. Tabletop, carry, and
travel map to the standing torso. Floor uses the certified whole-body
pregrasp backend when an object is available so a low support is not reached
by a guessed crouch vector.
"""

from __future__ import annotations

from typing import Any, Callable

from r1pro_data_gen.robot.robot_config import (
    R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD,
    R1PRO_WORKSPACE_TORSO_Q,
)

from ..core.base import ParamSpec, SkillResult
from ..core.sides import require_side, resolve_side
from .torso import TORSO_JOINTS


WORKSPACE_PROFILES = ("tabletop", "floor", "carry", "travel")
_STANDING_PROFILES = frozenset({"tabletop", "carry", "travel"})


class PrepareWorkspace:
    """Move the upper body to a named workspace profile."""

    name = "prepare_workspace"
    tier = "semantic"
    exposed = True
    description = (
        "Move the torso and upper body to a named workspace profile. "
        "Use tabletop when a table grasp needs a standing height, floor before "
        "a ground or low-support grasp, carry after attachment, and travel "
        "before a long navigation. Do not pass joint angles."
    )
    parameters: dict[str, ParamSpec] = {
        "profile": ParamSpec(
            "string",
            "Named workspace height/posture",
            required=True,
            enum=WORKSPACE_PROFILES,
        ),
        "object_name": ParamSpec(
            "string",
            "Live object used by the floor profile; omit to infer the lowest nearby graspable object",
            default=None,
        ),
        "side": ParamSpec(
            "string",
            "Arm side used by the floor profile; omit or use auto",
            default="auto",
            enum=("auto", "left", "right"),
        ),
    }

    def __init__(self, torso_move_to: Any, whole_body_pregrasp: Any = None) -> None:
        self.torso_move_to = torso_move_to
        self.whole_body_pregrasp = whole_body_pregrasp

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        profile: str | None = None,
        object_name: str | None = None,
        side: str = "auto",
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if profile not in WORKSPACE_PROFILES:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": f"unknown workspace profile: {profile!r}",
                    "failure_code": "unknown_workspace_profile",
                },
            )
        if profile in _STANDING_PROFILES:
            return self._move_standing(adapter, scene, profile, step_hook)
        return self._prepare_floor(
            adapter,
            scene=scene,
            object_name=object_name,
            side=side,
            step_hook=step_hook,
        )

    def _move_standing(
        self,
        adapter: Any,
        scene: Any,
        profile: str,
        step_hook: Callable[[], None] | None,
    ) -> SkillResult:
        if self.torso_move_to is None:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "standing workspace backend is unavailable",
                    "failure_code": "workspace_backend_unavailable",
                    "profile": profile,
                },
            )
        torso_q = list(R1PRO_WORKSPACE_TORSO_Q[profile])
        if _torso_already_at(adapter, torso_q):
            return SkillResult(
                True,
                self.name,
                metrics={"final_error_rad": 0.0, "steps": 0.0},
                details={"profile": profile, "torso_q": torso_q, "already_prepared": True},
            )
        result = self.torso_move_to.execute(
            adapter,
            scene=scene,
            target_q=torso_q,
            step_hook=step_hook,
        )
        details = dict(getattr(result, "details", {}) or {})
        details.update({"profile": profile, "torso_q": torso_q})
        return SkillResult(
            bool(result.success),
            self.name,
            metrics=dict(getattr(result, "metrics", {}) or {}),
            details=details,
        )

    def _prepare_floor(
        self,
        adapter: Any,
        *,
        scene: Any,
        object_name: str | None,
        side: str,
        step_hook: Callable[[], None] | None,
    ) -> SkillResult:
        resolved_object = object_name or _lowest_graspable_object(adapter, scene)
        if not resolved_object:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "floor profile needs a live graspable object",
                    "failure_code": "floor_target_unavailable",
                    "profile": "floor",
                },
            )
        if self.whole_body_pregrasp is None:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "floor workspace backend is unavailable",
                    "failure_code": "workspace_backend_unavailable",
                    "profile": "floor",
                    "object_name": resolved_object,
                },
            )
        requested = require_side(side, allow_auto=True)
        resolved_side = resolve_side(
            requested, adapter, object_name=resolved_object
        )
        result = self.whole_body_pregrasp.execute(
            adapter,
            scene=scene,
            object_name=resolved_object,
            side=resolved_side,
            step_hook=step_hook,
        )
        details = dict(getattr(result, "details", {}) or {})
        details.update(
            {
                "profile": "floor",
                "object_name": resolved_object,
                "side": resolved_side,
            }
        )
        return SkillResult(
            bool(result.success),
            self.name,
            metrics=dict(getattr(result, "metrics", {}) or {}),
            details=details,
        )


def _lowest_graspable_object(adapter: Any, scene: Any) -> str | None:
    objects = tuple(getattr(scene, "objects", ()) or ())
    lowest_name: str | None = None
    lowest_z: float | None = None
    for obj in objects:
        name = getattr(obj, "name", None)
        if not isinstance(name, str) or not name:
            continue
        if not _is_graspable(obj):
            continue
        position = _object_xyz(adapter, obj)
        if position is None:
            continue
        height = float(position[2])
        if lowest_z is None or height < lowest_z:
            lowest_name = name
            lowest_z = height
    return lowest_name


def _torso_already_at(adapter: Any, target_q: list[float]) -> bool:
    try:
        observation = adapter.read_observation(0.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    positions = getattr(observation, "joint_positions", {}) or {}
    try:
        error = max(
            abs(float(positions[name]) - float(target_q[index]))
            for index, name in enumerate(TORSO_JOINTS)
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return error <= R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD


def _is_graspable(obj: Any) -> bool:
    capabilities = {
        str(item)
        for item in (getattr(obj, "capabilities", ()) or ())
    }
    if "graspable" in capabilities or "movable" in capabilities:
        return True
    physics = getattr(obj, "physics", None)
    kinematic = bool(getattr(physics, "kinematic", False))
    return not kinematic and not _is_support_capability(capabilities)


def _is_support_capability(capabilities: set[str]) -> bool:
    return "supports_objects" in capabilities


def _object_xyz(adapter: Any, obj: Any) -> tuple[float, float, float] | None:
    name = getattr(obj, "name", None)
    if name and adapter is not None and hasattr(adapter, "object_position"):
        try:
            position = adapter.object_position(name)
            if position is not None and len(position) >= 3:
                return (float(position[0]), float(position[1]), float(position[2]))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
    authored = getattr(obj, "pos", None)
    if authored is not None and len(authored) >= 3:
        try:
            return (float(authored[0]), float(authored[1]), float(authored[2]))
        except (TypeError, ValueError):
            return None
    return None


__all__ = ["PrepareWorkspace", "WORKSPACE_PROFILES"]
