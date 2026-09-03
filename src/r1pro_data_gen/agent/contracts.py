"""Contracts for one-step agent actions.

The closed-loop agent never emits a multi-stage Plan. Each LLM response is a
single public skill call. Geometric micro-skills stay registered for replay
and for composite skills, but they are outside this policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from r1pro_data_gen.planning.llm.contracts import LLMPlanValidationError, LLM_PUBLIC_SKILLS, parse_json_object


AGENT_SCHEMA_VERSION = "agent_action.v1"

# The step-wise agent and the full-plan LLM share one policy boundary.  Low
# level arm/torso/gripper/joint-mask commands remain registered for trusted
# replay and internal composite skills, but are never model-callable.
AGENT_PUBLIC_SKILLS = LLM_PUBLIC_SKILLS

_ALLOWED_ENVELOPE_KEYS = frozenset({"schema_version", "status", "reason", "action"})
_ALLOWED_ACTION_KEYS = frozenset({"skill", "parameters"})
_ENTITY_PARAM_NAMES = frozenset({"object_name", "support_surface_name"})
_NAVIGATION_PURPOSES = frozenset(
    {"pregrasp", "observe", "park", "dropoff", "navigation", "staging"}
)
_MAX_REASON = 2048
_MAX_STRING = 2048


class AgentActionValidationError(ValueError):
    """Raised when an untrusted agent response violates the action contract."""


@dataclass(frozen=True, slots=True)
class AgentAction:
    """One validated semantic skill call."""

    skill: str
    parameters: Mapping[str, Any]


def validate_action_envelope(
    data: Mapping[str, Any],
    *,
    skill_catalog: Sequence[Mapping[str, Any]] | None = None,
    registry: Any = None,
    scene_object_names: Sequence[str] = (),
    scene: Any = None,
    attachments: Mapping[str, Any] | None = None,
    object_positions: Mapping[str, Any] | None = None,
    base_pose: Sequence[float] | None = None,
) -> AgentAction | None:
    """Validate one agent response.

    Returns ``None`` for a deliberate ``unsupported`` response. Every other
    invalid payload raises :class:`AgentActionValidationError`.
    """
    if not isinstance(data, Mapping):
        raise AgentActionValidationError("agent response must be a JSON object")
    unknown = set(data) - _ALLOWED_ENVELOPE_KEYS
    if unknown:
        raise AgentActionValidationError(
            f"agent response contains unknown fields: {sorted(unknown)}"
        )
    if data.get("schema_version") != AGENT_SCHEMA_VERSION:
        raise AgentActionValidationError(
            f"unsupported agent schema_version: {data.get('schema_version')!r}"
        )
    status = data.get("status")
    if status not in {"act", "unsupported"}:
        raise AgentActionValidationError("agent status must be 'act' or 'unsupported'")
    reason = data.get("reason", "")
    if not isinstance(reason, str) or len(reason) > _MAX_REASON:
        raise AgentActionValidationError("agent reason must be a bounded string")
    if status == "unsupported":
        if data.get("action") is not None:
            raise AgentActionValidationError("unsupported response must not contain an action")
        if not reason.strip():
            raise AgentActionValidationError("unsupported response requires a reason")
        return None
    action = data.get("action")
    if not isinstance(action, Mapping):
        raise AgentActionValidationError("act response requires an action object")
    return validate_action(
        action,
        skill_catalog=skill_catalog,
        registry=registry,
        scene_object_names=scene_object_names,
        scene=scene,
        attachments=attachments,
        object_positions=object_positions,
        base_pose=base_pose,
    )


def validate_action(
    action: Mapping[str, Any],
    *,
    skill_catalog: Sequence[Mapping[str, Any]] | None = None,
    registry: Any = None,
    scene_object_names: Sequence[str] = (),
    scene: Any = None,
    attachments: Mapping[str, Any] | None = None,
    object_positions: Mapping[str, Any] | None = None,
    base_pose: Sequence[float] | None = None,
) -> AgentAction:
    """Validate the action object of an ``act`` response."""
    unknown = set(action) - _ALLOWED_ACTION_KEYS
    if unknown:
        raise AgentActionValidationError(
            f"action contains unknown fields: {sorted(unknown)}"
        )
    skill = action.get("skill")
    if not isinstance(skill, str) or skill not in AGENT_PUBLIC_SKILLS:
        raise AgentActionValidationError(
            f"skill is outside the agent policy: {skill!r}"
        )
    catalog = {item.get("name"): item for item in skill_catalog or ()}
    if catalog and skill not in catalog:
        raise AgentActionValidationError(f"skill catalogue does not describe {skill!r}")
    parameters = action.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise AgentActionValidationError("action.parameters must be an object")
    if "skill" in parameters:
        raise AgentActionValidationError("action.parameters must not repeat the skill name")
    _validate_entity_names(parameters, scene_object_names)
    if skill == "base_navigate_to":
        _validate_navigation_parameters(parameters)
        _reject_same_support_navigation(
            parameters,
            scene=scene,
            attachments=attachments,
            object_positions=object_positions,
            base_pose=base_pose,
        )
    if skill == "push_object_to":
        _validate_push_target_parameters(parameters)
    if registry is not None and hasattr(registry, "validate_plan_params"):
        try:
            registry.validate_plan_params(skill, dict(parameters))
        except (KeyError, ValueError, TypeError) as exc:
            raise AgentActionValidationError(str(exc)) from exc
    return AgentAction(skill=skill, parameters=dict(parameters))


def parse_agent_response(text: str) -> Mapping[str, Any]:
    """Parse a provider body into a JSON object."""
    try:
        payload = parse_json_object(text)
    except LLMPlanValidationError as exc:
        raise AgentActionValidationError(str(exc)) from exc
    if not isinstance(payload, Mapping):
        raise AgentActionValidationError("agent response must be a JSON object")
    return payload


def _validate_entity_names(
    parameters: Mapping[str, Any],
    scene_object_names: Sequence[str],
) -> None:
    allowed = set(scene_object_names)
    if not allowed:
        return
    for key in _ENTITY_PARAM_NAMES:
        value = parameters.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > _MAX_STRING:
            raise AgentActionValidationError(f"{key} must be a bounded scene object name")
        if value not in allowed:
            raise AgentActionValidationError(
                f"{key} {value!r} is not a top-level scene object; "
                f"valid names are {sorted(allowed)}"
            )
    target_ref = parameters.get("target_ref")
    if target_ref is not None:
        if not isinstance(target_ref, str) or not target_ref.startswith("scene://"):
            raise AgentActionValidationError(
                "target_ref must be a scene://<object> reference"
            )
        name = target_ref.split("scene://", 1)[-1].split("/", 1)[0]
        if name not in allowed:
            raise AgentActionValidationError(
                f"target_ref {target_ref!r} is not a top-level scene object; "
                f"valid names are {sorted(allowed)}"
            )
    target_region_name = parameters.get("target_region_name")
    if target_region_name is not None:
        if (
            not isinstance(target_region_name, str)
            or not target_region_name.strip()
            or len(target_region_name) > _MAX_STRING
        ):
            raise AgentActionValidationError(
                "target_region_name must be a bounded scene object[/region] reference"
            )
        region_object = target_region_name.split("/", 1)[0]
        if region_object not in allowed:
            raise AgentActionValidationError(
                f"target_region_name {target_region_name!r} is not a top-level scene object; "
                f"valid names are {sorted(allowed)}"
            )


def _validate_navigation_parameters(parameters: Mapping[str, Any]) -> None:
    target = parameters.get("target")
    target_ref = parameters.get("target_ref")
    if (target is None) == (target_ref is None):
        raise AgentActionValidationError(
            "base_navigate_to requires exactly one of target=[x,y,yaw] or "
            "target_ref='scene://<object>'"
        )
    if target is not None:
        if (
            not isinstance(target, (list, tuple))
            or len(target) != 3
            or any(isinstance(value, bool) for value in target)
        ):
            raise AgentActionValidationError(
                "base_navigate_to target must be a finite [x, y, yaw] array"
            )
        try:
            numeric_target = tuple(float(value) for value in target)
        except (TypeError, ValueError) as exc:
            raise AgentActionValidationError(
                "base_navigate_to target must be a finite [x, y, yaw] array"
            ) from exc
        import math

        if not all(math.isfinite(value) for value in numeric_target):
            raise AgentActionValidationError(
                "base_navigate_to target must be a finite [x, y, yaw] array"
            )
    elif not isinstance(target_ref, str) or not target_ref.startswith("scene://"):
        raise AgentActionValidationError(
            "base_navigate_to target_ref must use 'scene://<object>'"
        )
    purpose = parameters.get("purpose")
    if purpose is not None:
        if (
            not isinstance(purpose, str)
            or not purpose.strip()
            or len(purpose) > _MAX_STRING
        ):
            raise AgentActionValidationError("purpose must be a bounded string")
        if purpose not in _NAVIGATION_PURPOSES:
            raise AgentActionValidationError(
                "purpose must be one of pregrasp, observe, park, dropoff, "
                "navigation, or staging"
            )
    approach_side = parameters.get("approach_side")
    if approach_side is not None and approach_side not in {"west", "east", "south", "north"}:
        raise AgentActionValidationError(
            "approach_side must be one of west, east, south, north"
        )


def _reject_same_support_navigation(
    parameters: Mapping[str, Any],
    *,
    scene: Any,
    attachments: Mapping[str, Any] | None,
    object_positions: Mapping[str, Any] | None,
    base_pose: Sequence[float] | None,
) -> None:
    """Forbid navigating back to the support that currently holds an attached object."""
    attached = _attached_object_names(attachments)
    if not attached:
        return
    target_name = _navigation_target_name(parameters)
    if target_name is not None and target_name in attached:
        raise AgentActionValidationError(
            "same_support_navigation_forbidden: do not navigate to an attached object; "
            "use arm_carry_object_to on the current support"
        )
    current_support = _support_for_attached_object(
        attached,
        scene=scene,
        object_positions=object_positions,
        base_pose=base_pose,
    )
    target_support = _navigation_target_support(
        parameters,
        scene=scene,
        object_positions=object_positions,
        attached=attached,
        current_support=current_support,
        base_pose=base_pose,
    )
    if current_support and target_support and current_support == target_support:
        raise AgentActionValidationError(
            "same_support_navigation_forbidden: the place region is on the current "
            "support; use arm_carry_object_to instead of base_navigate_to"
        )


def _attached_object_names(attachments: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(attachments, Mapping) or not attachments:
        return ()
    names = []
    for key, value in attachments.items():
        if not isinstance(key, str) or not key:
            continue
        if value in {None, False, ""}:
            continue
        names.append(key)
    return tuple(names)


def _navigation_target_name(parameters: Mapping[str, Any]) -> str | None:
    target_ref = parameters.get("target_ref")
    if not isinstance(target_ref, str) or not target_ref.startswith("scene://"):
        return None
    return target_ref.split("scene://", 1)[-1].split("/", 1)[0]


def _navigation_target_xy(
    parameters: Mapping[str, Any],
    *,
    scene: Any,
    object_positions: Mapping[str, Any] | None,
) -> tuple[float, float] | None:
    target = parameters.get("target")
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        try:
            return (float(target[0]), float(target[1]))
        except (TypeError, ValueError):
            return None
    name = _navigation_target_name(parameters)
    if name is None:
        return None
    return _named_xy(name, scene=scene, object_positions=object_positions)


def _navigation_target_support(
    parameters: Mapping[str, Any],
    *,
    scene: Any,
    object_positions: Mapping[str, Any] | None,
    attached: Sequence[str],
    current_support: str | None,
    base_pose: Sequence[float] | None,
) -> str | None:
    del attached, base_pose
    name = _navigation_target_name(parameters)
    if name is not None:
        obj = _scene_object(scene, name)
        if obj is not None and _is_support_object(obj):
            return name
        if name == current_support:
            return name
    xy = _navigation_target_xy(
        parameters, scene=scene, object_positions=object_positions
    )
    if xy is None:
        return None
    return _support_at_xy(scene, xy, object_positions=object_positions)


def _support_for_attached_object(
    attached: Sequence[str],
    *,
    scene: Any,
    object_positions: Mapping[str, Any] | None,
    base_pose: Sequence[float] | None,
) -> str | None:
    for name in attached:
        xy = _named_xy(name, scene=scene, object_positions=object_positions)
        if xy is None:
            continue
        support = _support_at_xy(scene, xy, object_positions=object_positions)
        if support is not None:
            return support
    if base_pose is not None and len(base_pose) >= 2:
        try:
            xy = (float(base_pose[0]), float(base_pose[1]))
        except (TypeError, ValueError):
            xy = None
        if xy is not None:
            return _support_at_xy(scene, xy, object_positions=object_positions)
    return None


def _support_at_xy(
    scene: Any,
    xy: tuple[float, float],
    *,
    object_positions: Mapping[str, Any] | None,
) -> str | None:
    best_name: str | None = None
    best_z: float | None = None
    for obj in tuple(getattr(scene, "objects", ()) or ()):
        if not _is_support_object(obj):
            continue
        name = getattr(obj, "name", None)
        if not isinstance(name, str):
            continue
        position = _named_xyz(name, scene=scene, object_positions=object_positions)
        if position is None or not _xy_inside_object(obj, position, xy):
            continue
        height = float(position[2])
        if best_z is None or height > best_z:
            best_name = name
            best_z = height
    return best_name


def _is_support_object(obj: Any) -> bool:
    capabilities = {str(item) for item in (getattr(obj, "capabilities", ()) or ())}
    if "supports_objects" in capabilities:
        return True
    surfaces = getattr(obj, "surfaces", None)
    return bool(surfaces)


def _scene_object(scene: Any, name: str) -> Any:
    for obj in tuple(getattr(scene, "objects", ()) or ()):
        if getattr(obj, "name", None) == name:
            return obj
    getter = getattr(scene, "object", None)
    if callable(getter):
        try:
            return getter(name)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _named_xy(
    name: str,
    *,
    scene: Any,
    object_positions: Mapping[str, Any] | None,
) -> tuple[float, float] | None:
    xyz = _named_xyz(name, scene=scene, object_positions=object_positions)
    if xyz is None:
        return None
    return (xyz[0], xyz[1])


def _named_xyz(
    name: str,
    *,
    scene: Any,
    object_positions: Mapping[str, Any] | None,
) -> tuple[float, float, float] | None:
    if isinstance(object_positions, Mapping):
        raw = object_positions.get(name)
        if raw is not None and len(raw) >= 3:
            try:
                return (float(raw[0]), float(raw[1]), float(raw[2]))
            except (TypeError, ValueError):
                pass
    obj = _scene_object(scene, name)
    authored = getattr(obj, "pos", None) if obj is not None else None
    if authored is not None and len(authored) >= 3:
        try:
            return (float(authored[0]), float(authored[1]), float(authored[2]))
        except (TypeError, ValueError):
            return None
    return None


def _xy_inside_object(
    obj: Any,
    origin: Sequence[float],
    xy: tuple[float, float],
) -> bool:
    from r1pro_data_gen.domain import object_xy_half_extents_m

    try:
        half_x, half_y = object_xy_half_extents_m(obj)
    except (AttributeError, TypeError, ValueError):
        return False
    try:
        return abs(float(xy[0]) - float(origin[0])) <= float(half_x) and abs(
            float(xy[1]) - float(origin[1])
        ) <= float(half_y)
    except (TypeError, ValueError):
        return False


def _validate_push_target_parameters(parameters: Mapping[str, Any]) -> None:
    """Require one unambiguous semantic/world target for the push skill."""
    provided = sum(
        parameters.get(name) is not None
        for name in ("target_ref", "target_region_name", "target_pose")
    )
    if provided != 1:
        raise AgentActionValidationError(
            "push_object_to requires exactly one of target_ref, "
            "target_region_name, or target_pose"
        )


__all__ = [
    "AGENT_PUBLIC_SKILLS",
    "AGENT_SCHEMA_VERSION",
    "AgentAction",
    "AgentActionValidationError",
    "parse_agent_response",
    "validate_action",
    "validate_action_envelope",
]
