"""Compact per-step observations for the closed-loop agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from r1pro_data_gen.evaluation.predicates import VerificationReport
from r1pro_data_gen.skills import SkillResult


def build_agent_observation(
    *,
    adapter: Any,
    scene: Any = None,
    scene_facts: Mapping[str, Any] | None = None,
    last_result: SkillResult | None = None,
    last_skill: str | None = None,
    last_parameters: Mapping[str, Any] | None = None,
    progress: VerificationReport | None = None,
    remaining_actions: int,
    skill_catalogue: Sequence[Mapping[str, Any]] = (),
    prior_feedback: Sequence[Mapping[str, Any]] = (),
    plan_skeleton: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded JSON observation. Missing sensors become null, not guesses."""
    live = _live_state(adapter, scene)
    navigation = {}
    if isinstance(scene_facts, Mapping):
        navigation = scene_facts.get("navigation", {}) if isinstance(scene_facts.get("navigation"), Mapping) else {}
    return {
        "goal_progress": _progress_payload(progress),
        "terminal": bool(
            progress is not None and progress.status.value in {"succeeded", "failed"}
        ),
        "live": live,
        "last_action": _last_action(last_skill, last_parameters, last_result),
        "navigation_candidates": _compact_navigation_candidates(
            list(navigation.get("approach_candidates") or ()),
            scene,
        ),
        "remaining_actions": int(remaining_actions),
        "skill_catalogue": list(skill_catalogue),
        "plan_skeleton": dict(plan_skeleton or {}),
        # Feedback is deliberately factual and bounded. The provider still
        # chooses the next semantic skill; this field is not a repair policy.
        "prior_attempt_feedback": [dict(item) for item in prior_feedback],
    }


def _progress_payload(progress: VerificationReport | None) -> dict[str, Any]:
    if progress is None:
        return {"status": "unknown", "predicates": []}
    return {
        "status": progress.status.value,
        "predicates": [
            {
                "predicate": item.predicate,
                "status": item.status.value,
                "invariant": item.invariant,
                "reason": item.reason,
            }
            for item in progress.predicates
        ],
        "failure_reason": progress.failure_reason,
    }


def _last_action(
    skill: str | None,
    parameters: Mapping[str, Any] | None,
    result: SkillResult | None,
) -> dict[str, Any] | None:
    if skill is None and result is None:
        return None
    details = result.details if result is not None else {}
    metrics = result.metrics if result is not None else {}
    return {
        "skill": skill or (result.skill if result is not None else None),
        "parameters": dict(parameters or {}),
        "success": None if result is None else bool(result.success),
        "failure_code": (
            details.get("failure_code")
            or details.get("error_code")
            or metrics.get("failure_code")
        ),
        "reason": details.get("reason"),
    }


_FURNITURE_MAX_XY_M = 2.0


def _compact_navigation_candidates(
    candidates: Sequence[Any],
    scene: Any,
) -> list[Any]:
    """Drop perimeter-wall candidates that are not interaction stances.

    Scene facts emit an approach pose for every collision cuboid, including
    room fences.  Those poses drown the agent observation and are not valid
    stances for grasping a movable object.  Keep furniture-scale obstacles,
    authored regions/surfaces, detected supports, and the movable objects
    themselves.
    """
    if scene is None or not hasattr(scene, "objects"):
        return list(candidates)
    keep = _interaction_obstacle_names(scene)
    return [
        item
        for item in candidates
        if isinstance(item, Mapping) and item.get("obstacle_name") in keep
    ]


def _interaction_obstacle_names(scene: Any) -> set[str]:
    names: set[str] = set()
    objects = tuple(getattr(scene, "objects", ()) or ())
    for obj in objects:
        name = getattr(obj, "name", None)
        if not isinstance(name, str):
            continue
        physics = getattr(obj, "physics", None)
        kinematic = bool(getattr(physics, "kinematic", False))
        if not kinematic:
            names.add(name)
        if getattr(obj, "regions", None) or getattr(obj, "surfaces", None):
            names.add(name)
        size = getattr(obj, "size", None)
        if (
            isinstance(size, (tuple, list))
            and len(size) >= 2
            and max(float(size[0]), float(size[1])) <= _FURNITURE_MAX_XY_M
        ):
            names.add(name)
    return names


def _live_state(adapter: Any, scene: Any) -> dict[str, Any]:
    observation = None
    if adapter is not None and hasattr(adapter, "read_observation"):
        try:
            observation = adapter.read_observation(0.0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            observation = None
    base_pose = list(getattr(observation, "base_pose", None) or ())
    objects: dict[str, Any] = {}
    if scene is not None and hasattr(scene, "objects") and hasattr(adapter, "object_position"):
        for item in scene.objects:
            name = getattr(item, "name", None)
            if not isinstance(name, str):
                continue
            try:
                objects[name] = list(adapter.object_position(name))
            except (KeyError, RuntimeError, TypeError, ValueError):
                objects[name] = None
    contacts_left = _finger_contacts(adapter, "left")
    contacts_right = _finger_contacts(adapter, "right")
    attachments = None
    if adapter is not None and hasattr(adapter, "attachment_state"):
        try:
            attachments = dict(adapter.attachment_state())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            attachments = None
    physical = _physical_state(adapter, observation)
    return {
        "base_pose": base_pose or None,
        "base_orientation": _optional_sequence(
            getattr(observation, "base_orientation", None), 4
        ),
        "base_height_m": _optional_number(
            getattr(observation, "base_height_m", None)
        ),
        "base_velocity": _optional_sequence(
            getattr(observation, "base_velocity", None), 3
        ),
        "objects": objects,
        "contacts_left": contacts_left,
        "contacts_right": contacts_right,
        "contacts": {"left": contacts_left, "right": contacts_right},
        "attachments": attachments,
        "physical_integrity": physical,
    }


def _physical_state(adapter: Any, observation: Any) -> dict[str, Any]:
    """Expose bounded physical facts without exposing low-level commands.

    The semantic agent needs to know when a posture or contact has become
    physically unsafe so it can stop choosing the same high-level action.  It
    does not need joint targets or actuator APIs; those remain behind the
    skill boundary.  Missing telemetry is represented explicitly and is
    therefore handled as an execution/acceptance failure rather than guessed
    as a safe state.
    """
    metrics = getattr(observation, "physical_metrics", None)
    if not isinstance(metrics, Mapping) and adapter is not None:
        reader = getattr(adapter, "physical_metrics", None)
        if callable(reader):
            try:
                metrics = reader()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                metrics = None
    result: dict[str, Any] = dict(metrics) if isinstance(metrics, Mapping) else {}
    support = getattr(observation, "support_contacts", None)
    if not isinstance(support, Mapping) and adapter is not None:
        reader = getattr(adapter, "support_contact_forces", None)
        if callable(reader):
            try:
                support = reader()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                support = None
    result["support_contacts"] = (
        {str(key): float(value) for key, value in support.items()}
        if isinstance(support, Mapping)
        else None
    )
    result["imu_linear_acceleration"] = _optional_sequence(
        getattr(observation, "imu_linear_acceleration", None), 3
    )
    result["imu_angular_velocity"] = _optional_sequence(
        getattr(observation, "imu_angular_velocity", None), 3
    )
    violation = None
    if adapter is not None:
        reader = getattr(adapter, "physical_safety_violation", None)
        if callable(reader):
            try:
                violation = reader()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                violation = None
    result["safety_violation"] = violation
    return result


def _optional_sequence(value: Any, length: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return values if len(values) == length else None


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _finger_contacts(adapter: Any, side: str) -> list[float] | None:
    if adapter is None or not hasattr(adapter, "finger_contact_forces"):
        return None
    try:
        return list(adapter.finger_contact_forces(side=side))
    except TypeError:
        if side != "left":
            return None
        try:
            return list(adapter.finger_contact_forces())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


__all__ = ["build_agent_observation"]
