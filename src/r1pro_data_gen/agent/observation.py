"""Compact per-step observations for the closed-loop agent."""

from __future__ import annotations

import math
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
) -> dict[str, Any]:
    """Build a bounded JSON observation. Missing sensors become null, not guesses.

    The skill catalogue lives in the system prompt, not here. Do not include an
    ordered plan skeleton. Task-family recovery belongs in
    last_action.recovery_hint and the system prompt.
    """
    del skill_catalogue
    navigation = {}
    if isinstance(scene_facts, Mapping):
        navigation = scene_facts.get("navigation", {}) if isinstance(scene_facts.get("navigation"), Mapping) else {}
    candidates = _compact_navigation_candidates(
        list(navigation.get("approach_candidates") or ()),
        scene,
    )
    live = _live_state(
        adapter,
        scene,
        navigation_candidates=candidates,
        navigation_facts=navigation,
    )
    return {
        "goal_progress": _progress_payload(progress),
        "terminal": bool(
            progress is not None and progress.status.value in {"succeeded", "failed"}
        ),
        "live": live,
        "last_action": _last_action(last_skill, last_parameters, last_result),
        "navigation_candidates": candidates,
        "remaining_actions": int(remaining_actions),
        # Feedback is factual plus optional task-family recovery_hint.
        # It must not name a scene-specific stage list.
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
    failure_code = (
        details.get("failure_code")
        or details.get("error_code")
        or metrics.get("failure_code")
    )
    payload = {
        "skill": skill or (result.skill if result is not None else None),
        "parameters": dict(parameters or {}),
        "success": None if result is None else bool(result.success),
        "failure_code": failure_code,
        "reason": details.get("reason"),
    }
    hint = _general_recovery_hint(failure_code)
    if hint is not None:
        payload["recovery_hint"] = hint
    return payload


_GENERAL_RECOVERY_HINTS = {
    "workspace_not_prepared": (
        "Current upper-body height cannot reach the target. Call "
        "prepare_workspace with a profile that matches the support "
        "(tabletop vs floor), then retry the manipulation."
    ),
    "unreachable_from_base": (
        "Current base stance cannot reach the target. Change "
        "base_navigate_to (approach_side, target_ref, or purpose) rather "
        "than only switching arm side."
    ),
    "target_contact_not_established": (
        "Grasp did not make contact. If the object is still in reach, retry "
        "grasp_object from the live stance; navigate only if it is out of reach."
    ),
    "contact_not_centered": (
        "Object is not centered in the gripper. If it is still in reach, retry "
        "grasp_object from the live stance; otherwise change the base approach."
    ),
    "one_finger_contact": (
        "Only one finger contacted the object. If it is still in reach, retry "
        "grasp_object; do not treat one-sided contact as a successful grasp."
    ),
    "grasp_not_attached": (
        "Object did not attach. If it is still in reach, retry grasp_object "
        "from the live stance; navigate only if it is out of reach."
    ),
    "same_support_navigation_forbidden": (
        "Destination is on the support that currently holds the attached "
        "object. Use arm_carry_object_to rather than driving around the same "
        "support."
    ),
}


def _general_recovery_hint(failure_code: object) -> str | None:
    if not isinstance(failure_code, str) or not failure_code:
        return None
    return _GENERAL_RECOVERY_HINTS.get(failure_code)


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


def _live_state(
    adapter: Any,
    scene: Any,
    *,
    navigation_candidates: Sequence[Any] = (),
    navigation_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observation = None
    if adapter is not None and hasattr(adapter, "read_observation"):
        try:
            observation = adapter.read_observation(0.0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            observation = None
    base_pose = list(getattr(observation, "base_pose", None) or ())
    objects: dict[str, Any] = {}
    if scene is not None and hasattr(scene, "objects"):
        supports = [item for item in scene.objects if _is_support_object(item)]
        for item in scene.objects:
            name = getattr(item, "name", None)
            if not isinstance(name, str):
                continue
            position = _live_object_position(adapter, item)
            objects[name] = _object_record(
                item,
                position=position,
                base_pose=base_pose,
                supports=supports,
                navigation_candidates=navigation_candidates,
                navigation_facts=navigation_facts,
            )
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


def _live_object_position(adapter: Any, item: Any) -> list[float] | None:
    name = getattr(item, "name", None)
    if adapter is not None and hasattr(adapter, "object_position") and isinstance(name, str):
        try:
            values = [float(value) for value in adapter.object_position(name)]
        except (KeyError, RuntimeError, TypeError, ValueError):
            values = []
        if len(values) >= 3 and all(math.isfinite(value) for value in values[:3]):
            return values[:3]
    pos = getattr(item, "pos", None)
    if isinstance(pos, (tuple, list)) and len(pos) >= 3:
        try:
            values = [float(pos[0]), float(pos[1]), float(pos[2])]
        except (TypeError, ValueError):
            return None
        if all(math.isfinite(value) for value in values):
            return values
    return None


def _object_record(
    item: Any,
    *,
    position: list[float] | None,
    base_pose: Sequence[float],
    supports: Sequence[Any],
    navigation_candidates: Sequence[Any],
    navigation_facts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    name = str(getattr(item, "name", ""))
    capabilities = [
        str(value)
        for value in (getattr(item, "capabilities", ()) or ())
        if str(value)
    ]
    return {
        "position": position,
        "size": _object_size(item),
        "capabilities": capabilities,
        "on_support": _infer_on_support(item, position, supports),
        "planar_distance_m": _planar_distance_m(base_pose, position),
        "reachable_from_here": _reachable_from_here(
            name,
            base_pose,
            navigation_candidates=navigation_candidates,
            navigation_facts=navigation_facts,
        ),
    }


def _object_size(item: Any) -> dict[str, Any] | None:
    size = getattr(item, "size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 3:
        try:
            values = [float(size[0]), float(size[1]), float(size[2])]
        except (TypeError, ValueError):
            values = []
        if len(values) == 3 and all(math.isfinite(value) and value > 0.0 for value in values):
            return {"shape": "cuboid", "xyz": values}
    radius = getattr(item, "radius", None)
    height = getattr(item, "height", None)
    try:
        radius_m = float(radius) if radius is not None else None
        height_m = float(height) if height is not None else None
    except (TypeError, ValueError):
        return None
    if (
        radius_m is not None
        and height_m is not None
        and math.isfinite(radius_m)
        and math.isfinite(height_m)
        and radius_m > 0.0
        and height_m > 0.0
    ):
        return {"shape": "cylinder", "radius": radius_m, "height": height_m}
    return None


def _is_support_object(item: Any) -> bool:
    capabilities = {str(value) for value in (getattr(item, "capabilities", ()) or ())}
    if "supports_objects" in capabilities:
        return True
    return bool(getattr(item, "surfaces", None))


def _infer_on_support(
    item: Any,
    position: list[float] | None,
    supports: Sequence[Any],
) -> str | None:
    """Nearest supporting object by XY footprint and top-to-bottom gap.

    Geometry only: no scene names, task families, or authored recipes.
    """
    if position is None or len(position) < 3:
        return None
    item_name = getattr(item, "name", None)
    object_bottom = position[2] - 0.5 * _vertical_extent_m(item)
    object_radius = _xy_radius_m(item)
    ranked: list[tuple[float, float, str]] = []
    for support in supports:
        name = getattr(support, "name", None)
        if not isinstance(name, str) or name == item_name:
            continue
        centre = getattr(support, "pos", None)
        if not isinstance(centre, (tuple, list)) or len(centre) < 2:
            continue
        if not _xy_inside_support(position, support, object_radius):
            continue
        try:
            top_z = float(getattr(support, "top_z"))
        except (TypeError, ValueError, AttributeError):
            top_z = float(centre[2]) + 0.5 * _vertical_extent_m(support) if len(centre) >= 3 else None
        if top_z is None:
            continue
        gap = abs(top_z - object_bottom)
        if gap > 0.05:
            continue
        ranked.append((gap, _footprint_area_m2(support), name))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]


def _xy_inside_support(position: Sequence[float], support: Any, object_radius: float) -> bool:
    centre = getattr(support, "pos", None)
    if not isinstance(centre, (tuple, list)) or len(centre) < 2:
        return False
    try:
        dx = abs(float(position[0]) - float(centre[0]))
        dy = abs(float(position[1]) - float(centre[1]))
    except (TypeError, ValueError, IndexError):
        return False
    size = getattr(support, "size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        try:
            half_x = 0.5 * float(size[0]) + object_radius + 0.01
            half_y = 0.5 * float(size[1]) + object_radius + 0.01
        except (TypeError, ValueError):
            return False
        return dx <= half_x and dy <= half_y
    radius = getattr(support, "radius", None)
    if radius is None:
        return False
    try:
        limit = float(radius) + object_radius + 0.01
    except (TypeError, ValueError):
        return False
    return (dx * dx + dy * dy) <= limit * limit


def _vertical_extent_m(item: Any) -> float:
    size = getattr(item, "size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 3:
        try:
            value = float(size[2])
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            return value
    height = getattr(item, "height", None)
    try:
        value = float(height) if height is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _xy_radius_m(item: Any) -> float:
    radius = getattr(item, "radius", None)
    if radius is not None:
        try:
            value = float(radius)
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            return value
    size = getattr(item, "size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        try:
            value = 0.5 * min(float(size[0]), float(size[1]))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value > 0.0 else 0.0
    return 0.0


def _footprint_area_m2(item: Any) -> float:
    size = getattr(item, "size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        try:
            area = abs(float(size[0]) * float(size[1]))
        except (TypeError, ValueError):
            return float("inf")
        return area if math.isfinite(area) else float("inf")
    radius = getattr(item, "radius", None)
    try:
        value = float(radius) if radius is not None else 0.0
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(value) or value <= 0.0:
        return float("inf")
    return math.pi * value * value


def _planar_distance_m(
    base_pose: Sequence[float],
    position: Sequence[float] | None,
) -> float | None:
    if position is None or len(position) < 2 or len(base_pose) < 2:
        return None
    try:
        dx = float(position[0]) - float(base_pose[0])
        dy = float(position[1]) - float(base_pose[1])
    except (TypeError, ValueError):
        return None
    distance = math.hypot(dx, dy)
    return round(distance, 4) if math.isfinite(distance) else None


def _reachable_from_here(
    object_name: str,
    base_pose: Sequence[float],
    *,
    navigation_candidates: Sequence[Any],
    navigation_facts: Mapping[str, Any] | None,
) -> bool | None:
    """Whether the current base is already at an IK-reachable approach stance.

    Uses scene-fact approach candidates only. Missing IK data stays null so
    the agent does not invent a workspace radius.
    """
    if len(base_pose) < 2:
        return None
    try:
        base_x = float(base_pose[0])
        base_y = float(base_pose[1])
    except (TypeError, ValueError):
        return None
    reachable_poses: list[tuple[float, float]] = []
    saw_ik = False
    for candidate in navigation_candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("obstacle_name") != object_name:
            continue
        annotations = candidate.get("ik_reachability")
        if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
            continue
        saw_ik = True
        if not any(
            isinstance(item, Mapping) and item.get("reachable") is True
            for item in annotations
        ):
            continue
        pose = candidate.get("pose")
        if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)) or len(pose) < 2:
            continue
        try:
            reachable_poses.append((float(pose[0]), float(pose[1])))
        except (TypeError, ValueError):
            continue
    if not saw_ik:
        return None
    if not reachable_poses:
        return False
    tolerance = _stance_tolerance_m(navigation_facts)
    if tolerance is None:
        return None
    return any(
        math.hypot(base_x - pose_x, base_y - pose_y) <= tolerance
        for pose_x, pose_y in reachable_poses
    )


def _stance_tolerance_m(navigation_facts: Mapping[str, Any] | None) -> float | None:
    if not isinstance(navigation_facts, Mapping):
        return None
    radius = _optional_number(navigation_facts.get("footprint_radius_m"))
    if radius is None:
        return None
    clearance = _optional_number(navigation_facts.get("inflation_clearance_m")) or 0.0
    return radius + clearance + 0.05


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
