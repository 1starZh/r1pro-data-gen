"""Generic, feasibility-gated scene randomization.

The task layer supplies only scene YAML, natural-language instructions, and an
optional randomization YAML.  This module deliberately does not know any task
name, object name, action sequence, or benchmark-specific coordinate.  Rules
select scene entities through capabilities/semantic roles and perturb the
authored geometry in a relationship-preserving way.

Randomization is rejection sampled.  A candidate is usable only after the
same pure :class:`SceneModel` parser used by planning accepts it and the
geometric feasibility checks pass.  This keeps invalid initial states out of a
success-rate denominator instead of silently treating them as controller
failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import random
from typing import Any

from r1pro_data_gen.domain import (
    ObjectCapability,
    ObjectModel,
    SceneModel,
    object_vertical_extent_m,
    object_xy_half_extents_m,
)
from r1pro_data_gen.planning.navigation.contract import (
    NAVIGATION_GRID_RESOLUTION_M,
    NAVIGATION_INFLATION_CLEARANCE_M,
)
from r1pro_data_gen.robot.chassis import default_footprint_radius_m


RANDOMIZATION_SCHEMA_VERSION = "scene_randomization.v1"
_FEASIBILITY_TOLERANCE_M = 1e-3
_SUPPORT_HEIGHT_TOLERANCE_M = 0.04
_ROBOT_COLLISION_HEIGHT_M = 0.75


class SceneRandomizationError(ValueError):
    """Raised when no valid candidate can be sampled within the retry budget."""

    def __init__(self, message: str, *, diagnostics: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.diagnostics = tuple(str(item) for item in diagnostics)


@dataclass(frozen=True, slots=True)
class SceneFeasibilityReport:
    """Pure geometric validity report for one serialized scene candidate."""

    valid: bool
    reasons: tuple[str, ...] = ()
    support_relations: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "support_relations": [
                {"parent": parent, "child": child}
                for parent, child in self.support_relations
            ],
        }


def default_randomization_spec() -> dict[str, Any]:
    """Return the compatibility default used by the single-scene runner.

    Formal benchmark suites should provide an explicit spec.  The compatibility
    default retains the historical robot-pose perturbation while still passing
    through the common feasibility gate.
    """

    return {
        "schema_version": RANDOMIZATION_SCHEMA_VERSION,
        "max_attempts": 64,
        "preserve_relations": True,
        "robot": {"xy_radius_m": 0.50, "yaw_range_rad": math.pi},
        "objects": [],
        "physics": [],
    }


def randomize_scene_data(
    base_data: Mapping[str, Any],
    rng: random.Random,
    spec: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample one valid scene and return ``(scene_data, provenance)``.

    ``spec`` is intentionally a data contract rather than a Python callback.
    Transform rules use ``match`` selectors and symmetric perturbation bounds:

    ``xy_radius_m``
        Uniform disk radius around the authored pose.
    ``yaw_range_rad``
        Uniform symmetric yaw perturbation.
    ``z_jitter_m``
        Uniform symmetric vertical perturbation.  Relationship checks reject
        support penetration or unsupported objects.

    Object selectors may use ``role`` (``object``, ``target``, ``support``,
    ``obstacle``, ``collision``, ``all``), ``capability``, ``capabilities``,
    ``names``, ``semantic_class`` and ``aliases``.  Physics rules only scale
    already-authored ``mass``, friction, contact offset, and planning margin;
    they never toggle rigid-body or collision semantics.
    """

    if not isinstance(base_data, Mapping):
        raise TypeError("base scene data must be a mapping")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    normalized = _normalize_spec(spec)
    base = copy.deepcopy(dict(base_data))
    base_model = _parse_scene(base, "base scene")
    base_report = check_scene_feasibility(base)
    if not base_report.valid:
        raise SceneRandomizationError(
            "base scene is not geometrically feasible",
            diagnostics=base_report.reasons,
        )

    object_rules = _parse_object_rules(normalized.get("objects", ()), "objects")
    physics_rules = _parse_object_rules(normalized.get("physics", ()), "physics")
    relations = _infer_support_relations(base_model)
    active_relations = relations if normalized["preserve_relations"] else ()
    diagnostics: list[str] = []
    for attempt in range(1, normalized["max_attempts"] + 1):
        candidate = copy.deepcopy(base)
        object_raw = candidate.get("objects", [])
        if not isinstance(object_raw, list):  # guarded by SceneModel parser
            raise SceneRandomizationError("scene objects must be an array")
        raw_by_name = {
            item.get("name"): item
            for item in object_raw
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        if len(raw_by_name) != len(object_raw):
            raise SceneRandomizationError("scene objects must have unique names")

        local_transforms = {
            model.name: _sample_local_transform(model, object_rules, rng)
            for model in base_model.objects
        }
        resolved_transforms: dict[str, dict[str, float]] = {}
        visiting: set[str] = set()
        for model in base_model.objects:
            _resolve_transform(
                model.name,
                base_model,
                active_relations,
                local_transforms,
                resolved_transforms,
                visiting,
            )

        changed_objects: dict[str, Any] = {}
        for model in base_model.objects:
            raw = raw_by_name[model.name]
            transform = resolved_transforms[model.name]
            old_pos = tuple(float(value) for value in model.pos)
            new_pos = (
                old_pos[0] + transform["dx"],
                old_pos[1] + transform["dy"],
                old_pos[2] + transform["dz"],
            )
            raw["pos"] = [_round_float(value) for value in new_pos]
            if abs(transform["yaw"]) > 1e-12:
                authored_quat = raw.get("quat", [1.0, 0.0, 0.0, 0.0])
                raw["quat"] = [
                    _round_float(value)
                    for value in _quat_multiply(
                        _yaw_quaternion(transform["yaw"]), authored_quat
                    )
                ]
            if any(abs(transform[key]) > 1e-12 for key in ("dx", "dy", "dz", "yaw")):
                changed_objects[model.name] = {
                    "authored_pos": list(old_pos),
                    "sampled_pos": list(new_pos),
                    "delta_xy_m": [transform["dx"], transform["dy"]],
                    "delta_z_m": transform["dz"],
                    "delta_yaw_rad": transform["yaw"],
                }

        physics_changes = _apply_physics_rules(
            object_raw,
            base_model,
            physics_rules,
            rng,
        )
        robot_change = _sample_robot_change(normalized.get("robot"), rng)
        robot_raw = candidate.get("robot")
        if not isinstance(robot_raw, dict):
            raise SceneRandomizationError("scene robot must be a mapping")
        authored_robot_pose = list(base_model.robot.init_pose)
        robot_raw["init_pose"] = [
            _round_float(authored_robot_pose[0] + robot_change["dx"]),
            _round_float(authored_robot_pose[1] + robot_change["dy"]),
            _round_float(robot_change["yaw"] + authored_robot_pose[2]),
        ]

        report = check_scene_feasibility(
            candidate,
            expected_support_relations=relations if normalized["preserve_relations"] else None,
        )
        if report.valid:
            provenance = {
                "schema_version": RANDOMIZATION_SCHEMA_VERSION,
                "attempt": attempt,
                "base_scene_sha256": _sha256_json(base),
                "spec": normalized,
                "robot": {
                    "authored_init_pose": authored_robot_pose,
                    "sampled_init_pose": robot_raw["init_pose"],
                    "delta_xy_m": [robot_change["dx"], robot_change["dy"]],
                    "delta_yaw_rad": robot_change["yaw"],
                },
                "objects": changed_objects,
                "physics": physics_changes,
                "support_relations": [
                    {"parent": parent, "child": child}
                    for parent, child in relations
                ],
                "feasibility": report.to_dict(),
            }
            return candidate, provenance
        diagnostics.extend(report.reasons[:8])

    unique_diagnostics = tuple(dict.fromkeys(diagnostics))
    raise SceneRandomizationError(
        f"could not sample a feasible scene in {normalized['max_attempts']} attempts",
        diagnostics=unique_diagnostics,
    )


def check_scene_feasibility(
    scene_data: Mapping[str, Any],
    *,
    expected_support_relations: Sequence[tuple[str, str]] | None = None,
) -> SceneFeasibilityReport:
    """Check bounds, primitive overlap, support relations, and robot placement."""

    try:
        scene = _parse_scene(scene_data, "candidate scene")
    except (TypeError, ValueError) as exc:
        return SceneFeasibilityReport(False, (str(exc),))

    reasons: list[str] = []
    half_world_x, half_world_y = (value / 2.0 for value in scene.world.ground_size)
    relations = _infer_support_relations(scene)
    extents = {
        obj.name: object_xy_half_extents_m(obj)
        for obj in scene.objects
    }
    vertical = {
        obj.name: object_vertical_extent_m(obj)
        for obj in scene.objects
    }
    for obj in scene.objects:
        hx, hy = extents[obj.name]
        x, y, z = obj.pos
        if abs(x) + hx > half_world_x + _FEASIBILITY_TOLERANCE_M:
            reasons.append(f"object {obj.name!r} exceeds world x bounds")
        if abs(y) + hy > half_world_y + _FEASIBILITY_TOLERANCE_M:
            reasons.append(f"object {obj.name!r} exceeds world y bounds")
        bottom = z - vertical[obj.name] / 2.0
        if bottom < -_FEASIBILITY_TOLERANCE_M:
            reasons.append(f"object {obj.name!r} penetrates the ground")

    for first_index, first in enumerate(scene.objects):
        if not first.physics.collision_enabled:
            continue
        for second in scene.objects[first_index + 1 :]:
            if not second.physics.collision_enabled:
                continue
            if _objects_overlap(first, second, extents, vertical) and not _is_structural_static_overlap(
                first, second, extents
            ):
                reasons.append(
                    f"collision overlap between objects {first.name!r} and {second.name!r}"
                )

    checked_relations = (
        tuple(expected_support_relations)
        if expected_support_relations is not None
        else relations
    )
    for parent_name, child_name in checked_relations:
        try:
            parent = scene.object(parent_name)
            child = scene.object(child_name)
        except KeyError as exc:
            reasons.append(f"support relation references unknown object: {exc}")
            continue
        parent_hx, parent_hy = extents[parent.name]
        child_hx, child_hy = extents[child.name]
        dx = abs(child.pos[0] - parent.pos[0])
        dy = abs(child.pos[1] - parent.pos[1])
        if dx + child_hx > parent_hx + _FEASIBILITY_TOLERANCE_M:
            reasons.append(f"supported object {child.name!r} leaves {parent.name!r} in x")
        if dy + child_hy > parent_hy + _FEASIBILITY_TOLERANCE_M:
            reasons.append(f"supported object {child.name!r} leaves {parent.name!r} in y")
        child_bottom = child.pos[2] - vertical[child.name] / 2.0
        parent_top = parent.pos[2] + vertical[parent.name] / 2.0
        if abs(child_bottom - parent_top) > _SUPPORT_HEIGHT_TOLERANCE_M:
            reasons.append(
                f"supported object {child.name!r} is not on {parent.name!r}"
            )

    robot_x, robot_y, _ = scene.robot.init_pose
    # Keep the randomization gate aligned with the runtime navigation skill.
    # The old check used a 0.5 m centre-point overlap test, while
    # ``base_navigate_to`` inflates collision boxes by the actual chassis
    # footprint and rasterizes them at 5 cm resolution.  A pose can therefore
    # pass this gate and still start inside the skill's first occupied cell.
    footprint = scene.robot.navigation_footprint_radius_m or default_footprint_radius_m()
    navigation_clearance = (
        footprint
        + NAVIGATION_INFLATION_CLEARANCE_M
        + NAVIGATION_GRID_RESOLUTION_M
    )
    if abs(robot_x) + footprint > half_world_x + _FEASIBILITY_TOLERANCE_M:
        reasons.append("robot initial pose exceeds world x bounds")
    if abs(robot_y) + footprint > half_world_y + _FEASIBILITY_TOLERANCE_M:
        reasons.append("robot initial pose exceeds world y bounds")
    for obj in scene.objects:
        if not obj.physics.collision_enabled:
            continue
        bottom = obj.pos[2] - vertical[obj.name] / 2.0
        top = obj.pos[2] + vertical[obj.name] / 2.0
        if bottom > _ROBOT_COLLISION_HEIGHT_M or top < -_FEASIBILITY_TOLERANCE_M:
            continue
        hx, hy = extents[obj.name]
        x_distance = abs(robot_x - obj.pos[0])
        y_distance = abs(robot_y - obj.pos[1])
        if x_distance <= footprint + hx and y_distance <= footprint + hy:
            reasons.append(f"robot initial pose overlaps object {obj.name!r}")
        elif (
            x_distance <= navigation_clearance + hx
            and y_distance <= navigation_clearance + hy
        ):
            reasons.append(
                f"robot initial pose lacks navigation clearance from object {obj.name!r}"
            )

    return SceneFeasibilityReport(
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        support_relations=tuple(relations),
    )


def _normalize_spec(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    if spec is None:
        return default_randomization_spec()
    if not isinstance(spec, Mapping):
        raise TypeError("randomization spec must be a mapping")
    allowed = {
        "schema_version",
        "max_attempts",
        "preserve_relations",
        "robot",
        "objects",
        "physics",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"randomization spec contains unknown fields: {sorted(unknown)}")
    schema_version = spec.get("schema_version", RANDOMIZATION_SCHEMA_VERSION)
    if schema_version != RANDOMIZATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported randomization schema_version: {schema_version!r}")
    max_attempts = spec.get("max_attempts", 64)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("randomization max_attempts must be a positive integer")
    preserve_relations = spec.get("preserve_relations", True)
    if not isinstance(preserve_relations, bool):
        raise TypeError("randomization preserve_relations must be a boolean")
    robot = spec.get("robot", {}) or {}
    if not isinstance(robot, Mapping):
        raise TypeError("randomization robot must be a mapping")
    _validate_robot_spec(robot)
    objects = spec.get("objects", []) or []
    physics = spec.get("physics", []) or []
    if not isinstance(objects, list) or not isinstance(physics, list):
        raise TypeError("randomization objects and physics must be arrays")
    normalized = {
        "schema_version": RANDOMIZATION_SCHEMA_VERSION,
        "max_attempts": max_attempts,
        "preserve_relations": preserve_relations,
        "robot": {
            "xy_radius_m": _non_negative(robot.get("xy_radius_m", 0.0), "robot.xy_radius_m"),
            "yaw_range_rad": _non_negative(robot.get("yaw_range_rad", 0.0), "robot.yaw_range_rad"),
        },
        "objects": list(objects),
        "physics": list(physics),
    }
    # Validate selectors/ranges here so a malformed suite fails before it can
    # produce a partially valid batch.
    _parse_object_rules(normalized["objects"], "objects")
    _parse_object_rules(normalized["physics"], "physics")
    return normalized


def _validate_robot_spec(robot: Mapping[str, Any]) -> None:
    unknown = set(robot) - {"xy_radius_m", "yaw_range_rad"}
    if unknown:
        raise ValueError(f"randomization robot contains unknown fields: {sorted(unknown)}")
    _non_negative(robot.get("xy_radius_m", 0.0), "robot.xy_radius_m")
    _non_negative(robot.get("yaw_range_rad", 0.0), "robot.yaw_range_rad")


def _parse_object_rules(value: Sequence[Mapping[str, Any]], what: str) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"randomization {what}[{index}] must be a mapping")
        allowed = {
            "match",
            "xy_radius_m",
            "yaw_range_rad",
            "z_jitter_m",
            "mass_scale",
            "friction_scale",
            "contact_offset_scale",
            "planning_margin_scale",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"randomization {what}[{index}] contains unknown fields: {sorted(unknown)}"
            )
        match = raw.get("match", {}) or {}
        if not isinstance(match, Mapping):
            raise TypeError(f"randomization {what}[{index}].match must be a mapping")
        _validate_match(match, f"{what}[{index}].match")
        normalized = {"match": dict(match)}
        for key in ("xy_radius_m", "yaw_range_rad", "z_jitter_m"):
            normalized[key] = _non_negative(raw.get(key, 0.0), f"{what}[{index}].{key}")
        for key in (
            "mass_scale",
            "friction_scale",
            "contact_offset_scale",
            "planning_margin_scale",
        ):
            if key in raw:
                normalized[key] = _ordered_range(raw[key], f"{what}[{index}].{key}")
        if not any(normalized[key] > 0.0 for key in ("xy_radius_m", "yaw_range_rad", "z_jitter_m")) and not any(
            key in normalized
            for key in (
                "mass_scale",
                "friction_scale",
                "contact_offset_scale",
                "planning_margin_scale",
            )
        ):
            raise ValueError(f"randomization {what}[{index}] has no perturbation")
        rules.append(normalized)
    return rules


def _validate_match(match: Mapping[str, Any], what: str) -> None:
    allowed = {"role", "capability", "capabilities", "names", "semantic_class", "aliases"}
    unknown = set(match) - allowed
    if unknown:
        raise ValueError(f"{what} contains unknown fields: {sorted(unknown)}")
    if "role" in match and match["role"] not in {
        "object", "target", "support", "obstacle", "collision", "all"
    }:
        raise ValueError(f"{what}.role is unsupported: {match['role']!r}")
    for key in ("capability", "semantic_class"):
        if key in match and (not isinstance(match[key], str) or not match[key].strip()):
            raise ValueError(f"{what}.{key} must be a non-empty string")
    if "capability" in match:
        try:
            ObjectCapability(match["capability"])
        except ValueError as exc:
            raise ValueError(f"{what}.capability is unsupported: {match['capability']!r}") from exc
    for key in ("capabilities", "names", "aliases"):
        if key in match:
            values = match[key]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(item, str) or not item.strip() for item in values)
            ):
                raise ValueError(f"{what}.{key} must be a non-empty string array")
            if key == "capabilities":
                for value in values:
                    try:
                        ObjectCapability(value)
                    except ValueError as exc:
                        raise ValueError(f"{what}.capabilities contains unsupported capability: {value!r}") from exc


def _sample_local_transform(
    model: ObjectModel,
    rules: Sequence[Mapping[str, Any]],
    rng: random.Random,
) -> dict[str, float]:
    transform = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw": 0.0}
    for rule in rules:
        if not _matches(model, rule["match"]):
            continue
        radius = float(rule["xy_radius_m"])
        if radius:
            sample_radius = radius * math.sqrt(rng.random())
            angle = rng.uniform(-math.pi, math.pi)
            transform["dx"] += sample_radius * math.cos(angle)
            transform["dy"] += sample_radius * math.sin(angle)
        transform["yaw"] += rng.uniform(-float(rule["yaw_range_rad"]), float(rule["yaw_range_rad"]))
        transform["dz"] += rng.uniform(-float(rule["z_jitter_m"]), float(rule["z_jitter_m"]))
    return transform


def _sample_robot_change(
    robot_spec: Mapping[str, Any] | None,
    rng: random.Random,
) -> dict[str, float]:
    robot_spec = robot_spec or {}
    radius = float(robot_spec.get("xy_radius_m", 0.0))
    if radius:
        sample_radius = radius * math.sqrt(rng.random())
        angle = rng.uniform(-math.pi, math.pi)
        dx = sample_radius * math.cos(angle)
        dy = sample_radius * math.sin(angle)
    else:
        dx = dy = 0.0
    return {
        "dx": dx,
        "dy": dy,
        "yaw": rng.uniform(
            -float(robot_spec.get("yaw_range_rad", 0.0)),
            float(robot_spec.get("yaw_range_rad", 0.0)),
        ),
    }


def _resolve_transform(
    name: str,
    scene: SceneModel,
    relations: Sequence[tuple[str, str]],
    local: Mapping[str, Mapping[str, float]],
    resolved: dict[str, dict[str, float]],
    visiting: set[str],
) -> dict[str, float]:
    if name in resolved:
        return resolved[name]
    if name in visiting:
        # Relation inference is acyclic by construction, but fail closed if a
        # future capability extension accidentally introduces a cycle.
        raise SceneRandomizationError(f"support relation cycle includes {name!r}")
    visiting.add(name)
    authored = scene.object(name)
    transform = dict(local[name])
    parent_name = next((parent for parent, child in relations if child == name), None)
    if parent_name is not None:
        parent = scene.object(parent_name)
        parent_transform = _resolve_transform(
            parent_name, scene, relations, local, resolved, visiting
        )
        relative = (
            authored.pos[0] - parent.pos[0],
            authored.pos[1] - parent.pos[1],
            authored.pos[2] - parent.pos[2],
        )
        rotated_relative = _rotate_xy(relative[0], relative[1], parent_transform["yaw"])
        transform["dx"] += parent_transform["dx"] + rotated_relative[0] - relative[0]
        transform["dy"] += parent_transform["dy"] + rotated_relative[1] - relative[1]
        transform["dz"] += parent_transform["dz"]
        transform["yaw"] += parent_transform["yaw"]
    visiting.remove(name)
    resolved[name] = transform
    return transform


def _apply_physics_rules(
    raw_objects: Sequence[Mapping[str, Any]],
    scene: SceneModel,
    rules: Sequence[Mapping[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    by_name = {obj.name: obj for obj in scene.objects}
    changes: dict[str, Any] = {}
    for raw in raw_objects:
        name = raw.get("name") if isinstance(raw, Mapping) else None
        if not isinstance(name, str) or name not in by_name:
            continue
        model = by_name[name]
        for rule in rules:
            if not _matches(model, rule["match"]):
                continue
            for key, scene_key in (
                ("mass_scale", "mass"),
                ("friction_scale", "static_friction"),
                ("friction_scale", "dynamic_friction"),
                ("contact_offset_scale", "contact_offset"),
                ("planning_margin_scale", "planning_margin"),
            ):
                if key not in rule or scene_key not in raw or raw.get(scene_key) is None:
                    continue
                scale = rng.uniform(*rule[key])
                old = float(raw[scene_key])
                new = old * scale
                if scene_key == "planning_margin":
                    new = max(0.0, new)
                if scene_key in {"mass", "contact_offset"}:
                    new = max(1e-8, new)
                if scene_key in {"static_friction", "dynamic_friction"}:
                    new = max(0.0, new)
                raw[scene_key] = _round_float(new)
                changes.setdefault(name, {})[scene_key] = {
                    "authored": old,
                    "sampled": new,
                    "scale": scale,
                }
    return changes


def _matches(model: ObjectModel, match: Mapping[str, Any]) -> bool:
    if not match:
        return True
    if "names" in match and model.name not in set(match["names"]):
        return False
    if "aliases" in match and not (set(model.aliases) & set(match["aliases"])):
        return False
    if "semantic_class" in match and model.semantic_class != match["semantic_class"]:
        return False
    if "capability" in match:
        try:
            capability = ObjectCapability(match["capability"])
        except ValueError:
            return False
        if capability not in model.capabilities:
            return False
    if "capabilities" in match and not set(ObjectCapability(item) for item in match["capabilities"]).issubset(set(model.capabilities)):
        return False
    role = match.get("role")
    if role is None or role == "all":
        return True
    capabilities = set(model.capabilities)
    if role == "object":
        return ObjectCapability.MOVABLE in capabilities
    if role == "target":
        return ObjectCapability.CONTAINS_OBJECTS in capabilities
    if role == "support":
        return ObjectCapability.SUPPORTS_OBJECTS in capabilities
    if role == "collision":
        return model.physics.collision_enabled
    if role == "obstacle":
        return model.physics.collision_enabled and not bool(
            capabilities
            & {
                ObjectCapability.MOVABLE,
                ObjectCapability.SUPPORTS_OBJECTS,
                ObjectCapability.CONTAINS_OBJECTS,
            }
        )
    return False


def _infer_support_relations(scene: SceneModel) -> tuple[tuple[str, str], ...]:
    parents = [
        obj
        for obj in scene.objects
        if ObjectCapability.SUPPORTS_OBJECTS in obj.capabilities
    ]
    relations: list[tuple[str, str]] = []
    for child in scene.objects:
        if child in parents:
            continue
        candidates: list[tuple[float, str]] = []
        child_hx, child_hy = object_xy_half_extents_m(child)
        child_bottom = child.pos[2] - object_vertical_extent_m(child) / 2.0
        for parent in parents:
            parent_hx, parent_hy = object_xy_half_extents_m(parent)
            if abs(child.pos[0] - parent.pos[0]) + child_hx > parent_hx + _FEASIBILITY_TOLERANCE_M:
                continue
            if abs(child.pos[1] - parent.pos[1]) + child_hy > parent_hy + _FEASIBILITY_TOLERANCE_M:
                continue
            parent_top = parent.pos[2] + object_vertical_extent_m(parent) / 2.0
            height_error = abs(child_bottom - parent_top)
            if height_error <= _SUPPORT_HEIGHT_TOLERANCE_M:
                area = parent_hx * parent_hy
                candidates.append((area, parent.name))
        if candidates:
            candidates.sort()
            relations.append((candidates[0][1], child.name))
    return tuple(relations)


def _objects_overlap(
    first: ObjectModel,
    second: ObjectModel,
    extents: Mapping[str, tuple[float, float]],
    vertical: Mapping[str, float],
) -> bool:
    first_hx, first_hy = extents[first.name]
    second_hx, second_hy = extents[second.name]
    xy_overlap = (
        abs(first.pos[0] - second.pos[0]) < first_hx + second_hx - _FEASIBILITY_TOLERANCE_M
        and abs(first.pos[1] - second.pos[1]) < first_hy + second_hy - _FEASIBILITY_TOLERANCE_M
    )
    if not xy_overlap:
        return False
    first_bottom = first.pos[2] - vertical[first.name] / 2.0
    first_top = first.pos[2] + vertical[first.name] / 2.0
    second_bottom = second.pos[2] - vertical[second.name] / 2.0
    second_top = second.pos[2] + vertical[second.name] / 2.0
    return min(first_top, second_top) - max(first_bottom, second_bottom) > _FEASIBILITY_TOLERANCE_M


def _is_structural_static_overlap(
    first: ObjectModel,
    second: ObjectModel,
    extents: Mapping[str, tuple[float, float]],
) -> bool:
    """Allow small corner joints between authored static wall primitives.

    Scene authors commonly close a perimeter with perpendicular kinematic
    cuboids whose volumes intersect at a corner.  Treating that construction
    detail as a spawn failure would reject otherwise valid arenas.  The area
    cap prevents a randomization that puts two large obstacles through each
    other from passing this exception.
    """
    if not first.physics.kinematic or not second.physics.kinematic:
        return False
    if ObjectCapability.MOVABLE in first.capabilities or ObjectCapability.MOVABLE in second.capabilities:
        return False
    first_hx, first_hy = extents[first.name]
    second_hx, second_hy = extents[second.name]
    overlap_x = max(0.0, min(first.pos[0] + first_hx, second.pos[0] + second_hx) - max(first.pos[0] - first_hx, second.pos[0] - second_hx))
    overlap_y = max(0.0, min(first.pos[1] + first_hy, second.pos[1] + second_hy) - max(first.pos[1] - first_hy, second.pos[1] - second_hy))
    intersection_area = overlap_x * overlap_y
    smaller_area = min(4.0 * first_hx * first_hy, 4.0 * second_hx * second_hy)
    return smaller_area > 0.0 and intersection_area <= 0.15 * smaller_area


def _parse_scene(data: Mapping[str, Any], label: str) -> SceneModel:
    try:
        return SceneModel.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc


def _ordered_range(value: object, what: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{what} must be a two-element numeric array")
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be a two-element numeric array") from exc
    if len(values) != 2 or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{what} must contain two finite values")
    if values[0] <= 0.0 or values[1] <= 0.0 or values[0] > values[1]:
        raise ValueError(f"{what} must contain positive ordered scale bounds")
    return values


def _non_negative(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{what} must be finite and non-negative")
    return result


def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return cosine * x - sine * y, sine * x + cosine * y


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2.0
    return math.cos(half), 0.0, 0.0, math.sin(half)


def _quat_multiply(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = (float(value) for value in first)
    w2, x2, y2, z2 = (float(value) for value in second)
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RANDOMIZATION_SCHEMA_VERSION",
    "SceneFeasibilityReport",
    "SceneRandomizationError",
    "check_scene_feasibility",
    "default_randomization_spec",
    "randomize_scene_data",
]
