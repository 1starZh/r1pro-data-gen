"""Compile semantic navigation intents into safe executable base poses.

An external planner should express *why* the base is moving (for example,
``target_ref=scene://pick_cylinder, purpose=pregrasp``), not guess the final
world pose.  This module is the deterministic boundary between that semantic
intent and ``base_navigate_to``'s concrete ``(x, y, yaw)`` target.

The resolver never changes the task goal.  It only selects a collision-free
approach candidate that preserves the requested purpose.  A literal ``target``
from an older plan remains supported as a preferred/executable pose for
backward compatibility, but new plans should use ``target_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any

from r1pro_data_gen.domain import SceneModel, object_xy_half_extents_m
from r1pro_data_gen.robot.chassis import default_footprint_radius_m

from .contract import (
    NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M,
    NAVIGATION_INFLATION_CLEARANCE_M,
)
from ..context.facts import scene_to_facts


_PURPOSES = frozenset({"navigation", "pregrasp", "dropoff", "staging", "observe", "park"})
_SIDES = frozenset({"west", "east", "south", "north"})


class NavigationTargetError(ValueError):
    """Raised when a semantic navigation target has no safe executable pose."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class NavigationTargetResolution:
    """Immutable record of semantic-to-executable navigation resolution."""

    target_ref: str
    purpose: str
    resolved_pose: tuple[float, float, float]
    source: str
    candidate_count: int
    alternatives: tuple[tuple[float, float, float], ...] = ()
    approach_side: str | None = None
    clearance_m: float | None = None

    def to_details(self) -> dict[str, Any]:
        """Return bounded JSON-compatible evidence for execution/feedback."""
        result: dict[str, Any] = {
            "target_ref": self.target_ref,
            "purpose": self.purpose,
            "resolved_target": [round(float(v), 4) for v in self.resolved_pose],
            "resolution_source": self.source,
            "candidate_count": self.candidate_count,
            "alternative_targets": [
                [round(float(v), 4) for v in pose] for pose in self.alternatives
            ],
        }
        if self.approach_side is not None:
            result["approach_side"] = self.approach_side
        if self.clearance_m is not None and math.isfinite(float(self.clearance_m)):
            result["clearance_m"] = round(float(self.clearance_m), 4)
        return result


def resolve_navigation_target(
    scene: SceneModel,
    target_ref: str,
    *,
    purpose: str = "navigation",
    preferred_pose: Sequence[float] | None = None,
    approach_side: str | None = None,
    kinematics: Any = None,
) -> NavigationTargetResolution:
    """Resolve a semantic scene reference to a safe navigation pose.

    Candidate poses come from geometry-derived scene facts.  For a movable
    object on a support surface, the support's approach candidates are used;
    this keeps the robot outside the table footprint instead of placing its
    base on the tabletop.  If no support is identified, candidates are
    generated around the referenced object itself.

    ``preferred_pose`` is intentionally soft: it affects ranking only.  A
    caller that needs an exact pose must use the legacy literal ``target`` and
    accept a normal navigation failure if that pose is occupied.
    """
    if not isinstance(scene, SceneModel):
        raise NavigationTargetError("INVALID_SCENE", "navigation target resolution needs a SceneModel")
    name = _parse_target_ref(target_ref)
    if not isinstance(purpose, str) or purpose not in _PURPOSES:
        raise NavigationTargetError(
            "INVALID_PURPOSE",
            f"unsupported navigation purpose: {purpose!r}",
            details={"purpose": purpose},
        )
    if approach_side is not None and approach_side not in _SIDES:
        raise NavigationTargetError(
            "INVALID_APPROACH_SIDE",
            f"unsupported approach side: {approach_side!r}",
            details={"approach_side": approach_side},
        )
    target = scene.object(name)
    preferred = _validate_pose(preferred_pose, "preferred_pose") if preferred_pose is not None else None

    # The registry passes the trusted arm kinematics at runtime so candidate
    # poses can carry the same reachability annotations used during planning.
    # A mapping is accepted for the two-arm registry; navigation uses the
    # primary left-arm model for its conservative pre-grasp probe.
    if isinstance(kinematics, Mapping):
        kinematics = kinematics.get("left") or kinematics.get("right")
    facts = scene_to_facts(scene, kinematics=kinematics)
    navigation = facts.get("navigation", {})
    authored_candidates = navigation.get("approach_candidates", [])
    support_names = _support_names(scene, target)
    candidates = _select_fact_candidates(
        authored_candidates,
        target_name=name,
        support_names=support_names,
        approach_side=approach_side,
    )

    # If kinematics were used while exporting facts, reject *support/target*
    # candidates known to be unreachable.  Facts without IK annotations remain
    # valid geometry-only candidates.  Unreachable support poses fail closed
    # because driving onto the support itself would be unsafe.  A movable
    # object with no support (for example a cylinder on the ground) must fall
    # through to object-geometry candidates instead of inheriting fence/wall
    # annotations that mention it only as the nearest dynamic body.
    annotated = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and any(
            isinstance(entry, Mapping) and entry.get("name") == name
            for entry in item.get("ik_reachability", ())
        )
    ]
    if annotated:
        reachable = [
            item for item in annotated
            if any(
                isinstance(entry, Mapping)
                and entry.get("name") == name
                and entry.get("reachable") is True
                for entry in item.get("ik_reachability", ())
            )
        ]
        if reachable:
            candidates = reachable
        elif support_names:
            raise NavigationTargetError(
                "NO_REACHABLE_APPROACH",
                f"no candidate approach pose can reach {name!r}",
                details={
                    "target_ref": target_ref,
                    "purpose": purpose,
                    "candidate_count": len(candidates),
                    "approach_sides": sorted({item.get("side") for item in candidates if item.get("side")}),
                },
            )
        else:
            candidates = []

    if not candidates:
        candidates = _geometry_candidates(
            scene,
            target,
            purpose=purpose,
            approach_side=approach_side,
        )
    if not candidates:
        raise NavigationTargetError(
            "NO_SAFE_APPROACH",
            f"no collision-free approach candidate exists for {name!r}",
            details={"target_ref": target_ref, "purpose": purpose},
        )

    ordered = sorted(
        candidates,
        key=lambda item: _candidate_distance(item, preferred, target),
    )
    selected = ordered[0]
    pose = _validate_pose(selected.get("pose"), "candidate.pose")
    alternatives = tuple(
        _validate_pose(item.get("pose"), "candidate.pose")
        for item in ordered[1:4]
        if isinstance(item, Mapping) and item.get("pose") is not None
    )
    return NavigationTargetResolution(
        target_ref=f"scene://{name}",
        purpose=purpose,
        resolved_pose=tuple(pose),
        source="scene_approach_candidate" if selected in authored_candidates else "geometry_candidate",
        candidate_count=len(ordered),
        alternatives=alternatives,
        approach_side=str(selected.get("side")) if selected.get("side") is not None else None,
        clearance_m=_candidate_clearance(selected, scene),
    )


def _parse_target_ref(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NavigationTargetError("INVALID_TARGET_REF", "target_ref must be a non-empty scene reference")
    ref = value.strip()
    if not ref.startswith("scene://"):
        raise NavigationTargetError(
            "INVALID_TARGET_REF",
            "target_ref must use the scene://<object> form",
            details={"target_ref": ref},
        )
    name = ref[len("scene://") :]
    if not name or "/" in name or "\\" in name:
        raise NavigationTargetError("INVALID_TARGET_REF", f"invalid scene target reference: {ref!r}")
    return name


def _validate_pose(value: Sequence[float] | None, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise NavigationTargetError("INVALID_TARGET_POSE", f"{field} must be a finite (x, y, yaw) pose")
    pose = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in pose):
        raise NavigationTargetError("INVALID_TARGET_POSE", f"{field} must contain finite numbers")
    return pose


def _support_names(scene: SceneModel, target: Any) -> set[str]:
    """Return static supports whose top surface contains a movable target."""
    if getattr(target.physics, "kinematic", False):
        return set()
    result: set[str] = set()
    target_x, target_y, target_z = (float(value) for value in target.pos)
    for obj in scene.objects:
        if obj.name == target.name or not obj.physics.collision_enabled or not obj.physics.kinematic:
            continue
        half_x, half_y = object_xy_half_extents_m(obj)
        xy_inside = (
            abs(target_x - float(obj.pos[0])) <= half_x + 0.03
            and abs(target_y - float(obj.pos[1])) <= half_y + 0.03
        )
        vertical_gap = target_z - float(obj.top_z)
        if xy_inside and -0.08 <= vertical_gap <= 0.20:
            result.add(obj.name)
    return result


def _select_fact_candidates(
    candidates: Sequence[Any],
    *,
    target_name: str,
    support_names: set[str],
    approach_side: str | None,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping) or not isinstance(item.get("pose"), Sequence):
            continue
        obstacle_name = item.get("obstacle_name")
        matches_target = obstacle_name == target_name
        matches_support = bool(support_names and obstacle_name in support_names)
        # Authored cuboid candidates belong to the named obstacle or to a
        # detected support (table under a cylinder).  IK annotations on
        # unrelated walls/fences mention the movable object only because it is
        # the nearest dynamic body; those poses are not approach stances.
        if not (matches_target or matches_support):
            continue
        if approach_side is not None and item.get("side") != approach_side:
            continue
        selected.append(item)
    return selected


def _geometry_candidates(
    scene: SceneModel,
    target: Any,
    *,
    purpose: str,
    approach_side: str | None,
) -> list[dict[str, Any]]:
    """Generate safe poses from scene geometry and the robot footprint.

    The resolver does not branch on object height or on a benchmark layout.
    The semantic purpose is retained as evidence, while executable poses are
    derived from the same primitive extents and collision margin for every
    object type.
    """
    nav_footprint = float(scene.robot.navigation_footprint_radius_m or default_footprint_radius_m())
    clearance = NAVIGATION_INFLATION_CLEARANCE_M
    half_x, half_y = object_xy_half_extents_m(target)
    extra = NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M
    anchors = {
        "west": (target.pos[0] - half_x - nav_footprint - clearance - extra, target.pos[1], 0.0),
        "east": (target.pos[0] + half_x + nav_footprint + clearance + extra, target.pos[1], math.pi),
        "south": (target.pos[0], target.pos[1] - half_y - nav_footprint - clearance - extra, math.pi / 2.0),
        "north": (target.pos[0], target.pos[1] + half_y + nav_footprint + clearance + extra, -math.pi / 2.0),
    }
    result: list[dict[str, Any]] = []
    for side, pose in anchors.items():
        if approach_side is not None and side != approach_side:
            continue
        candidate = {
            "obstacle_name": target.name,
            "side": side,
            "pose": list(pose),
            "footprint_radius_m": nav_footprint,
            "purpose": purpose,
        }
        if _pose_is_free(scene, pose, target_name=target.name, footprint=nav_footprint):
            result.append(candidate)
    return result


def _pose_is_free(scene: SceneModel, pose: Sequence[float], *, target_name: str, footprint: float) -> bool:
    x, y = float(pose[0]), float(pose[1])
    inflate = footprint + NAVIGATION_INFLATION_CLEARANCE_M
    for obj in scene.objects:
        if not obj.physics.collision_enabled:
            continue
        half_x, half_y = object_xy_half_extents_m(obj)
        if abs(x - float(obj.pos[0])) <= half_x + inflate and abs(y - float(obj.pos[1])) <= half_y + inflate:
            return False
    return True


def _candidate_distance(item: Mapping[str, Any], preferred: Sequence[float] | None, target: Any) -> float:
    pose = _validate_pose(item.get("pose"), "candidate.pose")
    if preferred is not None:
        return math.hypot(pose[0] - preferred[0], pose[1] - preferred[1]) + 0.05 * abs(
            _wrap_pi(pose[2] - preferred[2])
        )
    return math.hypot(pose[0] - float(target.pos[0]), pose[1] - float(target.pos[1]))


def _candidate_clearance(item: Mapping[str, Any], scene: SceneModel) -> float | None:
    value = item.get("clearance_m")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    footprint = float(item.get("footprint_radius_m", scene.robot.navigation_footprint_radius_m or default_footprint_radius_m()))
    return footprint + NAVIGATION_INFLATION_CLEARANCE_M


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "NavigationTargetError",
    "NavigationTargetResolution",
    "resolve_navigation_target",
]
