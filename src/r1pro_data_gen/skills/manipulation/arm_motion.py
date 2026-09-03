"""Arm manipulation skills: semantic motion, trajectory following, EE rotation, push.

These extend the arm capability beyond single pose targets:
- ``arm_move_to``: solve a final EE goal, plan around the current scene, and execute it;
- ``arm_trajectory_follow``: execute a certified joint trajectory;
  (wiping, scanning, pouring arcs);
- ``arm_move_directional``: advance the end-effector along a direction until contact
  or a distance (pressing, inserting, pulling);
- ``arm_rotate_ee``: rotate the end-effector about an axis while holding position
  (valves, tilting to pour);

All positions are base-frame. ``arm_move_to`` is the public semantic action; the
trajectory and joint skills are kept as lower-level backends for replay and
diagnostics. Every arm skill selects the same reusable implementation with a
``side=left|right`` parameter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .arm import ARM_JOINTS_BY_SIDE, ArmSegmentExecutor
from ..core.base import ParamSpec, SkillResult, stabilize_base
from ..core.sides import for_side, require_side

_FINAL_ERROR_TOL = 0.08
_TRACKING_RECOVERY_FACTOR = 12


class _HeldContextLost(RuntimeError):
    pass


class _ObjectMovedBeforeGrasp(RuntimeError):
    """Abort an open-gripper alignment when its target starts moving.

    An open gripper is allowed to approach a movable object, but it is not
    allowed to turn an alignment retry into an uncontrolled push.  Keeping
    this as an internal control-flow signal lets the low-level trajectory
    executor stop on the first violating physics step while the semantic
    skill still returns a structured, replannable ``SkillResult``.
    """

    def __init__(self, details: dict[str, Any]):
        self.details = dict(details)
        super().__init__(str(self.details.get("reason", "object moved before grasp")))


class _AlignmentCollisionCheckUnavailable(RuntimeError):
    """Abort alignment when a requested live collision certificate is absent."""

    def __init__(self, details: dict[str, Any]):
        self.details = dict(details)
        super().__init__(str(self.details.get("reason", "alignment collision check unavailable")))


class _AlignmentCollisionDetected(RuntimeError):
    """Abort a live alignment after a non-finger link enters the target."""

    def __init__(self, details: dict[str, Any]):
        self.details = dict(details)
        super().__init__(str(self.details.get("reason", "alignment collision detected")))


def _wxyz_rotation(quaternion: Any) -> np.ndarray:
    """Convert a finite world ``wxyz`` quaternion to a rotation matrix."""
    quat = np.asarray(quaternion, dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must be a finite wxyz vector")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    quat = quat / norm
    return Rotation.from_quat(
        [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]
    ).as_matrix()


def _finger_collision_box_specs(side: str) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Return robot-profiled collision boxes in the two finger link frames."""
    from r1pro_data_gen.robot.robot_config import R1PRO_GRIPPER_FINGER_COLLISION_BOXES_LOCAL

    specs = []
    for suffix, profile_name in (
        ("finger_link1", "finger_link1"),
        ("finger_link2", "finger_link2"),
    ):
        profile = R1PRO_GRIPPER_FINGER_COLLISION_BOXES_LOCAL[profile_name]
        center = np.asarray(profile["center"], dtype=float)
        half_extents = np.asarray(profile["half_extents"], dtype=float)
        if (
            center.shape != (3,)
            or half_extents.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(half_extents))
            or np.any(half_extents <= 0.0)
        ):
            raise ValueError(f"invalid collision profile for {suffix}")
        specs.append((f"{side}_gripper_{suffix}", center, half_extents))
    return tuple(specs)


def _json_safe_path_details(value: Any) -> Any:
    """Convert a local collision diagnostic into result-contract values."""
    if isinstance(value, dict):
        return {str(key): _json_safe_path_details(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe_path_details(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_path_details(value.tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe_path_details(item())
    return value


def _finger_collision_mesh(body_name: str) -> Any | None:
    """Load the supplied R1Pro finger collision mesh once per process.

    The profiled boxes remain a safe fallback for lightweight installations,
    but they are deliberately conservative envelopes.  During intentional
    object acquisition that conservatism can stop the arm several millimetres
    before the actual finger skin reaches the object.  The repository ships
    the same link meshes used by the R1Pro asset, so use them for the terminal
    finger/object certificate whenever they are available.
    """
    from r1pro_data_gen.methods.collision import collision_mesh_for_body

    return collision_mesh_for_body(body_name)


def _finger_box_collision(
    object_model: Any,
    finger_poses: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    margin: float,
) -> tuple[bool, dict[str, Any]]:
    """Check measured/predicted finger mesh envelopes against one object.

    The supplied finger collision meshes are used when available.  Their
    profiled boxes remain a conservative fallback for lightweight installs;
    using the boxes as the primary terminal test would report contact before
    the actual R1Pro finger geometry reaches a small object.  The arm path
    checker continues to use its existing link-specific spheres for the rest
    of the robot.  A collision at a geometrically valid two-sided window is
    reported to the caller as the terminal acquisition contact, not as a
    successful attachment.
    """
    from r1pro_data_gen.methods.collision import object_obstacle
    import hppfcl

    obstacle = object_obstacle(object_model, float(max(0.0, margin)))
    request = hppfcl.CollisionRequest()
    result = hppfcl.CollisionResult()
    for body_name, position, rotation in finger_poses:
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if (
            position.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(rotation))
        ):
            return False, {
                "checked": False,
                "reason": "finger collision pose is invalid",
                "body_name": body_name,
            }
        mesh = _finger_collision_mesh(body_name)
        shape_source = "asset_mesh"
        if mesh is None:
            profile = next(
                item for item in _finger_collision_box_specs(body_name.split("_", 1)[0])
                if item[0] == body_name
            )
            _, center_local, half_extents = profile
            shape = hppfcl.Box(*(2.0 * half_extents))
            shape_source = "profiled_box_fallback"
            box_center_world = position + rotation @ center_local
        else:
            shape = mesh
            # Mesh vertices are authored in the link frame; unlike the
            # profiled box, no additional local center offset is required.
            box_center_world = position
        transform = hppfcl.Transform3f(rotation, box_center_world)
        result.clear()
        if hppfcl.collide(
            obstacle.shape,
            obstacle.transform,
            shape,
            transform,
            request,
            result,
        ):
            return False, {
                "checked": True,
                "reason": "finger collision envelope entered target",
                "body_name": body_name,
                "body_position_world": position.round(6).tolist(),
                "box_center_world": box_center_world.round(6).tolist(),
                "object_position_world": [float(value) for value in object_model.pos],
                "inflation_margin_m": float(margin),
                "shape_source": shape_source,
            }
    return True, {
        "checked": True,
        "checked_finger_count": len(finger_poses),
        "inflation_margin_m": float(margin),
    }


def _alignment_world_model_transform(
    base_pose: Any,
    model_to_world_rotation: np.ndarray | None,
    model_to_world_translation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the model-to-world transform used by alignment certificates."""
    if model_to_world_rotation is not None and model_to_world_translation is not None:
        rotation = np.asarray(model_to_world_rotation, dtype=float)
        translation = np.asarray(model_to_world_translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("model-to-world transform has invalid shape")
        return rotation, translation
    x, y, yaw = (float(value) for value in tuple(base_pose)[:3])
    return Rotation.from_euler("z", yaw).as_matrix(), np.asarray([x, y, 0.0], dtype=float)


def _alignment_finger_opening_offsets(
    kin: Any,
    adapter: Any,
    side: str,
    start_q: np.ndarray,
    base_pose: Any,
    model_to_world_rotation: np.ndarray | None,
    model_to_world_translation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Measure live opening displacement in each finger link's local frame."""
    if not hasattr(kin, "finger_frame_fk") or not hasattr(adapter, "body_position"):
        return None
    rotation, translation = _alignment_world_model_transform(
        base_pose, model_to_world_rotation, model_to_world_translation
    )
    poses = kin.finger_frame_fk(np.asarray(start_q, dtype=float))
    result = []
    for index, (position_model, rotation_model) in enumerate(
        ((poses[0], poses[1]), (poses[2], poses[3])), start=1
    ):
        body_name = f"{side}_gripper_finger_link{index}"
        measured = np.asarray(adapter.body_position(body_name), dtype=float)
        world_rotation = rotation @ np.asarray(rotation_model, dtype=float)
        zero_world = rotation @ np.asarray(position_model, dtype=float) + translation
        if (
            measured.shape != (3,)
            or not np.all(np.isfinite(measured))
            or not np.all(np.isfinite(world_rotation))
            or not np.all(np.isfinite(zero_world))
        ):
            return None
        local_offset = world_rotation.T @ (measured - zero_world)
        if not np.all(np.isfinite(local_offset)) or float(np.linalg.norm(local_offset)) > 0.10:
            return None
        result.append(local_offset)
    return result[0], result[1]


def _predicted_alignment_finger_poses(
    kin: Any,
    adapter: Any,
    side: str,
    q_target: np.ndarray,
    opening_offsets: tuple[np.ndarray, np.ndarray],
    base_pose: Any,
    model_to_world_rotation: np.ndarray | None,
    model_to_world_translation: np.ndarray | None,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Predict physical finger link poses for an arm-only target sample."""
    del adapter
    rotation, translation = _alignment_world_model_transform(
        base_pose, model_to_world_rotation, model_to_world_translation
    )
    poses = kin.finger_frame_fk(np.asarray(q_target, dtype=float))
    result = []
    for index, (position_model, rotation_model, local_offset) in enumerate(
        ((poses[0], poses[1], opening_offsets[0]), (poses[2], poses[3], opening_offsets[1])),
        start=1,
    ):
        model_rotation = np.asarray(rotation_model, dtype=float)
        world_rotation = rotation @ model_rotation
        position = rotation @ np.asarray(position_model, dtype=float) + translation
        position = position + world_rotation @ np.asarray(local_offset, dtype=float)
        result.append(
            (
                f"{side}_gripper_finger_link{index}",
                np.asarray(position, dtype=float),
                np.asarray(world_rotation, dtype=float),
            )
        )
    return tuple(result)


def _alignment_support_top_z(scene: Any, object_model: Any) -> float | None:
    """Infer the plane currently below an object for finger clearance checks.

    The source plane is a geometric fact of the live scene, not a task pose.
    Ground is considered first; a nearby cuboid whose top is coincident with
    the object's bottom is used for elevated supports.  A destination table
    that is far below/above the object is therefore not accidentally treated
    as the source support.
    """
    if scene is None or object_model is None:
        return None
    try:
        from r1pro_data_gen.domain import object_vertical_extent_m, object_xy_radius_m

        object_position = np.asarray(object_model.pos, dtype=float)
        if object_position.shape != (3,) or not np.all(np.isfinite(object_position)):
            return None
        object_bottom = float(object_position[2]) - 0.5 * float(
            object_vertical_extent_m(object_model)
        )
        support_top: float | None = (
            0.0 if bool(getattr(getattr(scene, "world", None), "ground", False)) else None
        )
        radius = float(object_xy_radius_m(object_model))
        candidates: list[tuple[float, float, float]] = []
        for candidate in getattr(scene, "objects", ()):
            if getattr(candidate, "name", None) == getattr(object_model, "name", None):
                continue
            size = getattr(candidate, "size", None)
            position = getattr(candidate, "pos", None)
            if size is None or position is None:
                continue
            size = np.asarray(size, dtype=float)
            position = np.asarray(position, dtype=float)
            if size.shape != (3,) or position.shape != (3,):
                continue
            if not np.all(np.isfinite(size)) or not np.all(np.isfinite(position)):
                continue
            if float(np.linalg.norm(object_position[:2] - position[:2])) > float(
                np.linalg.norm(size[:2] / 2.0) + radius + 0.02
            ):
                continue
            top = float(position[2] + 0.5 * size[2])
            gap = abs(top - object_bottom)
            if gap <= 0.03:
                candidates.append((gap, -float(size[0] * size[1]), top))
        if candidates:
            candidates.sort()
            support_top = candidates[0][2]
        return support_top
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _alignment_finger_bottom_z(
    side: str,
    finger_poses: tuple[tuple[str, np.ndarray, np.ndarray], ...],
) -> float | None:
    """Return the lowest point of the profiled finger boxes in world Z."""
    try:
        specs = {name: (center, half) for name, center, half in _finger_collision_box_specs(side)}
        bottoms: list[float] = []
        for name, position, rotation in finger_poses:
            if name not in specs:
                continue
            center, half_extents = specs[name]
            position = np.asarray(position, dtype=float)
            rotation = np.asarray(rotation, dtype=float)
            if (
                position.shape != (3,)
                or rotation.shape != (3, 3)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(rotation))
            ):
                return None
            box_center = position + rotation @ center
            world_half_z = float(np.sum(np.abs(rotation[2, :]) * half_extents))
            bottoms.append(float(box_center[2] - world_half_z))
        return min(bottoms) if bottoms else None
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _finger_window_geometry_ready(
    side: str,
    finger_poses: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    object_model: Any,
    *,
    surface_tolerance_m: float,
    vertical_margin_m: float,
) -> tuple[bool, dict[str, Any]]:
    """Check a two-sided jaw window using the physical finger boxes.

    Finger-link origins are not the contact points on the R1Pro gripper: the
    supplied collision boxes extend below and around those origins.  For a
    floor object, a valid pinch can therefore have the object below the line
    joining the origins while still overlapping both physical finger boxes.
    Project the window onto the active support plane and require vertical
    overlap with both boxes.  This keeps endpoint/one-finger contacts out of
    the terminal condition without treating the origin-to-object 3-D distance
    as the contact distance.
    """
    try:
        if len(finger_poses) != 2:
            return False, {"checked": False, "reason": "two finger poses are required"}
        p1 = np.asarray(finger_poses[0][1], dtype=float)
        p2 = np.asarray(finger_poses[1][1], dtype=float)
        object_position = np.asarray(object_model.pos, dtype=float)
        if any(
            value.shape != (3,) or not np.all(np.isfinite(value))
            for value in (p1, p2, object_position)
        ):
            return False, {"checked": False, "reason": "window geometry is invalid"}
        span_xy = p2[:2] - p1[:2]
        denominator = float(np.dot(span_xy, span_xy))
        if denominator <= 1.0e-10:
            return False, {"checked": False, "reason": "finger span has no support-plane projection"}
        alpha = float(np.dot(object_position[:2] - p1[:2], span_xy) / denominator)
        closest_xy = p1[:2] + np.clip(alpha, 0.0, 1.0) * span_xy
        from r1pro_data_gen.domain import object_xy_radius_m, object_vertical_extent_m

        planar_surface_distance = max(
            0.0,
            float(np.linalg.norm(object_position[:2] - closest_xy))
            - float(object_xy_radius_m(object_model)),
        )
        object_height = float(object_vertical_extent_m(object_model))
        object_half_height = 0.5 * object_height
        from r1pro_data_gen.robot import gripper_min_vertical_overlap_m

        required_vertical_overlap = gripper_min_vertical_overlap_m(object_height)
        object_bottom = float(object_position[2] - object_half_height)
        object_top = float(object_position[2] + object_half_height)
        specs = {
            name: (center, half_extents)
            for name, center, half_extents in _finger_collision_box_specs(side)
        }
        overlaps: list[float] = []
        intervals: list[dict[str, float]] = []
        for body_name, position, rotation in finger_poses:
            position = np.asarray(position, dtype=float)
            rotation = np.asarray(rotation, dtype=float)
            profile = specs.get(body_name)
            if (
                profile is None
                or position.shape != (3,)
                or rotation.shape != (3, 3)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(rotation))
            ):
                return False, {"checked": False, "reason": "finger box geometry is invalid"}
            center_local, half_extents = profile
            box_center = position + rotation @ center_local
            half_z = float(np.sum(np.abs(rotation[2, :]) * half_extents))
            bottom = float(box_center[2] - half_z)
            top = float(box_center[2] + half_z)
            overlap = min(top, object_top + float(vertical_margin_m)) - max(
                bottom, object_bottom - float(vertical_margin_m)
            )
            overlaps.append(float(overlap))
            intervals.append({"bottom_z_m": bottom, "top_z_m": top, "overlap_m": float(overlap)})
        alpha_ready = 0.08 <= alpha <= 0.92
        planar_ready = planar_surface_distance <= float(surface_tolerance_m)
        vertical_ready = bool(overlaps) and all(
            value >= required_vertical_overlap for value in overlaps
        )
        ready = bool(alpha_ready and planar_ready and vertical_ready)
        return ready, {
            "checked": True,
            "window_geometry_source": "projected_finger_boxes",
            "segment_fraction_xy": alpha,
            "planar_surface_distance_m": planar_surface_distance,
            "vertical_overlap_m": min(overlaps) if overlaps else float("nan"),
            "required_vertical_overlap_m": required_vertical_overlap,
            "finger_vertical_intervals": intervals,
            "alpha_ready": alpha_ready,
            "planar_ready": planar_ready,
            "vertical_ready": vertical_ready,
        }
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return False, {"checked": False, "reason": "window geometry unavailable"}


def _finger_interaction_window_ready(window_details: Mapping[str, Any]) -> bool:
    """Return whether a first one-finger interaction is geometrically bounded.

    The terminal certificate requires both profiled finger boxes to overlap the
    object's vertical extent.  During a downward acquisition, the upper box
    can reach the object a few millimetres before the lower box does.  That is
    a valid *intermediate* state only when the jaw is already centered in the
    support plane and at least one finger has entered the physical height
    band.  The object-motion guard and the terminal two-finger check remain
    mandatory; this helper must never be used as a grasp-success predicate.
    """
    if not isinstance(window_details, Mapping):
        return False
    if not bool(window_details.get("alpha_ready", False)) or not bool(
        window_details.get("planar_ready", False)
    ):
        return False
    intervals = window_details.get("finger_vertical_intervals", ())
    if not isinstance(intervals, (tuple, list)):
        return False
    overlaps: list[float] = []
    for interval in intervals:
        if not isinstance(interval, Mapping):
            continue
        try:
            value = float(interval.get("overlap_m"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            overlaps.append(value)
    return bool(overlaps) and any(value >= 0.0 for value in overlaps)


def _alignment_finger_support_clearance(
    side: str,
    finger_poses: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    support_top_z: float | None,
    clearance_m: float,
) -> tuple[bool, dict[str, Any]]:
    """Check that the full finger collision boxes stay above the source plane."""
    if support_top_z is None or not math.isfinite(float(support_top_z)):
        return True, {"checked": False}
    bottom_z = _alignment_finger_bottom_z(side, finger_poses)
    if bottom_z is None:
        return False, {"checked": False, "reason": "finger support clearance unavailable"}
    clearance = float(bottom_z) - float(support_top_z)
    details = {
        "checked": True,
        "reason": "clear" if clearance >= float(clearance_m) else "finger_support_collision",
        "support_top_z_m": float(support_top_z),
        "minimum_finger_bottom_z_m": float(bottom_z),
        "support_clearance_m": clearance,
        "required_clearance_m": float(clearance_m),
    }
    return clearance >= float(clearance_m), details


def _alignment_minimum_midpoint_z(
    adapter: Any,
    side: str,
    midpoint: np.ndarray,
    scene: Any,
    object_model: Any,
) -> float | None:
    """Derive a safe lower bound for the physical finger midpoint.

    The bound is evaluated from the current measured finger orientation and
    the actual profiled collision boxes.  It is used only to keep a later
    Cartesian correction from asking the jaw to descend through its source
    plane; it does not prescribe a scene-specific grasp height.
    """
    support_top_z = _alignment_support_top_z(scene, object_model)
    if support_top_z is None or not hasattr(adapter, "body_pose"):
        return None
    try:
        poses = []
        for body_name, _, _ in _finger_collision_box_specs(side):
            position, quaternion = adapter.body_pose(body_name)
            poses.append(
                (
                    body_name,
                    np.asarray(position, dtype=float),
                    _wxyz_rotation(quaternion),
                )
            )
        bottom_z = _alignment_finger_bottom_z(side, tuple(poses))
        midpoint = np.asarray(midpoint, dtype=float)
        if bottom_z is None or midpoint.shape != (3,) or not np.all(np.isfinite(midpoint)):
            return None
        # Preserve the current orientation-dependent bottom-to-midpoint
        # offset while descending.  If the wrist rotates, the static/runtime
        # box certificates above remain authoritative for the new orientation.
        bottom_offset = float(bottom_z) - float(midpoint[2])
        from r1pro_data_gen.robot.robot_config import (
            R1PRO_GRIPPER_PREGRASP_CLEARANCE_M,
            R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M,
        )

        return float(support_top_z) + float(R1PRO_GRIPPER_PREGRASP_CLEARANCE_M) - bottom_offset
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _predicted_alignment_window_ready(
    finger_poses: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    object_model: Any,
) -> bool:
    """Return whether predicted physical finger boxes form a terminal window."""
    p1, p2 = (np.asarray(item[1], dtype=float) for item in finger_poses)
    object_position = np.asarray(object_model.pos, dtype=float)
    span = p2 - p1
    denominator = float(np.dot(span, span))
    if denominator <= 1.0e-10:
        return False
    alpha = float(np.dot(object_position - p1, span) / denominator)
    closest = p1 + np.clip(alpha, 0.0, 1.0) * span
    ready, _ = _finger_window_geometry_ready(
        "left" if "left_" in finger_poses[0][0] else "right",
        finger_poses,
        object_model,
        surface_tolerance_m=0.012,
        # This is the terminal contact window, not the non-contact
        # pregrasp envelope.  Inflating the vertical band here can report a
        # valid-looking window while the fingers are still above the object;
        # the subsequent prismatic close cannot correct that Z error.
        vertical_margin_m=0.0,
    )
    return ready


def _alignment_finger_path_certificate(
    kin: Any,
    adapter: Any,
    side: str,
    trajectory: np.ndarray,
    start_q: np.ndarray,
    object_model: Any,
    base_pose: Any,
    model_to_world_rotation: np.ndarray | None,
    model_to_world_translation: np.ndarray | None,
    support_top_z: float | None = None,
) -> tuple[bool, np.ndarray, dict[str, Any]]:
    """Certify/truncate an arm path against the measured finger mesh envelope."""
    opening_offsets = _alignment_finger_opening_offsets(
        kin,
        adapter,
        side,
        start_q,
        base_pose,
        model_to_world_rotation,
        model_to_world_translation,
    )
    if opening_offsets is None:
        return True, np.asarray(trajectory, dtype=float), {"checked": False}
    from r1pro_data_gen.robot.robot_config import R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M

    interaction_started = False
    for index, q_target in enumerate(np.asarray(trajectory, dtype=float)):
        finger_poses = _predicted_alignment_finger_poses(
            kin,
            adapter,
            side,
            q_target,
            opening_offsets,
            base_pose,
            model_to_world_rotation,
            model_to_world_translation,
        )
        support_free, support_details = _alignment_finger_support_clearance(
            side,
            finger_poses,
            support_top_z,
            # The profile's support reserve is the physical minimum used
            # during a low-object reorientation. The larger pre-grasp
            # clearance is still used when deriving the nominal center target,
            # but applying it as a hard path gate rejects millimetre-level
            # drive tracking variation despite a real positive gap to the
            # support surface.
            float(R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M),
        )
        if not support_free:
            return False, np.asarray(trajectory, dtype=float), {
                **support_details,
                "terminal_contact_index": int(index),
                "terminal_window_ready": False,
            }
        free, details = _finger_box_collision(
            object_model,
            finger_poses,
            # The finger boxes are the permitted acquisition envelope.  Use
            # exact object geometry for the terminal sample; the source
            # support reserve and the palm collision checker still provide
            # the non-contact safety margin around this intentional contact.
            0.0,
        )
        if not free:
            window_ready, window_details = _finger_window_geometry_ready(
                side,
                finger_poses,
                object_model,
                surface_tolerance_m=0.012,
                vertical_margin_m=0.0,
            )
            if window_ready:
                # The first valid physical contact is the terminal sample;
                # never execute the remaining precomputed samples through it.
                return True, np.asarray(trajectory, dtype=float)[: index + 1], {
                    **details,
                    "window_geometry": window_details,
                    "terminal_contact_index": int(index),
                    "terminal_window_ready": True,
                }
            if _finger_interaction_window_ready(window_details):
                # The open jaw may already have one finger inside the exact
                # contact envelope while the other box is still a few mm
                # above the object.  Keep checking the complete trajectory,
                # but allow this bounded intermediate interaction so the
                # vertical correction can establish the second side.  Runtime
                # live collision and object-motion gates remain active for
                # every executed sample; this is never a success predicate.
                interaction_started = True
                continue
            if interaction_started:
                # Once a branch has entered the physical contact band, a later
                # sample outside that band is not a safe prefix: the branch is
                # leaving the object while still colliding with it. Treating
                # the prefix as a successful correction (especially when it
                # contains only the current sample) lets the outer loop repeat
                # the same no-op indefinitely. Reject this branch and let the
                # bounded step subdivision try a genuinely local one.
                return False, np.asarray(trajectory, dtype=float), {
                    **details,
                    "window_geometry": window_details,
                    "terminal_contact_index": int(index),
                    "terminal_window_ready": False,
                    "interaction_window_lost": True,
                }
            # The commanded local correction can become unsafe only after a
            # safe prefix, for example when a redundant branch drifts toward
            # one side of a low object before the next measured replan. Keep
            # the safe prefix and stop before the first colliding sample; a
            # receding-horizon caller will remeasure the physical jaw and
            # choose a new branch. This is still fail-closed at the physical
            # envelope: the colliding sample itself is never executed, and a
            # collision at the first sample remains a hard rejection.
            if index >= 2:
                return True, np.asarray(trajectory, dtype=float)[:index], {
                    **details,
                    "window_geometry": window_details,
                    "terminal_contact_index": int(index),
                    "terminal_window_ready": False,
                    "truncated_before_unsafe_contact": True,
                    "safe_prefix_sample_count": int(index),
                }
            return False, np.asarray(trajectory, dtype=float), {
                **details,
                "window_geometry": window_details,
                "terminal_contact_index": int(index),
                "terminal_window_ready": False,
            }
    return True, np.asarray(trajectory, dtype=float), {
        "checked": True,
        "checked_sample_count": int(len(trajectory)),
    }


def _live_alignment_window_ready(adapter: Any, object_name: str | None, side: str) -> bool:
    """Read the live semantic window before accepting finger-envelope contact."""
    if object_name is None or not hasattr(adapter, "gripper_object_alignment"):
        return False
    try:
        details = adapter.gripper_object_alignment(object_name, side=side)
        return bool(details.get("between_fingers", False))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False


def _live_alignment_interaction_window_ready(
    adapter: Any,
    object_name: str | None,
    side: str,
) -> bool:
    """Return whether finger/object contact is geometrically intentional.

    ``between_fingers`` is the final close-ready certificate.  During the
    approach, however, the open finger mesh can touch the object before the
    projected zero-opening test is true.  Treating that first mesh contact as
    an ordinary collision makes the receding-horizon correction unable to
    reach the close-ready pose.  The adapter exposes the less strict physical
    window separately; it still requires both finger boxes to straddle the
    object in the support plane and vertical band, while the outer alignment
    loop continues to require the stricter predicted-close certificate.
    """
    if object_name is None or not hasattr(adapter, "gripper_object_alignment"):
        return False
    try:
        details = adapter.gripper_object_alignment(object_name, side=side)
        if details.get("window_geometry_source") != "projected_finger_boxes":
            return False
        fraction = float(details.get("segment_fraction_xy"))
        planar = float(details.get("planar_surface_distance_m"))
        tolerance = float(details.get("surface_tolerance_m", 0.012))
        intervals = details.get("finger_vertical_intervals", ())
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    interaction_window = {
        "alpha_ready": 0.08 <= fraction <= 0.92,
        "planar_ready": planar <= tolerance,
        "finger_vertical_intervals": intervals,
    }
    return bool(
        np.isfinite(fraction)
        and np.isfinite(planar)
        and np.isfinite(tolerance)
        and _finger_interaction_window_ready(interaction_window)
    )


def _live_alignment_finger_collision(
    adapter: Any,
    scene: Any,
    side: str,
    protected_object_name: str | None,
    exclude_objects: tuple[str, ...],
) -> tuple[bool, dict[str, Any]]:
    """Check actual finger-link mesh envelopes against the live object."""
    telemetry_adapter = getattr(adapter, "_adapter", adapter)

    def finish(free: bool, details: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        try:
            telemetry_adapter._alignment_finger_collision_checks = int(
                getattr(telemetry_adapter, "_alignment_finger_collision_checks", 0)
            ) + 1
            telemetry_adapter._alignment_finger_collision_last = str(
                details.get("reason", "clear" if free else "rejected")
            )
        except (AttributeError, TypeError, ValueError):
            pass
        return free, details

    if protected_object_name is None or scene is None:
        return finish(True, {"checked": False})
    if not hasattr(adapter, "body_pose"):
        return finish(True, {"checked": False, "reason": "adapter has no body_pose"})
    try:
        from r1pro_data_gen.robot.robot_config import R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M
        from r1pro_data_gen.skills.planning import runtime_scene_snapshot

        if protected_object_name in set(exclude_objects):
            return finish(False, {
                "checked": False,
                "reason": "protected alignment object was excluded from collision scene",
                "object_name": protected_object_name,
            })
        live_scene = runtime_scene_snapshot(scene, adapter, exclude_objects=())
        object_model = live_scene.object(protected_object_name)
        finger_poses = []
        telemetry_records = []
        for body_name, _, _ in _finger_collision_box_specs(side):
            position, quaternion = adapter.body_pose(body_name)
            position = np.asarray(position, dtype=float)
            rotation = _wxyz_rotation(quaternion)
            finger_poses.append((body_name, position, rotation))
            profile = next(
                item
                for item in _finger_collision_box_specs(side)
                if item[0] == body_name
            )
            box_center = position + rotation @ profile[1]
            telemetry_records.append(
                f"{body_name}:p=({','.join(f'{float(v):.6f}' for v in position)});"
                f"q=({','.join(f'{float(v):.6f}' for v in quaternion)});"
                f"box=({','.join(f'{float(v):.6f}' for v in box_center)})"
            )
        try:
            telemetry_adapter._alignment_finger_pose_last = ";".join(
                telemetry_records
            ) + f";object=({','.join(f'{float(v):.6f}' for v in object_model.pos)})"
        except (AttributeError, TypeError, ValueError):
            pass
        support_free, support_details = _alignment_finger_support_clearance(
            side,
            tuple(finger_poses),
            _alignment_support_top_z(live_scene, object_model),
            float(R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M),
        )
        if not support_free:
            return finish(False, {**support_details, "object_name": protected_object_name})
        free, details = _finger_box_collision(
            object_model,
            tuple(finger_poses),
            # This is an intentional finger/object interaction check.  Do
            # not reuse the free-space pregrasp inflation or the alignment
            # loop will stop above a low object before the jaws can touch it.
            0.0,
        )
        return finish(free, {**details, "object_name": protected_object_name})
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return finish(False, {
            "checked": False,
            "reason": "live finger collision check unavailable",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "object_name": protected_object_name,
        })


def _slow_retime_same_path(
    trajectory: np.ndarray,
    factor: int = _TRACKING_RECOVERY_FACTOR,
) -> np.ndarray:
    """Slow a certified joint path without changing its geometry.

    Repeating each already-certified sample is deliberately conservative: it
    cannot invent a new IK branch or cut a corner through an obstacle.  This is
    used only after physical tracking reports a large error, when a compliant
    gravity-loaded joint needs more settling time than the planner's nominal
    profile.  The first and last samples remain the original endpoints.
    """
    path = np.asarray(trajectory, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
        raise ValueError("trajectory must have shape (n>=2, 7)")
    repeats = max(1, int(factor))
    if repeats == 1:
        return path.copy()
    return np.repeat(path, repeats, axis=0)


def rotate_quat_about_axis(quat: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate quaternion (w, x, y, z) about ``axis`` (world frame) by ``angle``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    # dq = cos(a/2) + sin(a/2) * axis  (wxyz)
    half = angle / 2.0
    dq = np.array(
        [math.cos(half), axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half)]
    )
    # Hamilton product q * dq
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = dq
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    return result / np.linalg.norm(result)


def direction_steps(p0: np.ndarray, direction: np.ndarray, distance: float, step: float) -> list[np.ndarray]:
    """End-effector positions along ``direction`` (base frame), step increments."""
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        raise ValueError("direction must be non-zero")
    d = d / norm
    steps: list[np.ndarray] = []
    dist = 0.0
    while dist < distance - 1e-9:
        dist = min(dist + step, distance)
        steps.append(p0 + d * dist)
    return steps


def _slerp_wxyz(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    rots = Rotation.from_quat([[q1[1], q1[2], q1[3], q1[0]], [q2[1], q2[2], q2[3], q2[0]]])
    q = Slerp([0, 1], rots)(t).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


class ArmTrajectoryFollow:
    """Follow a planned joint trajectory (from the planning layer).

    The trajectory is a sequence of 7-DOF joint configurations (produced by
    ``query_arm_path`` / MPlib, or by any other planner). This skill only
    executes it with speed-limited interpolation -- it does not re-solve IK.
    """

    name = "arm_trajectory_follow"
    tier = "backend"
    exposed = False
    description = (
        "Execute a planned joint trajectory (sequence of 7-DOF configurations, "
        "optionally with per-point joint velocities) with speed-limited or "
        "velocity-referenced tracking. Takes a trajectory from the planning "
        "layer; does not re-solve IK."
    )
    parameters: dict[str, ParamSpec] = {
        "trajectory": ParamSpec("array", "List of joint configurations (each 7 values, rad)", required=True),
        "velocities": ParamSpec("array", "Optional per-point joint velocities (each 7 values, rad/s)", default=None),
        "side": ParamSpec("string", "Which arm ('left' or 'right')", default="left", enum=("left", "right")),
        "speed_scale": ParamSpec("number", "Fraction of the joint velocity limits", default=0.3),
        "trajectory_dt": ParamSpec("number", "Time between trajectory samples (s); defaults to simulator dt", default=None, minimum=1e-4),
    }

    def __init__(self, kin: Any, vel_limits: np.ndarray, speed_scale: float = 0.3):
        self.kin = kin
        self.vel_limits = vel_limits
        self.speed_scale = speed_scale

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        trajectory: list[list[float]] = None,
        velocities: list[list[float]] | None = None,
        side: str = "left",
        speed_scale: float = 0.3,
        trajectory_dt: float | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        if trajectory is None or len(trajectory) < 2:
            raise ValueError("arm_trajectory_follow requires a trajectory of >=2 joint configs")
        side = require_side(side)
        kin = for_side(self.kin, side)
        vel_limits = np.asarray(for_side(self.vel_limits, side), dtype=float)
        joints = ARM_JOINTS_BY_SIDE[side]
        traj = [np.asarray(q, dtype=float) for q in trajectory]
        for i, q in enumerate(traj):
            if q.shape != (7,):
                raise ValueError(f"arm_trajectory_follow waypoint {i} must be 7 values, got {q.shape}")

        # When per-point joint velocities are supplied (a TOPP time-parameterized
        # trajectory), the points are tracked at the TOPP cadence with PURE
        # position references: the velocity feed-forward was removed because it
        # commands instant joint direction changes at the shortcut corner
        # waypoints, which saturates the weak joints (measured: joint4 tracking
        # error diverged to 0.85 rad with feed-forward, 0.02 rad without --
        # the reference project also plays its references position-only).
        if velocities is not None and len(velocities) == len(traj):
            # The planner owns the reference clock. Do not silently play a 10 Hz
            # path at 60 Hz. MPlib currently returns 60 Hz samples, while this
            # resampling also makes externally supplied trajectories safe.
            reference_dt = float(trajectory_dt or getattr(adapter, "dt", 1.0 / 60.0))
            control_dt = float(getattr(adapter, "dt", 1.0 / 60.0))
            if abs(reference_dt - control_dt) > 1e-7:
                from r1pro_data_gen.methods.manipulation.mplib_path import resample_trajectory

                positions = np.asarray(traj, dtype=float)
                refs = np.asarray(velocities, dtype=float)
                sampled, _, _, _ = resample_trajectory(
                    positions, refs, None, dt_out=control_dt, dt_in=reference_dt
                )
                traj = [np.asarray(q, dtype=float) for q in sampled]
            max_tracking_error = 0.0
            max_actual_step = 0.0
            prev_actual: np.ndarray | None = None
            for q in traj:
                adapter.set_targets(
                    position={j: float(q[i]) for i, j in enumerate(joints)},
                    velocity={},
                )
                adapter.step()
                if step_hook is not None:
                    step_hook()
                obs = adapter.read_observation(0.0)
                actual = np.asarray([obs.joint_positions[j] for j in joints], dtype=float)
                max_tracking_error = max(max_tracking_error, float(np.max(np.abs(actual - q))))
                if prev_actual is not None:
                    max_actual_step = max(max_actual_step, float(np.max(np.abs(actual - prev_actual))))
                prev_actual = actual
        else:
            segment = ArmSegmentExecutor(kin, vel_limits, speed_scale, hold_steps=0)
            merge_eps = 0.02  # rad: merge waypoints this close together
            q_prev = traj[0]
            for q in traj[1:]:
                if float(np.max(np.abs(q - q_prev))) < merge_eps:
                    q_prev = q
                    continue
                final_err = segment.execute(adapter, side, q_prev, q, step_hook)
                if final_err >= _FINAL_ERROR_TOL:
                    return SkillResult(success=False, skill=self.name,
                                       metrics={"final_error_rad": float(final_err)})
                q_prev = q
        # Keep the final reference active until the gravity-loaded arm has
        # actually converged.  A fixed 0.25 s hold was too short once the base
        # was correctly locked: weak joints could still lag by >0.3 rad even
        # though the reference itself was smooth.
        settle_steps = 0
        stable_steps = 0
        final_actual = None
        for _ in range(18):
            adapter.step()
            if step_hook is not None:
                step_hook()
            settle_steps += 1
            obs = adapter.read_observation(0.0)
            final_actual = np.asarray([obs.joint_positions[j] for j in joints], dtype=float)
            if float(np.max(np.abs(final_actual - traj[-1]))) < _FINAL_ERROR_TOL:
                stable_steps += 1
                if stable_steps >= 3:
                    break
            else:
                stable_steps = 0
        if final_actual is None:  # pragma: no cover - loop always executes
            obs = adapter.read_observation(0.0)
            final_actual = np.asarray([obs.joint_positions[j] for j in joints], dtype=float)
        final_err = float(np.max(np.abs(final_actual - traj[-1])))
        return SkillResult(
            success=bool(final_err < _FINAL_ERROR_TOL),
            skill=self.name,
            metrics={
                "waypoints": float(len(traj)),
                "final_error_rad": float(final_err),
                "max_tracking_error_rad": float(locals().get("max_tracking_error", final_err)),
                "max_actual_step_rad": float(locals().get("max_actual_step", 0.0)),
                "settle_steps": float(settle_steps),
            },
            details={
                "final_target_q": np.asarray(traj[-1]).round(5).tolist(),
                "final_actual_q": final_actual.round(5).tolist(),
            },
        )


class ArmMoveThrough:
    """Plan all ordered EE waypoints first, then execute one certified path."""

    name = "arm_move_through"
    tier = "semantic"
    exposed = True
    description = (
        "Jointly select continuous IK branches for ordered end-effector poses, "
        "certify the complete collision-free trajectory, and execute it once."
    )
    parameters: dict[str, ParamSpec] = {
        "waypoints": ParamSpec(
            "array",
            "Ordered objects with name, EE pose alternatives, and incoming-edge collision exclusions",
            required=True,
            min_items=1,
        ),
        "side": ParamSpec("string", "Arm side", default="left", enum=("left", "right")),
        "planning_time": ParamSpec("number", "Planning time for each candidate edge (s)", default=0.8, minimum=0.1),
        "ik_candidates_per_waypoint": ParamSpec("integer", "Maximum IK branches retained across all poses at each waypoint", default=3, minimum=1),
        "beam_width": ParamSpec("integer", "Number of verified sequence prefixes retained", default=3, minimum=1),
        "max_planned_edges": ParamSpec("integer", "Hard planning budget across all candidate edges", default=72, minimum=1),
        "trajectory_speed_scale": ParamSpec("number", "Fraction of arm velocity limits used by the complete path", default=0.42, minimum=0.02),
        "local_radius_m": ParamSpec("number", "Obstacle culling radius around the live base (m)", default=2.0, minimum=0.5),
    }

    def __init__(self, kin: Any, vel_limits: np.ndarray, planner: Any):
        self.kin = kin
        self.vel_limits = vel_limits
        self.planner = planner

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        waypoints: list[dict[str, Any]] | None = None,
        side: str = "left",
        planning_time: float = 0.8,
        ik_candidates_per_waypoint: int = 3,
        beam_width: int = 3,
        max_planned_edges: int = 72,
        trajectory_speed_scale: float = 0.42,
        local_radius_m: float = 2.0,
        carried_context: Any = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if not waypoints:
            raise ValueError("arm_move_through requires at least one waypoint")
        if scene is None:
            return SkillResult(False, self.name, details={"reason": "arm_move_through requires a scene"})
        side = require_side(side)
        kin = for_side(self.kin, side)
        planner = for_side(self.planner, side)
        vel_limits = np.asarray(for_side(self.vel_limits, side), dtype=float)
        if kin is None:
            return SkillResult(False, self.name, details={"reason": "kinematics backend is unavailable"})

        from r1pro_data_gen.methods.manipulation.arm_path_optimizer import optimize_arm_waypoint_path
        from r1pro_data_gen.methods.manipulation.mplib_path import mplib_qpos_from_joint_positions
        from r1pro_data_gen.methods.manipulation.contracts import ArmWaypoint
        from r1pro_data_gen.skills.planning import runtime_scene_snapshot

        parsed: list[ArmWaypoint] = []
        for index, raw in enumerate(waypoints):
            if not isinstance(raw, dict):
                raise ValueError(f"arm_move_through waypoint {index} must be an object")
            pose_items = raw.get("poses")
            if not isinstance(pose_items, list) or not pose_items:
                raise ValueError(f"arm_move_through waypoint {index} requires poses")
            poses = []
            for pose_index, pose in enumerate(pose_items):
                if not isinstance(pose, dict):
                    raise ValueError(f"waypoint {index} pose {pose_index} must be an object")
                position = np.asarray(pose.get("position"), dtype=float)
                orientation = np.asarray(pose.get("orientation"), dtype=float)
                if position.shape != (3,) or orientation.shape != (4,):
                    raise ValueError(f"waypoint {index} pose {pose_index} has invalid shape")
                norm = float(np.linalg.norm(orientation))
                if not np.all(np.isfinite(position)) or not np.isfinite(norm) or norm < 1e-9:
                    raise ValueError(f"waypoint {index} pose {pose_index} must be finite")
                poses.append(
                    (
                        tuple(float(value) for value in position),
                        tuple(float(value) for value in orientation / norm),
                    )
                )
            exclusions = tuple(str(value) for value in raw.get("exclude_objects", ()))
            parsed.append(
                ArmWaypoint(
                    name=str(raw.get("name") or f"waypoint_{index + 1:02d}"),
                    poses=tuple(poses),
                    exclude_objects=exclusions,
                    contact=bool(raw.get("contact", False)),
                    speed_scale=(
                        None
                        if raw.get("speed_scale") is None
                        else float(raw["speed_scale"])
                    ),
                )
            )

        observation = adapter.read_observation(0.0)
        joints = ARM_JOINTS_BY_SIDE[side]
        q_current = np.asarray(
            [observation.joint_positions[name] for name in joints],
            dtype=float,
        )
        full_q_current = mplib_qpos_from_joint_positions(observation.joint_positions)
        base_pose = observation.base_pose or (0.0, 0.0, 0.0)
        base_xy = (float(base_pose[0]), float(base_pose[1]))
        base_yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
        live_scene = runtime_scene_snapshot(scene, adapter)

        def scene_for_exclusions(exclusions: tuple[str, ...]) -> Any:
            # Reuse the shared snapshot helper so object exclusions also prune
            # contact/collision sensor filters; rebuilding a SceneModel with a
            # stale filter is a contract error.
            # ``live_scene`` is already synchronized above; passing ``None``
            # keeps this pure exclusion/sensor pruning from re-reading moving
            # objects and changing the frozen planning snapshot.
            return runtime_scene_snapshot(live_scene, None, exclusions)

        # A carried-object sequence is usually a short local chain. Try the
        # deterministic, margin-best IK chain first so the online loop does
        # not spend an unbounded amount of time exploring OMPL branches for a
        # motion that can already be certified by dense collision checks. If
        # that chain is not feasible, fall through to the multi-candidate
        # planner below; the fallback never weakens the collision contract.
        if carried_context is not None and len(parsed) > 1:
            fast_path = _arm_move_through_margin_best_fallback(
                self,
                adapter,
                scene,
                waypoints=tuple(parsed),
                side=side,
                speed_scale=float(trajectory_speed_scale),
                step_hook=step_hook,
                carried_context=carried_context,
            )
            if fast_path is not None:
                return fast_path

        planning = optimize_arm_waypoint_path(
            planner,
            kin,
            q_current,
            tuple(parsed),
            live_scene,
            scene_for_exclusions=scene_for_exclusions,
            base_xy=base_xy,
            base_yaw=base_yaw,
            full_q_current=full_q_current,
            planning_time=float(planning_time),
            local_radius_m=float(local_radius_m),
            speed_scale=float(trajectory_speed_scale),
            side=side,
            ik_candidates_per_waypoint=int(ik_candidates_per_waypoint),
            beam_width=int(beam_width),
            max_planned_edges=int(max_planned_edges),
        )
        if not planning.success or planning.winner is None:
            # Multi-waypoint MPlib planning can fail when the authored waypoints
            # (e.g. a scene object's static position) disagree with the live
            # state -- the carried object is in the fingers, not at its reset
            # pose -- or when OMPL exhausts its budget in the real process.  As
            # a last resort, replay the waypoint sequence with the margin-best
            # IK executor that has proven reliable in the simulator for the
            # local align/descend/carry moves.
            fallback = _arm_move_through_margin_best_fallback(
                self,
                adapter,
                scene,
                waypoints=tuple(parsed),
                side=side,
                speed_scale=float(trajectory_speed_scale),
                step_hook=step_hook,
                carried_context=carried_context,
            )
            if fallback is not None:
                return fallback
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": planning.reason,
                    "planning_status": planning.status,
                    "request_hash": planning.request_hash,
                    "optimality_scope": planning.optimality_scope,
                    "candidate_failures": [
                        {
                            "sequence_id": candidate.sequence_id,
                            "failure_stage": candidate.constraints.stage,
                            "reasons": list(candidate.constraints.reasons),
                            "segments": list(candidate.segment_reports),
                        }
                        for candidate in planning.candidates
                        if not candidate.valid
                    ],
                },
            )

        execution_observation = adapter.read_observation(0.0)
        execution_q = mplib_qpos_from_joint_positions(execution_observation.joint_positions)
        execution_base = execution_observation.base_pose or (0.0, 0.0, 0.0)
        robot_changed = float(np.max(np.abs(execution_q - full_q_current))) > 0.01
        base_changed = (
            float(np.linalg.norm(np.asarray(execution_base[:2]) - np.asarray(base_pose[:2]))) > 0.005
            or abs(
                float(execution_base[2] if len(execution_base) > 2 else 0.0)
                - base_yaw
            ) > 0.01
        )
        execution_scene = runtime_scene_snapshot(scene, adapter)
        planned_objects = {obj.name: np.asarray(obj.pos, dtype=float) for obj in live_scene.objects}
        execution_objects = {
            obj.name: np.asarray(obj.pos, dtype=float) for obj in execution_scene.objects
        }
        object_set_changed = planned_objects.keys() != execution_objects.keys()
        scene_changed = object_set_changed or any(
            float(np.linalg.norm(execution_objects[name] - position)) > 0.005
            for name, position in planned_objects.items()
            if name in execution_objects
        )
        if robot_changed or base_changed or scene_changed:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "planning request became stale before execution",
                    "planning_status": "stale_scene",
                    "request_hash": planning.request_hash,
                    "robot_changed": robot_changed,
                    "base_changed": base_changed,
                    "scene_changed": scene_changed,
                    "object_set_changed": object_set_changed,
                },
            )

        if carried_context is not None:
            from r1pro_data_gen.methods.collision import carried_object_path_free

            carried_exclusions = tuple(sorted({
                str(name)
                for waypoint in parsed
                for name in waypoint.exclude_objects
            }))
            carried_free, carried_details = carried_object_path_free(
                kin,
                np.asarray(planning.winner.output["position"], dtype=float),
                live_scene,
                carried_context,
                base_xy=base_xy,
                base_yaw=base_yaw,
                exclude=carried_exclusions,
            )
            if not carried_free:
                return SkillResult(
                    False,
                    self.name,
                    details={
                        "reason": "carried object failed swept collision verification",
                        "planning_status": planning.status,
                        "carried_object_collision": carried_details,
                    },
                )

        output = dict(planning.winner.output or {})
        stabilize_base(adapter)
        held_context_verified = True
        attachment_failure: dict[str, object] | None = None
        execution_step_hook = step_hook
        if carried_context is not None:
            object_name = str(carried_context.object_name)
            expected_effector = f"{side}_gripper_finger_midpoint"

            def verify_held_context() -> None:
                nonlocal held_context_verified, attachment_failure
                if step_hook is not None:
                    step_hook()
                if not hasattr(adapter, "attachment_state"):
                    held_context_verified = False
                    attachment_failure = {
                        "reason": "adapter does not expose attachment state",
                        "object_name": object_name,
                    }
                    raise _HeldContextLost
                try:
                    state = adapter.attachment_state()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    held_context_verified = False
                    attachment_failure = {
                        "reason": "attachment state unavailable during carry",
                        "object_name": object_name,
                        "error": str(exc),
                    }
                    raise _HeldContextLost from exc
                observed_effector = state.get(object_name)
                if observed_effector != expected_effector:
                    held_context_verified = False
                    attachment_failure = {
                        "reason": "carried object attachment was lost",
                        "object_name": object_name,
                        "expected_effector": expected_effector,
                        "observed_effector": observed_effector,
                    }
                    raise _HeldContextLost

            try:
                verify_held_context()
            except _HeldContextLost:
                return SkillResult(
                    False,
                    self.name,
                    metrics={
                        "held_context_verified": False,
                        "failure_code": "attachment_lost",
                    },
                    details=attachment_failure or {},
                )
            execution_step_hook = verify_held_context
        try:
            execution = ArmTrajectoryFollow(kin, vel_limits).execute(
                adapter,
                scene=live_scene,
                trajectory=np.asarray(output["position"], dtype=float).tolist(),
                velocities=np.asarray(output["velocity"], dtype=float).tolist(),
                trajectory_dt=float(output["dt"]),
                side=side,
                step_hook=execution_step_hook,
            )
        except _HeldContextLost:
            return SkillResult(
                False,
                self.name,
                metrics={
                    "held_context_verified": False,
                    "failure_code": "attachment_lost",
                },
                details=attachment_failure or {},
            )
        if not execution.success:
            return SkillResult(
                False,
                self.name,
                metrics=dict(execution.metrics),
                details={"reason": "trajectory execution failed", **execution.details},
            )
        winner = planning.winner
        return SkillResult(
            True,
            self.name,
            metrics={
                **execution.metrics,
                **{key: float(value) for key, value in winner.metrics.items()},
                "held_context_verified": bool(held_context_verified),
                "failure_code": None,
            },
            details={
                "reason": "all waypoints reached on one certified trajectory",
                "planning_status": planning.status,
                "request_hash": planning.request_hash,
                "optimality_scope": planning.optimality_scope,
                "trajectory_points": len(output["position"]),
                "waypoint_candidates": [
                    {
                        "waypoint_id": candidate.waypoint_id,
                        "candidate_id": candidate.candidate_id,
                        "orientation_id": candidate.orientation_id,
                        "q_goal": np.asarray(candidate.q_goal).round(5).tolist(),
                        "minimum_limit_margin": candidate.minimum_limit_margin,
                        "minimum_singular_value": candidate.minimum_singular_value,
                        "wrist_motion": candidate.wrist_motion,
                    }
                    for candidate in winner.waypoint_candidates
                ],
                "segments": list(winner.segment_reports),
            },
        )


class ArmMoveTo:
    """Certified semantic arm motion: EE goal -> IK -> collision-free trajectory.

    This is the action exposed to a task planner. It accepts either a position
    only or a full quaternion pose, reads the live robot and object state, keeps
    the IK branch close to the current posture, plans a verified joint path, and
    executes the time-sampled reference. A caller does not need to manually
    assemble ``query_ik_solution`` and ``query_arm_path`` for ordinary motion.
    """

    name = "arm_move_to"
    tier = "semantic"
    exposed = True
    description = (
        "Move either end-effector to a base-frame position or pose with a "
        "collision-checked, smooth joint trajectory. Position-only goals keep "
        "the current orientation as a soft preference and choose the minimum-"
        "motion redundant IK branch. A grasp_center goal may omit orientation; "
        "the robot-level calibrated gripper default is then used. Reads live "
        "scene state before planning."
    )
    parameters: dict[str, ParamSpec] = {
        "target_pos": ParamSpec("array", "Target EE position (x, y, z) in the current base frame", required=True, shape=(3,)),
        "target_quat": ParamSpec("array", "Optional target quaternion (w, x, y, z); grasp_center uses the robot-calibrated default when omitted", default=None, shape=(4,)),
        "target_z_axis": ParamSpec("array", "Optional desired EE z-axis (x, y, z); alternative to target_quat", default=None, shape=(3,)),
        "target_frame": ParamSpec("string", "Target frame: ee link origin or midpoint between the two fingers", default="ee", enum=("ee", "grasp_center")),
        "side": ParamSpec("string", "Arm side", default="left", enum=("left", "right")),
        "planning_time": ParamSpec("number", "Planning time per candidate (s)", default=3.0, minimum=0.1),
        "ik_candidates": ParamSpec("integer", "Number of online IK branches considered", default=4, minimum=1),
        "trajectory_speed_scale": ParamSpec("number", "Fraction of arm velocity limits used by the smooth reference", default=0.42, minimum=0.02),
        "local_radius_m": ParamSpec("number", "Arm-planning obstacle culling radius around the live base (m)", default=2.0, minimum=0.5),
        "exclude_objects": ParamSpec("array", "Scene object names excluded from this arm motion's obstacle set", default=[]),
        "position_tolerance": ParamSpec("number", "Final EE position tolerance (m)", default=0.03, minimum=1e-4),
        "orientation_tolerance": ParamSpec("number", "Final orientation tolerance (rad)", default=0.10, minimum=1e-4),
        "max_joint_winding": ParamSpec("number", "Optional task-level joint winding limit; omitted paths remain eligible", default=None, minimum=1.0),
        "max_ee_winding": ParamSpec("number", "Optional task-level end-effector winding limit; omitted paths remain eligible", default=None, minimum=1.0),
    }

    def __init__(self, kin: Any, vel_limits: np.ndarray, planner: Any):
        self.kin = kin
        self.vel_limits = vel_limits
        self.planner = planner

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target_pos: list[float] | None = None,
        target_quat: list[float] | None = None,
        target_z_axis: list[float] | None = None,
        target_frame: str = "ee",
        side: str = "left",
        planning_time: float = 3.0,
        ik_candidates: int = 4,
        trajectory_speed_scale: float = 0.42,
        local_radius_m: float = 2.0,
        exclude_objects: list[str] | None = None,
        position_tolerance: float = 0.03,
        orientation_tolerance: float = 0.10,
        max_joint_winding: float | None = None,
        max_ee_winding: float | None = None,
        prefer_local_certified_path: bool = False,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if target_pos is None:
            raise ValueError("arm_move_to requires target_pos")
        side = require_side(side)
        kin = for_side(self.kin, side)
        planner = for_side(self.planner, side)
        vel_limits = np.asarray(for_side(self.vel_limits, side), dtype=float)
        # Set when the grasp-center target is reachable only position-first;
        # the commanded orientation is then deferred to a measured align stage.
        orientation_deferred = False
        if target_quat is not None and target_z_axis is not None:
            raise ValueError("arm_move_to accepts target_quat or target_z_axis, not both")
        if kin is None:
            return SkillResult(False, self.name, details={"reason": "kinematics backend is unavailable"})
        if scene is None:
            return SkillResult(False, self.name, details={"reason": "arm_move_to requires a scene for collision checking"})

        obs = adapter.read_observation(0.0)
        joints = ARM_JOINTS_BY_SIDE[side]
        q_cur = np.asarray([obs.joint_positions[j] for j in joints], dtype=float)
        if hasattr(kin, "set_auxiliary_q"):
            positions = getattr(obs, "joint_positions", {}) or {}
            kin.set_auxiliary_q(
                {
                    f"torso_joint{index}": float(positions[f"torso_joint{index}"])
                    for index in range(1, 5)
                    if f"torso_joint{index}" in positions
                }
            )
        base_pose_obs = obs.base_pose or (0.0, 0.0, 0.0)
        requested_pos = np.asarray(target_pos, dtype=float)
        if requested_pos.shape != (3,):
            raise ValueError(f"arm_move_to target_pos must be shape (3,), got {requested_pos.shape}")
        if target_frame not in {"ee", "grasp_center"}:
            raise ValueError("arm_move_to target_frame must be 'ee' or 'grasp_center'")
        # Keep the semantic grasp-center goal in world coordinates as well as
        # in the live base frame.  A mobile/torso robot can change its model
        # registration while a local arm segment is executing; the measured
        # local fallback uses this immutable world goal and re-solves from the
        # actual finger midpoint after every short segment.
        target_center_world = (
            _base_point_to_world(requested_pos, base_pose_obs)
            if target_frame == "grasp_center"
            else None
        )
        if target_quat is not None:
            quat = np.asarray(target_quat, dtype=float)
            if quat.shape != (4,) or np.linalg.norm(quat) < 1e-9:
                raise ValueError("target_quat must be a non-zero (w, x, y, z) quaternion")
            quat = quat / np.linalg.norm(quat)
        elif target_z_axis is not None:
            from .arm import quat_from_z_axis

            quat = quat_from_z_axis(np.asarray(target_z_axis, dtype=float))
        else:
            quat = None
        if target_frame == "grasp_center":
            if quat is None:
                from r1pro_data_gen.robot.robot_config import R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE

                # The parallel-jaw yaw is a robot capability, not a task
                # detail the LLM should invent.  A caller can still override
                # it with an observed/declared full quaternion.
                quat = np.asarray(
                    R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE[side], dtype=float
                )

            if prefer_local_certified_path:
                # A calibrated live loop can reach a semantic center through
                # several short, measured segments even when the final
                # low-workspace pose has no single-shot IK solution.  Invoke it
                # before the one-shot IK gate; otherwise the latter would
                # incorrectly classify a segmented physical approach as
                # unreachable.  Backends without the live measurement contract
                # return ``None`` and continue through the normal path below.
                measured_result = _arm_move_to_measured_grasp_center_local(
                    self,
                    adapter,
                    scene,
                    target_center_world=target_center_world,
                    target_quat=quat,
                    side=side,
                    speed_scale=float(trajectory_speed_scale),
                    exclude_objects=list(exclude_objects or ()),
                    position_tolerance=float(position_tolerance),
                    step_hook=step_hook,
                )
                if measured_result is not None:
                    return measured_result
            pos = kin.ee_target_from_grasp_center(requested_pos, quat)
            # Measured-tool correction: the URDF finger-midpoint offset is
            # pose-dependent relative to the simulated gripper (millimetres at
            # home, centimetres at pre-grasp poses), so commanding the model
            # grasp center onto the object makes the physical wrist lead the
            # descent and knock the object sideways.  Measure the live delta
            # once here so the move aims the actual finger gap; align still
            # closes any residual afterwards.
            measured_delta = _measured_grasp_center_delta(
                kin, adapter, side, base_pose_obs, q_cur
            )
            if measured_delta is not None:
                pos = pos - measured_delta
        else:
            pos = requested_pos

        if hasattr(kin, "ik_candidates"):
            solutions = kin.ik_candidates(
                pos, quat, q_cur, max_candidates=max(1, int(ik_candidates))
            )
        else:
            fallback = kin.ik(pos, quat, q_init=q_cur)
            solutions = [fallback] if fallback.success and fallback.q_arm is not None else []
        if not solutions and quat is not None:
            # One-shot IK from a stretched carry/extend seed can miss a
            # reachable descend. Advance the seed along short Cartesian
            # substeps (same helper used by waypoint sequences) so the
            # later screw interpolant still has a live-branch goal.
            from r1pro_data_gen.methods.manipulation.arm_path_optimizer import (
                _continuous_waypoint_ik_candidates,
            )

            solutions = _continuous_waypoint_ik_candidates(
                kin,
                pos,
                quat,
                q_cur,
                max(1, int(ik_candidates)),
            )
        if solutions:
            # Prefer the branch most continuous with the current posture.
            # Online IK ranks by naturalness, so a small correction (alignment,
            # tabletop approach) can list a distant branch first even when a
            # near-continuous one exists; OMPL then wastes its budget trying to
            # connect through a narrow gap.  Re-sorting by continuity keeps the
            # first candidates short, plan-able segments while the final winner
            # is still chosen by verified path quality.
            span = np.maximum(np.asarray(kin.upper, dtype=float) - np.asarray(kin.lower, dtype=float), 1e-9)
            solutions.sort(
                key=lambda item: float(
                    np.linalg.norm((np.asarray(item.q_arm, dtype=float) - q_cur) / span)
                )
            )
        if not solutions:
            if quat is None:
                # Position-only target: ik_candidates' QP/DLS seed loop can
                # return zero for targets that the single DLS solver still
                # converges on (same effect as the note in the grasp_center
                # fallback below).  Retry once with kin.ik so an ee position
                # goal does not fail purely because of solver diversity.
                single = kin.ik(pos, None, q_init=q_cur)
                if single.success and single.q_arm is not None:
                    solutions = [single]
            elif target_frame == "grasp_center":
                # Position-only fallback: navigation residuals and IK nulls can
                # make a grasp-center target reachable only without the commanded
                # orientation.  Move position-first and defer the orientation to
                # a measured align stage (arm_align_gripper) instead of failing.
                # Use the single DLS solver: ik_candidates' QP/DLS seed loop can
                # return zero for position-only targets that kin.ik reaches.
                pos_only_solution = kin.ik(pos, None, q_init=q_cur)
                pos_solutions = (
                    [pos_only_solution]
                    if pos_only_solution.success and pos_only_solution.q_arm is not None
                    else []
                )
                if pos_solutions:
                    span = np.maximum(
                        np.asarray(kin.upper, dtype=float)
                        - np.asarray(kin.lower, dtype=float),
                        1e-9,
                    )
                    pos_solutions.sort(
                        key=lambda item: float(
                            np.linalg.norm(
                                (np.asarray(item.q_arm, dtype=float) - q_cur) / span
                            )
                        )
                    )
                    solutions = pos_solutions
                    quat = None
                    orientation_deferred = True
        # A reachable screw/Cartesian chain does not need a one-shot IK goal.
        # Dummy unit-test scenes have no ``objects`` and must keep the factual
        # IK diagnosis; physical scenes continue into certified task-space.
        allow_task_space_without_ik = (
            not solutions
            and not prefer_local_certified_path
            and hasattr(scene, "objects")
        )
        if not solutions and not allow_task_space_without_ik:
            failed = kin.ik(pos, quat, q_init=q_cur)
            # Distinguish "target position is outside the arm workspace" from
            # "the position is reachable but not with the commanded pose".
            # The planner can act on that difference (move the base / pick a
            # closer stance vs relax or rotate the commanded orientation)
            # without guessing, and the paired tolerances feed the factual
            # feedback discrepancies.
            position_only = kin.ik(pos, None, q_init=q_cur)
            position_reachable = bool(
                position_only.success and position_only.q_arm is not None
            )
            base_pose_obs = obs.base_pose or (0.0, 0.0, 0.0)
            if quat is None:
                # The command already was position-only; there is no commanded
                # orientation to relax, so this is purely a workspace failure.
                reason = (
                    "IK failed: target position is outside the arm workspace "
                    "from this base pose"
                )
            else:
                reason = (
                    "IK failed: target position is outside the arm workspace from "
                    "this base pose"
                    if not position_reachable
                    else "IK failed: position reachable only without the commanded "
                    "orientation (relax or rotate it, or realign after moving)"
                )
            return SkillResult(
                False,
                self.name,
                metrics={
                    "ik_error_m": float(failed.position_error),
                    "ik_tolerance_m": float(position_tolerance),
                    "rotation_error_rad": float(failed.rotation_error),
                    "rotation_tolerance_rad": float(orientation_tolerance),
                    "position_reachable_without_orientation": bool(position_reachable),
                },
                details={
                    "reason": reason,
                    "q_current": q_cur.round(5).tolist(),
                    "target_frame": target_frame,
                    "base_pose_world": [round(float(v), 5) for v in base_pose_obs],
                    "target_pos_base": [round(float(v), 5) for v in pos],
                },
            )

        if (
            prefer_local_certified_path
            and target_frame == "grasp_center"
        ):
            # Ground-level grasp approaches are short local motions. Prefer
            # the already certified multi-seed IK path for this robot-level
            # workspace mode so a global planner cannot spend an unbounded
            # amount of time resolving a narrow near-ground approach. The
            # fallback performs the same dense collision check and is only
            # accepted when it reaches the requested pose.
            # The measured-center loop was attempted above, before one-shot IK.
            # If the backend cannot provide that capability, use the bounded
            # model-space fallbacks below; a real physical failure is returned
            # by the measured loop and never silently downgraded.
            if quat is not None:
                local_result = _arm_move_to_margin_best(
                    self,
                    adapter,
                    scene,
                    goal_ee=pos,
                    quat=quat,
                    side=side,
                    speed_scale=float(trajectory_speed_scale),
                    exclude_objects=list(exclude_objects or ()),
                    step_hook=step_hook,
                )
            else:
                local_result = _arm_move_to_position_only_local(
                    self,
                    adapter,
                    scene,
                    goal_ee=pos,
                    side=side,
                    speed_scale=float(trajectory_speed_scale),
                    exclude_objects=list(exclude_objects or ()),
                    position_tolerance=float(position_tolerance),
                    step_hook=step_hook,
                )
            if local_result is not None and local_result.success:
                return local_result
            if prefer_local_certified_path:
                # A caller explicitly selected the bounded local-motion
                # capability (currently used for low-workspace measured
                # approaches). Do not silently fall through to randomized
                # OMPL when that certified local path is rejected: the global
                # planner cannot repair a physically colliding torso/start
                # state and would consume the entire action budget without a
                # new fact for replanning.
                if local_result is not None:
                    return local_result
                return SkillResult(
                    False,
                    self.name,
                    details={
                        "reason": "certified local arm path unavailable",
                        "failure_code": "local_certified_path_unavailable",
                        "target_frame": target_frame,
                    },
                )

        from r1pro_data_gen.methods.manipulation.arm_path_optimizer import optimize_arm_path
        from r1pro_data_gen.methods.manipulation.mplib_path import (
            mplib_qpos_from_joint_positions,
            path_collision_free,
        )
        from r1pro_data_gen.skills.planning import runtime_scene_snapshot

        live_scene = runtime_scene_snapshot(scene, adapter, tuple(exclude_objects or ()))
        full_q_current = mplib_qpos_from_joint_positions(obs.joint_positions)
        base_pose = obs.base_pose or (0.0, 0.0, 0.0)
        planning = optimize_arm_path(
            planner,
            kin,
            q_cur,
            solutions,
            live_scene,
            base_xy=(float(base_pose[0]), float(base_pose[1])),
            base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
            full_q_current=full_q_current,
            planning_time=float(planning_time),
            local_radius_m=float(local_radius_m),
            speed_scale=float(trajectory_speed_scale),
            side=side,
            attempts_per_candidate=2,
            fallback_attempts_per_candidate=1,
            max_joint_winding=max_joint_winding,
            max_ee_winding=max_ee_winding,
            target_pos=pos,
            target_quat=quat,
        )
        if not planning.success or planning.winner is None:
            # A direct home->goal path can swoop under the table edge or
            # exhaust OMPL's randomized budget in a narrow approach.  Reach a
            # raised staging point first, then descend along the same pose:
            # the gripper stays above the tabletop and the final segment is
            # short.  Reuse arm_move_through so the two segments share the same
            # joint candidate selection, retiming and certification as one path.
            if quat is not None:
                staged = _arm_move_to_staged_fallback(
                    self,
                    adapter,
                    scene,
                    goal_ee=pos,
                    quat=quat,
                    side=side,
                    planning_time=float(planning_time),
                    local_radius_m=float(local_radius_m),
                    speed_scale=float(trajectory_speed_scale),
                    ik_candidates=max(4, int(ik_candidates)),
                    exclude_objects=list(exclude_objects or ()),
                    step_hook=step_hook,
                )
                if staged is not None:
                    return staged
            # OMPL state can degrade after the heavy multi-candidate use of a
            # prior arm_move_through on the same planner instance; rebuilding
            # the planner restores the deterministic direct-verified fast path.
            # Retry once before reporting a hard failure.
            from r1pro_data_gen.methods.manipulation.mplib_path import build_planner

            fresh = build_planner(side=side)
            retry = optimize_arm_path(
                fresh,
                kin,
                q_cur,
                solutions,
                live_scene,
                base_xy=(float(base_pose[0]), float(base_pose[1])),
                base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
                full_q_current=full_q_current,
                planning_time=float(planning_time),
                local_radius_m=float(local_radius_m),
                speed_scale=float(trajectory_speed_scale),
                side=side,
                attempts_per_candidate=2,
                fallback_attempts_per_candidate=1,
                max_joint_winding=max_joint_winding,
                max_ee_winding=max_ee_winding,
                target_pos=pos,
                target_quat=quat,
            )
            if retry.success and retry.winner is not None:
                planner = fresh
                planning = retry
                # Persist the rebuilt planner: the registry-injected mapping
                # is shared across every later skill call in this episode, so
                # a local-only swap never let a fresh OMPL state cure the
                # suspected degradation on subsequent calls.
                if isinstance(self.planner, dict):
                    self.planner[side] = fresh
                else:
                    self.planner = fresh
            else:
                # Last resort for grasp_center targets: the measured correction
                # path (align) has proven reliable in the simulator, and a
                # descending approach to the object is a short local move next
                # to the support surface.  Solve the goal once with multi-seed
                # IK preferring the branch with the most joint-limit margin and
                # execute it as a smooth trajectory, bypassing the OMPL planner
                # that has repeatedly timed out on this exact request.
                if target_frame == "grasp_center" and quat is not None:
                    fallback_result = _arm_move_to_margin_best(
                        self,
                        adapter,
                        scene,
                        goal_ee=pos,
                        quat=quat,
                        side=side,
                        speed_scale=float(trajectory_speed_scale),
                        exclude_objects=list(exclude_objects or ()),
                        step_hook=step_hook,
                    )
                    if fallback_result is not None:
                        return fallback_result
                if not solutions:
                    failed = kin.ik(pos, quat, q_init=q_cur)
                    position_only = kin.ik(pos, None, q_init=q_cur)
                    return SkillResult(
                        False,
                        self.name,
                        metrics={
                            "ik_error_m": float(failed.position_error),
                            "ik_tolerance_m": float(position_tolerance),
                            "rotation_error_rad": float(failed.rotation_error),
                            "rotation_tolerance_rad": float(orientation_tolerance),
                            "position_reachable_without_orientation": bool(
                                position_only.success and position_only.q_arm is not None
                            ),
                        },
                        details={
                            "reason": planning.reason,
                            "planning_status": planning.status,
                            "q_current": q_cur.round(5).tolist(),
                            "target_frame": target_frame,
                            "target_pos_base": [round(float(v), 5) for v in pos],
                        },
                    )
                solution = solutions[0]
                return SkillResult(
                    False,
                    self.name,
                    metrics={
                        "ik_error_m": float(solution.position_error),
                        "ik_tolerance_m": float(position_tolerance),
                        "rotation_error_rad": float(solution.rotation_error),
                        "rotation_tolerance_rad": float(orientation_tolerance),
                    },
                    details={
                        "reason": planning.reason,
                        "planning_status": planning.status,
                        "optimality_scope": planning.optimality_scope,
                        "planner_seed_controlled": planning.planner_seed_controlled,
                        "request_hash": planning.request_hash,
                        "failure_stage_summary": _summarize_failure_stages(planning.candidates),
                        "q_current": q_cur.round(5).tolist(),
                        "base_pose": [round(float(v), 5) for v in (base_pose or ())],
                        "planning_time": float(planning_time),
                        "speed_scale": float(trajectory_speed_scale),
                        "scene_objects": {
                            obj.name: [round(float(v), 5) for v in obj.pos]
                            for obj in getattr(live_scene, "objects", ())
                        },
                        "candidate_failures": [
                            {
                                "candidate_id": candidate.candidate_id,
                                "attempt_id": candidate.attempt_id,
                                "fallback": candidate.fallback,
                                "q_goal": np.asarray(candidate.q_goal).round(4).tolist(),
                                "status": candidate.planner_status,
                                "failure_stage": candidate.constraints.stage,
                                "reasons": list(candidate.constraints.reasons),
                                **{key: float(value) for key, value in candidate.metrics.items()},
                            }
                            for candidate in planning.candidates
                        ],
                    },
                )
        winner = planning.winner
        if 0 <= int(winner.candidate_id) < len(solutions):
            solution = solutions[winner.candidate_id]
        else:
            from types import SimpleNamespace

            reference = solutions[0] if solutions else None
            solution = SimpleNamespace(
                q_arm=np.asarray(winner.q_goal, dtype=float),
                position_error=float(getattr(reference, "position_error", 0.0)),
                rotation_error=float(getattr(reference, "rotation_error", 0.0)),
            )
        out = dict(winner.output or {})
        posture_cost = float(winner.metrics.get("posture_cost", 0.0))

        execution_obs = adapter.read_observation(0.0)
        execution_q = mplib_qpos_from_joint_positions(execution_obs.joint_positions)
        execution_base = execution_obs.base_pose or (0.0, 0.0, 0.0)
        execution_scene = runtime_scene_snapshot(
            scene,
            adapter,
            tuple(exclude_objects or ()),
        )
        robot_changed = float(np.max(np.abs(execution_q - full_q_current))) > 0.01
        base_changed = (
            float(
                np.linalg.norm(
                    np.asarray(execution_base[:2], dtype=float)
                    - np.asarray(base_pose[:2], dtype=float)
                )
            )
            > 0.005
            or abs(
                float(execution_base[2] if len(execution_base) > 2 else 0.0)
                - float(base_pose[2] if len(base_pose) > 2 else 0.0)
            )
            > 0.01
        )
        planned_objects = {
            obj.name: np.asarray(obj.pos, dtype=float)
            for obj in live_scene.objects
        }
        execution_objects = {
            obj.name: np.asarray(obj.pos, dtype=float)
            for obj in execution_scene.objects
        }
        scene_changed = planned_objects.keys() != execution_objects.keys() or any(
            float(np.linalg.norm(position - execution_objects[name])) > 0.005
            for name, position in planned_objects.items()
            if name in execution_objects
        )
        if robot_changed or base_changed or scene_changed:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "planning request became stale before execution",
                    "planning_status": "stale_scene",
                    "request_hash": planning.request_hash,
                    "robot_changed": robot_changed,
                    "base_changed": base_changed,
                    "scene_changed": scene_changed,
                },
            )
        # ``plan_arm_path`` already runs the hppfcl link-level proof for its
        # deterministic direct branch.  Re-running MPlib's environment check
        # on that branch is both redundant and very expensive in a room scene;
        # OMPL-produced paths still receive the independent final check.
        if out.get("status") not in {
            "DirectVerified",
            "RRTFallback",
            "RRTConnectVerified",
            "TaskSpaceVerified",
            "SequenceVerified",
        }:
            free, collision = path_collision_free(
                planner, out["position"], live_scene,
                base_xy=(float(base_pose[0]), float(base_pose[1])),
                base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
                dense=20, side=side, full_q_current=full_q_current,
            )
            if not free:
                return SkillResult(False, self.name, details={"reason": "final dense collision check failed", "collision": collision})

        stabilize_base(adapter)
        execution = ArmTrajectoryFollow(kin, vel_limits).execute(
            adapter,
            scene=live_scene,
            trajectory=np.asarray(out["position"]).tolist(),
            velocities=np.asarray(out["velocity"]).tolist() if out.get("velocity") is not None else None,
            trajectory_dt=float(out.get("dt", getattr(adapter, "dt", 1.0 / 60.0))),
            side=side,
            step_hook=step_hook,
        )
        recovery_mode: str | None = None
        if not execution.success:
            # A certified path can still outrun a compliant, gravity-loaded
            # joint.  Give the same geometric path one bounded, collision-
            # checked slow replay before reporting failure.  This is a generic
            # controller recovery, not a task-specific alternate target/IK
            # branch: no waypoint or safety exclusion is changed.
            tracking_error = max(
                float(execution.metrics.get("final_error_rad", 0.0)),
                float(execution.metrics.get("max_tracking_error_rad", 0.0)),
                float(execution.details.get("final_error_rad", 0.0)),
                float(execution.details.get("max_tracking_error_rad", 0.0)),
            )
            recovery_metrics: dict[str, Any] = {}
            if tracking_error > _FINAL_ERROR_TOL and hasattr(adapter, "read_observation"):
                from r1pro_data_gen.methods.collision import (
                    CollisionChecker,
                    check_path,
                    obstacles_from_scene,
                )

                checker = CollisionChecker(
                    kin,
                    obstacles_from_scene(live_scene, include_ground=True),
                )

                # If the direct certified branch is physically untrackable,
                # try the task-agnostic raised two-waypoint planner already
                # used for planning failures.  It preserves the same target,
                # orientation, exclusions and collision gates while giving a
                # loaded joint a different, measured approach posture.
                if target_frame == "grasp_center" and quat is not None:
                    staged = _arm_move_to_staged_fallback(
                        self,
                        adapter,
                        scene,
                        goal_ee=pos,
                        quat=quat,
                        side=side,
                        planning_time=float(planning_time),
                        local_radius_m=float(local_radius_m),
                        speed_scale=float(trajectory_speed_scale),
                        ik_candidates=max(4, int(ik_candidates)),
                        exclude_objects=list(exclude_objects or ()),
                        step_hook=step_hook,
                    )
                    if staged is not None and staged.success:
                        staged.skill = self.name
                        staged.metrics.update(
                            {
                                "tracking_recovery_attempts": 1.0,
                                "tracking_recovery_original_error_rad": float(tracking_error),
                            }
                        )
                        staged.details["tracking_recovery_mode"] = "raised_two_waypoint_path"
                        return staged

                def _collision_checked_replay(path: np.ndarray):
                    replay_obs = adapter.read_observation(0.0)
                    replay_q = np.asarray(
                        [replay_obs.joint_positions[j] for j in joints], dtype=float
                    )
                    replay_path = np.asarray(path, dtype=float)
                    if float(np.max(np.abs(replay_q - replay_path[0]))) > 1e-8:
                        replay_path = np.vstack([replay_q, replay_path])
                    replay_base = replay_obs.base_pose or base_pose
                    # The planner already certified the candidate path. The
                    # extra bridge starts at the measured post-failure pose.
                    free, _, link = check_path(
                        checker,
                        list(np.vstack([replay_q, np.asarray(path, dtype=float)])),
                        base_xy=(float(replay_base[0]), float(replay_base[1])),
                        base_yaw=float(replay_base[2]) if len(replay_base) > 2 else 0.0,
                        dense=20,
                    )
                    return replay_path, replay_base, free, link

                # The naturalness shortcut can erase a useful elbow detour:
                # the resulting direct joint path is collision-free but a
                # gravity-loaded joint may be unable to track it.  Replan once
                # from the measured state with shortcutting/direct fast-path
                # disabled.  This remains finite-budget, same-goal and fully
                # collision-certified; it is not a task-specific coordinate
                # repair.
                long_path_succeeded = False
                try:
                    from r1pro_data_gen.methods.manipulation.mplib_path import (
                        mplib_qpos_from_joint_positions,
                        plan_arm_path,
                    )

                    long_obs = adapter.read_observation(0.0)
                    long_q = np.asarray(
                        [long_obs.joint_positions[j] for j in joints], dtype=float
                    )
                    long_full_q = mplib_qpos_from_joint_positions(long_obs.joint_positions)
                    long_base = long_obs.base_pose or base_pose
                    long_out = plan_arm_path(
                        planner,
                        long_q,
                        np.asarray(solution.q_arm, dtype=float),
                        live_scene,
                        base_xy=(float(long_base[0]), float(long_base[1])),
                        base_yaw=float(long_base[2]) if len(long_base) > 2 else 0.0,
                        planning_time=float(planning_time),
                        kin=kin,
                        side=side,
                        local_radius_m=float(local_radius_m),
                        speed_scale=float(trajectory_speed_scale),
                        mplib_attempts=3,
                        allow_rrt_fallback=True,
                        rrt_connect_mode="second_opinion",
                        full_q_current=long_full_q,
                        shortcut_iterations=0,
                        direct_path=False,
                    )
                    recovery_metrics["tracking_recovery_long_path_planned"] = float(
                        bool(long_out.get("success"))
                    )
                    if long_out.get("success"):
                        long_path, _, long_free, long_link = _collision_checked_replay(
                            np.asarray(long_out["position"], dtype=float)
                        )
                        recovery_metrics["tracking_recovery_long_path_points"] = float(
                            len(long_path)
                        )
                        if long_free:
                            long_execution = ArmTrajectoryFollow(kin, vel_limits).execute(
                                adapter,
                                scene=live_scene,
                                trajectory=long_path.tolist(),
                                velocities=np.zeros_like(long_path).tolist(),
                                trajectory_dt=float(
                                    long_out.get("dt", getattr(adapter, "dt", 1.0 / 60.0))
                                ),
                                side=side,
                                step_hook=step_hook,
                            )
                            if long_execution.success:
                                execution = long_execution
                                out = dict(long_out)
                                execution.metrics.update(
                                    {
                                        "tracking_recovery_attempts": 1.0,
                                        "tracking_recovery_original_error_rad": float(tracking_error),
                                    }
                                )
                                recovery_mode = "uncut_verified_path"
                                long_path_succeeded = True
                            else:
                                recovery_metrics["tracking_recovery_long_path_final_error_rad"] = float(
                                    long_execution.metrics.get("final_error_rad", 0.0)
                                )
                        else:
                            recovery_metrics["tracking_recovery_long_path_blocked"] = True
                            recovery_metrics["tracking_recovery_long_path_collision"] = str(long_link or "unknown")
                except Exception as exc:  # pragma: no cover - defensive recovery
                    recovery_metrics["tracking_recovery_long_path_error"] = str(exc)

                # First try one other *already certified* planner candidate.
                # The winner is selected for geometric naturalness, but a
                # gravity-loaded joint may track a longer valid branch better.
                winner_key = (winner.candidate_id, winner.attempt_id, winner.fallback)
                alternatives = [
                    candidate
                    for candidate in planning.candidates
                    if candidate.valid
                    and (candidate.candidate_id, candidate.attempt_id, candidate.fallback)
                    != winner_key
                ]
                alternatives.sort(
                    key=lambda candidate: (
                        float(candidate.metrics.get("duration_s", 0.0)),
                        float(candidate.metrics.get("joint_winding", 0.0)),
                        float(candidate.metrics.get("ee_winding", 0.0)),
                    ),
                    reverse=True,
                )
                recovery_metrics["tracking_recovery_alternate_candidates"] = float(
                    len(alternatives)
                )
                alternate_failed = False
                if not long_path_succeeded and alternatives:
                    alternate = alternatives[0]
                    recovery_metrics["tracking_recovery_alternate_duration_s"] = float(
                        alternate.metrics.get("duration_s", 0.0)
                    )
                    alternate_out = dict(alternate.output or {})
                    alternate_path, alternate_base, alternate_free, alternate_link = _collision_checked_replay(
                        np.asarray(alternate_out.get("position", ()), dtype=float)
                    )
                    if alternate_free:
                        alternate_execution = ArmTrajectoryFollow(kin, vel_limits).execute(
                            adapter,
                            scene=live_scene,
                            trajectory=alternate_path.tolist(),
                            # Position-only references avoid replaying stale
                            # feed-forward velocities after the measured
                            # post-failure bridge was prepended.
                            velocities=np.zeros_like(alternate_path).tolist(),
                            trajectory_dt=float(
                                alternate_out.get("dt", getattr(adapter, "dt", 1.0 / 60.0))
                            ),
                            side=side,
                            step_hook=step_hook,
                        )
                        if alternate_execution.success:
                            execution = alternate_execution
                            execution.metrics.update(
                                {
                                    "tracking_recovery_attempts": 1.0,
                                    "tracking_recovery_alternate_duration_s": float(
                                        alternate.metrics.get("duration_s", 0.0)
                                    ),
                                    "tracking_recovery_original_error_rad": float(tracking_error),
                                }
                            )
                            out = alternate_out
                            winner = alternate
                            solution = solutions[alternate.candidate_id]
                            recovery_mode = "alternate_verified_path"
                        else:
                            alternate_failed = True
                    else:
                        alternate_failed = True
                if not execution.success and not long_path_succeeded:
                    retry_path = _slow_retime_same_path(np.asarray(out["position"], dtype=float))
                    retry_path, retry_base, free, collision_link = _collision_checked_replay(
                        retry_path
                    )
                    if free:
                        # Force the position-reference execution branch even
                        # when a planner did not provide velocities.
                        retry = ArmTrajectoryFollow(kin, vel_limits).execute(
                            adapter,
                            scene=live_scene,
                            trajectory=retry_path.tolist(),
                            velocities=np.zeros_like(retry_path).tolist(),
                            trajectory_dt=float(getattr(adapter, "dt", 1.0 / 60.0)),
                            side=side,
                            step_hook=step_hook,
                        )
                        recovery_metrics.update(
                            {
                                "tracking_recovery_attempts": 1.0,
                                "tracking_recovery_factor": float(_TRACKING_RECOVERY_FACTOR),
                                "tracking_recovery_original_error_rad": float(tracking_error),
                            }
                        )
                        if retry.success:
                            execution = retry
                            execution.metrics.update(recovery_metrics)
                            recovery_mode = "slow_same_path"
                        else:
                            recovery_metrics.update(
                                {
                                    "tracking_recovery_final_error_rad": float(
                                        retry.metrics.get("final_error_rad", 0.0)
                                    ),
                                    "tracking_recovery_max_tracking_error_rad": float(
                                        retry.metrics.get("max_tracking_error_rad", 0.0)
                                    ),
                                }
                            )
                            return SkillResult(
                                False,
                                self.name,
                                metrics={**dict(execution.metrics), **recovery_metrics},
                                details={
                                    "reason": "trajectory execution failed after bounded recovery",
                                    "tracking_recovery_collision_link": collision_link,
                                    "tracking_recovery_alternate_failed": alternate_failed,
                                    **execution.details,
                                },
                            )
                    else:
                        recovery_metrics.update(
                            {
                                "tracking_recovery_attempts": 0.0,
                                "tracking_recovery_factor": float(_TRACKING_RECOVERY_FACTOR),
                                "tracking_recovery_original_error_rad": float(tracking_error),
                                "tracking_recovery_blocked": True,
                            }
                        )
                        return SkillResult(
                            False,
                            self.name,
                            metrics={**dict(execution.metrics), **recovery_metrics},
                            details={
                                "reason": "trajectory execution failed; recovery rejected by collision gate",
                                "tracking_recovery_collision_link": collision_link,
                                **execution.details,
                            },
                        )
            if not execution.success:
                return SkillResult(
                    False,
                    self.name,
                    metrics={**dict(execution.metrics), **recovery_metrics},
                    details={"reason": "trajectory execution failed", **execution.details},
                )

        final_obs = adapter.read_observation(0.0)
        q_final = np.asarray([final_obs.joint_positions[j] for j in joints], dtype=float)
        if target_frame == "grasp_center":
            final_pos, final_quat = kin.grasp_center_fk(q_final)
            position_goal = requested_pos
        else:
            final_pos, final_quat = kin.fk(q_final)
            position_goal = pos
        pos_error = float(np.linalg.norm(final_pos - position_goal))
        rot_error = 0.0 if quat is None else float(np.linalg.norm(_quat_error(quat, final_quat)))
        success = pos_error <= float(position_tolerance) and rot_error <= float(orientation_tolerance)
        lock_metrics = adapter.joint_lock_metrics() if hasattr(adapter, "joint_lock_metrics") else {}
        return SkillResult(
            success,
            self.name,
            metrics={
                **execution.metrics,
                "ik_error_m": float(solution.position_error),
                "final_position_error_m": pos_error,
                "final_orientation_error_rad": rot_error,
                "duration_s": float(out.get("duration", 0.0)),
                "winding": float(out.get("winding", 0.0)),
                "ee_winding": float(out.get("ee_winding", 1.0)),
                "posture_cost": posture_cost,
                "ee_path_length_m": float(winner.metrics.get("ee_path_length_m", 0.0)),
                "normalized_joint_path_length": float(winner.metrics.get("normalized_joint_path_length", 0.0)),
                "smoothness_cost": float(winner.metrics.get("smoothness_cost", 0.0)),
                **lock_metrics,
            },
            details={
                "trajectory_points": len(out["position"]),
                "planner_status": out.get("status"),
                "target_frame": target_frame,
                "orientation_deferred": bool(orientation_deferred),
                "chosen_q_goal": np.asarray(solution.q_arm).round(5).tolist(),
                "winner_candidate_id": winner.candidate_id,
                "winner_attempt_id": winner.attempt_id,
                "optimality_scope": planning.optimality_scope,
                "planner_seed_controlled": planning.planner_seed_controlled,
                "request_hash": planning.request_hash,
                "candidate_count": len(planning.candidates),
                **(
                    {"tracking_recovery_mode": recovery_mode}
                    if recovery_mode is not None
                    else {}
                ),
                "reason": "goal reached" if success else "final target-frame tolerance failed",
            },
        )


# Staging retry height above a grasp_center target.  Large enough to clear the
# tabletop so the descent is short and collision-free, small enough to stay
# inside the reachable workspace.
_STAGING_RETRY_LIFT_M = 0.30

# A long close-approach target can be reachable in pose space but poorly
# tracked by the calibrated EE model near a support surface.  Replanning short
# measured-midpoint segments keeps the live physical reference authoritative.
_MEASURED_DIRECTIONAL_CHUNK_M = 0.015

# Maximum Cartesian correction proposed by one arm_align_gripper iteration.
# The actual executable step is still bounded by the local joint-continuity
# certificate below: if this proposal has no bounded IK/path branch, the
# caller retries the same measured correction with a smaller step.  A single
# fixed 2 cm cap made a normal tabletop descent require many expensive
# simulator actions, while allowing the full correction to reach IK branch
# jumps.  Adaptive subdivision preserves the safety contract and keeps the
# number of physical corrections practical for different robot postures.
_ALIGN_MAX_STEP_M = 0.10
_ALIGN_MIN_STEP_M = 0.005
_ALIGN_STEP_RETRY_LIMIT = 4

# A full jaw reorientation can be reachable in the reduced kinematic model but
# still demand a large instantaneous wrist/forearm change from the physical
# position drive.  Treat jaw direction as a progressive acquisition variable:
# this is a robot-control capability bound, independent of any scene geometry.
_ALIGN_MAX_WINDOW_DIRECTION_STEP_RAD = math.radians(15.0)
# The safe outer-clearance reorientation may need a larger *joint-space*
# branch change near a redundant-arm singularity even when the requested jaw
# direction changes by only a few degrees.  This is allowed only for the
# clearance phase; every trajectory is still swept-volume and effort checked,
# while the final object-window segment keeps the tighter local branch bound.
_ALIGN_MAX_CLEARANCE_ORIENTATION_JOINT_STEP_RAD = 1.20

# Vertical margin (m) for "aligned" when the object is approached from above.
# The finger-link origin sits on the upper part of the finger mesh, so a
# correct pinch over a tabletop object ends with the finger midpoint several
# cm above the object centre (the finger top touches the object top).  A strict
# |dz| <= tolerance would never be satisfied and would push the object trying
# to descend into it.  The margin must still be small enough that the fingers
# reach the object's side (grasp height), otherwise gripper_grasp closes on
# empty space above the object: half the object height (~0.06 m for the
# standard cylinder) plus a couple of cm of mesh offset is the useful band.
_ALIGN_ABOVE_OBJECT_MARGIN_M = 0.05

# Contact force (N) at which a continuous alignment move stops: it has already
# reached the object, so pushing further would only shove it.
_ALIGN_CONTACT_STOP_N = 0.15

# The complete trajectory is already statically swept/certified before it is
# sent to the simulator.  Re-running the full Pinocchio + live mesh check
# twice for every 60-Hz sample made a short measured correction dominate the
# action budget.  Keep periodic live checks (plus both endpoints); contact
# sensors and the object-motion guard still run at every physics step.
_ALIGN_LIVE_GEOMETRY_CHECK_STRIDE = 4


def _ik_with_retry(
    kin: Any,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_seed: np.ndarray,
) -> np.ndarray | None:
    """Solve online IK for ``target_pos``, retrying from the natural posture.

    The continuous current-branch solver can hit a joint limit or singular
    configuration near the end of a long correction.  Re-seeding from the
    relaxed reach posture returns a different redundant branch instead of
    failing the whole alignment.
    """
    seeds = [np.asarray(q_seed, dtype=float)]
    natural = getattr(kin, "natural_reach_q", None)
    if natural is not None:
        seeds.append(np.asarray(natural, dtype=float))
    for seed in seeds:
        if hasattr(kin, "_ik_once"):
            solution = kin._ik_once(
                np.asarray(target_pos, dtype=float),
                np.asarray(target_quat, dtype=float),
                seed,
                pos_tol=0.003,
                rot_tol=0.02,
            )
        else:
            solution = kin.ik(np.asarray(target_pos, dtype=float), np.asarray(target_quat, dtype=float), q_init=seed)
        if solution.success and solution.q_arm is not None:
            return np.asarray(solution.q_arm, dtype=float)
    return None


def _align_continuous_move(
    kin: Any,
    adapter: Any,
    side: str,
    start_q: np.ndarray,
    target_ee: np.ndarray,
    speed_scale: float,
    step_hook: Callable[[], None] | None,
    scene: Any = None,
    exclude_objects: tuple[str, ...] = (),
    object_motion_guard: Callable[[], dict[str, Any] | None] | None = None,
    target_quat: np.ndarray | None = None,
    target_center_world: np.ndarray | None = None,
    protected_object_name: str | None = None,
    ik_candidates: int = 8,
    direction_span_override: np.ndarray | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Move the EE to ``target_ee`` with bounded, certified IK branches.

    Used for the short measured recenter/descent of ``arm_align_gripper``.
    Solving the whole correction once with bounded multi-seed IK (preferring
    continuity) avoids the accumulation toward joint limits that per-frame
    stepping exhibits on a sideways reach.  Every branch is still checked by
    the swept-volume certificate before it can execute.  The branch budget is
    supplied by the public alignment skill so orientation/solver sources do
    not multiply into an unbounded search.  No MPlib and no OMPL.
    """
    from r1pro_data_gen.methods.manipulation.mplib_path import _minimum_jerk_trajectory
    from r1pro_data_gen.robot.robot_config import (
        R1PRO_ALIGNMENT_MAX_LOCAL_JOINT_STEP_RAD,
        R1PRO_ALIGNMENT_MIN_PHASE_S,
    )

    joints = ARM_JOINTS_BY_SIDE[side]
    start_q = np.asarray(start_q, dtype=float)
    target_ee = np.asarray(target_ee, dtype=float)
    base_pose: Any = (0.0, 0.0, 0.0)
    opening_offsets: tuple[np.ndarray, np.ndarray] | None = None
    target_center_model: np.ndarray | None = None
    model_to_world_rotation: np.ndarray | None = None
    model_to_world_translation: np.ndarray | None = None
    calibration_rms: float | None = None
    try:
        observation = adapter.read_observation(0.0)
        _sync_kinematics_auxiliary_q(kin, observation)
        base_pose = getattr(observation, "base_pose", None) or (0.0, 0.0, 0.0)
        # Arm IK freezes the two prismatic finger joints at zero. Measure the
        # live open-jaw offsets once and use the same fixed offsets in the
        # window objective and in the swept-volume certificate below.
        opening_offsets = _alignment_finger_opening_offsets(
            kin,
            adapter,
            side,
            start_q,
            base_pose,
            None,
            None,
        )
        if (
            target_center_world is not None
            and hasattr(adapter, "end_effector_poses")
            and hasattr(adapter, "body_position")
        ):
            mapped = _calibrated_model_center_target(
                kin,
                adapter,
                side,
                observation,
                start_q,
                np.asarray(target_center_world, dtype=float),
            )
            if mapped is not None:
                (
                    target_center_model,
                    calibration_rms,
                    model_to_world_rotation,
                    model_to_world_translation,
                ) = mapped
                measured_opening_offsets = _alignment_finger_opening_offsets(
                    kin,
                    adapter,
                    side,
                    start_q,
                    base_pose,
                    model_to_world_rotation,
                    model_to_world_translation,
                )
                if measured_opening_offsets is not None:
                    opening_offsets = measured_opening_offsets
                # Registration aligns the rigid arm links, but the loaded
                # gripper can retain a configuration-dependent tool residual.
                # Anchor this short correction at the measured physical
                # midpoint and apply only the requested world displacement;
                # an absolute nominal-URDF target can otherwise overshoot when
                # the wrist reorients during a floor-level grasp.
                measured_center = _measured_grasp_center_world(adapter, side)
                try:
                    if opening_offsets is not None and hasattr(
                        kin, "finger_geometry_fk"
                    ):
                        current_center_model = np.asarray(
                            kin.finger_geometry_fk(start_q, opening_offsets)[4],
                            dtype=float,
                        )
                    else:
                        current_center_model = np.asarray(
                            kin.grasp_center_fk(start_q)[0], dtype=float
                        )
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    current_center_model = None
                if (
                    measured_center is not None
                    and current_center_model is not None
                    and np.all(np.isfinite(current_center_model))
                ):
                    target_center_model = current_center_model + np.asarray(
                        model_to_world_rotation, dtype=float
                    ).T @ (
                        np.asarray(target_center_world, dtype=float)
                        - measured_center
                    )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    _, quat0 = kin.fk(start_q)
    # Prefer the calibrated vertical grasp orientation over the current
    # posture's orientation.  A position-only pre-grasp (used when the exact
    # grasp pose was outside the IK basin) can leave the wrist at an arbitrary
    # yaw; keeping that orientation through alignment makes the gripper pinch
    # the object sideways instead of from above.  The grasp orientation is the
    # robot capability, and a short measured correction that reaches it is the
    # goal of alignment.
    from r1pro_data_gen.robot.robot_config import R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE
    from r1pro_data_gen.robot.kinematics import (
        GRASP_WINDOW_IK_DIRECTION_TOL,
        GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD,
    )

    grasp_quat = np.asarray(R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE[side], dtype=float)
    geometry_span_world = _object_window_span_direction(
        adapter,
        side,
        protected_object_name,
    )
    if direction_span_override is not None:
        override = np.asarray(direction_span_override, dtype=float)
        if (
            override.shape == (3,)
            and np.all(np.isfinite(override))
            and float(np.linalg.norm(override[:2])) > 1.0e-8
        ):
            # The first measured object-to-midpoint approach defines one
            # local acquisition direction.  Keep that direction fixed for
            # the whole alignment invocation; recomputing it after every
            # imperfect redundant-arm translation can make the target yaw
            # chase the current error and repeatedly retreat the gripper.
            geometry_span_world = override.copy()
    orientation_candidates = []
    # A position-only whole-body pregrasp is intentionally allowed to choose
    # any redundant wrist posture.  At the final acquisition boundary that
    # posture is not neutral: the object can arrive at a finger endpoint while
    # the midpoint is still far away.  Derive a yaw correction from the live
    # finger segment and the live object direction, so the object is approached
    # through the middle of the jaw rather than through one fingertip.  This is
    # a robot-geometry capability; it contains no scene coordinate or object
    # name and is used only when the physical adapter exposes the measurements.
    geometry_quat = _object_window_orientation_candidate(
        kin,
        adapter,
        side,
        start_q,
        model_to_world_rotation=model_to_world_rotation,
        protected_object_name=protected_object_name,
        desired_span_world=geometry_span_world,
    )
    for q_candidate in (geometry_quat, target_quat, grasp_quat, quat0):
        if q_candidate is None:
            continue
        q_norm = q_candidate / np.linalg.norm(q_candidate)
        if not any(
            abs(float(np.dot(q_norm, existing))) > 1.0 - 1e-7
            for existing in orientation_candidates
        ):
            orientation_candidates.append(q_norm)
    # Keep every bounded local branch.  A single continuity-best branch can
    # still sweep the palm through the object; the executable path certificate
    # below is the authority for this measured correction.  The candidate
    # count is bounded by each backend's own IK call, so this remains a local
    # correction rather than an unbounded search.
    branch_limit = max(1, min(int(ik_candidates), 8))
    candidate_options: list[tuple[np.ndarray, float, float, bool, bool]] = []
    local_branch_rejections: list[dict[str, Any]] = []
    max_local_joint_step = float(R1PRO_ALIGNMENT_MAX_LOCAL_JOINT_STEP_RAD)
    # A staged correction is a sequence of local branches.  Once a clearance
    # or orientation waypoint has been selected, the next IK branch must be
    # measured from that waypoint, not from the original pre-grasp posture.
    # Comparing every later goal to ``start_q`` incorrectly discards a valid
    # multi-stage path whose total displacement is larger than one local step.
    candidate_reference_q = np.asarray(start_q, dtype=float).copy()

    def append_candidates(
        solutions: Any,
        *,
        orientation_relaxed: bool,
        center_position_ik: bool,
    ) -> None:
        for q, continuity, margin in _rank_continuous_ik_solutions(
            kin, solutions, candidate_reference_q
        ):
            max_joint_delta = float(np.max(np.abs(q - candidate_reference_q)))
            if max_joint_delta > max_local_joint_step:
                local_branch_rejections.append(
                    {
                        "max_joint_delta_rad": max_joint_delta,
                        "limit_rad": max_local_joint_step,
                    "q_goal": np.asarray(q, dtype=float).round(5).tolist(),
                    "reference_q": np.asarray(
                        candidate_reference_q, dtype=float
                    ).round(5).tolist(),
                }
            )
                continue
            # The same branch can be returned by the center and orientation
            # solvers.  Preserve the first (most semantically specific) entry
            # and avoid replaying an identical trajectory.
            if any(float(np.linalg.norm(q - existing[0])) < 1.0e-6 for existing in candidate_options):
                continue
            candidate_options.append(
                (
                    np.asarray(q, dtype=float),
                    float(continuity),
                    float(margin),
                    bool(orientation_relaxed),
                    bool(center_position_ik),
                )
            )

    # Candidate generation and path certification are intentionally lazy.  A
    # measured correction normally has a continuous local solution; generating
    # and certifying every orientation/solver branch up front made the first
    # correction safe but could spend the entire action budget on alternatives
    # before the next measured correction was attempted.  The phases below
    # preserve the generic fallback order while only expanding after the
    # preceding phase has been rejected by the geometry certificate.
    path_rejections: list[dict[str, Any]] = []
    telemetry_adapter = getattr(adapter, "_adapter", adapter)
    orientation_clearance_m: float | None = None
    window_geometry_details: dict[str, Any] = {}

    def mark_planning_phase(phase: str) -> None:
        try:
            telemetry_adapter._alignment_planning_phase = str(phase)
            telemetry_adapter._alignment_planning_candidate_count = int(
                len(candidate_options)
            )
        except (AttributeError, TypeError, ValueError):
            pass

    def execute_options(
        options: list[tuple[np.ndarray, float, float, bool, bool]],
        standoff_q: np.ndarray | None = None,
        clearance_q: np.ndarray | None = None,
        orientation_waypoints: tuple[np.ndarray, ...] = (),
    ) -> tuple[bool | None, dict[str, Any] | None]:
        """Try only this phase's new candidates, then report its outcome."""
        orientation_stages = tuple(
            np.asarray(item, dtype=float) for item in orientation_waypoints
        )
        if standoff_q is not None:
            orientation_stages = orientation_stages + (
                np.asarray(standoff_q, dtype=float),
            )
        staged_orientation = bool(orientation_stages)
        for (
            q_goal,
            continuity,
            margin,
            candidate_orientation_relaxed,
            candidate_center_position_ik,
        ) in options:
            mark_planning_phase("candidate_path_and_execution")
            # Build every segment explicitly. A low-support acquisition must
            # not rotate the open jaw while its center is crossing the object:
            # retreat first, apply bounded orientation waypoints in free space,
            # and only then translate to the final window with the solved jaw
            # direction fixed.
            stage_points: list[np.ndarray] = []
            if clearance_q is not None:
                stage_points.append(np.asarray(clearance_q, dtype=float))
            stage_points.extend(orientation_stages)
            stage_points.append(np.asarray(q_goal, dtype=float))
            trajectory_parts: list[np.ndarray] = []
            previous = np.asarray(start_q, dtype=float)
            for stage in stage_points:
                segment, _, _ = _minimum_jerk_trajectory(
                    np.asarray([previous, stage]),
                    speed_scale=float(speed_scale),
                    side=side,
                    # A short correction still changes several gravity-loaded
                    # joints at once. The longer minimum phase gives the
                    # position drive time to follow the C2 reference without
                    # a damping spike at the branch endpoint; effort and
                    # physical collision gates remain active at every step.
                    min_duration_s=R1PRO_ALIGNMENT_MIN_PHASE_S,
                )
                trajectory_parts.append(segment)
                previous = stage
            trajectory = trajectory_parts[0]
            for segment in trajectory_parts[1:]:
                trajectory = np.vstack((trajectory, segment[1:]))
            try:
                executed = _execute_alignment_trajectory(
                    kin, adapter, side, trajectory, start_q, step_hook,
                    scene=scene,
                    exclude_objects=exclude_objects,
                    object_motion_guard=object_motion_guard,
                    model_to_world_rotation=model_to_world_rotation,
                    model_to_world_translation=model_to_world_translation,
                    protected_object_name=protected_object_name,
                )
            except _ObjectMovedBeforeGrasp as exc:
                return False, {
                    "reason": "movable object shifted before attachment",
                    "failure_code": "object_moved_before_grasp",
                    "goal_margin_rad": margin,
                    "continuity_cost": continuity,
                    "orientation_relaxed": candidate_orientation_relaxed,
                    "grasp_center_position_ik": candidate_center_position_ik,
                    "alignment_candidate_count": len(candidate_options),
                    "alignment_ik_branch_limit": branch_limit,
                    "alignment_local_joint_step_limit_rad": max_local_joint_step,
                    "alignment_local_branch_rejected_count": len(local_branch_rejections),
                    **exc.details,
                }
            except _AlignmentCollisionDetected as exc:
                return False, {
                    "reason": "live non-finger collision detected during alignment",
                    "failure_code": "alignment_collision_detected",
                    "goal_margin_rad": margin,
                    "continuity_cost": continuity,
                    "orientation_relaxed": candidate_orientation_relaxed,
                    "grasp_center_position_ik": candidate_center_position_ik,
                    "alignment_candidate_count": len(candidate_options),
                    "alignment_ik_branch_limit": branch_limit,
                    "alignment_local_joint_step_limit_rad": max_local_joint_step,
                    "alignment_local_branch_rejected_count": len(local_branch_rejections),
                    **exc.details,
                }
            except _AlignmentCollisionCheckUnavailable as exc:
                return False, {
                    "reason": "live alignment collision certificate unavailable",
                    "failure_code": "alignment_collision_check_unavailable",
                    "goal_margin_rad": margin,
                    "continuity_cost": continuity,
                    "orientation_relaxed": candidate_orientation_relaxed,
                    "grasp_center_position_ik": candidate_center_position_ik,
                    "alignment_candidate_count": len(candidate_options),
                    "alignment_ik_branch_limit": branch_limit,
                    "alignment_local_joint_step_limit_rad": max_local_joint_step,
                    "alignment_local_branch_rejected_count": len(local_branch_rejections),
                    **exc.details,
                }
            if executed:
                return True, {
                    "goal_margin_rad": margin,
                    "continuity_cost": continuity,
                    "orientation_relaxed": candidate_orientation_relaxed,
                    "grasp_center_position_ik": candidate_center_position_ik,
                    "calibration_rms_m": calibration_rms,
                    "alignment_candidate_count": len(candidate_options),
                    "alignment_path_rejected_count": len(path_rejections),
                    "alignment_ik_branch_limit": branch_limit,
                    "alignment_local_joint_step_limit_rad": max_local_joint_step,
                    "alignment_local_branch_rejected_count": len(local_branch_rejections),
                    "staged_orientation": staged_orientation,
                    "staged_clearance": clearance_q is not None,
                    "orientation_clearance_m": orientation_clearance_m,
                    "window_geometry": _json_safe_path_details(
                        window_geometry_details
                    ),
                }
            path_rejections.append(
                {
                    "q_goal": np.asarray(q_goal, dtype=float).round(5).tolist(),
                    "trajectory_sample_count": int(len(trajectory)),
                    "goal_margin_rad": margin,
                    "continuity_cost": continuity,
                    "orientation_relaxed": candidate_orientation_relaxed,
                    "grasp_center_position_ik": candidate_center_position_ik,
                    "staged_orientation": staged_orientation,
                    "staged_clearance": clearance_q is not None,
                    "orientation_clearance_m": orientation_clearance_m,
                    "window_geometry": _json_safe_path_details(
                        window_geometry_details
                    ),
                    "clearance_q": (
                        None
                        if clearance_q is None
                        else np.asarray(clearance_q, dtype=float).round(5).tolist()
                    ),
                    "standoff_q": (
                        None
                        if standoff_q is None
                        else np.asarray(standoff_q, dtype=float).round(5).tolist()
                    ),
                    "alignment_local_joint_step_limit_rad": max_local_joint_step,
                    "certificate": _json_safe_path_details(
                        getattr(
                            telemetry_adapter,
                            "_alignment_last_path_rejection",
                            None,
                        )
                    ),
                }
            )
        return None, None

    def append_and_execute(
        solutions: Any,
        *,
        orientation_relaxed: bool,
        center_position_ik: bool,
        phase: str,
        standoff_q: np.ndarray | None = None,
        clearance_q: np.ndarray | None = None,
        orientation_waypoints: tuple[np.ndarray, ...] = (),
    ) -> tuple[bool | None, dict[str, Any] | None]:
        before = len(candidate_options)
        append_candidates(
            solutions,
            orientation_relaxed=orientation_relaxed,
            center_position_ik=center_position_ik,
        )
        if len(candidate_options) == before:
            return None, None
        mark_planning_phase(phase)
        return execute_options(
            candidate_options[before:],
            standoff_q=standoff_q,
            clearance_q=clearance_q,
            orientation_waypoints=orientation_waypoints,
        )

    def execute_live_orientation_sequence(
        q_initial: np.ndarray,
        q_clearance: np.ndarray,
        desired_span_world: np.ndarray,
    ) -> tuple[bool, dict[str, Any]]:
        """Rotate the open jaw with measured re-planning at every waypoint.

        The arm-link registration is fitted from live frames, but a loaded
        gripper can still have a configuration-dependent tool residual.  A
        precomputed multi-waypoint orientation path can therefore be
        geometrically safe in the reduced model while the USD finger boxes
        drift into an object in the real stage.  Execute the clearance move
        once, then re-read the physical finger segment and re-solve one small
        direction step at a time.  Each segment gets a fresh registration,
        opening-offset measurement, static certificate, and live collision
        gate.  The returned state is ready for a separately certified final
        window approach.
        """
        from r1pro_data_gen.robot.kinematics import (
            GRASP_WINDOW_IK_DIRECTION_TOL,
            GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD,
        )

        if target_center_world is None or not hasattr(adapter, "body_position"):
            return False, {
                "failure_code": "alignment_orientation_unavailable",
                "reason": "live orientation measurements are unavailable",
            }

        def read_state() -> dict[str, Any] | None:
            try:
                observation = adapter.read_observation(0.0)
                _sync_kinematics_auxiliary_q(kin, observation)
                q_actual = np.asarray(
                    [
                        observation.joint_positions[name]
                        for name in ARM_JOINTS_BY_SIDE[side]
                    ],
                    dtype=float,
                )
                base_pose_actual = getattr(observation, "base_pose", None) or (
                    0.0,
                    0.0,
                    0.0,
                )
                measured_center = _measured_grasp_center_world(adapter, side)
                if measured_center is None:
                    return None
                rotation_actual = model_to_world_rotation
                translation_actual = model_to_world_translation
                rms_actual = calibration_rms
                if hasattr(kin, "calibrated_base_transform"):
                    mapped = _calibrated_model_center_target(
                        kin,
                        adapter,
                        side,
                        observation,
                        q_actual,
                        measured_center,
                    )
                    if mapped is not None:
                        _, rms_actual, rotation_actual, translation_actual = mapped
                if rotation_actual is None or translation_actual is None:
                    rotation_actual, translation_actual = _alignment_world_model_transform(
                        base_pose_actual,
                        None,
                        None,
                    )
                offsets_actual = _alignment_finger_opening_offsets(
                    kin,
                    adapter,
                    side,
                    q_actual,
                    base_pose_actual,
                    rotation_actual,
                    translation_actual,
                )
                if offsets_actual is None:
                    offsets_actual = opening_offsets
                if offsets_actual is not None and hasattr(kin, "finger_geometry_fk"):
                    center_model_actual = np.asarray(
                        kin.finger_geometry_fk(q_actual, offsets_actual)[4],
                        dtype=float,
                    )
                else:
                    center_model_actual = np.asarray(
                        kin.grasp_center_fk(q_actual)[0],
                        dtype=float,
                    )
                if (
                    center_model_actual.shape != (3,)
                    or not np.all(np.isfinite(center_model_actual))
                    or rotation_actual is None
                    or translation_actual is None
                    or offsets_actual is None
                ):
                    return None
                return {
                    "q": q_actual,
                    "base_pose": base_pose_actual,
                    "center_model": center_model_actual,
                    "center_world": np.asarray(measured_center, dtype=float),
                    "rotation": np.asarray(rotation_actual, dtype=float),
                    "translation": np.asarray(translation_actual, dtype=float),
                    "offsets": offsets_actual,
                    "calibration_rms": rms_actual,
                }
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                return None

        def direction_angle(
            current_span: np.ndarray,
            desired_span: np.ndarray,
        ) -> float:
            current_vector = np.asarray(current_span, dtype=float)
            desired_vector = np.asarray(desired_span, dtype=float)
            current_norm = float(np.linalg.norm(current_vector))
            desired_norm = float(np.linalg.norm(desired_vector))
            if (
                current_vector.shape != (3,)
                or desired_vector.shape != (3,)
                or not np.all(np.isfinite(current_vector))
                or not np.all(np.isfinite(desired_vector))
                or current_norm <= 1.0e-8
                or desired_norm <= 1.0e-8
            ):
                return math.pi
            current_unit = current_vector / current_norm
            desired_unit = desired_vector / desired_norm
            # The two finger endpoints are interchangeable, hence the
            # absolute dot product. Height skew is not interchangeable: an XY
            # yaw-only comparison hid a lower finger that entered contact well
            # before its mate. The desired span is horizontal, so this full
            # 3-D angle levels the jaw without constraining the full wrist pose.
            return math.acos(
                float(
                    np.clip(
                        abs(float(np.dot(current_unit, desired_unit))),
                        -1.0,
                        1.0,
                    )
                )
            )

        def run_segment(
            state: dict[str, Any],
            q_goal: np.ndarray,
            role: str,
        ) -> tuple[bool, dict[str, Any]]:
            q_start_actual = np.asarray(state["q"], dtype=float)
            q_goal = np.asarray(q_goal, dtype=float)
            trajectory, _, _ = _minimum_jerk_trajectory(
                np.asarray([q_start_actual, q_goal], dtype=float),
                speed_scale=float(speed_scale),
                side=side,
                min_duration_s=R1PRO_ALIGNMENT_MIN_PHASE_S,
            )
            mark_planning_phase(f"live_{role}_path_and_execution")
            try:
                executed = _execute_alignment_trajectory(
                    kin,
                    adapter,
                    side,
                    trajectory,
                    q_start_actual,
                    step_hook,
                    scene=scene,
                    exclude_objects=exclude_objects,
                    object_motion_guard=object_motion_guard,
                    model_to_world_rotation=state["rotation"],
                    model_to_world_translation=state["translation"],
                    protected_object_name=protected_object_name,
                )
            except _ObjectMovedBeforeGrasp as exc:
                return False, {
                    "failure_code": "object_moved_before_grasp",
                    "reason": "movable object shifted before attachment",
                    **exc.details,
                }
            except _AlignmentCollisionDetected as exc:
                return False, {
                    "failure_code": "alignment_collision_detected",
                    "reason": "live non-finger collision detected during alignment",
                    **exc.details,
                }
            except _AlignmentCollisionCheckUnavailable as exc:
                return False, {
                    "failure_code": "alignment_collision_check_unavailable",
                    "reason": "live alignment collision certificate unavailable",
                    **exc.details,
                }
            if not executed:
                return False, {
                    "failure_code": "alignment_path_unavailable",
                    "reason": "live orientation segment was rejected",
                    "certificate": _json_safe_path_details(
                        getattr(
                            telemetry_adapter,
                            "_alignment_last_path_rejection",
                            None,
                        )
                    ),
                }
            # A position drive may still be converging when the trajectory's
            # final command is sent.  Reading the jaw immediately at that
            # instant can feed a transient tool direction into the next IK
            # solve and make an otherwise reachable progressive rotation look
            # discontinuous.  Hold the last command for a short bounded
            # settling window, with the same live collision/object-motion
            # gates active during the hold.
            for _ in range(18):
                adapter.step()
                if step_hook is not None:
                    step_hook()
                live_free, live_details = _live_alignment_body_collision(
                    adapter,
                    scene,
                    side,
                    protected_object_name,
                    exclude_objects,
                )
                if not live_free:
                    if not bool(live_details.get("checked", False)):
                        raise _AlignmentCollisionCheckUnavailable(live_details)
                    raise _AlignmentCollisionDetected(live_details)
                finger_free, finger_details = _live_alignment_finger_collision(
                    adapter,
                    scene,
                    side,
                    protected_object_name,
                    exclude_objects,
                )
                if not finger_free and not _live_alignment_window_ready(
                    adapter, protected_object_name, side
                ):
                    if not bool(finger_details.get("checked", False)):
                        raise _AlignmentCollisionCheckUnavailable(finger_details)
                    raise _AlignmentCollisionDetected(finger_details)
                if object_motion_guard is not None:
                    violation = object_motion_guard()
                    if violation is not None:
                        raise _ObjectMovedBeforeGrasp(violation)
            refreshed = read_state()
            if refreshed is None:
                return False, {
                    "failure_code": "alignment_orientation_unavailable",
                    "reason": "live orientation state could not be refreshed",
                }
            return True, refreshed

        state = read_state()
        if state is None:
            return False, {
                "failure_code": "alignment_orientation_unavailable",
                "reason": "live orientation state could not be read",
            }
        if float(np.max(np.abs(np.asarray(q_clearance) - state["q"]))) > 1.0e-5:
            ok, segment_details = run_segment(state, q_clearance, "clearance")
            if not ok:
                return False, segment_details
            state = segment_details

        angle_history: list[float] = []
        max_orientation_steps = 12
        support_lift_retry_count = 0
        support_lift_attempts: list[dict[str, Any]] = []
        for step_index in range(1, max_orientation_steps + 1):
            try:
                p1 = np.asarray(
                    adapter.body_position(f"{side}_gripper_finger_link1"),
                    dtype=float,
                )
                p2 = np.asarray(
                    adapter.body_position(f"{side}_gripper_finger_link2"),
                    dtype=float,
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                return False, {
                    "failure_code": "alignment_orientation_unavailable",
                    "reason": "live finger direction could not be read",
                }
            live_span = p2 - p1
            desired_live = _object_window_span_direction(
                adapter,
                side,
                protected_object_name,
            )
            if desired_live is None:
                desired_live = np.asarray(desired_span_world, dtype=float)
            angle = direction_angle(live_span, desired_live)
            angle_history.append(float(angle))
            if angle <= float(GRASP_WINDOW_IK_DIRECTION_TOL):
                break
            q_next = None
            selected_substep_rad: float | None = None
            orientation_solver_counts: list[int] = []
            used_full_direction_fallback = False
            # A 15-degree geometric step can still map to a large joint-space
            # change near a redundant-arm singularity.  Retry the same live
            # correction with smaller angular targets before relaxing the
            # local branch bound; the swept-volume and effort gates remain
            # unchanged for every accepted segment.
            substep_limits = (
                _ALIGN_MAX_WINDOW_DIRECTION_STEP_RAD,
                math.radians(10.0),
                math.radians(5.0),
                math.radians(3.0),
            )
            for substep_limit in substep_limits:
                direction_target = _object_window_direction_step(
                    adapter,
                    side,
                    protected_object_name,
                    np.asarray(desired_live, dtype=float),
                    max_step_rad=substep_limit,
                )
                if direction_target is None:
                    direction_target = np.asarray(desired_live, dtype=float)
                try:
                        direction_solutions = kin.ik_grasp_window_candidates(
                            state["center_model"],
                            direction_target,
                            state["q"],
                            max_candidates=branch_limit,
                            direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                            opening_offsets=state["offsets"],
                            span_to_constraint_rotation=state["rotation"],
                        )
                except TypeError:
                    # Lightweight compatibility backends may not expose the
                    # optional world-direction argument; production R1Pro uses
                    # the calibrated branch above.
                        direction_solutions = kin.ik_grasp_window_candidates(
                            state["center_model"],
                            direction_target,
                            state["q"],
                            max_candidates=branch_limit,
                            direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                            opening_offsets=state["offsets"],
                        )
                except (
                    AttributeError,
                    RuntimeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    direction_solutions = []
                orientation_solver_counts.append(len(direction_solutions))
                q_candidate = choose_local_window_solution(
                    direction_solutions,
                    np.asarray(state["q"], dtype=float),
                    "live_orientation",
                    require_progress=True,
                    max_step_rad=_ALIGN_MAX_CLEARANCE_ORIENTATION_JOINT_STEP_RAD,
                )
                if q_candidate is not None:
                    q_next = q_candidate
                    selected_substep_rad = float(substep_limit)
                    break
            if q_next is None:
                # The reduced arm manifold can have a discontinuous inverse
                # image for an intermediate yaw: the small target may expose
                # only a vertical redundancy branch, while the complete live
                # direction has a nearby usable branch.  Try that complete
                # target once, still using the same bounded local-joint,
                # tilt, support, collision, and effort gates. This is a
                # generic finite-budget replan, not a task-specific jump.
                full_direction_target = np.asarray(desired_live, dtype=float)
                try:
                    full_direction_solutions = kin.ik_grasp_window_candidates(
                        state["center_model"],
                        full_direction_target,
                        state["q"],
                        max_candidates=branch_limit,
                        direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                        opening_offsets=state["offsets"],
                        span_to_constraint_rotation=state["rotation"],
                    )
                except TypeError:
                    full_direction_solutions = kin.ik_grasp_window_candidates(
                        state["center_model"],
                        full_direction_target,
                        state["q"],
                        max_candidates=branch_limit,
                        direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                        opening_offsets=state["offsets"],
                    )
                except (
                    AttributeError,
                    RuntimeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    full_direction_solutions = []
                orientation_solver_counts.append(len(full_direction_solutions))
                q_candidate = choose_local_window_solution(
                    full_direction_solutions,
                    np.asarray(state["q"], dtype=float),
                    "live_orientation_full_direction",
                    require_progress=True,
                    max_step_rad=_ALIGN_MAX_CLEARANCE_ORIENTATION_JOINT_STEP_RAD,
                )
                if q_candidate is not None:
                    q_next = q_candidate
                    used_full_direction_fallback = True
            if q_next is None:
                return False, {
                    "failure_code": "alignment_orientation_ik_failed",
                    "reason": "no bounded live orientation branch was found",
                    "orientation_step_index": step_index,
                    "orientation_angle_history_rad": angle_history,
                    "orientation_solution_counts": orientation_solver_counts,
                    "orientation_full_direction_fallback": used_full_direction_fallback,
                    "alignment_local_branch_rejected_count": len(
                        local_branch_rejections
                    ),
                    "alignment_local_branch_rejections": _json_safe_path_details(
                        local_branch_rejections
                    ),
                }
            # A direction-only IK solution can still lower one profiled finger
            # box below the support reserve because the box envelope changes
            # with wrist orientation. If the static certificate reports only
            # this failure, lift the temporary center by the measured deficit
            # plus a small robot-level margin and solve the same direction
            # step again. This is bounded closed-loop re-planning, not a
            # collision-check bypass: every retry is re-certified by the
            # complete static/live trajectory gates in ``run_segment``.
            support_step_retries = 0
            while True:
                ok, segment_details = run_segment(state, q_next, "orientation")
                if ok:
                    break
                certificate = segment_details.get("certificate", {})
                certificate_details = (
                    certificate.get("details", {})
                    if isinstance(certificate, dict)
                    else {}
                )
                support_failure = (
                    segment_details.get("failure_code")
                    == "alignment_path_unavailable"
                    and certificate.get("phase") == "static_finger_path"
                    and certificate_details.get("reason")
                    == "finger_support_collision"
                )
                if not support_failure or support_step_retries >= 3:
                    segment_details["orientation_step_index"] = step_index
                    segment_details["orientation_angle_history_rad"] = angle_history
                    segment_details["orientation_selected_substep_rad"] = (
                        selected_substep_rad
                    )
                    segment_details["orientation_full_direction_fallback"] = (
                        used_full_direction_fallback
                    )
                    segment_details["support_lift_retry_count"] = (
                        support_lift_retry_count
                    )
                    segment_details["support_lift_attempts"] = support_lift_attempts
                    return False, segment_details
                lift_world = 0.0
                try:
                    from r1pro_data_gen.robot.robot_config import (
                        R1PRO_GRIPPER_PREGRASP_CLEARANCE_M,
                    )
                    required_clearance = float(
                        certificate_details["required_clearance_m"]
                    )
                    observed_clearance = float(
                        certificate_details["support_clearance_m"]
                    )
                    deficit = required_clearance - observed_clearance
                    if not np.isfinite(deficit):
                        raise ValueError("support clearance deficit is not finite")
                    # The increment is deliberately applied only after a
                    # measured certificate rejection. Use one full
                    # pregrasp-clearance length rather than a millimetre-scale
                    # nudge: the window IK accepts up to 8 mm of center
                    # residual, so repeated tiny lifts can be entirely
                    # absorbed by the solver while the physical box remains
                    # at the same height. The exact static/live certificate
                    # still decides whether this candidate is executable.
                    lift_world = max(
                        float(R1PRO_GRIPPER_PREGRASP_CLEARANCE_M),
                        deficit + 0.002,
                    )
                    lifted_center_model = np.asarray(
                        state["center_model"], dtype=float
                    ) + np.asarray(state["rotation"], dtype=float).T @ np.asarray(
                        [0.0, 0.0, lift_world],
                        dtype=float,
                    )
                    # Separate the support lift from the jaw rotation. A
                    # direct joint-space segment to the elevated/oriented
                    # goal can dip through the support in the middle even
                    # when both endpoints are safe. Holding the measured
                    # jaw direction while lifting gives the trajectory
                    # certificate a monotone, physically meaningful escape
                    # path; only after it succeeds do we solve the desired
                    # orientation again at the new live center.
                    try:
                        lift_solutions = kin.ik_grasp_window_candidates(
                            lifted_center_model,
                            np.asarray(live_span, dtype=float),
                            state["q"],
                            max_candidates=branch_limit,
                            position_tolerance=0.002,
                            direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                            opening_offsets=state["offsets"],
                            span_to_constraint_rotation=state["rotation"],
                        )
                    except TypeError:
                        lift_solutions = kin.ik_grasp_window_candidates(
                            lifted_center_model,
                            np.asarray(live_span, dtype=float),
                            state["q"],
                            max_candidates=branch_limit,
                            position_tolerance=0.002,
                            direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                            opening_offsets=state["offsets"],
                        )
                    q_lift = choose_local_window_solution(
                        lift_solutions,
                        np.asarray(state["q"], dtype=float),
                        "live_orientation_support_lift",
                        require_progress=True,
                        max_step_rad=_ALIGN_MAX_CLEARANCE_ORIENTATION_JOINT_STEP_RAD,
                    )
                    if q_lift is None:
                        raise ValueError("support lift has no bounded IK branch")
                    lift_ok, lift_details = run_segment(
                        state,
                        q_lift,
                        "support_lift",
                    )
                except (
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    q_lift = None
                    lift_ok = False
                    lift_details = {
                        "failure_code": "alignment_orientation_ik_failed",
                        "reason": "support lift has no bounded IK branch",
                    }
                if q_lift is None or not lift_ok:
                    segment_details["orientation_step_index"] = step_index
                    segment_details["orientation_angle_history_rad"] = angle_history
                    segment_details["orientation_selected_substep_rad"] = (
                        selected_substep_rad
                    )
                    segment_details["support_lift_retry_count"] = (
                        support_lift_retry_count
                    )
                    segment_details["support_lift_attempts"] = support_lift_attempts
                    segment_details["support_lift_failed_m"] = float(lift_world)
                    segment_details["support_lift_details"] = lift_details
                    return False, segment_details
                support_lift_attempts.append(
                    {
                        "orientation_step_index": step_index,
                        "lift_world_m": float(lift_world),
                        "target_center_model": np.asarray(
                            lifted_center_model,
                            dtype=float,
                        ).round(6).tolist(),
                        "q_goal": np.asarray(q_lift, dtype=float).round(6).tolist(),
                        "max_joint_delta_rad": float(
                            np.max(
                                np.abs(
                                    np.asarray(q_lift, dtype=float)
                                    - np.asarray(state["q"], dtype=float)
                                )
                            )
                        ),
                    }
                )
                state = lift_details
                try:
                    retry_solutions = kin.ik_grasp_window_candidates(
                        np.asarray(state["center_model"], dtype=float),
                        direction_target,
                        state["q"],
                        max_candidates=branch_limit,
                        direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                        opening_offsets=state["offsets"],
                        span_to_constraint_rotation=state["rotation"],
                    )
                except TypeError:
                    retry_solutions = kin.ik_grasp_window_candidates(
                        np.asarray(state["center_model"], dtype=float),
                        direction_target,
                        state["q"],
                        max_candidates=branch_limit,
                        direction_tolerance=float(GRASP_WINDOW_IK_DIRECTION_TOL),
                        opening_offsets=state["offsets"],
                    )
                q_retry = choose_local_window_solution(
                    retry_solutions,
                    np.asarray(state["q"], dtype=float),
                    "live_orientation_after_support_lift",
                    require_progress=True,
                    max_step_rad=_ALIGN_MAX_CLEARANCE_ORIENTATION_JOINT_STEP_RAD,
                )
                if q_retry is None:
                    segment_details = {
                        "failure_code": "alignment_orientation_ik_failed",
                        "reason": "no bounded orientation branch after support lift",
                    }
                    segment_details["orientation_step_index"] = step_index
                    segment_details["orientation_angle_history_rad"] = angle_history
                    segment_details["orientation_selected_substep_rad"] = (
                        selected_substep_rad
                    )
                    segment_details["support_lift_retry_count"] = (
                        support_lift_retry_count + 1
                    )
                    segment_details["support_lift_attempts"] = support_lift_attempts
                    return False, segment_details
                q_next = q_retry
                support_step_retries += 1
                support_lift_retry_count += 1
            state = segment_details
        else:
            return False, {
                "failure_code": "alignment_orientation_unavailable",
                "reason": "live jaw direction did not converge within the bounded sequence",
                "orientation_angle_history_rad": angle_history,
            }

        desired_final = np.asarray(desired_span_world, dtype=float)
        if (
            desired_final.shape != (3,)
            or not np.all(np.isfinite(desired_final))
            or float(np.linalg.norm(desired_final[:2])) <= 1.0e-8
        ):
            desired_final = _object_window_span_direction(
                adapter,
                side,
                protected_object_name,
            )
        if desired_final is None:
            desired_final = np.asarray(desired_span_world, dtype=float)
        # The orientation stage intentionally retreats the physical jaw from
        # the object.  The requested correction was computed before that
        # retreat, so mapping the complete residual into one final IK solve
        # would turn a short measured correction into a large branch jump.
        # Advance only one bounded physical chunk; the enclosing alignment
        # loop remeasures the center and repeats from the actual state.  This
        # keeps the same semantic target for arbitrary objects while making
        # the approach obey the local joint-continuity contract.
        target_delta_world = np.asarray(target_center_world, dtype=float) - np.asarray(
            state["center_world"], dtype=float
        )
        target_delta_norm = float(np.linalg.norm(target_delta_world))
        if target_delta_norm > float(_ALIGN_MAX_STEP_M):
            target_delta_world *= float(_ALIGN_MAX_STEP_M) / max(
                target_delta_norm, 1.0e-12
            )
        window_target_world = np.asarray(state["center_world"], dtype=float) + (
            target_delta_world
        )
        target_model = np.asarray(state["center_model"], dtype=float) + np.asarray(
            state["rotation"], dtype=float
        ).T @ (
            window_target_world - np.asarray(state["center_world"], dtype=float)
        )
        return True, {
            "q": np.asarray(state["q"], dtype=float),
            "target_center_model": target_model,
            "target_center_world": window_target_world,
            "target_chunk_m": float(np.linalg.norm(target_delta_world)),
            "direction_span_target": np.asarray(desired_final, dtype=float),
            "rotation": np.asarray(state["rotation"], dtype=float),
            "translation": np.asarray(state["translation"], dtype=float),
            "offsets": state["offsets"],
            "calibration_rms": state["calibration_rms"],
            "orientation_steps": len(angle_history),
            "support_lift_retry_count": support_lift_retry_count,
            "support_lift_attempts": support_lift_attempts,
            "orientation_angle_history_rad": angle_history,
            "final_orientation_error_rad": angle_history[-1]
            if angle_history
            else 0.0,
        }

    # A full wrist orientation is unnecessarily strong for parallel-jaw
    # acquisition.  If the live jaw direction can be solved together with the
    # physical midpoint, use that underconstrained branch first.  This is what
    # prevents a low-workspace position-only fallback from driving one finger
    # endpoint into the object while retaining full path/effort/contact gates.
    if (
        target_center_model is not None
        and geometry_span_world is not None
        and hasattr(kin, "ik_grasp_window_candidates")
    ):
        desired_span_step_world = _object_window_direction_step(
            adapter,
            side,
            protected_object_name,
            np.asarray(geometry_span_world, dtype=float),
            max_step_rad=_ALIGN_MAX_WINDOW_DIRECTION_STEP_RAD,
        )
        direction_rotation: np.ndarray | None = None
        try:
            direction_rotation, _ = _alignment_world_model_transform(
                base_pose,
                model_to_world_rotation,
                model_to_world_translation,
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            direction_rotation = None
        desired_span_model = desired_span_step_world
        window_span_model = np.asarray(geometry_span_world, dtype=float)
        if direction_rotation is not None and desired_span_step_world is not None:
            # Position targets remain in model coordinates, but jaw direction
            # is constrained in the physical/world frame.  In particular, do
            # not inverse-map a world-horizontal vector and then drop its
            # model Z component: a calibrated pitch/roll would change the
            # direction seen by the real USD fingers.
            desired_span_model = (
                np.asarray(direction_rotation, dtype=float).T
                @ np.asarray(desired_span_step_world, dtype=float)
            )
            window_span_model = (
                np.asarray(direction_rotation, dtype=float).T
                @ window_span_model
            )
        elif desired_span_step_world is None:
            desired_span_model = None
        direction_span_target = (
            np.asarray(geometry_span_world, dtype=float)
            if direction_rotation is not None
            else np.asarray(window_span_model, dtype=float)
        )
        if desired_span_model is not None:
            standoff_q: np.ndarray | None = None
            clearance_q: np.ndarray | None = None
            orientation_waypoints: tuple[np.ndarray, ...] = ()
            if hasattr(kin, "finger_geometry_fk"):
                try:
                    current_center_model = np.asarray(
                        kin.finger_geometry_fk(start_q, opening_offsets)[4],
                        dtype=float,
                    )
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    current_center_model = None
            else:
                current_center_model = np.asarray(
                    kin.grasp_center_fk(start_q)[0],
                    dtype=float,
                )

            # A jaw reorientation must not happen at the current center when
            # that center is still close enough for a finger envelope to
            # sweep the object.  Derive an outer clearance center from the
            # live object footprint and the robot gripper envelope, then use
            # separate waypoints: retreat with the current jaw direction,
            # rotate in free space, and only then close the measured window.
            orientation_center_model = current_center_model
            orientation_seed_q = start_q
            orientation_already_aligned = False
            # Once the measured jaw direction is aligned, subsequent center
            # corrections must preserve the current physical wrist pose.  A
            # direction-only window solve is an underconstrained fallback;
            # on the loaded USD tool it can satisfy the model direction while
            # the real jaw rotates during the translation and sends the outer
            # alignment loop back into retreat/reorientation forever.
            window_orientation_lock = False
            window_geometry_details["current_center_model"] = (
                None
                if current_center_model is None
                else np.asarray(current_center_model, dtype=float).round(6).tolist()
            )
            if current_center_model is not None:
                try:
                    measured_center_world = _measured_grasp_center_world(
                        adapter, side
                    )
                    object_world = np.asarray(
                        adapter.object_position(protected_object_name),
                        dtype=float,
                    )
                    if measured_center_world is None:
                        raise ValueError("live grasp center is unavailable")
                    approach_xy = object_world[:2] - measured_center_world[:2]
                    approach_norm = float(np.linalg.norm(approach_xy))
                    orientation_clearance_m = _alignment_orientation_clearance(
                        scene, protected_object_name
                    )
                    if (
                        measured_center_world is not None
                        and object_world.shape == (3,)
                        and np.all(np.isfinite(object_world))
                        and approach_norm > 1.0e-7
                        and orientation_clearance_m is not None
                    ):
                        retreat_world = np.zeros(3, dtype=float)
                        retreat_world[:2] = (
                            -approach_xy / approach_norm
                        ) * float(orientation_clearance_m)
                        registration, _ = _alignment_world_model_transform(
                            base_pose,
                            model_to_world_rotation,
                            model_to_world_translation,
                        )
                        orientation_center_model = current_center_model + (
                            np.asarray(registration, dtype=float).T @ retreat_world
                        )
                        window_geometry_details.update(
                            {
                                "retreat_world": retreat_world.round(6).tolist(),
                                "retreat_derived": True,
                            }
                        )
                    else:
                        window_geometry_details["retreat_derived"] = False
                except (
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    orientation_center_model = current_center_model
                    window_geometry_details["retreat_derived"] = False

            window_geometry_details["orientation_center_model"] = (
                None
                if orientation_center_model is None
                else np.asarray(orientation_center_model, dtype=float).round(6).tolist()
            )

            # Reorientation is performed in free space, but the temporary
            # center still has to clear the source support for *all* jaw
            # orientations. The live pregrasp center can be a few millimetres
            # below the capability-derived support-aware height after a
            # compliant whole-body transition. Lift only this temporary
            # staging center in world Z; the final measured window approach
            # remains responsible for descending to the object. This keeps
            # the rule independent of object names and scene coordinates and
            # lets the exact finger-box certificate reject any residual risk.
            if orientation_center_model is not None and protected_object_name is not None:
                try:
                    support_object = scene.object(protected_object_name)
                    support_center_z = _alignment_support_clearance_center_z(
                        scene,
                        support_object,
                    )
                    registration, registration_translation = (
                        _alignment_world_model_transform(
                            base_pose,
                            model_to_world_rotation,
                            model_to_world_translation,
                        )
                    )
                    orientation_center_world = (
                        np.asarray(registration, dtype=float)
                        @ np.asarray(orientation_center_model, dtype=float)
                        + np.asarray(registration_translation, dtype=float)
                    )
                    support_lift = max(
                        0.0,
                        float(support_center_z) - float(orientation_center_world[2]),
                    ) if support_center_z is not None else 0.0
                    if support_lift > 0.0:
                        orientation_center_model = np.asarray(
                            orientation_center_model,
                            dtype=float,
                        ) + np.asarray(registration, dtype=float).T @ np.asarray(
                            [0.0, 0.0, support_lift],
                            dtype=float,
                        )
                    window_geometry_details.update(
                        {
                            "orientation_support_center_target_z_m": (
                                None
                                if support_center_z is None
                                else float(support_center_z)
                            ),
                            "orientation_support_lift_m": float(support_lift),
                        }
                    )
                except (
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    window_geometry_details["orientation_support_lift_m"] = 0.0
            window_geometry_details["orientation_center_model"] = (
                None
                if orientation_center_model is None
                else np.asarray(orientation_center_model, dtype=float).round(6).tolist()
            )

            if orientation_center_model is not None:
                # The loaded USD gripper can have a configuration-dependent
                # prismatic/tool residual even after the arm-link calibration
                # is excellent. Use the live finger vector as the orientation
                # reference and only map that vector into the reduced IK frame;
                # using nominal FK here was observed to report a one-step yaw
                # error while the real jaw was still tangent to the object.
                current_span_world: np.ndarray | None = None
                if hasattr(adapter, "body_position"):
                    try:
                        live_p1 = np.asarray(
                            adapter.body_position(
                                f"{side}_gripper_finger_link1"
                            ),
                            dtype=float,
                        )
                        live_p2 = np.asarray(
                            adapter.body_position(
                                f"{side}_gripper_finger_link2"
                            ),
                            dtype=float,
                        )
                        live_span = live_p2 - live_p1
                        if (
                            live_p1.shape == (3,)
                            and live_p2.shape == (3,)
                            and np.all(np.isfinite(live_p1))
                            and np.all(np.isfinite(live_p2))
                            and float(np.linalg.norm(live_span)) > 1.0e-7
                        ):
                            current_span_world = live_span
                    except (
                        AttributeError,
                        KeyError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        current_span_world = None
                if current_span_world is not None:
                    current_span_constraint = np.asarray(current_span_world, dtype=float)
                    window_geometry_details["current_span_source"] = "live_mapped"
                else:
                    try:
                        _, _, current_span_model = kin.finger_span_fk(
                            start_q, opening_offsets
                        )
                        current_span_constraint = np.asarray(
                            current_span_model, dtype=float
                        )
                        if direction_rotation is not None:
                            current_span_constraint = (
                                np.asarray(direction_rotation, dtype=float)
                                @ current_span_constraint
                            )
                    except (
                        AttributeError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        current_span_constraint = None
                    window_geometry_details["current_span_source"] = "kinematic_fk"

                # Once a previous measured correction has already aligned the
                # jaw, do not retreat and reorient it again on every outer
                # center iteration.  The alignment loop is intentionally
                # stateless across calls, so this physical check is the
                # generic state hand-off that prevents repeated retreats.
                if current_span_world is not None and direction_span_target is not None:
                    current_vector = np.asarray(current_span_world, dtype=float)
                    desired_vector = np.asarray(direction_span_target, dtype=float)
                    current_norm = float(np.linalg.norm(current_vector))
                    desired_norm = float(np.linalg.norm(desired_vector))
                    orientation_error = math.pi
                    if (
                        current_vector.shape == (3,)
                        and desired_vector.shape == (3,)
                        and np.all(np.isfinite(current_vector))
                        and np.all(np.isfinite(desired_vector))
                        and current_norm > 1.0e-8
                        and desired_norm > 1.0e-8
                    ):
                        current_xy = current_vector[:2]
                        desired_xy = desired_vector[:2]
                        current_xy_norm = float(np.linalg.norm(current_xy))
                        desired_xy_norm = float(np.linalg.norm(desired_xy))
                        if current_xy_norm > 1.0e-8 and desired_xy_norm > 1.0e-8:
                            current_xy /= current_xy_norm
                            desired_xy /= desired_xy_norm
                            orientation_error = math.acos(
                                float(
                                    np.clip(
                                        abs(float(np.dot(current_xy, desired_xy))),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                            tilt = math.asin(
                                min(1.0, abs(float(current_vector[2])) / current_norm)
                            )
                            tilt_violation = max(
                                0.0,
                                tilt - float(GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD),
                            )
                            if tilt_violation > 0.0:
                                orientation_error = max(
                                    orientation_error,
                                    tilt_violation
                                    + float(GRASP_WINDOW_IK_DIRECTION_TOL),
                                )
                        orientation_already_aligned = (
                            orientation_error <= GRASP_WINDOW_IK_DIRECTION_TOL
                        )
                        window_geometry_details["current_orientation_error_rad"] = float(
                            orientation_error
                        )
                if orientation_already_aligned:
                    orientation_center_model = current_center_model
                    orientation_seed_q = start_q
                    clearance_q = None
                    window_orientation_lock = True
                    # The live USD jaw is the authoritative orientation after
                    # a measured correction.  Preserve that vector for the
                    # local center/window solve instead of asking the nominal
                    # model to reproduce the original direction target.  A
                    # small calibration/tool residual can make those two
                    # directions differ enough for the unconstrained IK to
                    # select a distant redundancy branch even though the
                    # physical jaw is already aligned.
                    direction_span_target = np.asarray(
                        current_span_world,
                        dtype=float,
                    )
                    if direction_rotation is not None:
                        window_span_model = (
                            np.asarray(direction_rotation, dtype=float).T
                            @ direction_span_target
                        )
                    else:
                        window_span_model = direction_span_target.copy()
                    window_geometry_details["window_direction_source"] = (
                        "live_current_preserved"
                    )
                    window_geometry_details["retreat_derived"] = False
                    window_geometry_details["orientation_already_aligned"] = True

                def choose_local_window_solution(
                    solutions: Any,
                    reference_q: np.ndarray,
                    role: str,
                    require_progress: bool = False,
                    max_step_rad: float | None = None,
                ) -> np.ndarray | None:
                    step_limit = (
                        max_local_joint_step
                        if max_step_rad is None
                        else float(max_step_rad)
                    )
                    for candidate_q, _, _ in _rank_continuous_ik_solutions(
                        kin, solutions, reference_q
                    ):
                        max_joint_delta = float(
                            np.max(np.abs(candidate_q - reference_q))
                        )
                        if require_progress and max_joint_delta <= 1.0e-4:
                            continue
                        if max_joint_delta <= step_limit:
                            return np.asarray(candidate_q, dtype=float)
                        local_branch_rejections.append(
                            {
                                "max_joint_delta_rad": max_joint_delta,
                                "limit_rad": step_limit,
                                "q_goal": np.asarray(
                                    candidate_q, dtype=float
                                ).round(5).tolist(),
                                "reference_q": np.asarray(
                                    reference_q, dtype=float
                                ).round(5).tolist(),
                                "role": role,
                            }
                        )
                    return None

                if (
                    not orientation_already_aligned
                    and
                    orientation_center_model is not current_center_model
                    and current_span_constraint is not None
                ):
                    try:
                        clearance_solutions = kin.ik_grasp_window_candidates(
                            orientation_center_model,
                            current_span_constraint,
                            start_q,
                            max_candidates=branch_limit,
                            opening_offsets=opening_offsets,
                            span_to_constraint_rotation=direction_rotation,
                        )
                    except (
                        AttributeError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        clearance_solutions = []
                    window_geometry_details["clearance_solution_count"] = len(
                        clearance_solutions
                    )
                    clearance_q = choose_local_window_solution(
                        clearance_solutions,
                        start_q,
                        "clearance",
                    )
                    if clearance_q is not None:
                        orientation_seed_q = clearance_q

                # If retreat was not needed or could not be solved, retain the
                # original same-center orientation fallback. Otherwise solve
                # the jaw direction progressively at the retreat center and
                # seed each branch from the preceding one so the arm does not
                # jump to a different redundancy branch.
                if clearance_q is None:
                    orientation_center_model = current_center_model
                    orientation_seed_q = start_q
                direction_targets: list[np.ndarray] = []
                if (
                    not orientation_already_aligned
                    and
                    current_span_constraint is not None
                    and direction_span_target is not None
                ):
                    current_xy = np.asarray(current_span_constraint[:2], dtype=float)
                    target_xy = np.asarray(direction_span_target[:2], dtype=float)
                    current_norm = float(np.linalg.norm(current_xy))
                    target_norm = float(np.linalg.norm(target_xy))
                    if current_norm > 1.0e-8 and target_norm > 1.0e-8:
                        current_xy /= current_norm
                        target_xy /= target_norm
                        if float(np.dot(current_xy, target_xy)) < 0.0:
                            target_xy = -target_xy
                        signed_angle = math.atan2(
                            float(
                                current_xy[0] * target_xy[1]
                                - current_xy[1] * target_xy[0]
                            ),
                            float(np.dot(current_xy, target_xy)),
                        )
                        direction_step_count = max(
                            1,
                            int(
                                math.ceil(
                                    abs(signed_angle)
                                    / _ALIGN_MAX_WINDOW_DIRECTION_STEP_RAD
                                )
                            ),
                        )
                        current_angle = math.atan2(
                            float(current_xy[1]), float(current_xy[0])
                        )
                        for direction_index in range(1, direction_step_count + 1):
                            angle = current_angle + signed_angle * (
                                direction_index / direction_step_count
                            )
                            direction_targets.append(
                                np.array(
                                    [math.cos(angle), math.sin(angle), 0.0],
                                    dtype=float,
                                )
                            )
                    window_geometry_details[
                        "orientation_direction_step_count"
                    ] = len(direction_targets)
                if (
                    not direction_targets
                    and window_span_model is not None
                    and not orientation_already_aligned
                ):
                    direction_targets = [np.asarray(window_span_model, dtype=float)]
                orientation_solution_counts: list[int] = []
                orientation_qs: list[np.ndarray] = []
                for direction_index, direction_target in enumerate(direction_targets):
                    try:
                        direction_solutions = kin.ik_grasp_window_candidates(
                            orientation_center_model,
                            direction_target,
                            orientation_seed_q,
                            max_candidates=branch_limit,
                            opening_offsets=opening_offsets,
                            span_to_constraint_rotation=direction_rotation,
                        )
                    except (
                        AttributeError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        direction_solutions = []
                    orientation_solution_counts.append(len(direction_solutions))
                    if not direction_solutions:
                        break
                    role = (
                        "orientation_final"
                        if direction_index == len(direction_targets) - 1
                        else "orientation_standoff"
                    )
                    direction_q = choose_local_window_solution(
                        direction_solutions,
                        orientation_seed_q,
                        role,
                    )
                    if direction_q is None:
                        break
                    orientation_qs.append(direction_q)
                    orientation_seed_q = direction_q
                if orientation_qs:
                    if len(orientation_qs) == len(direction_targets):
                        # The final orientation is reached before the object;
                        # intermediate solutions are explicit free-space
                        # waypoints, while the last one is the fixed-orientation
                        # approach start.
                        standoff_q = orientation_qs[-1]
                        orientation_waypoints = tuple(orientation_qs[:-1])
                    else:
                        # Do not claim the final direction was solved. The
                        # partial sequence is still useful as a bounded local
                        # correction, but the window goal must be filtered from
                        # its actual last waypoint.
                        standoff_q = orientation_qs[-1]
                window_geometry_details["orientation_standoff_solution_count"] = (
                    orientation_solution_counts[0]
                    if orientation_solution_counts
                    else 0
                )
                window_geometry_details["orientation_final_solution_count"] = (
                    orientation_solution_counts[-1]
                    if orientation_solution_counts
                    and len(orientation_qs) == len(direction_targets)
                    else 0
                )
                window_geometry_details["staged_clearance"] = clearance_q is not None
                window_geometry_details["staged_orientation"] = (
                    standoff_q is not None or bool(orientation_waypoints)
                )
                window_geometry_details["orientation_waypoint_count"] = len(
                    orientation_waypoints
                )

            # A static sequence is only a seed for a real USD gripper: its
            # tool residual can change as the wrist rotates.  Once a safe
            # outer clearance branch exists, execute the reorientation as
            # measured local segments and rebuild the final window target
            # from the post-rotation physical state.  This keeps the object
            # stationary while retaining the same collision/effort gates.
            if (
                clearance_q is not None
                and target_center_world is not None
                and model_to_world_rotation is not None
                and hasattr(adapter, "read_observation")
                and hasattr(adapter, "body_position")
            ):
                live_orientation_ok, live_orientation_details = (
                    execute_live_orientation_sequence(
                        np.asarray(start_q, dtype=float),
                        np.asarray(clearance_q, dtype=float),
                        np.asarray(direction_span_target, dtype=float),
                    )
                )
                if not live_orientation_ok:
                    window_geometry_details.update(
                        {
                            "live_staged_orientation": False,
                            "live_orientation_failure": live_orientation_details,
                        }
                    )
                    return False, {
                        "reason": "live staged orientation failed",
                        "failure_code": live_orientation_details.get(
                            "failure_code", "alignment_orientation_unavailable"
                        ),
                        "target_pos": target_ee.round(5).tolist(),
                        "q_current": start_q.round(5).tolist(),
                        "orientation_clearance_m": orientation_clearance_m,
                        "window_geometry": _json_safe_path_details(
                            window_geometry_details
                        ),
                        **{
                            key: value
                            for key, value in live_orientation_details.items()
                            if key not in {"failure_code"}
                        },
                    }
                start_q = np.asarray(live_orientation_details["q"], dtype=float)
                target_center_model = np.asarray(
                    live_orientation_details["target_center_model"], dtype=float
                )
                direction_span_target = np.asarray(
                    live_orientation_details["direction_span_target"], dtype=float
                )
                model_to_world_rotation = np.asarray(
                    live_orientation_details["rotation"], dtype=float
                )
                model_to_world_translation = np.asarray(
                    live_orientation_details["translation"], dtype=float
                )
                # ``direction_rotation`` was computed before the measured
                # clearance/orientation sequence.  Reuse the registration
                # fitted at the actual post-rotation state for the final
                # physical window solve; otherwise a pitch/roll residual can
                # invalidate the calibrated world-direction constraint.
                direction_rotation = np.asarray(
                    model_to_world_rotation, dtype=float
                )
                opening_offsets = live_orientation_details["offsets"]
                calibration_rms = live_orientation_details["calibration_rms"]
                desired_span_model = (
                    np.asarray(model_to_world_rotation, dtype=float).T
                    @ np.asarray(direction_span_target, dtype=float)
                )
                window_span_model = desired_span_model.copy()
                window_orientation_lock = True
                # The live orientation executor reports convergence against
                # the object-derived direction, but the final center move
                # must preserve the *measured post-rotation* jaw vector. A
                # loaded USD gripper can have a small configuration-dependent
                # tool residual; reusing the nominal object direction here
                # lets the window IK undo the just-completed orientation and
                # makes the outer alignment loop retreat repeatedly. Read the
                # actual span at the hand-off and constrain only the next
                # local center correction to that stable physical direction.
                try:
                    live_p1_after = np.asarray(
                        adapter.body_position(
                            f"{side}_gripper_finger_link1"
                        ),
                        dtype=float,
                    )
                    live_p2_after = np.asarray(
                        adapter.body_position(
                            f"{side}_gripper_finger_link2"
                        ),
                        dtype=float,
                    )
                    measured_span_after = live_p2_after - live_p1_after
                    if (
                        live_p1_after.shape == (3,)
                        and live_p2_after.shape == (3,)
                        and np.all(np.isfinite(live_p1_after))
                        and np.all(np.isfinite(live_p2_after))
                        and float(np.linalg.norm(measured_span_after)) > 1.0e-7
                    ):
                        direction_span_target = np.asarray(
                            measured_span_after,
                            dtype=float,
                        )
                        desired_span_model = (
                            np.asarray(model_to_world_rotation, dtype=float).T
                            @ direction_span_target
                        )
                        window_span_model = desired_span_model.copy()
                        window_geometry_details[
                            "window_direction_source"
                        ] = "live_post_orientation_preserved"
                except (
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    pass
                # The physical clearance/orientation segments have already
                # been executed.  The remaining candidate paths must start
                # at the measured post-rotation state and contain no stale
                # nominal waypoints.
                clearance_q = None
                standoff_q = None
                orientation_waypoints = ()
                window_geometry_details.update(
                    {
                        "live_staged_orientation": True,
                        "live_orientation_steps": live_orientation_details.get(
                            "orientation_steps", 0
                        ),
                        "live_orientation_angle_history_rad": live_orientation_details.get(
                            "orientation_angle_history_rad", []
                        ),
                        "live_final_orientation_error_rad": live_orientation_details.get(
                            "final_orientation_error_rad", 0.0
                        ),
                        "orientation_waypoint_count": 0,
                        "staged_clearance": False,
                        "staged_orientation": False,
                    }
                )
                # Candidate continuity must be measured from the actual
                # post-rotation state.  The final fixed-pose correction below
                # is deliberately local, so comparing it with the original
                # pre-grasp q would discard the valid branch before the path
                # certificate gets a chance to evaluate it.
                candidate_reference_q = np.asarray(start_q, dtype=float).copy()

            # A measured orientation correction establishes a physical
            # jaw-pose lock.  Move the center by a short model-frame delta
            # while holding the current EE quaternion, then let the existing
            # finite-window solver handle any residual.  The target EE is
            # anchored at the measured/open-jaw center predicted at the
            # current q, so the tool offset is not reintroduced as a nominal
            # absolute pose error.
            if (
                window_orientation_lock
                and target_center_model is not None
                and hasattr(kin, "ik_candidates")
            ):
                try:
                    current_ee_model, current_quat_model = kin.fk(
                        np.asarray(start_q, dtype=float)
                    )
                    if hasattr(kin, "finger_geometry_fk"):
                        current_window_center_model = np.asarray(
                            kin.finger_geometry_fk(
                                np.asarray(start_q, dtype=float), opening_offsets
                            )[4],
                            dtype=float,
                        )
                    else:
                        current_window_center_model = np.asarray(
                            kin.grasp_center_fk(np.asarray(start_q, dtype=float))[0],
                            dtype=float,
                        )
                    current_ee_model = np.asarray(current_ee_model, dtype=float)
                    current_quat_model = np.asarray(current_quat_model, dtype=float)
                    target_center_model = np.asarray(target_center_model, dtype=float)
                    hold_target_ee = current_ee_model + (
                        target_center_model - current_window_center_model
                    )
                    if (
                        current_ee_model.shape != (3,)
                        or current_quat_model.shape != (4,)
                        or current_window_center_model.shape != (3,)
                        or hold_target_ee.shape != (3,)
                        or not np.all(np.isfinite(current_ee_model))
                        or not np.all(np.isfinite(current_quat_model))
                        or not np.all(np.isfinite(current_window_center_model))
                        or not np.all(np.isfinite(hold_target_ee))
                        or float(np.linalg.norm(current_quat_model)) <= 1.0e-8
                    ):
                        raise ValueError("fixed-pose window target is not finite")
                    current_quat_model /= float(np.linalg.norm(current_quat_model))
                    hold_solutions = kin.ik_candidates(
                        hold_target_ee,
                        current_quat_model,
                        np.asarray(start_q, dtype=float),
                        max_candidates=branch_limit,
                    )
                    window_geometry_details[
                        "fixed_orientation_window_solution_count"
                    ] = len(hold_solutions or ())
                    outcome, details = append_and_execute(
                        hold_solutions,
                        orientation_relaxed=False,
                        center_position_ik=False,
                        phase="fixed_orientation_window_candidate_path_check",
                    )
                    if outcome is not None:
                        return bool(outcome), details
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    window_geometry_details[
                        "fixed_orientation_window_solution_count"
                    ] = 0
            mark_planning_phase("window_geometry_candidate_generation")
            if standoff_q is not None:
                window_seed_q = standoff_q
            elif clearance_q is not None:
                window_seed_q = clearance_q
            else:
                window_seed_q = start_q
            window_geometry_details["window_seed"] = (
                "standoff"
                if standoff_q is not None
                else "clearance"
                if clearance_q is not None
                else "start"
            )
            # The executable window segment starts at the last solved staging
            # waypoint.  Keep candidate filtering aligned with that segment;
            # the trajectory certificate below still checks the complete
            # start -> clearance -> standoff -> window path.
            candidate_reference_q = np.asarray(window_seed_q, dtype=float).copy()
            try:
                window_solutions = kin.ik_grasp_window_candidates(
                    target_center_model,
                    direction_span_target,
                    window_seed_q,
                    max_candidates=branch_limit,
                    opening_offsets=opening_offsets,
                    span_to_constraint_rotation=direction_rotation,
                )
            except (
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                window_solutions = []
                window_geometry_details["window_solver_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            window_geometry_details["window_solution_count"] = len(window_solutions)
            window_geometry_details["window_target_center_model"] = np.asarray(
                target_center_model,
                dtype=float,
            ).round(6).tolist()
            window_geometry_details["window_direction_target_world"] = np.asarray(
                direction_span_target,
                dtype=float,
            ).round(6).tolist()
            window_geometry_details["window_registration_rotation"] = (
                None
                if direction_rotation is None
                else np.asarray(direction_rotation, dtype=float).round(6).tolist()
            )
            window_geometry_details["window_opening_offsets"] = (
                None
                if opening_offsets is None
                else [
                    np.asarray(offset, dtype=float).round(6).tolist()
                    for offset in opening_offsets
                ]
            )
            window_geometry_details["window_solution_max_delta_rad"] = [
                float(
                    np.max(
                        np.abs(
                            np.asarray(solution.q_arm, dtype=float)
                            - np.asarray(window_seed_q, dtype=float)
                        )
                    )
                )
                for solution in window_solutions
                if getattr(solution, "q_arm", None) is not None
            ]
            outcome, details = append_and_execute(
                window_solutions,
                # The jaw direction is constrained, but no arbitrary full
                # wrist orientation is required by this branch.
                orientation_relaxed=True,
                center_position_ik=False,
                phase="window_geometry_candidate_path_check",
                standoff_q=standoff_q,
                clearance_q=clearance_q,
                orientation_waypoints=orientation_waypoints,
            )
            if outcome is not None:
                return bool(outcome), details
            # All later fallbacks are direct corrections from the measured
            # pre-grasp, so restore the original local-branch reference.
            candidate_reference_q = np.asarray(start_q, dtype=float).copy()

    # A relaxed EE-position IK is not equivalent to a relaxed grasp-center
    # goal: the wrist orientation changes the physical finger-midpoint offset.
    # Try orientation-constrained candidates first.  The live jaw-orientation
    # candidate is especially important after a whole-body position-only
    # pregrasp; the center-only solver remains a bounded fallback for robots or
    # poses where the orientation-constrained basin is unavailable.
    for candidate_index, quat_goal in enumerate(orientation_candidates):
        mark_planning_phase(f"orientation_{candidate_index}_generation")
        candidate_target_ee = target_ee
        if target_center_model is not None and hasattr(
            kin, "ee_target_from_grasp_center"
        ):
            candidate_target_ee = np.asarray(
                kin.ee_target_from_grasp_center(target_center_model, quat_goal),
                dtype=float,
            )
        if hasattr(kin, "ik_candidates"):
            solutions = kin.ik_candidates(
                candidate_target_ee,
                quat_goal,
                start_q,
                max_candidates=branch_limit,
            )
        else:
            solution = kin.ik(candidate_target_ee, quat_goal, q_init=start_q)
            solutions = [solution] if solution.success and solution.q_arm is not None else []
        outcome, details = append_and_execute(
            solutions,
            orientation_relaxed=candidate_index > 0,
            center_position_ik=False,
            phase=f"orientation_{candidate_index}_path_check",
        )
        if outcome is not None:
            return bool(outcome), details

    if target_center_model is not None and hasattr(kin, "ik_grasp_center_candidates"):
        mark_planning_phase("center_candidate_generation")
        try:
            center_solutions = kin.ik_grasp_center_candidates(
                target_center_model,
                start_q,
                max_candidates=branch_limit,
            )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            center_solutions = []
            window_geometry_details["center_solver_exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        outcome, details = append_and_execute(
            center_solutions,
            orientation_relaxed=True,
            center_position_ik=True,
            phase="center_candidate_path_check",
        )
        if outcome is not None:
            return bool(outcome), details

    if hasattr(kin, "ik_candidates"):
        # Alignment is a measured position task.  If neither the grasp
        # orientation nor the current posture solves, relax orientation only as
        # a last fallback: near the table the exact orientation can be outside
        # the DLS solver's convergence basin even when position-only IK has a
        # continuous solution.
        mark_planning_phase("position_only_generation")
        position_target = target_ee
        if target_center_model is not None and hasattr(
            kin, "ee_target_from_grasp_center"
        ):
            position_target = np.asarray(
                kin.ee_target_from_grasp_center(
                    target_center_model,
                    np.asarray(quat0, dtype=float),
                ),
                dtype=float,
            )
        position_solutions = kin.ik_candidates(
            position_target, None, start_q, max_candidates=branch_limit
        )
        outcome, details = append_and_execute(
            position_solutions,
            orientation_relaxed=True,
            center_position_ik=False,
            phase="position_only_path_check",
        )
        if outcome is not None:
            return bool(outcome), details

    if not candidate_options:
        return False, {
            "reason": "alignment IK failed",
            "failure_code": "alignment_ik_failed",
            "target_pos": target_ee.round(5).tolist(),
            "q_current": start_q.round(5).tolist(),
            "current_ee": np.asarray(kin.fk(start_q)[0]).round(5).tolist(),
            "orientation_relaxed": False,
            "grasp_center_position_ik": False,
            "calibration_rms_m": calibration_rms,
            "alignment_ik_branch_limit": branch_limit,
            "alignment_local_joint_step_limit_rad": max_local_joint_step,
            "alignment_local_branch_rejected_count": len(local_branch_rejections),
            "alignment_local_branch_rejections": _json_safe_path_details(
                local_branch_rejections
            ),
            "window_geometry": _json_safe_path_details(window_geometry_details),
        }

    return False, {
        "reason": "alignment candidate paths rejected",
        "failure_code": "alignment_path_unavailable",
        "target_pos": target_ee.round(5).tolist(),
        "q_current": start_q.round(5).tolist(),
        "current_ee": np.asarray(kin.fk(start_q)[0]).round(5).tolist(),
        "orientation_relaxed": any(item[3] for item in candidate_options),
        "grasp_center_position_ik": any(item[4] for item in candidate_options),
        "calibration_rms_m": calibration_rms,
        "alignment_candidate_count": len(candidate_options),
        "alignment_path_rejected_count": len(path_rejections),
        "alignment_path_rejections": _json_safe_path_details(path_rejections),
        "alignment_ik_branch_limit": branch_limit,
        "alignment_local_joint_step_limit_rad": max_local_joint_step,
        "alignment_local_branch_rejected_count": len(local_branch_rejections),
        "alignment_local_branch_rejections": _json_safe_path_details(
            local_branch_rejections
        ),
        "orientation_clearance_m": orientation_clearance_m,
        "window_geometry": _json_safe_path_details(window_geometry_details),
    }


def _execute_alignment_trajectory(
    kin: Any,
    adapter: Any,
    side: str,
    trajectory: np.ndarray,
    start_q: np.ndarray,
    step_hook: Callable[[], None] | None,
    scene: Any = None,
    exclude_objects: tuple[str, ...] = (),
    object_motion_guard: Callable[[], dict[str, Any] | None] | None = None,
    model_to_world_rotation: np.ndarray | None = None,
    model_to_world_translation: np.ndarray | None = None,
    protected_object_name: str | None = None,
) -> bool:
    """Execute one alignment correction, returning False when it would collide.

    Alignment corrections are short measured moves solved without a planner.
    The margin-best IK branch can pass through the target object (or push it)
    when the descent is longer than the gripper window; the contact stop only
    fires after contact registers, which is too late on a fast push.  A dense
    hppfcl check of the *commanded trajectory itself* rejects a correction that
    would sweep through the object, so alignment keeps the object stationary
    and hands the grasp primitive a clean, two-sided contact.  No MPlib, no
    OMPL -- just a cheap link-sphere check of the few trajectory samples.
    """
    from r1pro_data_gen.methods.collision import (
        CollisionChecker,
        check_path,
        collision_mesh_for_body,
        obstacles_from_scene,
    )
    from r1pro_data_gen.methods.manipulation.mplib_path import _ARM_LIMITS_BY_SIDE, _ARM_SLICE_BY_SIDE

    joints = ARM_JOINTS_BY_SIDE[side]
    trajectory_to_execute = np.asarray(trajectory, dtype=float)
    telemetry_adapter = getattr(adapter, "_adapter", adapter)
    execution_telemetry: dict[str, Any] = {
        "requested_sample_count": int(len(trajectory_to_execute)),
        "certified_sample_count": int(len(trajectory_to_execute)),
        "executed_sample_count": 0,
        "stopped_on_contact": False,
        "start_q": np.asarray(trajectory_to_execute[0], dtype=float).round(5).tolist()
        if len(trajectory_to_execute)
        else [],
    }
    try:
        telemetry_adapter._alignment_last_path_rejection = None
        telemetry_adapter._alignment_last_path_execution = execution_telemetry
        history = getattr(telemetry_adapter, "_alignment_path_history", None)
        if not isinstance(history, list):
            history = []
            telemetry_adapter._alignment_path_history = history
        history.append(execution_telemetry)
        del history[:-128]
    except (AttributeError, TypeError, ValueError):
        pass
    if not hasattr(adapter, "read_observation"):
        execution_telemetry["certified_sample_count"] = int(len(trajectory))
        for q_target in trajectory[1:]:
            adapter.set_targets(
                position={joint: float(q_target[i]) for i, joint in enumerate(joints)},
                velocity={},
            )
            adapter.step()
            execution_telemetry["executed_sample_count"] = int(
                execution_telemetry["executed_sample_count"]
            ) + 1
            if step_hook is not None:
                step_hook()
        return True
    base_pose = adapter.read_observation(0.0).base_pose or (0.0, 0.0, 0.0)
    scene_model = scene if scene is not None else getattr(adapter, "scene_model", None)
    if scene_model is None:
        # Without scene geometry there is nothing to check against; keep the
        # historical behaviour for minimal adapters.
        pass
    else:
        alignment_object_model = None
        if hasattr(scene_model, "objects") and protected_object_name is None:
            # The semantic alignment API always knows which object it is
            # approaching.  Dropping that identity (for example through an
            # outdated wrapper signature) would disable the live rigid-body
            # gate, so refuse to execute an uncertified physical correction.
            raise _AlignmentCollisionCheckUnavailable(
                {
                    "checked": False,
                    "reason": "protected alignment object was not provided",
                }
            )
        # The alignment is a controlled local correction toward the object, so
        # the support surface is the only geometry excluded (it would block the
        # final short descent).  The target object itself must stay in the
        # obstacle set so the correction cannot sweep through it.  The obstacle
        # set is built from a runtime snapshot so a pushed/moving object is
        # checked at its actual position, not its reset pose.
        try:
            from r1pro_data_gen.skills.planning import runtime_scene_snapshot

            live_scene = runtime_scene_snapshot(scene_model, adapter, exclude_objects=exclude_objects)
            if protected_object_name is not None:
                alignment_object_model = live_scene.object(protected_object_name)
            # Keep the palm/gripper body in the swept-volume check.  Alignment
            # may legitimately finish with the two finger links around the
            # object, but the palm and proximal links must never enter the
            # object's collision envelope before the attachment is established.
            # The previous filter removed the palm as well as the fingers;
            # that allowed a palm-first approach to push a low object before
            # either finger sensor could report contact.
            from r1pro_data_gen.methods.collision import LINK_SPHERE_RADII_BY_SIDE

            full_radii = dict(LINK_SPHERE_RADII_BY_SIDE[side])
            arm_radii = {
                name: radius
                for name, radius in full_radii.items()
                if not name.endswith("gripper_finger_link1")
                and not name.endswith("gripper_finger_link2")
            }
            gripper_link_name = f"{side}_gripper_link"
            gripper_mesh = collision_mesh_for_body(gripper_link_name)
            interaction_margins: dict[str, float] = {}
            if protected_object_name is not None:
                try:
                    interaction_object = live_scene.object(protected_object_name)
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    interaction_object = None
                if interaction_object is not None:
                    # Keep the target in the obstacle set, but use its
                    # physical contact envelope rather than the larger
                    # free-space planning buffer.  The palm remains in the
                    # checked link set and the live object-motion guard below
                    # rejects any actual pre-attachment push.
                    interaction_margins[protected_object_name] = (
                        _alignment_interaction_margin(interaction_object)
                    )
            checker = CollisionChecker(
                kin,
                obstacles_from_scene(
                    live_scene,
                    margins=interaction_margins,
                    include_ground=True,
                ),
                link_radii=arm_radii,
                link_meshes=(
                    {gripper_link_name: gripper_mesh}
                    if gripper_mesh is not None
                    else None
                ),
            )
            free, body_details, _ = check_path(
                checker,
                [np.asarray(q, dtype=float) for q in trajectory],
                base_xy=(float(base_pose[0]), float(base_pose[1])),
                base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
                # ``trajectory`` is already sampled at the simulator's
                # physics cadence by _minimum_jerk_trajectory and its
                # adjacent joint increments are velocity-limited.  Endpoint
                # checking therefore covers every commanded physical
                # interval; an additional 4x interpolation here repeats the
                # same certificate for every IK candidate and can consume the
                # entire action wall-time budget before the next correction.
                dense=1,
                model_to_world_rotation=model_to_world_rotation,
                model_to_world_translation=model_to_world_translation,
            )
            if not free:
                collision_segment = int(body_details)
                collision_samples: list[dict[str, Any]] = []
                for collision_sample in sorted(
                    {
                        min(max(0, collision_segment), len(trajectory) - 1),
                        min(max(0, collision_segment + 1), len(trajectory) - 1),
                    }
                ):
                    collision_samples.append(
                        {
                            "sample_index": int(collision_sample),
                            "collision_link": checker.first_collision_link(
                                np.asarray(trajectory[collision_sample], dtype=float),
                                base_xy=(float(base_pose[0]), float(base_pose[1])),
                                base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
                                model_to_world_rotation=model_to_world_rotation,
                                model_to_world_translation=model_to_world_translation,
                            ),
                            "q": np.asarray(
                                trajectory[collision_sample], dtype=float
                            ).round(5).tolist(),
                        }
                    )
                body_failure_details = {
                    "segment_index": collision_segment,
                    "samples": collision_samples,
                }
                execution_telemetry["failure_phase"] = "static_body_path"
                execution_telemetry["failure_details"] = body_failure_details
                try:
                    telemetry_adapter._alignment_last_path_rejection = {
                        "phase": "static_body_path",
                        "details": body_failure_details,
                    }
                except (AttributeError, TypeError, ValueError):
                    pass
                return False
            if protected_object_name is not None and alignment_object_model is not None:
                support_top_z = _alignment_support_top_z(
                    live_scene,
                    alignment_object_model,
                )
                finger_free, trajectory_to_execute, finger_details = (
                    _alignment_finger_path_certificate(
                        kin,
                        adapter,
                        side,
                        trajectory_to_execute,
                        start_q,
                        alignment_object_model,
                        base_pose,
                        model_to_world_rotation,
                        model_to_world_translation,
                        support_top_z=support_top_z,
                    )
                )
                if not finger_free:
                    execution_telemetry["failure_phase"] = "static_finger_path"
                    execution_telemetry["failure_details"] = _json_safe_path_details(
                        finger_details
                    )
                    try:
                        telemetry_adapter._alignment_last_path_rejection = {
                            "phase": "static_finger_path",
                            "details": dict(_json_safe_path_details(finger_details)),
                        }
                    except (AttributeError, TypeError, ValueError):
                        pass
                    return False
                execution_telemetry["certified_sample_count"] = int(
                    len(trajectory_to_execute)
                )
                execution_telemetry["finger_certificate"] = _json_safe_path_details(
                    finger_details
                )
        except Exception as exc:
            # A physical alignment must never proceed without its requested
            # geometry certificate.  Minimal adapters that do not provide a
            # scene model take the compatibility path above; once a real scene
            # is supplied, a checker failure is a safety failure, not a reason
            # to execute an uncertified trajectory.
            raise _AlignmentCollisionCheckUnavailable(
                {
                    "checked": False,
                    "reason": "static alignment collision check unavailable",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
            ) from exc
    for trajectory_index, q_target in enumerate(trajectory_to_execute[1:], start=1):
        check_live_geometry = (
            trajectory_index == 1
            or trajectory_index == len(trajectory_to_execute) - 1
            or trajectory_index % _ALIGN_LIVE_GEOMETRY_CHECK_STRIDE == 0
        )
        if check_live_geometry:
            live_free, live_details = _live_alignment_body_collision(
                adapter,
                scene_model,
                side,
                protected_object_name,
                exclude_objects,
            )
            if not live_free:
                if not bool(live_details.get("checked", False)):
                    raise _AlignmentCollisionCheckUnavailable(live_details)
                # Before the command, this candidate can still be rejected and
                # a different continuous branch may be tried without
                # advancing the physical state.
                try:
                    telemetry_adapter._alignment_last_path_rejection = {
                        "phase": "live_body_before_step",
                        "details": dict(_json_safe_path_details(live_details)),
                    }
                except (AttributeError, TypeError, ValueError):
                    pass
                execution_telemetry["failure_phase"] = "live_body_before_step"
                execution_telemetry["failure_details"] = _json_safe_path_details(
                    live_details
                )
                return False
            finger_free, finger_details = _live_alignment_finger_collision(
                adapter,
                scene_model,
                side,
                protected_object_name,
                exclude_objects,
            )
            if not finger_free:
                if not bool(finger_details.get("checked", False)):
                    raise _AlignmentCollisionCheckUnavailable(finger_details)
                # Contact can begin before the predicted zero-opening
                # certificate is true.  It is allowed to remain inside the
                # intentional two-sided interaction window so the outer loop
                # can take another measured correction; ``between_fingers``
                # remains the strict close-ready gate.
                if not _live_alignment_interaction_window_ready(
                    adapter,
                    protected_object_name,
                    side,
                ):
                    try:
                        telemetry_adapter._alignment_last_path_rejection = {
                            "phase": "live_finger_before_step",
                            "details": dict(_json_safe_path_details(finger_details)),
                        }
                    except (AttributeError, TypeError, ValueError):
                        pass
                    execution_telemetry["failure_phase"] = "live_finger_before_step"
                    execution_telemetry["failure_details"] = _json_safe_path_details(
                        finger_details
                    )
                    return False
        try:
            telemetry_adapter._alignment_last_path_execution["executed_sample_count"] = (
                int(
                    telemetry_adapter._alignment_last_path_execution.get(
                        "executed_sample_count", 0
                    )
                )
                + 1
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        adapter.set_targets(
            position={joint: float(q_target[i]) for i, joint in enumerate(joints)},
            velocity={},
        )
        adapter.step()
        # A measured alignment is a contact-sensitive operation.  The target
        # object remains dynamic until the gripper establishes the attachment,
        # so executing the entire precomputed correction after the first
        # finger touch can turn a valid approach into a push.  Check the live
        # finger sensors at physics cadence and stop at the first contact; the
        # semantic caller then refreshes the window and decides whether this
        # contact is a valid two-sided grasp or a replannable side hit.
        contact_detected = _alignment_finger_contact_detected(
            adapter,
            side,
            _ALIGN_CONTACT_STOP_N,
        )
        if step_hook is not None:
            step_hook()
        if check_live_geometry:
            live_free, live_details = _live_alignment_body_collision(
                adapter,
                scene_model,
                side,
                protected_object_name,
                exclude_objects,
            )
            if not live_free:
                if not bool(live_details.get("checked", False)):
                    raise _AlignmentCollisionCheckUnavailable(live_details)
                try:
                    telemetry_adapter._alignment_last_path_rejection = {
                        "phase": "live_body_after_step",
                        "details": dict(_json_safe_path_details(live_details)),
                    }
                except (AttributeError, TypeError, ValueError):
                    pass
                raise _AlignmentCollisionDetected(live_details)
            finger_free, finger_details = _live_alignment_finger_collision(
                adapter,
                scene_model,
                side,
                protected_object_name,
                exclude_objects,
            )
            if not finger_free:
                if not bool(finger_details.get("checked", False)):
                    raise _AlignmentCollisionCheckUnavailable(finger_details)
                if not _live_alignment_interaction_window_ready(
                    adapter,
                    protected_object_name,
                    side,
                ):
                    try:
                        telemetry_adapter._alignment_last_path_rejection = {
                            "phase": "live_finger_after_step",
                            "details": dict(_json_safe_path_details(finger_details)),
                        }
                    except (AttributeError, TypeError, ValueError):
                        pass
                    raise _AlignmentCollisionDetected(finger_details)
        if object_motion_guard is not None:
            violation = object_motion_guard()
            if violation is not None:
                raise _ObjectMovedBeforeGrasp(violation)
        # Contact sensors can be unavailable or one frame late on a GPU PhysX
        # filter.  The finite finger segment is a second, measured contact
        # signal: stop when its closest point reaches the physical contact
        # envelope, including endpoint contacts that are not a valid pinch.
        # The semantic alignment loop then remeasures and can rotate/replan a
        # bounded branch before sending another approach command.
        geometry_contact = _alignment_geometry_contact_detected(
            adapter,
            side,
            protected_object_name,
        )
        if contact_detected or geometry_contact:
            # A first finger can touch a few millimetres before the second
            # finger enters the vertical contact band.  If the live support-
            # plane window is already centered, keep this short vertical
            # segment running so the second side can be acquired.  The
            # object-motion guard above still aborts on any displacement, and
            # the outer alignment loop never treats this intermediate state as
            # success.  Any contact outside that bounded window stops here.
            interaction_window = _live_alignment_interaction_window_ready(
                adapter,
                protected_object_name,
                side,
            )
            terminal_window = _live_alignment_window_ready(
                adapter,
                protected_object_name,
                side,
            )
            if not (interaction_window and not terminal_window):
                try:
                    telemetry_adapter._alignment_last_path_execution[
                        "stopped_on_contact"
                    ] = True
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
                break
    return True


def _alignment_finger_contact_detected(
    adapter: Any,
    side: str,
    threshold_n: float,
) -> bool:
    """Return whether either live finger sensor has reached ``threshold_n``.

    Contact telemetry is an optional adapter capability.  Lightweight
    backends may not expose it, in which case the geometry and object-motion
    gates remain responsible for the alignment result.  A malformed or
    temporarily unavailable sensor sample is treated as unavailable rather
    than as contact; it must never create a false grasp success.
    """
    reader = getattr(adapter, "finger_contact_forces", None)
    if reader is None:
        return False
    try:
        try:
            values = reader(side=side)
        except TypeError:
            values = reader()
        forces = tuple(float(value) for value in values)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return bool(forces) and max(forces) > float(threshold_n)


def _alignment_geometry_contact_detected(
    adapter: Any,
    side: str,
    object_name: str | None,
) -> bool:
    """Stop an open-jaw correction at the measured physical contact envelope.

    Finger force sensors are authoritative when available, but GPU contact
    filters can report one frame late or have no filtered force for a valid
    geometry contact.  The live finite-segment measurement is therefore a
    conservative second signal.  It does not declare a grasp: the caller
    still has to verify the segment fraction, vertical band, two-sided contact
    and attachment.
    """
    if object_name is None or not hasattr(adapter, "gripper_object_alignment"):
        return False
    telemetry_adapter = getattr(adapter, "_adapter", adapter)
    try:
        alignment = dict(adapter.gripper_object_alignment(object_name, side=side))
        p1_raw = alignment.get("finger_position_1")
        p2_raw = alignment.get("finger_position_2")
        object_position_raw = alignment.get("object_position")
        p1 = (
            np.asarray(p1_raw, dtype=float)
            if p1_raw is not None
            else np.full(3, np.nan, dtype=float)
        )
        p2 = (
            np.asarray(p2_raw, dtype=float)
            if p2_raw is not None
            else np.full(3, np.nan, dtype=float)
        )
        object_position = (
            np.asarray(object_position_raw, dtype=float)
            if object_position_raw is not None
            else np.full(3, np.nan, dtype=float)
        )
        surface_distance = float(alignment.get("surface_distance_m"))
        segment_fraction = float(alignment.get("segment_fraction"))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    if not np.isfinite(surface_distance) or not np.isfinite(segment_fraction):
        return False
    try:
        scene = getattr(adapter, "scene_model", None)
        model = scene.object(object_name) if scene is not None else None
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        model = None
    if model is None:
        contact_margin = 0.005
    else:
        contact_margin = max(0.002, _alignment_interaction_margin(model))
    try:
        telemetry_adapter._alignment_geometry_surface_distance_m = float(
            surface_distance
        )
        telemetry_adapter._alignment_geometry_segment_fraction = float(
            segment_fraction
        )
        telemetry_adapter._alignment_geometry_contact_margin_m = float(
            contact_margin
        )
        if p1.shape == (3,) and np.all(np.isfinite(p1)):
            telemetry_adapter._alignment_geometry_finger_p1 = ",".join(
                f"{float(value):.6f}" for value in p1
            )
        if p2.shape == (3,) and np.all(np.isfinite(p2)):
            telemetry_adapter._alignment_geometry_finger_p2 = ",".join(
                f"{float(value):.6f}" for value in p2
            )
        if object_position.shape == (3,) and np.all(np.isfinite(object_position)):
            telemetry_adapter._alignment_geometry_object_position = ",".join(
                f"{float(value):.6f}" for value in object_position
            )
    except (AttributeError, TypeError, ValueError):
        pass
    return surface_distance <= contact_margin


def _object_window_span_direction(
    adapter: Any,
    side: str,
    protected_object_name: str | None,
) -> np.ndarray | None:
    """Return a live world jaw direction normal to the object approach.

    The returned line is sign-invariant because the two parallel fingers are
    interchangeable.  It is a geometric acquisition target only; the caller
    still verifies the finite segment and physical contact after execution.
    """
    if protected_object_name is None or not hasattr(adapter, "body_position"):
        return None
    try:
        p1 = np.asarray(
            adapter.body_position(f"{side}_gripper_finger_link1"), dtype=float
        )
        p2 = np.asarray(
            adapter.body_position(f"{side}_gripper_finger_link2"), dtype=float
        )
        object_position = np.asarray(
            adapter.object_position(protected_object_name), dtype=float
        )
        if any(
            value.shape != (3,) or not np.all(np.isfinite(value))
            for value in (p1, p2, object_position)
        ):
            return None
        midpoint = 0.5 * (p1 + p2)
        approach = object_position[:2] - midpoint[:2]
        approach_norm = float(np.linalg.norm(approach))
        span = p2[:2] - p1[:2]
        span_norm = float(np.linalg.norm(span))
        if approach_norm <= 1.0e-7 or span_norm <= 1.0e-7:
            return None
        desired = np.asarray([-approach[1], approach[0], 0.0], dtype=float)
        desired /= float(np.linalg.norm(desired))
        current = np.asarray([span[0], span[1], 0.0], dtype=float) / span_norm
        if float(np.dot(current, desired)) < 0.0:
            desired = -desired
        return desired
    except (
        AttributeError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return None


def _object_window_orientation_candidate(
    kin: Any,
    adapter: Any,
    side: str,
    q_current: np.ndarray,
    *,
    model_to_world_rotation: np.ndarray | None,
    protected_object_name: str | None,
    desired_span_world: np.ndarray | None = None,
) -> np.ndarray | None:
    """Orient the jaw span perpendicular to the live approach direction.

    A low-support pregrasp may be solved position-only, leaving the parallel
    jaw line tangent to the object.  Translating that posture toward the
    object then contacts one finger endpoint and pushes the object.  Rotating
    the current measured posture about world up so the jaw line is normal to
    the approach is the minimum-change geometric correction.  IK and all
    collision/effort gates remain responsible for deciding whether the
    candidate is executable.
    """
    if protected_object_name is None or not hasattr(adapter, "body_position"):
        return None
    try:
        p1 = np.asarray(
            adapter.body_position(f"{side}_gripper_finger_link1"), dtype=float
        )
        p2 = np.asarray(
            adapter.body_position(f"{side}_gripper_finger_link2"), dtype=float
        )
        object_position = np.asarray(
            adapter.object_position(protected_object_name), dtype=float
        )
        if any(
            value.shape != (3,) or not np.all(np.isfinite(value))
            for value in (p1, p2, object_position)
        ):
            return None
        midpoint = 0.5 * (p1 + p2)
        span = p2[:2] - p1[:2]
        span_norm = float(np.linalg.norm(span))
        if span_norm <= 1.0e-7:
            return None
        current_span = span / span_norm
        if desired_span_world is None:
            approach = object_position[:2] - midpoint[:2]
            approach_norm = float(np.linalg.norm(approach))
            if approach_norm <= 1.0e-7:
                return None
            desired_span = np.asarray([-approach[1], approach[0]], dtype=float)
        else:
            desired_span_value = np.asarray(desired_span_world, dtype=float)
            if (
                desired_span_value.shape != (3,)
                or not np.all(np.isfinite(desired_span_value))
            ):
                return None
            desired_span = desired_span_value[:2]
        desired_norm = float(np.linalg.norm(desired_span))
        if desired_norm <= 1.0e-7:
            return None
        desired_span /= desired_norm
        # The two finger endpoints are interchangeable. Keep the closest
        # signed direction to avoid an unnecessary 180-degree wrist turn.
        if float(np.dot(current_span, desired_span)) < 0.0:
            desired_span = -desired_span
        delta = math.atan2(
            float(current_span[0] * desired_span[1] - current_span[1] * desired_span[0]),
            float(np.dot(current_span, desired_span)),
        )
        if abs(delta) <= math.radians(3.0):
            return None
        _, current_quat = kin.fk(np.asarray(q_current, dtype=float))
        current_model_rotation = Rotation.from_quat(
            [
                float(current_quat[1]),
                float(current_quat[2]),
                float(current_quat[3]),
                float(current_quat[0]),
            ]
        ).as_matrix()
        if model_to_world_rotation is not None:
            registration = np.asarray(model_to_world_rotation, dtype=float)
            if registration.shape != (3, 3) or not np.all(np.isfinite(registration)):
                return None
            current_world_rotation = registration @ current_model_rotation
            desired_world_rotation = (
                Rotation.from_euler("z", delta).as_matrix() @ current_world_rotation
            )
            desired_model_rotation = registration.T @ desired_world_rotation
        else:
            desired_model_rotation = (
                Rotation.from_euler("z", delta).as_matrix() @ current_model_rotation
            )
        quat_xyzw = Rotation.from_matrix(desired_model_rotation).as_quat()
        candidate = np.asarray(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=float,
        )
        return candidate if np.all(np.isfinite(candidate)) else None
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _measured_grasp_center_delta(
    kin: Any,
    adapter: Any,
    side: str,
    base_pose: tuple[float, float, float],
    q_cur: np.ndarray,
) -> np.ndarray | None:
    """Base-frame delta between the simulated finger midpoint and the model
    grasp center at the current configuration; None when unmeasurable."""
    if not hasattr(adapter, "body_position") or base_pose is None:
        return None
    try:
        _sync_kinematics_auxiliary_q(kin, adapter.read_observation(0.0))
        p1 = np.asarray(adapter.body_position(f"{side}_gripper_finger_link1"), dtype=float)
        p2 = np.asarray(adapter.body_position(f"{side}_gripper_finger_link2"), dtype=float)
        midpoint_world = 0.5 * (p1 + p2)
        if midpoint_world.shape != (3,) or not np.all(np.isfinite(midpoint_world)):
            return None
    except Exception:
        return None
    yaw = float(base_pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    dx = float(midpoint_world[0]) - float(base_pose[0])
    dy = float(midpoint_world[1]) - float(base_pose[1])
    midpoint_base = np.array([c * dx + s * dy, -s * dx + c * dy, float(midpoint_world[2])])
    try:
        model_gc_base, _ = kin.grasp_center_fk(np.asarray(q_cur, dtype=float))
    except Exception:
        return None
    delta = midpoint_base - model_gc_base
    # Guard against calibration blowups (e.g. gripper not yet spawned).
    if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) > 0.15:
        return None
    return delta


def _summarize_failure_stages(candidates: Any) -> dict[str, int]:
    """Count candidate failures by pipeline stage.

    Separates "the planner found no path" (``mplib_plan`` / RRT stages) from
    "a path was found but rejected by verification" (collision stages) — the
    two need completely different remedies.
    """
    summary: dict[str, int] = {}
    for candidate in candidates or ():
        stage = getattr(getattr(candidate, "constraints", None), "stage", None)
        if stage:
            key = str(stage)
            summary[key] = summary.get(key, 0) + 1
    return summary


def _select_continuous_ik_solution(
    kin: Any,
    solutions: Any,
    q_reference: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Choose a short local IK correction without changing redundant branches.

    Margin-best IK is useful for an isolated tabletop goal, but it is the wrong
    objective for a measured correction: selecting a distant elbow branch can
    make the next correction or the subsequent carry target unreachable.  Keep
    continuity as the primary key and use joint-limit margin only to break
    near-ties.
    """
    ranked = _rank_continuous_ik_solutions(kin, solutions, q_reference)
    return ranked[0] if ranked else None


def _rank_continuous_ik_solutions(
    kin: Any,
    solutions: Any,
    q_reference: np.ndarray,
) -> list[tuple[np.ndarray, float, float]]:
    """Return all valid local IK branches ordered by continuity.

    A local correction must not silently switch to a distant redundant branch,
    but continuity alone is not enough: the closest branch may sweep the palm
    through the target object while a slightly less-close branch is safe.  The
    alignment executor therefore consumes this complete, bounded ranking and
    lets the live swept-volume certificate choose the first executable branch.
    """
    q_reference = np.asarray(q_reference, dtype=float)
    lower = np.asarray(kin.lower, dtype=float)
    upper = np.asarray(kin.upper, dtype=float)
    span = np.maximum(upper - lower, 1e-9)
    ranked: list[tuple[tuple[float, float, float], tuple[np.ndarray, float, float]]] = []
    for solution in solutions:
        if not solution.success or solution.q_arm is None:
            continue
        q = np.asarray(solution.q_arm, dtype=float)
        if q.shape != q_reference.shape or not np.all(np.isfinite(q)):
            continue
        continuity = float(np.linalg.norm((q - q_reference) / span))
        margin = float(np.min(np.minimum(q - lower, upper - q)))
        residual = float(solution.position_error + solution.rotation_error)
        key = (round(continuity, 10), round(-margin, 10), round(residual, 10))
        ranked.append((key, (q, continuity, margin)))
    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked]


def _alignment_interaction_margin(object_model: Any) -> float:
    """Return the physical clearance used while intentionally acquiring an object.

    ``planning_margin`` is a free-space buffer: it is deliberately larger than
    the contact envelope and is appropriate while routing around an object.
    During a grasp, the selected object remains an obstacle for the palm and
    proximal links, but the fingers must be allowed to enter its contact
    neighborhood.  Use the authored PhysX contact offset for that neighborhood
    and fall back to a small scale-independent buffer when an older scene does
    not declare one.  The target is never removed from the obstacle set.
    """
    physics = getattr(object_model, "physics", None)
    contact_offset = getattr(physics, "contact_offset", None)
    try:
        value = float(contact_offset)
    except (TypeError, ValueError):
        value = float("nan")
    if np.isfinite(value) and value >= 0.0:
        return value
    planning_margin = getattr(physics, "planning_margin", None)
    try:
        planning_value = float(planning_margin)
    except (TypeError, ValueError):
        planning_value = 0.01
    if not np.isfinite(planning_value) or planning_value < 0.0:
        planning_value = 0.01
    return min(planning_value, 0.01)


def _alignment_orientation_clearance(
    scene: Any,
    object_name: str | None,
) -> float | None:
    """Return a geometry-derived retreat distance for jaw reorientation.

    A position-only pregrasp can leave the open jaw tangent to a low object.
    Rotating that jaw at the same center sweeps a finger box through the
    object even when the center itself is still far away.  Before changing
    orientation, the local controller therefore retreats by the object's
    footprint plus the robot gripper envelope and the live contact margin.
    This is a capability-scaled clearance, not a task waypoint; if the live
    object geometry is unavailable the caller keeps its ordinary fallback.
    """
    if scene is None or object_name is None or not hasattr(scene, "object"):
        return None
    try:
        from r1pro_data_gen.domain import object_xy_radius_m
        from r1pro_data_gen.robot.robot_config import (
            R1PRO_GRIPPER_COLLISION_ENVELOPE_M,
            R1PRO_GRIPPER_PREGRASP_CLEARANCE_M,
        )

        object_model = scene.object(object_name)
        object_radius = float(object_xy_radius_m(object_model))
        if not np.isfinite(object_radius) or object_radius < 0.0:
            return None
        envelope = float(R1PRO_GRIPPER_COLLISION_ENVELOPE_M)
        if not np.isfinite(envelope) or envelope <= 0.0:
            return None
        return float(
            envelope
            + object_radius
            + _alignment_interaction_margin(object_model)
            + float(R1PRO_GRIPPER_PREGRASP_CLEARANCE_M)
        )
    except (
        AttributeError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return None


def _alignment_support_clearance_center_z(
    scene: Any,
    object_model: Any,
) -> float | None:
    """Return a support-relative center height for free-space reorientation.

    The final approach may intentionally place the finger boxes close to a
    floor or tabletop. Reorienting an open jaw at that height is different:
    the lowest box corner changes with wrist orientation, so a pose that is
    safe before rotation can become support-adjacent halfway through the
    rotation. Use the same robot-capability envelope as support-aware
    pregrasp derivation to lift the temporary reorientation center. The exact
    swept finger-box certificate remains authoritative for every path.
    """
    support_top_z = _alignment_support_top_z(scene, object_model)
    if support_top_z is None:
        return None
    try:
        from r1pro_data_gen.robot.robot_config import (
            R1PRO_GRIPPER_FINGER_SUPPORT_ENVELOPE_Z_M,
            R1PRO_GRIPPER_PREGRASP_CLEARANCE_M,
            R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M,
        )

        return float(
            support_top_z
            + float(R1PRO_GRIPPER_FINGER_SUPPORT_ENVELOPE_Z_M)
            + _alignment_interaction_margin(object_model)
            + float(R1PRO_GRIPPER_PREGRASP_CLEARANCE_M)
            + float(R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M)
        )
    except (
        AttributeError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return None


def _live_alignment_body_collision(
    adapter: Any,
    scene: Any,
    side: str,
    protected_object_name: str | None,
    exclude_objects: tuple[str, ...],
) -> tuple[bool, dict[str, Any]]:
    """Check measured non-finger link origins against the live target object.

    The offline Pinocchio certificate protects the commanded joint path.  A
    real controller can still lag or overshoot between two short corrections,
    so alignment also checks the measured USD/PhysX body positions immediately
    before and after every physics step.  Finger links are intentionally
    omitted: they are the permitted acquisition envelope, while the palm and
    proximal links must remain outside the object until attachment.
    """
    telemetry_adapter = getattr(adapter, "_adapter", adapter)

    def finish(free: bool, details: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        # Record coverage in the physical adapter so an outer safety abort in
        # the same step can still show whether this gate actually ran.
        try:
            telemetry_adapter._alignment_live_collision_checks = int(
                getattr(telemetry_adapter, "_alignment_live_collision_checks", 0)
            ) + 1
            telemetry_adapter._alignment_live_collision_object = str(
                protected_object_name or ""
            )
            telemetry_adapter._alignment_live_collision_last = str(
                details.get("reason", "clear" if free else "rejected")
            )
        except (AttributeError, TypeError, ValueError):
            pass
        return free, details

    if protected_object_name is None or scene is None:
        return finish(True, {"checked": False})
    if not hasattr(adapter, "body_position"):
        # The model-space path certificate remains available for lightweight
        # adapters.  Physical adapters expose body_position, so they never
        # silently skip this live check.
        return finish(True, {"checked": False, "reason": "adapter has no body_position"})
    try:
        from r1pro_data_gen.methods.collision import (
            LINK_SPHERE_RADII_BY_SIDE,
            LINK_SPHERE_OFFSETS_BY_SIDE,
            collision_mesh_for_body,
            object_obstacle,
        )
        from r1pro_data_gen.skills.planning import runtime_scene_snapshot
        import hppfcl

        if protected_object_name in set(exclude_objects):
            return finish(False, {
                "checked": False,
                "reason": "protected alignment object was excluded from collision scene",
                "object_name": protected_object_name,
            })
        live_scene = runtime_scene_snapshot(scene, adapter, exclude_objects=())
        object_model = live_scene.object(protected_object_name)
        margin = _alignment_interaction_margin(object_model)
        obstacle = object_obstacle(object_model, margin)
        radii = {
            name: radius
            for name, radius in LINK_SPHERE_RADII_BY_SIDE[side].items()
            if not name.endswith("gripper_finger_link1")
            and not name.endswith("gripper_finger_link2")
        }
        request = hppfcl.CollisionRequest()
        result = hppfcl.CollisionResult()
        for body_name, radius in radii.items():
            position = np.asarray(adapter.body_position(body_name), dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                return finish(False, {
                    "checked": False,
                    "reason": "live alignment body position is invalid",
                    "body_name": body_name,
                    "object_name": protected_object_name,
                })
            offset = LINK_SPHERE_OFFSETS_BY_SIDE[side].get(body_name)
            body_rotation = None
            if body_name == f"{side}_gripper_link" and hasattr(adapter, "body_pose"):
                try:
                    _, quaternion = adapter.body_pose(body_name)
                    body_rotation = _wxyz_rotation(quaternion)
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                    return finish(False, {
                        "checked": False,
                        "reason": "live gripper-link orientation is unavailable",
                        "body_name": body_name,
                        "object_name": protected_object_name,
                    })
            if offset is not None and body_rotation is None and hasattr(adapter, "body_pose"):
                try:
                    _, quaternion = adapter.body_pose(body_name)
                    position = position + _wxyz_rotation(quaternion) @ np.asarray(
                        offset, dtype=float
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                    return finish(False, {
                        "checked": False,
                        "reason": "live alignment body orientation is unavailable",
                        "body_name": body_name,
                        "object_name": protected_object_name,
                    })
            mesh = collision_mesh_for_body(body_name)
            if mesh is not None and body_rotation is not None:
                shape = mesh
                shape_transform = hppfcl.Transform3f(body_rotation, position)
                shape_source = "asset_mesh"
            else:
                shape = hppfcl.Sphere(float(radius))
                shape_transform = hppfcl.Transform3f(position)
                shape_source = "sphere_proxy"
            result.clear()
            if hppfcl.collide(
                obstacle.shape,
                obstacle.transform,
                shape,
                shape_transform,
                request,
                result,
            ):
                return finish(False, {
                    "checked": True,
                    "reason": "non-finger link entered target collision envelope",
                    "body_name": body_name,
                    "object_name": protected_object_name,
                    "body_position_world": position.round(6).tolist(),
                    "object_position_world": list(
                        float(value) for value in object_model.pos
                    ),
                    "inflation_margin_m": margin,
                    "shape_source": shape_source,
                })
        return finish(True, {
            "checked": True,
            "object_name": protected_object_name,
            "checked_link_count": len(radii),
            "inflation_margin_m": margin,
        })
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return finish(False, {
            "checked": False,
            "reason": "live alignment collision check unavailable",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "object_name": protected_object_name,
        })


def _arm_move_to_staged_fallback(
    move_to: "ArmMoveTo",
    adapter: Any,
    scene: Any,
    *,
    goal_ee: np.ndarray,
    quat: np.ndarray,
    side: str,
    planning_time: float,
    local_radius_m: float,
    speed_scale: float,
    ik_candidates: int,
    exclude_objects: list[str] | None,
    step_hook: Callable[[], None] | None,
) -> SkillResult | None:
    """Retry an unreachable ``arm_move_to`` as a raised two-waypoint path.

    Builds ``[staging above the target, target]`` and delegates to
    :class:`ArmMoveThrough`, whose joint candidate selection, verified
    retiming and single-shot execution already handle multi-waypoint paths.
    ``goal_ee`` is the already frame-resolved EE goal (grasp_center or ee);
    the staging point is the same pose lifted by ``_STAGING_RETRY_LIFT_M``.
    Returns ``None`` when the staging plan itself cannot be produced, so the
    caller reports the original planning failure.
    """
    staging_ee = np.asarray(goal_ee, dtype=float) + np.array(
        [0.0, 0.0, _STAGING_RETRY_LIFT_M]
    )
    goal_ee = np.asarray(goal_ee, dtype=float)
    orientation = tuple(float(v) for v in (quat / np.linalg.norm(quat)))
    staged = ArmMoveThrough(move_to.kin, move_to.vel_limits, move_to.planner).execute(
        adapter,
        scene=scene,
        waypoints=[
            {
                "name": "staging",
                "poses": [{"position": [float(v) for v in staging_ee], "orientation": orientation}],
                "exclude_objects": list(exclude_objects or ()),
                "speed_scale": speed_scale,
            },
            {
                "name": "goal",
                "poses": [{"position": [float(v) for v in goal_ee], "orientation": orientation}],
                "exclude_objects": list(exclude_objects or ()),
            },
        ],
        side=side,
        planning_time=planning_time,
        ik_candidates_per_waypoint=ik_candidates,
        beam_width=3,
        max_planned_edges=72,
        trajectory_speed_scale=speed_scale,
        local_radius_m=local_radius_m,
        step_hook=step_hook,
    )
    if not staged.success:
        return None
    return staged


def _base_point_to_world(point: np.ndarray, base_pose: tuple[float, float, float]) -> np.ndarray:
    """Convert a base-frame point to a flat-world point.

    The domain observation intentionally exposes the mobile base as
    ``(x, y, yaw)``.  Manipulation scenes keep the root on the ground plane, so
    the point's z component is already expressed in world metres.  Keeping
    this conversion here (rather than in a task policy) lets every measured
    grasp-center action use the same frame contract.
    """
    point = np.asarray(point, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"point must be a 3-vector, got {point.shape}")
    x, y, yaw = (float(value) for value in base_pose[:3])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        [x + cosine * point[0] - sine * point[1],
         y + sine * point[0] + cosine * point[1],
         point[2]],
        dtype=float,
    )


def _sync_kinematics_auxiliary_q(kin: Any, observation: Any) -> None:
    """Register measured non-arm joints before FK/IK.

    The arm solver is intentionally a 7-DOF reduced model.  Its auxiliary
    torso configuration must nevertheless follow the physical articulation;
    otherwise a torso residual is silently interpreted as an arm/tool offset.
    Backends without auxiliary joints simply ignore this hook.
    """
    if not hasattr(kin, "set_auxiliary_q"):
        return
    positions = getattr(observation, "joint_positions", {}) or {}
    kin.set_auxiliary_q(
        {
            f"torso_joint{index}": float(positions[f"torso_joint{index}"])
            for index in range(1, 5)
            if f"torso_joint{index}" in positions
        }
    )


def _measured_grasp_center_world(adapter: Any, side: str) -> np.ndarray | None:
    """Read the physical finger midpoint from the adapter, if available."""
    try:
        if hasattr(adapter, "end_effector_poses"):
            poses = adapter.end_effector_poses() or {}
            pose = poses.get(f"{side}_gripper_finger_midpoint")
            if pose is not None:
                midpoint = np.asarray(pose[:3], dtype=float)
                if midpoint.shape == (3,) and np.all(np.isfinite(midpoint)):
                    return midpoint
        if hasattr(adapter, "body_position"):
            p1 = np.asarray(adapter.body_position(f"{side}_gripper_finger_link1"), dtype=float)
            p2 = np.asarray(adapter.body_position(f"{side}_gripper_finger_link2"), dtype=float)
            midpoint = 0.5 * (p1 + p2)
            if midpoint.shape == (3,) and np.all(np.isfinite(midpoint)):
                return midpoint
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    return None


def _object_window_direction_step(
    adapter: Any,
    side: str,
    protected_object_name: str | None,
    desired_world: np.ndarray,
    *,
    max_step_rad: float,
) -> np.ndarray | None:
    """Limit one live jaw-direction correction to a bounded angular step.

    The desired line is computed from the object geometry, while the current
    line is measured from the actual finger links.  Returning an intermediate
    line lets the arm acquire a difficult low-support pose over multiple
    short, effort-certified corrections instead of asking the position drive
    for a single large wrist change.
    """
    if protected_object_name is None or not hasattr(adapter, "body_position"):
        return np.asarray(desired_world, dtype=float)
    try:
        p1 = np.asarray(
            adapter.body_position(f"{side}_gripper_finger_link1"), dtype=float
        )
        p2 = np.asarray(
            adapter.body_position(f"{side}_gripper_finger_link2"), dtype=float
        )
        desired = np.asarray(desired_world, dtype=float)
        if (
            p1.shape != (3,)
            or p2.shape != (3,)
            or desired.shape != (3,)
            or not np.all(np.isfinite(p1))
            or not np.all(np.isfinite(p2))
            or not np.all(np.isfinite(desired))
        ):
            return None
        current = np.asarray(p2 - p1, dtype=float)
        current_norm = float(np.linalg.norm(current))
        desired_norm = float(np.linalg.norm(desired))
        if current_norm <= 1.0e-7 or desired_norm <= 1.0e-7:
            return None
        current /= current_norm
        desired /= desired_norm
        # The two jaw endpoints are interchangeable. Choose the sign closest
        # to the measured *3-D* span before interpolating; using only the XY
        # projection makes a nearly vertical open jaw appear horizontal and
        # defeats the angular step limit at a floor object.
        if float(np.dot(current, desired)) < 0.0:
            desired = -desired
        cosine = float(np.clip(np.dot(current, desired), -1.0, 1.0))
        angle = math.acos(cosine)
        step = min(angle, abs(float(max_step_rad)))
        if angle <= 1.0e-8 or step <= 1.0e-8:
            intermediate = current if step <= 1.0e-8 else desired
        else:
            sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
            if sine <= 1.0e-8:
                intermediate = current
            else:
                intermediate = (
                    current * math.cos(step)
                    + (desired - current * cosine)
                    * (math.sin(step) / sine)
                )
                intermediate /= max(float(np.linalg.norm(intermediate)), 1.0e-12)
        # The direction is returned in world frame.  The model registration is
        # consumed by the caller when it maps this measured target into IK.
        return intermediate
    except (
        AttributeError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return None


def _calibrated_model_center_target(
    kin: Any,
    adapter: Any,
    side: str,
    observation: Any,
    q_arm: np.ndarray,
    target_world: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray] | None:
    """Map a world grasp-center target into the current reduced model frame.

    Return the fitted rigid transform as well as the mapped point.  The same
    transform must be used for collision certification: otherwise a calibrated
    target can be mapped into the model correctly while the collision checker
    tests the resulting links in a different (simplified base-only) frame.
    """
    if not hasattr(kin, "calibrated_base_transform") or not hasattr(adapter, "body_position"):
        return None
    frame_names = tuple(getattr(kin, "base_calibration_frames", ()))
    if len(frame_names) < 3:
        return None
    try:
        _sync_kinematics_auxiliary_q(kin, observation)
        measured = np.asarray(
            [adapter.body_position(name) for name in frame_names],
            dtype=float,
        )
        rotation, translation, rms_error = kin.calibrated_base_transform(
            np.asarray(q_arm, dtype=float),
            measured,
            frame_names=frame_names,
        )
        if not np.isfinite(rms_error) or float(rms_error) > 0.03:
            return None
        target_world = np.asarray(target_world, dtype=float)
        if target_world.shape != (3,) or not np.all(np.isfinite(target_world)):
            return None
        target_model = np.asarray(rotation, dtype=float).T @ (
            target_world - np.asarray(translation, dtype=float)
        )
        if not np.all(np.isfinite(target_model)):
            return None
        return target_model, float(rms_error), np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


class _CalibratedCollisionProxy:
    """Adapt the generic joint-space RRT interface to a live rigid transform."""

    def __init__(
        self,
        checker: Any,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        self._checker = checker
        self._rotation = np.asarray(rotation, dtype=float)
        self._translation = np.asarray(translation, dtype=float)

    def is_collision_free(
        self,
        q_arm: np.ndarray,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
    ) -> bool:
        return self._checker.is_collision_free(
            q_arm,
            base_xy,
            base_yaw,
            model_to_world_rotation=self._rotation,
            model_to_world_translation=self._translation,
        )

    def first_collision_link(
        self,
        q_arm: np.ndarray,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
    ) -> str | None:
        return self._checker.first_collision_link(
            q_arm,
            base_xy,
            base_yaw,
            model_to_world_rotation=self._rotation,
            model_to_world_translation=self._translation,
        )


def _measured_center_trajectory(
    checker: Any,
    kin: Any,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    base_pose: Sequence[float],
    model_to_world_rotation: np.ndarray,
    model_to_world_translation: np.ndarray,
    side: str,
    speed_scale: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Find a collision-certified local arm trajectory in the live frame.

    Short grasp corrections normally use a direct minimum-jerk segment.  If a
    scene obstacle blocks that segment, use the generic joint-space
    RRT-Connect method with the same measured model registration.  No object
    name, task ordering, or scene-specific detour is introduced here.
    """
    from r1pro_data_gen.methods.collision import check_path
    from r1pro_data_gen.methods.manipulation.mplib_path import _linear_resample, _minimum_jerk_trajectory
    from r1pro_data_gen.methods.navigation.rrt import RRTConnectPlanner

    q_path = np.asarray([q_start, q_goal], dtype=float)
    base_xy = (float(base_pose[0]), float(base_pose[1]))
    base_yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
    from r1pro_data_gen.robot.robot_config import (
        R1PRO_FREE_ARM_SPEED_SCALE,
        R1PRO_LOCAL_ARM_MIN_TRAJECTORY_S,
    )

    direct_trajectory, _, _ = _minimum_jerk_trajectory(
        q_path,
        speed_scale=min(float(speed_scale), R1PRO_FREE_ARM_SPEED_SCALE),
        side=side,
        min_duration_s=R1PRO_LOCAL_ARM_MIN_TRAJECTORY_S,
    )
    free, _, collision_link = check_path(
        checker,
        [np.asarray(q, dtype=float) for q in direct_trajectory],
        base_xy=base_xy,
        base_yaw=base_yaw,
        dense=8,
        model_to_world_rotation=model_to_world_rotation,
        model_to_world_translation=model_to_world_translation,
    )
    if free:
        return direct_trajectory, {
            "planning_status": "MeasuredCenterDirectVerified",
            "collision_checked": True,
        }

    proxy = _CalibratedCollisionProxy(
        checker,
        model_to_world_rotation,
        model_to_world_translation,
    )
    for seed in (11, 23, 37):
        planner = RRTConnectPlanner(
            kin,
            proxy,
            step=0.18,
            max_iters=1200,
            connect_depth_cap=48,
            seed=seed,
        )
        ok, geometric, stats = planner.plan(
            q_start,
            q_goal,
            base_xy=base_xy,
            base_yaw=base_yaw,
        )
        if not ok:
            continue
        trajectory, _, _ = _linear_resample(
            np.asarray(geometric, dtype=float),
            speed_scale=min(float(speed_scale), 0.15),
            side=side,
        )
        valid, _, final_collision_link = check_path(
            proxy,
            [np.asarray(q, dtype=float) for q in trajectory],
            base_xy=base_xy,
            base_yaw=base_yaw,
            dense=4,
        )
        if valid:
            return trajectory, {
                "planning_status": "MeasuredCenterRRTConnectVerified",
                "collision_checked": True,
                "rrt_seed": seed,
                "rrt_stats": stats,
                "direct_collision_link": collision_link,
            }
        collision_link = final_collision_link
    return None, {
        "planning_status": "MeasuredCenterCollisionRejected",
        "collision_checked": True,
        "collision_link": collision_link,
    }


def _arm_move_to_measured_grasp_center_local(
    move_to: "ArmMoveTo",
    adapter: Any,
    scene: Any,
    *,
    target_center_world: np.ndarray | None,
    target_quat: np.ndarray | None,
    side: str,
    speed_scale: float,
    exclude_objects: list[str] | None,
    position_tolerance: float,
    step_hook: Callable[[], None] | None,
) -> SkillResult | None:
    """Reach a grasp center with short closed-loop measured corrections.

    A single model-space trajectory is unsafe for a floating-base robot when
    the torso or tool geometry has a physical residual.  This helper keeps the
    semantic target fixed in world space, advances at most a small Cartesian
    increment, recalibrates the reduced model from measured link origins, and
    verifies the *physical* finger midpoint after every segment.  It returns
    ``None`` only when the adapter/backend cannot provide the measurements; in
    that case the caller may use the legacy model-only path for compatible
    lightweight adapters.
    """
    if target_center_world is None:
        return None
    required = ("read_observation", "set_targets", "step", "body_position")
    if not all(hasattr(adapter, name) for name in required):
        return None
    kin = for_side(move_to.kin, side)
    if kin is None or not hasattr(kin, "calibrated_base_transform"):
        return None
    target_center_world = np.asarray(target_center_world, dtype=float)
    if target_center_world.shape != (3,) or not np.all(np.isfinite(target_center_world)):
        return None
    joints = ARM_JOINTS_BY_SIDE[side]
    from r1pro_data_gen.methods.collision import CollisionChecker, check_path, obstacles_from_scene

    tolerance = max(0.045, 1.75 * float(position_tolerance))
    max_iterations = 10
    max_step_m = 0.09
    records: list[dict[str, Any]] = []
    orientation_relaxed = False
    last_collision_link = None

    def solve_local_ik(
        target: np.ndarray,
        orientation: np.ndarray | None,
    ) -> tuple[tuple[np.ndarray, float, float] | None, dict[str, Any]]:
        """Solve one local correction without making a solver-specific assumption.

        Candidate IK is preferred because it exposes the redundant-arm branch
        set.  Some lightweight or older kinematics backends expose only the
        single-solution ``ik`` API, while some candidate implementations can
        legitimately return no branch for a numerically difficult waypoint.
        Falling back to the public single-solution API keeps the closed loop
        generic and bounded; it never bypasses the subsequent collision and
        measured-center checks.
        """
        diagnostics: dict[str, Any] = {
            "ik_api": "none",
            "ik_candidate_count": 0,
            "ik_single_fallback": False,
        }
        selected = None
        if hasattr(kin, "ik_candidates"):
            diagnostics["ik_api"] = "ik_candidates"
            try:
                solutions = kin.ik_candidates(
                    target, orientation, q_start, max_candidates=8
                )
                diagnostics["ik_candidate_count"] = len(solutions or ())
            except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                solutions = []
            selected = _select_continuous_ik_solution(kin, solutions, q_start)
        if selected is None and hasattr(kin, "ik"):
            diagnostics["ik_single_fallback"] = True
            if diagnostics["ik_api"] == "none":
                diagnostics["ik_api"] = "ik"
            try:
                solution = kin.ik(target, orientation, q_init=q_start)
            except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                solution = None
            if solution is not None and getattr(solution, "success", False):
                q_solution = getattr(solution, "q_arm", None)
                if q_solution is not None:
                    selected = _select_continuous_ik_solution(
                        kin, [solution], q_start
                    )
        return selected, diagnostics

    for iteration in range(1, max_iterations + 1):
        observation = adapter.read_observation(0.0)
        _sync_kinematics_auxiliary_q(kin, observation)
        q_start = np.asarray(
            [observation.joint_positions[name] for name in joints],
            dtype=float,
        )
        midpoint = _measured_grasp_center_world(adapter, side)
        if midpoint is None:
            return None
        error = target_center_world - midpoint
        error_norm = float(np.linalg.norm(error))
        if error_norm <= tolerance:
            return SkillResult(
                True,
                move_to.name,
                metrics={
                    "physical_center_error_m": error_norm,
                    "physical_center_tolerance_m": tolerance,
                    "closed_loop_iterations": float(iteration - 1),
                },
                details={
                    "reason": "measured grasp center reached",
                    "planning_status": "MeasuredCenterClosedLoop",
                    "collision_checked": True,
                    "target_center_world": target_center_world.round(5).tolist(),
                    "final_center_world": midpoint.round(5).tolist(),
                    "corrections": records,
                },
            )

        step = min(error_norm, max_step_m)
        waypoint_world = midpoint + error * (step / max(error_norm, 1.0e-9))
        mapped = _calibrated_model_center_target(
            kin,
            adapter,
            side,
            observation,
            q_start,
            waypoint_world,
        )
        if mapped is None:
            return None
        (
            center_model,
            calibration_rms,
            model_to_world_rotation,
            model_to_world_translation,
        ) = mapped
        # The rigid arm-link registration does not fully model the loaded
        # gripper/tool residual. Preserve the measured current grasp center as
        # the local anchor, then apply only the requested physical
        # displacement. Mapping an absolute target through the nominal URDF
        # offset alone can make a short correction jump to a model center that
        # is centimetres away once the wrist posture changes.
        measured_center = _measured_grasp_center_world(adapter, side)
        try:
            current_center_model = np.asarray(
                kin.grasp_center_fk(q_start)[0], dtype=float
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
            current_center_model = None
        if (
            measured_center is not None
            and current_center_model is not None
            and np.all(np.isfinite(current_center_model))
        ):
            center_model = current_center_model + np.asarray(
                model_to_world_rotation, dtype=float
            ).T @ (
                np.asarray(waypoint_world, dtype=float) - measured_center
            )
        _, current_quat = kin.fk(q_start)
        quat_goal = None
        if target_quat is not None:
            # The public arm contract expresses orientation in the current
            # base frame.  Convert it into the registered reduced-model frame
            # just as the target point is converted below.  For the common
            # zero-yaw case this is numerically the old quaternion; with a
            # rotated/mobile base it remains a physically meaningful pose.
            base_yaw = float((observation.base_pose or (0.0, 0.0, 0.0))[2])
            base_rotation = Rotation.from_euler("z", base_yaw).as_matrix()
            target_rotation_base = Rotation.from_quat(
                [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
            ).as_matrix()
            target_rotation_model = (
                np.asarray(model_to_world_rotation, dtype=float).T
                @ base_rotation
                @ target_rotation_base
            )
            q_target_model = Rotation.from_matrix(target_rotation_model).as_quat()
            quat_goal = np.asarray(
                [q_target_model[3], q_target_model[0], q_target_model[1], q_target_model[2]],
                dtype=float,
            )
            rotation = Rotation.from_quat(
                [quat_goal[1], quat_goal[2], quat_goal[3], quat_goal[0]]
            )
            target_ee = center_model - rotation.apply(
                np.asarray(kin.grasp_center_offset_local, dtype=float)
            )
            selected, ik_diagnostics = solve_local_ik(target_ee, quat_goal)
            if selected is None:
                quat_goal = None
                orientation_relaxed = True
        if quat_goal is None:
            current_rotation = Rotation.from_quat(
                [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
            )
            target_ee = center_model - current_rotation.apply(
                np.asarray(kin.grasp_center_offset_local, dtype=float)
            )
            selected, ik_diagnostics = solve_local_ik(target_ee, None)
        if selected is None:
            return SkillResult(
                False,
                move_to.name,
                metrics={
                    "physical_center_error_m": error_norm,
                    "physical_center_tolerance_m": tolerance,
                    "closed_loop_iterations": float(iteration - 1),
                    "calibration_rms_m": calibration_rms,
                },
                details={
                    "reason": "measured-center IK failed",
                    "failure_code": "measured_center_ik_failed",
                    "target_center_world": target_center_world.round(5).tolist(),
                    "current_center_world": midpoint.round(5).tolist(),
                    "target_ee_model": np.asarray(target_ee).round(5).tolist(),
                    "q_start": q_start.round(5).tolist(),
                    "orientation_relaxed": orientation_relaxed,
                    **ik_diagnostics,
                    "corrections": records,
                },
            )
        q_goal, continuity, margin = selected
        base_pose = observation.base_pose or (0.0, 0.0, 0.0)
        checker = CollisionChecker(
            kin,
            obstacles_from_scene(
                scene,
                exclude=tuple(exclude_objects or ()),
                include_ground=True,
            ),
        )
        trajectory, trajectory_details = _measured_center_trajectory(
            checker,
            kin,
            q_start,
            q_goal,
            base_pose=base_pose,
            model_to_world_rotation=model_to_world_rotation,
            model_to_world_translation=model_to_world_translation,
            side=side,
            speed_scale=float(speed_scale),
        )
        if trajectory is None:
            last_collision_link = trajectory_details.get("collision_link")
            return SkillResult(
                False,
                move_to.name,
                metrics={
                    "physical_center_error_m": error_norm,
                    "physical_center_tolerance_m": tolerance,
                    "closed_loop_iterations": float(iteration - 1),
                    "calibration_rms_m": calibration_rms,
                },
                details={
                    "reason": "measured-center local path collides",
                    "failure_code": "measured_center_collision",
                    "collision_checked": True,
                    "collision_link": last_collision_link,
                    "corrections": records,
                    **trajectory_details,
                },
            )
        for q_target in trajectory[1:]:
            adapter.set_targets(
                position={joint: float(q_target[index]) for index, joint in enumerate(joints)},
                velocity={},
            )
            adapter.step()
            if step_hook is not None:
                step_hook()
        # Give the physical drive a short settling window before measuring the
        # next correction.  The next iteration still has a bounded target and
        # never trusts the nominal model endpoint as evidence.
        for _ in range(12):
            adapter.step()
            if step_hook is not None:
                step_hook()
        final_midpoint = _measured_grasp_center_world(adapter, side)
        if final_midpoint is None:
            return None
        final_error = float(np.linalg.norm(target_center_world - final_midpoint))
        records.append(
            {
                "iteration": iteration,
                "error_before_m": error_norm,
                "error_after_m": final_error,
                "step_m": step,
                "calibration_rms_m": calibration_rms,
                **trajectory_details,
                "continuity_cost": float(continuity),
                "goal_margin_rad": float(margin),
                "q_start": q_start.round(5).tolist(),
                "q_goal": np.asarray(q_goal).round(5).tolist(),
            }
        )
        if final_error > error_norm + 0.02 and error_norm > tolerance:
            # A live torso/base change made this correction move away from the
            # physical target.  Stop with a factual failure so the outer agent
            # can replan from the new observation instead of repeating an
            # uninformative model-only trajectory.
            return SkillResult(
                False,
                move_to.name,
                metrics={
                    "physical_center_error_m": final_error,
                    "physical_center_tolerance_m": tolerance,
                    "closed_loop_iterations": float(iteration),
                    "calibration_rms_m": calibration_rms,
                },
                details={
                    "reason": "measured-center correction made no progress",
                    "failure_code": "measured_center_no_progress",
                    "collision_checked": True,
                    "collision_link": last_collision_link,
                    "orientation_relaxed": orientation_relaxed,
                    "corrections": records,
                },
            )

    final_midpoint = _measured_grasp_center_world(adapter, side)
    final_error = (
        float(np.linalg.norm(target_center_world - final_midpoint))
        if final_midpoint is not None
        else float("inf")
    )
    return SkillResult(
        False,
        move_to.name,
        metrics={
            "physical_center_error_m": final_error,
            "physical_center_tolerance_m": tolerance,
            "closed_loop_iterations": float(len(records)),
        },
        details={
            "reason": "measured grasp center tolerance not reached",
            "failure_code": "measured_center_tolerance_not_reached",
            "collision_checked": True,
            "collision_link": last_collision_link,
            "orientation_relaxed": orientation_relaxed,
            "target_center_world": target_center_world.round(5).tolist(),
            "final_center_world": final_midpoint.round(5).tolist() if final_midpoint is not None else None,
            "corrections": records,
        },
    )


def _arm_move_to_position_only_local(
    move_to: "ArmMoveTo",
    adapter: Any,
    scene: Any,
    *,
    goal_ee: np.ndarray,
    side: str,
    speed_scale: float,
    exclude_objects: list[str] | None,
    position_tolerance: float,
    step_hook: Callable[[], None] | None,
) -> SkillResult | None:
    """Execute a position-only local approach with a certified IK branch.

    A low-workspace grasp may not admit the calibrated tool orientation from
    the current safe posture, even though the target position is reachable.
    This fallback keeps the approach bounded and collision checked; the later
    measured alignment primitive remains responsible for the final finger
    orientation and window.
    """
    kin = for_side(move_to.kin, side)
    if kin is None:
        return None
    joints = ARM_JOINTS_BY_SIDE[side]
    target = np.asarray(goal_ee, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return None
    obs = adapter.read_observation(0.0)
    _sync_kinematics_auxiliary_q(kin, obs)
    q_start = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
    if hasattr(kin, "ik_candidates"):
        solutions = kin.ik_candidates(target, None, q_start, max_candidates=8)
    elif hasattr(kin, "ik"):
        solution = kin.ik(target, None, q_init=q_start)
        solutions = [solution] if solution.success and solution.q_arm is not None else []
    else:
        return None
    selected = _select_continuous_ik_solution(kin, solutions, q_start)
    if selected is None:
        return None
    q_goal, continuity, margin = selected

    from r1pro_data_gen.methods.collision import CollisionChecker, check_path, obstacles_from_scene
    from r1pro_data_gen.methods.manipulation.mplib_path import _minimum_jerk_trajectory
    from r1pro_data_gen.robot.robot_config import R1PRO_LOCAL_ARM_MIN_TRAJECTORY_S

    trajectory, _, _ = _minimum_jerk_trajectory(
        np.asarray([q_start, q_goal]),
        speed_scale=float(speed_scale),
        side=side,
        min_duration_s=R1PRO_LOCAL_ARM_MIN_TRAJECTORY_S,
    )
    base_pose = obs.base_pose or (0.0, 0.0, 0.0)
    free, _, collision_link = check_path(
        CollisionChecker(
            kin,
            obstacles_from_scene(
                scene,
                exclude=tuple(exclude_objects or ()),
                include_ground=True,
            ),
        ),
        list(trajectory),
        base_xy=(float(base_pose[0]), float(base_pose[1])),
        base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
        dense=8,
    )
    if not free:
        return None
    for q_target in trajectory[1:]:
        adapter.set_targets(
            position={joint: float(q_target[index]) for index, joint in enumerate(joints)},
            velocity={},
        )
        adapter.step()
        if step_hook is not None:
            step_hook()
    final_actual = q_start.copy()
    stable_steps = 0
    for _ in range(30):
        adapter.step()
        if step_hook is not None:
            step_hook()
        settled = adapter.read_observation(0.0)
        final_actual = np.asarray([settled.joint_positions[name] for name in joints], dtype=float)
        if float(np.max(np.abs(final_actual - q_goal))) < _FINAL_ERROR_TOL:
            stable_steps += 1
            if stable_steps >= 5:
                break
        else:
            stable_steps = 0
    final_pos, _ = kin.fk(final_actual)
    position_error = float(np.linalg.norm(np.asarray(final_pos) - target))
    tolerance = max(0.06, 2.0 * float(position_tolerance))
    success = position_error <= tolerance
    return SkillResult(
        success,
        move_to.name,
        metrics={
            "final_position_error_m": position_error,
            "position_tolerance_m": tolerance,
            "goal_margin_rad": float(margin),
            "continuity_cost": float(continuity),
            "stable_steps": float(stable_steps),
        },
        details={
            "reason": "position-only local IK reached" if success else "position-only local tracking error",
            "planning_status": "PositionOnlyMarginBestIK",
            "collision_checked": True,
            "collision_link": collision_link,
            "final_target_q": np.asarray(q_goal).round(5).tolist(),
            "final_actual_q": np.asarray(final_actual).round(5).tolist(),
        },
    )


def _arm_move_to_margin_best(
    move_to: "ArmMoveTo",
    adapter: Any,
    scene: Any,
    *,
    goal_ee: np.ndarray,
    quat: np.ndarray,
    side: str,
    speed_scale: float,
    exclude_objects: list[str] | None,
    step_hook: Callable[[], None] | None,
) -> SkillResult | None:
    """Fallback for grasp_center targets: margin-best IK + smooth execution.

    Used only after direct planning and the staging two-waypoint path both
    fail.  The goal is a short local move next to the support surface (align or
    descend to grasp), where multi-seed IK prefers the branch with the most
    joint-limit margin and the trajectory is short enough that the omitted OMPL
    planning does not jeopardize safety.  Returns ``None`` so the caller keeps
    the original planning failure when even this cannot be solved.
    """
    kin = for_side(move_to.kin, side)
    if kin is None:
        return None
    obs = adapter.read_observation(0.0)
    _sync_kinematics_auxiliary_q(kin, obs)
    joints = ARM_JOINTS_BY_SIDE[side]
    q_start = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
    goal_ee = np.asarray(goal_ee, dtype=float)
    _, quat0 = kin.fk(q_start)
    requested_rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    center_goal = goal_ee + requested_rot.apply(
        np.asarray(kin.grasp_center_offset_local, dtype=float)
    )
    candidates: list[tuple[np.ndarray, float, np.ndarray]] = []

    def _evaluate(quat_goal: np.ndarray) -> None:
        quat_goal = np.asarray(quat_goal, dtype=float) / np.linalg.norm(quat_goal)
        # Re-project the grasp-center goal to the EE link for this orientation:
        # the finger-midpoint offset lives in the gripper frame, so rotating
        # the grasp changes the EE target even for the same object point.
        rot = Rotation.from_quat([quat_goal[1], quat_goal[2], quat_goal[3], quat_goal[0]])
        ee_target = np.asarray(center_goal, dtype=float) - rot.apply(
            np.asarray(kin.grasp_center_offset_local, dtype=float)
        )
        if hasattr(kin, "ik_candidates"):
            solutions = kin.ik_candidates(ee_target, quat_goal, q_start, max_candidates=8)
        elif hasattr(kin, "ik"):
            solution = kin.ik(ee_target, quat_goal, q_init=q_start)
            solutions = [solution] if solution.success and solution.q_arm is not None else []
        else:
            solutions = []
        for solution in solutions:
            if not solution.success or solution.q_arm is None:
                continue
            q = np.asarray(solution.q_arm, dtype=float)
            margin = float(np.min(np.minimum(q - kin.lower, kin.upper - q)))
            candidates.append((q, margin, quat_goal.copy()))

    # Evaluate the caller's pose first, then the calibrated robot capability and
    # the current pose.  A target_z_axis or mirrored LLM quaternion can therefore
    # recover through the same certified collision gate without changing the
    # task-level request.
    orientation_candidates: list[np.ndarray] = [np.asarray(quat, dtype=float)]
    from r1pro_data_gen.robot.robot_config import R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE

    orientation_candidates.append(
        np.asarray(R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE[side], dtype=float)
    )
    orientation_candidates.append(np.asarray(quat0, dtype=float))
    unique_orientations: list[np.ndarray] = []
    for candidate in orientation_candidates:
        normalized = candidate / np.linalg.norm(candidate)
        if not any(
            abs(float(np.dot(normalized, existing))) > 1.0 - 1e-7
            for existing in unique_orientations
        ):
            unique_orientations.append(normalized)
    for candidate in unique_orientations:
        _evaluate(candidate)
    if not candidates:
        # The requested and calibrated grasp orientations may both be close to
        # a joint limit. Try small tool-axis yaw variations as a generic final
        # recovery, keeping the current object center fixed.
        base_rot = Rotation.from_quat([quat0[1], quat0[2], quat0[3], quat0[0]])
        for offset_deg in (15.0, -15.0, 30.0, -30.0, 45.0, -45.0):
            rot = base_rot * Rotation.from_euler("z", np.deg2rad(offset_deg))
            qv = rot.as_quat()
            quat_off = np.asarray([qv[3], qv[0], qv[1], qv[2]], dtype=float)
            _evaluate(quat_off)
    if not candidates:
        return None
    from r1pro_data_gen.methods.manipulation.mplib_path import _minimum_jerk_trajectory
    from r1pro_data_gen.methods.collision import CollisionChecker, check_path, obstacles_from_scene

    base_pose = obs.base_pose or (0.0, 0.0, 0.0)
    collision_checker = CollisionChecker(
        kin,
        obstacles_from_scene(
            scene,
            exclude=tuple(exclude_objects or ()),
            include_ground=True,
        ),
    )
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected: tuple[np.ndarray, float, np.ndarray, Any] | None = None
    for q_goal, margin, quat_goal in candidates:
        trajectory, _, _ = _minimum_jerk_trajectory(
            np.asarray([q_start, q_goal]), speed_scale=float(speed_scale), side=side
        )
        free, _, collision_link = check_path(
            collision_checker,
            list(trajectory),
            base_xy=(float(base_pose[0]), float(base_pose[1])),
            base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
            dense=8,
        )
        if free:
            selected = (q_goal, margin, quat_goal, trajectory)
            break
    if selected is None:
        return None
    q_goal, margin, quat_goal, trajectory = selected
    collision_link = None
    for q_target in trajectory[1:]:
        adapter.set_targets(
            position={joint: float(q_target[i]) for i, joint in enumerate(joints)},
            velocity={},
        )
        adapter.step()
        if step_hook is not None:
            step_hook()
    final_obs = adapter.read_observation(0.0)
    q_final = np.asarray([final_obs.joint_positions[name] for name in joints], dtype=float)
    final_pos, final_quat = kin.grasp_center_fk(q_final)
    pos_error = float(np.linalg.norm(final_pos - center_goal))
    # The fallback runs without OMPL and ends with the fingers pressed against
    # the object (strong two-finger contact, the object nudged a few cm).  That
    # is a grasp-ready state even when the commanded finger-center differs from
    # the IK goal by the contact displacement, so accept a looser tolerance
    # here than the certified path gate.
    success = pos_error <= 0.06
    return SkillResult(
        success,
        move_to.name,
        metrics={
            "final_position_error_m": pos_error,
            "goal_margin_rad": float(margin),
        },
        details={
            "reason": "goal reached" if success else "margin-best IK fallback exceeded tolerance",
            "planning_status": "MarginBestIK",
            "q_goal": q_goal.round(5).tolist(),
            "collision_checked": True,
            "collision_link": collision_link,
        },
    )


def _arm_move_through_margin_best_fallback(
    move_through: "ArmMoveThrough",
    adapter: Any,
    scene: Any,
    *,
    waypoints: tuple[Any, ...],
    side: str,
    speed_scale: float,
    step_hook: Callable[[], None] | None,
    carried_context: Any = None,
) -> SkillResult | None:
    """Replay a waypoint sequence as one collision-certified trajectory.

    Each waypoint is first chained in Cartesian task space from the live
    branch so the arm does not snap to a distant IK solution. If that
    interpolant is unavailable, a collision-checked joint segment is used
    instead. Adjacent segments are concatenated and retimed once so the
    executed motion has a single C2 reference instead of a stop at every
    authored waypoint.
    """
    kin = for_side(move_through.kin, side)
    if kin is None:
        return None
    from r1pro_data_gen.methods.manipulation.mplib_path import _minimum_jerk_trajectory
    from r1pro_data_gen.methods.manipulation.taskspace import plan_task_path
    from r1pro_data_gen.methods.collision import CollisionChecker, check_path, obstacles_from_scene

    joints = ARM_JOINTS_BY_SIDE[side]
    obs = adapter.read_observation(0.0)
    q_cur = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
    base_pose = obs.base_pose or (0.0, 0.0, 0.0)
    base_xy = (float(base_pose[0]), float(base_pose[1]))
    base_yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
    geometric_parts: list[np.ndarray] = []
    planned_exclusions: list[tuple[str, ...]] = []
    vel_limits = np.asarray(for_side(move_through.vel_limits, side), dtype=float)

    def _make_checker(excluded: set[str] | tuple[str, ...]) -> CollisionChecker | None:
        try:
            return CollisionChecker(
                kin,
                obstacles_from_scene(scene, exclude=set(excluded), include_ground=True),
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _segment_free(checker: CollisionChecker, geometric: np.ndarray, segment_speed: float) -> np.ndarray | None:
        trajectory, _, _ = _minimum_jerk_trajectory(
            geometric,
            speed_scale=float(segment_speed),
            side=side,
        )
        free, _, _ = check_path(
            checker,
            list(trajectory),
            base_xy=base_xy,
            base_yaw=base_yaw,
            dense=8,
        )
        return trajectory if free else None

    for waypoint in waypoints:
        if not waypoint.poses:
            return None
        excluded = tuple(sorted(set(getattr(waypoint, "exclude_objects", ()) or ())))
        checker = _make_checker(excluded)
        if checker is None:
            return None
        segment_speed = float(
            waypoint.speed_scale if waypoint.speed_scale is not None else speed_scale
        )
        selected: np.ndarray | None = None
        if hasattr(kin, "ik") and hasattr(kin, "fk"):
            for position, orientation in waypoint.poses:
                position = np.asarray(position, dtype=float)
                orientation = np.asarray(orientation, dtype=float)
                orientation = orientation / max(float(np.linalg.norm(orientation)), 1e-12)
                planned = plan_task_path(kin, position, orientation, q_cur)
                if not planned.success or len(planned.waypoints) < 2:
                    continue
                geometric = np.asarray(planned.waypoints, dtype=float)
                trajectory = _segment_free(checker, geometric, segment_speed)
                if trajectory is not None:
                    selected = geometric
                    break
        if selected is None:
            candidates: list[tuple[np.ndarray, float]] = []
            for position, orientation in waypoint.poses:
                position = np.asarray(position, dtype=float)
                orientation = np.asarray(orientation, dtype=float)
                orientation = orientation / max(float(np.linalg.norm(orientation)), 1e-12)
                if hasattr(kin, "ik_candidates"):
                    solutions = kin.ik_candidates(position, orientation, q_cur, max_candidates=6)
                elif hasattr(kin, "ik"):
                    solution = kin.ik(position, orientation, q_init=q_cur)
                    solutions = [solution] if solution.success and solution.q_arm is not None else []
                else:
                    return None
                for solution in solutions:
                    if not solution.success or solution.q_arm is None:
                        continue
                    q = np.asarray(solution.q_arm, dtype=float)
                    margin = float(np.min(np.minimum(q - kin.lower, kin.upper - q)))
                    candidates.append((q, margin))
            if not candidates:
                return None
            for q_goal, _margin in sorted(candidates, key=lambda item: (-item[1], tuple(item[0]))):
                geometric = np.asarray([q_cur, q_goal], dtype=float)
                trajectory = _segment_free(checker, geometric, segment_speed)
                if trajectory is not None:
                    selected = geometric
                    break
        if selected is None:
            return None
        geometric_parts.append(selected)
        planned_exclusions.append(excluded)
        q_cur = np.asarray(selected[-1], dtype=float)

    complete_path = np.concatenate(
        [geometric_parts[0]] + [part[1:] for part in geometric_parts[1:]],
        axis=0,
    )
    complete_speed = float(
        max(
            (
                waypoint.speed_scale
                if waypoint.speed_scale is not None
                else speed_scale
            )
            for waypoint in waypoints
        )
    )
    trajectory, velocity, _acceleration = _minimum_jerk_trajectory(
        complete_path,
        speed_scale=complete_speed,
        side=side,
    )
    union_exclude = set().union(*[set(items) for items in planned_exclusions]) if planned_exclusions else set()
    checker = _make_checker(union_exclude)
    if checker is None:
        return None
    free, _, _ = check_path(
        checker,
        list(trajectory),
        base_xy=base_xy,
        base_yaw=base_yaw,
        dense=8,
    )
    if not free:
        return None

    if carried_context is not None:
        from r1pro_data_gen.methods.collision import carried_object_path_free

        held_exclusions = tuple(sorted(union_exclude))
        carried_free, _ = carried_object_path_free(
            kin,
            trajectory,
            scene,
            carried_context,
            base_xy=base_xy,
            base_yaw=base_yaw,
            exclude=held_exclusions,
        )
        if not carried_free:
            return None

    expected_effector = (
        None if carried_context is None else f"{side}_gripper_finger_midpoint"
    )
    attachment_failure: dict[str, object] | None = None

    def step_and_verify_attachment() -> None:
        nonlocal attachment_failure
        if step_hook is not None:
            step_hook()
        if carried_context is None:
            return
        if not hasattr(adapter, "attachment_state"):
            attachment_failure = {
                "reason": "adapter does not expose attachment state",
                "object_name": str(carried_context.object_name),
            }
            raise _HeldContextLost("adapter does not expose attachment state")
        state = adapter.attachment_state()
        if state.get(str(carried_context.object_name)) != expected_effector:
            attachment_failure = {
                "reason": "carried object attachment was lost",
                "object_name": str(carried_context.object_name),
            }
            raise _HeldContextLost("carried object attachment was lost")

    try:
        stabilize_base(adapter)
        step_and_verify_attachment()
        execution = ArmTrajectoryFollow(kin, vel_limits).execute(
            adapter,
            scene=scene,
            trajectory=np.asarray(trajectory, dtype=float).tolist(),
            velocities=np.asarray(velocity, dtype=float).tolist(),
            trajectory_dt=float(getattr(adapter, "dt", 1.0 / 60.0)),
            side=side,
            step_hook=step_and_verify_attachment,
        )
    except _HeldContextLost as exc:
        return SkillResult(
            False,
            move_through.name,
            metrics={"held_context_verified": False, "failure_code": "attachment_lost"},
            details={"reason": str(exc), "planning_status": "TaskSpaceChain"},
        )
    if not execution.success:
        return SkillResult(
            False,
            move_through.name,
            metrics=dict(execution.metrics),
            details={"reason": "trajectory execution failed", **execution.details, "planning_status": "TaskSpaceChain"},
        )
    return SkillResult(
        True,
        move_through.name,
        metrics={
            **dict(execution.metrics),
            "fallback_waypoints": float(len(waypoints)),
            "held_context_verified": True,
        },
        details={
            "reason": "all waypoints reached on one certified Cartesian chain",
            "planning_status": "TaskSpaceChain",
            "carried_object_collision_checked": carried_context is not None,
        },
    )


def _quat_error(q_ref: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Small-angle quaternion error vector in wxyz convention."""
    q_ref = q_ref / np.linalg.norm(q_ref)
    q_cur = q_cur / np.linalg.norm(q_cur)
    w1, x1, y1, z1 = q_ref
    w2, x2, y2, z2 = q_cur
    # q_error = q_ref * conjugate(q_cur).  Keeping this convention identical
    # to the IK solver is important: a malformed scalar term can make an
    # actually correct grasp orientation fail the final tolerance check.
    dq = np.array([
        w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
        -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2,
        -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2,
        -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2,
    ])
    if dq[0] < 0:
        dq = -dq
    return 2.0 * dq[1:]


class ArmAlignGripper:
    """Center a gripper around a live object using measured link geometry.

    The command is deliberately a reusable alignment primitive, not a grasp or
    pick-and-place policy.  It repeatedly measures the finger midpoint, maps
    the measured Cartesian correction into the Pinocchio model frame, and uses
    the same certified ``arm_move_to`` implementation as ordinary motion.
    """

    name = "arm_align_gripper"
    tier = "semantic"
    exposed = True
    description = (
        "Center a selected gripper around a live scene object using measured "
        "finger geometry and certified Cartesian arm motion."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Object to center between the gripper fingers", required=True),
        "side": ParamSpec("string", "Arm side", default="left", enum=("left", "right")),
        "position_tolerance": ParamSpec("number", "Maximum object-to-finger-center error (m)", default=0.015, minimum=1e-4),
        "require_vertical_alignment": ParamSpec("boolean", "Also descend until the measured finger center reaches the object height", default=False),
        "require_between_fingers": ParamSpec("boolean", "Require the live object-window geometry to contain the object before reporting alignment success", default=False),
        "surface_tolerance_m": ParamSpec("number", "Measured object-to-finger-window surface tolerance (m)", default=0.012, minimum=0.005, maximum=0.05),
        "max_iterations": ParamSpec("integer", "Maximum measured correction iterations", default=16, minimum=1, maximum=24),
        "trajectory_speed_scale": ParamSpec("number", "Fraction of arm velocity limits used for each correction", default=0.12, minimum=0.02),
        "planning_time": ParamSpec("number", "Planning time per correction (s)", default=1.2, minimum=0.1),
        "ik_candidates": ParamSpec("integer", "Number of online IK branches per correction", default=3, minimum=1),
        "local_radius_m": ParamSpec("number", "Arm-planning obstacle culling radius (m)", default=2.0, minimum=0.5),
        "exclude_objects": ParamSpec("array", "Scene objects excluded during intentional local alignment", default=[]),
        "contact_threshold": ParamSpec("number", "Contact force threshold (N) that stops a measured descent before it pushes the object", default=1.0, minimum=0.1),
        "object_motion_tolerance_m": ParamSpec(
            "number",
            "Maximum pre-attachment target motion (m); defaults to a geometry-derived settling tolerance",
            default=None,
            minimum=1e-4,
        ),
    }

    def __init__(self, kin: Any, vel_limits: np.ndarray, planner: Any):
        self.kin = kin
        self.vel_limits = vel_limits
        self.planner = planner

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        side: str = "left",
        position_tolerance: float = 0.015,
        require_vertical_alignment: bool = False,
        require_between_fingers: bool = False,
        surface_tolerance_m: float = 0.012,
        max_iterations: int = 16,
        trajectory_speed_scale: float = 0.12,
        planning_time: float = 1.2,
        ik_candidates: int = 3,
        local_radius_m: float = 2.0,
        exclude_objects: list[str] | None = None,
        contact_threshold: float = 1.0,
        object_motion_tolerance_m: float | None = None,
        object_motion_reference_position: Sequence[float] | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if not object_name:
            raise ValueError("arm_align_gripper requires object_name")
        side = require_side(side)
        if not hasattr(adapter, "gripper_object_alignment"):
            return SkillResult(False, self.name, details={"reason": "adapter does not provide gripper alignment measurements"})
        stabilize_base(adapter)
        kin = for_side(self.kin, side)
        planner = for_side(self.planner, side)
        vel_limits = np.asarray(for_side(self.vel_limits, side), dtype=float)
        if kin is None:
            return SkillResult(False, self.name, details={"reason": "kinematics backend is unavailable"})
        if scene is None:
            return SkillResult(False, self.name, details={"reason": "arm_align_gripper requires a scene for collision checking"})
        try:
            object_model = scene.object(object_name)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            # Minimal adapters/tests may provide measured alignment without a
            # full SceneModel.  The guard still works with its conservative
            # fallback tolerance; full physical runs always provide geometry.
            object_model = None

        # The measured correction is checked against the scene before execution
        # (see _execute_alignment_trajectory): only the support surface is
        # excluded so the final short descent is legal, while the target object
        # itself stays in the obstacle set so the correction cannot sweep
        # through and push it.
        try:
            adapter.scene_model = scene
        except Exception:
            pass
        last_alignment: dict[str, object] = {}
        lateral_correction_done = False
        # Keep one signed, measured jaw-span direction for this complete
        # local acquisition.  Recomputing it from the changing midpoint
        # after every redundant-arm correction makes the desired orientation
        # chase Cartesian tracking error and can create a retreat loop.
        direction_span_reference: np.ndarray | None = None
        approach_axis_reference: np.ndarray | None = None
        telemetry_adapter = getattr(adapter, "_adapter", adapter)
        try:
            telemetry_adapter._alignment_path_history = []
        except (AttributeError, TypeError, ValueError):
            pass
        object_motion_reference: np.ndarray | None = None
        if object_motion_reference_position is not None:
            reference = np.asarray(object_motion_reference_position, dtype=float)
            if reference.shape != (3,) or not np.all(np.isfinite(reference)):
                raise ValueError(
                    "object_motion_reference_position must be a finite 3-vector"
                )
            object_motion_reference = reference.copy()
        object_motion_tolerance = _pregrasp_object_motion_tolerance(
            object_model,
            object_motion_tolerance_m,
        )
        noncontact_rebaseline_count = 0
        noncontact_rebaseline_diagnostic: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "reason": "not_needed",
        }

        def rebaseline_noncontact_settling(
            current: np.ndarray,
            displacement: float,
        ) -> bool:
            """Rebaseline small support settling while the gripper is remote.

            A dynamic object can settle on its support after another action or
            after the arm reaches a high standoff.  That motion is not an open
            gripper push when the live finger geometry is still well clear and
            the contact sensors explicitly report no finger contact.  Keep the
            no-push guard fail-closed when either piece of evidence is absent.
            """
            nonlocal object_motion_reference, noncontact_rebaseline_count
            noncontact_rebaseline_diagnostic.update(
                {
                    "attempted": True,
                    "accepted": False,
                    "displacement_m": float(displacement),
                    "max_allowed_displacement_m": float(
                        max(2.0 * object_motion_tolerance, 0.006)
                    ),
                }
            )
            if displacement > max(2.0 * object_motion_tolerance, 0.006):
                noncontact_rebaseline_diagnostic["reason"] = "displacement_too_large"
                return False
            if not hasattr(adapter, "finger_contact_forces"):
                noncontact_rebaseline_diagnostic["reason"] = "contact_force_reader_unavailable"
                return False
            contact_events_reader = getattr(adapter, "contact_events", None)
            events_available = False
            target_contact = False
            try:
                alignment = dict(
                    adapter.gripper_object_alignment(object_name, side=side)
                )
                surface_distance = float(alignment.get("surface_distance_m"))
                noncontact_rebaseline_diagnostic["surface_distance_m"] = surface_distance
                clearance_threshold = max(
                    0.03,
                    2.0 * float(surface_tolerance_m),
                )
                if (
                    not np.isfinite(surface_distance)
                    or surface_distance <= clearance_threshold
                ):
                    noncontact_rebaseline_diagnostic["reason"] = "gripper_not_remote"
                    return False
                try:
                    contact_values = adapter.finger_contact_forces(side=side)
                except TypeError:
                    contact_values = adapter.finger_contact_forces()
                forces = tuple(float(value) for value in contact_values)
                threshold = float(contact_threshold)
                noncontact_rebaseline_diagnostic["finger_contact_forces_n"] = [
                    float(value) for value in forces
                ]
                noncontact_rebaseline_diagnostic["contact_threshold_n"] = threshold
                if contact_events_reader is not None:
                    events = tuple(contact_events_reader())
                    events_available = True
                    for event in events:
                        try:
                            event_force = float(event.force_n)
                            body_a = str(event.body_a)
                            body_b = str(event.body_b)
                        except (AttributeError, TypeError, ValueError):
                            continue
                        if event_force > threshold and object_name in {body_a, body_b}:
                            target_contact = True
                            break
                    noncontact_rebaseline_diagnostic["target_contact_event"] = bool(
                        target_contact
                    )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                noncontact_rebaseline_diagnostic["reason"] = "contact_evidence_unavailable"
                return False
            if target_contact:
                noncontact_rebaseline_diagnostic["reason"] = "target_contact_detected"
                return False
            if (
                not forces
                or not np.isfinite(threshold)
                or any(not np.isfinite(value) for value in forces)
            ):
                noncontact_rebaseline_diagnostic["reason"] = "invalid_contact_force_sample"
                return False
            # ``finger_contact_forces`` is an aggregate fallback for adapters
            # without identity-aware events.  When identity-aware events are
            # available, a nonzero force from an unrelated body (for example a
            # support surface) must not be mistaken for contact with the target.
            if max(forces) > threshold and not events_available:
                noncontact_rebaseline_diagnostic["reason"] = "non_target_contact_unverified"
                return False
            object_motion_reference = current.copy()
            noncontact_rebaseline_count += 1
            noncontact_rebaseline_diagnostic.update(
                {
                    "accepted": True,
                    "reason": "remote_without_target_contact",
                    "events_available": events_available,
                }
            )
            return True

        def object_motion_violation(
            current_position: np.ndarray | None = None,
        ) -> dict[str, Any] | None:
            """Return a structured violation if the open gripper pushed the object.

            Prefer the adapter's direct live pose API.  The alignment
            measurement is a compatible fallback for lightweight adapters and
            keeps this guard independent of any particular simulator.
            """
            if object_motion_reference is None:
                return None
            if hasattr(adapter, "is_object_attached"):
                try:
                    if bool(adapter.is_object_attached(object_name)):
                        return None
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    pass
            current = (
                np.asarray(current_position, dtype=float)
                if current_position is not None
                else _alignment_object_position(adapter, object_name)
            )
            if current is None:
                return None
            if current.shape != (3,) or not np.all(np.isfinite(current)):
                return None
            displacement = float(np.linalg.norm(current - object_motion_reference))
            if displacement <= object_motion_tolerance:
                return None
            if rebaseline_noncontact_settling(current, displacement):
                return None
            return {
                "reason": "target object moved before the gripper established attachment",
                "object_name": object_name,
                "object_motion_m": displacement,
                "object_motion_tolerance_m": object_motion_tolerance,
                "initial_object_position": object_motion_reference.round(6).tolist(),
                "current_object_position": current.round(6).tolist(),
                "noncontact_rebaseline": dict(noncontact_rebaseline_diagnostic),
            }

        for iteration in range(1, int(max_iterations) + 1):
            try:
                _sync_kinematics_auxiliary_q(kin, adapter.read_observation(0.0))
                last_alignment = dict(adapter.gripper_object_alignment(object_name, side=side))
            except (KeyError, RuntimeError, ValueError) as exc:
                return SkillResult(False, self.name, details={"reason": str(exc), "iteration": iteration})
            midpoint = np.asarray(last_alignment.get("finger_midpoint", ()), dtype=float)
            object_position = np.asarray(last_alignment.get("object_position", ()), dtype=float)
            if midpoint.shape != (3,) or object_position.shape != (3,):
                return SkillResult(False, self.name, details={"reason": "alignment measurement must contain two 3-vectors", "iteration": iteration})
            if object_motion_reference is None:
                object_motion_reference = object_position.copy()
            else:
                violation = object_motion_violation(object_position)
                if violation is not None:
                    return _object_motion_failure(
                        self.name,
                        iteration=iteration,
                        alignment=last_alignment,
                        violation=violation,
                    )
            if direction_span_reference is None:
                direction_span_reference = _object_window_span_direction(
                    adapter,
                    side,
                    object_name,
                )
            if approach_axis_reference is None:
                approach_xy = object_position[:2] - midpoint[:2]
                approach_norm = float(np.linalg.norm(approach_xy))
                if approach_norm > 1.0e-8 and np.all(np.isfinite(approach_xy)):
                    approach_axis_reference = np.asarray(
                        [approach_xy[0] / approach_norm, approach_xy[1] / approach_norm, 0.0],
                        dtype=float,
                    )
            # The finite finger segment is the *acceptance* geometry, but the
            # motion anchor is the physical midpoint of that same segment.
            # Driving the whole gripper by ``object - closest_point`` can move
            # the segment's midpoint away from the object as its orientation
            # changes, leaving the object just outside one endpoint (observed
            # as a small surface distance with segment_fraction < 0).  Move
            # the measured midpoint toward the object center; keep the
            # closest-point/finite-segment test below as the independent
            # grasp-window gate.  Adapters predating ``closest_point`` remain
            # fully supported because the midpoint is always required.
            alignment_reference = midpoint
            closest_point = _optional_alignment_point(last_alignment.get("closest_point"))
            if closest_point is not None:
                alignment_reference = closest_point
            window_correction = np.asarray(
                object_position - alignment_reference,
                dtype=float,
            )
            correction = np.asarray(object_position - midpoint, dtype=float)
            # Correct in full XYZ.  A pure horizontal recenter at the pre-grasp
            # height drives the online IK branch into its joint limits (the arm
            # reaches sideways), while descending to the object height together
            # with the horizontal correction keeps the joint solution inside
            # the workspace -- the measured delta is a short local move next to
            # the support surface, not a table-edge traversal.
            horizontal_error = float(np.linalg.norm(correction[:2]))
            window_horizontal_error = float(np.linalg.norm(window_correction[:2]))
            full_error = float(np.linalg.norm(correction))
            between = _alignment_window_ready(last_alignment, float(surface_tolerance_m))
            raw_between = bool(last_alignment.get("between_fingers", False))
            # Success is measured by the object being inside the gripper window
            # (between_fingers) or horizontally centered.  The finger-link
            # origins sit on the upper part of the finger meshes, so the
            # finger-midpoint can legitimately sit several cm above the object
            # centre even in a correct pinch; a full-XYZ distance is therefore
            # not the alignment gate.  Vertical alignment means the fingers have
            # descended to the object's height band: the midpoint may be above
            # the object centre (the finger mesh upper edge touches the object
            # top), but must not be far below it or hang far above it.
            vertical_offset = float(correction[2])
            physical_vertical_window = _alignment_physical_vertical_window_ready(
                last_alignment
            )
            vertically_aligned = (
                physical_vertical_window
                or (
                    -_ALIGN_ABOVE_OBJECT_MARGIN_M <= vertical_offset
                    <= float(position_tolerance) + _ALIGN_ABOVE_OBJECT_MARGIN_M
                )
            )
            centered = between if require_between_fingers else (
                between or window_horizontal_error <= float(position_tolerance)
            )
            if centered and (
                not require_vertical_alignment or vertically_aligned
            ):
                if not _one_sided_finger_contact(adapter, side, float(contact_threshold)):
                    return SkillResult(
                        True,
                        self.name,
                        metrics=_alignment_metrics(
                            horizontal_error=horizontal_error,
                            vertical_offset=float(correction[2]),
                            vertical_tolerance=float(position_tolerance),
                            surface_distance=last_alignment.get("surface_distance_m"),
                            between=between,
                            iterations=iteration,
                            failure_code=None,
                        ),
                        details={
                            "finger_midpoint": midpoint.tolist(),
                            "object_position": object_position.tolist(),
                            "vertical_offset_m": float(correction[2]),
                            "between_fingers": between,
                            "raw_between_fingers": raw_between,
                            "require_between_fingers": bool(require_between_fingers),
                            "surface_tolerance_m": float(surface_tolerance_m),
                            **{
                                key: last_alignment[key]
                                for key in (
                                    "segment_fraction",
                                    "surface_distance_m",
                                )
                                if key in last_alignment
                            },
                        },
                    )
                # A geometric window without two-sided load is not close-ready.
                # Keep the bounded descent instead of handing a one-finger graze
                # to gripper_grasp.
            # Cap each correction at a small step.  A single large Cartesian
            # jump lets online IK flip to a distant branch whose joint-space
            # path through the tabletop gap is much harder for OMPL; stepping
            # a few cm at a time keeps the branch continuous and the segments
            # short and certifiable (same rationale as the trusted policy's
            # recenter loop).
            # When the object is not yet inside the jaw window, do not make a
            # diagonal descent toward it while also correcting lateral error.
            # A finger can touch the object's side first (especially with a
            # posture-dependent USD/URDF tool offset), which causes the
            # contact guard below to stop before vertical alignment and leaves
            # every retry at the same bad height.  Center horizontally at the
            # current standoff first; only the following iteration descends.
            correction_for_step = np.asarray(correction, dtype=float)
            if (
                require_between_fingers
                and not between
                and abs(vertical_offset) > float(position_tolerance)
                and (
                    not lateral_correction_done
                    or horizontal_error > float(position_tolerance)
                )
            ):
                correction_for_step = correction_for_step.copy()
                correction_for_step[2] = 0.0
                # Once the measured lateral error is within the declared
                # position tolerance, the next correction may descend.  A
                # previous implementation gated on a tiny 1e-4 m residual,
                # which could keep a gripper permanently above an object when
                # its physical finger midpoint was a few millimetres off.
                lateral_correction_done = True
            if approach_axis_reference is not None:
                # A redundant arm can realize a short Cartesian target with
                # a small tangential drift even when its midpoint error gets
                # smaller.  For a parallel jaw this changes the object-to-jaw
                # normal and makes the next fixed-direction window approach
                # hit one finger first.  Treat the first measured object
                # approach as a local line: correct tangential error while
                # still at a geometry-derived safe radial distance, then
                # advance toward the object along that line.  If the current
                # point is already too close to the object for a tangential
                # correction, retreat along the same line before correcting
                # the tangent.  The actual finger-box certificate remains the
                # authority for every resulting trajectory.
                axis = np.asarray(approach_axis_reference, dtype=float)
                correction_xy = np.asarray(correction[:2], dtype=float)
                axis_xy = axis[:2]
                radial_error = float(np.dot(correction_xy, axis_xy))
                tangent_error = correction_xy - radial_error * axis_xy
                tangent_norm = float(np.linalg.norm(tangent_error))
                tangent_tolerance = max(0.005, float(position_tolerance))
                line_recovery = False
                if tangent_norm > tangent_tolerance:
                    safe_distance = _alignment_orientation_clearance(
                        scene,
                        object_name,
                    )
                    if (
                        safe_distance is not None
                        and np.isfinite(float(safe_distance))
                        and radial_error < float(safe_distance)
                    ):
                        correction_for_step[:2] = -axis_xy * (
                            float(safe_distance) - radial_error
                        )
                        line_recovery = True
                    else:
                        correction_for_step[:2] = tangent_error
                    # Do not descend while recovering the horizontal line;
                    # the object remains protected by the support and finger
                    # path gates until the approach is radial again.
                    correction_for_step[2] = 0.0
                else:
                    correction_for_step[:2] = radial_error * axis_xy
                try:
                    telemetry_adapter._alignment_line_recovery = bool(line_recovery)
                    telemetry_adapter._alignment_radial_error_m = float(radial_error)
                    telemetry_adapter._alignment_tangential_error_m = float(
                        tangent_norm
                    )
                except (AttributeError, TypeError, ValueError):
                    pass
            minimum_midpoint_z = _alignment_minimum_midpoint_z(
                adapter,
                side,
                midpoint,
                scene,
                object_model,
            )
            if minimum_midpoint_z is not None:
                # A center-only height target is insufficient for the R1Pro
                # mesh: the lower finger box extends below its link origin.
                # Keep the entire profiled box above the source plane before
                # allowing a measured descent; the static and live box gates
                # remain authoritative if the wrist reorients afterwards.
                correction_for_step = correction_for_step.copy()
                correction_for_step[2] = max(
                    float(correction_for_step[2]),
                    float(minimum_midpoint_z) - float(midpoint[2]),
                )
            step_norm = float(np.linalg.norm(correction_for_step))
            if step_norm <= 1e-9:
                # The midpoint may already be centered while the finite
                # segment is still outside its valid window (for example at
                # an endpoint). Use the closest-point residual only as a
                # bounded window/orientation correction in that case.
                correction_for_step = np.asarray(window_correction, dtype=float)
                step_norm = float(np.linalg.norm(correction_for_step))
            if step_norm <= 1e-9:
                return SkillResult(
                    False,
                    self.name,
                    metrics=_alignment_metrics(
                        horizontal_error=horizontal_error,
                        vertical_offset=float(correction[2]),
                        vertical_tolerance=float(position_tolerance),
                        surface_distance=last_alignment.get("surface_distance_m"),
                        between=between,
                        iterations=iteration,
                        failure_code="object_window_not_reached",
                    ),
                    details={
                        "reason": "gripper segment cannot improve its measured window",
                        "between_fingers": between,
                        "raw_between_fingers": raw_between,
                    },
                )
            step = min(step_norm, _ALIGN_MAX_STEP_M)
            moved = False
            move_details: dict[str, Any] | None = None
            step_retry_count = 0
            # A Cartesian cap alone is not a joint-space continuity proof:
            # near a singular posture, even a 10 cm vertical move can map to
            # a branch whose largest joint delta exceeds the local limit.  A
            # pre-execution rejection is safe to retry because the measured
            # robot/object state has not advanced.  Halve only for those
            # bounded IK/path rejections; collision, object motion, and
            # runtime-certificate failures remain fail-closed.
            while True:
                step_vector = correction_for_step * (step / step_norm)
                target_ee, target_quat = _measured_gripper_correction(
                    kin, adapter, side, step_vector
                )
                if target_ee is None or target_quat is None:
                    return SkillResult(
                        False,
                        self.name,
                        details={
                            "reason": "could not register live gripper geometry",
                            "iteration": iteration,
                        },
                    )
                # Execute the correction as a continuous IK motion instead of
                # a single MPlib plan. Re-read the state for each subdivision
                # so a future adapter can safely implement a pre-execution
                # recovery without reusing a stale joint vector.
                obs = adapter.read_observation(0.0)
                _sync_kinematics_auxiliary_q(kin, obs)
                q_now = np.asarray(
                    [obs.joint_positions[j] for j in ARM_JOINTS_BY_SIDE[side]],
                    dtype=float,
                )
                try:
                    moved, move_details = _align_continuous_move(
                        kin,
                        adapter,
                        side,
                        q_now,
                        np.asarray(target_ee, dtype=float),
                        float(trajectory_speed_scale),
                        step_hook,
                        scene=scene,
                        exclude_objects=tuple(exclude_objects or ()),
                        object_motion_guard=object_motion_violation,
                        target_quat=np.asarray(target_quat, dtype=float),
                        protected_object_name=object_name,
                        target_center_world=(
                            midpoint + step_vector
                            if hasattr(adapter, "end_effector_poses")
                            else None
                        ),
                        ik_candidates=max(1, int(ik_candidates)),
                        direction_span_override=direction_span_reference,
                    )
                except TypeError as exc:
                    # Keep compatibility with external/test doubles that wrap
                    # the pre-extension private helper signature. The
                    # production helper accepts the guard, so a TypeError from
                    # it is not silently swallowed here.
                    if "object_motion_guard" not in str(exc):
                        raise
                    moved, move_details = _align_continuous_move(
                        kin,
                        adapter,
                        side,
                        q_now,
                        np.asarray(target_ee, dtype=float),
                        float(trajectory_speed_scale),
                        step_hook,
                        scene=scene,
                        exclude_objects=tuple(exclude_objects or ()),
                    )
                if moved:
                    break
                failure_code = (
                    move_details.get("failure_code")
                    if isinstance(move_details, dict)
                    else None
                )
                if (
                    failure_code
                    not in {"alignment_path_unavailable", "alignment_ik_failed"}
                    or step_retry_count >= _ALIGN_STEP_RETRY_LIMIT
                    or step <= _ALIGN_MIN_STEP_M
                ):
                    break
                step = max(_ALIGN_MIN_STEP_M, 0.5 * step)
                step_retry_count += 1
            if not moved:
                if isinstance(move_details, dict) and move_details.get(
                    "failure_code"
                ) == "object_moved_before_grasp":
                    return _object_motion_failure(
                        self.name,
                        iteration=iteration,
                        alignment=last_alignment,
                        violation=move_details,
                        motion=move_details,
                    )
                return SkillResult(
                    False,
                    self.name,
                    metrics=_alignment_metrics(
                        horizontal_error=horizontal_error,
                        vertical_offset=float(correction[2]),
                        vertical_tolerance=float(position_tolerance),
                        surface_distance=last_alignment.get("surface_distance_m"),
                        between=between,
                        iterations=iteration,
                        failure_code="correction_motion_failed",
                    ),
                    details={"reason": "certified correction motion failed", "motion": move_details},
                )
            # Contact stop: a measured descent can reach the object before the
            # geometric window closes (e.g. a standoff taller than the window
            # tolerance).  Both fingers must be loaded before this is treated
            # as grasp-ready. A one-sided graze with the object still high in
            # the jaw is not a pinch; keep the bounded descent. A one-sided
            # graze at the pinch height is a failure, not a close.
            try:
                try:
                    contact_values = adapter.finger_contact_forces(side=side)
                except TypeError:
                    contact_values = adapter.finger_contact_forces()
                forces = tuple(float(v) for v in contact_values)
                if forces and max(forces) > float(contact_threshold):
                    # The motion above may have closed the jaw window.  The
                    # alignment sampled before the move is not authoritative
                    # once contact is observed: using it here can report
                    # ``contact_not_centered`` even though the post-motion
                    # geometry is now inside the finite finger window.  Refresh
                    # the measurement before applying the fail-closed contact
                    # gate.  If the adapter cannot provide a second sample,
                    # retain the pre-motion fact and keep the conservative
                    # behavior.
                    try:
                        refreshed = dict(
                            adapter.gripper_object_alignment(object_name, side=side)
                        )
                        refreshed_midpoint = np.asarray(
                            refreshed.get("finger_midpoint", ()), dtype=float
                        )
                        refreshed_object = np.asarray(
                            refreshed.get("object_position", ()), dtype=float
                        )
                        if (
                            refreshed_midpoint.shape == (3,)
                            and refreshed_object.shape == (3,)
                        ):
                            last_alignment = refreshed
                            # Use the same physical grasp-center anchor as the
                            # motion controller.  ``closest_point`` remains
                            # part of the independent finite-segment gate,
                            # but using it for the contact correction mixes an
                            # endpoint error into the center error.
                            refreshed_delta = refreshed_object - refreshed_midpoint
                            horizontal_error = float(
                                np.linalg.norm(refreshed_delta[:2])
                            )
                            vertical_offset = float(refreshed_delta[2])
                            between = _alignment_window_ready(
                                refreshed, float(surface_tolerance_m)
                            )
                            raw_between = bool(
                                refreshed.get("between_fingers", False)
                            )
                            violation = object_motion_violation(refreshed_object)
                            if violation is not None:
                                return _object_motion_failure(
                                    self.name,
                                    iteration=iteration,
                                    alignment=refreshed,
                                    violation=violation,
                                )
                    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                        pass
                    both_contact = (
                        len(forces) >= 2
                        and forces[0] > float(contact_threshold)
                        and forces[1] > float(contact_threshold)
                    )
                    # A geometric jaw window plus one-sided graze is not a
                    # pinch. The first tabletop failure closed on that state
                    # (one finger 14 N, the other 0 N). Stop pushing only when
                    # both fingers are loaded; if the object is still high in
                    # the window, keep the bounded descent.
                    if both_contact:
                        return SkillResult(
                            True,
                            self.name,
                            metrics=_alignment_metrics(
                                horizontal_error=horizontal_error,
                                vertical_offset=float(vertical_offset),
                                vertical_tolerance=float(position_tolerance),
                                surface_distance=last_alignment.get("surface_distance_m"),
                                between=between,
                                iterations=iteration,
                                failure_code=None,
                            ),
                            details={
                                "reason": "contact reached during alignment",
                                "contact_detected": True,
                                "contact_force_n": [round(float(v), 3) for v in forces],
                                "between_fingers": between,
                                "raw_between_fingers": bool(last_alignment.get("between_fingers", False)),
                                "require_between_fingers": bool(require_between_fingers),
                                "surface_tolerance_m": float(surface_tolerance_m),
                                "noncontact_rebaseline_count": noncontact_rebaseline_count,
                            },
                        )
                    if between and abs(float(vertical_offset)) > float(position_tolerance):
                        continue
                    if require_between_fingers and not between:
                        if _live_alignment_interaction_window_ready(
                            adapter,
                            object_name,
                            side,
                        ):
                            # This is an allowed one-sided intermediate
                            # interaction, not a grasp result. Re-enter the
                            # measured loop and let the next bounded descent
                            # acquire the other finger while the object-motion
                            # guard continues to protect the target.
                            continue
                        # A single finger touching while the object is still
                        # outside the jaw window means the object is off to one
                        # side, not ready for a pinch.  Declaring success here
                        # lets gripper_grasp close on empty air (or push the
                        # object).  Report failure so the planner re-approaches
                        # instead of wasting an attempt on a bad pinch.
                        return SkillResult(
                            False,
                            self.name,
                            metrics=_alignment_metrics(
                                horizontal_error=horizontal_error,
                                vertical_offset=float(correction[2]),
                                vertical_tolerance=float(position_tolerance),
                                surface_distance=last_alignment.get("surface_distance_m"),
                                between=between,
                                iterations=iteration,
                                failure_code="contact_not_centered",
                            ),
                            details={
                                "reason": "contact reached but object not between fingers",
                                "contact_detected": True,
                                "contact_force_n": [round(float(v), 3) for v in forces],
                                "between_fingers": between,
                                "raw_between_fingers": bool(last_alignment.get("between_fingers", False)),
                                "require_between_fingers": bool(require_between_fingers),
                        "surface_tolerance_m": float(surface_tolerance_m),
                        "direction_span_reference_world": (
                            None
                            if direction_span_reference is None
                            else np.asarray(direction_span_reference, dtype=float).round(6).tolist()
                        ),
                    },
                )
                    if not both_contact and not between:
                        if _live_alignment_interaction_window_ready(
                            adapter,
                            object_name,
                            side,
                        ):
                            continue
                        # A single finger touching while the object is outside
                        # the jaw window means the object is off to one side.
                        # Even when the caller did not require between_fingers,
                        # closing the gripper now would pinch empty air on one
                        # side and push the object on the other, so this is not
                        # a grasp-ready state.
                        return SkillResult(
                            False,
                            self.name,
                            metrics=_alignment_metrics(
                                horizontal_error=horizontal_error,
                                vertical_offset=float(correction[2]),
                                vertical_tolerance=float(position_tolerance),
                                surface_distance=last_alignment.get("surface_distance_m"),
                                between=between,
                                iterations=iteration,
                                failure_code="contact_not_centered",
                            ),
                            details={
                                "reason": "contact reached but object not centered in jaw",
                                "contact_detected": True,
                                "contact_force_n": [round(float(v), 3) for v in forces],
                                "between_fingers": between,
                                "raw_between_fingers": bool(last_alignment.get("between_fingers", False)),
                                "require_between_fingers": bool(require_between_fingers),
                                "surface_tolerance_m": float(surface_tolerance_m),
                            },
                        )
                    return SkillResult(
                        False,
                        self.name,
                        metrics=_alignment_metrics(
                            horizontal_error=horizontal_error,
                            vertical_offset=float(vertical_offset),
                            vertical_tolerance=float(position_tolerance),
                            surface_distance=last_alignment.get("surface_distance_m"),
                            between=between,
                            iterations=iteration,
                            failure_code="one_finger_contact",
                        ),
                        details={
                            "reason": "contact reached on one finger; jaw is not ready to pinch",
                            "contact_detected": True,
                            "contact_force_n": [round(float(v), 3) for v in forces],
                            "between_fingers": between,
                            "raw_between_fingers": bool(last_alignment.get("between_fingers", False)),
                            "require_between_fingers": bool(require_between_fingers),
                            "surface_tolerance_m": float(surface_tolerance_m),
                        },
                    )
            except (AttributeError, TypeError, ValueError):
                pass

        try:
            last_alignment = dict(adapter.gripper_object_alignment(object_name, side=side))
            midpoint = np.asarray(last_alignment.get("finger_midpoint", ()), dtype=float)
            object_position = np.asarray(last_alignment.get("object_position", ()), dtype=float)
            # Final alignment metrics use the physical grasp center, matching
            # the controller.  The finite segment/closest-point geometry is
            # still evaluated by ``between`` and remains the fallback gate
            # when the caller does not explicitly require a two-sided window.
            delta = np.asarray(object_position, dtype=float) - midpoint
            final_closest = _optional_alignment_point(
                last_alignment.get("closest_point")
            )
            if final_closest is None:
                final_closest = midpoint
            window_delta = np.asarray(object_position, dtype=float) - final_closest
            error = float(np.linalg.norm(delta[:2]))
            window_error = float(np.linalg.norm(window_delta[:2]))
            between = _alignment_window_ready(last_alignment, float(surface_tolerance_m))
            raw_between = bool(last_alignment.get("between_fingers", False))
        except (KeyError, RuntimeError, ValueError) as exc:
            return SkillResult(False, self.name, details={"reason": str(exc), "iterations": float(max_iterations)})
        vertically_aligned = (
            _alignment_physical_vertical_window_ready(last_alignment)
            or abs(float(delta[2])) <= float(position_tolerance)
        )
        centered = between if require_between_fingers else (
            between or window_error <= float(position_tolerance)
        )
        if centered and (
            not require_vertical_alignment or vertically_aligned
        ) and not _one_sided_finger_contact(adapter, side, float(contact_threshold)):
            return SkillResult(
                True,
                self.name,
                metrics=_alignment_metrics(
                    horizontal_error=error,
                    vertical_offset=float(delta[2]),
                    vertical_tolerance=float(position_tolerance),
                    surface_distance=last_alignment.get("surface_distance_m"),
                    between=between,
                    iterations=max_iterations,
                    failure_code=None,
                ),
                details={
                    "finger_midpoint": midpoint.tolist(),
                    "object_position": object_position.tolist(),
                    "between_fingers": between,
                    "raw_between_fingers": raw_between,
                    "require_between_fingers": bool(require_between_fingers),
                    "surface_tolerance_m": float(surface_tolerance_m),
                    "direction_span_reference_world": (
                        None
                        if direction_span_reference is None
                        else np.asarray(direction_span_reference, dtype=float).round(6).tolist()
                    ),
                    "noncontact_rebaseline_count": noncontact_rebaseline_count,
                    **{
                        key: last_alignment[key]
                        for key in (
                            "segment_fraction",
                            "surface_distance_m",
                        )
                        if key in last_alignment
                    },
                },
            )
        if require_vertical_alignment and not vertically_aligned:
            failure_code = "vertical_alignment_not_reached"
        elif require_between_fingers and not between:
            failure_code = "object_window_not_reached"
        elif _one_sided_finger_contact(adapter, side, float(contact_threshold)):
            failure_code = "one_finger_contact"
        else:
            failure_code = "horizontal_alignment_not_reached"
        return SkillResult(
            False,
            self.name,
            metrics=_alignment_metrics(
                horizontal_error=error,
                vertical_offset=float(delta[2]),
                vertical_tolerance=float(position_tolerance),
                surface_distance=last_alignment.get("surface_distance_m"),
                between=between,
                iterations=max_iterations,
                failure_code=failure_code,
            ),
            details={
                "reason": "alignment tolerance not reached",
                "finger_midpoint": midpoint.tolist(),
                "object_position": object_position.tolist(),
                "between_fingers": between,
                "raw_between_fingers": raw_between,
                "require_between_fingers": bool(require_between_fingers),
                "surface_tolerance_m": float(surface_tolerance_m),
                "noncontact_rebaseline_count": noncontact_rebaseline_count,
                **{
                    key: last_alignment[key]
                    for key in (
                        "segment_fraction",
                        "surface_distance_m",
                    )
                    if key in last_alignment
                },
            },
        )


def _alignment_metrics(
    *,
    horizontal_error: float,
    vertical_offset: float,
    vertical_tolerance: float,
    surface_distance: object,
    between: bool,
    iterations: int,
    failure_code: str | None,
) -> dict[str, Any]:
    try:
        measured_surface_distance = float(surface_distance)
    except (TypeError, ValueError):
        measured_surface_distance = float("nan")
    return {
        "alignment_error_m": float(horizontal_error),
        "horizontal_error_m": float(horizontal_error),
        "vertical_error_m": abs(float(vertical_offset)),
        "vertical_tolerance_m": float(vertical_tolerance),
        "surface_distance_m": measured_surface_distance,
        "between_fingers": bool(between),
        "iterations": float(iterations),
        "failure_code": failure_code,
    }


def _alignment_object_position(adapter: Any, object_name: str) -> np.ndarray | None:
    """Read one live object position without assuming a simulator backend.

    This helper intentionally does not call ``gripper_object_alignment`` as a
    fallback: that measurement is the alignment loop's observation and some
    adapters advance a synthetic observation on each call.  The loop passes
    its just-read position explicitly at phase boundaries; per-physics-step
    guarding uses only a direct live pose API.
    """
    candidates: list[object] = []
    if hasattr(adapter, "object_position"):
        try:
            candidates.append(adapter.object_position(object_name))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
    if not candidates and hasattr(adapter, "object_state"):
        try:
            state = adapter.object_state(object_name)
            candidates.append(getattr(state, "position", None))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
    for value in candidates:
        try:
            position = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if position.shape == (3,) and np.all(np.isfinite(position)):
            return position
    return None


def _pregrasp_object_motion_tolerance(model: Any, override: float | None) -> float:
    """Derive a scale-aware no-push tolerance for an open-gripper approach.

    The threshold is intentionally based on the target geometry and authored
    contact/planning margins.  A single cylinder radius or a scene coordinate
    must not leak into the generic alignment skill.  The upper cap prevents a
    large object from being pushed a visibly meaningful distance before the
    guard reacts; the lower bound tolerates ordinary physics settling noise.

    This is a *pre-attachment* tolerance, not a grasp clearance.  It must be
    much smaller than the object's footprint: allowing half a footprint would
    turn a failed approach into a valid-looking new object pose and permit
    repeated retries to accumulate a push.
    """
    if override is not None:
        value = float(override)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("object_motion_tolerance_m must be a finite positive number")
        return value
    from r1pro_data_gen.skills.manipulation.support_aware_grasp import pregrasp_motion_tolerance

    return pregrasp_motion_tolerance(model)


def _object_motion_failure(
    skill_name: str,
    *,
    iteration: int,
    alignment: dict[str, object],
    violation: dict[str, Any],
    motion: dict[str, Any] | None = None,
) -> SkillResult:
    """Build a replannable result for a pre-attachment push."""
    midpoint = np.asarray(alignment.get("finger_midpoint", ()), dtype=float)
    object_position = np.asarray(alignment.get("object_position", ()), dtype=float)
    if midpoint.shape == (3,) and object_position.shape == (3,):
        reference = midpoint
        closest = _optional_alignment_point(alignment.get("closest_point"))
        if closest is not None:
            reference = closest
        delta = object_position - reference
        horizontal_error = float(np.linalg.norm(delta[:2]))
        vertical_offset = float(delta[2])
    else:
        horizontal_error = float("nan")
        vertical_offset = float("nan")
    metrics = _alignment_metrics(
        horizontal_error=horizontal_error,
        vertical_offset=vertical_offset,
        vertical_tolerance=float("nan"),
        surface_distance=alignment.get("surface_distance_m"),
        between=_alignment_window_ready(alignment, 0.012),
        iterations=iteration,
        failure_code="object_moved_before_grasp",
    )
    metrics.update(
        {
            key: value
            for key, value in violation.items()
            if key in {"object_motion_m", "object_motion_tolerance_m"}
        }
    )
    details: dict[str, Any] = {
        "reason": "target object moved before the gripper established attachment",
        "failure_code": "object_moved_before_grasp",
        "iteration": iteration,
        "alignment": {
            key: alignment[key]
            for key in (
                "finger_midpoint",
                "closest_point",
                "object_position",
                "between_fingers",
                "segment_fraction",
                "surface_distance_m",
            )
            if key in alignment
        },
        **violation,
    }
    if motion is not None:
        details["motion"] = motion
    return SkillResult(False, skill_name, metrics=metrics, details=details)


def _measured_gripper_correction(
    kin: Any,
    adapter: Any,
    side: str,
    correction_world: np.ndarray,
) -> tuple[list[float] | None, list[float] | None]:
    """Map a live world-frame finger-center correction into the model frame.

    The correction is a world-frame displacement; the base pose is used for
    the world->base rotation (online URDF/USD calibration drifts several cm at
    pre-grasp postures and made these short corrections point the wrong way).
    """
    if not hasattr(adapter, "read_observation"):
        return None, None
    try:
        obs = adapter.read_observation(0.0)
        _sync_kinematics_auxiliary_q(kin, obs)
        base_pose = obs.base_pose
        if base_pose is None or len(base_pose) < 3:
            return None, None
        yaw = float(base_pose[2])
        c, s = math.cos(yaw), math.sin(yaw)
        # World->base rotation of a displacement (the base translation cancels).
        base_vector = np.array(
            [
                c * float(correction_world[0]) + s * float(correction_world[1]),
                -s * float(correction_world[0]) + c * float(correction_world[1]),
                float(correction_world[2]),
            ]
        )
        joints = ARM_JOINTS_BY_SIDE[side]
        q_arm = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
        ee_position, ee_quat = kin.fk(q_arm)
        ee_quat = np.asarray(ee_quat, dtype=float)
        if hasattr(kin, "grasp_center_fk") and hasattr(kin, "ee_target_from_grasp_center"):
            # The measured alignment is defined by the physical finger
            # midpoint, not the wrist/EE-link origin.  Adding a correction to
            # ``fk(q)`` omits that robot-tool offset and can send a low grasp
            # sideways into the object.  Express the same short correction in
            # the model's grasp-center frame, then convert that center back to
            # the EE target with the current orientation.  The next IK step
            # receives this same quaternion, so position and orientation use a
            # consistent frame contract.
            model_center, _ = kin.grasp_center_fk(q_arm)
            target_center = np.asarray(model_center, dtype=float) + base_vector
            target_ee = kin.ee_target_from_grasp_center(target_center, ee_quat)
        else:
            # Lightweight backends predating the grasp-center contract retain
            # the original EE-position behavior.
            target_ee = np.asarray(ee_position, dtype=float) + base_vector
        return np.asarray(target_ee, dtype=float).tolist(), ee_quat.tolist()
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None, None


def _one_sided_finger_contact(
    adapter: Any,
    side: str,
    contact_threshold: float,
) -> bool:
    """True when contact exists on exactly one finger.

    A loaded graze is not a pinch. Zero-zero forces mean the jaw is still a
    non-contact pregrasp and closing is allowed. Missing sensors do not block.
    """
    if not hasattr(adapter, "finger_contact_forces"):
        return False
    try:
        try:
            values = adapter.finger_contact_forces(side=side)
        except TypeError:
            values = adapter.finger_contact_forces()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if values is None or len(values) < 2:
        return False
    try:
        left = float(values[0])
        right = float(values[1])
    except (TypeError, ValueError):
        return False
    threshold = float(contact_threshold)
    loaded = (left > threshold, right > threshold)
    return any(loaded) and not all(loaded)


def _alignment_window_ready(alignment: Mapping[str, Any], surface_tolerance_m: float) -> bool:
    """Evaluate the finite gripper window with an optional measured tolerance."""
    if bool(alignment.get("between_fingers", False)):
        return True
    try:
        fraction = float(alignment.get("segment_fraction"))
        surface_distance = float(alignment.get("surface_distance_m"))
    except (TypeError, ValueError):
        return False
    return 0.08 <= fraction <= 0.92 and surface_distance <= float(surface_tolerance_m)


def _alignment_physical_vertical_window_ready(
    alignment: Mapping[str, Any],
) -> bool:
    """Return whether measured finger boxes, not link origins, clear the band.

    The R1Pro finger-link origins are above the actual contact boxes.  A valid
    tabletop pinch can therefore have a midpoint several centimetres above
    the object's centre even though both physical boxes overlap the object's
    vertical band.  Use that measured box certificate as the vertical gate;
    adapters without it retain the conservative origin-based fallback.
    """
    if alignment.get("window_geometry_source") != "projected_finger_boxes":
        return False
    intervals = alignment.get("finger_vertical_intervals")
    if not isinstance(intervals, (tuple, list)) or len(intervals) < 2:
        return False
    try:
        overlaps = [float(item["overlap_m"]) for item in intervals]
        required_overlap = float(alignment.get("required_vertical_overlap_m", 0.0))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        np.isfinite(required_overlap)
        and required_overlap >= 0.0
        and bool(overlaps)
        and all(
            np.isfinite(value) and value >= required_overlap for value in overlaps
        )
    )


def _optional_alignment_point(value: object) -> np.ndarray | None:
    """Parse an optional finite 3-vector without weakening the measurement gate."""
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point


def _directional_contact_identity(
    adapter: Any,
    object_name: str,
    contact_threshold: float,
) -> tuple[bool, str | None]:
    """Match contact-limited motion to the requested object identity."""
    if not hasattr(adapter, "contact_events"):
        return False, None
    observed_object: str | None = None
    try:
        events = adapter.contact_events()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False, None
    for event in events:
        try:
            if float(event.force_n) <= float(contact_threshold):
                continue
            body_a = str(event.body_a)
            body_b = str(event.body_b)
        except (AttributeError, TypeError, ValueError):
            continue
        if object_name in {body_a, body_b}:
            return True, object_name
        observed_object = body_b
    return False, observed_object


class ArmMoveDirectional:
    """Advance the end-effector along a direction until contact or a distance."""

    name = "arm_move_directional"
    description = "Move the arm end-effector along a direction (base frame) until contact or a set distance."
    parameters: dict[str, ParamSpec] = {
        "direction": ParamSpec("array", "Direction vector (xyz, base frame)", required=True),
        "distance": ParamSpec("number", "Max distance to move (m)", default=0.1),
        "step": ParamSpec("number", "Increment per IK step (m)", default=0.01),
        "until_contact": ParamSpec("boolean", "Stop early when contact is detected", default=False),
        "contact_threshold": ParamSpec("number", "Contact force threshold (N)", default=1.0),
        "speed_scale": ParamSpec("number", "Fraction of arm velocity limits used by the continuous Cartesian move", default=0.10, minimum=0.02),
        "object_name": ParamSpec("string", "Optional object used to measure the live gripper midpoint during a precision approach", default=None),
        "support_surface_name": ParamSpec("string", "Optional support surface intentionally approached during a precision grasp descent", default=None),
        "side": ParamSpec("string", "Which arm ('left' or 'right')", default="left", enum=("left", "right")),
    }

    def __init__(self, kin: Any, vel_limits: np.ndarray, planner: Any = None, speed_scale: float = 0.2):
        self.kin = kin
        self.vel_limits = vel_limits
        self.planner = planner
        self.speed_scale = speed_scale

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        direction: list[float] = None,
        distance: float = 0.1,
        step: float = 0.01,
        until_contact: bool = False,
        contact_threshold: float = 1.0,
        speed_scale: float = 0.10,
        object_name: str | None = None,
        support_surface_name: str | None = None,
        side: str = "left",
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if direction is None:
            raise ValueError("arm_move_directional requires direction")
        if (
            isinstance(distance, bool)
            or not math.isfinite(float(distance))
            or float(distance) <= 0.0
        ):
            raise ValueError("arm_move_directional distance must be finite and positive")
        if (
            isinstance(step, bool)
            or not math.isfinite(float(step))
            or float(step) <= 0.0
        ):
            raise ValueError("arm_move_directional step must be finite and positive")
        if until_contact and (not isinstance(object_name, str) or not object_name.strip()):
            raise ValueError("contact-limited arm_move_directional requires object_name")
        side = require_side(side)
        kin = for_side(self.kin, side)
        obs = adapter.read_observation(0.0)
        q_cur = np.array([obs.joint_positions[j] for j in ARM_JOINTS_BY_SIDE[side]])
        pos0, quat0 = kin.fk(q_cur)
        moved = 0.0
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        stabilize_base(adapter)

        # The generic FK end-effector frame and the live midpoint between the
        # physical finger links are not identical for every calibrated robot.
        # For a finite object approach, use the measured midpoint as the
        # motion reference and the certified arm_move_to executor.  This keeps
        # the primitive task-agnostic while preventing a model-frame descent
        # from reporting motion that the actual fingers did not make.
        if object_name and self.planner is not None and not until_contact:
            measured = _measured_midpoint_direction_target(
                kin, adapter, side, object_name, d, float(distance)
            )
            if measured is not None:
                move_to = ArmMoveTo(self.kin, self.vel_limits, self.planner)
                excluded_objects = [str(object_name)]
                if isinstance(support_surface_name, str) and support_surface_name.strip():
                    excluded_objects.append(support_surface_name.strip())
                requested_distance = float(distance)
                moved = 0.0
                midpoint1 = None
                target_ee = measured[0]
                target_quat = measured[1]
                result = SkillResult(False, "arm_move_to", details={"reason": "no measured segment executed"})
                segment_records: list[dict[str, Any]] = []
                remaining = requested_distance
                # A posture-dependent URDF/USD offset can make one local
                # model segment produce only a fraction of the requested live
                # midpoint displacement. Allow a bounded second pass of local
                # segments, while still stopping immediately on a no-progress
                # segment and never creating an unbounded retry loop.
                max_segments = max(
                    1,
                    2 * int(math.ceil(requested_distance / _MEASURED_DIRECTIONAL_CHUNK_M)) + 2,
                )
                for segment_index in range(max_segments):
                    if remaining <= 1e-4:
                        break
                    segment_distance = min(remaining, _MEASURED_DIRECTIONAL_CHUNK_M)
                    segment = _measured_midpoint_direction_target(
                        kin, adapter, side, object_name, d, segment_distance
                    )
                    if segment is None:
                        break
                    target_ee, target_quat, midpoint0, direction_world = segment
                    result = move_to.execute(
                        adapter,
                        scene=scene,
                        target_pos=target_ee,
                        target_quat=target_quat,
                        target_frame="ee",
                        side=side,
                        planning_time=1.2,
                        ik_candidates=3,
                        trajectory_speed_scale=float(speed_scale),
                        local_radius_m=2.0,
                        exclude_objects=excluded_objects,
                        step_hook=step_hook,
                    )
                    segment_moved = 0.0
                    motion_target_frame = "ee"
                    try:
                        alignment1 = adapter.gripper_object_alignment(object_name, side=side)
                        midpoint1 = np.asarray(alignment1.get("finger_midpoint", ()), dtype=float)
                        if midpoint1.shape == (3,):
                            segment_moved = max(
                                0.0,
                                float(np.dot(midpoint1 - midpoint0, direction_world)),
                            )
                    except (KeyError, RuntimeError, ValueError):
                        midpoint1 = None
                    # A certified EE-origin path can report success while the
                    # physical gripper midpoint does not advance because the
                    # URDF tool offset is not configuration-invariant.  Retry
                    # that short segment once in the calibrated grasp-center
                    # frame before declaring no progress.
                    if segment_moved <= 0.001:
                        target_center_base = _measured_midpoint_grasp_center_target(
                            kin, adapter, side, midpoint0 + direction_world * segment_distance
                        )
                        if target_center_base is not None:
                            center_result = move_to.execute(
                                adapter,
                                scene=scene,
                                target_pos=target_center_base,
                                target_quat=target_quat,
                                target_frame="grasp_center",
                                side=side,
                                planning_time=1.2,
                                ik_candidates=3,
                                trajectory_speed_scale=float(speed_scale),
                                local_radius_m=2.0,
                                exclude_objects=excluded_objects,
                                step_hook=step_hook,
                            )
                            result = center_result
                            motion_target_frame = "grasp_center"
                            try:
                                alignment1 = adapter.gripper_object_alignment(object_name, side=side)
                                midpoint1 = np.asarray(alignment1.get("finger_midpoint", ()), dtype=float)
                                if midpoint1.shape == (3,):
                                    segment_moved = max(
                                        0.0,
                                        float(np.dot(midpoint1 - midpoint0, direction_world)),
                                    )
                            except (KeyError, RuntimeError, ValueError):
                                midpoint1 = None
                    # If both certified target-frame attempts report success
                    # without moving the physical midpoint, solve this short
                    # measured displacement directly from the live posture.
                    # MPlib can select a nearby model-frame solution whose EE
                    # residual is acceptable while the USD finger midpoint is
                    # stationary because the open-gripper offset is not
                    # invariant.  The direct branch keeps the current IK
                    # branch, verifies the dense joint path against the same
                    # filtered obstacle set, and only then executes it.
                    if segment_moved <= 0.001:
                        # The URDF/USD grasp-center discrepancy is posture
                        # dependent. A model displacement equal to the live
                        # requested displacement can therefore be absorbed by
                        # the changing finger-link offset. Try a bounded set
                        # of larger model-space commands, measuring after each
                        # collision-checked execution. The largest command is
                        # still only four local chunks, so this cannot turn a
                        # precision descent into an unbounded plunge.
                        for command_scale in (4.0, 3.0, 2.0, 1.0):
                            direct_segment = _measured_midpoint_direction_target(
                                kin,
                                adapter,
                                side,
                                object_name,
                                d,
                                segment_distance * command_scale,
                            )
                            if direct_segment is None:
                                continue
                            (
                                direct_target_ee,
                                direct_target_quat,
                                direct_midpoint0,
                                direct_direction_world,
                            ) = direct_segment
                            direct_result = _measured_midpoint_direct_move(
                                kin,
                                adapter,
                                scene,
                                side,
                                direct_target_ee,
                                direct_target_quat,
                                float(speed_scale),
                                excluded_objects,
                                step_hook,
                            )
                            result = direct_result
                            motion_target_frame = "direct_continuous_ik"
                            midpoint0 = direct_midpoint0
                            direction_world = direct_direction_world
                            try:
                                alignment1 = adapter.gripper_object_alignment(object_name, side=side)
                                midpoint1 = np.asarray(alignment1.get("finger_midpoint", ()), dtype=float)
                                if midpoint1.shape == (3,):
                                    segment_moved = max(
                                        0.0,
                                        float(np.dot(midpoint1 - midpoint0, direction_world)),
                                    )
                            except (KeyError, RuntimeError, ValueError):
                                midpoint1 = None
                            if segment_moved > 0.001:
                                break
                    moved += segment_moved
                    remaining = max(0.0, requested_distance - moved)
                    segment_records.append(
                        {
                            "index": segment_index,
                            "requested_m": segment_distance,
                            "moved_m": segment_moved,
                            "underlying_success": bool(result.success),
                            "underlying_reason": result.details.get("reason"),
                            "underlying_details": dict(result.details),
                            "motion_target_frame": motion_target_frame,
                        }
                    )
                    # A failed model-frame certification is recoverable when
                    # the physical midpoint actually advanced.  Stop only on
                    # a no-progress segment to avoid repeating an identical
                    # impossible request indefinitely.
                    if segment_moved <= 0.001:
                        break
                contact = False
                try:
                    forces = adapter.finger_contact_forces(side=side)
                    contact = bool(forces and max(forces) > contact_threshold)
                except (AttributeError, TypeError, RuntimeError, ValueError):
                    pass
                # This branch is deliberately defined in the live finger-
                # midpoint frame.  ArmMoveTo also reports the model EE-origin
                # error, but that origin is not coincident with the physical
                # midpoint on this robot.  A small EE-origin miss must not turn
                # an otherwise completed measured descent into a false skill
                # failure.  Keep the underlying collision/path result in the
                # diagnostics; accept only when the measured midpoint itself
                # reached the requested displacement within a generic geometric
                # residual.
                measured_completion_tolerance = max(
                    0.008, min(0.015, float(distance) * 0.20)
                )
                endpoint_error = max(0.0, requested_distance - moved)
                progress_threshold = min(0.002, requested_distance * 0.25)
                if moved < progress_threshold:
                    measured_success = False
                    failure_code = "no_progress"
                elif endpoint_error > measured_completion_tolerance:
                    measured_success = False
                    failure_code = "endpoint_not_reached"
                else:
                    measured_success = True
                    failure_code = None
                return SkillResult(
                    success=measured_success,
                    skill=self.name,
                    metrics={
                        "moved_m": moved,
                        "contact": float(contact),
                        "measured_midpoint_motion": 1.0,
                        "requested_distance_m": requested_distance,
                        "actual_displacement_m": moved,
                        "endpoint_error_m": endpoint_error,
                        "contact_detected": False,
                        "contact_object": None,
                        "failure_code": failure_code,
                        "measured_completion_tolerance_m": measured_completion_tolerance,
                        "underlying_success": float(bool(result.success)),
                        "measured_segment_count": float(len(segment_records)),
                    },
                    details={
                        "motion_reference": "measured_gripper_midpoint",
                        "object_name": str(object_name),
                        "support_surface_name": support_surface_name,
                        "excluded_objects": excluded_objects,
                        "requested_distance_m": requested_distance,
                        "target_ee": np.asarray(target_ee).round(5).tolist(),
                        "motion_target_frame": segment_records[-1].get("motion_target_frame") if segment_records else "ee",
                        "measured_target_error_m": max(0.0, requested_distance - moved),
                        "measured_completion_tolerance_m": measured_completion_tolerance,
                        "underlying_reason": result.details.get("reason"),
                        "underlying_metrics": dict(result.metrics),
                        "measured_segments": segment_records,
                    },
                )

        # Build one online Cartesian IK chain, always seeded from the previous
        # frame. Executing a fresh accelerate/hold/decelerate segment for every
        # centimetre was the source of the visible stop-start descent.
        geometric = [q_cur.copy()]
        ik_step = min(max(float(step), 0.002), 0.01)
        for target in direction_steps(np.asarray(pos0, dtype=float), d, distance, ik_step):
            if hasattr(kin, "_solve_seed"):
                sol = kin._solve_seed(
                    target, quat0, geometric[-1],
                    pos_tol=min(0.003, max(0.0015, ik_step * 0.50)), rot_tol=0.02,
                )
            else:
                sol = kin.ik(target, quat0, q_init=geometric[-1])
            if not sol.success:
                return SkillResult(
                    success=False, skill=self.name,
                    metrics={"moved_m": moved, "ik_error_m": sol.position_error},
                    details={
                        "reason": "directional IK failed",
                        "side": side,
                        "q_current": np.asarray(geometric[-1]).round(5).tolist(),
                        "target_pos": np.asarray(target).round(5).tolist(),
                        "rotation_error_rad": float(sol.rotation_error),
                    },
                )
            if float(np.max(np.abs(sol.q_arm - geometric[-1]))) > 0.20:
                return SkillResult(
                    success=False, skill=self.name,
                    metrics={"moved_m": moved},
                    details={"reason": "directional IK changed redundant branch"},
                )
            geometric.append(np.asarray(sol.q_arm, dtype=float).copy())

        from r1pro_data_gen.methods.manipulation.mplib_path import _minimum_jerk_trajectory

        trajectory, _, _ = _minimum_jerk_trajectory(
            np.asarray(geometric), speed_scale=float(speed_scale), side=side
        )
        joints = ARM_JOINTS_BY_SIDE[side]
        contact = False
        last_target = trajectory[0]
        for q_target in trajectory[1:]:
            last_target = q_target
            adapter.set_targets(
                position={joint: float(q_target[i]) for i, joint in enumerate(joints)},
                velocity={},
            )
            adapter.step()
            if step_hook is not None:
                step_hook()
            try:
                forces = adapter.finger_contact_forces(side=side)
            except TypeError:  # legacy/test adapters
                forces = adapter.finger_contact_forces()
            if until_contact and forces and max(forces) > contact_threshold:
                contact = True
                break
        adapter.set_targets(
            position={joint: float(last_target[i]) for i, joint in enumerate(joints)},
            velocity={},
        )
        for _ in range(10):
            adapter.step()
            if step_hook is not None:
                step_hook()
        final_obs = adapter.read_observation(0.0)
        q_actual = np.asarray([final_obs.joint_positions[j] for j in joints], dtype=float)
        final_pos, _ = kin.fk(q_actual)
        moved = max(0.0, float(np.dot(final_pos - np.asarray(pos0, dtype=float), d)))
        final_error = float(np.max(np.abs(q_actual - last_target)))
        target_contact, contact_object = _directional_contact_identity(
            adapter,
            str(object_name),
            float(contact_threshold),
        ) if until_contact else (False, None)
        endpoint_error = max(0.0, float(distance) - moved)
        progress_threshold = min(0.002, float(distance) * 0.25)
        if moved < progress_threshold:
            success = False
            failure_code = "no_progress"
        elif until_contact and not target_contact:
            success = False
            failure_code = "contact_not_established"
        elif not until_contact and endpoint_error > max(0.003, float(step)):
            success = False
            failure_code = "endpoint_not_reached"
        elif final_error >= _FINAL_ERROR_TOL:
            success = False
            failure_code = "joint_tracking_error"
        else:
            success = True
            failure_code = None
        return SkillResult(
            success=success,
            skill=self.name,
            metrics={
                "moved_m": moved,
                "contact": float(target_contact),
                "final_error_rad": final_error,
                "trajectory_points": float(len(trajectory)),
                "requested_distance_m": float(distance),
                "actual_displacement_m": moved,
                "endpoint_error_m": endpoint_error,
                "contact_detected": bool(target_contact),
                "contact_object": contact_object,
                "failure_code": failure_code,
            },
            details={
                "reason": failure_code,
                "until_contact": bool(until_contact),
                "object_name": object_name,
            },
        )


def _measured_midpoint_direction_target(
    kin: Any,
    adapter: Any,
    side: str,
    object_name: str,
    direction_base: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Build a finite EE goal from a live finger-midpoint displacement."""
    if not hasattr(adapter, "gripper_object_alignment") or not hasattr(adapter, "body_position"):
        return None
    if not hasattr(kin, "calibrated_base_transform"):
        return None
    try:
        alignment = adapter.gripper_object_alignment(object_name, side=side)
        midpoint = np.asarray(alignment.get("finger_midpoint", ()), dtype=float)
        if midpoint.shape != (3,):
            return None
        obs = adapter.read_observation(0.0)
        joints = ARM_JOINTS_BY_SIDE[side]
        q_arm = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
        frame_names = getattr(kin, "base_calibration_frames", ())
        measured = np.asarray([adapter.body_position(name) for name in frame_names], dtype=float)
        rotation, _, rms_error = kin.calibrated_base_transform(
            q_arm, measured, frame_names=frame_names
        )
        if not np.isfinite(rms_error) or rms_error > 0.02:
            return None
        direction_world = np.asarray(rotation, dtype=float) @ np.asarray(direction_base, dtype=float)
        ee_position, ee_quat = kin.fk(q_arm)
        target_midpoint = midpoint + direction_world * float(distance)
        target_ee = np.asarray(ee_position, dtype=float) + np.asarray(rotation, dtype=float).T @ (
            target_midpoint - midpoint
        )
        return target_ee, np.asarray(ee_quat, dtype=float), midpoint, direction_world
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _measured_midpoint_grasp_center_target(
    kin: Any,
    adapter: Any,
    side: str,
    target_midpoint_world: np.ndarray,
) -> list[float] | None:
    """Map a live midpoint target into the calibrated model grasp-center frame."""
    if not hasattr(adapter, "body_position") or not hasattr(kin, "calibrated_base_transform"):
        return None
    try:
        obs = adapter.read_observation(0.0)
        joints = ARM_JOINTS_BY_SIDE[side]
        q_arm = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
        frame_names = getattr(kin, "base_calibration_frames", ())
        measured = np.asarray([adapter.body_position(name) for name in frame_names], dtype=float)
        rotation, translation, rms_error = kin.calibrated_base_transform(
            q_arm, measured, frame_names=frame_names
        )
        if not np.isfinite(rms_error) or rms_error > 0.02:
            return None
        target = np.asarray(target_midpoint_world, dtype=float)
        if target.shape != (3,):
            return None
        return (np.asarray(rotation, dtype=float).T @ (target - np.asarray(translation, dtype=float))).tolist()
    except (KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _measured_midpoint_direct_move(
    kin: Any,
    adapter: Any,
    scene: Any,
    side: str,
    target_ee: np.ndarray,
    target_quat: np.ndarray,
    speed_scale: float,
    exclude_objects: list[str],
    step_hook: Callable[[], None] | None,
) -> SkillResult:
    """Execute one live-measured Cartesian segment with continuous IK.

    This is a generic local-motion fallback for measured approaches.  It is
    deliberately narrower than a task policy: the caller supplies only the
    live target pose and obstacle exclusions.  The target is solved from the
    current posture, the resulting joint trajectory is collision checked, and
    the same trajectory is then tracked directly.  Keeping this branch local
    avoids asking a global randomized planner to resolve a 1.5 cm correction
    whose authoritative reference is a simulator finger midpoint.
    """
    if scene is None:
        return SkillResult(False, "arm_move_to", details={"reason": "measured direct move requires a scene"})
    joints = ARM_JOINTS_BY_SIDE[side]
    obs = adapter.read_observation(0.0)
    q_start = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
    target_ee = np.asarray(target_ee, dtype=float)
    target_quat = np.asarray(target_quat, dtype=float)
    if target_ee.shape != (3,) or target_quat.shape != (4,) or not np.all(np.isfinite(target_ee)):
        return SkillResult(False, "arm_move_to", details={"reason": "invalid measured direct target"})
    target_quat = target_quat / max(float(np.linalg.norm(target_quat)), 1e-12)
    if hasattr(kin, "ik_candidates"):
        solutions = kin.ik_candidates(target_ee, target_quat, q_start, max_candidates=8)
    elif hasattr(kin, "ik"):
        solution = kin.ik(target_ee, target_quat, q_init=q_start)
        solutions = [solution] if solution.success and solution.q_arm is not None else []
    else:
        solutions = []
    selected = _select_continuous_ik_solution(kin, solutions, q_start)
    if selected is None:
        return SkillResult(False, "arm_move_to", details={"reason": "measured direct IK failed"})
    q_goal, continuity, margin = selected

    from r1pro_data_gen.methods.collision import CollisionChecker, check_path, obstacles_from_scene
    from r1pro_data_gen.methods.manipulation.mplib_path import _minimum_jerk_trajectory
    from r1pro_data_gen.robot.robot_config import R1PRO_LOCAL_ARM_MIN_TRAJECTORY_S

    trajectory, _, _ = _minimum_jerk_trajectory(
        np.asarray([q_start, q_goal]),
        speed_scale=float(speed_scale),
        side=side,
        min_duration_s=R1PRO_LOCAL_ARM_MIN_TRAJECTORY_S,
    )
    base_pose = obs.base_pose or (0.0, 0.0, 0.0)
    free, _, collision_link = check_path(
        CollisionChecker(
            kin,
            obstacles_from_scene(
                scene,
                exclude=tuple(exclude_objects or ()),
                include_ground=True,
            ),
        ),
        list(trajectory),
        base_xy=(float(base_pose[0]), float(base_pose[1])),
        base_yaw=float(base_pose[2]) if len(base_pose) > 2 else 0.0,
        dense=8,
    )
    if not free:
        return SkillResult(
            False,
            "arm_move_to",
            details={
                "reason": "measured direct collision check failed",
                "collision_link": collision_link,
            },
        )

    for q_target in trajectory[1:]:
        adapter.set_targets(
            position={joint: float(q_target[index]) for index, joint in enumerate(joints)},
            velocity={},
        )
        adapter.step()
        if step_hook is not None:
            step_hook()
    # Keep the final target active briefly so the measured midpoint is sampled
    # after the same controller settling used by the ordinary arm executor.
    final_actual = q_start.copy()
    stable_steps = 0
    for _ in range(60):
        adapter.step()
        if step_hook is not None:
            step_hook()
        settled = adapter.read_observation(0.0)
        final_actual = np.asarray([settled.joint_positions[name] for name in joints], dtype=float)
        if float(np.max(np.abs(final_actual - q_goal))) < _FINAL_ERROR_TOL:
            stable_steps += 1
            if stable_steps >= 5:
                break
        else:
            stable_steps = 0
    final_error = float(np.max(np.abs(final_actual - q_goal)))
    return SkillResult(
        bool(final_error < _FINAL_ERROR_TOL),
        "arm_move_to",
        metrics={
            "final_error_rad": final_error,
            "continuity_cost": float(continuity),
            "goal_margin_rad": float(margin),
            "trajectory_points": float(len(trajectory)),
        },
        details={
            "reason": "measured direct continuous IK reached" if final_error < _FINAL_ERROR_TOL else "measured direct tracking error",
            "planning_status": "MeasuredDirectIK",
            "collision_checked": True,
            "collision_link": collision_link,
            "final_target_q": np.asarray(q_goal).round(5).tolist(),
            "final_actual_q": np.asarray(final_actual).round(5).tolist(),
        },
    )


class ArmRotateEE:
    """Rotate the end-effector about an axis while holding its position."""

    name = "arm_rotate_ee"
    description = (
        "Rotate the arm end-effector about an axis while holding position. "
        "frame='end_effector' rotates about a gripper-local axis (valves); "
        "frame='world' about a world axis (tilting to pour)."
    )
    parameters: dict[str, ParamSpec] = {
        "axis": ParamSpec("array", "Rotation axis (xyz)", required=True),
        "angle": ParamSpec("number", "Rotation angle (rad)", required=True),
        "frame": ParamSpec("string", "'end_effector' or 'world'", default="end_effector"),
        "steps": ParamSpec("number", "Interpolation steps", default=20),
        "side": ParamSpec("string", "Which arm ('left' or 'right')", default="left", enum=("left", "right")),
    }

    def __init__(self, kin: Any, vel_limits: np.ndarray, speed_scale: float = 0.2):
        self.kin = kin
        self.vel_limits = vel_limits
        self.speed_scale = speed_scale

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        axis: list[float] = None,
        angle: float = 0.0,
        frame: str = "end_effector",
        steps: int = 20,
        side: str = "left",
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        if axis is None:
            raise ValueError("arm_rotate_ee requires axis")
        side = require_side(side)
        kin = for_side(self.kin, side)
        segment = ArmSegmentExecutor(kin, np.asarray(for_side(self.vel_limits, side), dtype=float), self.speed_scale, hold_steps=10)
        obs = adapter.read_observation(0.0)
        q_cur = np.array([obs.joint_positions[j] for j in ARM_JOINTS_BY_SIDE[side]])
        pos0, quat0 = kin.fk(q_cur)

        axis_w = np.asarray(axis, dtype=float)
        if frame == "end_effector":
            # Express the gripper-local axis in the world frame: R(q0) @ axis.
            axis_w = Rotation.from_quat([quat0[1], quat0[2], quat0[3], quat0[0]]).apply(axis_w)
        axis_w = axis_w / np.linalg.norm(axis_w)
        quat_target = rotate_quat_about_axis(quat0, axis_w, float(angle))

        # Solve the target (rotated) joint configuration once, then interpolate
        # in joint space from the current config. Iterating IK frame-by-frame
        # lets the 1 redundant DOF of the 7-DOF arm wander arbitrarily, so the
        # elbow/wrist "waves" while the end-effector pose is correct. Joint-space
        # interpolation keeps every joint smooth and continuous.
        sol = kin.ik(np.asarray(pos0, dtype=float), quat_target, q_init=q_cur)
        if not sol.success:
            return SkillResult(
                success=False, skill=self.name,
                metrics={"ik_error_m": sol.position_error},
                details={"reason": "rotate IK failed"},
            )
        q_goal = sol.q_arm
        final_err = segment.execute(adapter, side, q_cur, q_goal, step_hook)
        if final_err >= _FINAL_ERROR_TOL:
            return SkillResult(success=False, skill=self.name,
                               metrics={"final_error_rad": float(final_err)})
        return SkillResult(success=True, skill=self.name,
                           metrics={"angle_rad": float(angle), "steps": float(steps)})


__all__ = [
    "ArmAlignGripper",
    "ArmMoveDirectional",
    "ArmRotateEE",
    "ArmTrajectoryFollow",
    "direction_steps",
    "rotate_quat_about_axis",
]
