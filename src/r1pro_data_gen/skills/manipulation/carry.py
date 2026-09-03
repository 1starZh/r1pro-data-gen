"""Generic motion for carrying a measured grasp to a scene destination."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from r1pro_data_gen.domain import GraspContext, object_vertical_extent_m, object_xy_radius_m
from r1pro_data_gen.robot.kinematics import BASE_CALIBRATION_FRAMES_BY_SIDE

from .arm import ARM_JOINTS_BY_SIDE
from .arm_motion import ArmMoveTo, ArmMoveThrough
from ..core.base import ParamSpec, SkillResult, stabilize_base
from ..core.sides import for_side, resolve_side, require_side


DEFAULT_CARRY_YAW_OFFSETS = (0.0, 0.261799, -0.261799, 0.523599, -0.523599)
# Same-support tabletop place inside this XY radius can go extend→descend
# without the 12 cm radial retract that looks like pulling the arm back.
_SHORT_SAME_SUPPORT_RETRACT_SKIP_M = 0.25


def _world_point_to_base(
    world: np.ndarray | tuple[float, float, float],
    base_pose: tuple[float, ...] | list[float],
) -> list[float]:
    """Convert a world point into the live mobile-base frame."""
    dx = float(world[0]) - float(base_pose[0])
    dy = float(world[1]) - float(base_pose[1])
    yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return [cosine * dx + sine * dy, -sine * dx + cosine * dy, float(world[2])]


def _yaw_rotated_quaternion(quaternion: np.ndarray, yaw: float) -> np.ndarray:
    """Apply a base-frame yaw while preserving the current tool tilt."""
    half = float(yaw) / 2.0
    left = np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)
    right = np.asarray(quaternion, dtype=float)
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    result = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    return result / np.linalg.norm(result)


def live_grasp_context(adapter: Any, object_name: str, side: str) -> GraspContext:
    """Read a live grasp context from an adapter with a narrow fallback API."""
    if hasattr(adapter, "get_grasp_context"):
        context = adapter.get_grasp_context(object_name, side=side)
        if not isinstance(context, GraspContext):
            raise TypeError("adapter.get_grasp_context must return GraspContext")
        return context
    if not hasattr(adapter, "gripper_object_alignment"):
        raise RuntimeError("adapter does not expose live grasp geometry")
    alignment = adapter.gripper_object_alignment(object_name, side=side)
    object_position = tuple(float(v) for v in alignment["object_position"])
    grasp_center = tuple(float(v) for v in alignment["finger_midpoint"])
    attached = bool(
        adapter.is_object_attached(object_name)
        if hasattr(adapter, "is_object_attached")
        else alignment.get("between_fingers", False)
    )
    attachment_error = (
        float(adapter.grasp_attachment_error(object_name))
        if attached and hasattr(adapter, "grasp_attachment_error")
        else None
    )
    return GraspContext(
        object_name=object_name,
        side=side,
        attached=attached,
        object_position_world=object_position,
        grasp_center_world=grasp_center,
        object_to_grasp_center_world=tuple(
            grasp_center[index] - object_position[index] for index in range(3)
        ),
        attachment_error_m=attachment_error,
    )


def calibrated_model_transform(
    kin: Any,
    adapter: Any,
    side: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Register the live simulator links to the selected planning model."""
    if kin is None or not hasattr(adapter, "body_position") or not hasattr(kin, "calibrated_base_transform"):
        return None
    try:
        joints = ARM_JOINTS_BY_SIDE[side]
        obs = adapter.read_observation(0.0)
        q_arm = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
        frames = BASE_CALIBRATION_FRAMES_BY_SIDE[side]
        measured = np.asarray([adapter.body_position(name) for name in frames], dtype=float)
        rotation, translation, rms_error = kin.calibrated_base_transform(q_arm, measured, frames)
        if not np.isfinite(rms_error) or rms_error > 0.02:
            return None
        return np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _xy_in_footprint(
    position: np.ndarray,
    centre: Any,
    size: Any,
    radius: float,
) -> bool:
    """True when the live object centre sits inside an XY footprint."""
    if centre is None or size is None:
        return False
    size_xy = np.asarray(size, dtype=float)[:2]
    centre_xy = np.asarray(centre, dtype=float)[:2]
    if size_xy.shape != (2,) or centre_xy.shape != (2,):
        return False
    half = size_xy / 2.0
    delta = np.abs(np.asarray(position[:2], dtype=float) - centre_xy)
    return bool(np.all(delta <= half + float(radius) + 0.01))


def _xy_on_support(position: np.ndarray, support: Any, radius: float) -> bool:
    """True when the live object centre sits on the destination support footprint."""
    return _xy_in_footprint(
        position,
        getattr(support, "pos", None),
        getattr(support, "size", None),
        radius,
    )


def _target_region_pose(adapter: Any, scene: Any, target_region_name: str) -> np.ndarray:
    if hasattr(adapter, "object_position"):
        return np.asarray(adapter.object_position(target_region_name), dtype=float)
    return np.asarray(scene.object(target_region_name).pos, dtype=float)


def _infer_source_support_surface(
    scene: Any,
    object_model: Any,
    object_position: tuple[float, float, float],
    *,
    vertical_tolerance_m: float = 0.03,
) -> str | None:
    """Infer the static surface currently supporting a live object.

    Carry receives the destination support surface, which is not necessarily
    the surface from which the object was picked.  During the small vertical
    lift we may temporarily ignore only that *source* contact; all other
    obstacles remain in the planning scene.  The inference is geometric and
    scene-generic: a cuboid whose XY footprint contains the measured object
    centre and whose top is close to the object's bottom is a source support.
    """
    if scene is None or not hasattr(scene, "objects"):
        return None
    position = np.asarray(object_position, dtype=float)
    object_bottom = float(position[2]) - object_vertical_extent_m(object_model) / 2.0
    candidates: list[tuple[float, float, str]] = []
    radius = object_xy_radius_m(object_model)
    for candidate in scene.objects:
        if candidate.name == object_model.name or getattr(candidate, "size", None) is None:
            continue
        size = np.asarray(candidate.size, dtype=float)
        centre = np.asarray(candidate.pos, dtype=float)
        half_xy = size[:2] / 2.0
        # Keep a small footprint margin for measured object-centre noise, but
        # do not classify a nearby destination marker as the source surface.
        inside = np.all(np.abs(position[:2] - centre[:2]) <= half_xy + radius + 0.01)
        if not inside:
            continue
        top_gap = abs(float(candidate.top_z) - object_bottom)
        if top_gap > float(vertical_tolerance_m):
            continue
        footprint_area = float(size[0] * size[1])
        candidates.append((top_gap, footprint_area, str(candidate.name)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _infer_support_surface_below_target(
    scene: Any,
    target_model: Any,
    support_model: Any,
    *,
    vertical_tolerance_m: float = 0.03,
) -> str | None:
    """Find a physical support directly below a non-colliding target marker."""
    if scene is None or not hasattr(scene, "objects"):
        return None
    target_position = np.asarray(target_model.pos, dtype=float)
    target_size = getattr(support_model, "size", None)
    if target_size is None:
        return None
    target_bottom = float(support_model.pos[2]) - float(target_size[2]) / 2.0
    candidates: list[tuple[float, float, str]] = []
    for candidate in scene.objects:
        if candidate.name in {target_model.name, support_model.name}:
            continue
        size = getattr(candidate, "size", None)
        if size is None:
            continue
        size_arr = np.asarray(size, dtype=float)
        centre = np.asarray(candidate.pos, dtype=float)
        if not np.all(np.abs(target_position[:2] - centre[:2]) <= size_arr[:2] / 2.0 + 0.01):
            continue
        top_gap = abs(float(candidate.top_z) - target_bottom)
        if top_gap > float(vertical_tolerance_m):
            continue
        candidates.append((top_gap, float(size_arr[0] * size_arr[1]), str(candidate.name)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


class ArmCarryObjectTo:
    """Carry an attached object using its live grasp transform.

    The planner chooses the object, destination and semantic parameters. This
    skill owns only the generic state transition from an existing attachment
    to a verified destination; it has no task, object, or scene constants.
    """

    name = "arm_carry_object_to"
    tier = "semantic"
    exposed = True
    description = (
        "Carry an already attached object to a named destination region on a "
        "named support. Use for placement when live on_support matches the "
        "destination support; if the destination is a different support, "
        "navigate first. Requires the object to be attached. "
        "target_region_name is the destination object; support_surface_name is "
        "the physical support under that destination. Do not use if the object "
        "is not attached, and do not use this for a no-grasp push goal."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Name of the currently attached object", required=True),
        "target_region_name": ParamSpec("string", "Name of the destination region or marker", required=True),
        "support_surface_name": ParamSpec("string", "Name of the support surface below the destination", required=True),
        "side": ParamSpec(
            "string",
            "Arm side; auto selects the side holding the live attachment",
            default="auto",
            enum=("auto", "left", "right"),
        ),
        "clearance_m": ParamSpec("number", "Object-center clearance above the destination surface (m)", default=0.13, minimum=0.01, exposed=False),
        "retract_distance_m": ParamSpec("number", "Distance to retract toward the robot before traversing (m)", default=0.12, minimum=0.0, exposed=False),
        "inward_offset_m": ParamSpec("number", "Destination offset toward the robot within the region (m)", default=0.05, minimum=0.0, exposed=False),
        "yaw_offsets_rad": ParamSpec("array", "Alternative vertical-tool yaw offsets (rad)", default=list(DEFAULT_CARRY_YAW_OFFSETS), min_items=1, max_items=8, exposed=False),
        "planning_time": ParamSpec("number", "Planning time per candidate edge (s)", default=0.4, minimum=0.1, exposed=False),
        "ik_candidates_per_waypoint": ParamSpec("integer", "IK branches retained per waypoint", default=3, minimum=1, exposed=False),
        "beam_width": ParamSpec("integer", "Verified waypoint prefixes retained", default=3, minimum=1, exposed=False),
        "max_planned_edges": ParamSpec("integer", "Planning edge budget", default=72, minimum=1, exposed=False),
        "trajectory_speed_scale": ParamSpec("number", "Carry trajectory speed scale", default=0.36, minimum=0.02, exposed=False),
        "descend_speed_scale": ParamSpec("number", "Final contact descent speed scale", default=0.18, minimum=0.02, exposed=False),
        "local_radius_m": ParamSpec("number", "Obstacle culling radius around the live base (m)", default=2.0, minimum=0.5, exposed=False),
        "place_xy_tolerance_m": ParamSpec("number", "Allowed final object XY error (m)", default=0.015, minimum=1e-4, exposed=False),
        "place_z_tolerance_m": ParamSpec("number", "Allowed final object Z error (m); held objects sit above rest height until release", default=0.05, minimum=1e-4, exposed=False),
    }

    def __init__(
        self,
        kin: Any,
        vel_limits: Any,
        planner: Any,
        move_through: ArmMoveThrough,
        move_to: ArmMoveTo,
    ) -> None:
        self.kin = kin
        self.vel_limits = vel_limits
        self.planner = planner
        self.move_through = move_through
        self.move_to = move_to

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        target_region_name: str | None = None,
        support_surface_name: str | None = None,
        side: str = "auto",
        clearance_m: float = 0.13,
        retract_distance_m: float = 0.12,
        inward_offset_m: float = 0.05,
        yaw_offsets_rad: list[float] | None = None,
        planning_time: float = 0.4,
        ik_candidates_per_waypoint: int = 3,
        beam_width: int = 3,
        max_planned_edges: int = 72,
        trajectory_speed_scale: float = 0.36,
        descend_speed_scale: float = 0.18,
        local_radius_m: float = 2.0,
        place_xy_tolerance_m: float = 0.015,
        place_z_tolerance_m: float = 0.05,
        skip_lift: bool = False,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if scene is None:
            return SkillResult(False, self.name, details={"reason": "carry requires a scene"})
        if not object_name or not target_region_name or not support_surface_name:
            raise ValueError("arm_carry_object_to requires object, target region, and support surface names")
        side = resolve_side(
            require_side(side, allow_auto=True),
            adapter,
            object_name=object_name,
        )
        stabilize_base(adapter)
        kin = for_side(self.kin, side)
        if kin is None:
            return SkillResult(False, self.name, details={"reason": "kinematics backend is unavailable"})
        try:
            context = live_grasp_context(adapter, object_name, side)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return SkillResult(False, self.name, details={"reason": "live grasp context unavailable", "error": str(exc)})
        if not context.attached:
            return SkillResult(False, self.name, details={"reason": "object is not attached", "grasp_context": context.to_dict()})

        try:
            target_region = scene.object(target_region_name)
            support_surface = scene.object(support_surface_name)
            object_model = scene.object(object_name)
        except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
            return SkillResult(False, self.name, details={"reason": "carry scene role is unavailable", "error": str(exc)})
        region_world = _target_region_pose(adapter, scene, target_region_name)
        target_world = np.asarray(region_world, dtype=float).copy()
        target_half_extent = float(min(target_region.size[:2])) / 2.0 if target_region.size else 0.0
        object_radius = object_xy_radius_m(object_model)
        maximum_inward = max(0.0, target_half_extent - object_radius)
        obs = adapter.read_observation(0.0)
        base_pose = obs.base_pose or (0.0, 0.0, 0.0)
        place_direction = np.asarray(base_pose[:2], dtype=float) - target_world[:2]
        place_norm = float(np.linalg.norm(place_direction))
        if place_norm > 1e-9:
            target_world = target_world.copy()
            target_world[:2] += place_direction / place_norm * min(inward_offset_m, maximum_inward)
        target_center_z = float(support_surface.top_z) + object_vertical_extent_m(object_model) / 2.0

        source_support_name = _infer_source_support_surface(
            scene,
            object_model,
            context.object_position_world,
        )
        destination_backing_support_name = _infer_support_surface_below_target(
            scene,
            target_region,
            support_surface,
        )

        calibrated = calibrated_model_transform(kin, adapter, side)
        if calibrated is None:
            return SkillResult(False, self.name, details={"reason": "live model calibration unavailable"})
        rotation, translation = calibrated
        joints = ARM_JOINTS_BY_SIDE[side]
        try:
            q_arm = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
        except KeyError as exc:
            return SkillResult(False, self.name, details={"reason": "arm observation is incomplete", "error": str(exc)})
        _, current_quat = kin.fk(q_arm)
        carry_height = target_center_z + float(clearance_m)

        # Ensure the object is above the destination before lateral traversal.
        # This is a generic relative move and is derived from the live grasp,
        # never from the object's reset-pose YAML position.
        if not skip_lift and context.object_position_world[2] < carry_height - 0.005:
            ee_position, ee_quat = kin.fk(q_arm)
            ee_delta_model = rotation.T @ np.asarray([0.0, 0.0, carry_height - context.object_position_world[2]])
            raise_exclusions = [object_name]
            if source_support_name is not None:
                raise_exclusions.append(source_support_name)
            raise_result = self.move_to.execute(
                adapter,
                scene=scene,
                target_pos=(np.asarray(ee_position) + ee_delta_model).tolist(),
                target_quat=np.asarray(ee_quat).tolist(),
                target_frame="ee",
                side=side,
                planning_time=0.4,
                trajectory_speed_scale=float(trajectory_speed_scale),
                local_radius_m=local_radius_m,
                exclude_objects=raise_exclusions,
                step_hook=step_hook,
            )
            if not raise_result.success:
                return SkillResult(
                    False,
                    self.name,
                    details={
                        "reason": "unable to raise object to carry clearance",
                        "motion": raise_result.details,
                        "source_support_surface": source_support_name,
                        "raise_exclusions": raise_exclusions,
                    },
                )
            context = live_grasp_context(adapter, object_name, side)
            # The raise is an actual motion phase. Refresh every stateful input
            # used by the carry planner so the first waypoint starts from the
            # post-raise robot/base/calibration state.
            obs = adapter.read_observation(0.0)
            base_pose = obs.base_pose or (0.0, 0.0, 0.0)
            calibrated = calibrated_model_transform(kin, adapter, side)
            if calibrated is None:
                return SkillResult(False, self.name, details={"reason": "live model calibration unavailable after raise"})
            rotation, translation = calibrated
            try:
                q_arm = np.asarray([obs.joint_positions[name] for name in joints], dtype=float)
            except KeyError as exc:
                return SkillResult(False, self.name, details={"reason": "arm observation is incomplete after raise", "error": str(exc)})
            _, current_quat = kin.fk(q_arm)

        object_position = np.asarray(context.object_position_world, dtype=float)
        object_to_center = np.asarray(context.object_to_grasp_center_world, dtype=float)
        radial = object_position[:2] - np.asarray(base_pose[:2], dtype=float)
        radial_norm = float(np.linalg.norm(radial))
        direction = radial / radial_norm if radial_norm > 1e-9 else np.asarray([1.0, 0.0])
        retract_world = object_position.copy()
        retract_world[:2] -= direction * min(float(retract_distance_m), max(0.0, radial_norm - 0.10))
        traverse_world = retract_world.copy()
        traverse_world[:2] += target_world[:2] - object_position[:2]
        above_world = object_position.copy()
        above_world[:2] = target_world[:2]
        place_world = above_world.copy()
        place_world[2] = target_center_z
        yaw_offsets = tuple(float(value) for value in (yaw_offsets_rad or DEFAULT_CARRY_YAW_OFFSETS))
        waypoint_diagnostics: dict[str, dict[str, Any]] = {}

        def poses_for_object_target(name: str, object_target: np.ndarray) -> list[dict[str, list[float]]]:
            midpoint_target_world = np.asarray(object_target, dtype=float) + object_to_center
            midpoint_target_model = rotation.T @ (midpoint_target_world - translation)
            poses = []
            for yaw in yaw_offsets:
                orientation = _yaw_rotated_quaternion(np.asarray(current_quat), yaw)
                position = kin.ee_target_from_grasp_center(midpoint_target_model, orientation)
                poses.append({"position": np.asarray(position, dtype=float).tolist(), "orientation": orientation.tolist()})
            waypoint_diagnostics[name] = {
                "object_target_world": np.asarray(object_target, dtype=float).tolist(),
                "grasp_center_target_world": midpoint_target_world.tolist(),
                "grasp_center_target_model": midpoint_target_model.tolist(),
                "poses": poses,
            }
            return poses

        descend_exclusions = [object_name, support_surface_name]
        if destination_backing_support_name is not None:
            descend_exclusions.append(destination_backing_support_name)
        if source_support_name is not None and source_support_name not in descend_exclusions:
            descend_exclusions.append(source_support_name)
        # Keep transit collision-free and solve the final drop separately.
        # Descend is one certified Cartesian interpolant from the live extend
        # pose; a measured local crawl would segment the drop and can tilt the
        # gripper before release.
        poses_for_object_target("place_descend", place_world)
        extend_waypoint = {
            "name": "carry_extend",
            "poses": poses_for_object_target("carry_extend", above_world),
            "exclude_objects": [object_name],
            "speed_scale": trajectory_speed_scale,
        }
        full_waypoints = [
            {"name": "carry_retract", "poses": poses_for_object_target("carry_retract", retract_world), "exclude_objects": [object_name], "speed_scale": trajectory_speed_scale},
            {"name": "carry_traverse", "poses": poses_for_object_target("carry_traverse", traverse_world), "exclude_objects": [object_name], "speed_scale": trajectory_speed_scale},
            extend_waypoint,
        ]
        xy_place = float(np.hypot(target_world[0] - object_position[0], target_world[1] - object_position[1]))
        skip_retract = (
            xy_place < _SHORT_SAME_SUPPORT_RETRACT_SKIP_M
            and _xy_on_support(
                object_position,
                support_surface,
                object_xy_radius_m(object_model),
            )
        )
        transit_waypoints = [extend_waypoint] if skip_retract else full_waypoints
        motion = self.move_through.execute(
            adapter,
            scene=scene,
            waypoints=transit_waypoints,
            side=side,
            planning_time=planning_time,
            ik_candidates_per_waypoint=ik_candidates_per_waypoint,
            beam_width=beam_width,
            max_planned_edges=max_planned_edges,
            trajectory_speed_scale=trajectory_speed_scale,
            local_radius_m=local_radius_m,
            carried_context=context,
            step_hook=step_hook,
        )
        if not motion.success and skip_retract:
            transit_waypoints = full_waypoints
            motion = self.move_through.execute(
                adapter,
                scene=scene,
                waypoints=transit_waypoints,
                side=side,
                planning_time=planning_time,
                ik_candidates_per_waypoint=ik_candidates_per_waypoint,
                beam_width=beam_width,
                max_planned_edges=max_planned_edges,
                trajectory_speed_scale=trajectory_speed_scale,
                local_radius_m=local_radius_m,
                carried_context=context,
                step_hook=step_hook,
            )
        if not motion.success:
            return SkillResult(
                False,
                self.name,
                metrics=dict(motion.metrics),
                details={
                    "reason": "carry motion failed",
                    "motion": motion.details,
                    "grasp_context": context.to_dict(),
                    "target_position_world": target_world.tolist(),
                    "target_center_z": target_center_z,
                    "carry_height": carry_height,
                    "pre_lift_handled_externally": bool(skip_lift),
                    "source_support_surface": source_support_name,
                    "destination_backing_support": destination_backing_support_name,
                    "base_pose": list(base_pose),
                    "current_ee_quaternion": np.asarray(current_quat, dtype=float).tolist(),
                    "calibration_rotation": rotation.tolist(),
                    "calibration_translation": translation.tolist(),
                    "waypoint_diagnostics": waypoint_diagnostics,
                },
            )

        try:
            context = live_grasp_context(adapter, object_name, side)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return SkillResult(
                False,
                self.name,
                details={"reason": "live grasp context unavailable after transit", "error": str(exc)},
            )
        if not context.attached:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "attachment lost after carry transit",
                    "failure_code": "attachment_lost",
                    "grasp_context": context.to_dict(),
                    "motion": motion.details,
                },
            )
        obs = adapter.read_observation(0.0)
        base_pose = obs.base_pose or (0.0, 0.0, 0.0)
        # Set-down is a vertical-only motion from the live EE: transit already
        # placed the object above the region. Re-solving a grasp-center IK at
        # a new XY/orientation is what turned this phase into a drop.
        object_z = float(context.object_position_world[2])
        dz = float(target_center_z) - object_z
        live_quat = None
        target_ee = None
        if kin is not None and hasattr(kin, "fk"):
            try:
                q_arm = np.asarray(
                    [obs.joint_positions[name] for name in joints], dtype=float
                )
                pos_ee, live_quat = kin.fk(q_arm)
                target_ee = np.asarray(pos_ee, dtype=float).copy()
                target_ee[2] += dz
            except (KeyError, TypeError, ValueError):
                live_quat = None
                target_ee = None
        if target_ee is None:
            grasp_center_world = np.asarray(place_world, dtype=float) + np.asarray(
                context.object_to_grasp_center_world, dtype=float
            )
            target_ee = np.asarray(
                _world_point_to_base(grasp_center_world, base_pose), dtype=float
            )
        def _vertical_descend(remaining_dz: float) -> SkillResult:
            obs_now = adapter.read_observation(0.0)
            quat_now = live_quat
            ee_now = np.asarray(target_ee, dtype=float)
            if kin is not None and hasattr(kin, "fk"):
                try:
                    q_now = np.asarray(
                        [obs_now.joint_positions[name] for name in joints], dtype=float
                    )
                    pos_now, quat_now = kin.fk(q_now)
                    ee_now = np.asarray(pos_now, dtype=float).copy()
                    ee_now[2] += float(remaining_dz)
                except (KeyError, TypeError, ValueError):
                    ee_now = np.asarray(target_ee, dtype=float)
            return self.move_to.execute(
                adapter,
                scene=scene,
                target_pos=[float(value) for value in ee_now],
                target_quat=(
                    None
                    if quat_now is None
                    else [float(value) for value in np.asarray(quat_now, dtype=float)]
                ),
                target_frame="ee",
                side=side,
                exclude_objects=descend_exclusions,
                prefer_local_certified_path=False,
                planning_time=max(0.4, float(planning_time)),
                trajectory_speed_scale=float(descend_speed_scale),
                local_radius_m=local_radius_m,
                step_hook=step_hook,
            )

        descend = _vertical_descend(dz)
        for _ in range(2):
            if descend.success:
                break
            try:
                context = live_grasp_context(adapter, object_name, side)
            except (KeyError, RuntimeError, TypeError, ValueError):
                break
            if not context.attached:
                break
            remaining = float(target_center_z) - float(context.object_position_world[2])
            if abs(remaining) < 0.01:
                break
            descend = _vertical_descend(remaining)

        placed = np.asarray(adapter.object_position(object_name), dtype=float)
        xy_error = float(np.linalg.norm(placed[:2] - target_world[:2]))
        z_error = abs(float(placed[2] - target_center_z))
        in_region = _xy_in_footprint(
            placed,
            region_world,
            getattr(target_region, "size", None),
            object_xy_radius_m(object_model),
        )
        object_at_place = z_error <= float(place_z_tolerance_m) and (
            xy_error <= float(place_xy_tolerance_m) or in_region
        )
        if not object_at_place and not descend.success:
            return SkillResult(
                False,
                self.name,
                metrics=dict(descend.metrics),
                details={
                    "reason": "place descend failed",
                    "failure_code": descend.details.get("failure_code") or "place_descend_failed",
                    "motion": motion.details,
                    "descend": descend.details,
                    "grasp_context": context.to_dict(),
                    "target_position_world": target_world.tolist(),
                    "target_center_z": target_center_z,
                    "descend_exclusions": descend_exclusions,
                    "waypoint_diagnostics": waypoint_diagnostics,
                },
            )
        success = object_at_place
        return SkillResult(
            success,
            self.name,
            metrics={"object_xy_error_m": xy_error, "object_z_error_m": z_error, "waypoints": 4.0},
            details={
                "reason": "carried object reached verified release pose" if success else "carried object missed verified release pose",
                "grasp_context": context.to_dict(),
                "target_position_world": target_world.tolist(),
                "target_center_z": target_center_z,
                "release_pose": placed.tolist(),
                "motion": motion.details,
                "descend": descend.details,
                "pre_lift_handled_externally": bool(skip_lift),
            },
        )


__all__ = ["ArmCarryObjectTo", "calibrated_model_transform", "live_grasp_context"]
