"""Support-aware grasp acquisition for low-clearance scene objects.

The ordinary grasp contract is intentionally semantic, but a movable object
near a ground or support plane cannot always be reached safely by descending
from above.  This module adds a reusable acquisition layer that derives a
non-contact, plane-parallel pregrasp from live object/support geometry and
robot gripper dimensions.  It then delegates the actual motion, measured
alignment, contact verification, and attachment to the existing generic grasp
skill.

No object name, scene coordinate, fixed trajectory, or task-specific pose is
stored here.  The layer is a capability of the R1Pro gripper and consumes only
the current scene snapshot and live robot/object state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from r1pro_data_gen.domain import object_vertical_extent_m, object_xy_half_extents_m
from r1pro_data_gen.robot.robot_config import (
    R1PRO_GRIPPER_COLLISION_ENVELOPE_M,
    R1PRO_GRIPPER_FINGER_HALF_HEIGHT_M,
    R1PRO_GRIPPER_FINGER_HALF_LENGTH_M,
    R1PRO_GROUND_INTERACTION_MAX_STAGED_EE_Z_M,
    R1PRO_GRIPPER_PREGRASP_CLEARANCE_M,
    R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M,
    R1PRO_GRIPPER_FINGER_SUPPORT_ENVELOPE_Z_M,
    R1PRO_SUPPORT_AWARE_APPROACH_OFFSETS_RAD,
    R1PRO_SUPPORT_AWARE_YAW_OFFSETS_RAD,
    R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE,
    R1PRO_TRANSFER_HOLD_CENTER_TOL_M,
)
from r1pro_data_gen.methods.collision import LINK_SPHERE_RADII_BY_SIDE

from ..core.base import SkillResult
from .grasp import GraspObject
from ..core.sides import require_side


class SupportAwareGraspObject(GraspObject):
    """Acquire a low-clearance object through a support-aware pregrasp.

    The public operation remains a complete grasp transaction.  For a low
    object, the implementation first creates a safe high posture (through the
    inherited whole-body transition), then moves to a geometry-derived
    plane-parallel standoff.  Final alignment approaches the object laterally
    at its grasp height, preventing a top-down finger collision from turning
    into a push.  Objects with a normal elevated clearance retain the parent
    behavior, so this skill is suitable for mixed scenes.
    """

    name = "support_aware_grasp_object"
    tier = "semantic"
    exposed = False
    description = (
        "Safely acquire a movable object near a ground or support plane: derive "
        "a geometry-scaled non-contact side/plane standoff from live scene facts, "
        "align the gripper without pushing, verify attachment, and return a "
        "replannable failure if the object moves before grasping."
    )
    parameters = dict(GraspObject.parameters)

    def _whole_body_pregrasp_parameters(
        self,
        adapter: Any,
        *,
        scene: Any,
        object_name: str,
        object_model: Any,
        object_world: Sequence[float],
        support_name: str | None,
        low_object: bool,
        side: str,
    ) -> dict[str, Any]:
        """Give the whole-body solver the live low-side interaction target.

        A generic high waypoint is useful for clearing obstacles, but it leaves
        the arm at its hanging home posture and makes the later descent to a
        floor object an impossible one-shot IK request.  The support-aware
        target is derived from the object/support/gripper geometry and remains
        a non-contact standoff.  The whole-body solver then chooses a
        collision- and stability-certified arm/torso configuration that can
        actually reach this low workspace.
        """
        if not low_object:
            return {}
        candidates = derive_support_aware_pregrasp_candidates(
            adapter,
            scene,
            object_name,
            object_model,
            object_world,
            support_name=support_name,
            side=side,
        )
        if not candidates:
            return {}
        target_world, geometry = candidates[0]
        target_span_world = _jaw_span_from_approach_geometry(geometry)
        return {
            "target_center_world": [float(value) for value in target_world],
            "target_span_world": [float(value) for value in target_span_world],
        }

    def _whole_body_pregrasp_parameter_candidates(
        self,
        adapter: Any,
        *,
        scene: Any,
        object_name: str,
        object_model: Any,
        object_world: Sequence[float],
        support_name: str | None,
        low_object: bool,
        side: str,
    ) -> tuple[dict[str, Any], ...]:
        """Expose all bounded live-geometry directions to the whole-body solver.

        The first direction is the current base-to-object radial approach, but
        it is not privileged after planning proves its path unstable.  The
        remaining tangent/opposite directions are derived from the same live
        object, support, and gripper envelope.  They are alternatives within
        one grasp transaction, never task coordinates supplied by the planner.
        """
        if not low_object:
            return ({},)
        candidates = derive_support_aware_pregrasp_candidates(
            adapter,
            scene,
            object_name,
            object_model,
            object_world,
            support_name=support_name,
            side=side,
        )
        if not candidates:
            # Preserve the parent contract: the whole-body skill can still
            # report a generic high-posture reachability failure when live
            # support geometry is unavailable.
            return ({},)
        return tuple(
            {
                "target_center_world": [float(value) for value in target_world],
                "target_span_world": [
                    float(value)
                    for value in _jaw_span_from_approach_geometry(candidate_geometry)
                ],
            }
            for target_world, candidate_geometry in candidates
        )

    def _prepare_alignment_standoff(
        self,
        adapter: Any,
        *,
        scene: Any,
        object_name: str,
        object_model: Any,
        object_world: Sequence[float],
        support_name: str | None,
        low_object: bool,
        side: str,
        step_hook: Any = None,
        pregrasp_established: bool = False,
    ) -> SkillResult | None:
        if not low_object:
            return None
        side = require_side(side)
        pregrasp_candidates = derive_support_aware_pregrasp_candidates(
            adapter,
            scene,
            object_name,
            object_model,
            object_world,
            support_name=support_name,
            side=side,
        )
        if not pregrasp_candidates:
            return SkillResult(
                False,
                self.name,
                details={
                    "failure_code": "support_geometry_unavailable",
                    "reason": "support-aware pregrasp could not derive a finite plane-parallel target",
                },
            )
        observation = adapter.read_observation(0.0)
        base_pose = getattr(observation, "base_pose", None) or (0.0, 0.0, 0.0)
        initial_geometry = pregrasp_candidates[0][1]
        # Keep the target object in the collision set.  Only a source support
        # may be excluded for the final local approach; this pregrasp must be
        # certified as non-contact rather than relying on a blind object
        # exclusion.
        exclusions = [support_name] if support_name else []
        # Do not connect the old high pose directly to the low side standoff:
        # that diagonal can cut through the target even when both endpoints
        # are collision-free.  First translate at the existing high height,
        # then descend outside the object's inflated footprint.  The final
        # planar closure remains owned by measured alignment.
        high_z = _live_grasp_center_height(adapter, side)
        if high_z is None:
            high_z = float(object_world[2]) + max(
                0.20,
                0.5 * float(initial_geometry["object_vertical_extent_m"])
                + R1PRO_GRIPPER_FINGER_HALF_HEIGHT_M
                + float(initial_geometry["planning_margin_m"])
                + float(initial_geometry["contact_offset_m"]),
            )
        approach_results: list[dict[str, Any]] = []

        if pregrasp_established:
            # The whole-body transition has already certified and physically
            # executed one of these live low-side targets.  Calling the
            # ordinary arm IK again here would discard that certified branch
            # and can select a different self-motion solution with a larger
            # static torque (the observed joint-5 limit hit).  Verify the
            # measured finger midpoint and reuse the current posture; the
            # later arm-align phase owns only the final contact closure.
            current_center = _live_grasp_center(adapter, side)
            if current_center is None:
                return SkillResult(
                    False,
                    self.name,
                    details={
                        "failure_code": "pregrasp_center_unavailable",
                        "reason": "certified whole-body pregrasp has no live finger-midpoint measurement",
                    },
                )
            distances = [
                float(np.linalg.norm(current_center - np.asarray(candidate_world, dtype=float)))
                for candidate_world, _ in pregrasp_candidates
            ]
            candidate_index = int(np.argmin(distances))
            center_error = distances[candidate_index]
            if center_error > 0.5 * float(R1PRO_TRANSFER_HOLD_CENTER_TOL_M):
                return SkillResult(
                    False,
                    self.name,
                    metrics={"pregrasp_center_error_m": center_error},
                    details={
                        "failure_code": "pregrasp_center_mismatch",
                        "reason": "measured finger midpoint is outside the certified support-aware pregrasp target",
                        "pregrasp_center_error_m": center_error,
                        "pregrasp_center_tolerance_m": 0.5 * float(R1PRO_TRANSFER_HOLD_CENTER_TOL_M),
                    },
                )
            target_world = np.asarray(pregrasp_candidates[candidate_index][0], dtype=float).copy()
            target_base = world_point_to_base(target_world, base_pose)
            geometry = dict(pregrasp_candidates[candidate_index][1])
            return SkillResult(
                True,
                self.name,
                metrics={"pregrasp_center_error_m": center_error},
                details={
                    "target_pos": [float(value) for value in target_base],
                    "target_world": [float(value) for value in target_world],
                    "high_target_pos": None,
                    "high_target_world": None,
                    "approach_mode": "reuse_certified_pregrasp",
                    "source_support_name": support_name,
                    "geometry": geometry,
                    "approach_candidate_index": candidate_index,
                    "approach_candidate_count": len(pregrasp_candidates),
                    "approach_candidates": [
                        {
                            "target_world": [float(value) for value in candidate_world],
                            "geometry": dict(candidate_geometry),
                        }
                        for candidate_world, candidate_geometry in pregrasp_candidates
                    ],
                    "object_name": object_name,
                    "approach_results": [],
                    "chosen_orientation_quat": None,
                    "orientation_candidate_count": len(support_aware_orientation_candidates(side)),
                    "pregrasp_reused": True,
                    "pregrasp_center_error_m": center_error,
                },
            )

        def record(
            phase: str,
            waypoint: Sequence[float],
            result: SkillResult,
            target_quat: Sequence[float] | None = None,
        ) -> None:
            item: dict[str, Any] = {
                "phase": phase,
                "target_pos": [float(value) for value in waypoint],
                "success": bool(result.success),
                "metrics": dict(result.metrics),
                "details": dict(result.details),
            }
            if target_quat is not None:
                item["target_quat"] = [float(value) for value in target_quat]
            approach_results.append(item)

        def motion_failure(result: SkillResult) -> SkillResult | None:
            current = _live_object_position(adapter, object_name)
            if current is None:
                return None
            displacement = float(
                np.linalg.norm(current - np.asarray(object_world, dtype=float))
            )
            tolerance = pregrasp_motion_tolerance(object_model)
            if displacement <= tolerance:
                return None
            return SkillResult(
                False,
                self.name,
                metrics={
                    **dict(result.metrics),
                    "object_motion_m": displacement,
                    "object_motion_tolerance_m": tolerance,
                },
                details={
                    **dict(result.details),
                    "failure_code": "object_moved_before_grasp",
                    "reason": "target object moved during support-aware pregrasp",
                    "object_motion_m": displacement,
                    "object_motion_tolerance_m": tolerance,
                    "initial_object_position": [float(value) for value in object_world],
                    "current_object_position": current.tolist(),
                },
            )

        # Establish a certified high lateral pose before trying any low
        # orientation.  Radial approach is the first candidate, but a close
        # target can be inside a robot's local shoulder singularity even when
        # it is collision-safe.  Tangent/opposite directions are derived from
        # the same live object geometry and are selected only after IK/path
        # certification; they are not scene-specific waypoints.
        moved: SkillResult | None = None
        target_world: np.ndarray | None = None
        target_base: list[float] | None = None
        geometry: dict[str, Any] = {}
        high_target_world: np.ndarray | None = None
        high_target_base: list[float] | None = None
        approach_candidate_index: int | None = None
        last_high_failure: SkillResult | None = None
        for candidate_index, (candidate_world, candidate_geometry) in enumerate(pregrasp_candidates):
            candidate_high_world = np.asarray(candidate_world, dtype=float).copy()
            candidate_high_world[2] = max(float(high_z), float(candidate_world[2]))
            candidate_high_base = world_point_to_base(candidate_high_world, base_pose)
            high_attempt = self._approach(
                adapter,
                scene=scene,
                target=candidate_high_base,
                side=side,
                exclude=exclusions,
                prefer_local_certified_path=True,
                step_hook=step_hook,
            )
            record(
                f"high_lateral_clearance_candidate_{candidate_index}",
                candidate_high_base,
                high_attempt,
            )
            failed_motion = motion_failure(high_attempt)
            if failed_motion is not None:
                moved = failed_motion
                last_high_failure = failed_motion
                break
            last_high_failure = high_attempt
            if high_attempt.success:
                moved = high_attempt
                target_world = np.asarray(candidate_world, dtype=float).copy()
                target_base = world_point_to_base(target_world, base_pose)
                geometry = dict(candidate_geometry)
                high_target_world = candidate_high_world
                high_target_base = candidate_high_base
                approach_candidate_index = candidate_index
                break

        if moved is None:
            moved = last_high_failure or SkillResult(
                False,
                self.name,
                details={
                    "failure_code": "support_aware_high_approach_failed",
                    "reason": "no high lateral support-aware approach candidate was certified",
                },
            )
        chosen_quat: list[float] | None = None
        last_low_attempt = moved
        if moved.success and target_base is not None and high_target_world is not None and high_target_base is not None:
            # A parallel gripper has a bounded in-plane orientation redundancy.
            # Test it at the low standoff rather than assuming the default
            # wrist yaw is compatible with every live obstacle arrangement.
            for index, candidate_quat in enumerate(support_aware_orientation_candidates(side)):
                low_attempt = self._approach(
                    adapter,
                    scene=scene,
                    target=target_base,
                    side=side,
                    exclude=exclusions,
                    target_quat=candidate_quat,
                    prefer_local_certified_path=True,
                    step_hook=step_hook,
                )
                record(
                    f"low_side_standoff_candidate_{index}",
                    target_base,
                    low_attempt,
                    candidate_quat,
                )
                failed_motion = motion_failure(low_attempt)
                if failed_motion is not None:
                    moved = failed_motion
                    last_low_attempt = failed_motion
                    break
                last_low_attempt = low_attempt
                if low_attempt.success:
                    moved = low_attempt
                    chosen_quat = candidate_quat
                    break

                # The local measured controller may have executed several
                # safe corrections before rejecting its next segment. Return
                # to the already certified high waypoint before trying a new
                # yaw; never let a failed candidate silently bias the next.
                current_center = _live_grasp_center(adapter, side)
                recovery_height = max(
                    float(high_target_world[2]),
                    float(current_center[2]) if current_center is not None else 0.0,
                )
                recovery_height = min(
                    recovery_height,
                    R1PRO_GROUND_INTERACTION_MAX_STAGED_EE_Z_M,
                )
                # Preserve the current planar position while escaping.  A
                # failed local correction may be outside the IK basin of the
                # original high target; moving only away from the support is
                # the smallest certified recovery and lets the next candidate
                # start from a measured, collision-free state.
                recovery_world = (
                    np.asarray(current_center, dtype=float).copy()
                    if current_center is not None
                    else np.asarray(high_target_world, dtype=float).copy()
                )
                recovery_world[2] = recovery_height
                recovery_base = world_point_to_base(recovery_world, base_pose)
                recovery_quat = _live_grasp_orientation_base(adapter, side, base_pose)
                escape = self._approach(
                    adapter,
                    scene=scene,
                    target=recovery_base,
                    side=side,
                    exclude=exclusions,
                    # Keep the candidate's wrist orientation during
                    # recovery.  First use a higher escape waypoint so a
                    # partially descended local IK branch does not have to
                    # connect directly back through the same singular region.
                    target_quat=recovery_quat,
                    prefer_local_certified_path=True,
                    step_hook=step_hook,
                )
                record("recover_high_escape", recovery_base, escape, recovery_quat)
                failed_motion = motion_failure(escape)
                if failed_motion is not None:
                    moved = failed_motion
                    last_low_attempt = failed_motion
                    break
                if not escape.success:
                    moved = SkillResult(
                        False,
                        self.name,
                        metrics=dict(escape.metrics),
                        details={
                            **dict(escape.details),
                            "failure_code": "support_aware_recovery_failed",
                            "reason": "failed to reach a collision-free high escape between orientation trials",
                        },
                    )
                    last_low_attempt = moved
                    break
            else:
                moved = last_low_attempt

        details = dict(moved.details)
        details.update(
            {
                "target_pos": None if target_base is None else [float(value) for value in target_base],
                "target_world": None if target_world is None else [float(value) for value in target_world],
                "high_target_pos": None if high_target_base is None else [float(value) for value in high_target_base],
                "high_target_world": None if high_target_world is None else [float(value) for value in high_target_world],
                "approach_mode": "plane_parallel_non_contact",
                "source_support_name": support_name,
                "geometry": geometry,
                "approach_candidate_index": approach_candidate_index,
                "approach_candidate_count": len(pregrasp_candidates),
                "approach_candidates": [
                    {
                        "target_world": [float(value) for value in candidate_world],
                        "geometry": dict(candidate_geometry),
                    }
                    for candidate_world, candidate_geometry in pregrasp_candidates
                ],
                "object_name": object_name,
                "approach_results": approach_results,
                "chosen_orientation_quat": chosen_quat,
                "orientation_candidate_count": len(support_aware_orientation_candidates(side)),
            }
        )
        current = _live_object_position(adapter, object_name)
        if current is not None:
            displacement = float(np.linalg.norm(current - np.asarray(object_world, dtype=float)))
            tolerance = pregrasp_motion_tolerance(object_model)
            details.update(
                {
                    "object_motion_m": displacement,
                    "object_motion_tolerance_m": tolerance,
                    "initial_object_position": [float(value) for value in object_world],
                    "current_object_position": current.tolist(),
                }
            )
            if displacement > tolerance:
                return SkillResult(
                    False,
                    self.name,
                    metrics={
                        **dict(moved.metrics),
                        "object_motion_m": displacement,
                        "object_motion_tolerance_m": tolerance,
                    },
                    details={
                        **details,
                        "failure_code": "object_moved_before_grasp",
                        "reason": "target object moved before the support-aware grasp established attachment",
                    },
                )
        return SkillResult(
            bool(moved.success),
            self.name,
            metrics=dict(moved.metrics),
            details=details,
        )


def derive_support_aware_pregrasp(
    adapter: Any,
    scene: Any,
    object_name: str,
    object_model: Any,
    object_world: Sequence[float],
    *,
    support_name: str | None,
    side: str = "left",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Derive a non-contact standoff from live geometry.

    The approach direction is from the current robot base toward the object;
    the target lies one geometry-scaled clearance back along that direction.
    Its height is a geometry-derived point in the object's upper grasp band,
    clamped above the active support plane by the gripper's collision
    half-height.  A finger body has non-zero vertical extent, so using the
    object centre verbatim can put its lower edge unnecessarily close to the
    support while still asking the later alignment phase to descend through a
    large vertical interval.
    """
    candidates = derive_support_aware_pregrasp_candidates(
        adapter,
        scene,
        object_name,
        object_model,
        object_world,
        support_name=support_name,
        side=side,
    )
    if candidates:
        return candidates[0]
    return None, {"reason": "live support/object geometry is unavailable"}


def _jaw_span_from_approach_geometry(geometry: dict[str, Any]) -> np.ndarray:
    """Return a horizontal jaw-span direction from a live approach ray.

    The support-aware target is placed on the opposite side of the object from
    ``approach_direction_world``.  A parallel-jaw gripper must therefore span
    the direction perpendicular to that ray so the object enters the opening
    during the final radial closure.  This is a local geometric relation, not
    an authored wrist pose; any object shape that can provide the same
    support-aware approach facts uses the same rule.
    """
    approach = np.asarray(
        geometry.get("approach_direction_world", ()),
        dtype=float,
    )
    if approach.shape != (2,) or not np.all(np.isfinite(approach)):
        raise ValueError("approach_direction_world must be a finite 2-vector")
    norm = float(np.linalg.norm(approach))
    if norm <= 1.0e-8:
        raise ValueError("approach_direction_world must be non-zero")
    approach = approach / norm
    return np.asarray([-approach[1], approach[0], 0.0], dtype=float)


def derive_support_aware_pregrasp_candidates(
    adapter: Any,
    scene: Any,
    object_name: str,
    object_model: Any,
    object_world: Sequence[float],
    *,
    support_name: str | None,
    side: str = "left",
) -> tuple[tuple[np.ndarray, dict[str, Any]], ...]:
    """Derive bounded plane-parallel targets around a live object.

    The radial direction is always first.  If that point falls inside the
    robot's local workspace singularity or an obstacle, the motion layer can
    try tangent/opposite directions without changing the object geometry or
    inventing task-specific coordinates.  Every returned target shares the
    same measured safety envelope and is still subject to live IK/path
    certification by the caller.
    """
    del object_name  # retained in the API for future live-object adapters
    side = require_side(side)
    try:
        world = np.asarray(object_world, dtype=float)
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            raise ValueError("object_world must be a finite 3-vector")
        observation = adapter.read_observation(0.0)
        base_pose = getattr(observation, "base_pose", None) or (0.0, 0.0, 0.0)
        if len(base_pose) < 3 or not all(math.isfinite(float(value)) for value in base_pose[:3]):
            raise ValueError("base pose is not finite")
        base_xy = np.asarray([float(base_pose[0]), float(base_pose[1])], dtype=float)
        radial = world[:2] - base_xy
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1.0e-8:
            radial_angle = float(base_pose[2])
        else:
            radial_angle = math.atan2(float(radial[1]), float(radial[0]))

        half_x, half_y = object_xy_half_extents_m(object_model)
        object_height = object_vertical_extent_m(object_model)
        physics = getattr(object_model, "physics", None)
        planning_margin = max(0.0, float(getattr(physics, "planning_margin", 0.0) or 0.0))
        contact_offset = max(0.0, float(getattr(physics, "contact_offset", 0.0) or 0.0))
        gripper_envelope, envelope_source, envelope_by_link = _gripper_collision_envelope(
            adapter,
            side,
        )
        # The old radial standoff used the maximum distance of the *current*
        # open-finger links from their midpoint.  That is a circumscribed
        # envelope and is unnecessarily large when the intended side grasp
        # rotates the jaw span perpendicular to the approach ray: the two
        # fingers then occupy the lateral direction, while the palm and the
        # finger body thickness are the relevant radial clearances.  Use a
        # directional envelope when the runtime link measurements are
        # available; keep the conservative robot-profile envelope for legacy
        # adapters that cannot expose the per-link measurements.  The final
        # whole-body swept certificate remains authoritative either way.
        if envelope_by_link:
            radii = LINK_SPHERE_RADII_BY_SIDE[side]
            palm_envelope = float(
                envelope_by_link.get(
                    f"{side}_gripper_link",
                    R1PRO_GRIPPER_COLLISION_ENVELOPE_M,
                )
            )
            finger_radial_envelope = float(
                R1PRO_GRIPPER_FINGER_HALF_LENGTH_M
                + max(
                    float(radii[f"{side}_gripper_finger_link1"]),
                    float(radii[f"{side}_gripper_finger_link2"]),
                )
            )
            directional_gripper_envelope = max(
                palm_envelope,
                finger_radial_envelope,
            )
            directional_envelope_source = "runtime_palm_and_finger_profile"
        else:
            directional_gripper_envelope = float(gripper_envelope)
            directional_envelope_source = envelope_source
        support_top_z = 0.0 if bool(getattr(getattr(scene, "world", None), "ground", False)) else -float("inf")
        if support_name and hasattr(scene, "object"):
            support = scene.object(support_name)
            support_top_z = float(getattr(support, "top_z"))
        # Side acquisition keeps the *complete* finger collision boxes above
        # the active support and places them inside the object's upper grasp
        # band.  The vertical envelope is a robot capability measured from
        # the supplied USD boxes; using only the link-origin half-height lets
        # the lower R1Pro finger penetrate the floor before contact.
        minimum_center_z = (
            support_top_z
            + R1PRO_GRIPPER_FINGER_SUPPORT_ENVELOPE_Z_M
            + contact_offset
            + R1PRO_GRIPPER_PREGRASP_CLEARANCE_M
            + R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M
        )
        object_top_z = float(world[2]) + 0.5 * float(object_height)
        grasp_band_center_z = (
            object_top_z
            - R1PRO_GRIPPER_FINGER_HALF_HEIGHT_M
            - contact_offset
            - R1PRO_GRIPPER_PREGRASP_CLEARANCE_M
        )
        targets: list[tuple[np.ndarray, dict[str, Any]]] = []
        for offset in R1PRO_SUPPORT_AWARE_APPROACH_OFFSETS_RAD:
            angle = radial_angle + float(offset)
            approach_direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
            # Projection of the authored object footprint along this approach
            # ray; cylinders and oriented cuboids therefore share the same
            # geometry contract.
            object_extent_along_approach = (
                abs(float(approach_direction[0])) * half_x
                + abs(float(approach_direction[1])) * half_y
            )
            direction_standoff = (
                object_extent_along_approach
                + max(
                    R1PRO_GRIPPER_FINGER_HALF_LENGTH_M,
                    directional_gripper_envelope,
                )
                + planning_margin
                + contact_offset
                + R1PRO_GRIPPER_PREGRASP_CLEARANCE_M
            )
            direction_standoff = max(
                direction_standoff,
                R1PRO_GRIPPER_FINGER_HALF_LENGTH_M + 0.02,
            )
            target = world.copy()
            target[:2] -= approach_direction * direction_standoff
            target[2] = max(float(world[2]), grasp_band_center_z, minimum_center_z)
            if not np.all(np.isfinite(target)):
                continue
            targets.append(
                (
                    target,
                    {
                        "approach_direction_world": approach_direction.tolist(),
                        "approach_offset_rad": float(offset),
                        "object_extent_along_approach_m": float(object_extent_along_approach),
                        "object_vertical_extent_m": float(object_height),
                        "standoff_m": float(direction_standoff),
                        "gripper_collision_envelope_m": float(gripper_envelope),
                        "gripper_envelope_source": envelope_source,
                        "gripper_envelope_by_link_m": envelope_by_link,
                        "directional_gripper_envelope_m": float(
                            directional_gripper_envelope
                        ),
                        "directional_envelope_source": directional_envelope_source,
                        "support_top_z_m": float(support_top_z),
                        "object_top_z_m": float(object_top_z),
                        "grasp_band_center_z_m": float(grasp_band_center_z),
                        "target_grasp_height_m": float(target[2]),
                        "planning_margin_m": float(planning_margin),
                        "contact_offset_m": float(contact_offset),
                        "side": side,
                    },
                )
            )
        return tuple(targets)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return ()


def _gripper_collision_envelope(
    adapter: Any,
    side: str,
) -> tuple[float, str, dict[str, float]]:
    """Return a measured-or-calibrated planar gripper collision envelope.

    The object is an obstacle during pregrasp certification.  Therefore the
    relevant radial clearance is not only the finger half-length: the palm
    link and the finger collision spheres are offset from the physical
    midpoint used as the motion target.  Measuring those offsets at runtime
    captures loaded-torso/tool residuals; the robot profile remains a
    conservative lower bound for adapters that do not expose link poses.
    """
    side = require_side(side)
    profile = float(R1PRO_GRIPPER_COLLISION_ENVELOPE_M)
    names = (
        f"{side}_gripper_link",
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
    )
    radii = LINK_SPHERE_RADII_BY_SIDE[side]
    midpoint = _live_grasp_center(adapter, side)
    if midpoint is None or not hasattr(adapter, "body_position"):
        return profile, "robot_profile", {}
    measured: dict[str, float] = {}
    for name in names:
        try:
            point = np.asarray(adapter.body_position(name), dtype=float)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            continue
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        planar_offset = float(np.linalg.norm(point[:2] - midpoint[:2]))
        measured[name] = planar_offset + float(radii[name])
    if not measured:
        return profile, "robot_profile", {}
    return max(profile, max(measured.values())), "runtime_and_profile", measured


def pregrasp_motion_tolerance(model: Any) -> float:
    """Return the maximum allowed object displacement before attachment.

    This is deliberately tighter than the planning clearance.  A pre-grasp
    correction is not allowed to turn incidental palm contact into a push; the
    object-motion guard must stop on the first measurable displacement while
    still allowing small rigid-body solver noise.
    """
    try:
        half_x, half_y = object_xy_half_extents_m(model)
        footprint = max(0.01, float(max(half_x, half_y)))
    except (AttributeError, TypeError, ValueError):
        footprint = 0.03
    physics = getattr(model, "physics", None)
    planning_margin = max(0.0, float(getattr(physics, "planning_margin", 0.0) or 0.0))
    contact_offset = max(0.0, float(getattr(physics, "contact_offset", 0.0) or 0.0))
    nominal = max(
        0.001,
        0.05 * footprint,
        0.5 * contact_offset,
        0.10 * planning_margin,
    )
    # Cap below a visible push, but above ordinary PhysX settling after a
    # long navigation. A 3 mm hard cap rejected ~3.02 mm remote motion while
    # the gripper was still ~10 cm from the object.
    return float(min(0.008, nominal))


def _live_object_position(adapter: Any, object_name: str) -> np.ndarray | None:
    if not hasattr(adapter, "object_position"):
        return None
    try:
        value = np.asarray(adapter.object_position(object_name), dtype=float)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    return value


def _live_grasp_center(adapter: Any, side: str) -> np.ndarray | None:
    """Read the physical midpoint used as the grasp/recovery anchor."""
    try:
        poses = adapter.end_effector_poses() if hasattr(adapter, "end_effector_poses") else {}
        pose = (poses or {}).get(f"{side}_gripper_finger_midpoint")
        if pose is not None:
            value = np.asarray(pose[:3], dtype=float)
            if value.shape == (3,) and np.all(np.isfinite(value)):
                return value
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        pass
    return None


def _live_grasp_center_height(adapter: Any, side: str) -> float | None:
    """Read the current physical grasp-center height when the adapter exposes it."""
    try:
        poses = adapter.end_effector_poses() if hasattr(adapter, "end_effector_poses") else {}
        pose = (poses or {}).get(f"{side}_gripper_finger_midpoint")
        if pose is None:
            return None
        value = float(pose[2])
        return value if math.isfinite(value) else None
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _live_grasp_orientation_base(
    adapter: Any,
    side: str,
    base_pose: Sequence[float],
) -> list[float] | None:
    """Convert the measured world gripper orientation into the base frame."""
    try:
        poses = adapter.end_effector_poses() if hasattr(adapter, "end_effector_poses") else {}
        pose = (poses or {}).get(f"{side}_gripper_finger_midpoint")
        if pose is None or len(pose) < 7:
            return None
        quat_world = np.asarray(pose[3:7], dtype=float)
        if quat_world.shape != (4,) or not np.all(np.isfinite(quat_world)):
            return None
        quat_world = quat_world / np.linalg.norm(quat_world)
        world_rotation = Rotation.from_quat(
            [quat_world[1], quat_world[2], quat_world[3], quat_world[0]]
        )
        yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
        base_rotation = Rotation.from_euler("z", -yaw) * world_rotation
        quat = base_rotation.as_quat()
        return [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def support_aware_orientation_candidates(side: str) -> tuple[list[float], ...]:
    """Return robot-level in-plane grasp orientations in deterministic order.

    The candidates rotate the calibrated gripper orientation around the
    support normal in the current base frame.  They are deliberately limited
    to the robot's planar symmetry instead of exposing arbitrary quaternions
    to a task planner; every candidate still goes through live IK and dense
    collision certification before execution.
    """
    side = require_side(side)
    default = np.asarray(R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE[side], dtype=float)
    if default.shape != (4,) or not np.all(np.isfinite(default)):
        raise ValueError("robot default grasp orientation must be a finite quaternion")
    base = Rotation.from_quat([default[1], default[2], default[3], default[0]])
    candidates: list[list[float]] = []
    for offset in R1PRO_SUPPORT_AWARE_YAW_OFFSETS_RAD:
        rotation = Rotation.from_euler("z", float(offset)) * base
        quat = rotation.as_quat()
        candidate = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]
        if not any(abs(float(np.dot(candidate, previous))) > 1.0 - 1.0e-7 for previous in candidates):
            candidates.append(candidate)
    return tuple(candidates)


def world_point_to_base(point: Sequence[float], base_pose: Sequence[float]) -> list[float]:
    """Convert a world point to the robot base frame without changing height."""
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point must be a finite 3-vector")
    if len(base_pose) < 3:
        raise ValueError("base_pose must contain x, y, yaw")
    dx = float(point[0]) - float(base_pose[0])
    dy = float(point[1]) - float(base_pose[1])
    cosine = math.cos(float(base_pose[2]))
    sine = math.sin(float(base_pose[2]))
    return [
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        float(point[2]),
    ]


__all__ = [
    "SupportAwareGraspObject",
    "derive_support_aware_pregrasp",
    "derive_support_aware_pregrasp_candidates",
    "pregrasp_motion_tolerance",
    "support_aware_orientation_candidates",
    "world_point_to_base",
]
