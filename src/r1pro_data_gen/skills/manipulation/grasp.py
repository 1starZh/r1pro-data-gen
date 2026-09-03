"""Semantic grasp skill: open, approach, align, and attach in one call."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from r1pro_data_gen.domain import object_vertical_extent_m, object_xy_radius_m
from r1pro_data_gen.robot.robot_config import (
    R1PRO_ALIGNMENT_MIN_PHASE_S,
    R1PRO_ALIGNMENT_SPEED_SCALE,
    R1PRO_GRASP_APPROACH_SPEED_SCALE,
    R1PRO_READY_POSE_SPEED_SCALE,
)

from ..core.base import ParamSpec, SkillResult, stabilize_base
from .gripper import GRIPPER_OPEN
from ..core.sides import rank_arm_sides, require_side


_GRASP_PLANNING_TIME_S = 0.4
_GRASP_SPEED_SCALE = R1PRO_GRASP_APPROACH_SPEED_SCALE
_ALIGN_PLANNING_TIME_S = 0.35
# The measured alignment runs against the compliant, gravity-loaded arm drive.
# Keep this local correction below the ordinary arm-motion profile so a
# newly-solved branch does not create a velocity/damping torque spike while
# the controller is still catching up. The physical effort gate remains
# authoritative for every simulator step.
_ALIGN_SPEED_SCALE = R1PRO_ALIGNMENT_SPEED_SCALE
_ALIGN_MAX_ITERATIONS = 16
# A floor object can require more than the normal tabletop correction distance
# between the safe high approach and the measured finger window. Keep the
# correction incremental, but give the generic low-workspace mode enough
# bounded iterations to descend rather than inventing an unreachable fixed
# standoff.
_GROUND_ALIGN_MAX_ITERATIONS = 48
_HANGING_EE_Z_M = 0.90
_HOME_JOINT_ABS_MAX = 0.20


class GraspObject:
    """Establish a verified two-finger attachment around a named object.

    The caller names the object and optional arm side. Standoff height, grasp
    frame, measured alignment and contact recovery stay inside the skill.
    Low objects require prepare_workspace first; this skill does not crouch
    or run a whole-body pregrasp internally.
    """

    name = "grasp_object"
    tier = "semantic"
    exposed = True
    description = (
        "Attach a named graspable object at the current base stance. Observe "
        "live size and pose; leave side=auto unless geometry or a failure says "
        "otherwise. Use when GoalSpec still needs attached and the object is "
        "not already attached. Low or floor objects need prepare_workspace "
        "first. If contact fails while reachable_from_here is true, retry this "
        "skill; if unreachable, navigate instead. Do not use when the goal "
        "forbids grasping or the object is only pushable. This skill does not "
        "crouch or navigate."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Scene object to grasp", required=True),
        "side": ParamSpec(
            "string",
            "Arm side; omit or use auto to select from live geometry",
            default="auto",
            enum=("auto", "left", "right"),
        ),
    }

    def __init__(
        self,
        gripper_set: Any,
        arm_move_to: Any,
        arm_align_gripper: Any,
        gripper_grasp: Any,
        arm_joint_to: Any = None,
        torso_move_to: Any = None,
        *,
        whole_body_pregrasp: Any = None,
        attempts: Sequence[Mapping[str, float]] | None = None,
    ) -> None:
        self.gripper_set = gripper_set
        self.arm_move_to = arm_move_to
        self.arm_align_gripper = arm_align_gripper
        self.gripper_grasp = gripper_grasp
        self.arm_joint_to = arm_joint_to
        self.torso_move_to = torso_move_to
        self.whole_body_pregrasp = whole_body_pregrasp
        self.attempts = None if attempts is None else tuple(dict(item) for item in attempts)

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
        step_hook: Callable[[], None] | None,
        pregrasp_established: bool = False,
    ) -> SkillResult | None:
        """Optionally establish a safe pre-contact pose before alignment.

        The base grasp primitive keeps the historical high standoff behavior.
        Higher-level grasp implementations may override this hook to derive a
        support-aware approach from live geometry.  Returning ``None`` means
        that no additional pre-alignment phase is needed; returning a result
        makes the phase part of the same atomic grasp transaction.
        """
        del (
            adapter,
            scene,
            object_name,
            object_model,
            object_world,
            support_name,
            low_object,
            side,
            step_hook,
            pregrasp_established,
        )
        return None

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
        """Return optional live-geometry inputs for the low-body solver.

        The base grasp contract keeps the historical object-derived high
        target.  Support-aware subclasses can provide a geometry-derived
        low-side target without exposing a task coordinate or a fixed torso
        vector to the planner.
        """
        del (
            adapter,
            scene,
            object_name,
            object_model,
            object_world,
            support_name,
            low_object,
            side,
        )
        return {}

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
        """Return bounded whole-body pregrasp alternatives for one grasp.

        The base skill exposes one target for compatibility with existing
        subclasses.  A geometry-aware skill may override this hook with a
        finite set of live-derived alternatives.  Alternatives are retried
        only when the child solver rejects them before issuing a physical
        waypoint; once execution has started, the grasp transaction fails
        closed and lets the outer closed loop replan from measured state.
        """
        return (
            self._whole_body_pregrasp_parameters(
                adapter,
                scene=scene,
                object_name=object_name,
                object_model=object_model,
                object_world=object_world,
                support_name=support_name,
                low_object=low_object,
                side=side,
            ),
        )

    def _approach(
        self,
        adapter: Any,
        *,
        scene: Any,
        target: Sequence[float],
        side: str,
        exclude: Sequence[str],
        target_quat: Sequence[float] | None = None,
        prefer_local_certified_path: bool = False,
        step_hook: Callable[[], None] | None,
    ) -> SkillResult:
        parameters: dict[str, Any] = {
            "scene": scene,
            "target_pos": list(target),
            "target_frame": "grasp_center",
            "side": side,
            "exclude_objects": list(exclude),
            "planning_time": _GRASP_PLANNING_TIME_S,
            "trajectory_speed_scale": _GRASP_SPEED_SCALE,
            "prefer_local_certified_path": prefer_local_certified_path,
            "step_hook": step_hook,
        }
        if target_quat is not None:
            parameters["target_quat"] = list(target_quat)
        return self.arm_move_to.execute(
            adapter,
            **parameters,
        )

    def _retreat_to_standoff(
        self,
        adapter: Any,
        *,
        scene: Any,
        target: Sequence[float] | None,
        side: str,
        exclude: Sequence[str],
        prefer_local_certified_path: bool = False,
        step_hook: Callable[[], None] | None,
    ) -> None:
        if target is None:
            return
        self._approach(
            adapter,
            scene=scene,
            target=target,
            side=side,
            exclude=exclude,
            prefer_local_certified_path=prefer_local_certified_path,
            step_hook=step_hook,
        )

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        side: str = "auto",
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        """Run the grasp transaction, resolving ``auto`` to one arm.

        If the first live-geometry-ranked arm fails before moving the object,
        the other arm receives one bounded opportunity in the same physical
        episode.  No simulator reset or state teleport is used; all later
        checks still come from the live adapter.
        """
        requested_side = require_side(side, allow_auto=True)
        # A complete grasp transaction is an operation phase, even when its
        # first internal action is geometric planning.  Establish the
        # wheel/steer/torso hold before side ranking or any child skill so a
        # planning rejection cannot leave the articulated robot in the
        # navigation controller's free-spin mode.  An existing explicit mask
        # remains authoritative inside ``stabilize_base``.
        stabilize_base(adapter)
        if requested_side != "auto":
            return self._execute_single_side(
                adapter,
                scene=scene,
                object_name=object_name,
                side=requested_side,
                step_hook=step_hook,
            )
        side_attempts: list[dict[str, Any]] = []
        last_result: SkillResult | None = None
        for candidate in rank_arm_sides(adapter, object_name=object_name):
            result = self._execute_single_side(
                adapter,
                scene=scene,
                object_name=object_name,
                side=candidate,
                step_hook=step_hook,
            )
            last_result = result
            side_attempts.append(
                {
                    "side": candidate,
                    "success": bool(result.success),
                    "failure_code": _failure_code(result),
                    "metrics": dict(result.metrics),
                    "details": dict(result.details),
                }
            )
            if result.success:
                return SkillResult(
                    True,
                    self.name,
                    metrics=dict(result.metrics),
                    details={
                        **dict(result.details),
                        "requested_side": "auto",
                        "side": candidate,
                        "side_attempts": side_attempts,
                    },
                )
            # Once a failed attempt has displaced the target, trying another
            # arm with stale geometry is unsafe. The same boundary applies
            # when a whole-body pregrasp or an arm alignment has already
            # executed but the object stayed still: the other arm would be
            # planned against a robot whose first arm/torso is no longer in
            # the ranked initial state. Let the outer closed loop observe and
            # replan instead.
            if (
                _failure_code(result) == "object_moved_before_grasp"
                or _grasp_result_started_physical_motion(result)
            ):
                break
        if last_result is None:
            return _failure(
                "no_arm_available",
                "no arm side is available for the live grasp",
                attempts=side_attempts,
                skill_name=self.name,
            )
        return SkillResult(
            False,
            self.name,
            metrics=dict(last_result.metrics),
            details={
                **dict(last_result.details),
                "requested_side": "auto",
                "side_attempts": side_attempts,
                "failure_code": _failure_code(last_result) or "grasp_failed",
            },
        )

    def _execute_single_side(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        side: str = "left",
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if not object_name:
            raise ValueError("grasp_object requires object_name")
        if scene is None or not hasattr(scene, "object"):
            return SkillResult(
                False,
                self.name,
                details={"reason": "grasp_object requires a scene", "failure_code": "missing_scene"},
            )
        try:
            object_model = scene.object(object_name)
        except KeyError:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": f"object {object_name!r} is not in the scene",
                    "failure_code": "unknown_object",
                },
            )
        side = require_side(side)
        world = _object_world_position(adapter, object_name)
        support_name = _source_support_name(adapter, scene, object_name)
        low_object = _needs_ground_posture(world, support_name)
        ground_posture_fallback = False
        attempts = self.attempts or _geometry_standoff_attempts(object_model)
        ground_arm_ready = False
        workspace_prepared = _workspace_is_prepared(adapter)
        # A low object is reachable only after prepare_workspace has already
        # moved the torso. This skill must not crouch or run whole-body itself.
        whole_body_prepared = bool(low_object and workspace_prepared)
        whole_body_pregrasp_attempts: list[dict[str, Any]] = []
        if low_object and not workspace_prepared:
            return _failure(
                "workspace_not_prepared",
                "object is below the current torso workspace; call prepare_workspace then retry grasp_object",
                attempts=(),
                skill_name=self.name,
            )
        stabilize_base(adapter, replace_wheel_only=low_object)
        from r1pro_data_gen.skills.mobility.base_motion import _brake_until_stopped

        # Locking the wheels does not instantly kill chassis momentum. Grasp
        # targets are in the live base frame, so a still-spinning yaw would
        # slide the object out of the finger window.
        _brake_until_stopped(adapter, step_hook=step_hook)
        # Do not force a high joint-space ready pose after entering the low
        # torso configuration.  That pose is useful for clearing a tabletop
        # when starting from home, but its long lateral moment is not a
        # robot-independent prerequisite for a floor grasp: on the R1Pro it
        # loads the pitched torso and makes the calibrated low posture collapse
        # before the Cartesian approach begins.  The generic local approach
        # below starts from the measured arm state (home is already within the
        # low-workspace envelope) and will report a bounded reachability/path
        # failure if that measured state cannot reach the live object.
        opened = self.gripper_set.execute(
            adapter,
            scene=scene,
            open_value=GRIPPER_OPEN,
            side=side,
            hold_steps=12,
            step_hook=step_hook,
        )
        if not opened.success:
            return _failure(
                "gripper_not_open",
                "failed to open the gripper before alignment",
                attempts=(),
                last=opened,
                skill_name=self.name,
            )
        hanging = (not low_object) and _arm_is_hanging(adapter, side)
        ready_used = False
        if hanging and self.arm_joint_to is not None:
            # From home the hanging arm saturates the shoulder if it goes
            # straight to a tabletop standoff. Raise to ready first; an arm
            # that is already in the workspace skips this.
            ready = self.arm_joint_to.execute(
                adapter,
                scene=scene,
                target_q=_ready_q(side),
                side=side,
                speed_scale=R1PRO_READY_POSE_SPEED_SCALE,
                step_hook=step_hook,
            )
            ready_used = True
            if not ready.success:
                return _failure(
                    "ready_pose_failed",
                    "failed to raise the arm to a collision-free ready posture",
                    attempts=(),
                    last=ready,
                    skill_name=self.name,
                )

        attempt_records: list[dict[str, Any]] = []
        last_align: SkillResult | None = None
        last_safe_standoff: list[float] | None = None
        exclude = [name for name in (object_name, support_name) if name]
        if low_object and not whole_body_prepared:
            # A stability-certified whole-body pregrasp already placed the
            # gripper at the geometry-derived low interaction standoff.  Do
            # not replay the legacy high grasp-center approach here: it is a
            # separate arm-only motion, can select a different self-motion
            # branch, and would discard the certified low posture before the
            # measured alignment phase.  This branch remains only for a
            # compatible low-object caller that has no whole-body pregrasp
            # result and therefore still needs the generic high waypoint.
            high_height = _safe_approach_center_z(object_model, world) - float(world[2])
            high_target = _standoff_target(
                adapter,
                object_name,
                {"height_m": high_height, "yaw_rad": 0.0, "nudge_m": 0.0},
            )
            if high_target is not None:
                high_move = self._approach(
                    adapter,
                    scene=scene,
                    target=high_target,
                    side=side,
                    exclude=exclude,
                    prefer_local_certified_path=low_object,
                    step_hook=step_hook,
                )
                attempt_records.append(
                    {
                        "index": 0,
                        "standoff": {"height_m": high_height, "yaw_rad": 0.0, "nudge_m": 0.0},
                        "target_pos": list(high_target),
                        "approach_success": bool(high_move.success),
                        "approach_failure_code": _failure_code(high_move),
                        "approach_metrics": dict(high_move.metrics),
                        "approach_details": dict(high_move.details),
                        "phase": "high_approach",
                    }
                )
                if high_move.success:
                    last_safe_standoff = list(high_target)
                else:
                    self._retreat_to_standoff(
                        adapter,
                        scene=scene,
                        target=last_safe_standoff,
                        side=side,
                        exclude=exclude,
                        prefer_local_certified_path=low_object,
                        step_hook=step_hook,
                    )

        # A low object is not necessarily safely acquired by descending from
        # above.  Advanced grasp implementations can insert a geometry-derived
        # side/plane approach here; the resulting pose becomes the new safe
        # standoff for every alignment attempt in this atomic transaction.
        if world is not None:
            prepared_alignment = self._prepare_alignment_standoff(
                adapter,
                scene=scene,
                object_name=object_name,
                object_model=object_model,
                object_world=world,
                support_name=support_name,
                low_object=low_object,
                side=side,
                step_hook=step_hook,
                pregrasp_established=whole_body_prepared,
            )
            if prepared_alignment is not None:
                prepared_target = prepared_alignment.details.get("target_pos")
                attempt_records.append(
                    {
                        "index": -1,
                        "phase": "support_aware_pregrasp",
                        "target_pos": list(prepared_target) if isinstance(prepared_target, (list, tuple)) else prepared_target,
                        "approach_success": bool(prepared_alignment.success),
                        "approach_failure_code": _failure_code(prepared_alignment),
                        "approach_metrics": dict(prepared_alignment.metrics),
                        "approach_details": dict(prepared_alignment.details),
                    }
                )
                if not prepared_alignment.success:
                    prepared_failure_code = _failure_code(prepared_alignment)
                    failure_code = (
                        "object_moved_before_grasp"
                        if prepared_failure_code == "object_moved_before_grasp"
                        else "support_aware_pregrasp_failed"
                    )
                    failure_reason = (
                        "target object moved before attachment; aborting grasp retries"
                        if failure_code == "object_moved_before_grasp"
                        else "support-aware pregrasp could not establish a collision-free non-contact approach"
                    )
                    return _failure(
                        failure_code,
                        failure_reason,
                        attempts=attempt_records,
                        last=prepared_alignment,
                        skill_name=self.name,
                    )
                if (
                    isinstance(prepared_target, (list, tuple))
                    and len(prepared_target) == 3
                ):
                    last_safe_standoff = [float(value) for value in prepared_target]

        for index, attempt in enumerate(attempts, start=1):
            target = _standoff_target(adapter, object_name, attempt)
            if target is None:
                return _failure(
                    "object_pose_unavailable",
                    "live object pose could not be read",
                    attempts=attempt_records,
                    skill_name=self.name,
                )
            reuse_high_standoff = bool(low_object and last_safe_standoff is not None)
            if reuse_high_standoff:
                # The low workspace is approached from the already certified
                # high pose. A second fixed z standoff can be outside the
                # current torso/arm workspace; measured alignment below owns
                # the continuous descent and the final jaw-window check.
                moved = SkillResult(
                    True,
                    "arm_move_to",
                    details={
                        "reason": "reusing high standoff for measured low-workspace descent",
                        "reused_high_standoff": True,
                    },
                )
            else:
                moved = self._approach(
                    adapter,
                    scene=scene,
                    target=target,
                    side=side,
                    exclude=exclude,
                    prefer_local_certified_path=low_object,
                    step_hook=step_hook,
                )
                if (
                    not moved.success
                    and hanging
                    and not ready_used
                    and self.arm_joint_to is not None
                ):
                    ready = self.arm_joint_to.execute(
                        adapter,
                        scene=scene,
                        target_q=_ready_q(side),
                        side=side,
                        speed_scale=R1PRO_READY_POSE_SPEED_SCALE,
                        step_hook=step_hook,
                    )
                    ready_used = True
                    if not ready.success:
                        return _failure(
                            "ready_pose_failed",
                            "failed to raise the arm to a collision-free ready posture",
                            attempts=attempt_records,
                            last=ready,
                            skill_name=self.name,
                        )
                    moved = self._approach(
                        adapter,
                        scene=scene,
                        target=target,
                        side=side,
                        exclude=exclude,
                        prefer_local_certified_path=low_object,
                        step_hook=step_hook,
                    )
            record = {
                "index": index,
                "standoff": dict(attempt),
                "target_pos": list(target),
                "approach_success": bool(moved.success),
                "approach_failure_code": _failure_code(moved),
                "approach_metrics": dict(moved.metrics),
                "approach_details": dict(moved.details),
                "reused_high_standoff": reuse_high_standoff,
            }
            if not moved.success:
                attempt_records.append(record)
                if moved.details.get("position_reachable_without_orientation") is False:
                    return _failure(
                        "unreachable_from_base",
                        "object is not reachable from the current base pose",
                        attempts=attempt_records,
                        last=moved,
                        skill_name=self.name,
                    )
                self._retreat_to_standoff(
                    adapter,
                    scene=scene,
                    target=last_safe_standoff,
                    side=side,
                    exclude=exclude,
                    prefer_local_certified_path=low_object,
                    step_hook=step_hook,
                )
                continue
            if not reuse_high_standoff:
                last_safe_standoff = list(target)
            aligned = self.arm_align_gripper.execute(
                adapter,
                scene=scene,
                object_name=object_name,
                side=side,
                require_between_fingers=True,
                require_vertical_alignment=True,
                exclude_objects=[support_name] if support_name else [],
                trajectory_speed_scale=_ALIGN_SPEED_SCALE,
                planning_time=_ALIGN_PLANNING_TIME_S,
                max_iterations=(
                    _GROUND_ALIGN_MAX_ITERATIONS
                    if low_object
                    else _ALIGN_MAX_ITERATIONS
                ),
                # Keep the no-push baseline across every bounded attempt in
                # this single grasp transaction.  Resetting it after a failed
                # alignment would convert several small contacts into a large
                # cumulative displacement before the outer transfer can
                # replan from live state.
                object_motion_reference_position=world,
                step_hook=step_hook,
            )
            last_align = aligned
            record["align_success"] = bool(aligned.success)
            record["align_failure_code"] = _failure_code(aligned)
            record["between_fingers"] = aligned.details.get("between_fingers")
            record["align_metrics"] = dict(aligned.metrics)
            record["support_name"] = support_name
            attempt_records.append(record)
            if aligned.success:
                grasped = self.gripper_grasp.execute(
                    adapter,
                    scene=scene,
                    object_name=object_name,
                    side=side,
                    step_hook=step_hook,
                )
                if grasped.success:
                    return SkillResult(
                        True,
                        self.name,
                        metrics={
                            "attempts": float(index),
                            "attached": 1.0,
                        },
                        details={
                            "object_name": object_name,
                            "side": side,
                            "attempts": attempt_records,
                            "grasp": grasped.details,
                            "ground_posture_fallback": ground_posture_fallback,
                            "ground_arm_ready": ground_arm_ready,
                            "whole_body_prepared": whole_body_prepared,
                            "whole_body_pregrasp_attempts": whole_body_pregrasp_attempts,
                            "failure_code": None,
                        },
                    )
                record["grasp_success"] = False
                record["grasp_failure_code"] = _failure_code(grasped)
                record["grasp_metrics"] = dict(grasped.metrics)
                if _failure_code(grasped) in {
                    "object_moved_before_grasp",
                    "object_moved_before_attachment",
                }:
                    return _failure(
                        "object_moved_before_grasp",
                        "target object moved before attachment; aborting grasp retries",
                        attempts=attempt_records,
                        last=grasped,
                        skill_name=self.name,
                    )
                # A close that did not pinch is still the same grasp
                # transaction. Reopen, retreat, and try the next geometry so
                # the episode keeps moving instead of idling while an outer
                # planner selects another skill.
                self.gripper_set.execute(
                    adapter,
                    side=side,
                    open_value=GRIPPER_OPEN,
                    step_hook=step_hook,
                )
                self._retreat_to_standoff(
                    adapter,
                    scene=scene,
                    target=last_safe_standoff,
                    side=side,
                    exclude=exclude,
                    prefer_local_certified_path=low_object,
                    step_hook=step_hook,
                )
                continue
            if _failure_code(aligned) == "object_moved_before_grasp":
                # Re-approaching after an un-attached object moved is unsafe:
                # the original geometric plan is stale and another retry can
                # push it farther.  Abort the atomic grasp phase and let the
                # agent replan from the new observation.
                return _failure(
                    "object_moved_before_grasp",
                    "target object moved before attachment; aborting grasp retries",
                    attempts=attempt_records,
                    last=aligned,
                    skill_name=self.name,
                )
            if _alignment_failed_before_execution(aligned):
                # A static swept-volume/IK rejection did not advance the
                # physical arm.  Retreating through the generic arm_move_to
                # path here is actively harmful: it can select a different
                # redundant branch and turn a repeatable safe rejection into
                # the observed joint drift.  End this atomic grasp attempt
                # with the certificate so the outer task loop can replan from
                # the unchanged live state.
                return _failure(
                    _failure_code(aligned) or "alignment_path_unavailable",
                    "alignment candidates were rejected before execution; replan from the live pregrasp",
                    attempts=attempt_records,
                    last=aligned,
                    skill_name=self.name,
                )
            self._retreat_to_standoff(
                adapter,
                scene=scene,
                target=last_safe_standoff,
                side=side,
                exclude=exclude,
                prefer_local_certified_path=low_object,
                step_hook=step_hook,
            )
        code = _failure_code(last_align) if last_align is not None else "no_collision_free_approach"
        if code == "contact_not_centered":
            reason = "measured alignment stayed outside the finger window"
        elif last_align is None:
            code = "no_collision_free_approach"
            reason = "no collision-free non-contact standoff was found"
        else:
            reason = "measured alignment did not reach a grasp-ready window"
        return _failure(
            code,
            reason,
            attempts=attempt_records,
            last=last_align,
            skill_name=self.name,
        )


def _source_support_name(adapter: Any, scene: Any, object_name: str) -> str | None:
    from r1pro_data_gen.skills.manipulation.carry import _infer_source_support_surface

    world = _object_world_position(adapter, object_name)
    try:
        model = scene.object(object_name)
    except (AttributeError, KeyError):
        return None
    if world is None:
        return None
    return _infer_source_support_surface(scene, model, world)


def _workspace_is_prepared(adapter: Any) -> bool:
    """True when the live torso has left the standing workspace profile.

    Floor objects need prepare_workspace(floor) first. Standing joints near
    the calibrated tabletop/carry/travel profile mean that skill has not run.
    """
    from r1pro_data_gen.robot.robot_config import (
        R1PRO_TRANSFER_TORSO_Q,
        R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD,
    )
    from r1pro_data_gen.skills.posture.torso import TORSO_JOINTS

    try:
        observation = adapter.read_observation(0.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    positions = getattr(observation, "joint_positions", None) or {}
    standing = tuple(float(value) for value in R1PRO_TRANSFER_TORSO_Q)
    if len(standing) != len(TORSO_JOINTS):
        return False
    errors = []
    for index, name in enumerate(TORSO_JOINTS):
        if name not in positions:
            return False
        try:
            errors.append(abs(float(positions[name]) - standing[index]))
        except (TypeError, ValueError):
            return False
    return max(errors) > float(R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD)


def _needs_ground_posture(
    world: Sequence[float] | None,
    support_name: str | None,
) -> bool:
    """Select the low-workspace posture from measured object height.

    A named source support does not imply elevated clearance: a floor can be
    represented by a scene object, and a low platform can support an object
    below the robot's normal manipulation envelope.  The live object height
    is the invariant that selects the posture; support identity is consumed by
    the support-aware geometry solver instead.
    """
    del support_name
    if world is None:
        return False
    from r1pro_data_gen.robot.robot_config import R1PRO_GROUND_INTERACTION_CENTER_Z_M

    return float(world[2]) < float(R1PRO_GROUND_INTERACTION_CENTER_Z_M)


def _safe_approach_center_z(model: Any, world: Sequence[float]) -> float:
    """Compute a collision-safe high waypoint from object extent and robot limits."""
    from r1pro_data_gen.robot.robot_config import R1PRO_SAFE_APPROACH_CENTER_Z_M

    try:
        extent = max(0.02, float(object_vertical_extent_m(model)))
    except (AttributeError, TypeError, ValueError):
        extent = 0.10
    geometry_clearance = max(0.02, 0.25 * extent)
    return max(
        float(R1PRO_SAFE_APPROACH_CENTER_Z_M),
        float(world[2]) + 0.5 * extent + geometry_clearance,
    )


def _geometry_standoff_attempts(model: Any) -> tuple[dict[str, float], ...]:
    """Build bounded grasp retries from the selected object's geometry.

    The old implementation embedded a list tuned for one cylinder. This
    generator keeps the same recovery idea—small height/orientation/radial
    variations—but scales it with the actual object footprint and height, so
    cuboids, cylinders, and future primitive types share one skill contract.
    """
    try:
        footprint = max(0.01, float(object_xy_radius_m(model)))
    except (AttributeError, TypeError, ValueError):
        footprint = 0.03
    try:
        height = max(0.02, float(object_vertical_extent_m(model)))
    except (AttributeError, TypeError, ValueError):
        height = 0.10
    approach_height = max(0.12, 1.5 * height)
    nudge = min(0.05, max(0.015, 1.5 * footprint))
    yaw_offset = min(0.40, max(0.15, nudge / max(footprint, 0.02)))
    lower_height = max(0.08, 0.75 * approach_height)
    higher_height = min(0.30, 1.25 * approach_height)
    return (
        {"height_m": approach_height, "yaw_rad": 0.0, "nudge_m": -nudge},
        {"height_m": approach_height, "yaw_rad": 0.0, "nudge_m": 0.0},
        {"height_m": approach_height, "yaw_rad": yaw_offset, "nudge_m": -nudge},
        {"height_m": approach_height, "yaw_rad": -yaw_offset, "nudge_m": -nudge},
        {"height_m": lower_height, "yaw_rad": 0.0, "nudge_m": -nudge},
        {"height_m": approach_height, "yaw_rad": 0.0, "nudge_m": nudge},
        {"height_m": higher_height, "yaw_rad": 0.0, "nudge_m": 0.0},
        {"height_m": approach_height, "yaw_rad": yaw_offset, "nudge_m": -0.5 * nudge},
        {"height_m": approach_height, "yaw_rad": -yaw_offset, "nudge_m": -0.5 * nudge},
    )


def _ready_q(side: str) -> list[float]:
    from r1pro_data_gen.robot.robot_config import R1PRO_ARM_READY_Q_BY_SIDE

    return list(R1PRO_ARM_READY_Q_BY_SIDE[side])


def _arm_is_hanging(adapter: Any, side: str) -> bool:
    """True when the arm is at home or the gripper has dropped below chest height.

    Ready is only needed from those postures. An arm already at a tabletop
    standoff must not be commanded through joint-space ready.
    """
    from .arm import ARM_JOINTS_BY_SIDE

    observation = adapter.read_observation(0.0)
    positions = getattr(observation, "joint_positions", {}) or {}
    joints = ARM_JOINTS_BY_SIDE[side]
    if all(name in positions for name in joints):
        peak = max(abs(float(positions[name])) for name in joints)
        if peak < _HOME_JOINT_ABS_MAX:
            return True
    if not hasattr(adapter, "end_effector_poses"):
        return False
    try:
        poses = adapter.end_effector_poses() or {}
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    ee = poses.get(f"{side}_ee")
    return ee is not None and len(ee) >= 3 and float(ee[2]) < _HANGING_EE_Z_M


def _grasp_quat(side: str) -> tuple[float, float, float, float]:
    from r1pro_data_gen.robot.robot_config import R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE

    return R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE[side]


def _standoff_target(
    adapter: Any,
    object_name: str,
    attempt: Mapping[str, float],
) -> list[float] | None:
    world = _object_world_position(adapter, object_name)
    observation = adapter.read_observation(0.0)
    base_pose = getattr(observation, "base_pose", None) or (0.0, 0.0, 0.0)
    if world is None:
        return None
    local = _world_xy_to_base(world, base_pose)
    yaw = float(attempt.get("yaw_rad", 0.0))
    nudge = float(attempt.get("nudge_m", 0.0))
    height = float(attempt.get("height_m", 0.20))
    if yaw != 0.0 or nudge != 0.0:
        radius = math.hypot(local[0], local[1])
        heading = math.atan2(local[1], local[0]) + yaw
        radius = max(0.01, radius + nudge)
        local = [radius * math.cos(heading), radius * math.sin(heading), local[2]]
    return [float(local[0]), float(local[1]), float(local[2] + height)]


def _object_world_position(adapter: Any, object_name: str) -> tuple[float, float, float] | None:
    if hasattr(adapter, "object_position"):
        try:
            position = adapter.object_position(object_name)
        except (KeyError, RuntimeError, TypeError, ValueError):
            position = None
        if position is not None and len(position) >= 3:
            return (float(position[0]), float(position[1]), float(position[2]))
    if hasattr(adapter, "gripper_object_alignment"):
        try:
            alignment = adapter.gripper_object_alignment(object_name)
            position = alignment.get("object_position")
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None
        if position is not None and len(position) >= 3:
            return (float(position[0]), float(position[1]), float(position[2]))
    return None


def _world_xy_to_base(
    world: tuple[float, float, float],
    base_pose: Sequence[float],
) -> list[float]:
    dx = float(world[0]) - float(base_pose[0])
    dy = float(world[1]) - float(base_pose[1])
    yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return [cosine * dx + sine * dy, -sine * dx + cosine * dy, float(world[2])]


def _failure_code(result: SkillResult | None) -> str | None:
    if result is None:
        return None
    for source in (result.details, result.metrics):
        value = source.get("failure_code")
        if isinstance(value, str) and value.strip():
            return value
    reason = result.details.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason
    return None


def _grasp_result_started_physical_motion(result: SkillResult | None) -> bool:
    """Return whether a failed side attempt crossed the physical boundary.

    ``side='auto'`` may try a second arm only when the first side was rejected
    before issuing a manipulation motion.  A successful approach, a measured
    alignment attempt, or a certified whole-body pregrasp means the live robot
    state has changed even if the object did not move.  Reusing the original
    side ranking after that point can command the other arm from stale torso,
    collision, and effort assumptions.
    """
    if result is None:
        return False
    details = result.details or {}
    if bool(details.get("whole_body_prepared", False)):
        return True
    attempts = details.get("attempts")
    if not isinstance(attempts, (list, tuple)):
        return False
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        if attempt.get("align_success") is not None:
            return True
        if bool(attempt.get("approach_success", False)):
            return True
    return False


def _alignment_failed_before_execution(result: SkillResult | None) -> bool:
    """Return whether an alignment failure contains no executed arm segment."""
    if result is None or result.success:
        return False
    if _failure_code(result) != "correction_motion_failed":
        return False
    motion = result.details.get("motion")
    if not isinstance(motion, Mapping):
        return False
    return motion.get("failure_code") in {
        "alignment_path_unavailable",
        "alignment_ik_failed",
        "alignment_collision_check_unavailable",
        # These codes can be returned after the measured clearance sequence
        # has already moved the arm.  Retrying another standoff from that
        # altered physical state would invalidate the frozen collision and
        # object-motion assumptions of this atomic grasp phase.
        "alignment_collision_detected",
        "alignment_orientation_ik_failed",
        "alignment_orientation_unavailable",
    }


def _failure(
    code: str,
    reason: str,
    *,
    attempts: Sequence[Mapping[str, Any]],
    last: SkillResult | None = None,
    skill_name: str = "grasp_object",
) -> SkillResult:
    details: dict[str, Any] = {
        "reason": reason,
        "failure_code": code,
        "attempts": list(attempts),
    }
    if last is not None:
        details["last_skill"] = last.skill
        details["last_metrics"] = last.metrics
        details["last_details"] = last.details
    return SkillResult(False, skill_name, details=details)


__all__ = ["GraspObject"]
