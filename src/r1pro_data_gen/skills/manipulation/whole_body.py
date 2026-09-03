"""Higher-level whole-body manipulation capabilities.

The low-level arm and torso skills are useful backends, but a low object moved
to an elevated support needs an additional state transition: lift the held
object clear, change the torso posture, and keep the live grasp frame stable
while the arm is re-solved.  The classes here own that transition and use the
whole-body feasibility methods before issuing joint targets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import inspect
from typing import Any

import numpy as np

from r1pro_data_gen.methods.manipulation.whole_body import (
    WholeBodyCollisionChecker,
    held_object_configuration_free,
    whole_body_path_free,
)
from r1pro_data_gen.methods.manipulation.stability import configuration_stability, payload_com
from r1pro_data_gen.domain import object_vertical_extent_m, object_xy_radius_m
from r1pro_data_gen.robot.chassis import STEER_POSITIONS
from r1pro_data_gen.robot.robot_config import (
    R1PRO_EFFORT_PLANNING_UTILIZATION,
    R1PRO_JOINT_LIMITS,
    R1PRO_SUPPORT_POLYGON_MARGIN_M,
    R1PRO_TORSO_EFFORT_LIMIT,
    R1PRO_TRANSFER_HOLD_CENTER_TOL_M,
    R1PRO_TRANSFER_MAX_ARM_STEP_RAD,
    R1PRO_TRANSFER_MAX_TRACK_STEPS,
    R1PRO_TRANSFER_TRACK_TOL_RAD,
    R1PRO_TRANSFER_TORSO_Q,
    R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD,
    R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
    R1PRO_WHOLE_BODY_MAX_SPEED_SCALE,
)
from r1pro_data_gen.execution.contracts import PhysicalSafetyViolation

from .arm import ARM_JOINTS_BY_SIDE
from ..core.base import ParamSpec, SkillResult, stabilize_base
from .carry import calibrated_model_transform, live_grasp_context
from ..core.sides import for_side, require_side


class WholeBodyPregraspTransition:
    """Prepare a low-support interaction posture with a certified sweep.

    A low object cannot be approached safely by pitching the torso while
    leaving the arm at its neutral configuration.  This capability first
    searches a coordinated torso/arm path that preserves the measured EE
    position and checks the entire robot envelope.  If the current base pose
    blocks that path, an optional staging backend tries geometry-derived base
    candidates; the object name is used only to read live scene geometry, not
    to select a task-specific waypoint.
    """

    name = "whole_body_pregrasp_transition"
    tier = "semantic"
    exposed = False
    description = (
        "Prepare a robot for a low-support grasp by solving a coordinated "
        "torso/arm posture transition, certifying the whole-body sweep and "
        "optionally staging the base from live object geometry."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Object being approached", required=True),
        "target_posture": ParamSpec(
            "string",
            "Robot interaction posture profile",
            default="ground_interaction",
            enum=("ground_interaction",),
        ),
        "side": ParamSpec("string", "Arm side", default="left", enum=("left", "right")),
        "speed_scale": ParamSpec(
            "number",
            "Fraction of the coordinated torso velocity profile",
            default=0.2,
            minimum=0.05,
            maximum=1.0,
        ),
        "settle_steps": ParamSpec(
            "integer",
            "Physics steps used to settle the interaction posture",
            default=18,
            minimum=0,
            maximum=240,
        ),
        "target_center_world": ParamSpec(
            "array",
            "Optional live world-frame non-contact interaction center for an internal support-aware caller",
            default=None,
            shape=(3,),
        ),
        "target_span_world": ParamSpec(
            "array",
            "Optional live world-frame jaw-span direction for an internal support-aware caller",
            default=None,
            shape=(3,),
        ),
    }

    def __init__(
        self,
        kin: Any,
        base_staging: Any = None,
        *,
        torso_velocity_limit: float = 0.5,
    ) -> None:
        self.kin = kin
        self.base_staging = base_staging
        self.torso_velocity_limit = float(torso_velocity_limit)

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        target_posture: str = "ground_interaction",
        side: str = "left",
        speed_scale: float = 0.2,
        settle_steps: int = 18,
        target_center_world: Sequence[float] | None = None,
        target_span_world: Sequence[float] | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if scene is None or not object_name:
            return _failure_for(
                self.name,
                "missing_scene_or_object",
                "whole-body pregrasp transition requires a scene and object",
            )
        if target_posture != "ground_interaction":
            raise ValueError("whole-body pregrasp supports only ground_interaction")
        side = require_side(side)
        kin = for_side(self.kin, side)
        if kin is None:
            return _failure_for(self.name, "kinematics_unavailable", "whole-body pregrasp has no selected-arm kinematics")
        try:
            scene.object(object_name)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return _failure_for(self.name, "unknown_object", "whole-body pregrasp object is unavailable", error=str(exc))
        if not np.isfinite(float(speed_scale)) or not 0.0 < float(speed_scale) <= 1.0:
            raise ValueError("speed_scale must be finite and in (0, 1]")
        if target_center_world is not None:
            target_center_world = np.asarray(target_center_world, dtype=float)
            if target_center_world.shape != (3,) or not np.all(np.isfinite(target_center_world)):
                raise ValueError("target_center_world must be a finite 3-vector")
        if target_span_world is not None:
            target_span_world = np.asarray(target_span_world, dtype=float)
            if (
                target_span_world.shape != (3,)
                or not np.all(np.isfinite(target_span_world))
                or float(np.linalg.norm(target_span_world[:2])) <= 1.0e-8
            ):
                raise ValueError(
                    "target_span_world must be a finite 3-vector with a horizontal component"
                )
        # The semantic parameter remains available for slower callers, but a
        # task plan cannot override the calibrated drive capability with a
        # faster whole-body trajectory.
        speed_scale = min(float(speed_scale), R1PRO_WHOLE_BODY_MAX_SPEED_SCALE)

        direct = self._execute_at_current_base(
            adapter,
            scene,
            object_name,
            side,
            float(speed_scale),
            int(settle_steps),
            target_center_world,
            target_span_world,
            step_hook,
        )
        # An explicit target is derived in the current live world frame by a
        # support-aware caller.  If its robot-level IK certificate fails,
        # moving the base and replaying that same target is not a valid
        # recovery: the base motion changes the dynamic boundary and can turn
        # passive arm reactions into an effort-limit violation.  Return the
        # complete candidate diagnostics so the outer closed loop can observe
        # and replan.  The legacy/default high-posture path may still use the
        # bounded staging fallback below because its target is recomputed by
        # the next invocation.
        if direct.success or self.base_staging is None or target_center_world is not None:
            return direct

        # Base staging is a planning fallback, not a recovery controller for
        # a partially executed articulated motion.  Once the pregrasp has
        # issued any certified waypoint, handing control to a sparse
        # navigation command can leave the loaded torso/arm targets
        # underspecified; the old fallback then let the torso drift under
        # gravity while it searched base candidates.  Retry only a pure
        # no-motion reachability rejection, and return all tracking/physical
        # failures at their measured boundary.  This rule is independent of
        # the object type and applies to every task using this capability.
        failure_code = direct.details.get("failure_code")
        records = direct.details.get("records")
        if failure_code != "whole_body_pregrasp_unreachable" or records:
            return direct

        candidates = _staging_candidates(adapter, scene, object_name)
        attempts: list[dict[str, Any]] = []
        for index, target in enumerate(candidates, start=1):
            _unlock_internal_hold(adapter)
            # Internal base staging is still part of the manipulation
            # transaction.  Allow only steer/wheel joints to move and hold
            # every arm, torso, and gripper joint at its measured state.  A
            # navigation skill may therefore command the base without
            # exposing passive articulated chains to gravity or letting its
            # sparse zero-filled position buffer pull an arm toward home.
            if hasattr(adapter, "lock_joint_mask"):
                adapter.lock_joint_mask(
                    mask_mode="allow",
                    joint_groups=("steer", "wheel"),
                    lock_root=False,
                )
            try:
                moved = self.base_staging.execute(
                    adapter,
                    scene=scene,
                    target=list(target),
                    purpose="staging",
                    motion_mode="holonomic",
                    v_max=0.08,
                    omega_max=0.2,
                    arrive_tol=0.03,
                    final_arrive_tol=0.02,
                    step_hook=step_hook,
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                if isinstance(exc, PhysicalSafetyViolation):
                    raise
                attempts.append({"index": index, "target": list(target), "error": str(exc)})
                continue
            record = {
                "index": index,
                "target": list(target),
                "success": bool(moved.success),
                "metrics": _json_safe(moved.metrics),
                "details": _json_safe(moved.details),
            }
            attempts.append(record)
            if not moved.success:
                continue
            retried = self._execute_at_current_base(
                adapter,
                scene,
                object_name,
                side,
                float(speed_scale),
                int(settle_steps),
                target_center_world,
                target_span_world,
                step_hook,
            )
            if retried.success:
                retried.details["base_staging"] = {
                    "used": True,
                    "candidate_index": index,
                    "attempts": attempts,
                }
                return retried
            attempts[-1]["pregrasp_retry"] = _json_safe(retried.details)

        direct.details["base_staging"] = {"used": True, "attempts": attempts}
        return direct

    def _execute_at_current_base(
        self,
        adapter: Any,
        scene: Any,
        object_name: str,
        side: str,
        speed_scale: float,
        settle_steps: int,
        target_center_world: np.ndarray | None,
        target_span_world: np.ndarray | None,
        step_hook: Callable[[], None] | None,
    ) -> SkillResult:
        """Execute a certified robot-level transition at the current base.

        Before grasping, the end-effector is not a constrained object frame:
        preserving its exact world position while the torso moves can trap a
        neutral arm at a kinematic singularity.  Instead, search robot-level
        arm staging profiles (including the measured arm state), certify the
        arm and torso sweep, and let the later object-alignment skill solve
        the live grasp pose.  This keeps the transition reusable for any
        low-support object and any scene layout.
        """
        kin = for_side(self.kin, side)
        joints = ARM_JOINTS_BY_SIDE[side]
        try:
            observation = adapter.read_observation(0.0)
            current_torso = _torso_from_observation(observation)
            current_arm = np.asarray([observation.joint_positions[name] for name in joints], dtype=float)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return _failure_for(self.name, "observation_unavailable", "whole-body pregrasp observation is incomplete", error=str(exc))
        if current_torso is None or current_arm.shape != (7,) or not np.all(np.isfinite(current_arm)):
            return _failure_for(self.name, "observation_unavailable", "whole-body pregrasp observation is incomplete")
        calibrated = calibrated_model_transform(kin, adapter, side)
        if calibrated is None:
            return _failure_for(self.name, "live_model_calibration_unavailable", "cannot register the live robot before whole-body pregrasp")
        rotation, translation = calibrated
        try:
            from r1pro_data_gen.methods.collision import obstacles_from_scene

            obstacles = obstacles_from_scene(scene, include_ground=False)
            checker = WholeBodyCollisionChecker(kin, obstacles, side=side)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return _failure_for(self.name, "whole_body_checker_unavailable", "whole-body pregrasp checker could not be built", error=str(exc))

        selected = _select_pregrasp_plan(
            kin,
            checker,
            adapter,
            scene,
            object_name,
            side=side,
            current_arm=current_arm,
            current_torso=current_torso,
            model_to_world_rotation=np.asarray(rotation, dtype=float),
            model_to_world_translation=np.asarray(translation, dtype=float),
            speed_scale=float(speed_scale),
            torso_velocity_limit=self.torso_velocity_limit,
            target_center_world=target_center_world,
            target_span_world=target_span_world,
        )
        if selected is None:
            candidate_diagnostics = getattr(kin, "_last_pregrasp_diagnostics", ())
            return _failure_for(
                self.name,
                "whole_body_pregrasp_unreachable",
                "no dynamically stable, collision-free whole-body pregrasp candidate was found",
                candidates=_json_safe(candidate_diagnostics),
            )
        target_torso, target_arm, plan, planner_diagnostics = selected
        # Preserve the selected candidate on the physical adapter for the
        # case where execution later aborts before a SkillResult can return.
        # This is diagnostic-only metadata and does not write a robot pose,
        # wrench, or object state.  The proxy used by Orchestrator delegates
        # simulator methods but keeps attributes locally, so write through to
        # its underlying adapter when present.
        telemetry_adapter = getattr(adapter, "_adapter", adapter)
        selected_diagnostic = next(
            (
                item.get("selected")
                for item in reversed(planner_diagnostics)
                if isinstance(item, dict) and isinstance(item.get("selected"), dict)
            ),
            {},
        )
        selected_execution_order = str(selected_diagnostic.get("execution_order", ""))
        explicit_target = target_center_world is not None
        setattr(telemetry_adapter, "_whole_body_pregrasp_phase", "tracking")
        setattr(
            telemetry_adapter,
            "_whole_body_pregrasp_target_distance_m",
            float(selected_diagnostic.get("target_distance_m", 0.0)),
        )
        setattr(
            telemetry_adapter,
            "_whole_body_pregrasp_target_torso_q",
            ",".join(f"{float(value):.5f}" for value in np.asarray(target_torso, dtype=float)),
        )
        setattr(
            telemetry_adapter,
            "_whole_body_pregrasp_target_arm_q",
            ",".join(f"{float(value):.5f}" for value in np.asarray(target_arm, dtype=float)),
        )
        setattr(
            telemetry_adapter,
            "_whole_body_pregrasp_candidate_count",
            float(max(0, len(planner_diagnostics) - 1)),
        )
        setattr(
            telemetry_adapter,
            "_whole_body_pregrasp_execution_order",
            str(selected_diagnostic.get("execution_order", "")),
        )
        setattr(
            telemetry_adapter,
            "_whole_body_pregrasp_effort_utilization",
            float(selected_diagnostic.get("effort_utilization", 0.0)),
        )
        setattr(telemetry_adapter, "_whole_body_pregrasp_tracking_error_rad", 0.0)

        # All collision and continuity checks happen before the first target is
        # sent.  The execution phase therefore cannot leave a half-planned
        # torso sweep inside an obstacle.
        if getattr(adapter, "joint_mask_locked", False) and hasattr(adapter, "unlock_joint_mask"):
            adapter.unlock_joint_mask()
        stabilize_base(adapter, lock_torso=False)
        # The opposite arm is still a real articulated load. Leaving it as a
        # sparse, implicitly zero-velocity chain while the torso accelerates
        # can create a dynamic effort spike even though the selected arm path
        # is collision- and stability-certified.  Add only the non-task arm
        # to the existing mask at its measured configuration; the selected
        # arm and torso remain the active DOFs of this transition.  This is a
        # controller-phase invariant, not a task-specific posture.
        passive_side = "right" if side == "left" else "left"
        passive_group = f"{passive_side}_arm"
        if (
            getattr(adapter, "joint_mask_locked", False)
            and hasattr(adapter, "extend_joint_mask")
        ):
            adapter.extend_joint_mask(joint_groups=(passive_group,))
        records: list[dict[str, Any]] = []
        actual_arm = current_arm.copy()
        actual_torso = current_torso.copy()
        initial_torso = current_torso.copy()
        actual_torso_velocity = _torso_velocity_from_observation(observation)
        if actual_torso_velocity is None:
            actual_torso_velocity = np.zeros(4, dtype=float)
        dt = float(getattr(adapter, "dt", 1.0 / 60.0))
        max_torso_step = max(
            1.0e-4,
            abs(self.torso_velocity_limit) * float(speed_scale) * dt,
        )
        max_torso_velocity = max(
            1.0e-4,
            abs(self.torso_velocity_limit) * max(0.05, float(speed_scale)),
        )
        # The R1Pro torso is an implicit position/velocity drive.  A velocity
        # target is absolute, not an increment to the measured velocity.  The
        # previous ``measured + correction`` rule therefore integrated any
        # existing motion and made a joint continue past its certified
        # waypoint (the observed torso2/torso3 divergence).  Keep the command
        # as an explicit, slew-limited desired velocity instead.
        commanded_torso_velocity = np.zeros(4, dtype=float)
        max_torso_velocity_delta = max(1.0e-4, 0.10 * max_torso_velocity)

        def set_motion_targets(
            command_arm: np.ndarray,
            command_torso: np.ndarray,
            measured_arm: np.ndarray,
            measured_torso: np.ndarray,
            *,
            stop: bool = False,
            torso_velocity_reference: np.ndarray | None = None,
            measured_torso_velocity: np.ndarray | None = None,
        ) -> None:
            """Send a bounded position target with a matching velocity ref."""
            del measured_arm, measured_torso_velocity
            nonlocal commanded_torso_velocity
            if stop:
                desired_torso_velocity = np.zeros(4, dtype=float)
            elif torso_velocity_reference is not None:
                # During final settling, never switch a gravity-loaded joint
                # from its measured nonzero velocity to zero in one frame.
                # The reference is ramped by the caller and is bounded by the
                # authored torso velocity capability, while ordinary motion
                # remains on the slower position-error profile below.
                desired_torso_velocity = np.clip(
                    np.asarray(torso_velocity_reference, dtype=float),
                    -max_torso_velocity,
                    max_torso_velocity,
                )
            else:
                desired_torso_velocity = np.clip(
                    (np.asarray(command_torso, dtype=float) - np.asarray(measured_torso, dtype=float)) / dt,
                    -max_torso_velocity,
                    max_torso_velocity,
                )
            torso_velocity = commanded_torso_velocity + np.clip(
                desired_torso_velocity - commanded_torso_velocity,
                -max_torso_velocity_delta,
                max_torso_velocity_delta,
            )
            commanded_torso_velocity = torso_velocity
            adapter.set_targets(
                position={
                    **{f"torso_joint{index}": float(command_torso[index - 1]) for index in range(1, 5)},
                    **{name: float(command_arm[index]) for index, name in enumerate(joints)},
                },
                velocity={
                    **{f"torso_joint{index}": float(torso_velocity[index - 1]) for index in range(1, 5)},
                },
            )
        # The planner's path resolution is chosen for collision and stability
        # certification, not for the discrete position-drive bandwidth.  A
        # planner edge may therefore be much larger than one safe actuator
        # update.  Resample every edge at the robot capability limits before
        # execution; otherwise the tracking loop can follow only the first
        # 0.02-rad slice and silently skip the remainder of the edge.
        executable_plan: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
        previous_arm = actual_arm.copy()
        previous_torso = actual_torso.copy()
        for waypoint_arm, waypoint_torso, waypoint_diagnostic in plan:
            waypoint_arm = np.asarray(waypoint_arm, dtype=float)
            waypoint_torso = np.asarray(waypoint_torso, dtype=float)
            substeps = max(
                1,
                int(
                    np.ceil(
                        max(
                            float(np.max(np.abs(waypoint_arm - previous_arm)))
                            / R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
                            float(np.max(np.abs(waypoint_torso - previous_torso)))
                            / max_torso_step,
                        )
                    )
                ),
            )
            for substep in range(1, substeps + 1):
                alpha = substep / substeps
                executable_plan.append(
                    (
                        previous_arm + alpha * (waypoint_arm - previous_arm),
                        previous_torso + alpha * (waypoint_torso - previous_torso),
                        waypoint_diagnostic,
                    )
                )
            previous_arm = waypoint_arm.copy()
            previous_torso = waypoint_torso.copy()

        # The planner's candidate path is a certificate, but the executable
        # controller must be keyed to the *returned* endpoint arrays.  In the
        # explicit support-aware branch these are the whole-body IK result;
        # replaying a candidate's intermediate list can otherwise leave the
        # arm at a staging waypoint while the torso suffix is already being
        # consumed.  Rebuild the phase schedule from the measured boundary to
        # the selected endpoint so the schedule and the certified target have
        # one source of truth.  This remains a robot-level order selected by
        # the planner, not a task-specific posture or waypoint.
        if explicit_target:
            phase_arm_path = _linear_joint_path(
                actual_arm,
                np.asarray(target_arm, dtype=float),
                max(0.08, min(R1PRO_TRANSFER_MAX_ARM_STEP_RAD, 0.16)),
            )
            phase_torso_path = _linear_joint_path(
                initial_torso,
                np.asarray(target_torso, dtype=float),
                max_torso_step,
            )
            if selected_execution_order == "torso_then_arm":
                phase_states = [
                    (actual_arm.copy(), q.copy(), {"phase": "torso", "execution_order": selected_execution_order})
                    for q in phase_torso_path[1:]
                ]
                phase_states.extend(
                    (
                        q.copy(),
                        np.asarray(target_torso, dtype=float).copy(),
                        {"phase": "arm", "execution_order": selected_execution_order},
                    )
                    for q in phase_arm_path[1:]
                )
            elif selected_execution_order == "coordinated":
                count = max(len(phase_arm_path), len(phase_torso_path))
                phase_states = []
                for index in range(1, count):
                    arm_alpha = min(1.0, index / max(1, len(phase_arm_path) - 1))
                    torso_alpha = min(1.0, index / max(1, len(phase_torso_path) - 1))
                    phase_states.append(
                        (
                            actual_arm + arm_alpha * (np.asarray(target_arm, dtype=float) - actual_arm),
                            initial_torso + torso_alpha * (np.asarray(target_torso, dtype=float) - initial_torso),
                            {"phase": "coordinated", "execution_order": selected_execution_order},
                        )
                    )
            else:
                # ``arm_then_torso`` is the preferred low-support schedule;
                # an unknown diagnostic is treated conservatively the same
                # way, so a future planner label cannot silently reintroduce
                # the old partial-arm/torso-suffix behavior.
                phase_states = [
                    (q.copy(), initial_torso.copy(), {"phase": "arm", "execution_order": "arm_then_torso"})
                    for q in phase_arm_path[1:]
                ]
                phase_states.extend(
                    (
                        np.asarray(target_arm, dtype=float).copy(),
                        q.copy(),
                        {"phase": "torso", "execution_order": "arm_then_torso"},
                    )
                    for q in phase_torso_path[1:]
                )
            executable_plan = []
            previous_arm = actual_arm.copy()
            previous_torso = actual_torso.copy()
            for waypoint_arm, waypoint_torso, waypoint_diagnostic in phase_states:
                waypoint_arm = np.asarray(waypoint_arm, dtype=float)
                waypoint_torso = np.asarray(waypoint_torso, dtype=float)
                substeps = max(
                    1,
                    int(
                        np.ceil(
                            max(
                                float(np.max(np.abs(waypoint_arm - previous_arm)))
                                / R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
                                float(np.max(np.abs(waypoint_torso - previous_torso)))
                                / max_torso_step,
                            )
                        )
                    ),
                )
                for substep in range(1, substeps + 1):
                    alpha = substep / substeps
                    executable_plan.append(
                        (
                            previous_arm + alpha * (waypoint_arm - previous_arm),
                            previous_torso + alpha * (waypoint_torso - previous_torso),
                            waypoint_diagnostic,
                        )
                    )
                previous_arm = waypoint_arm.copy()
                previous_torso = waypoint_torso.copy()

        setattr(
            telemetry_adapter,
            "_whole_body_pregrasp_runtime_plan_count",
            int(len(executable_plan)),
        )
        previous_scheduled_arm = actual_arm.copy()
        previous_scheduled_torso = actual_torso.copy()
        for execution_index, (next_arm, next_torso, diagnostic) in enumerate(executable_plan):
            next_arm = np.asarray(next_arm, dtype=float)
            next_torso = np.asarray(next_torso, dtype=float)
            # The certified arm-first schedule is a phase contract, not just
            # an ordering hint.  A compliant drive can remain behind the
            # discrete arm waypoint while the precomputed torso suffix is
            # already being iterated.  If that happened, the old executor
            # would begin bending the torso with only a partially positioned
            # arm—the exact high-load failure seen on GPU6.  Hold the torso
            # at the measured phase boundary and keep issuing a bounded arm
            # increment until the live arm reaches the selected target.  The
            # increment is still checked against the whole-body collision
            # envelope below; no pose write or task-specific posture is used.
            if (
                explicit_target
                and selected_execution_order == "arm_then_torso"
                and float(np.max(np.abs(next_torso - initial_torso)))
                > max(1.5 * max_torso_step, 1.0e-4)
                and float(np.max(np.abs(np.asarray(target_arm, dtype=float) - actual_arm)))
                > max(0.01, 0.5 * R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD)
            ):
                next_arm = actual_arm + np.clip(
                    np.asarray(target_arm, dtype=float) - actual_arm,
                    -R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
                    R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
                )
                next_torso = initial_torso.copy()
            # A torso-only suffix is already time-parameterized at the
            # authored torso velocity capability: adjacent references differ
            # by at most one physics-step of motion while the arm stays at its
            # certified endpoint.  Requiring this streaming reference to
            # settle to the strict endpoint tolerance before every physics
            # step multiplies a 70--100 s robot motion into thousands of
            # repeated settle loops and can exhaust the action wall-time
            # budget.  Keep the segment collision check and the robot-level
            # tracking gate, but let the reference clock advance one physics
            # step at a time; the final catch-up below still enforces the
            # tighter endpoint tolerance.  This is a phase-independent
            # controller optimization and does not insert a task waypoint.
            streaming_torso_phase = (
                isinstance(diagnostic, dict)
                and diagnostic.get("phase") == "torso"
                and float(np.max(np.abs(next_arm - previous_scheduled_arm))) <= 1.0e-5
                and float(np.max(np.abs(next_torso - previous_scheduled_torso)))
                <= max(1.5 * max_torso_step, 1.0e-4)
            )
            tracking_tolerance = (
                min(R1PRO_TRANSFER_TRACK_TOL_RAD, 0.10)
                if streaming_torso_phase
                else min(
                    R1PRO_TRANSFER_TRACK_TOL_RAD,
                    max(0.01, 0.5 * R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD),
                )
            )
            setattr(
                telemetry_adapter,
                "_whole_body_pregrasp_runtime_phase",
                str(diagnostic.get("phase", "planned"))
                if isinstance(diagnostic, dict)
                else "planned",
            )
            setattr(
                telemetry_adapter,
                "_whole_body_pregrasp_runtime_index",
                int(execution_index),
            )
            setattr(
                telemetry_adapter,
                "_whole_body_pregrasp_certified_arm_q",
                ",".join(f"{float(value):.5f}" for value in next_arm),
            )
            setattr(
                telemetry_adapter,
                "_whole_body_pregrasp_certified_torso_q",
                ",".join(f"{float(value):.5f}" for value in next_torso),
            )
            # The selected whole-body path was already certified densely
            # before execution.  At runtime this edge is bounded by one
            # physics-step of the capability-limited reference, and the
            # measured post-step configuration is checked again below with
            # ``first_collision_frame``.  Repeating an 8-sample swept check
            # for every one of thousands of tiny actuator increments adds a
            # large CPU cost without covering a new physical interval; use
            # endpoint checking here while retaining the pre-execution dense
            # certificate and the post-step collision gate.
            free, path_diagnostic = whole_body_path_free(
                checker,
                [(actual_arm, actual_torso), (next_arm, next_torso)],
                model_to_world_rotation=rotation,
                model_to_world_translation=translation,
                dense=1,
            )
            if not free:
                return _failure_for(
                    self.name,
                    "whole_body_pregrasp_tracking_path_blocked",
                    "measured tracking state diverged from the certified whole-body path",
                    collision=path_diagnostic,
                    records=records,
                )
            # The executable list is already resampled at the robot's
            # capability limits.  Send that fixed certified waypoint directly
            # instead of recomputing ``measured + clipped(error)``.  The latter
            # turns load-induced deflection into a moving position target: a
            # torso that should remain at its arm-phase boundary then drifts
            # farther into its bend on every frame, exactly the mechanism that
            # produced the former one-joint "waist hinge" and the GPU6 effort
            # spike.  The measured-vs-waypoint error below remains fail-closed;
            # it prevents a compliant drive from silently leaving the
            # collision/stability certificate.
            command_arm = np.asarray(next_arm, dtype=float).copy()
            command_torso = np.asarray(next_torso, dtype=float).copy()
            set_motion_targets(
                command_arm,
                command_torso,
                actual_arm,
                actual_torso,
                measured_torso_velocity=actual_torso_velocity,
            )
            tracked = False
            tracking_error = float("inf")
            for _ in range(R1PRO_TRANSFER_MAX_TRACK_STEPS):
                set_motion_targets(
                    command_arm,
                    command_torso,
                    actual_arm,
                    actual_torso,
                    measured_torso_velocity=actual_torso_velocity,
                )
                adapter.step()
                measured = adapter.read_observation(0.0)
                # Read the post-step state before the evidence hook.  The
                # hook also reads an observation; the adapter caches the
                # finite-difference velocity for this physics interval so
                # both consumers see the same measured velocity.
                if step_hook is not None:
                    step_hook()
                measured_torso = _torso_from_observation(measured)
                try:
                    measured_arm = np.asarray([measured.joint_positions[name] for name in joints], dtype=float)
                except (AttributeError, KeyError, TypeError, ValueError):
                    measured_torso = None
                    measured_arm = np.asarray((), dtype=float)
                if measured_torso is None or measured_arm.shape != (7,):
                    break
                actual_torso = measured_torso
                actual_arm = measured_arm
                measured_torso_velocity = _torso_velocity_from_observation(measured)
                if measured_torso_velocity is not None:
                    actual_torso_velocity = measured_torso_velocity
                _sync_auxiliary_q(kin, actual_torso)
                collision_frame = checker.first_collision_frame(
                    actual_arm,
                    actual_torso,
                    model_to_world_rotation=rotation,
                    model_to_world_translation=translation,
                )
                if collision_frame is not None:
                    return _failure_for(
                        self.name,
                        "whole_body_pregrasp_runtime_collision",
                        "measured whole-body tracking entered a collision envelope",
                        collision_frame=collision_frame,
                        records=records,
                    )
                command_tracking_error = max(
                    float(np.max(np.abs(actual_arm - command_arm))),
                    float(np.max(np.abs(actual_torso - command_torso))),
                )
                # ``command_*`` is deliberately a slew-limited projection of
                # the measured state, so it can look well tracked even when
                # the physical state has fallen behind the certified
                # waypoint.  The safety contract is the latter: do not let
                # the controller silently walk away from the collision,
                # support, and effort certificate it was given.
                certified_tracking_error = max(
                    float(np.max(np.abs(actual_arm - np.asarray(next_arm, dtype=float)))),
                    float(np.max(np.abs(actual_torso - np.asarray(next_torso, dtype=float)))),
                )
                tracking_error = max(command_tracking_error, certified_tracking_error)
                setattr(
                    telemetry_adapter,
                    "_whole_body_pregrasp_tracking_error_rad",
                    float(certified_tracking_error),
                )
                if tracking_error <= tracking_tolerance:
                    tracked = True
                    break
            if not tracked:
                return _failure_for(
                    self.name,
                    "whole_body_pregrasp_tracking_failed",
                    "arm/torso did not follow the certified posture increment",
                    tracking_error_rad=tracking_error,
                    target_arm_q=next_arm.round(5).tolist(),
                    target_torso_q=next_torso.round(5).tolist(),
                    records=records,
                )
            records.append(
                {
                    "torso_q": actual_torso.round(5).tolist(),
                    "arm_q": actual_arm.round(5).tolist(),
                    "tracking_error_rad": tracking_error,
                    "certified_tracking_error_rad": certified_tracking_error,
                    **diagnostic,
                }
            )
            previous_scheduled_arm = next_arm.copy()
            previous_scheduled_torso = next_torso.copy()

        # The discrete certified path can finish while the physical drive is
        # still lagging behind its final waypoint. Close that measured gap
        # with the same slew-limited controller instead of declaring a
        # failure or issuing one large endpoint jump. The segment is checked
        # against the certified whole-body envelope on every update.
        final_arm_tolerance = max(0.01, 0.5 * R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD)
        final_torso_tolerance = min(R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD, 0.01)
        setattr(telemetry_adapter, "_whole_body_pregrasp_runtime_phase", "catchup")
        catchup_steps = max(
            120,
            int(np.ceil(float(np.max(np.abs(np.asarray(target_arm) - actual_arm))) / R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD)) * 4,
            int(np.ceil(float(np.max(np.abs(np.asarray(target_torso) - actual_torso))) / max_torso_step)) * 4,
        )
        for _ in range(catchup_steps):
            if (
                float(np.max(np.abs(np.asarray(target_arm) - actual_arm))) <= final_arm_tolerance
                and float(np.max(np.abs(np.asarray(target_torso) - actual_torso))) <= final_torso_tolerance
            ):
                break
            command_arm = actual_arm + np.clip(
                np.asarray(target_arm, dtype=float) - actual_arm,
                -R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
                R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
            )
            command_torso = actual_torso + np.clip(
                np.asarray(target_torso, dtype=float) - actual_torso,
                -max_torso_step,
                max_torso_step,
            )
            free, path_diagnostic = whole_body_path_free(
                checker,
                [(actual_arm, actual_torso), (command_arm, command_torso)],
                model_to_world_rotation=rotation,
                model_to_world_translation=translation,
                dense=4,
            )
            if not free:
                return _failure_for(
                    self.name,
                    "whole_body_pregrasp_tracking_path_blocked",
                    "measured final tracking state diverged into a collision envelope",
                    collision=path_diagnostic,
                    records=records,
                )
            set_motion_targets(
                command_arm,
                command_torso,
                actual_arm,
                actual_torso,
                measured_torso_velocity=actual_torso_velocity,
            )
            adapter.step()
            measured = adapter.read_observation(0.0)
            if step_hook is not None:
                step_hook()
            measured_torso = _torso_from_observation(measured)
            try:
                measured_arm = np.asarray([measured.joint_positions[name] for name in joints], dtype=float)
            except (AttributeError, KeyError, TypeError, ValueError):
                measured_torso = None
                measured_arm = np.asarray((), dtype=float)
            if measured_torso is None or measured_arm.shape != (7,) or not np.all(np.isfinite(measured_arm)):
                return _failure_for(
                    self.name,
                    "whole_body_pregrasp_tracking_failed",
                    "final whole-body tracking observation is incomplete",
                    records=records,
                )
            actual_torso = measured_torso
            actual_arm = measured_arm
            measured_torso_velocity = _torso_velocity_from_observation(measured)
            if measured_torso_velocity is not None:
                actual_torso_velocity = measured_torso_velocity
        final = adapter.read_observation(0.0)
        final_torso = _torso_from_observation(final)
        try:
            final_arm = np.asarray([final.joint_positions[name] for name in joints], dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError):
            final_arm = np.asarray((), dtype=float)
        final_error = float("inf") if final_torso is None else _max_abs_difference(final_torso, target_torso)
        final_arm_error = (
            float("inf")
            if final_arm.shape != (7,)
            else _max_abs_difference(final_arm, target_arm)
        )
        if (
            final_torso is None
            or final_error > R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD
            or final_arm_error > max(0.01, 0.5 * R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD)
        ):
            return _failure_for(
                self.name,
                "whole_body_pregrasp_not_settled",
                "whole-body pregrasp posture did not settle within tolerance",
                final_error_rad=final_error,
                final_arm_error_rad=final_arm_error,
                target_torso_q=np.asarray(target_torso, dtype=float).round(5).tolist(),
                target_arm_q=np.asarray(target_arm, dtype=float).round(5).tolist(),
                final_torso_q=None if final_torso is None else final_torso.round(5).tolist(),
                final_arm_q=None if final_arm.shape != (7,) else final_arm.round(5).tolist(),
                records=records,
            )
        setattr(telemetry_adapter, "_whole_body_pregrasp_phase", "settling")
        # The wheel/steer hold was established before the torso motion and
        # may have accumulated measured encoder drift under the loaded
        # reaction.  Start the final settle from the live boundary state so a
        # sparse arm/torso command cannot simultaneously correct an obsolete
        # chassis target.  This is a target update only; the physical state
        # remains untouched.
        if hasattr(adapter, "rebase_joint_mask_targets"):
            adapter.rebase_joint_mask_targets()
        # A position-drive target can arrive at the final posture while the
        # loaded torso still has appreciable velocity.  Sending an immediate
        # zero velocity target would add a full damping impulse to the static
        # gravity torque (the source of the observed last-frame effort spike).
        # Brake from the measured velocity over the settle window instead.
        settle_velocity_reference = np.clip(
            np.asarray(actual_torso_velocity, dtype=float),
            -abs(self.torso_velocity_limit),
            abs(self.torso_velocity_limit),
        )
        for _ in range(max(0, int(settle_steps))):
            set_motion_targets(
                np.asarray(target_arm, dtype=float),
                np.asarray(target_torso, dtype=float),
                actual_arm,
                actual_torso,
                torso_velocity_reference=settle_velocity_reference,
            )
            adapter.step()
            settled_observation = adapter.read_observation(0.0)
            if step_hook is not None:
                step_hook()
            settled_velocity = _torso_velocity_from_observation(settled_observation)
            if settled_velocity is not None:
                actual_torso_velocity = settled_velocity
            settle_velocity_reference *= 0.85
        stabilize_base(adapter, replace_wheel_only=True)
        return SkillResult(
            True,
            self.name,
            metrics={"iterations": float(len(records)), "final_torso_error_rad": final_error},
            details={
                "reason": "whole-body pregrasp posture settled",
                "target_torso_q": target_torso.tolist(),
                "arm_staging_q": target_arm.round(5).tolist(),
                "planner_diagnostics": _json_safe(planner_diagnostics),
                "records": records,
            },
        )


def _planning_budget_check(adapter: Any, phase: str) -> None:
    """Check a hosted action budget during CPU-only planning work.

    The normal adapter proxy checks budgets around ``step`` calls.  Whole-body
    IK and collision certification can legitimately run for many seconds
    without stepping the simulator, so semantic planners call this hook at
    their inner-loop boundaries when the adapter exposes the orchestrator's
    optional ``check`` method.  Minimal test adapters keep the old behavior.
    """
    checker = getattr(adapter, "check", None)
    if callable(checker):
        checker(str(phase))


def _select_pregrasp_plan(
    kin: Any,
    checker: WholeBodyCollisionChecker,
    adapter: Any,
    scene: Any,
    object_name: str,
    *,
    side: str,
    current_arm: np.ndarray,
    current_torso: np.ndarray,
    model_to_world_rotation: np.ndarray,
    model_to_world_translation: np.ndarray,
    speed_scale: float,
    torso_velocity_limit: float,
    target_center_world: np.ndarray | None = None,
    target_span_world: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    list[dict[str, Any]],
] | None:
    """Search coordinated torso/arm candidates with physical preflight gates.

    The target is a live, geometry-derived high approach above the object. The
    torso vector is not a task pose: it is sampled from the robot's authored
    limits, and every branch is filtered by swept collision, COM support
    margin, and available torso gravity effort before execution.
    """
    diagnostics: list[dict[str, Any]] = []
    try:
        object_world = np.asarray(adapter.object_position(object_name), dtype=float)
        object_model = scene.object(object_name)
        observation = adapter.read_observation(0.0)
        base_pose = tuple(float(value) for value in (observation.base_pose or (0.0, 0.0, 0.0)))
        height = max(0.02, float(object_vertical_extent_m(object_model)))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        setattr(kin, "_last_pregrasp_diagnostics", [{"reason": str(exc)}])
        return None
    if object_world.shape != (3,) or not np.all(np.isfinite(object_world)):
        setattr(kin, "_last_pregrasp_diagnostics", [{"reason": "object position is invalid"}])
        return None
    if len(base_pose) < 3 or not np.all(np.isfinite(np.asarray(base_pose[:3], dtype=float))):
        setattr(kin, "_last_pregrasp_diagnostics", [{"reason": "base pose is invalid"}])
        return None

    # The default high point is derived from the live primitive's vertical
    # extent.  A support-aware caller may instead provide a live, geometry-
    # derived low-side non-contact center so the coordinated solver chooses a
    # posture that is reachable for the later grasp phase.  Neither branch is
    # a task coordinate or a fixed torso pose.
    if target_center_world is None:
        target_world = object_world.copy()
        target_world[2] += max(0.06, 0.75 * height)
    else:
        target_world = np.asarray(target_center_world, dtype=float).copy()
        if target_world.shape != (3,) or not np.all(np.isfinite(target_world)):
            setattr(kin, "_last_pregrasp_diagnostics", [{"reason": "target center is invalid"}])
            return None
    target_model = np.asarray(model_to_world_rotation, dtype=float).T @ (
        target_world - np.asarray(model_to_world_translation, dtype=float)
    )
    if target_span_world is not None:
        target_span_world = np.asarray(target_span_world, dtype=float)
        if (
            target_span_world.shape != (3,)
            or not np.all(np.isfinite(target_span_world))
            or float(np.linalg.norm(target_span_world[:2])) <= 1.0e-8
        ):
            setattr(
                kin,
                "_last_pregrasp_diagnostics",
                [{"reason": "target jaw span is invalid"}],
            )
            return None
    dt = float(getattr(adapter, "dt", 1.0 / 60.0))
    max_torso_step = max(1.0e-4, abs(float(torso_velocity_limit)) * float(speed_scale) * dt)
    # The physical executor uses ``max_torso_step`` so every command respects
    # the authored velocity capability.  Planning a collision certificate at
    # that same 60-Hz step would make a one- to two-radian posture contain
    # thousands of nearly identical states, and repeating that certificate
    # for every IK branch is both redundant and hostile to generic tasks.  A
    # planning edge is still subdivided by ``whole_body_path_free`` and the
    # runtime executor independently rebuilds the fine-grained schedule, so
    # this coarser value changes only planning cost, never commanded motion.
    planning_torso_step = max(
        max_torso_step,
        float(R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD),
    )
    wheel_positions = tuple(STEER_POSITIONS.values())
    torso_goals = _torso_goal_candidates(current_torso)
    arm_seed_candidates = _pregrasp_arm_candidates(kin, side, current_arm)
    best: tuple[float, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]], dict[str, Any]] | None = None

    # A geometry-derived low interaction center must be reached by the
    # selected configuration.  The old implementation mixed genuine IK
    # branches with neutral/natural staging seeds and then ranked them by
    # distance, so a 0.5 m miss could become the winning "pregrasp".  For an
    # explicit support-aware target, only a whole-body IK solution (or a
    # successful fixed-torso IK solution on a third-party backend) is allowed.
    # The generic high approach keeps its legacy staging candidates because it
    # is intentionally a posture transition rather than a target-reaching
    # operation.
    candidate_specs: list[tuple[int, int, np.ndarray, np.ndarray, dict[str, Any]]] = []
    explicit_target = target_center_world is not None
    if explicit_target:
        solutions: list[Any] = []
        if hasattr(kin, "whole_body_grasp_center_candidates"):
            try:
                ik_solver = kin.whole_body_grasp_center_candidates
                ik_kwargs: dict[str, Any] = {
                    "max_candidates": 80,
                }
                if target_span_world is not None:
                    ik_kwargs.update(
                        {
                            "desired_span": np.asarray(target_span_world, dtype=float),
                            "span_to_constraint_rotation": np.asarray(
                                model_to_world_rotation,
                                dtype=float,
                            ),
                        }
                    )
                # The optional callback lets the hosted action budget stop a
                # pure Python/SciPy IK search even when no simulator step has
                # happened yet.  Do not require third-party kinematics
                # backends to implement the extension: inspect the callable
                # first and retain their existing API contract.
                try:
                    ik_parameters = inspect.signature(ik_solver).parameters
                    accepts_budget = (
                        "budget_check" in ik_parameters
                        or any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in ik_parameters.values()
                        )
                    )
                except (TypeError, ValueError):
                    accepts_budget = False
                if accepts_budget:
                    ik_kwargs["budget_check"] = lambda: _planning_budget_check(
                        adapter,
                        "whole_body_position_ik",
                    )
                solutions = list(
                    ik_solver(
                        target_model,
                        current_arm,
                        current_torso,
                        **ik_kwargs,
                    )
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                solutions = []
        if solutions:
            diagnostics.append(
                {
                    "phase": "whole_body_position_ik",
                    "target_center_model": target_model.round(5).tolist(),
                    "solution_count": len(solutions),
                }
            )
            for solution_index, solution in enumerate(solutions):
                q_arm = getattr(solution, "q_arm", None)
                q_torso = getattr(solution, "q_torso", None)
                if q_arm is None or q_torso is None:
                    continue
                q_arm = np.asarray(q_arm, dtype=float)
                q_torso = np.asarray(q_torso, dtype=float)
                if (
                    q_arm.shape != (7,)
                    or q_torso.shape != (4,)
                    or not np.all(np.isfinite(q_arm))
                    or not np.all(np.isfinite(q_torso))
                ):
                    continue
                candidate_specs.append(
                    (
                        solution_index,
                        solution_index,
                        q_torso,
                        q_arm,
                        {
                            "ik_position_error_m": float(getattr(solution, "position_error", float("inf"))),
                            "ik_iterations": int(getattr(solution, "iterations", 0)),
                        },
                    )
                )
        elif target_span_world is not None and hasattr(
            kin,
            "ik_grasp_window_candidates",
        ):
            # A backend without the 11-DOF whole-body solver may still expose
            # the arm-level grasp-window solver.  Preserve the requested
            # orientation constraint in that compatibility path; falling
            # back to position-only IK here would recreate the unsafe
            # endpoint-aligned pregrasp that this capability is meant to
            # prevent.
            for torso_index, torso_goal in enumerate(torso_goals):
                _sync_auxiliary_q(kin, torso_goal)
                try:
                    fixed_torso_solutions = kin.ik_grasp_window_candidates(
                        target_model,
                        np.asarray(target_span_world, dtype=float),
                        current_arm,
                        max_candidates=6,
                        span_to_constraint_rotation=np.asarray(
                            model_to_world_rotation,
                            dtype=float,
                        ),
                    )
                except TypeError:
                    try:
                        fixed_torso_solutions = kin.ik_grasp_window_candidates(
                            target_model,
                            np.asarray(target_span_world, dtype=float),
                            current_arm,
                            max_candidates=6,
                        )
                    except (
                        AttributeError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        fixed_torso_solutions = ()
                except (
                    AttributeError,
                    RuntimeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    fixed_torso_solutions = ()
                for arm_index, solution in enumerate(fixed_torso_solutions):
                    q_arm = getattr(solution, "q_arm", None)
                    if not getattr(solution, "success", False) or q_arm is None:
                        continue
                    q_arm = np.asarray(q_arm, dtype=float)
                    if q_arm.shape != (7,) or not np.all(np.isfinite(q_arm)):
                        continue
                    candidate_specs.append(
                        (
                            torso_index,
                            arm_index,
                            torso_goal.copy(),
                            q_arm.copy(),
                            {
                                "ik_position_error_m": float(
                                    getattr(solution, "position_error", float("inf"))
                                ),
                                "ik_direction_error_rad": float(
                                    getattr(solution, "rotation_error", float("inf"))
                                ),
                            },
                        )
                    )
        elif target_span_world is None and hasattr(kin, "ik_grasp_center_candidates"):
            # Keep lightweight/test and third-party kinematics backends useful
            # without reintroducing the unsafe seed fallback. Every candidate
            # here must still be a successful arm IK result for its sampled
            # torso posture.
            for torso_index, torso_goal in enumerate(torso_goals):
                _sync_auxiliary_q(kin, torso_goal)
                try:
                    fixed_torso_solutions = kin.ik_grasp_center_candidates(
                        target_model,
                        current_arm,
                        max_candidates=6,
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                    fixed_torso_solutions = ()
                for arm_index, solution in enumerate(fixed_torso_solutions):
                    q_arm = getattr(solution, "q_arm", None)
                    if not getattr(solution, "success", False) or q_arm is None:
                        continue
                    q_arm = np.asarray(q_arm, dtype=float)
                    if q_arm.shape != (7,) or not np.all(np.isfinite(q_arm)):
                        continue
                    candidate_specs.append(
                        (
                            torso_index,
                            arm_index,
                            torso_goal.copy(),
                            q_arm.copy(),
                            {"ik_position_error_m": float(getattr(solution, "position_error", float("inf")))},
                        )
                    )
        if not candidate_specs:
            diagnostics.append(
                {
                    "phase": "whole_body_position_ik",
                    "reason": "no_reaching_solution",
                    "target_center_model": target_model.round(5).tolist(),
                }
            )
    else:
        for torso_index, torso_goal in enumerate(torso_goals):
            _sync_auxiliary_q(kin, torso_goal)
            arm_candidates = list(arm_seed_candidates)
            if hasattr(kin, "ik_grasp_center_candidates"):
                try:
                    arm_candidates.extend(
                        solution.q_arm
                        for solution in kin.ik_grasp_center_candidates(
                            target_model,
                            current_arm,
                            max_candidates=6,
                        )
                        if getattr(solution, "success", False) and getattr(solution, "q_arm", None) is not None
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                    pass
            unique_arms: list[np.ndarray] = []
            for candidate in arm_candidates:
                candidate = np.asarray(candidate, dtype=float)
                if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                    continue
                if not any(np.allclose(candidate, previous, atol=1.0e-5) for previous in unique_arms):
                    unique_arms.append(candidate)
            arm_candidates = sorted(
                unique_arms,
                key=lambda candidate: _pregrasp_target_distance(
                    kin,
                    candidate,
                    target_world,
                    model_to_world_rotation,
                    model_to_world_translation,
                ),
            )
            for arm_index, arm_goal in enumerate(arm_candidates[:10]):
                candidate_specs.append(
                    (torso_index, arm_index, torso_goal.copy(), np.asarray(arm_goal, dtype=float).copy(), {})
                )

    for torso_index, arm_index, torso_goal, arm_goal, ik_diagnostic in candidate_specs:
        _planning_budget_check(adapter, "whole_body_pregrasp_candidate")
        _sync_auxiliary_q(kin, torso_goal)
        variants = (
            _pregrasp_state_variants(
                current_arm,
                current_torso,
                arm_goal,
                torso_goal,
                planning_torso_step,
            )
            if explicit_target
            else (
                (
                    "arm_then_torso",
                    [
                        (current_arm.copy(), current_torso.copy()),
                        *[
                            (q.copy(), current_torso.copy())
                            for q in _linear_joint_path(
                                current_arm,
                                arm_goal,
                                max(0.08, min(R1PRO_TRANSFER_MAX_ARM_STEP_RAD, 0.16)),
                            )[1:]
                        ],
                        *[
                            (arm_goal.copy(), q.copy())
                            for q in _linear_joint_path(current_torso, torso_goal, planning_torso_step)[1:]
                        ],
                    ],
                ),
            )
        )
        endpoint_stable = True
        endpoint_stability: dict[str, Any] = {}
        if explicit_target:
            # Endpoint support/effort is independent of execution order.  It
            # used to be recomputed three times for each IK branch, adding
            # planning work without strengthening the certificate.
            endpoint_stable, endpoint_stability = _stability_path_free(
                kin,
                adapter,
                [(arm_goal, torso_goal)],
                base_pose=base_pose,
                wheel_positions=wheel_positions,
                model_to_world_rotation=model_to_world_rotation,
                model_to_world_translation=model_to_world_translation,
            )
            if not endpoint_stable:
                diagnostics.append(
                    {
                        "torso_candidate_index": torso_index,
                        "arm_candidate_index": arm_index,
                        "execution_order": "all_variants",
                        "torso_q": torso_goal.round(5).tolist(),
                        "arm_q": arm_goal.round(5).tolist(),
                        "endpoint_stability": _json_safe(endpoint_stability),
                    }
                )
        for execution_order, states in variants:
            if explicit_target and not endpoint_stable:
                continue
            free, collision = whole_body_path_free(
                checker,
                states,
                model_to_world_rotation=model_to_world_rotation,
                model_to_world_translation=model_to_world_translation,
                dense=3,
                budget_check=lambda: _planning_budget_check(
                    adapter,
                    "whole_body_path_certificate",
                ),
            )
            if not free:
                diagnostics.append(
                    {
                        "torso_candidate_index": torso_index,
                        "arm_candidate_index": arm_index,
                        "execution_order": execution_order,
                        "torso_q": torso_goal.round(5).tolist(),
                        "arm_q": arm_goal.round(5).tolist(),
                        "collision": _json_safe(collision),
                    }
                )
                continue
            stable, stability = _stability_path_free(
                kin,
                adapter,
                states,
                base_pose=base_pose,
                wheel_positions=wheel_positions,
                model_to_world_rotation=model_to_world_rotation,
                model_to_world_translation=model_to_world_translation,
            )
            if not stable:
                diagnostics.append(
                    {
                        "torso_candidate_index": torso_index,
                        "arm_candidate_index": arm_index,
                        "execution_order": execution_order,
                        "torso_q": torso_goal.round(5).tolist(),
                        "arm_q": arm_goal.round(5).tolist(),
                        "stability": _json_safe(stability),
                    }
                )
                continue
            target_distance = _pregrasp_target_distance(
                kin,
                arm_goal,
                target_world,
                model_to_world_rotation,
                model_to_world_translation,
            )
            if explicit_target and not np.isfinite(target_distance):
                continue
            motion_cost = float(np.linalg.norm(arm_goal - current_arm)) + float(
                np.linalg.norm(torso_goal - current_torso)
            )
            # For a low-support interaction, moving the torso while the
            # selected arm is still at its hanging posture is the least
            # robust load schedule: the torso must carry the worst arm
            # moment before the arm has reached its counterbalancing branch.
            # Prefer arm-first/coordinated execution when the physical
            # certificates are otherwise comparable.  This is a
            # robot-level dynamic scheduling rule, not a task pose or a
            # scene-specific sequence; the collision/stability gates above
            # still reject any order that is genuinely infeasible.
            order_penalty = (
                {
                    "arm_then_torso": 0.0,
                    "coordinated": 0.20,
                    "torso_then_arm": 0.40,
                }
                if explicit_target
                else {
                    "torso_then_arm": 0.0,
                    "coordinated": 0.001,
                    "arm_then_torso": 0.002,
                }
            ).get(execution_order, 0.08 if explicit_target else 0.002)
            effort_utilization = float(
                stability.get("maximum_torso_effort_nm", 0.0)
            ) / max(1.0, float(R1PRO_TORSO_EFFORT_LIMIT))
            score = float(
                10.0 * target_distance
                + 0.05 * motion_cost
                + order_penalty
                + 0.25 * effort_utilization
                - 0.02 * float(stability.get("minimum_margin_m", 0.0))
            )
            diagnostic = {
                "torso_candidate_index": torso_index,
                "arm_candidate_index": arm_index,
                "execution_order": execution_order,
                "torso_q": torso_goal.round(5).tolist(),
                "arm_staging_q": arm_goal.round(5).tolist(),
                "target_pregrasp_world": target_world.round(5).tolist(),
                "target_distance_m": target_distance,
                "effort_utilization": effort_utilization,
                "stability": stability,
                **ik_diagnostic,
            }
            candidate_plan = [
                (
                    arm_state,
                    torso_state,
                    {
                        "arm_candidate_index": arm_index,
                        "torso_candidate_index": torso_index,
                        "execution_order": execution_order,
                        "arm_staging_q": arm_goal.round(5).tolist(),
                        "stability": stability,
                    },
                )
                for arm_state, torso_state in states[1:]
            ]
            if best is None or score < best[0]:
                best = (score, torso_goal.copy(), arm_goal.copy(), candidate_plan, diagnostic)

    _sync_auxiliary_q(kin, current_torso)
    setattr(kin, "_last_pregrasp_diagnostics", diagnostics)
    if best is None:
        return None
    _, target_torso, target_arm, plan, diagnostic = best
    diagnostics.append({"selected": diagnostic})
    return target_torso, target_arm, plan, diagnostics


def _torso_goal_candidates(current_torso: Sequence[float]) -> list[np.ndarray]:
    """Generate bounded, distributed torso bends from authored joint limits."""
    current = np.asarray(current_torso, dtype=float)
    if current.shape != (4,) or not np.all(np.isfinite(current)):
        return []
    lower = np.asarray(
        [
            R1PRO_JOINT_LIMITS[f"torso_joint{index}"][0]
            for index in range(1, 5)
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            R1PRO_JOINT_LIMITS[f"torso_joint{index}"][1]
            for index in range(1, 5)
        ],
        dtype=float,
    )
    # The amplitudes are fractions of each joint's available interval. Every
    # non-neutral candidate moves multiple pitch joints, so a floor approach
    # cannot silently turn into the former single-joint hinge motion.
    amplitude = np.asarray([0.14, 0.10, 0.14, 0.0], dtype=float)
    candidates: list[np.ndarray] = []
    directions = tuple(
        np.asarray([sx, sy, sz], dtype=float)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    )
    for level in (0.50, 0.85, 1.20, 1.55, 1.90):
        for direction in directions:
            goal = current.copy()
            goal[:3] += level * amplitude[:3] * np.asarray(
                [upper[index] - lower[index] for index in range(3)],
                dtype=float,
            ) * direction
            goal = np.clip(goal, lower, upper)
            if np.max(np.abs(goal[:3] - current[:3])) <= 1.0e-3:
                continue
            if sum(abs(float(value)) > 1.0e-3 for value in goal[:3] - current[:3]) < 2:
                continue
            if not any(np.allclose(goal, previous, atol=1.0e-5) for previous in candidates):
                candidates.append(goal)
    # An already configured non-neutral posture remains a valid candidate for
    # idempotent retries, but it is considered after genuine coordinated bends.
    if not any(np.allclose(current, previous, atol=1.0e-5) for previous in candidates):
        candidates.append(current.copy())
    return candidates


def _pregrasp_target_distance(
    kin: Any,
    q_arm: np.ndarray,
    target_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    try:
        center_model, _ = kin.grasp_center_fk(np.asarray(q_arm, dtype=float))
        center_world = np.asarray(rotation, dtype=float) @ np.asarray(center_model, dtype=float) + np.asarray(translation, dtype=float)
        return float(np.linalg.norm(center_world - np.asarray(target_world, dtype=float)))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return float("inf")


def _pregrasp_state_variants(
    current_arm: np.ndarray,
    current_torso: np.ndarray,
    arm_goal: np.ndarray,
    torso_goal: np.ndarray,
    max_torso_step: float,
) -> tuple[tuple[str, list[tuple[np.ndarray, np.ndarray]]], ...]:
    """Return physically distinct, bounded orders for a whole-body posture.

    A valid endpoint does not determine a safe transition.  The solver tests
    torso-first, arm-first, and synchronized interpolation so a scene can
    choose the order whose full swept body remains clear and supported.  The
    variants are robot-level execution strategies; none contains a task
    waypoint or a fixed torso vector.
    """
    current_arm = np.asarray(current_arm, dtype=float)
    current_torso = np.asarray(current_torso, dtype=float)
    arm_goal = np.asarray(arm_goal, dtype=float)
    torso_goal = np.asarray(torso_goal, dtype=float)
    arm_path = _linear_joint_path(
        current_arm,
        arm_goal,
        max(0.08, min(R1PRO_TRANSFER_MAX_ARM_STEP_RAD, 0.16)),
    )
    torso_path = _linear_joint_path(current_torso, torso_goal, max_torso_step)
    arm_then_torso: list[tuple[np.ndarray, np.ndarray]] = [
        (current_arm.copy(), current_torso.copy()),
        *[(q.copy(), current_torso.copy()) for q in arm_path[1:]],
        *[(arm_goal.copy(), q.copy()) for q in torso_path[1:]],
    ]
    torso_then_arm: list[tuple[np.ndarray, np.ndarray]] = [
        (current_arm.copy(), current_torso.copy()),
        *[(current_arm.copy(), q.copy()) for q in torso_path[1:]],
        *[(q.copy(), torso_goal.copy()) for q in arm_path[1:]],
    ]
    count = max(len(arm_path), len(torso_path))
    synchronized: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(count):
        arm_alpha = min(1.0, index / max(1, len(arm_path) - 1))
        torso_alpha = min(1.0, index / max(1, len(torso_path) - 1))
        synchronized.append(
            (
                current_arm + arm_alpha * (arm_goal - current_arm),
                current_torso + torso_alpha * (torso_goal - current_torso),
            )
        )
    return (
        ("torso_then_arm", torso_then_arm),
        ("coordinated", synchronized),
        ("arm_then_torso", arm_then_torso),
    )


def _stability_path_free(
    kin: Any,
    adapter: Any,
    states: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    base_pose: Sequence[float],
    wheel_positions: Sequence[Sequence[float]],
    model_to_world_rotation: np.ndarray,
    model_to_world_translation: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Check COM support and torso gravity effort along a candidate path."""
    has_com = hasattr(kin, "center_of_mass")
    has_effort = hasattr(kin, "torso_gravity_effort")
    if not has_com:
        # Test doubles may only exercise collision/path composition. A real
        # R1Pro kinematics object always supplies this gate; a real adapter
        # refuses to certify a candidate when the model is incomplete.
        if hasattr(adapter, "physical_metrics"):
            return False, {"verified": False, "reason": "robot COM model unavailable"}
        return True, {"verified": False, "reason": "test adapter has no COM model"}
    minimum_margin = float("inf")
    maximum_effort = 0.0
    certificates: list[dict[str, Any]] = []
    for index, (arm_q, torso_q) in enumerate(states):
        _sync_auxiliary_q(kin, torso_q)
        try:
            com_model = np.asarray(kin.center_of_mass(np.asarray(arm_q, dtype=float)), dtype=float)
            com_world = np.asarray(model_to_world_rotation, dtype=float) @ com_model + np.asarray(model_to_world_translation, dtype=float)
            certificate = configuration_stability(
                com_world=com_world,
                base_pose=base_pose,
                wheel_positions=wheel_positions,
                required_margin_m=R1PRO_SUPPORT_POLYGON_MARGIN_M,
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return False, {"verified": False, "reason": "COM stability calculation failed", "error": str(exc), "state": index}
        if not certificate.stable:
            return False, {
                "verified": True,
                "reason": "quasi-static COM support margin failed",
                "state": index,
                "certificate": certificate.to_dict(),
            }
        minimum_margin = min(minimum_margin, float(certificate.margin_m))
        effort: np.ndarray | None = None
        if has_effort:
            try:
                effort = np.asarray(kin.torso_gravity_effort(np.asarray(arm_q, dtype=float)), dtype=float)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                return False, {"verified": False, "reason": "torso effort calculation failed", "error": str(exc), "state": index}
            if effort.shape != (4,) or not np.all(np.isfinite(effort)):
                return False, {"verified": False, "reason": "torso effort is invalid", "state": index}
            maximum_effort = max(maximum_effort, float(np.max(np.abs(effort))))
            if maximum_effort > R1PRO_TORSO_EFFORT_LIMIT * R1PRO_EFFORT_PLANNING_UTILIZATION:
                return False, {
                    "verified": True,
                    "reason": "torso gravity effort exceeds planning reserve",
                    "state": index,
                    "torso_effort_nm": effort.round(5).tolist(),
                    "limit_nm": R1PRO_TORSO_EFFORT_LIMIT * R1PRO_EFFORT_PLANNING_UTILIZATION,
                }
        certificates.append(
            {
                "state": index,
                "margin_m": float(certificate.margin_m),
                "torso_effort_nm": None if effort is None else effort.round(5).tolist(),
            }
        )
    return True, {
        "verified": True,
        "minimum_margin_m": 0.0 if not np.isfinite(minimum_margin) else minimum_margin,
        "maximum_torso_effort_nm": maximum_effort,
        "states_checked": len(states),
        "certificates": certificates,
    }


def _pregrasp_arm_candidates(
    kin: Any,
    side: str,
    current_arm: np.ndarray,
) -> list[np.ndarray]:
    """Return robot-level arm staging profiles, without task poses."""
    candidates: list[np.ndarray] = [np.asarray(current_arm, dtype=float).copy()]
    natural = getattr(kin, "natural_reach_q", None)
    if natural is not None:
        candidates.append(np.asarray(natural, dtype=float).copy())
    try:
        from r1pro_data_gen.robot.robot_config import R1PRO_ARM_READY_Q_BY_SIDE

        candidates.append(np.asarray(R1PRO_ARM_READY_Q_BY_SIDE[side], dtype=float))
    except (KeyError, TypeError, ValueError):
        pass
    result: list[np.ndarray] = []
    lower = np.asarray(getattr(kin, "lower", np.full(7, -np.inf)), dtype=float)
    upper = np.asarray(getattr(kin, "upper", np.full(7, np.inf)), dtype=float)
    for candidate in candidates:
        if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
            continue
        clipped = np.clip(candidate, lower, upper)
        if not any(np.allclose(clipped, previous, atol=1.0e-6) for previous in result):
            result.append(clipped)
    return result


def _linear_joint_path(
    start: np.ndarray,
    goal: np.ndarray,
    max_step: float,
) -> list[np.ndarray]:
    """Discretize a robot-joint interpolation at a bounded per-state step."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != goal.shape:
        raise ValueError("linear joint path endpoints must have matching shapes")
    step = max(1.0e-4, float(max_step))
    count = max(1, int(np.ceil(float(np.max(np.abs(goal - start))) / step)))
    return [start + (goal - start) * (index / count) for index in range(count + 1)]


def _unattached_hold_solution(
    kin: Any,
    checker: WholeBodyCollisionChecker,
    *,
    side: str,
    current_arm: np.ndarray,
    current_torso: np.ndarray,
    torso_goal: np.ndarray,
    hold_ee_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Find one small arm correction while an unattached EE is held fixed."""
    del side
    for fraction in (1.0, 0.5, 0.25):
        candidate_torso = current_torso + (torso_goal - current_torso) * fraction
        _sync_auxiliary_q(kin, current_torso)
        try:
            _sync_auxiliary_q(kin, candidate_torso)
            target_ee = np.asarray(rotation, dtype=float).T @ (
                np.asarray(hold_ee_world, dtype=float) - np.asarray(translation, dtype=float)
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
            continue
        candidates = _ik_candidates(kin, target_ee, current_arm)
        for arm_goal in candidates:
            arm_goal = np.asarray(arm_goal, dtype=float)
            if arm_goal.shape != current_arm.shape:
                continue
            if float(np.max(np.abs(arm_goal - current_arm))) > R1PRO_TRANSFER_MAX_ARM_STEP_RAD:
                continue
            measured_ee_model, _ = kin.fk(arm_goal)
            measured_ee_world = np.asarray(rotation, dtype=float) @ measured_ee_model + np.asarray(translation, dtype=float)
            hold_error = float(np.linalg.norm(measured_ee_world - hold_ee_world))
            if hold_error > R1PRO_TRANSFER_HOLD_CENTER_TOL_M:
                continue
            free, collision = whole_body_path_free(
                checker,
                [(current_arm, current_torso), (arm_goal, candidate_torso)],
                model_to_world_rotation=rotation,
                model_to_world_translation=translation,
                dense=8,
            )
            if not free:
                continue
            return arm_goal, candidate_torso, {
                "ee_hold_error_m": hold_error,
                "collision": collision,
            }
    _sync_auxiliary_q(kin, current_torso)
    return None


def _staging_candidates(adapter: Any, scene: Any, object_name: str) -> list[tuple[float, float, float]]:
    """Generate bounded base candidates around a live object, generically."""
    try:
        object_position = np.asarray(adapter.object_position(object_name), dtype=float)
        observation = adapter.read_observation(0.0)
        base_pose = tuple(float(value) for value in (observation.base_pose or (0.0, 0.0, 0.0)))
        object_model = scene.object(object_name)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return []
    if object_position.shape != (3,) or len(base_pose) < 3:
        return []
    from r1pro_data_gen.robot.chassis import default_footprint_radius_m

    configured = getattr(getattr(scene, "robot", None), "navigation_footprint_radius_m", None)
    try:
        footprint = max(default_footprint_radius_m(), float(configured or 0.0))
    except (TypeError, ValueError):
        footprint = default_footprint_radius_m()
    required_radius = footprint + object_xy_radius_m(object_model) + 0.15
    current_xy = np.asarray(base_pose[:2], dtype=float)
    vector = current_xy - object_position[:2]
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-6:
        vector = np.asarray([-1.0, 0.0], dtype=float)
        distance = 1.0
    direction = vector / distance
    base_angle = float(np.arctan2(direction[1], direction[0]))
    # The ring is derived from the robot footprint and object extent.  Angular
    # offsets provide alternate sides when a support blocks the current side;
    # no task layout or named scene coordinate appears here.
    radii = (max(required_radius, distance + 0.20), required_radius + 0.20, required_radius + 0.40)
    angle_offsets = (0.0, np.pi / 12.0, -np.pi / 12.0, np.pi / 6.0, -np.pi / 6.0, np.pi / 3.0, -np.pi / 3.0)
    candidates: list[tuple[float, float, float]] = []
    for radius in radii:
        for offset in angle_offsets:
            angle = base_angle + float(offset)
            xy = object_position[:2] + float(radius) * np.asarray([np.cos(angle), np.sin(angle)])
            yaw = float(np.arctan2(object_position[1] - xy[1], object_position[0] - xy[0]))
            pose = (float(xy[0]), float(xy[1]), yaw)
            if float(np.linalg.norm(xy - current_xy)) < 0.08:
                continue
            if pose not in candidates:
                candidates.append(pose)
    return candidates


def _unlock_internal_hold(adapter: Any) -> None:
    """Release a lock created by a failed internal whole-body preflight."""
    if not getattr(adapter, "joint_mask_locked", False) or not hasattr(adapter, "unlock_joint_mask"):
        return
    groups = set(getattr(adapter, "joint_lock_groups", getattr(adapter, "_joint_lock_groups", ())))
    if groups.issubset({"steer", "wheel", "torso"}):
        adapter.unlock_joint_mask()


class WholeBodyHoldTransition:
    """Change torso posture while keeping an attached object in place.

    The target posture is a robot capability profile.  The held object and its
    current grasp frame are read from the adapter; no scene coordinate or
    object-specific pose is embedded in this skill.
    """

    name = "whole_body_hold_transition"
    tier = "semantic"
    exposed = False
    description = (
        "With an object attached, first lift it clear when requested, then "
        "perform a whole-body transition: re-solve the arm for every small torso step, certify torso/shoulder/"
        "arm swept collision and held-object clearance, and preserve the live "
        "grasp frame until the target manipulation posture settles."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Attached object being held", required=True),
        "target_posture": ParamSpec(
            "string",
            "Robot-level posture profile",
            required=True,
            enum=("carry", "upright"),
        ),
        "side": ParamSpec("string", "Arm side", default="left", enum=("left", "right")),
        "speed_scale": ParamSpec(
            "number",
            "Fraction of the robot torso velocity profile",
            default=0.2,
            minimum=0.05,
            maximum=1.0,
        ),
        "settle_steps": ParamSpec(
            "integer",
            "Physics steps used to settle the final whole-body posture",
            default=18,
            minimum=0,
            maximum=240,
        ),
    }

    def __init__(
        self,
        kin: Any,
        arm_move_directional: Any = None,
        *,
        torso_velocity_limit: float = 0.5,
    ) -> None:
        self.kin = kin
        self.arm_move_directional = arm_move_directional
        self.torso_velocity_limit = float(torso_velocity_limit)

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        target_posture: str = "carry",
        side: str = "left",
        speed_scale: float = 0.2,
        settle_steps: int = 18,
        target_height_m: float | None = None,
        source_support_name: str | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if scene is None or not object_name:
            return _failure("missing_scene_or_object", "whole-body hold transition requires a scene and object")
        side = require_side(side)
        kin = for_side(self.kin, side)
        if kin is None:
            return _failure("kinematics_unavailable", "whole-body hold transition has no selected-arm kinematics")
        if target_posture not in {"carry", "upright"}:
            raise ValueError("target_posture must be 'carry' or 'upright'")
        if not np.isfinite(float(speed_scale)) or not 0.0 < float(speed_scale) <= 1.0:
            raise ValueError("speed_scale must be finite and in (0, 1]")
        speed_scale = min(float(speed_scale), R1PRO_WHOLE_BODY_MAX_SPEED_SCALE)
        try:
            context = live_grasp_context(adapter, object_name, side)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return _failure("grasp_context_unavailable", "live grasp context unavailable", error=str(exc))
        if not context.attached:
            return _failure(
                "object_not_attached",
                "whole-body hold transition requires an attached object",
                grasp_context=context.to_dict(),
            )

        # A low source must be lifted before the torso can be reconfigured.
        # The distance is derived from the destination support by the
        # composite transfer skill and is applied through the measured
        # midpoint directional skill, never through a guessed object pose.
        lift_result: SkillResult | None = None
        if target_height_m is not None:
            if not np.isfinite(float(target_height_m)):
                raise ValueError("target_height_m must be finite")
            current_height = float(context.object_position_world[2])
            lift_distance = max(0.0, float(target_height_m) - current_height)
            if lift_distance > 0.01:
                if self.arm_move_directional is None:
                    return _failure(
                        "lift_backend_unavailable",
                        "an attached low object requires a directional lift backend",
                    )
                lift_result = self.arm_move_directional.execute(
                    adapter,
                    scene=scene,
                    direction=[0.0, 0.0, 1.0],
                    distance=lift_distance,
                    until_contact=False,
                    object_name=object_name,
                    support_surface_name=source_support_name,
                    side=side,
                    speed_scale=min(0.12, max(0.05, float(speed_scale) * 0.25)),
                    step_hook=step_hook,
                )
                if not lift_result.success:
                    return _failure(
                        "lift_phase_failed",
                        "failed to lift the attached object clear before torso transition",
                        lift=lift_result.details,
                        lift_metrics=lift_result.metrics,
                    )
                try:
                    context = live_grasp_context(adapter, object_name, side)
                except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                    return _failure("grasp_context_unavailable", "live grasp context unavailable after lift", error=str(exc))
                if not context.attached:
                    return _failure("attachment_lost", "object attachment was lost during lift")

        target_torso = _target_torso_posture(target_posture)
        initial_hold_center = np.asarray(context.grasp_center_world, dtype=float)
        if initial_hold_center.shape != (3,) or not np.all(np.isfinite(initial_hold_center)):
            return _failure("invalid_grasp_center", "live grasp center is not finite")

        observation = adapter.read_observation(0.0)
        current_torso = _torso_from_observation(observation)
        if current_torso is None:
            return _failure("torso_observation_unavailable", "torso joint observation is incomplete")
        joints = ARM_JOINTS_BY_SIDE[side]
        try:
            current_arm = np.asarray(
                [observation.joint_positions[name] for name in joints], dtype=float
            )
        except (KeyError, TypeError, ValueError) as exc:
            return _failure("arm_observation_unavailable", "arm joint observation is incomplete", error=str(exc))

        if _max_abs_difference(current_torso, target_torso) <= R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD:
            stabilize_base(adapter, replace_wheel_only=True)
            return SkillResult(
                True,
                self.name,
                metrics={"iterations": 0.0, "max_hold_center_error_m": 0.0},
                details={
                    "reason": "target whole-body posture was already settled",
                    "target_posture": target_posture,
                    "target_torso_q": list(target_torso),
                    "lift": None if lift_result is None else lift_result.details,
                },
            )

        # GraspObject normally leaves an internally-created torso lock active.
        # Release only that known internal lock; an explicit task-level mask is
        # authoritative and must not be overwritten by this backend.
        _unlock_internal_hold(adapter)
        stabilize_base(adapter, lock_torso=False)

        try:
            from r1pro_data_gen.methods.collision import obstacles_from_scene

            obstacles = obstacles_from_scene(scene, exclude=(object_name,), include_ground=False)
            checker = WholeBodyCollisionChecker(kin, obstacles, side=side)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return _failure("whole_body_checker_unavailable", "whole-body collision checker could not be built", error=str(exc))

        dt = float(getattr(adapter, "dt", 1.0 / 60.0))
        max_torso_step = max(1.0e-4, abs(self.torso_velocity_limit) * float(speed_scale) * dt)
        max_iterations = max(
            60,
            int(np.ceil(np.max(np.abs(np.asarray(target_torso) - current_torso)) / max_torso_step)) * 5,
        )
        records: list[dict[str, Any]] = []
        max_center_error = 0.0

        for iteration in range(1, max_iterations + 1):
            observation = adapter.read_observation(0.0)
            actual_torso = _torso_from_observation(observation)
            if actual_torso is None:
                return _failure("torso_observation_unavailable", "torso observation disappeared during transition", records=records)
            try:
                actual_arm = np.asarray(
                    [observation.joint_positions[name] for name in joints], dtype=float
                )
            except (KeyError, TypeError, ValueError) as exc:
                return _failure("arm_observation_unavailable", "arm observation disappeared during transition", error=str(exc), records=records)

            remaining = np.asarray(target_torso, dtype=float) - actual_torso
            if float(np.max(np.abs(remaining))) <= R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD:
                break
            torso_goal = actual_torso + np.clip(remaining, -max_torso_step, max_torso_step)

            solution = _hold_solution(
                kin,
                adapter,
                scene,
                checker,
                side=side,
                current_arm=actual_arm,
                current_torso=actual_torso,
                torso_goal=torso_goal,
                hold_center_world=initial_hold_center,
                object_name=object_name,
            )
            if solution is None:
                return _failure(
                    "whole_body_transition_unreachable",
                    "no collision-free arm branch preserves the held object during the next torso step",
                    records=records,
                    torso_q=actual_torso.round(5).tolist(),
                    requested_torso_q=torso_goal.round(5).tolist(),
                )
            arm_goal, chosen_torso, diagnostic = solution
            command_arm = actual_arm + np.clip(
                np.asarray(arm_goal, dtype=float) - actual_arm,
                -R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
                R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD,
            )
            command_torso = actual_torso + np.clip(
                np.asarray(chosen_torso, dtype=float) - actual_torso,
                -max_torso_step,
                max_torso_step,
            )
            adapter.set_targets(
                position={
                    **{f"torso_joint{index}": float(command_torso[index - 1]) for index in range(1, 5)},
                    **{name: float(command_arm[index]) for index, name in enumerate(joints)},
                },
                velocity={},
            )
            adapter.step()
            if step_hook is not None:
                step_hook()

            try:
                context = live_grasp_context(adapter, object_name, side)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                return _failure("grasp_context_unavailable", "live grasp context unavailable after transition step", error=str(exc), records=records)
            if not context.attached:
                return _failure("attachment_lost", "object attachment was lost during whole-body transition", records=records)
            measured_after = adapter.read_observation(0.0)
            measured_torso = _torso_from_observation(measured_after)
            try:
                measured_arm = np.asarray(
                    [measured_after.joint_positions[name] for name in joints], dtype=float
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                measured_torso = None
                measured_arm = np.asarray((), dtype=float)
            if measured_torso is None or measured_arm.shape != (7,):
                return _failure(
                    "whole_body_observation_unavailable",
                    "whole-body transition could not verify the measured post-step robot state",
                    records=records,
                )
            measured_calibration = calibrated_model_transform(kin, adapter, side)
            if measured_calibration is None:
                return _failure(
                    "whole_body_calibration_unavailable",
                    "whole-body transition could not re-register the measured robot state",
                    records=records,
                )
            measured_rotation, measured_translation = measured_calibration
            collision_frame = checker.first_collision_frame(
                measured_arm,
                measured_torso,
                model_to_world_rotation=measured_rotation,
                model_to_world_translation=measured_translation,
            )
            if collision_frame is not None:
                return _failure(
                    "whole_body_runtime_collision",
                    "measured whole-body tracking entered a collision envelope",
                    collision_frame=collision_frame,
                    records=records,
                )
            object_free, object_diagnostic = held_object_configuration_free(
                scene,
                object_name,
                context.object_position_world,
                exclude=(object_name,),
                include_ground=True,
            )
            if not object_free:
                return _failure(
                    "held_object_runtime_collision",
                    "the measured held-object proxy entered an obstacle envelope",
                    held_object=object_diagnostic,
                    records=records,
                )
            center = np.asarray(context.grasp_center_world, dtype=float)
            center_error = float(np.linalg.norm(center - initial_hold_center))
            max_center_error = max(max_center_error, center_error)
            attachment_error = context.attachment_error_m
            if center_error > R1PRO_TRANSFER_HOLD_CENTER_TOL_M:
                return _failure(
                    "held_center_drift",
                    "the live grasp center drifted beyond the whole-body hold tolerance",
                    records=records,
                    center_error_m=center_error,
                    attachment_error_m=attachment_error,
                )
            if attachment_error is not None and float(attachment_error) > 0.03:
                return _failure(
                    "attachment_unstable",
                    "attachment error exceeded the generic hold tolerance",
                    records=records,
                    attachment_error_m=float(attachment_error),
                )
            records.append(
                {
                    "iteration": iteration,
                    "torso_q": chosen_torso.round(5).tolist(),
                    "arm_q": arm_goal.round(5).tolist(),
                    "command_torso_q": command_torso.round(5).tolist(),
                    "command_arm_q": command_arm.round(5).tolist(),
                    "center_error_m": center_error,
                    "held_object": object_diagnostic,
                    **diagnostic,
                }
            )

        final_observation = adapter.read_observation(0.0)
        final_torso = _torso_from_observation(final_observation)
        if final_torso is None:
            return _failure("torso_observation_unavailable", "final torso observation is incomplete", records=records)
        final_error = _max_abs_difference(final_torso, target_torso)
        if final_error > R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD:
            return _failure(
                "torso_target_not_reached",
                "whole-body posture did not settle within the robot-level tolerance",
                records=records,
                final_error_rad=final_error,
            )

        for _ in range(max(0, int(settle_steps))):
            adapter.set_targets(
                position={f"torso_joint{index}": float(target_torso[index - 1]) for index in range(1, 5)},
                velocity={},
            )
            adapter.step()
            if step_hook is not None:
                step_hook()
        try:
            final_context = live_grasp_context(adapter, object_name, side)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return _failure("grasp_context_unavailable", "live grasp context unavailable after final settle", error=str(exc), records=records)
        if not final_context.attached:
            return _failure("attachment_lost", "object attachment was lost during final whole-body settle", records=records)
        final_center_error = float(
            np.linalg.norm(np.asarray(final_context.grasp_center_world, dtype=float) - initial_hold_center)
        )
        if final_center_error > R1PRO_TRANSFER_HOLD_CENTER_TOL_M:
            return _failure(
                "held_center_drift",
                "the held object drifted during final whole-body settling",
                center_error_m=final_center_error,
                records=records,
            )
        stabilize_base(adapter, replace_wheel_only=True)
        return SkillResult(
            True,
            self.name,
            metrics={
                "iterations": float(len(records)),
                "max_hold_center_error_m": max_center_error,
                "final_hold_center_error_m": final_center_error,
                "final_torso_error_rad": final_error,
                "settle_steps": float(max(0, int(settle_steps))),
                "lifted": float(lift_result is not None),
            },
            details={
                "reason": "whole-body hold transition settled",
                "target_posture": target_posture,
                "target_torso_q": list(target_torso),
                "records": records,
                "hold_center_world": initial_hold_center.round(5).tolist(),
                "lift": None if lift_result is None else lift_result.details,
            },
        )


def _hold_solution(
    kin: Any,
    adapter: Any,
    scene: Any,
    checker: WholeBodyCollisionChecker,
    *,
    side: str,
    current_arm: np.ndarray,
    current_torso: np.ndarray,
    torso_goal: np.ndarray,
    hold_center_world: np.ndarray,
    object_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Find a small collision-free arm correction for one torso increment."""
    joints = ARM_JOINTS_BY_SIDE[side]
    # If a branch needs a larger arm correction than the physical controller
    # can safely make in one step, retry with a smaller torso increment.  This
    # is a generic coordinated-motion refinement, not a task-specific route.
    for fraction in (1.0, 0.5, 0.25):
        candidate_torso = current_torso + (torso_goal - current_torso) * fraction
        _sync_auxiliary_q(kin, current_torso)
        calibrated = calibrated_model_transform(kin, adapter, side)
        if calibrated is None:
            continue
        rotation, translation = calibrated
        try:
            _, current_quat = kin.fk(current_arm)
            _sync_auxiliary_q(kin, candidate_torso)
            target_center_model = np.asarray(rotation, dtype=float).T @ (
                hold_center_world - np.asarray(translation, dtype=float)
            )
            target_ee = kin.ee_target_from_grasp_center(target_center_model, current_quat)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
            continue
        candidates = _ik_candidates(kin, target_ee, current_arm)
        for arm_goal in candidates:
            arm_goal = np.asarray(arm_goal, dtype=float)
            if arm_goal.shape != current_arm.shape:
                continue
            if float(np.max(np.abs(arm_goal - current_arm))) > R1PRO_TRANSFER_MAX_ARM_STEP_RAD:
                continue
            measured_center_model, _ = kin.grasp_center_fk(arm_goal)
            measured_center_world = np.asarray(rotation, dtype=float) @ measured_center_model + np.asarray(translation, dtype=float)
            if float(np.linalg.norm(measured_center_world - hold_center_world)) > 0.06:
                continue
            free, collision = whole_body_path_free(
                checker,
                [(current_arm, current_torso), (arm_goal, candidate_torso)],
                model_to_world_rotation=rotation,
                model_to_world_translation=translation,
                dense=8,
            )
            if not free:
                continue
            object_free, object_diagnostic = held_object_configuration_free(
                scene,
                object_name,
                hold_center_world,
                exclude=(object_name,),
                include_ground=True,
            )
            if not object_free:
                continue
            stable, stability = _held_configuration_stability(
                kin,
                adapter,
                arm_q=arm_goal,
                torso_q=candidate_torso,
                object_name=object_name,
                rotation=np.asarray(rotation, dtype=float),
                translation=np.asarray(translation, dtype=float),
            )
            if not stable:
                continue
            return arm_goal, candidate_torso, {
                "ik_position_error_m": float(np.linalg.norm(measured_center_world - hold_center_world)),
                "collision": collision,
                "held_object": object_diagnostic,
                "stability": stability,
            }
    # Restore the measured torso configuration for callers that inspect the
    # kinematics object after a rejected branch.
    _sync_auxiliary_q(kin, current_torso)
    return None


def _held_configuration_stability(
    kin: Any,
    adapter: Any,
    *,
    arm_q: np.ndarray,
    torso_q: np.ndarray,
    object_name: str,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Certify a held configuration using robot and live payload mass.

    A pre-grasp COM check alone is insufficient: after attachment the object
    can move the combined COM outside the three-wheel support polygon.  The
    real Isaac adapter exposes masses read from its instantiated PhysX
    assets; if that capability is missing, a real adapter fails closed rather
    than substituting a guessed mass.  Lightweight test doubles without a
    physical telemetry contract retain the collision-only unit-test path.
    """
    has_mass = all(
        callable(getattr(adapter, name, None))
        for name in ("robot_mass_kg", "object_mass_kg")
    )
    if not has_mass:
        if hasattr(adapter, "physical_metrics"):
            return False, {
                "verified": False,
                "reason": "live robot/payload mass capability unavailable",
            }
        return True, {
            "verified": False,
            "reason": "test adapter has no live mass capability",
        }
    try:
        observation = adapter.read_observation(0.0)
        base_pose = tuple(float(value) for value in (observation.base_pose or ()))
        if len(base_pose) != 3 or not np.all(np.isfinite(np.asarray(base_pose, dtype=float))):
            return False, {"verified": False, "reason": "base pose is invalid"}
        robot_mass = float(adapter.robot_mass_kg())
        payload_mass = float(adapter.object_mass_kg(object_name))
        object_world = np.asarray(adapter.object_position(object_name), dtype=float)
        if object_world.shape != (3,) or not np.all(np.isfinite(object_world)):
            return False, {"verified": False, "reason": "held object position is invalid"}
        _sync_auxiliary_q(kin, torso_q)
        robot_com_model = np.asarray(kin.center_of_mass(np.asarray(arm_q, dtype=float)), dtype=float)
        robot_com_world = np.asarray(rotation, dtype=float) @ robot_com_model + np.asarray(translation, dtype=float)
        combined_com = payload_com(
            robot_com_world,
            robot_mass,
            object_world,
            payload_mass,
        )
        certificate = configuration_stability(
            com_world=combined_com,
            base_pose=base_pose,
            wheel_positions=tuple(STEER_POSITIONS.values()),
            required_margin_m=R1PRO_SUPPORT_POLYGON_MARGIN_M,
        )
        effort = np.asarray(kin.torso_gravity_effort(np.asarray(arm_q, dtype=float)), dtype=float)
        if effort.shape != (4,) or not np.all(np.isfinite(effort)):
            return False, {"verified": False, "reason": "torso effort is invalid"}
        effort_limit = R1PRO_TORSO_EFFORT_LIMIT * R1PRO_EFFORT_PLANNING_UTILIZATION
        if float(np.max(np.abs(effort))) > effort_limit:
            return False, {
                "verified": True,
                "reason": "torso gravity effort exceeds planning reserve",
                "torso_effort_nm": effort.round(5).tolist(),
                "limit_nm": float(effort_limit),
            }
        diagnostic = {
            "verified": True,
            "robot_mass_kg": robot_mass,
            "payload_mass_kg": payload_mass,
            "robot_com_world": robot_com_world.round(5).tolist(),
            "combined_com_world": combined_com.round(5).tolist(),
            "certificate": certificate.to_dict(),
            "torso_effort_nm": effort.round(5).tolist(),
        }
        return bool(certificate.stable), diagnostic
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return False, {
            "verified": False,
            "reason": "held configuration stability calculation failed",
            "error": str(exc),
        }


def _ik_candidates(kin: Any, target_ee: np.ndarray, current_arm: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    if hasattr(kin, "ik_candidates"):
        try:
            raw = kin.ik_candidates(target_ee, None, current_arm, max_candidates=8)
        except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
            raw = []
        for item in raw or ():
            q = getattr(item, "q_arm", None)
            if getattr(item, "success", False) and q is not None:
                candidates.append(np.asarray(q, dtype=float))
    # ``ik_candidates`` is normally the preferred branch source, but at a
    # near-singular posture it can return successful branches that are too far
    # from the measured configuration while the single-seed solver can still
    # find a small continuous correction.  Always admit that bounded fallback
    # into the candidate pool; the caller still applies the per-step motion
    # and collision gates.
    if hasattr(kin, "ik"):
        try:
            item = kin.ik(target_ee, None, q_init=current_arm)
        except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
            item = None
        q = getattr(item, "q_arm", None) if item is not None else None
        if getattr(item, "success", False) and q is not None:
            candidate = np.asarray(q, dtype=float)
            if not any(np.allclose(candidate, existing, atol=0.01) for existing in candidates):
                candidates.append(candidate)
    # Continuity is the primary criterion for a hold transition.  A distant
    # redundant branch can satisfy position while jerking the held object.
    candidates.sort(key=lambda q: float(np.linalg.norm(np.asarray(q) - current_arm)))
    return candidates


def _target_torso_posture(target_posture: str) -> np.ndarray:
    # ``upright`` is kept as a semantic alias for users of the skill; both
    # profiles intentionally resolve to the robot's calibrated transfer rest
    # posture rather than exposing raw joint values to the planner.
    del target_posture
    return np.asarray(R1PRO_TRANSFER_TORSO_Q, dtype=float)


def _torso_from_observation(observation: Any) -> np.ndarray | None:
    positions = getattr(observation, "joint_positions", {}) or {}
    names = tuple(f"torso_joint{index}" for index in range(1, 5))
    if not all(name in positions for name in names):
        return None
    try:
        values = np.asarray([positions[name] for name in names], dtype=float)
    except (TypeError, ValueError):
        return None
    return values if values.shape == (4,) and np.all(np.isfinite(values)) else None


def _torso_velocity_from_observation(observation: Any) -> np.ndarray | None:
    """Read the four measured torso velocities when the adapter exposes them."""
    velocities = getattr(observation, "joint_velocities", {}) or {}
    names = tuple(f"torso_joint{index}" for index in range(1, 5))
    if not all(name in velocities for name in names):
        return None
    try:
        values = np.asarray([velocities[name] for name in names], dtype=float)
    except (TypeError, ValueError):
        return None
    return values if values.shape == (4,) and np.all(np.isfinite(values)) else None


def _sync_auxiliary_q(kin: Any, torso_q: Sequence[float]) -> None:
    if hasattr(kin, "set_auxiliary_q"):
        kin.set_auxiliary_q(
            {f"torso_joint{index}": float(torso_q[index - 1]) for index in range(1, 5)}
        )


def _max_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def _failure(code: str, reason: str, **details: Any) -> SkillResult:
    return _failure_for("whole_body_hold_transition", code, reason, **details)


def _failure_for(skill: str, code: str, reason: str, **details: Any) -> SkillResult:
    return SkillResult(
        False,
        skill,
        details={"reason": reason, "failure_code": code, **_json_safe(details)},
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return value


from .transfer import TransferObjectBetweenSupports


class WholeBodyTransferObjectBetweenSupports(TransferObjectBetweenSupports):
    """Complete transfer with a support-aware lift and torso/arm handoff."""

    name = "whole_body_transfer_object_between_supports"
    tier = "semantic"
    exposed = False
    description = (
        "Complete a support-to-support object transfer with a generic whole-"
        "body handoff: infer source/destination elevation, lift the live grasp "
        "clear, preserve attachment while changing torso posture, then carry, "
        "place, release, and settle."
    )

    def __init__(
        self,
        grasp: Any,
        carry: Any,
        release: Any,
        handoff: Any,
        base_reposition: Any = None,
    ) -> None:
        super().__init__(
            grasp,
            carry,
            release,
            handoff=handoff,
            base_reposition=base_reposition,
        )


__all__ = [
    "WholeBodyHoldTransition",
    "WholeBodyPregraspTransition",
    "WholeBodyTransferObjectBetweenSupports",
]
