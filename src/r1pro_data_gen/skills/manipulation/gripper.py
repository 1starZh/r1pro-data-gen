"""Gripper skill: set the gripper opening (left/right)."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from r1pro_data_gen.domain import object_xy_radius_m

from ..core.base import ParamSpec, SkillResult, stabilize_base
from ..core.sides import require_side

GRIPPER_OPEN = 0.05  # m finger travel (full open ~12.7 cm gap)
GRIPPER_CLOSED = 0.0

_GRIPPER_JOINT1 = {
    "left": "left_gripper_finger_joint1",
    "right": "right_gripper_finger_joint1",
}
_GRIPPER_JOINT2 = {
    "left": "left_gripper_finger_joint2",
    "right": "right_gripper_finger_joint2",
}


class GripperSet:
    """Open or close a gripper to a target opening value (fingers spread by
    the object when closing around it, producing the pinch force)."""

    name = "gripper_set"
    description = "Set a gripper to a target opening (m): 0.05 full open, 0.0 closed."
    parameters: dict[str, ParamSpec] = {
        "open_value": ParamSpec("number", "Target finger opening (m)", default=GRIPPER_OPEN),
        "side": ParamSpec("string", "Which gripper ('left' or 'right')", default="left", enum=("left", "right")),
        "object_name": ParamSpec("string", "Optional object to detach before opening", default=None),
    }

    def __init__(self, hold_steps: int = 16):
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        open_value: float = GRIPPER_OPEN,
        side: str = "left",
        object_name: str | None = None,
        hold_steps: int | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        side = require_side(side)
        detached = False
        if object_name and hasattr(adapter, "detach_object"):
            detached = bool(adapter.detach_object(object_name))
        j1, j2 = _GRIPPER_JOINT1[side], _GRIPPER_JOINT2[side]
        adapter.set_targets(
            position={j1: open_value, j2: -open_value},
            velocity={},
        )
        held = self.hold_steps if hold_steps is None else max(1, int(hold_steps))
        for _ in range(held):
            adapter.step()
            if step_hook is not None:
                step_hook()
        obs = adapter.read_observation(0.0)
        actual = obs.joint_positions.get(j1, 0.0)
        separation = _object_effector_separation(adapter, object_name, side)
        if object_name and hasattr(adapter, "attachment_state"):
            try:
                detached = detached and object_name not in adapter.attachment_state()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                detached = False
        success = bool(abs(actual - open_value) < 0.01)
        if object_name:
            success = success and detached
        return SkillResult(
            success=success,
            skill=self.name,
            metrics={
                "final_finger_pos_m": float(actual),
                "detached": detached,
                "separation_m": separation,
                "failure_code": None if success else (
                    "detach_not_observed" if object_name and not detached else "opening_not_reached"
                ),
            },
        )


class GripperGrasp:
    """Close the gripper until finger contact is detected (robust grasp)."""

    name = "gripper_grasp"
    description = "Close a gripper in small steps until the fingers contact an object (or reach max_close), pinching it."
    parameters: dict[str, ParamSpec] = {
        "max_close": ParamSpec("number", "Maximum finger travel from the measured opening (m)", default=0.05),
        "contact_threshold": ParamSpec("number", "Per-finger contact force to treat as grasp (N)", default=1.0),
        "step": ParamSpec("number", "Finger close increment (m)", default=0.002),
        "side": ParamSpec("string", "Which gripper ('left' or 'right')", default="left", enum=("left", "right")),
        "object_name": ParamSpec("string", "Scene object that both fingers must contact and hold", required=True),
        "object_motion_tolerance_m": ParamSpec(
            "number",
            "Maximum target displacement before attachment; defaults to a geometry/physics-derived value",
            default=None,
            minimum=1e-4,
        ),
    }

    def __init__(self, hold_steps: int = 10):
        self.hold_steps = hold_steps

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        max_close: float = 0.05,
        contact_threshold: float = 1.0,
        step: float = 0.002,
        side: str = "left",
        object_name: str | None = None,
        object_motion_tolerance_m: float | None = None,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        side = require_side(side)
        if not object_name:
            raise ValueError("gripper_grasp requires object_name")
        if scene is None or not hasattr(scene, "object"):
            raise ValueError("gripper_grasp requires a scene that resolves object_name")
        try:
            object_model = scene.object(object_name)
        except KeyError as exc:
            raise ValueError(
                f"gripper_grasp object_name {object_name!r} is not present in the scene"
            ) from exc
        for name, value in (
            ("max_close", max_close),
            ("contact_threshold", contact_threshold),
            ("step", step),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        # A grasp is not allowed to turn a mis-centered contact into a push.
        # Keep a live reference until the fixed attachment is created.  The
        # tolerance is derived from object scale and authored PhysX margins so
        # this guard applies to arbitrary primitive objects, not one scene.
        motion_reference: np.ndarray | None = None
        if hasattr(adapter, "object_position"):
            try:
                measured_reference = np.asarray(
                    adapter.object_position(object_name), dtype=float
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                measured_reference = None
            if (
                measured_reference is not None
                and measured_reference.shape == (3,)
                and np.all(np.isfinite(measured_reference))
            ):
                motion_reference = measured_reference.copy()
        if object_motion_tolerance_m is None:
            try:
                footprint = max(0.01, float(object_xy_radius_m(object_model)))
            except (AttributeError, TypeError, ValueError):
                footprint = 0.03
            physics = getattr(object_model, "physics", None)
            try:
                planning_margin = max(
                    0.0, float(getattr(physics, "planning_margin", 0.0) or 0.0)
                )
            except (TypeError, ValueError):
                planning_margin = 0.0
            try:
                contact_offset = max(
                    0.0, float(getattr(physics, "contact_offset", 0.0) or 0.0)
                )
            except (TypeError, ValueError):
                contact_offset = 0.0
            # An open-gripper approach and a verified pinch are different
            # physical phases.  Once the adapter proves the object is inside
            # the two-finger window, symmetric closure may center it by up to
            # half the authored contact offset before both contacts become
            # stable.  Keep the original 3 mm no-push cap unless that geometry
            # certificate is present; even then the bounded pinch cap remains
            # far below the object's footprint.
            verified_pinch_window = False
            if hasattr(adapter, "gripper_object_alignment"):
                try:
                    alignment = adapter.gripper_object_alignment(
                        object_name,
                        side=side,
                    )
                    verified_pinch_window = bool(
                        alignment.get("between_fingers", False)
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    verified_pinch_window = False
            motion_tolerance = min(
                0.005 if verified_pinch_window else 0.003,
                max(
                    0.001,
                    0.05 * footprint,
                    (
                        contact_offset
                        if verified_pinch_window
                        else 0.5 * contact_offset
                    ),
                    0.10 * planning_margin,
                ),
            )
        else:
            motion_tolerance = float(object_motion_tolerance_m)
            if not math.isfinite(motion_tolerance) or motion_tolerance <= 0.0:
                raise ValueError(
                    "object_motion_tolerance_m must be finite and positive"
                )

        def object_motion_violation() -> dict[str, Any] | None:
            if motion_reference is None or not hasattr(adapter, "object_position"):
                return None
            try:
                current_position = np.asarray(
                    adapter.object_position(object_name), dtype=float
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                return None
            if current_position.shape != (3,) or not np.all(np.isfinite(current_position)):
                return None
            displacement = float(
                np.linalg.norm(current_position - motion_reference)
            )
            if displacement <= motion_tolerance:
                return None
            return {
                "reason": "target object moved before attachment",
                "failure_code": "object_moved_before_attachment",
                "object_name": object_name,
                "object_motion_m": displacement,
                "object_motion_tolerance_m": motion_tolerance,
                "initial_object_position": motion_reference.round(6).tolist(),
                "current_object_position": current_position.round(6).tolist(),
            }

        stabilize_base(adapter)
        # A preceding alignment trajectory may have stopped on a contact
        # sample while its last arm target is still ahead of the measured
        # state.  ``GripperGrasp`` is a new controller phase: closing the
        # fingers must not leave the arm/torso continuing that stale
        # trajectory.  Capture the measured non-gripper state once and reuse
        # it for every close/settle command.  This is deliberately derived
        # from the adapter observation, so it applies to either arm and to
        # arbitrary tasks without embedding a posture or scene coordinate.
        if hasattr(adapter, "rebase_joint_mask_targets"):
            try:
                adapter.rebase_joint_mask_targets()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Older/lightweight adapters do not expose a phase rebase;
                # their existing stabilize_base contract remains valid.
                pass
        j1, j2 = _GRIPPER_JOINT1[side], _GRIPPER_JOINT2[side]
        obs = adapter.read_observation(0.0)
        current1 = float(obs.joint_positions.get(j1, 0.0))
        raw_current2 = obs.joint_positions.get(j2)
        current2 = current1 if raw_current2 is None else -float(raw_current2)
        if not math.isfinite(current2):
            current2 = current1
        current1 = max(0.0, current1)
        current2 = max(0.0, current2)
        minimum_opening1 = max(0.0, current1 - float(max_close))
        minimum_opening2 = max(0.0, current2 - float(max_close))
        hold_positions = {
            str(name): float(value)
            for name, value in getattr(obs, "joint_positions", {}).items()
            if "gripper_finger_joint" not in str(name)
            and math.isfinite(float(value))
        }

        def set_grasp_targets(opening1: float, opening2: float) -> None:
            """Close the fingers while holding the measured robot phase."""
            position = dict(hold_positions)
            position[j1] = float(opening1)
            position[j2] = -float(opening2)
            adapter.set_targets(position=position, velocity={})

        c1, c2 = current1, current2
        initial_forces = _finger_forces(adapter, side)
        initial_f1 = initial_forces[0] if len(initial_forces) > 0 else 0.0
        initial_f2 = initial_forces[1] if len(initial_forces) > 1 else 0.0
        stop1 = initial_f1 > contact_threshold
        stop2 = initial_f2 > contact_threshold
        peak_f1, peak_f2 = initial_f1, initial_f2
        contact_bodies = _finger_contact_bodies(
            adapter, side, float(contact_threshold)
        )
        both_fingers = contact_bodies == [object_name, object_name]

        while (not stop1 and c1 > minimum_opening1) or (
            not stop2 and c2 > minimum_opening2
        ):
            if not stop1 and c1 > minimum_opening1:
                c1 = max(minimum_opening1, c1 - float(step))
            if not stop2 and c2 > minimum_opening2:
                c2 = max(minimum_opening2, c2 - float(step))
            set_grasp_targets(c1, c2)
            adapter.step()
            if step_hook is not None:
                step_hook()
            forces = _finger_forces(adapter, side)
            f1 = forces[0] if len(forces) > 0 else 0.0
            f2 = forces[1] if len(forces) > 1 else 0.0
            peak_f1, peak_f2 = max(peak_f1, f1), max(peak_f2, f2)
            contact_bodies = _finger_contact_bodies(
                adapter, side, float(contact_threshold)
            )
            both_fingers = contact_bodies == [object_name, object_name]
            # Read contact identity before the displacement guard.  The first
            # frame in which both fingers establish target contact is already
            # a valid transition out of the no-push phase: compliant PhysX
            # contact may move/settle the object before the fixed joint is
            # authored.  A frame with only one finger (or the wrong object)
            # remains subject to the strict pre-attachment motion cap.
            if not both_fingers:
                violation = object_motion_violation()
                if violation is not None:
                    return SkillResult(
                        False,
                        self.name,
                        metrics={
                            "failure_code": "object_moved_before_attachment",
                            "object_motion_m": violation["object_motion_m"],
                            "object_motion_tolerance_m": motion_tolerance,
                        },
                        details=violation,
                    )
            stop1 = stop1 or f1 > contact_threshold
            stop2 = stop2 or f2 > contact_threshold

        # Contact events can flicker for a frame while the closed-loop joint
        # drive settles.  Keep the safety gate fail-closed, but allow a bounded
        # re-close of only the finger that lost contact and require a stable
        # tail of observations before accepting the pinch.  This avoids
        # discarding a physically valid two-sided grasp because of one transient
        # event, without treating a single finger or wrong object as success.
        stable_tail = 1 if both_fingers else 0
        for _ in range(self.hold_steps):
            adapter.step()
            if step_hook is not None:
                step_hook()
            contact_bodies = _finger_contact_bodies(
                adapter, side, float(contact_threshold)
            )
            current_both = contact_bodies == [object_name, object_name]
            if current_both:
                stable_tail += 1
            else:
                # Once the two-sided contact frame has passed, the object is
                # allowed to settle only while that two-sided identity is
                # still present.  If contact is lost, return to the strict
                # no-push guard before attempting a bounded re-close.
                violation = object_motion_violation()
                if violation is not None:
                    return SkillResult(
                        False,
                        self.name,
                        metrics={
                            "failure_code": "object_moved_before_attachment",
                            "object_motion_m": violation["object_motion_m"],
                            "object_motion_tolerance_m": motion_tolerance,
                        },
                        details=violation,
                    )
                stable_tail = 0
                # If a target finger has lost contact, close it by one already
                # bounded increment.  The opposite finger is not moved, so a
                # wrong-object or one-sided contact cannot be converted into a
                # blind pinch.
                changed = False
                if contact_bodies[0] != object_name and c1 > minimum_opening1:
                    c1 = max(minimum_opening1, c1 - float(step))
                    changed = True
                if contact_bodies[1] != object_name and c2 > minimum_opening2:
                    c2 = max(minimum_opening2, c2 - float(step))
                    changed = True
                if changed:
                    set_grasp_targets(c1, c2)
            both_fingers = current_both
        contact_stable = both_fingers and stable_tail >= max(1, min(3, self.hold_steps))

        attached = False
        attachment_stable = False
        attachment_error = 0.0
        failure_code: str | None = None
        attachment_failure: dict[str, object] | None = None
        if not both_fingers or not contact_stable:
            failure_code = "target_contact_not_established"
        elif not hasattr(adapter, "attach_object"):
            failure_code = "attachment_not_established"
        else:
            effector = f"{side}_gripper_finger_midpoint"
            attached = bool(
                adapter.attach_object(object_name, body_name=effector)
            )
            if not attached:
                failure_code = "attachment_not_established"
                diagnostic = getattr(adapter, "last_grasp_attachment_failure", None)
                if isinstance(diagnostic, dict):
                    attachment_failure = dict(diagnostic)
            else:
                attachment_stable = True
                for _ in range(max(1, self.hold_steps)):
                    adapter.step()
                    if step_hook is not None:
                        step_hook()
                    contact_bodies = _finger_contact_bodies(
                        adapter, side, float(contact_threshold)
                    )
                    if contact_bodies != [object_name, object_name]:
                        contact_stable = False
                    if hasattr(adapter, "attachment_state"):
                        try:
                            state = adapter.attachment_state()
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            attachment_stable = False
                        else:
                            attachment_stable = (
                                attachment_stable
                                and state.get(object_name) == effector
                            )
                    if hasattr(adapter, "grasp_attachment_error"):
                        try:
                            current_error = float(
                                adapter.grasp_attachment_error(object_name)
                            )
                        except (KeyError, RuntimeError, TypeError, ValueError):
                            attachment_stable = False
                        else:
                            attachment_error = max(
                                attachment_error, current_error
                            )
                            if (
                                not math.isfinite(current_error)
                                or current_error > 0.03
                            ):
                                attachment_stable = False
                if not attachment_stable or not contact_stable:
                    if hasattr(adapter, "detach_object"):
                        adapter.detach_object(object_name)
                    attached = False
                    failure_code = (
                        "target_contact_not_established"
                        if not contact_stable
                        else "attachment_unstable"
                    )

        details: dict[str, Any] = {}
        if attachment_failure is not None:
            details["attachment_failure"] = attachment_failure
        if attached and hasattr(adapter, "get_grasp_context"):
            try:
                details["grasp_context"] = adapter.get_grasp_context(
                    object_name, side=side
                ).to_dict()
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
        return SkillResult(
            success=both_fingers and contact_stable and attached and attachment_stable,
            skill=self.name,
            metrics={
                "final_finger_pos_m": min(c1, c2),
                "contact_detected": both_fingers,
                "contact_bodies": contact_bodies,
                "both_fingers": both_fingers,
                "attached": attached,
                "attachment_stable": attachment_stable,
                "attachment_error_m": attachment_error,
                "peak_contact_force_n": [peak_f1, peak_f2],
                "failure_code": failure_code,
            },
            details=details,
        )


def _object_effector_separation(
    adapter: Any,
    object_name: str | None,
    side: str,
) -> float | None:
    if (
        not object_name
        or not hasattr(adapter, "object_position")
        or not hasattr(adapter, "end_effector_poses")
    ):
        return None
    try:
        object_position = adapter.object_position(object_name)
        effector_pose = adapter.end_effector_poses().get(f"{side}_ee")
        if effector_pose is None:
            return None
        return math.sqrt(
            sum(
                (float(object_position[index]) - float(effector_pose[index])) ** 2
                for index in range(3)
            )
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None


def _finger_contact_bodies(
    adapter: Any,
    side: str,
    contact_threshold: float,
) -> list[str | None]:
    bodies = [None, None]
    if not hasattr(adapter, "contact_events"):
        return bodies
    try:
        events = adapter.contact_events()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return bodies
    finger_names = (
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
    )
    for event in events:
        try:
            if float(event.force_n) <= contact_threshold:
                continue
            body_a = str(event.body_a)
            body_b = str(event.body_b)
        except (AttributeError, TypeError, ValueError):
            continue
        for index, finger_name in enumerate(finger_names):
            if body_a == finger_name:
                bodies[index] = body_b
            elif body_b == finger_name:
                bodies[index] = body_a
    return bodies


def _finger_forces(adapter: Any, side: str) -> tuple[float, ...]:
    """Read one gripper's contacts while keeping legacy adapters usable."""
    try:
        values = adapter.finger_contact_forces(side=side)
    except TypeError:
        values = adapter.finger_contact_forces()
    return tuple(float(value) for value in values)


__all__ = ["GRIPPER_CLOSED", "GRIPPER_OPEN", "GripperGrasp", "GripperSet"]
