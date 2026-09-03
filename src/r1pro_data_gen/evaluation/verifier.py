"""Fail-closed deterministic evaluation of frozen GoalSpec predicates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Callable

from r1pro_data_gen.domain import (
    ContactEvent,
    EntityState,
    EvidenceBundle,
    EvidenceFrame,
    GoalPredicate,
    GoalSpec,
)

from .policy import VerificationPolicy
from .predicates import (
    PredicateEvaluation,
    PredicateStatus,
    VerificationReport,
    VerificationStatus,
)


class PredicateVerifier:
    """Evaluate goal predicates only from recorded episode facts."""

    def verify(
        self,
        goal_spec: GoalSpec,
        evidence: EvidenceBundle,
        policy: VerificationPolicy,
    ) -> VerificationReport:
        evaluations = tuple(
            self._evaluate(item, goal_spec, evidence, policy, invariant=False)
            for item in goal_spec.required
        ) + tuple(
            self._evaluate(item, goal_spec, evidence, policy, invariant=True)
            for item in goal_spec.invariants
        )
        invariant_violated = any(
            item.invariant and item.status is PredicateStatus.VIOLATED
            for item in evaluations
        )
        required_violated = any(
            not item.invariant and item.status is PredicateStatus.VIOLATED
            for item in evaluations
        )
        unknown = any(item.status is PredicateStatus.UNKNOWN for item in evaluations)
        required_satisfied = bool(
            any(not item.invariant for item in evaluations)
        ) and all(
            item.status is PredicateStatus.SATISFIED
            for item in evaluations
            if not item.invariant
        )
        if invariant_violated or required_violated:
            status = VerificationStatus.FAILED
            reason = "one or more required predicates or invariants were violated"
        elif unknown:
            status = VerificationStatus.INCOMPLETE
            reason = "episode evidence is incomplete or a predicate is unknown"
        elif required_satisfied:
            # GoalSpec success is physical evidence, not "every skill returned
            # success". A failed intermediate skill must not keep a satisfied
            # goal incomplete.
            status = VerificationStatus.SUCCEEDED
            reason = None
        else:
            status = VerificationStatus.INCOMPLETE
            reason = "episode evidence is incomplete or a predicate is unknown"
        return VerificationReport(
            status=status,
            predicates=evaluations,
            evidence_complete=bool(evidence.complete and not unknown),
            failure_reason=reason,
        )

    def progress(
        self,
        goal_spec: GoalSpec,
        evidence: EvidenceBundle,
        policy: VerificationPolicy,
    ) -> VerificationReport:
        """Report predicate status without treating unfinished goals as failure.

        Mid-episode ``violated`` required predicates stay ``incomplete`` so the
        agent can keep acting. Invariant violations are still terminal. Full
        success still requires every required predicate to be satisfied.
        """
        evaluations = tuple(
            self._evaluate(item, goal_spec, evidence, policy, invariant=False)
            for item in goal_spec.required
        ) + tuple(
            self._evaluate(item, goal_spec, evidence, policy, invariant=True)
            for item in goal_spec.invariants
        )
        invariant_violated = any(
            item.invariant and item.status is PredicateStatus.VIOLATED
            for item in evaluations
        )
        required = tuple(item for item in evaluations if not item.invariant)
        all_required_satisfied = bool(required) and all(
            item.status is PredicateStatus.SATISFIED for item in required
        )
        if invariant_violated:
            status = VerificationStatus.FAILED
            reason = "one or more invariants were violated"
        elif all_required_satisfied:
            status = VerificationStatus.SUCCEEDED
            reason = None
        else:
            status = VerificationStatus.INCOMPLETE
            reason = "required predicates are not all satisfied yet"
        return VerificationReport(
            status=status,
            predicates=evaluations,
            evidence_complete=bool(evidence.complete),
            failure_reason=reason,
        )

    def _evaluate(
        self,
        predicate: GoalPredicate,
        spec: GoalSpec,
        evidence: EvidenceBundle,
        policy: VerificationPolicy,
        *,
        invariant: bool,
    ) -> PredicateEvaluation:
        handlers: dict[str, Callable[..., PredicateEvaluation]] = {
            "object_at_pose": self._object_at_pose,
            "within_tolerance": self._within_tolerance,
            "inside_region": self._inside_region,
            "on_support": self._on_support,
            "contact": self._contact,
            "attached": self._attached,
            "lifted": self._lifted,
            "released": self._released,
            "settled": self._settled,
            "base_at_pose": self._base_at_pose,
            "collision_free": self._collision_free,
        }
        try:
            result = handlers[predicate.predicate](
                predicate,
                spec,
                evidence,
                policy,
            )
        except (KeyError, TypeError, ValueError) as exc:
            result = self._result(
                predicate,
                PredicateStatus.UNKNOWN,
                reason=f"invalid or missing predicate evidence: {exc}",
            )
        return PredicateEvaluation(
            predicate=result.predicate,
            status=result.status,
            requested=result.requested,
            observed=result.observed,
            error=result.error,
            tolerance=result.tolerance,
            evidence_range=result.evidence_range,
            reason=result.reason,
            invariant=invariant,
        )

    def _object_at_pose(self, item, spec, evidence, policy) -> PredicateEvaluation:
        name = _entity_argument(item, spec, "subject")
        state = _final_state(evidence, name)
        target = _vector(item.arguments.get("position"), 3, "position")
        position_error = _distance(state.position, target)
        errors = {"position_m": position_error}
        tolerances = {"position_m": policy.position_tolerance_m}
        satisfied = position_error <= policy.position_tolerance_m
        target_quaternion = item.arguments.get("quaternion")
        if target_quaternion is not None:
            target_quaternion = _vector(target_quaternion, 4, "quaternion")
            orientation_error = _quaternion_error(state.quaternion, target_quaternion)
            errors["orientation_rad"] = orientation_error
            tolerances["orientation_rad"] = policy.orientation_tolerance_rad
            satisfied = satisfied and orientation_error <= policy.orientation_tolerance_rad
        return self._result(
            item,
            _binary(satisfied),
            observed={
                "position": list(state.position),
                "quaternion": list(state.quaternion),
            },
            error=errors,
            tolerance=tolerances,
            evidence_range=_final_range(evidence),
        )

    def _within_tolerance(self, item, spec, evidence, policy) -> PredicateEvaluation:
        name = _entity_argument(item, spec, "subject")
        state = _final_state(evidence, name)
        field_name = _required_string(item.arguments.get("field"), "field")
        if field_name not in {
            "position",
            "quaternion",
            "linear_velocity",
            "angular_velocity",
        }:
            raise ValueError(f"unsupported EntityState field {field_name!r}")
        observed = tuple(getattr(state, field_name))
        target = _vector(item.arguments.get("target"), len(observed), "target")
        tolerance = _positive_number(item.arguments.get("tolerance"), "tolerance")
        if tolerance > policy.max_explicit_tolerance_m:
            raise ValueError("explicit tolerance exceeds verification policy maximum")
        error = (
            _quaternion_error(observed, target)
            if field_name == "quaternion"
            else _distance(observed, target)
        )
        return self._result(
            item,
            _binary(error <= tolerance),
            observed={"field": field_name, "value": list(observed)},
            error={"absolute": error},
            tolerance={"absolute": tolerance},
            evidence_range=_final_range(evidence),
        )

    def _inside_region(self, item, spec, evidence, policy) -> PredicateEvaluation:
        subject = _final_state(evidence, _entity_argument(item, spec, "subject"))
        reference = _final_state(evidence, _entity_argument(item, spec, "reference"))
        region = _mapping(item.arguments.get("region"), "region")
        center = _vector(region.get("center"), 3, "region.center")
        local = _world_to_local(subject.position, reference.position, reference.quaternion)
        offset = tuple(local[index] - center[index] for index in range(3))
        shape = _required_string(region.get("shape"), "region.shape")
        tolerance = policy.region_boundary_tolerance_m
        if shape == "cuboid":
            size = _vector(region.get("size"), 3, "region.size")
            margins = tuple(size[index] / 2.0 + tolerance - abs(offset[index]) for index in range(3))
            satisfied = min(margins) >= 0.0
        elif shape == "cylinder":
            radius = _positive_number(region.get("radius"), "region.radius")
            height = _positive_number(region.get("height"), "region.height")
            margins = (
                radius + tolerance - math.hypot(offset[0], offset[1]),
                height / 2.0 + tolerance - abs(offset[2]),
            )
            satisfied = min(margins) >= 0.0
        else:
            raise ValueError(f"unsupported region shape {shape!r}")
        return self._result(
            item,
            _binary(satisfied),
            observed={"local_position": list(local), "boundary_margins": list(margins)},
            error={"outside_distance_m": max(0.0, -min(margins))},
            tolerance={"boundary_m": tolerance},
            evidence_range=_final_range(evidence),
        )

    def _base_at_pose(self, item, spec, evidence, policy) -> PredicateEvaluation:
        del spec
        pose = _vector(item.arguments.get("pose"), 3, "pose")
        observed = _final_frame(evidence).base_pose
        position_error = math.hypot(observed[0] - pose[0], observed[1] - pose[1])
        yaw_error = abs(_wrap_angle(observed[2] - pose[2]))
        return self._result(
            item,
            _binary(
                position_error <= policy.base_position_tolerance_m
                and yaw_error <= policy.base_yaw_tolerance_rad
            ),
            observed={"pose": list(observed)},
            error={"position_m": position_error, "yaw_rad": yaw_error},
            tolerance={
                "position_m": policy.base_position_tolerance_m,
                "yaw_rad": policy.base_yaw_tolerance_rad,
            },
            evidence_range=_final_range(evidence),
        )

    def _contact(self, item, spec, evidence, policy) -> PredicateEvaluation:
        first = _entity_argument(item, spec, "entity_a")
        second = _entity_argument(item, spec, "entity_b")
        matched = _matching_contacts(
            evidence.contacts,
            first,
            second,
            policy.contact_force_min_n,
        )
        if not matched:
            return self._result(
                item,
                PredicateStatus.VIOLATED if evidence.complete else PredicateStatus.UNKNOWN,
                observed={"matching_events": 0},
                tolerance={
                    "force_n": policy.contact_force_min_n,
                    "duration_s": policy.contact_duration_s,
                },
                reason="no matching contact evidence",
            )
        start, end = matched[0].timestamp, matched[-1].timestamp
        duration = end - start
        return self._result(
            item,
            _binary(duration >= policy.contact_duration_s),
            observed={
                "matching_events": len(matched),
                "duration_s": duration,
                "minimum_force_n": min(event.force_n for event in matched),
            },
            error={"duration_shortfall_s": max(0.0, policy.contact_duration_s - duration)},
            tolerance={
                "force_n": policy.contact_force_min_n,
                "duration_s": policy.contact_duration_s,
            },
            evidence_range=(start, end),
        )

    def _attached(self, item, spec, evidence, policy) -> PredicateEvaluation:
        subject = _entity_argument(item, spec, "subject")
        effector = _entity_argument(item, spec, "effector", required=False)
        matched = [
            frame
            for frame in _frames(evidence)
            if subject in frame.attachments
            and (
                effector is None
                or _effector_matches(frame.attachments.get(subject), effector)
            )
        ]
        if not matched:
            return self._result(
                item,
                PredicateStatus.VIOLATED if evidence.complete else PredicateStatus.UNKNOWN,
                reason="target attachment was not observed",
            )
        start, end = matched[0].timestamp, matched[-1].timestamp
        relative_positions: list[tuple[float, float, float]] = []
        speeds: list[float] = []
        for frame in matched:
            state = frame.entities.get(subject)
            actual_effector = frame.attachments.get(subject)
            pose = _effector_pose(frame, effector or actual_effector)
            if state is None or pose is None:
                return self._result(
                    item,
                    PredicateStatus.UNKNOWN,
                    reason="attachment frame lacks entity or end-effector pose",
                )
            relative_positions.append(
                tuple(state.position[index] - pose[index] for index in range(3))
            )
            speeds.append(_norm(state.linear_velocity))
        relative_drift = max(
            _distance(relative_positions[0], position)
            for position in relative_positions
        )
        max_speed = max(speeds)
        satisfied = (
            end - start >= policy.attachment_duration_s
            and relative_drift <= policy.attachment_position_tolerance_m
        )
        return self._result(
            item,
            _binary(satisfied),
            observed={
                "duration_s": end - start,
                "relative_position_drift_m": relative_drift,
                "max_entity_speed_mps": max_speed,
            },
            tolerance={
                "duration_s": policy.attachment_duration_s,
                "relative_position_m": policy.attachment_position_tolerance_m,
            },
            evidence_range=(start, end),
        )

    def _released(self, item, spec, evidence, policy) -> PredicateEvaluation:
        subject = _entity_argument(item, spec, "subject")
        effector = _entity_argument(item, spec, "effector", required=False)
        detachments = [
            event
            for event in evidence.attachment_events
            if event.entity == subject
            and not event.attached
            and (effector is None or _effector_matches(event.effector, effector))
        ]
        if not detachments:
            return self._result(
                item,
                PredicateStatus.VIOLATED if evidence.complete else PredicateStatus.UNKNOWN,
                reason="no matching detachment event",
            )
        detachment = detachments[-1]
        final = _final_frame(evidence)
        if subject in final.attachments:
            return self._result(item, PredicateStatus.VIOLATED, reason="entity is still attached")
        selected_effector = effector or detachment.effector
        entity_state = final.entities.get(subject)
        effector_pose = _effector_pose(final, selected_effector)
        if entity_state is None or effector_pose is None:
            return self._result(
                item,
                PredicateStatus.UNKNOWN,
                reason="release evidence lacks entity or end-effector pose",
            )
        separation = _distance(entity_state.position, effector_pose[:3])
        duration = final.timestamp - detachment.timestamp
        return self._result(
            item,
            _binary(
                separation >= policy.release_separation_m
                and duration >= policy.release_duration_s
            ),
            observed={"separation_m": separation, "duration_s": duration},
            tolerance={
                "separation_m": policy.release_separation_m,
                "duration_s": policy.release_duration_s,
            },
            evidence_range=(detachment.timestamp, final.timestamp),
        )

    def _lifted(self, item, spec, evidence, policy) -> PredicateEvaluation:
        subject = _entity_argument(item, spec, "subject")
        initial = evidence.initial.entities.get(subject)
        states = [
            frame.entities.get(subject)
            for frame in _frames(evidence)
        ]
        states = [state for state in states if state is not None]
        if initial is None or not states:
            return self._result(item, PredicateStatus.UNKNOWN, reason="lift evidence is missing entity state")
        # ``lifted`` is an event-style predicate: a pick-place task may lift
        # successfully and later return the object to a support surface. A
        # terminal-only final-Z check made such a valid plan impossible.
        displacement = max(state.position[2] - initial.position[2] for state in states)
        return self._result(
            item,
            _binary(displacement >= policy.lift_displacement_m),
            observed={"vertical_displacement_m": displacement},
            tolerance={"minimum_displacement_m": policy.lift_displacement_m},
            evidence_range=(evidence.initial.timestamp, _final_frame(evidence).timestamp),
        )

    def _settled(self, item, spec, evidence, policy) -> PredicateEvaluation:
        subject = _entity_argument(item, spec, "subject")
        return self._settled_evaluation(item, subject, evidence, policy)

    def _settled_evaluation(
        self,
        item: GoalPredicate,
        subject: str,
        evidence: EvidenceBundle,
        policy: VerificationPolicy,
    ) -> PredicateEvaluation:
        stable: list[tuple[EvidenceFrame, EntityState]] = []
        for frame in reversed(_frames(evidence)):
            state = frame.entities.get(subject)
            if state is None:
                return self._result(item, PredicateStatus.UNKNOWN, reason="settled evidence is missing entity state")
            if (
                _norm(state.linear_velocity) > policy.settled_linear_velocity_mps
                or _norm(state.angular_velocity) > policy.settled_angular_velocity_radps
            ):
                break
            stable.append((frame, state))
        stable.reverse()
        if not stable:
            return self._result(item, PredicateStatus.VIOLATED, reason="no stable final frame")
        start, end = stable[0][0].timestamp, stable[-1][0].timestamp
        max_linear = max(_norm(state.linear_velocity) for _, state in stable)
        max_angular = max(_norm(state.angular_velocity) for _, state in stable)
        return self._result(
            item,
            _binary(end - start >= policy.settled_duration_s),
            observed={
                "duration_s": end - start,
                "max_linear_velocity_mps": max_linear,
                "max_angular_velocity_radps": max_angular,
            },
            tolerance={
                "duration_s": policy.settled_duration_s,
                "linear_velocity_mps": policy.settled_linear_velocity_mps,
                "angular_velocity_radps": policy.settled_angular_velocity_radps,
            },
            evidence_range=(start, end),
        )

    def _on_support(self, item, spec, evidence, policy) -> PredicateEvaluation:
        subject_name = _entity_argument(item, spec, "subject")
        support_name = _entity_argument(item, spec, "support")
        subject = _final_state(evidence, subject_name)
        support = _final_state(evidence, support_name)
        surface = _mapping(item.arguments.get("surface"), "surface")
        center = _vector(surface.get("center"), 3, "surface.center")
        size = _vector(surface.get("size"), 2, "surface.size")
        half_height = _positive_number(
            item.arguments.get("subject_half_height_m"),
            "subject_half_height_m",
        )
        local = _world_to_local(subject.position, support.position, support.quaternion)
        horizontal_margins = (
            size[0] / 2.0 + policy.region_boundary_tolerance_m - abs(local[0] - center[0]),
            size[1] / 2.0 + policy.region_boundary_tolerance_m - abs(local[1] - center[1]),
        )
        expected_z = center[2] + half_height
        height_error = abs(local[2] - expected_z)
        # Contact telemetry is pair-specific.  A scene may expose unrelated
        # streams (for example finger/object grasp contacts) without exposing
        # a subject/support sensor.  Only a declared/observed pair can impose
        # the optional duration gate for this predicate.
        pair_contacts = _matching_contacts(
            evidence.contacts,
            subject_name,
            support_name,
            0.0,
        )
        contacts = [
            event
            for event in pair_contacts
            if event.force_n >= policy.contact_force_min_n
        ]
        contact_duration = (
            contacts[-1].timestamp - contacts[0].timestamp if contacts else 0.0
        )
        # Contact telemetry is an optional strengthening signal.  Scenes that
        # do not declare pairwise support sensors can still prove support from
        # exact relative geometry plus a continuous settled window.  If any
        # contact stream is present, however, a missing subject/support pair is
        # treated as a real violation rather than silently passing.
        contact_coverage_available = bool(pair_contacts)
        contact_satisfied = (
            contact_duration >= policy.contact_duration_s
            if contact_coverage_available
            else True
        )
        settled = self._settled_evaluation(item, subject_name, evidence, policy)
        satisfied = (
            min(horizontal_margins) >= 0.0
            and height_error <= policy.support_height_tolerance_m
            and contact_satisfied
            and settled.status is PredicateStatus.SATISFIED
        )
        return self._result(
            item,
            _binary(satisfied),
            observed={
                "local_position": list(local),
                "horizontal_margins_m": list(horizontal_margins),
                "contact_duration_s": contact_duration,
                "contact_coverage": "observed" if contact_coverage_available else "not_declared",
                "settled_status": settled.status.value,
            },
            error={"height_m": height_error},
            tolerance={
                "height_m": policy.support_height_tolerance_m,
                "contact_duration_s": policy.contact_duration_s,
                "settled_duration_s": policy.settled_duration_s,
            },
            evidence_range=_final_range(evidence),
        )

    def _collision_free(self, item, spec, evidence, policy) -> PredicateEvaluation:
        del policy
        subject = _entity_argument(item, spec, "subject")
        if not evidence.complete:
            return self._result(
                item,
                PredicateStatus.UNKNOWN,
                reason="collision evidence is incomplete",
            )
        if not evidence.collision_observation_complete:
            return self._result(
                item,
                PredicateStatus.UNKNOWN,
                reason="collision telemetry coverage is unavailable",
            )
        collisions = tuple(
            event
            for event in evidence.collisions
            if _collision_involves(event, subject)
        )
        if collisions:
            first = collisions[0]
            last = collisions[-1]
            return self._result(
                item,
                PredicateStatus.VIOLATED,
                observed={
                    "collision_count": len(collisions),
                    "first_pair": [first.body_a, first.body_b],
                },
                evidence_range=(first.timestamp, last.timestamp),
                reason="one or more disallowed collisions were recorded",
            )
        return self._result(
            item,
            PredicateStatus.SATISFIED,
            observed={"collision_count": 0, "subject": subject},
            evidence_range=(evidence.initial.timestamp, _final_frame(evidence).timestamp),
        )

    @staticmethod
    def _result(
        item: GoalPredicate,
        status: PredicateStatus,
        *,
        observed: Mapping[str, object] | None = None,
        error: Mapping[str, float] | None = None,
        tolerance: Mapping[str, float] | None = None,
        evidence_range: tuple[float, float] | None = None,
        reason: str | None = None,
    ) -> PredicateEvaluation:
        return PredicateEvaluation(
            predicate=item.predicate,
            status=status,
            requested=dict(item.arguments),
            observed=observed or {},
            error=error or {},
            tolerance=tolerance or {},
            evidence_range=evidence_range,
            reason=reason,
        )


def _frames(evidence: EvidenceBundle) -> tuple[EvidenceFrame, ...]:
    return (evidence.initial, *evidence.frames)


def _final_frame(evidence: EvidenceBundle) -> EvidenceFrame:
    return evidence.frames[-1] if evidence.frames else evidence.initial


def _final_state(evidence: EvidenceBundle, name: str) -> EntityState:
    state = _final_frame(evidence).entities.get(name)
    if state is None:
        raise KeyError(f"final evidence has no entity {name!r}")
    return state


def _final_range(evidence: EvidenceBundle) -> tuple[float, float]:
    final = _final_frame(evidence).timestamp
    return final, final


def _effector_pose(
    frame: EvidenceFrame,
    name: str | None,
) -> tuple[float, ...] | None:
    """Resolve a physical attachment body to a recorded public EE pose.

    Runtime attachment events may name the physical finger midpoint used as
    the grasp anchor, while EvidenceFrame exposes the public arm EE key
    (``left_ee``/``right_ee``).  These are two names for the same side's
    measured end-effector state.  Keep the aliasing in the verifier so the
    evidence schema remains stable and unknown evidence still fails closed.
    """
    if name is None:
        return None
    pose = frame.end_effectors.get(name)
    if pose is not None:
        return pose
    aliases = {
        "left_gripper": "left_ee",
        "left_gripper_finger_midpoint": "left_ee",
        "left_finger_midpoint": "left_ee",
        "right_gripper": "right_ee",
        "right_gripper_finger_midpoint": "right_ee",
        "right_finger_midpoint": "right_ee",
    }
    alias = aliases.get(name)
    return frame.end_effectors.get(alias) if alias is not None else None


def _effector_matches(actual: str | None, requested: str | None) -> bool:
    """Match public effector aliases to the physical attachment identity."""
    if actual is None or requested is None:
        return actual == requested
    aliases = {
        "left_gripper": "left_ee",
        "left_finger_midpoint": "left_ee",
        "left_gripper_finger_midpoint": "left_ee",
        "right_gripper": "right_ee",
        "right_finger_midpoint": "right_ee",
        "right_gripper_finger_midpoint": "right_ee",
    }
    return aliases.get(actual, actual) == aliases.get(requested, requested)


def _contact_identity_matches(actual: str, requested: str) -> bool:
    """Match semantic robot-base terms to the concrete sensor body name."""
    if actual == requested:
        return True
    if requested in {"robot", "base", "mobile_base"}:
        normalized = actual.casefold()
        return (
            normalized in {"base", "base_link", "mobile_base", "chassis"}
            and not any(
                token in normalized
                for token in ("arm", "gripper", "finger", "wheel", "steer")
            )
        )
    return False


def _collision_involves(event: ContactEvent, subject: str) -> bool:
    """Scope collision evidence to the entity named by the predicate."""
    if subject in {event.body_a, event.body_b}:
        return True
    if subject in {"robot", "base", "mobile_base"}:
        return any(
            token in body.lower()
            for body in (event.body_a, event.body_b)
            for token in ("robot", "base", "wheel", "gripper", "arm", "torso")
        )
    return False


def _entity_argument(
    item: GoalPredicate,
    spec: GoalSpec,
    key: str,
    *,
    required: bool = True,
) -> str | None:
    value = item.arguments.get(key)
    if value is None and not required:
        return None
    text = _required_string(value, key)
    reference = spec.bindings.get(text)
    if reference is not None:
        return reference.removeprefix("scene://")
    if text.startswith("scene://"):
        return text.removeprefix("scene://")
    return text


def _matching_contacts(
    events: tuple[ContactEvent, ...],
    first: str,
    second: str,
    minimum_force: float,
) -> list[ContactEvent]:
    return sorted(
        (
            event
            for event in events
            if (
                (
                    _contact_identity_matches(event.body_a, first)
                    and _contact_identity_matches(event.body_b, second)
                )
                or (
                    _contact_identity_matches(event.body_a, second)
                    and _contact_identity_matches(event.body_b, first)
                )
            ) and event.force_n >= minimum_force
        ),
        key=lambda event: event.timestamp,
    )


def _binary(value: bool) -> PredicateStatus:
    return PredicateStatus.SATISFIED if value else PredicateStatus.VIOLATED


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be an object")
    return value


def _required_string(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _positive_number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{what} must be finite and positive")
    return number


def _vector(value: object, length: int, what: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{what} must be a numeric array")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be a numeric array") from exc
    if len(values) != length or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{what} must contain {length} finite values")
    return values


def _distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _norm(value: tuple[float, ...]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _quaternion_error(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    norm_a = _norm(first)
    norm_b = _norm(second)
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("quaternion must not be zero")
    dot = abs(sum(a * b for a, b in zip(first, second)) / (norm_a * norm_b))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _world_to_local(
    point: tuple[float, float, float],
    origin: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    vector = tuple(point[index] - origin[index] for index in range(3))
    w, x, y, z = quaternion
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq <= 0.0:
        raise ValueError("reference quaternion must not be zero")
    inverse = (w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq)
    return _quaternion_rotate(inverse, vector)


def _quaternion_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["PredicateVerifier"]
