from __future__ import annotations

from dataclasses import replace

import pytest

from r1pro_data_gen.domain import (
    AttachmentEvent,
    ContactEvent,
    EntityState,
    EvidenceBundle,
    EvidenceFrame,
    GoalPredicate,
    GoalSpec,
)
from r1pro_data_gen.evaluation import (
    PredicateStatus,
    PredicateVerifier,
    VerificationPolicy,
    VerificationStatus,
)


def _state(
    position=(0.0, 0.0, 0.0),
    linear_velocity=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
) -> EntityState:
    return EntityState(
        position=position,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )


def _frame(
    timestamp: float,
    *,
    item: EntityState | None = None,
    support: EntityState | None = None,
    base_pose=(0.0, 0.0, 0.0),
    attachments: dict[str, str] | None = None,
    end_effectors: dict[str, tuple[float, ...]] | None = None,
) -> EvidenceFrame:
    entities = {}
    if item is not None:
        entities["item"] = item
    if support is not None:
        entities["support"] = support
    return EvidenceFrame(
        timestamp=timestamp,
        base_pose=base_pose,
        base_velocity=(0.0, 0.0, 0.0),
        joint_positions={},
        joint_velocities={},
        end_effectors=end_effectors or {},
        entities=entities,
        attachments=attachments or {},
    )


def _evidence(
    *,
    initial: EvidenceFrame | None = None,
    frames: tuple[EvidenceFrame, ...] = (),
    contacts: tuple[ContactEvent, ...] = (),
    collisions: tuple[ContactEvent, ...] = (),
    attachment_events: tuple[AttachmentEvent, ...] = (),
    complete: bool = True,
) -> EvidenceBundle:
    return EvidenceBundle(
        initial=initial or _frame(0.0, item=_state()),
        frames=frames,
        contacts=contacts,
        collisions=collisions,
        attachment_events=attachment_events,
        complete=complete,
        collision_observation_complete=True,
    )


def _spec(
    predicate: str,
    arguments: dict[str, object],
    *,
    invariant: bool = False,
) -> GoalSpec:
    item = GoalPredicate(predicate=predicate, arguments=arguments)
    return GoalSpec(
        schema_version=1,
        bindings={"item": "scene://item", "support": "scene://support"},
        required=(GoalPredicate("released", {"subject": "item"}),) if invariant else (item,),
        invariants=(item,) if invariant else (),
    )


def test_progress_does_not_fail_the_episode_for_unsatisfied_required_predicates() -> None:
    report = PredicateVerifier().progress(
        _spec("object_at_pose", {"subject": "item", "position": [1.0, 2.0, 0.5]}),
        _evidence(frames=(_frame(0.1, item=_state((0.0, 0.0, 0.0))),)),
        VerificationPolicy(position_tolerance_m=0.02),
    )
    assert report.status is VerificationStatus.INCOMPLETE
    assert report.predicates[0].status is PredicateStatus.VIOLATED


def test_verifier_is_fail_closed_when_required_evidence_is_missing() -> None:
    report = PredicateVerifier().verify(
        _spec("settled", {"subject": "item"}),
        _evidence(initial=_frame(0.0), complete=False),
        VerificationPolicy(),
    )

    assert report.status is VerificationStatus.INCOMPLETE
    assert report.predicates[0].status is PredicateStatus.UNKNOWN
    assert not report.evidence_complete


def test_verifier_succeeds_when_predicates_are_satisfied_despite_incomplete_bundle() -> None:
    report = PredicateVerifier().verify(
        _spec(
            "object_at_pose",
            {"subject": "item", "position": [1.0, 2.0, 0.5]},
        ),
        _evidence(
            frames=(_frame(0.1, item=_state((1.01, 2.0, 0.5))),),
            complete=False,
        ),
        VerificationPolicy(position_tolerance_m=0.02),
    )
    assert report.status is VerificationStatus.SUCCEEDED
    assert report.predicates[0].status is PredicateStatus.SATISFIED


def test_object_at_pose_uses_live_final_frame_and_reports_error() -> None:
    report = PredicateVerifier().verify(
        _spec(
            "object_at_pose",
            {"subject": "item", "position": [1.0, 2.0, 0.5]},
        ),
        _evidence(frames=(_frame(0.1, item=_state((1.01, 2.0, 0.5))),)),
        VerificationPolicy(position_tolerance_m=0.02),
    )

    evaluation = report.predicates[0]
    assert report.status is VerificationStatus.SUCCEEDED
    assert evaluation.status is PredicateStatus.SATISFIED
    assert evaluation.error["position_m"] == pytest.approx(0.01)
    assert evaluation.observed["position"] == pytest.approx([1.01, 2.0, 0.5])


def test_on_support_can_use_geometry_and_settled_window_without_pair_sensor() -> None:
    support = _state((1.0, 2.0, 0.4))
    item = _state((1.1, 2.0, 0.55))
    evidence = _evidence(
        initial=_frame(0.0, item=item, support=support),
        frames=(_frame(0.1, item=item, support=support), _frame(0.25, item=item, support=support)),
        contacts=(),
    )
    report = PredicateVerifier().verify(
        _spec(
            "on_support",
            {
                "subject": "item",
                "support": "support",
                "surface": {"center": [0.0, 0.0, 0.1], "size": [0.5, 0.5]},
                "subject_half_height_m": 0.05,
            },
        ),
        evidence,
        VerificationPolicy(support_height_tolerance_m=0.01, settled_duration_s=0.1),
    )

    assert report.status is VerificationStatus.SUCCEEDED
    assert report.predicates[0].observed["contact_coverage"] == "not_declared"


def test_on_support_ignores_unrelated_contact_streams() -> None:
    support = _state((1.0, 2.0, 0.4))
    item = _state((1.1, 2.0, 0.55))
    evidence = _evidence(
        initial=_frame(0.0, item=item, support=support),
        frames=(_frame(0.1, item=item, support=support), _frame(0.25, item=item, support=support)),
        contacts=(ContactEvent(0.1, "finger", "item", 2.0),),
    )
    report = PredicateVerifier().verify(
        _spec(
            "on_support",
            {
                "subject": "item",
                "support": "support",
                "surface": {"center": [0.0, 0.0, 0.1], "size": [0.5, 0.5]},
                "subject_half_height_m": 0.05,
            },
        ),
        evidence,
        VerificationPolicy(support_height_tolerance_m=0.01, settled_duration_s=0.1),
    )

    assert report.status is VerificationStatus.SUCCEEDED
    assert report.predicates[0].observed["contact_coverage"] == "not_declared"


def test_inside_region_and_base_at_pose_use_explicit_geometry() -> None:
    verifier = PredicateVerifier()
    evidence = _evidence(
        frames=(
            _frame(
                0.1,
                item=_state((1.1, 2.0, 0.5)),
                support=_state((1.0, 2.0, 0.4)),
                base_pose=(0.49, -0.01, 0.02),
            ),
        )
    )
    region = verifier.verify(
        _spec(
            "inside_region",
            {
                "subject": "item",
                "reference": "support",
                "region": {
                    "shape": "cuboid",
                    "center": [0.0, 0.0, 0.1],
                    "size": [0.4, 0.4, 0.2],
                },
            },
        ),
        evidence,
        VerificationPolicy(region_boundary_tolerance_m=0.001),
    )
    base = verifier.verify(
        _spec("base_at_pose", {"pose": [0.5, 0.0, 0.0]}),
        evidence,
        VerificationPolicy(base_position_tolerance_m=0.03, base_yaw_tolerance_rad=0.05),
    )

    assert region.status is VerificationStatus.SUCCEEDED
    assert base.status is VerificationStatus.SUCCEEDED


def test_contact_requires_matching_identity_force_and_duration() -> None:
    evidence = _evidence(
        frames=(_frame(0.2, item=_state()),),
        contacts=(
            ContactEvent(0.05, "finger", "item", 1.0),
            ContactEvent(0.12, "item", "finger", 1.1),
        ),
    )
    report = PredicateVerifier().verify(
        _spec(
            "contact",
            {"entity_a": "finger", "entity_b": "item"},
        ),
        evidence,
        VerificationPolicy(contact_force_min_n=0.5, contact_duration_s=0.05),
    )

    assert report.status is VerificationStatus.SUCCEEDED
    assert report.predicates[0].evidence_range == pytest.approx((0.05, 0.12))


def test_contact_accepts_robot_base_semantic_alias_for_concrete_sensor_body() -> None:
    evidence = _evidence(
        contacts=(
            ContactEvent(0.05, "base_link", "item", 1.0),
            ContactEvent(0.12, "base_link", "item", 1.0),
        )
    )

    report = PredicateVerifier().verify(
        _spec("contact", {"entity_a": "robot", "entity_b": "item"}),
        evidence,
        VerificationPolicy(contact_force_min_n=0.5, contact_duration_s=0.05),
    )

    assert report.status is VerificationStatus.SUCCEEDED


def test_attached_requires_stable_window_and_released_requires_detach_and_separation() -> None:
    initial = _frame(
        0.0,
        item=_state((0.0, 0.0, 0.5)),
        attachments={"item": "left_gripper"},
        end_effectors={"left_gripper": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
    )
    attached_evidence = _evidence(
        initial=initial,
        frames=(
            replace(initial, timestamp=0.1),
            replace(initial, timestamp=0.2),
        ),
        attachment_events=(AttachmentEvent(0.0, "item", "left_gripper", True),),
    )
    attached = PredicateVerifier().verify(
        _spec("attached", {"subject": "item", "effector": "left_gripper"}),
        attached_evidence,
        VerificationPolicy(attachment_duration_s=0.1),
    )

    released_evidence = _evidence(
        initial=initial,
        frames=(
            _frame(
                0.1,
                item=_state((0.0, 0.0, 0.5)),
                end_effectors={"left_gripper": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
            ),
            _frame(
                0.25,
                item=_state((0.0, 0.0, 0.5)),
                end_effectors={"left_gripper": (0.2, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
            ),
        ),
        attachment_events=(AttachmentEvent(0.1, "item", "left_gripper", False),),
    )
    released = PredicateVerifier().verify(
        _spec("released", {"subject": "item", "effector": "left_gripper"}),
        released_evidence,
        VerificationPolicy(release_duration_s=0.1, release_separation_m=0.05),
    )

    assert attached.status is VerificationStatus.SUCCEEDED
    assert released.status is VerificationStatus.SUCCEEDED


def test_attached_accepts_any_observed_end_effector_when_effector_is_omitted() -> None:
    initial = _frame(
        0.0,
        item=_state((0.0, 0.0, 0.5)),
        attachments={"item": "left_gripper_finger_midpoint"},
        end_effectors={"left_ee": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
    )
    evidence = _evidence(
        initial=initial,
        frames=(
            replace(initial, timestamp=0.1),
            replace(initial, timestamp=0.2),
        ),
        attachment_events=(
            AttachmentEvent(0.0, "item", "left_gripper_finger_midpoint", True),
        ),
    )

    report = PredicateVerifier().verify(
        _spec("attached", {"subject": "item"}),
        evidence,
        VerificationPolicy(attachment_duration_s=0.1),
    )

    assert report.status is VerificationStatus.SUCCEEDED


def test_attached_prefers_the_exact_physical_grasp_anchor_pose() -> None:
    initial = _frame(
        0.0,
        item=_state((0.10, 0.20, 0.50)),
        attachments={"item": "left_gripper_finger_midpoint"},
        end_effectors={
            "left_ee": (0.10, 0.20, 0.50, 1.0, 0.0, 0.0, 0.0),
            "left_gripper_finger_midpoint": (0.10, 0.20, 0.50, 1.0, 0.0, 0.0, 0.0),
        },
    )
    evidence = _evidence(
        initial=initial,
        frames=(
            _frame(
                0.1,
                item=_state((0.10, 0.20, 0.50)),
                attachments={"item": "left_gripper_finger_midpoint"},
                end_effectors={
                    "left_ee": (0.125, 0.20, 0.50, 1.0, 0.0, 0.0, 0.0),
                    "left_gripper_finger_midpoint": (0.10, 0.20, 0.50, 1.0, 0.0, 0.0, 0.0),
                },
            ),
            _frame(
                0.2,
                item=_state((0.10, 0.20, 0.50)),
                attachments={"item": "left_gripper_finger_midpoint"},
                end_effectors={
                    "left_ee": (0.15, 0.20, 0.50, 1.0, 0.0, 0.0, 0.0),
                    "left_gripper_finger_midpoint": (0.10, 0.20, 0.50, 1.0, 0.0, 0.0, 0.0),
                },
            ),
        ),
        attachment_events=(
            AttachmentEvent(0.0, "item", "left_gripper_finger_midpoint", True),
        ),
    )

    report = PredicateVerifier().verify(
        _spec("attached", {"subject": "item"}),
        evidence,
        VerificationPolicy(attachment_duration_s=0.1),
    )

    assert report.status is VerificationStatus.SUCCEEDED
    assert report.predicates[0].observed["relative_position_drift_m"] == pytest.approx(0.0)


def test_released_resolves_physical_finger_midpoint_to_public_ee_pose() -> None:
    evidence = _evidence(
        initial=_frame(
            0.0,
            item=_state((0.0, 0.0, 0.5)),
            attachments={"item": "left_gripper_finger_midpoint"},
            end_effectors={"left_ee": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
        ),
        frames=(
            _frame(
                0.1,
                item=_state((0.0, 0.0, 0.5)),
                end_effectors={"left_ee": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
            ),
            _frame(
                0.25,
                item=_state((0.2, 0.0, 0.5)),
                end_effectors={"left_ee": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
            ),
        ),
        attachment_events=(
            AttachmentEvent(0.1, "item", "left_gripper_finger_midpoint", False),
        ),
    )
    report = PredicateVerifier().verify(
        _spec("released", {"subject": "item"}),
        evidence,
        VerificationPolicy(release_duration_s=0.1, release_separation_m=0.05),
    )
    assert report.status is VerificationStatus.SUCCEEDED
    assert report.predicates[0].observed["separation_m"] == pytest.approx(0.2)


def test_released_accepts_public_gripper_alias_for_physical_midpoint_event() -> None:
    evidence = _evidence(
        initial=_frame(
            0.0,
            item=_state((0.0, 0.0, 0.5)),
            attachments={"item": "left_gripper_finger_midpoint"},
            end_effectors={"left_ee": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
        ),
        frames=(
            _frame(
                0.25,
                item=_state((0.2, 0.0, 0.5)),
                end_effectors={"left_ee": (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)},
            ),
        ),
        attachment_events=(
            AttachmentEvent(0.1, "item", "left_gripper_finger_midpoint", False),
        ),
    )
    report = PredicateVerifier().verify(
        _spec("released", {"subject": "item", "effector": "left_gripper"}),
        evidence,
        VerificationPolicy(release_duration_s=0.1, release_separation_m=0.05),
    )

    assert report.status is VerificationStatus.SUCCEEDED


def test_lifted_and_settled_use_initial_state_and_continuous_final_window() -> None:
    evidence = _evidence(
        initial=_frame(0.0, item=_state((0.0, 0.0, 0.1))),
        frames=(
            _frame(0.1, item=_state((0.0, 0.0, 0.2))),
            _frame(0.25, item=_state((0.0, 0.0, 0.2))),
        ),
    )
    lifted = PredicateVerifier().verify(
        _spec("lifted", {"subject": "item"}),
        evidence,
        VerificationPolicy(lift_displacement_m=0.05),
    )
    settled = PredicateVerifier().verify(
        _spec("settled", {"subject": "item"}),
        evidence,
        VerificationPolicy(settled_duration_s=0.1),
    )

    assert lifted.status is VerificationStatus.SUCCEEDED
    assert settled.status is VerificationStatus.SUCCEEDED


def test_lifted_is_satisfied_by_a_past_lift_event_before_final_placement() -> None:
    evidence = _evidence(
        initial=_frame(0.0, item=_state((0.0, 0.0, 0.1))),
        frames=(
            _frame(0.1, item=_state((0.0, 0.0, 0.2))),
            _frame(0.25, item=_state((0.0, 0.0, 0.1))),
        ),
    )
    report = PredicateVerifier().verify(
        _spec("lifted", {"subject": "item"}),
        evidence,
        VerificationPolicy(lift_displacement_m=0.05),
    )

    assert report.status is VerificationStatus.SUCCEEDED


def test_on_support_rejects_surface_without_contract_fields() -> None:
    with pytest.raises((TypeError, ValueError), match="surface"):
        _spec(
            "on_support",
            {
                "subject": "item",
                "support": "support",
                "surface": {"name": "top"},
                "subject_half_height_m": 0.05,
            },
        )


    support = _state((1.0, 2.0, 0.4))
    item = _state((1.1, 2.0, 0.55))
    evidence = _evidence(
        initial=_frame(0.0, item=item, support=support),
        frames=(
            _frame(0.1, item=item, support=support),
            _frame(0.25, item=item, support=support),
        ),
        contacts=(
            ContactEvent(0.1, "item", "support", 1.0),
            ContactEvent(0.2, "support", "item", 1.0),
        ),
    )
    report = PredicateVerifier().verify(
        _spec(
            "on_support",
            {
                "subject": "item",
                "support": "support",
                "surface": {
                    "center": [0.0, 0.0, 0.1],
                    "size": [0.5, 0.5],
                },
                "subject_half_height_m": 0.05,
            },
        ),
        evidence,
        VerificationPolicy(
            support_height_tolerance_m=0.01,
            contact_duration_s=0.05,
            settled_duration_s=0.1,
        ),
    )

    assert report.status is VerificationStatus.SUCCEEDED


def test_collision_free_invariant_fails_and_incomplete_collision_evidence_is_unknown() -> None:
    collision = ContactEvent(0.1, "robot", "wall", 4.0)
    failed = PredicateVerifier().verify(
        _spec("collision_free", {"subject": "robot"}, invariant=True),
        _evidence(frames=(_frame(0.2, item=_state()),), collisions=(collision,)),
        VerificationPolicy(),
    )
    incomplete = PredicateVerifier().verify(
        _spec("collision_free", {"subject": "robot"}),
        _evidence(complete=False),
        VerificationPolicy(),
    )

    assert failed.status is VerificationStatus.FAILED
    assert failed.predicates[-1].status is PredicateStatus.VIOLATED
    assert incomplete.predicates[0].status is PredicateStatus.UNKNOWN


def test_collision_free_is_scoped_to_subject_entity() -> None:
    unrelated = ContactEvent(0.1, "other_object", "wall", 4.0)
    report = PredicateVerifier().verify(
        _spec("collision_free", {"subject": "item"}),
        _evidence(frames=(_frame(0.2, item=_state()),), collisions=(unrelated,)),
        VerificationPolicy(),
    )

    assert report.status is VerificationStatus.SUCCEEDED


def test_collision_free_is_unknown_when_collision_telemetry_is_unavailable() -> None:
    evidence = EvidenceBundle(
        initial=_frame(0.0, item=_state()),
        frames=(_frame(0.1, item=_state()),),
        complete=True,
        collision_observation_complete=False,
    )
    report = PredicateVerifier().verify(
        _spec("collision_free", {"subject": "robot"}),
        evidence,
        VerificationPolicy(),
    )

    assert report.status is VerificationStatus.INCOMPLETE
    assert report.predicates[0].status is PredicateStatus.UNKNOWN


def test_within_tolerance_reads_observed_entity_field_not_goal_observation() -> None:
    report = PredicateVerifier().verify(
        _spec(
            "within_tolerance",
            {
                "subject": "item",
                "field": "position",
                "target": [0.0, 0.0, 0.2],
                "tolerance": 0.02,
            },
        ),
        _evidence(frames=(_frame(0.1, item=_state((0.0, 0.0, 0.21))),)),
        VerificationPolicy(max_explicit_tolerance_m=0.05),
    )

    assert report.status is VerificationStatus.SUCCEEDED
