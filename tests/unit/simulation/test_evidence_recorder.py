from __future__ import annotations

import math

import pytest

from r1pro_data_gen.domain import (
    AttachmentEvent,
    ContactEvent,
    EntityState,
    EvidenceBundle,
    EvidenceFrame,
    ObjectModel,
    ObjectType,
    RobotModel,
    SceneModel,
    WorldModel,
    evidence_from_dict,
    evidence_to_dict,
)
from r1pro_data_gen.simulation import EvidenceRecorder


def _entity() -> EntityState:
    return EntityState(
        position=(0.1, 0.2, 0.3),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        source="live",
    )


def _frame(timestamp: float) -> EvidenceFrame:
    return EvidenceFrame(
        timestamp=timestamp,
        base_pose=(0.0, 0.0, 0.0),
        base_velocity=(0.0, 0.0, 0.0),
        joint_positions={"joint": 0.1},
        joint_velocities={"joint": 0.0},
        end_effectors={"left_ee": (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0)},
        entities={"item": _entity()},
        attachments={"item": "left_gripper"},
        stage="move",
    )


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        initial=_frame(0.0),
        frames=(_frame(0.1), _frame(0.2)),
        contacts=(
            ContactEvent(
                timestamp=0.15,
                body_a="left_finger",
                body_b="item",
                force_n=1.2,
                point=(0.1, 0.2, 0.3),
                normal=(0.0, 0.0, 1.0),
            ),
        ),
        collisions=(),
        attachment_events=(
            AttachmentEvent(
                timestamp=0.12,
                entity="item",
                effector="left_gripper",
                attached=True,
            ),
        ),
        stage_windows={"move": (0.1, 0.2)},
        stage_status={"move": True},
        complete=True,
    )


def test_evidence_bundle_rejects_non_monotonic_frames() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        EvidenceBundle(
            initial=_frame(0.0),
            frames=(_frame(0.2), _frame(0.1)),
        )


def test_evidence_bundle_requires_initial_before_continuous_frames() -> None:
    with pytest.raises(ValueError, match="after initial"):
        EvidenceBundle(initial=_frame(0.2), frames=(_frame(0.1),))


def test_contact_event_requires_two_distinct_bodies() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ContactEvent(
            timestamp=0.0,
            body_a="item",
            body_b="item",
            force_n=1.0,
        )


def test_evidence_rejects_invalid_identity_and_numeric_values() -> None:
    with pytest.raises(ValueError, match="entity"):
        AttachmentEvent(
            timestamp=0.0,
            entity="",
            effector="left_gripper",
            attached=True,
        )


def test_physical_metrics_preserve_scalar_diagnostic_labels() -> None:
    frame = EvidenceFrame(
        timestamp=0.0,
        base_pose=(0.0, 0.0, 0.0),
        base_velocity=(0.0, 0.0, 0.0),
        joint_positions={"joint": 0.1},
        joint_velocities={"joint": 0.0},
        end_effectors={},
        entities={},
        attachments={},
        physical_metrics={
            "max_effort_joint": "right_arm_joint1",
            "max_effort_source": "applied_torque",
            "within_effort_limit": False,
        },
    )

    payload = evidence_to_dict(EvidenceBundle(initial=frame))
    assert payload["initial"]["physical_metrics"]["max_effort_joint"] == "right_arm_joint1"
    assert evidence_to_dict(evidence_from_dict(payload)) == payload
    with pytest.raises(ValueError, match="finite"):
        EntityState(
            position=(math.inf, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )


def test_evidence_bundle_rejects_invalid_stage_window() -> None:
    with pytest.raises(ValueError, match="stage window"):
        EvidenceBundle(
            initial=_frame(0.0),
            frames=(_frame(0.1),),
            stage_windows={"move": (0.2, 0.1)},
        )


def test_evidence_bundle_rejects_status_without_stage_window() -> None:
    with pytest.raises(ValueError, match="stage status"):
        EvidenceBundle(
            initial=_frame(0.0),
            frames=(_frame(0.1),),
            stage_status={"never_run": False},
        )


    payload = evidence_to_dict(_bundle())

    assert evidence_to_dict(evidence_from_dict(payload)) == payload
    assert payload["initial"]["entities"]["item"]["source"] == "live"
    assert "task_success" not in str(payload)
    assert "grasp_success" not in str(payload)
    assert "evaluator" not in str(payload)


class _FakeAdapter:
    def __init__(self, scene: SceneModel) -> None:
        self.scene_model = scene
        self._attached: dict[str, str] = {}
        self._collision_complete = True

    def collision_events(self):
        return ()

    @property
    def collision_observation_complete(self):
        return self._collision_complete

    def read_observation(self, timestamp: float):
        from r1pro_data_gen.domain import Observation

        return Observation(
            timestamp=timestamp,
            joint_positions={"joint": 0.1},
            joint_velocities={"joint": 0.0},
            base_pose=(0.0, 0.0, 0.0),
        )

    def all_object_states(self):
        return {
            obj.name: EntityState(
                position=obj.pos,
                quaternion=obj.quat,
                linear_velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                source="live" if obj.name == "item_a" else "declared",
            )
            for obj in self.scene_model.objects
        }

    def end_effector_poses(self):
        return {"left_ee": (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0)}

    def contact_events(self):
        return ()

    def attachment_state(self):
        return dict(self._attached)


def _scene_with_objects(*names: str) -> SceneModel:
    return SceneModel(
        name="recorder_scene",
        world=WorldModel(),
        robot=RobotModel(asset="asset/robot.usda"),
        objects=tuple(
            ObjectModel(
                name=name,
                type=ObjectType.CUBOID,
                pos=(float(index), 0.0, 0.1),
                size=(0.1, 0.1, 0.1),
            )
            for index, name in enumerate(names)
        ),
    )


def test_recorder_replaces_same_timestamp_after_existing_frame() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.1, stage="previous")
    recorder.capture(0.1, stage="move")
    recorder.finish_stage(0.1, "move", success=True)
    bundle = recorder.finish(complete=True, expected_stages=("move",))

    assert len(bundle.frames) == 1
    assert bundle.frames[0].timestamp == 0.1
    assert bundle.frames[0].stage == "move"
    assert bundle.complete


def test_recorder_decimates_continuous_frames_but_allows_forced_boundaries() -> None:
    scene = _scene_with_objects("item")
    recorder = EvidenceRecorder(_FakeAdapter(scene), scene)

    recorder.capture(0.0)
    assert recorder.capture_if_due(
        0.05,
        stage="move",
        min_interval_s=0.1,
    ) is None
    assert recorder.capture_if_due(
        0.1,
        stage="move",
        min_interval_s=0.1,
    ) is not None
    recorder.capture(0.15, stage="move")
    recorder.finish_stage(0.15, "move", success=True)
    bundle = recorder.finish(complete=True, expected_stages=("move",))

    assert [frame.timestamp for frame in bundle.frames] == [0.1, 0.15]
    assert bundle.stage_windows == {"move": (0.1, 0.15)}
    assert bundle.complete


@pytest.mark.parametrize("interval", [-0.1, math.inf, math.nan])
def test_recorder_rejects_invalid_sampling_interval(interval: float) -> None:
    scene = _scene_with_objects("item")
    recorder = EvidenceRecorder(_FakeAdapter(scene), scene)

    with pytest.raises(ValueError, match="min_interval_s"):
        recorder.capture_if_due(0.0, min_interval_s=interval)


def test_recorder_deduplicates_same_timestamp_stage_boundary() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.0, stage="move")
    recorder.capture(0.1, stage="move")
    recorder.finish_stage(0.1, "move", success=True)
    bundle = recorder.finish(complete=True, expected_stages=("move",))

    assert len(bundle.frames) == 1
    assert bundle.stage_windows == {"move": (0.0, 0.1)}
    assert bundle.complete


def test_recorder_captures_every_scene_object_without_task_configuration() -> None:
    scene = _scene_with_objects("item_a", "support_b", "obstacle_c")
    recorder = EvidenceRecorder(_FakeAdapter(scene), scene)

    initial = recorder.capture(0.0)
    recorder.capture(0.1, stage="observe")
    bundle = recorder.finish(complete=True)

    assert set(initial.entities) == {"item_a", "support_b", "obstacle_c"}
    assert set(bundle.initial.entities) == {"item_a", "support_b", "obstacle_c"}
    assert bundle.frames[0].stage == "observe"
    assert bundle.stage_windows == {"observe": (0.1, 0.1)}
    assert bundle.complete


def test_recorder_does_not_require_collision_coverage_for_non_collision_goal() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    adapter._collision_complete = False
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.1, stage="observe")
    recorder.finish_stage(0.1, "observe", success=True)
    bundle = recorder.finish(
        complete=True,
        expected_stages=("observe",),
        require_collision_observation=False,
    )

    assert bundle.complete
    assert not bundle.collision_observation_complete


def test_recorder_requires_collision_coverage_for_collision_goal() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    adapter._collision_complete = False
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.1, stage="observe")
    recorder.finish_stage(0.1, "observe", success=True)
    bundle = recorder.finish(
        complete=True,
        expected_stages=("observe",),
        require_collision_observation=True,
    )

    assert not bundle.complete
    assert not bundle.collision_observation_complete


    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    collision = ContactEvent(0.1, "robot", "obstacle", 2.0)
    adapter.collision_events = lambda: (collision,)
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.1, stage="move")
    recorder.finish_stage(0.1, "move", success=True)
    bundle = recorder.finish(complete=True, expected_stages=("move",))

    assert bundle.collisions == (collision,)
    assert bundle.complete


def test_recorder_closes_failed_stage_without_fabricating_unexecuted_stage() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.1, stage="failed")
    recorder.finish_stage(0.1, "failed", success=False)
    bundle = recorder.finish(complete=False, expected_stages=("failed", "never_run"))

    assert "failed" in bundle.stage_windows
    assert "never_run" not in bundle.stage_windows
    assert bundle.stage_status == {"failed": False}
    assert not bundle.complete


def test_recorder_keeps_failed_stage_fail_closed_even_if_completion_is_requested() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.1, stage="failed")
    recorder.finish_stage(0.1, "failed", success=False)
    bundle = recorder.finish(complete=True, expected_stages=("failed",))

    assert bundle.stage_status == {"failed": False}
    assert bundle.complete
    assert not bundle.stage_success_complete


def test_recorder_records_successful_stage_status() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    recorder.capture(0.1, stage="move")
    recorder.finish_stage(0.1, "move", success=True)
    bundle = recorder.finish(complete=True, expected_stages=("move",))

    assert bundle.stage_status == {"move": True}
    assert bundle.complete


def test_recorder_tracks_attachment_transitions() -> None:
    scene = _scene_with_objects("item")
    adapter = _FakeAdapter(scene)
    recorder = EvidenceRecorder(adapter, scene)

    recorder.capture(0.0)
    adapter._attached["item"] = "left_gripper"
    recorder.capture(0.1, stage="grasp")
    adapter._attached.clear()
    recorder.capture(0.2, stage="release")
    events = recorder.finish().attachment_events

    assert events == (
        AttachmentEvent(0.1, "item", "left_gripper", True),
        AttachmentEvent(0.2, "item", "left_gripper", False),
    )
