"""Task-independent collection of simulator state and events."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Protocol

from r1pro_data_gen.domain import (
    AttachmentEvent,
    ContactEvent,
    EntityState,
    EvidenceBundle,
    EvidenceFrame,
    SceneModel,
)


class EvidenceAdapter(Protocol):
    """The observation-only adapter surface required by EvidenceRecorder."""

    def read_observation(self, timestamp: float): ...

    def all_object_states(self) -> Mapping[str, EntityState]: ...

    def end_effector_poses(self) -> Mapping[str, tuple[float, ...]]: ...

    def contact_events(self) -> tuple[ContactEvent, ...]: ...

    def collision_events(self) -> tuple[ContactEvent, ...]: ...

    @property
    def collision_observation_complete(self) -> bool: ...

    def attachment_state(self) -> Mapping[str, str]: ...


class EvidenceRecorder:
    """Collect synchronized facts for every object declared by a scene."""

    def __init__(self, adapter: EvidenceAdapter, scene: SceneModel) -> None:
        self._adapter = adapter
        self._scene = scene
        self._initial: EvidenceFrame | None = None
        self._frames: list[EvidenceFrame] = []
        self._contacts: list[ContactEvent] = []
        self._collisions: list[ContactEvent] = []
        self._attachment_events: list[AttachmentEvent] = []
        self._previous_attachments: dict[str, str] = {}
        self._stage_windows: dict[str, tuple[float, float]] = {}
        self._stage_status: dict[str, bool] = {}
        self._last_capture_timestamp: float | None = None

    def capture_if_due(
        self,
        timestamp: float,
        stage: str | None = None,
        *,
        min_interval_s: float,
    ) -> EvidenceFrame | None:
        """Capture at a bounded cadence while leaving boundaries forceable.

        Long navigation stages can contain thousands of physics steps.  A full
        evidence frame performs several device-to-host synchronizations, so
        recording one on every step makes replay duration and output size grow
        unnecessarily.  Callers use :meth:`capture` directly at stage
        boundaries and this method for continuous sampling between them.
        """
        interval = float(min_interval_s)
        if not math.isfinite(interval) or interval < 0.0:
            raise ValueError("min_interval_s must be finite and non-negative")
        current = float(timestamp)
        if not math.isfinite(current):
            raise ValueError("evidence timestamp must be finite")
        if (
            interval > 0.0
            and self._last_capture_timestamp is not None
            and current - self._last_capture_timestamp < interval - 1e-9
        ):
            return None
        return self.capture(current, stage=stage)

    def capture(self, timestamp: float, stage: str | None = None) -> EvidenceFrame:
        """Capture one synchronized frame and append newly observed events."""
        observation = self._adapter.read_observation(timestamp)
        entities = dict(self._adapter.all_object_states())
        expected = {obj.name for obj in self._scene.objects}
        missing = expected - set(entities)
        if missing:
            raise RuntimeError(
                f"evidence adapter omitted scene objects: {sorted(missing)}"
            )
        attachments = dict(self._adapter.attachment_state())
        frame = EvidenceFrame(
            timestamp=timestamp,
            base_pose=_base_vector(observation.base_pose, "base_pose"),
            base_velocity=_base_velocity(observation),
            joint_positions=observation.joint_positions,
            joint_velocities=observation.joint_velocities,
            end_effectors=self._adapter.end_effector_poses(),
            entities=entities,
            attachments=attachments,
            stage=stage,
            base_orientation=_optional_vector(
                getattr(observation, "base_orientation", None),
                4,
            ),
            base_height_m=_optional_number(
                getattr(observation, "base_height_m", None),
            ),
            imu_linear_acceleration=_optional_vector(
                getattr(observation, "imu_linear_acceleration", None),
                3,
            ),
            imu_angular_velocity=_optional_vector(
                getattr(observation, "imu_angular_velocity", None),
                3,
            ),
            support_contacts=getattr(observation, "support_contacts", {}) or {},
            joint_efforts=getattr(observation, "joint_efforts", {}) or {},
            physical_metrics=getattr(observation, "physical_metrics", {}) or {},
        )
        if self._initial is None:
            self._initial = frame
        elif timestamp == self._initial.timestamp and not self._frames:
            if stage is not None:
                self._initial = EvidenceFrame(
                    timestamp=self._initial.timestamp,
                    base_pose=self._initial.base_pose,
                    base_velocity=self._initial.base_velocity,
                    joint_positions=self._initial.joint_positions,
                    joint_velocities=self._initial.joint_velocities,
                    end_effectors=self._initial.end_effectors,
                    entities=self._initial.entities,
                    attachments=self._initial.attachments,
                    stage=stage,
                    base_orientation=self._initial.base_orientation,
                    base_height_m=self._initial.base_height_m,
                    imu_linear_acceleration=self._initial.imu_linear_acceleration,
                    imu_angular_velocity=self._initial.imu_angular_velocity,
                    support_contacts=self._initial.support_contacts,
                    joint_efforts=self._initial.joint_efforts,
                    physical_metrics=self._initial.physical_metrics,
                )
        elif self._frames and timestamp == self._frames[-1].timestamp:
            previous = self._frames[-1]
            self._frames[-1] = EvidenceFrame(
                timestamp=timestamp,
                base_pose=frame.base_pose,
                base_velocity=frame.base_velocity,
                joint_positions=frame.joint_positions,
                joint_velocities=frame.joint_velocities,
                end_effectors=frame.end_effectors,
                entities=frame.entities,
                attachments=frame.attachments,
                stage=stage if stage is not None else previous.stage,
                base_orientation=frame.base_orientation,
                base_height_m=frame.base_height_m,
                imu_linear_acceleration=frame.imu_linear_acceleration,
                imu_angular_velocity=frame.imu_angular_velocity,
                support_contacts=frame.support_contacts,
                joint_efforts=frame.joint_efforts,
                physical_metrics=frame.physical_metrics,
            )
        else:
            self._frames.append(frame)
        self._record_stage_window(frame)
        self._record_contacts()
        self._record_collisions()
        self._record_attachment_transitions(timestamp, attachments)
        self._last_capture_timestamp = float(timestamp)
        return frame

    def finish_stage(self, timestamp: float, stage: str, *, success: bool) -> None:
        """Close a stage window from an observed execution boundary."""
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(success, bool):
            raise TypeError("stage success must be a boolean")
        window = self._stage_windows.get(stage)
        if window is None:
            return
        if timestamp < window[0]:
            raise ValueError("stage end timestamp must not precede stage start")
        self._stage_windows[stage] = (window[0], timestamp)
        self._stage_status[stage] = success

    def finish(
        self,
        complete: bool = False,
        *,
        expected_stages: tuple[str, ...] = (),
        require_collision_observation: bool = True,
    ) -> EvidenceBundle:
        """Freeze facts after checking observation coverage.

        Stage success is deliberately not part of ``EvidenceBundle.complete``.
        It is preserved as ``stage_success_complete`` so the final evaluator
        can accept a physically satisfied goal while still reporting that an
        intermediate skill returned failure.
        """
        if self._initial is None:
            raise RuntimeError("evidence recorder requires an initial capture")
        if not isinstance(require_collision_observation, bool):
            raise TypeError("require_collision_observation must be a boolean")
        covered_stages = set(self._stage_windows)
        stages_complete = set(expected_stages).issubset(covered_stages)
        stage_results_complete = all(
            self._stage_status.get(stage_name) is True
            for stage_name in expected_stages
        )
        collision_observation_complete = bool(
            self._adapter.collision_observation_complete
        )
        complete = bool(
            complete
            and stages_complete
            and (
                not require_collision_observation
                or collision_observation_complete
            )
        )
        return EvidenceBundle(
            initial=self._initial,
            frames=tuple(self._frames),
            contacts=tuple(self._contacts),
            collisions=tuple(self._collisions),
            attachment_events=tuple(self._attachment_events),
            stage_windows=self._stage_windows,
            stage_status=self._stage_status,
            complete=complete,
            stage_success_complete=bool(stages_complete and stage_results_complete),
            collision_observation_complete=collision_observation_complete,
        )

    def _record_stage_window(self, frame: EvidenceFrame) -> None:
        if frame.stage is None:
            return
        previous = self._stage_windows.get(frame.stage)
        start = frame.timestamp if previous is None else previous[0]
        self._stage_windows[frame.stage] = (start, frame.timestamp)

    def _record_contacts(self) -> None:
        for event in self._adapter.contact_events():
            if event not in self._contacts:
                self._contacts.append(event)

    def _record_collisions(self) -> None:
        for event in self._adapter.collision_events():
            if event not in self._collisions:
                self._collisions.append(event)

    def _record_attachment_transitions(
        self,
        timestamp: float,
        attachments: Mapping[str, str],
    ) -> None:
        for entity, previous_effector in self._previous_attachments.items():
            current_effector = attachments.get(entity)
            if current_effector != previous_effector:
                self._attachment_events.append(
                    AttachmentEvent(
                        timestamp=timestamp,
                        entity=entity,
                        effector=previous_effector,
                        attached=False,
                    )
                )
        for entity, effector in attachments.items():
            if self._previous_attachments.get(entity) != effector:
                self._attachment_events.append(
                    AttachmentEvent(
                        timestamp=timestamp,
                        entity=entity,
                        effector=effector,
                        attached=True,
                    )
                )
        self._previous_attachments = dict(attachments)


def _base_vector(value: object, what: str) -> tuple[float, float, float]:
    if value is None:
        raise RuntimeError(f"evidence adapter did not provide {what}")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise RuntimeError(f"evidence adapter {what} must contain 3 values")
    return values


def _base_velocity(observation: object) -> tuple[float, float, float]:
    raw = getattr(observation, "base_velocity", None)
    if raw is None:
        return (0.0, 0.0, 0.0)
    return _base_vector(raw, "base_velocity")


def _optional_vector(value: object, length: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != length:
        return None
    return values


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


__all__ = ["EvidenceAdapter", "EvidenceRecorder"]
