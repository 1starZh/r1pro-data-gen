"""Domain data structures shared by planning, control and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Mapping


class ControlMode(StrEnum):
    """Supported low-level command semantics."""

    POSITION = "position"
    VELOCITY = "velocity"


class TaskStatus(StrEnum):
    """Lifecycle status for a planned task or rollout."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PlanStage:
    """One semantic stage in a task plan."""

    name: str
    goal: str
    depends_on: tuple[str, ...] = ()
    parameters: Mapping[str, object] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    preconditions: tuple[Mapping[str, object], ...] = ()
    postconditions: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage name must not be empty")
        if not self.goal.strip():
            raise ValueError("stage goal must not be empty")
        if any(not isinstance(name, str) or not name.strip() for name in self.outputs):
            raise ValueError("stage outputs must contain non-empty strings")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("stage outputs must be unique")


@dataclass(frozen=True, slots=True)
class Plan:
    """Structured task intent; it does not contain low-level commands."""

    task_name: str
    stages: tuple[PlanStage, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("plan stage names must be unique")
        known = set(names)
        for stage in self.stages:
            missing = set(stage.depends_on) - known
            if missing:
                raise ValueError(
                    f"stage {stage.name!r} depends on unknown stages: {sorted(missing)}"
                )

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One time-sampled reference point in a unified trajectory."""

    timestamp: float
    joint_positions: Mapping[str, float] = field(default_factory=dict)
    joint_velocities: Mapping[str, float] = field(default_factory=dict)
    joint_accelerations: Mapping[str, float] = field(default_factory=dict)
    base_pose: tuple[float, ...] | None = None
    base_velocity: tuple[float, ...] | None = None
    gripper: float | None = None
    stage: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("trajectory timestamp must be non-negative")
        if self.base_pose is not None and len(self.base_pose) != 3:
            raise ValueError("base_pose must be (x, y, yaw)")
        if self.base_velocity is not None and len(self.base_velocity) != 3:
            raise ValueError("base_velocity must be (vx, vy, wz)")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Monotonic, time-parameterized reference trajectory."""

    joint_names: tuple[str, ...]
    points: tuple[TrajectoryPoint, ...]
    planner: str = "unknown"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("trajectory joint_names must be unique")
        if any(not name.strip() for name in self.joint_names):
            raise ValueError("trajectory joint names must not be empty")
        timestamps = [point.timestamp for point in self.points]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("trajectory timestamps must be strictly increasing")
        allowed = set(self.joint_names)
        for point in self.points:
            for values in (
                point.joint_positions,
                point.joint_velocities,
                point.joint_accelerations,
            ):
                unknown = set(values) - allowed
                if unknown:
                    raise ValueError(f"trajectory contains unknown joints: {sorted(unknown)}")

    @property
    def duration(self) -> float:
        return 0.0 if not self.points else self.points[-1].timestamp

    def sample(self, timestamp: float) -> TrajectoryPoint:
        """Return the latest point at or before ``timestamp``."""
        if not self.points:
            raise ValueError("cannot sample an empty trajectory")
        if timestamp < 0:
            raise ValueError("sample timestamp must be non-negative")
        point = self.points[0]
        for candidate in self.points:
            if candidate.timestamp > timestamp:
                break
            point = candidate
        return point


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """A command sent for one control step, separate from actual state."""

    timestamp: float
    mode_by_group: Mapping[str, ControlMode]
    position_targets: Mapping[str, float] = field(default_factory=dict)
    velocity_targets: Mapping[str, float] = field(default_factory=dict)
    gripper_target: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("command timestamp must be non-negative")
        for joint in self.position_targets:
            if joint in self.velocity_targets:
                raise ValueError(f"joint has both position and velocity targets: {joint}")


@dataclass(frozen=True, slots=True)
class Observation:
    """Actual feedback returned by a simulator or another execution backend."""

    timestamp: float
    joint_positions: Mapping[str, float] = field(default_factory=dict)
    joint_velocities: Mapping[str, float] = field(default_factory=dict)
    base_pose: tuple[float, ...] | None = None
    base_velocity: tuple[float, ...] | None = None
    end_effector_pose: tuple[float, ...] | None = None
    contacts: tuple[object, ...] = ()
    object_states: Mapping[str, object] = field(default_factory=dict)
    # Optional physical-integrity telemetry.  The original (x, y, yaw)
    # interface remains stable for planners and legacy adapters; these fields
    # expose the measurements needed to reject an apparently successful but
    # dynamically invalid manipulation.
    base_orientation: tuple[float, ...] | None = None
    base_height_m: float | None = None
    imu_linear_acceleration: tuple[float, ...] | None = None
    imu_angular_velocity: tuple[float, ...] | None = None
    support_contacts: Mapping[str, float] = field(default_factory=dict)
    joint_efforts: Mapping[str, float] = field(default_factory=dict)
    physical_metrics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("observation timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    """Structured evidence for a failed or near-miss rollout."""

    category: str
    stage: str | None
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.category.strip() or not self.reason.strip():
            raise ValueError("failure category and reason must not be empty")


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Evaluator result, based on execution observations rather than planner status."""

    status: TaskStatus
    task_name: str
    completed_stages: tuple[str, ...] = ()
    failure: FailureEvidence | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty")
        if self.status == TaskStatus.FAILED and self.failure is None:
            raise ValueError("failed results require failure evidence")
        if self.status == TaskStatus.SUCCEEDED and self.failure is not None:
            raise ValueError("succeeded results cannot contain failure evidence")


__all__ = [
    "ControlCommand",
    "ControlMode",
    "FailureEvidence",
    "Observation",
    "Plan",
    "PlanStage",
    "TaskResult",
    "TaskStatus",
    "Trajectory",
    "TrajectoryPoint",
]
