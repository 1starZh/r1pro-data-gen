"""Immutable, task-independent execution evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EntityState:
    """Observed world-frame state for one scene entity."""

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    source: str = "live"

    def __post_init__(self) -> None:
        _validate_vector(self.position, 3, "entity position")
        _validate_vector(self.quaternion, 4, "entity quaternion")
        _validate_vector(self.linear_velocity, 3, "entity linear_velocity")
        _validate_vector(self.angular_velocity, 3, "entity angular_velocity")
        _validate_identity(self.source, "entity source")


@dataclass(frozen=True, slots=True)
class ContactEvent:
    """Observed contact between two identified bodies."""

    timestamp: float
    body_a: str
    body_b: str
    force_n: float
    point: tuple[float, float, float] | None = None
    normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp, "contact timestamp")
        _validate_identity(self.body_a, "contact body_a")
        _validate_identity(self.body_b, "contact body_b")
        if self.body_a == self.body_b:
            raise ValueError("contact bodies must be distinct")
        _validate_non_negative(self.force_n, "contact force_n")
        if self.point is not None:
            _validate_vector(self.point, 3, "contact point")
        if self.normal is not None:
            _validate_vector(self.normal, 3, "contact normal")


@dataclass(frozen=True, slots=True)
class AttachmentEvent:
    """Observed attachment or detachment of an entity and an effector."""

    timestamp: float
    entity: str
    effector: str
    attached: bool

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp, "attachment timestamp")
        _validate_identity(self.entity, "attachment entity")
        _validate_identity(self.effector, "attachment effector")
        if not isinstance(self.attached, bool):
            raise TypeError("attachment attached must be a boolean")


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    """One synchronized snapshot of generic robot and scene state."""

    timestamp: float
    base_pose: tuple[float, float, float]
    base_velocity: tuple[float, float, float]
    joint_positions: Mapping[str, float]
    joint_velocities: Mapping[str, float]
    end_effectors: Mapping[str, tuple[float, ...]]
    entities: Mapping[str, EntityState]
    attachments: Mapping[str, str]
    stage: str | None = None
    base_orientation: tuple[float, ...] | None = None
    base_height_m: float | None = None
    imu_linear_acceleration: tuple[float, ...] | None = None
    imu_angular_velocity: tuple[float, ...] | None = None
    support_contacts: Mapping[str, float] = field(default_factory=dict)
    joint_efforts: Mapping[str, float] = field(default_factory=dict)
    # JSON scalar telemetry: numeric values, booleans, and optional labels
    # such as the name of the joint that reached the largest effort.  Keeping
    # labels in the same immutable snapshot avoids losing the diagnostic
    # identity when a physical gate trips.
    physical_metrics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp, "frame timestamp")
        _validate_vector(self.base_pose, 3, "frame base_pose")
        _validate_vector(self.base_velocity, 3, "frame base_velocity")
        object.__setattr__(
            self,
            "joint_positions",
            _freeze_numeric_mapping(self.joint_positions, "joint_positions"),
        )
        object.__setattr__(
            self,
            "joint_velocities",
            _freeze_numeric_mapping(self.joint_velocities, "joint_velocities"),
        )
        object.__setattr__(
            self,
            "end_effectors",
            _freeze_pose_mapping(self.end_effectors),
        )
        object.__setattr__(self, "entities", _freeze_entity_mapping(self.entities))
        object.__setattr__(
            self,
            "attachments",
            _freeze_identity_mapping(self.attachments, "attachments"),
        )
        if self.stage is not None:
            _validate_identity(self.stage, "frame stage")
        if self.base_orientation is not None:
            object.__setattr__(
                self,
                "base_orientation",
                _vector(self.base_orientation, 4, "frame base_orientation"),
            )
        if self.base_height_m is not None:
            object.__setattr__(
                self,
                "base_height_m",
                _number(self.base_height_m, "frame base_height_m"),
            )
        if self.imu_linear_acceleration is not None:
            object.__setattr__(
                self,
                "imu_linear_acceleration",
                _vector(
                    self.imu_linear_acceleration,
                    3,
                    "frame imu_linear_acceleration",
                ),
            )
        if self.imu_angular_velocity is not None:
            object.__setattr__(
                self,
                "imu_angular_velocity",
                _vector(self.imu_angular_velocity, 3, "frame imu_angular_velocity"),
            )
        object.__setattr__(
            self,
            "support_contacts",
            _freeze_numeric_mapping(self.support_contacts, "support_contacts"),
        )
        object.__setattr__(
            self,
            "joint_efforts",
            _freeze_numeric_mapping(self.joint_efforts, "joint_efforts"),
        )
        object.__setattr__(
            self,
            "physical_metrics",
            _freeze_scalar_mapping(self.physical_metrics, "physical_metrics"),
        )


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Initial state, continuous frames, and event facts for one episode.

    ``complete`` describes observation coverage, not whether every skill
    succeeded.  A physically successful episode may contain a failed
    intermediate skill (for example, a carry call that missed its local
    target while a later release left the object in the goal region).  Keep
    that execution fact separately in ``stage_success_complete``.
    """

    initial: EvidenceFrame
    frames: tuple[EvidenceFrame, ...] = ()
    contacts: tuple[ContactEvent, ...] = ()
    collisions: tuple[ContactEvent, ...] = ()
    attachment_events: tuple[AttachmentEvent, ...] = ()
    stage_windows: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    stage_status: Mapping[str, bool] = field(default_factory=dict)
    complete: bool = False
    stage_success_complete: bool = False
    collision_observation_complete: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.initial, EvidenceFrame):
            raise TypeError("evidence initial must be an EvidenceFrame")
        frames = tuple(self.frames)
        if any(not isinstance(frame, EvidenceFrame) for frame in frames):
            raise TypeError("evidence frames must contain EvidenceFrame values")
        timestamps = [self.initial.timestamp, *(frame.timestamp for frame in frames)]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            if frames and frames[0].timestamp <= self.initial.timestamp:
                raise ValueError("evidence frames must occur after initial frame")
            raise ValueError("evidence frame timestamps must be strictly increasing")
        contacts = _typed_tuple(self.contacts, ContactEvent, "contacts")
        collisions = _typed_tuple(self.collisions, ContactEvent, "collisions")
        attachment_events = _typed_tuple(
            self.attachment_events,
            AttachmentEvent,
            "attachment_events",
        )
        if not isinstance(self.complete, bool):
            raise TypeError("evidence complete must be a boolean")
        if not isinstance(self.stage_success_complete, bool):
            raise TypeError("evidence stage_success_complete must be a boolean")
        if not isinstance(self.collision_observation_complete, bool):
            raise TypeError("evidence collision_observation_complete must be a boolean")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "contacts", contacts)
        object.__setattr__(self, "collisions", collisions)
        object.__setattr__(self, "attachment_events", attachment_events)
        stage_windows = _freeze_stage_windows(self.stage_windows)
        stage_status = _freeze_stage_status(self.stage_status)
        if set(stage_status) - set(stage_windows):
            raise ValueError("stage status requires a corresponding stage window")
        object.__setattr__(self, "stage_windows", stage_windows)
        object.__setattr__(self, "stage_status", stage_status)


def evidence_to_dict(bundle: EvidenceBundle) -> dict[str, object]:
    """Serialize an EvidenceBundle to its JSON-compatible public shape."""
    if not isinstance(bundle, EvidenceBundle):
        raise TypeError("bundle must be an EvidenceBundle")
    return {
        "initial": _frame_to_dict(bundle.initial),
        "frames": [_frame_to_dict(frame) for frame in bundle.frames],
        "contacts": [_contact_to_dict(event) for event in bundle.contacts],
        "collisions": [_contact_to_dict(event) for event in bundle.collisions],
        "attachment_events": [
            {
                "timestamp": event.timestamp,
                "entity": event.entity,
                "effector": event.effector,
                "attached": event.attached,
            }
            for event in bundle.attachment_events
        ],
        "stage_windows": {
            name: [window[0], window[1]]
            for name, window in bundle.stage_windows.items()
        },
        "stage_status": dict(bundle.stage_status),
        "complete": bundle.complete,
        "stage_success_complete": bundle.stage_success_complete,
        "collision_observation_complete": bundle.collision_observation_complete,
    }


def evidence_from_dict(data: Mapping[str, object]) -> EvidenceBundle:
    """Parse the strict public EvidenceBundle shape."""
    _require_mapping(data, "evidence")
    required_fields = {
        "initial",
        "frames",
        "contacts",
        "collisions",
        "attachment_events",
        "stage_windows",
        "stage_status",
        "complete",
        "collision_observation_complete",
    }
    optional_fields = {"stage_success_complete"}
    unknown = set(data) - required_fields - optional_fields
    missing = required_fields - set(data)
    if unknown:
        raise ValueError(f"evidence contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"evidence is missing fields: {sorted(missing)}")
    initial = _frame_from_dict(data["initial"], "evidence initial")
    frames = tuple(
        _frame_from_dict(item, f"evidence frames[{index}]")
        for index, item in enumerate(_require_array(data["frames"], "evidence frames"))
    )
    contacts = tuple(
        _contact_from_dict(item, f"evidence contacts[{index}]")
        for index, item in enumerate(_require_array(data["contacts"], "evidence contacts"))
    )
    collisions = tuple(
        _contact_from_dict(item, f"evidence collisions[{index}]")
        for index, item in enumerate(_require_array(data["collisions"], "evidence collisions"))
    )
    attachment_events = tuple(
        _attachment_from_dict(item, f"evidence attachment_events[{index}]")
        for index, item in enumerate(
            _require_array(data["attachment_events"], "evidence attachment_events")
        )
    )
    raw_windows = _require_mapping(data["stage_windows"], "evidence stage_windows")
    raw_status = _require_mapping(data["stage_status"], "evidence stage_status")
    if not isinstance(data["complete"], bool):
        raise TypeError("evidence complete must be a boolean")
    if not isinstance(data["collision_observation_complete"], bool):
        raise TypeError("evidence collision_observation_complete must be a boolean")
    return EvidenceBundle(
        initial=initial,
        frames=frames,
        contacts=contacts,
        collisions=collisions,
        attachment_events=attachment_events,
        stage_windows={
            _identity(name, "stage window name"): _pair(value, f"stage window {name!r}")
            for name, value in raw_windows.items()
        },
        stage_status={
            _identity(name, "stage status name"): _boolean(
                value,
                f"stage status {name!r}",
            )
            for name, value in raw_status.items()
        },
        complete=data["complete"],
        # Legacy evidence did not distinguish observation coverage from stage
        # success.  Treat its old complete bit as the best available coverage
        # value while keeping the new field explicit in all new output.
        stage_success_complete=bool(data.get("stage_success_complete", data["complete"])),
        collision_observation_complete=data["collision_observation_complete"],
    )


def _frame_to_dict(frame: EvidenceFrame) -> dict[str, object]:
    return {
        "timestamp": frame.timestamp,
        "base_pose": list(frame.base_pose),
        "base_velocity": list(frame.base_velocity),
        "joint_positions": dict(frame.joint_positions),
        "joint_velocities": dict(frame.joint_velocities),
        "end_effectors": {
            name: list(pose) for name, pose in frame.end_effectors.items()
        },
        "entities": {
            name: {
                "position": list(state.position),
                "quaternion": list(state.quaternion),
                "linear_velocity": list(state.linear_velocity),
                "angular_velocity": list(state.angular_velocity),
                "source": state.source,
            }
            for name, state in frame.entities.items()
        },
        "attachments": dict(frame.attachments),
        "stage": frame.stage,
        "base_orientation": (
            list(frame.base_orientation) if frame.base_orientation is not None else None
        ),
        "base_height_m": frame.base_height_m,
        "imu_linear_acceleration": (
            list(frame.imu_linear_acceleration)
            if frame.imu_linear_acceleration is not None
            else None
        ),
        "imu_angular_velocity": (
            list(frame.imu_angular_velocity)
            if frame.imu_angular_velocity is not None
            else None
        ),
        "support_contacts": dict(frame.support_contacts),
        "joint_efforts": dict(frame.joint_efforts),
        "physical_metrics": dict(frame.physical_metrics),
    }


def _contact_to_dict(event: ContactEvent) -> dict[str, object]:
    return {
        "timestamp": event.timestamp,
        "body_a": event.body_a,
        "body_b": event.body_b,
        "force_n": event.force_n,
        "point": list(event.point) if event.point is not None else None,
        "normal": list(event.normal) if event.normal is not None else None,
    }


def _frame_from_dict(value: object, what: str) -> EvidenceFrame:
    raw = _require_mapping(value, what)
    required = {
            "timestamp",
            "base_pose",
            "base_velocity",
            "joint_positions",
            "joint_velocities",
            "end_effectors",
            "entities",
            "attachments",
            "stage",
        }
    optional = {
        "base_orientation",
        "base_height_m",
        "imu_linear_acceleration",
        "imu_angular_velocity",
        "support_contacts",
        "joint_efforts",
        "physical_metrics",
    }
    unknown = set(raw) - required - optional
    missing = required - set(raw)
    if unknown:
        raise ValueError(f"{what} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{what} is missing fields: {sorted(missing)}")
    raw_entities = _require_mapping(raw["entities"], f"{what} entities")
    return EvidenceFrame(
        timestamp=_number(raw["timestamp"], f"{what} timestamp"),
        base_pose=_triple(raw["base_pose"], f"{what} base_pose"),
        base_velocity=_triple(raw["base_velocity"], f"{what} base_velocity"),
        joint_positions=_number_mapping(raw["joint_positions"], f"{what} joint_positions"),
        joint_velocities=_number_mapping(raw["joint_velocities"], f"{what} joint_velocities"),
        end_effectors={
            _identity(name, f"{what} end effector name"): _pose(pose, f"{what} end_effectors.{name}")
            for name, pose in _require_mapping(
                raw["end_effectors"],
                f"{what} end_effectors",
            ).items()
        },
        entities={
            _identity(name, f"{what} entity name"): _entity_from_dict(
                state,
                f"{what} entities.{name}",
            )
            for name, state in raw_entities.items()
        },
        attachments={
            _identity(name, f"{what} attachment entity"): _identity(
                effector,
                f"{what} attachment effector",
            )
            for name, effector in _require_mapping(
                raw["attachments"],
                f"{what} attachments",
            ).items()
        },
        stage=(
            _identity(raw["stage"], f"{what} stage")
            if raw["stage"] is not None
            else None
        ),
        base_orientation=(
            _quad(raw["base_orientation"], f"{what} base_orientation")
            if raw.get("base_orientation") is not None
            else None
        ),
        base_height_m=(
            _number(raw["base_height_m"], f"{what} base_height_m")
            if raw.get("base_height_m") is not None
            else None
        ),
        imu_linear_acceleration=(
            _triple(
                raw["imu_linear_acceleration"],
                f"{what} imu_linear_acceleration",
            )
            if raw.get("imu_linear_acceleration") is not None
            else None
        ),
        imu_angular_velocity=(
            _triple(raw["imu_angular_velocity"], f"{what} imu_angular_velocity")
            if raw.get("imu_angular_velocity") is not None
            else None
        ),
        support_contacts=_number_mapping(
            raw.get("support_contacts", {}), f"{what} support_contacts"
        ),
        joint_efforts=_number_mapping(
            raw.get("joint_efforts", {}), f"{what} joint_efforts"
        ),
        physical_metrics=_scalar_mapping(
            raw.get("physical_metrics", {}), f"{what} physical_metrics"
        ),
    )


def _entity_from_dict(value: object, what: str) -> EntityState:
    raw = _require_mapping(value, what)
    _require_fields(
        raw,
        {"position", "quaternion", "linear_velocity", "angular_velocity", "source"},
        what,
    )
    return EntityState(
        position=_triple(raw["position"], f"{what} position"),
        quaternion=_quad(raw["quaternion"], f"{what} quaternion"),
        linear_velocity=_triple(raw["linear_velocity"], f"{what} linear_velocity"),
        angular_velocity=_triple(raw["angular_velocity"], f"{what} angular_velocity"),
        source=_identity(raw["source"], f"{what} source"),
    )


def _contact_from_dict(value: object, what: str) -> ContactEvent:
    raw = _require_mapping(value, what)
    _require_fields(
        raw,
        {"timestamp", "body_a", "body_b", "force_n", "point", "normal"},
        what,
    )
    return ContactEvent(
        timestamp=_number(raw["timestamp"], f"{what} timestamp"),
        body_a=_identity(raw["body_a"], f"{what} body_a"),
        body_b=_identity(raw["body_b"], f"{what} body_b"),
        force_n=_number(raw["force_n"], f"{what} force_n"),
        point=_triple(raw["point"], f"{what} point") if raw["point"] is not None else None,
        normal=_triple(raw["normal"], f"{what} normal") if raw["normal"] is not None else None,
    )


def _attachment_from_dict(value: object, what: str) -> AttachmentEvent:
    raw = _require_mapping(value, what)
    _require_fields(raw, {"timestamp", "entity", "effector", "attached"}, what)
    return AttachmentEvent(
        timestamp=_number(raw["timestamp"], f"{what} timestamp"),
        entity=_identity(raw["entity"], f"{what} entity"),
        effector=_identity(raw["effector"], f"{what} effector"),
        attached=raw["attached"],
    )


def _freeze_numeric_mapping(value: Mapping[str, float], what: str) -> Mapping[str, float]:
    return MappingProxyType(_number_mapping(value, what))


def _freeze_scalar_mapping(
    value: Mapping[str, object],
    what: str,
) -> Mapping[str, object]:
    return MappingProxyType(_scalar_mapping(value, what))


def _number_mapping(value: object, what: str) -> dict[str, float]:
    raw = _require_mapping(value, what)
    return {
        _identity(name, f"{what} name"): _number(number, f"{what}.{name}")
        for name, number in raw.items()
    }


def _scalar_mapping(value: object, what: str) -> dict[str, object]:
    raw = _require_mapping(value, what)
    normalized: dict[str, object] = {}
    for name, item in raw.items():
        identity = _identity(name, f"{what} name")
        if isinstance(item, bool):
            normalized[identity] = item
            continue
        if isinstance(item, str):
            normalized[identity] = item
            continue
        normalized[identity] = _number(item, f"{what}.{name}")
    return normalized


def _freeze_pose_mapping(
    value: Mapping[str, tuple[float, ...]],
) -> Mapping[str, tuple[float, ...]]:
    raw = _require_mapping(value, "end_effectors")
    return MappingProxyType(
        {
            _identity(name, "end effector name"): _pose(pose, f"end effector {name!r}")
            for name, pose in raw.items()
        }
    )


def _freeze_entity_mapping(
    value: Mapping[str, EntityState],
) -> Mapping[str, EntityState]:
    raw = _require_mapping(value, "entities")
    entities: dict[str, EntityState] = {}
    for name, state in raw.items():
        identity = _identity(name, "entity name")
        if not isinstance(state, EntityState):
            raise TypeError(f"entity {identity!r} state must be an EntityState")
        entities[identity] = state
    return MappingProxyType(entities)


def _freeze_identity_mapping(
    value: Mapping[str, str],
    what: str,
) -> Mapping[str, str]:
    raw = _require_mapping(value, what)
    return MappingProxyType(
        {
            _identity(name, f"{what} key"): _identity(target, f"{what} value")
            for name, target in raw.items()
        }
    )


def _freeze_stage_windows(
    value: Mapping[str, tuple[float, float]],
) -> Mapping[str, tuple[float, float]]:
    raw = _require_mapping(value, "stage_windows")
    windows: dict[str, tuple[float, float]] = {}
    for name, window in raw.items():
        identity = _identity(name, "stage window name")
        start, end = _pair(window, f"stage window {identity!r}")
        if end < start:
            raise ValueError(f"stage window {identity!r} endpoints must be ordered")
        windows[identity] = (start, end)
    return MappingProxyType(windows)


def _freeze_stage_status(value: Mapping[str, bool]) -> Mapping[str, bool]:
    raw = _require_mapping(value, "stage_status")
    return MappingProxyType(
        {
            _identity(name, "stage status name"): _boolean(
                status,
                f"stage status {name!r}",
            )
            for name, status in raw.items()
        }
    )


def _boolean(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{what} must be a boolean")
    return value

def _typed_tuple(value: object, expected: type, what: str) -> tuple[Any, ...]:
    items = tuple(value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"evidence {what} must contain {expected.__name__} values")
    return items


def _require_fields(raw: Mapping[str, object], fields: set[str], what: str) -> None:
    unknown = set(raw) - fields
    missing = fields - set(raw)
    if unknown:
        raise ValueError(f"{what} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{what} is missing fields: {sorted(missing)}")


def _require_mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be an object")
    return value


def _require_array(value: object, what: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{what} must be an array")
    return value


def _validate_identity(value: object, what: str) -> None:
    _identity(value, what)


def _identity(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _validate_timestamp(value: object, what: str) -> None:
    number = _number(value, what)
    if number < 0.0:
        raise ValueError(f"{what} must be non-negative")


def _validate_non_negative(value: object, what: str) -> None:
    number = _number(value, what)
    if number < 0.0:
        raise ValueError(f"{what} must be non-negative")


def _validate_vector(value: object, length: int, what: str) -> None:
    _vector(value, length, what)


def _vector(value: object, length: int, what: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{what} must be a numeric array")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{what} must be a numeric array") from exc
    if len(items) != length:
        raise ValueError(f"{what} must contain {length} values")
    return tuple(_number(item, f"{what}[{index}]") for index, item in enumerate(items))


def _pose(value: object, what: str) -> tuple[float, ...]:
    return _vector(value, 7, what)


def _pair(value: object, what: str) -> tuple[float, float]:
    first, second = _vector(value, 2, what)
    return first, second


def _triple(value: object, what: str) -> tuple[float, float, float]:
    first, second, third = _vector(value, 3, what)
    return first, second, third


def _quad(value: object, what: str) -> tuple[float, float, float, float]:
    first, second, third, fourth = _vector(value, 4, what)
    return first, second, third, fourth


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite")
    return result


__all__ = [
    "AttachmentEvent",
    "ContactEvent",
    "EntityState",
    "EvidenceBundle",
    "EvidenceFrame",
    "evidence_from_dict",
    "evidence_to_dict",
]
