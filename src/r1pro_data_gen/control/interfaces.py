"""Control contracts independent of a concrete actuator API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from r1pro_data_gen.domain import ControlCommand, ControlMode, Observation, TrajectoryPoint


@dataclass(frozen=True, slots=True)
class JointGroup:
    """Logical group and its selected low-level control mode."""

    name: str
    joints: tuple[str, ...]
    mode: ControlMode

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("joint group name must not be empty")
        if len(set(self.joints)) != len(self.joints):
            raise ValueError("joint group joints must be unique")


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Group configuration used by a pure logic command router."""

    groups: tuple[JointGroup, ...]
    limits: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [group.name for group in self.groups]
        if len(names) != len(set(names)):
            raise ValueError("joint group names must be unique")


class Controller(Protocol):
    """Convert a trajectory point and feedback into a command."""

    def command(
        self,
        point: TrajectoryPoint,
        observation: Observation,
        timestamp: float,
    ) -> ControlCommand:
        """Produce one command without changing simulator state."""
