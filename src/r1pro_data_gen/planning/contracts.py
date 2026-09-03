"""Planner contracts shared by all future planning backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from r1pro_data_gen.domain import Observation, Plan, Trajectory


@dataclass(frozen=True, slots=True)
class PlannerRequest:
    """Input to a planner for one semantic stage."""

    plan: Plan
    stage_name: str
    observation: Observation
    constraints: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage_name not in self.plan.stage_names:
            raise ValueError(f"unknown plan stage: {self.stage_name}")


@dataclass(frozen=True, slots=True)
class PlannerResult:
    """Planner output; physical success is evaluated elsewhere."""

    feasible: bool
    trajectory: Trajectory | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.feasible and self.trajectory is None:
            raise ValueError("feasible planner results require a trajectory")
        if not self.feasible and not self.reason:
            raise ValueError("infeasible planner results require a reason")


class Planner(Protocol):
    """Minimal protocol implemented by CPU, GPU and replay planners."""

    name: str

    def plan(self, request: PlannerRequest) -> PlannerResult:
        """Generate a candidate trajectory without advancing simulation."""
