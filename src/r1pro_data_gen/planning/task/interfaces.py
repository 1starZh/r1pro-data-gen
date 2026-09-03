"""Task-level planner contracts, separate from trajectory planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from r1pro_data_gen.domain import Plan


@dataclass(frozen=True, slots=True)
class TaskPlanningRequest:
    """Facts and intent supplied to a task-level planner."""

    task_description: str
    scene_facts: Mapping[str, Any]
    skill_catalog: tuple[Mapping[str, Any], ...]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    goal_spec: Mapping[str, Any] | None = None
    goal_spec_hash: str | None = None
    # Deterministic geometry/observation contract compiled from the frozen
    # GoalSpec. Legacy callers may omit it.
    goal_contract_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.task_description.strip():
            raise ValueError("task_description must not be empty")
        if not isinstance(self.scene_facts, Mapping):
            raise TypeError("scene_facts must be a mapping")
        if not isinstance(self.skill_catalog, tuple):
            raise TypeError("skill_catalog must be a tuple")
        if self.goal_spec_hash is not None:
            if self.goal_spec is None:
                raise ValueError("goal_spec_hash requires goal_spec")
            if not isinstance(self.goal_spec_hash, str) or not self.goal_spec_hash.strip():
                raise ValueError("goal_spec_hash must be a non-empty string")
        if self.goal_contract_hash is not None:
            if self.goal_spec is None:
                raise ValueError("goal_contract_hash requires goal_spec")
            if not isinstance(self.goal_contract_hash, str) or not self.goal_contract_hash.strip():
                raise ValueError("goal_contract_hash must be a non-empty string")
        if self.goal_spec is not None and not isinstance(self.goal_spec, Mapping):
            raise TypeError("goal_spec must be a mapping")


@dataclass(frozen=True, slots=True)
class TaskPlanningResult:
    """A validated task plan or an explicit unsupported result."""

    status: str
    plan: Plan | None = None
    reason: str = ""
    provider: str = ""
    model: str = ""
    raw_response: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"planned", "unsupported", "failed"}:
            raise ValueError(f"unsupported task planning status: {self.status!r}")
        if self.status == "planned" and self.plan is None:
            raise ValueError("planned result requires a plan")
        if self.status == "unsupported" and not self.reason.strip():
            raise ValueError("unsupported result requires a reason")
        if self.status == "failed" and not self.reason.strip():
            raise ValueError("failed result requires a reason")


class TaskPlanner(Protocol):
    """Protocol for task-level planners that produce semantic Plans."""

    name: str
    model: str

    def plan(self, request: TaskPlanningRequest) -> TaskPlanningResult:
        """Generate and validate a Plan without advancing simulation."""
        ...


class LLMProvider(Protocol):
    """Minimal provider interface used by :class:`LLMTaskPlanner`."""

    name: str
    model: str

    def complete(self, *, system: str, user: str) -> Any:
        """Return a provider response containing JSON text."""
        ...


__all__ = [
    "LLMProvider",
    "TaskPlanner",
    "TaskPlanningRequest",
    "TaskPlanningResult",
]
