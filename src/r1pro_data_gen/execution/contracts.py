"""Execution contracts for fake and future simulator-backed executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from r1pro_data_gen.domain import ControlCommand, Observation, TaskResult


class SimulationBackend(Protocol):
    """Minimal backend boundary for fake and future simulator implementations."""

    def reset(self) -> Observation:
        """Reset and return the actual initial observation."""

    def step(self, command: ControlCommand) -> Observation:
        """Apply one command and return actual feedback."""


class PhysicalSafetyViolation(RuntimeError):
    """Measured simulator safety gate crossing that must not be swallowed."""

    def __init__(self, code: str, *, metrics: Mapping[str, Any] | None = None) -> None:
        self.code = str(code)
        self._metrics = dict(metrics or {})
        super().__init__(f"physical safety gate failed: {self.code}")

    def metrics(self) -> dict[str, Any]:
        """Return the measured physical values at the gate crossing."""
        return dict(self._metrics)


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    """One command/observation pair from an execution rollout."""

    command: ControlCommand
    observation: Observation


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Execution output passed to task evaluation."""

    task_result: TaskResult
    steps: tuple[ExecutionStep, ...] = ()
