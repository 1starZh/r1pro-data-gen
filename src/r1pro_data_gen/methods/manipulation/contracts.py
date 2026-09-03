"""Typed contracts for finite-budget arm path optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


OPTIMALITY_SCOPE = "best_verified_candidate_within_budget"
SEQUENCE_OPTIMALITY_SCOPE = "best_verified_sequence_within_budget"


@dataclass(frozen=True, slots=True)
class ArmPlanningBudget:
    """Deterministic outer budget applied to every IK branch."""

    attempts_per_candidate: int = 2
    fallback_attempts_per_candidate: int = 1
    planning_time_per_attempt_s: float = 3.0

    def __post_init__(self) -> None:
        if self.attempts_per_candidate < 1:
            raise ValueError("attempts_per_candidate must be positive")
        if self.fallback_attempts_per_candidate < 0:
            raise ValueError("fallback_attempts_per_candidate must not be negative")
        if self.planning_time_per_attempt_s <= 0.0:
            raise ValueError("planning_time_per_attempt_s must be positive")


@dataclass(frozen=True, slots=True)
class IKCandidate:
    candidate_id: int
    q_goal: tuple[float, ...]
    position_error_m: float
    rotation_error_rad: float
    posture_cost: float


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """Hard-constraint result for one generated executable trajectory."""

    valid: bool
    stage: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PathCandidate:
    candidate_id: int
    attempt_id: int
    fallback: bool
    q_goal: tuple[float, ...]
    planner_status: str
    constraints: ConstraintReport
    metrics: Mapping[str, float] = field(default_factory=dict)
    score: tuple[int, ...] | None = None
    output: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return self.constraints.valid and self.output is not None


@dataclass(frozen=True, slots=True)
class ArmPlanningResult:
    success: bool
    status: str
    reason: str
    request_hash: str
    candidates: tuple[PathCandidate, ...]
    winner: PathCandidate | None = None
    optimality_scope: str = OPTIMALITY_SCOPE
    planner_seed_controlled: bool = False

    def __post_init__(self) -> None:
        if self.success != (self.winner is not None):
            raise ValueError("successful arm planning requires exactly one winner")


@dataclass(frozen=True, slots=True)
class ArmWaypoint:
    """One ordered EE goal and the collision semantics of its incoming edge."""

    name: str
    poses: tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float, float],
        ],
        ...,
    ]
    exclude_objects: tuple[str, ...] = ()
    contact: bool = False
    speed_scale: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm waypoint name must not be empty")
        if not self.poses:
            raise ValueError("arm waypoint requires at least one EE pose")
        if self.speed_scale is not None and self.speed_scale <= 0.0:
            raise ValueError("arm waypoint speed_scale must be positive")


@dataclass(frozen=True, slots=True)
class WaypointIKCandidate:
    waypoint_id: int
    candidate_id: int
    orientation_id: int
    q_goal: tuple[float, ...]
    position_error_m: float
    rotation_error_rad: float
    continuity_cost: float
    posture_cost: float
    minimum_limit_margin: float
    wrist_motion: float
    minimum_singular_value: float
    score: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArmSequenceCandidate:
    sequence_id: int
    waypoint_candidates: tuple[WaypointIKCandidate, ...]
    segment_reports: tuple[Mapping[str, Any], ...]
    constraints: ConstraintReport
    metrics: Mapping[str, float] = field(default_factory=dict)
    score: tuple[int, ...] | None = None
    output: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return self.constraints.valid and self.output is not None


@dataclass(frozen=True, slots=True)
class ArmSequencePlanningResult:
    success: bool
    status: str
    reason: str
    request_hash: str
    candidates: tuple[ArmSequenceCandidate, ...]
    winner: ArmSequenceCandidate | None = None
    optimality_scope: str = SEQUENCE_OPTIMALITY_SCOPE
    planner_seed_controlled: bool = False

    def __post_init__(self) -> None:
        if self.success != (self.winner is not None):
            raise ValueError("successful arm sequence planning requires exactly one winner")
