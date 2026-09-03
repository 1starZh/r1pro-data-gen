"""Deterministic replay planner used by offline contract tests and replays."""

from __future__ import annotations

from r1pro_data_gen.domain import Trajectory, TrajectoryPoint
from r1pro_data_gen.planning.contracts import PlannerRequest, PlannerResult


class ReplayPlanner:
    """Return a supplied trajectory without simulator or planner dependencies."""

    name = "replay"

    def __init__(self, trajectory: Trajectory | None = None) -> None:
        self._trajectory = trajectory

    def plan(self, request: PlannerRequest) -> PlannerResult:
        if self._trajectory is None:
            return PlannerResult(
                feasible=False,
                reason="replay planner has no trajectory for this request",
            )
        return PlannerResult(feasible=True, trajectory=self._trajectory)


def make_hold_trajectory(
    *,
    joint_names: tuple[str, ...] = (),
    stage: str | None = None,
) -> Trajectory:
    """Create a minimal one-step trajectory for contract tests."""
    return Trajectory(
        joint_names=joint_names,
        points=(TrajectoryPoint(timestamp=0.0, stage=stage),),
        planner="replay",
    )
