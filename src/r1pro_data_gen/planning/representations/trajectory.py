"""Small pure-Python helpers for constructing validated trajectories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from r1pro_data_gen.domain import Trajectory, TrajectoryPoint


def trajectory_from_points(
    joint_names: Iterable[str],
    points: Iterable[TrajectoryPoint],
    *,
    planner: str = "unknown",
    metadata: Mapping[str, object] | None = None,
) -> Trajectory:
    """Build a trajectory while keeping validation in the domain model."""
    return Trajectory(
        joint_names=tuple(joint_names),
        points=tuple(points),
        planner=planner,
        metadata={} if metadata is None else dict(metadata),
    )
