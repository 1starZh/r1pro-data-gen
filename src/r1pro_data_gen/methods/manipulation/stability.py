"""Task-independent quasi-static stability certificates.

This module contains the geometry and robot-state checks used before a
whole-body manipulation transition is sent to the simulator.  It deliberately
does not choose a task waypoint or a fixed torso pose.  Callers provide the
live support points, centre of mass, and candidate configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class StabilityCertificate:
    """Result of a quasi-static support-polygon check."""

    stable: bool
    margin_m: float
    com_xy: tuple[float, float]
    support_polygon: tuple[tuple[float, float], ...]
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "stable": bool(self.stable),
            "margin_m": float(self.margin_m),
            "com_xy": [float(value) for value in self.com_xy],
            "support_polygon": [list(point) for point in self.support_polygon],
            "reason": self.reason,
        }


def convex_hull(points: Iterable[Sequence[float]]) -> np.ndarray:
    """Return a counter-clockwise 2-D convex hull without duplicated points."""
    raw = np.asarray(list(points), dtype=float)
    if raw.size == 0:
        return np.empty((0, 2), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.all(np.isfinite(raw)):
        raise ValueError("support points must be a finite (N, 2) array")
    unique = sorted({(float(row[0]), float(row[1])) for row in raw})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float).reshape((-1, 2))

    def cross(origin, first, second) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float).reshape((-1, 2))


def support_polygon_margin(
    com_xy: Sequence[float],
    support_points: Iterable[Sequence[float]],
    *,
    required_margin_m: float = 0.0,
) -> StabilityCertificate:
    """Certify the inward distance of a COM projection from support edges.

    ``margin_m`` is positive only when the COM is inside the polygon and has
    at least the requested erosion margin.  A line or a point support is
    always reported unstable, which prevents a missing wheel-contact reading
    from being treated as a valid base.
    """
    com = np.asarray(com_xy, dtype=float)
    if com.shape != (2,) or not np.all(np.isfinite(com)):
        raise ValueError("com_xy must be a finite 2-vector")
    required = float(required_margin_m)
    if not np.isfinite(required) or required < 0.0:
        raise ValueError("required_margin_m must be finite and non-negative")
    hull = convex_hull(support_points)
    polygon = tuple(tuple(float(value) for value in row) for row in hull)
    if len(hull) < 3:
        return StabilityCertificate(
            False,
            float("-inf"),
            (float(com[0]), float(com[1])),
            polygon,
            "support polygon has fewer than three non-collinear contacts",
        )
    distances: list[float] = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1.0e-12:
            continue
        # Hull is CCW; the left side of each edge is inward.
        distances.append(float(np.cross(edge, com - start) / length))
    if not distances:
        return StabilityCertificate(
            False,
            float("-inf"),
            (float(com[0]), float(com[1])),
            polygon,
            "support polygon edges are degenerate",
        )
    raw_margin = min(distances)
    margin = raw_margin - required
    stable = bool(margin >= 0.0)
    return StabilityCertificate(
        stable,
        float(margin),
        (float(com[0]), float(com[1])),
        polygon,
        "" if stable else "COM projection is outside the eroded support polygon",
    )


def wheel_support_points(
    base_pose: Sequence[float],
    wheel_positions: Iterable[Sequence[float]],
) -> np.ndarray:
    """Transform authored wheel contact points into the current world frame."""
    pose = np.asarray(base_pose, dtype=float)
    local = np.asarray(list(wheel_positions), dtype=float)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("base_pose must be a finite (x, y, yaw) vector")
    if local.ndim != 2 or local.shape[1] != 2 or not np.all(np.isfinite(local)):
        raise ValueError("wheel_positions must be a finite (N, 2) array")
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    return (rotation @ local.T).T + pose[:2]


def payload_com(
    robot_com_world: Sequence[float],
    robot_mass_kg: float,
    payload_position_world: Sequence[float] | None = None,
    payload_mass_kg: float = 0.0,
) -> np.ndarray:
    """Combine robot and optional payload COMs without hidden task constants."""
    robot = np.asarray(robot_com_world, dtype=float)
    if robot.shape != (3,) or not np.all(np.isfinite(robot)):
        raise ValueError("robot_com_world must be a finite 3-vector")
    robot_mass = float(robot_mass_kg)
    payload_mass = float(payload_mass_kg)
    if not np.isfinite(robot_mass) or robot_mass <= 0.0:
        raise ValueError("robot_mass_kg must be positive and finite")
    if not np.isfinite(payload_mass) or payload_mass < 0.0:
        raise ValueError("payload_mass_kg must be finite and non-negative")
    if payload_mass == 0.0:
        return robot.copy()
    if payload_position_world is None:
        raise ValueError("payload position is required for a non-zero payload mass")
    payload = np.asarray(payload_position_world, dtype=float)
    if payload.shape != (3,) or not np.all(np.isfinite(payload)):
        raise ValueError("payload_position_world must be a finite 3-vector")
    return (robot_mass * robot + payload_mass * payload) / (robot_mass + payload_mass)


def configuration_stability(
    *,
    com_world: Sequence[float],
    base_pose: Sequence[float],
    wheel_positions: Iterable[Sequence[float]],
    required_margin_m: float,
) -> StabilityCertificate:
    """Convenience certificate for one candidate robot configuration."""
    support = wheel_support_points(base_pose, wheel_positions)
    return support_polygon_margin(
        np.asarray(com_world, dtype=float)[:2],
        support,
        required_margin_m=required_margin_m,
    )


__all__ = [
    "StabilityCertificate",
    "configuration_stability",
    "convex_hull",
    "payload_com",
    "support_polygon_margin",
    "wheel_support_points",
]
