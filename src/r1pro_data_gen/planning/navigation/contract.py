"""Shared constants for the generic 2D navigation contract."""

from __future__ import annotations

# Hard obstacle inflation used by both planner facts and runtime skills.
NAVIGATION_INFLATION_CLEARANCE_M = 0.05
# Candidates sit one full cell plus a small numerical boundary margin beyond
# the hard inflated boundary. This keeps the published (four-decimal) pose in
# the next free cell instead of exactly on the rasterization boundary.
NAVIGATION_GRID_RESOLUTION_M = 0.05
NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M = NAVIGATION_GRID_RESOLUTION_M + 0.001

__all__ = [
    "NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M",
    "NAVIGATION_GRID_RESOLUTION_M",
    "NAVIGATION_INFLATION_CLEARANCE_M",
]
