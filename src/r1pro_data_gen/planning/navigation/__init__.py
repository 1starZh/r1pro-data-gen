"""Semantic navigation target resolution and reachability checks.

The target resolver is loaded lazily because scene-fact generation imports the
low-level navigation constants.  Keeping that dependency one-way avoids a
package-initialisation cycle while preserving the concise public imports from
``r1pro_data_gen.planning.navigation``.
"""

from typing import TYPE_CHECKING

from .contract import (
    NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M,
    NAVIGATION_GRID_RESOLUTION_M,
    NAVIGATION_INFLATION_CLEARANCE_M,
)
from .reachability import InteractionReachabilityReport, assess_interaction_target

if TYPE_CHECKING:
    from .targets import NavigationTargetError, NavigationTargetResolution, resolve_navigation_target


def __getattr__(name: str):
    if name in {"NavigationTargetError", "NavigationTargetResolution", "resolve_navigation_target"}:
        from .targets import NavigationTargetError, NavigationTargetResolution, resolve_navigation_target

        return {
            "NavigationTargetError": NavigationTargetError,
            "NavigationTargetResolution": NavigationTargetResolution,
            "resolve_navigation_target": resolve_navigation_target,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "InteractionReachabilityReport",
    "NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M",
    "NAVIGATION_GRID_RESOLUTION_M",
    "NAVIGATION_INFLATION_CLEARANCE_M",
    "NavigationTargetError",
    "NavigationTargetResolution",
    "assess_interaction_target",
    "resolve_navigation_target",
]
