"""Deterministic, simulator-independent motion algorithms.

Methods answer "how can a requested motion be made feasible?". They are
called by skills and do not choose task order or read task-specific policy.
Navigation and manipulation algorithms have their own subpackages; collision
geometry remains at this level because both domains use it.
"""

from .collision import (
    LINK_SPHERE_RADII,
    LINK_SPHERE_OFFSETS_BY_SIDE,
    CollisionChecker,
    Obstacle,
    check_path,
    ground_obstacle,
    object_obstacle,
    obstacles_from_scene,
)
from .manipulation import (
    ArmPlanningBudget,
    ArmPlanningResult,
    ArmSequenceCandidate,
    ArmSequencePlanningResult,
    ArmWaypoint,
    ConstraintReport,
    IKCandidate,
    PathCandidate,
    StabilityCertificate,
    WaypointIKCandidate,
    configuration_stability,
    convex_hull,
    held_object_configuration_free,
    payload_com,
    plan_certified_task_path,
    plan_task_path,
    support_polygon_margin,
    wheel_support_points,
    WHOLE_BODY_FRAME_RADII_BY_SIDE,
    WholeBodyCollisionChecker,
    whole_body_path_free,
)
from .manipulation.taskspace import TaskPath
from .navigation import (
    RRTPlanner,
    astar_path,
    clearance_cost_grid,
    occupancy_from_boxes,
    path_to_world_waypoints,
    plan_rrt_path,
    simplify_grid_path,
)

__all__ = [
    "ArmPlanningBudget",
    "ArmPlanningResult",
    "ArmSequenceCandidate",
    "ArmSequencePlanningResult",
    "ArmWaypoint",
    "CollisionChecker",
    "ConstraintReport",
    "IKCandidate",
    "LINK_SPHERE_RADII",
    "LINK_SPHERE_OFFSETS_BY_SIDE",
    "Obstacle",
    "PathCandidate",
    "RRTPlanner",
    "StabilityCertificate",
    "TaskPath",
    "WaypointIKCandidate",
    "WHOLE_BODY_FRAME_RADII_BY_SIDE",
    "WholeBodyCollisionChecker",
    "astar_path",
    "check_path",
    "clearance_cost_grid",
    "configuration_stability",
    "convex_hull",
    "ground_obstacle",
    "held_object_configuration_free",
    "object_obstacle",
    "obstacles_from_scene",
    "occupancy_from_boxes",
    "path_to_world_waypoints",
    "payload_com",
    "plan_rrt_path",
    "plan_certified_task_path",
    "plan_task_path",
    "simplify_grid_path",
    "support_polygon_margin",
    "wheel_support_points",
    "whole_body_path_free",
]
