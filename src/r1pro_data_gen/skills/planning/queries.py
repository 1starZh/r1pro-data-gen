"""Solve/plan skills: the middle layers of the LLM pipeline.

These turn semantic goals into *plans* without executing anything, so the LLM
can reason about each step (is the target reachable? is the path collision-free?)
and re-plan from live simulation state whenever a deviation is observed.

- ``query_ik_solution``: target pose -> joint configuration (pure solve).
- ``query_arm_path``: target joint config -> collision-free smoothed trajectory
  via MPlib (OMPL + TOPP). Fallback to RRT when MPlib fails.
- ``query_base_path``: target pose -> 2D A* waypoint path.

State-sync contract (LLM re-planning prerequisite): every call reads the current
robot configuration and obstacle geometry from the adapter/scene -- the planner
never caches world state.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from ..core.base import ParamSpec, SkillResult
from ..manipulation.arm import ARM_JOINTS_BY_SIDE
from ..core.sides import for_side, require_side



def runtime_scene_snapshot(scene: Any, adapter: Any, exclude_objects: tuple[str, ...] = ()) -> Any:
    """Copy the declarative scene and refresh movable object poses from sim.

    Static YAML is the fallback for assets without a runtime pose API. Dynamic
    objects that have moved, been pushed, or are being carried therefore enter
    the next planner call at their actual position instead of their reset pose.

    When objects are excluded, the contact/collision sensor filters are pruned
    of those object names too: a sensor filter referencing an object that is
    no longer in the scene violates the SceneModel contract and breaks every
    downstream skill that reconstructs the scene model.
    """
    if scene is None or not hasattr(scene, "objects"):
        return scene
    objects = []
    excluded = set(exclude_objects)
    for obj in scene.objects:
        if obj.name in excluded:
            continue
        current = obj
        if hasattr(adapter, "object_position"):
            try:
                current = replace(obj, pos=tuple(float(v) for v in adapter.object_position(obj.name)))
            except (RuntimeError, KeyError, AttributeError):
                pass
        objects.append(current)
    # Prune sensor filters that referenced excluded objects so the rebuilt
    # SceneModel stays internally consistent (a filter naming an object that is
    # no longer present violates the scene contract).
    if excluded:
        def _prune_sensors(sensors: Any) -> tuple[Any, ...]:
            pruned = []
            for sensor in sensors:
                kept = tuple(name for name in sensor.filter if name not in excluded)
                pruned.append(replace(sensor, filter=kept))
            return tuple(pruned)

        scene = replace(
            scene,
            objects=tuple(objects),
            contact_sensors=_prune_sensors(scene.contact_sensors),
            collision_sensors=_prune_sensors(scene.collision_sensors),
        )
        return scene
    return replace(scene, objects=tuple(objects))


class QueryIKSolution:
    """Solve IK for a target pose without executing (pure solve)."""

    name = "query_ik_solution"
    description = (
        "Solve inverse kinematics for a target end-effector pose and return the "
        "joint configuration, reachability, and position/orientation errors. "
        "Does not move the robot."
    )
    parameters: dict[str, ParamSpec] = {
        "target_pos": ParamSpec("array", "Target end-effector position (base frame, xyz)", required=True, shape=(3,)),
        "target_z_axis": ParamSpec("array", "Desired gripper z-axis direction (xyz); omit for position-only IK", default=None, shape=(3,)),
        "target_quat": ParamSpec("array", "Optional target quaternion (w, x, y, z)", default=None, shape=(4,)),
        "side": ParamSpec("string", "Which arm ('left' or 'right')", default="left", enum=("left", "right")),
        "q_init": ParamSpec("array", "Initial joint configuration (7) for the IK seed", default=None, shape=(7,)),
    }

    def __init__(self, kin: Any):
        self.kin = kin

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target_pos: list[float] = None,
        target_z_axis: list[float] | None = None,
        target_quat: list[float] | None = None,
        side: str = "left",
        q_init: list[float] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        if target_pos is None:
            raise ValueError("query_ik_solution requires target_pos")
        side = require_side(side)
        kin = for_side(self.kin, side)
        from r1pro_data_gen.skills.manipulation.arm import quat_from_z_axis

        target_pos_arr = np.asarray(target_pos, dtype=float)
        if target_quat is not None and target_z_axis is not None:
            raise ValueError("query_ik_solution accepts target_quat or target_z_axis, not both")
        if target_quat is not None:
            target_quat = np.asarray(target_quat, dtype=float)
            if target_quat.shape != (4,) or np.linalg.norm(target_quat) < 1e-9:
                raise ValueError("target_quat must be a non-zero quaternion")
            target_quat = target_quat / np.linalg.norm(target_quat)
        else:
            target_quat = None if target_z_axis is None else quat_from_z_axis(np.asarray(target_z_axis, dtype=float))
        seed = None if q_init is None else np.asarray(q_init, dtype=float)
        if hasattr(kin, "ik_candidates"):
            candidates = kin.ik_candidates(
                target_pos_arr,
                target_quat,
                np.zeros(7) if seed is None else seed,
            )
            sol = candidates[0] if candidates else kin.ik(
                target_pos_arr,
                target_quat,
                q_init=seed,
            )
        else:
            candidates = []
            sol = kin.ik(target_pos_arr, target_quat, q_init=seed)
        return SkillResult(
            success=sol.success,
            skill=self.name,
            metrics={
                "ik_error_m": float(sol.position_error),
                "rotation_error_rad": float(sol.rotation_error),
                "ik_candidates": float(len(candidates) if candidates else int(sol.success)),
            },
            details={
                "q_arm": [round(float(v), 4) for v in sol.q_arm] if sol.q_arm is not None else None,
                "reason": sol.reason or ("reachable" if sol.success else "unreachable"),
            },
        )


class QueryArmPath:
    """Plan a collision-free smoothed arm trajectory to a target configuration."""

    name = "query_arm_path"
    description = (
        "Plan a collision-free, smoothed arm trajectory to a target joint "
        "configuration, avoiding the current scene obstacles (MPlib + TOPP). "
        "Returns the joint trajectory plus velocity/acceleration profiles. "
        "Does not move the robot."
    )
    parameters: dict[str, ParamSpec] = {
        "target_q": ParamSpec("array", "Target joint configuration (7, rad); use this or target_pos", default=None, shape=(7,)),
        "target_pos": ParamSpec("array", "Target EE position (3, base frame); use this or target_q", default=None, shape=(3,)),
        "target_quat": ParamSpec("array", "Optional target EE quaternion (wxyz)", default=None, shape=(4,)),
        "target_z_axis": ParamSpec("array", "Optional target EE z-axis", default=None, shape=(3,)),
        "side": ParamSpec("string", "Which arm ('left' or 'right')", default="left", enum=("left", "right")),
        "planning_time": ParamSpec("number", "Max planning seconds", default=3.0, minimum=0.1),
        "local_radius_m": ParamSpec("number", "Arm-planning obstacle culling radius around the live base (m)", default=2.0, minimum=0.5),
        "exclude_objects": ParamSpec("array", "Scene object names excluded from this arm motion's obstacle set", default=[]),
    }

    def __init__(self, planner: Any, kin: Any = None):
        self.planner = planner
        self.kin = kin

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target_q: list[float] = None,
        target_pos: list[float] | None = None,
        target_quat: list[float] | None = None,
        target_z_axis: list[float] | None = None,
        side: str = "left",
        planning_time: float = 3.0,
        local_radius_m: float = 2.0,
        exclude_objects: list[str] | None = None,
        **_: Any,
    ) -> SkillResult:
        side = require_side(side)
        kin = for_side(self.kin, side)
        planner = for_side(self.planner, side)
        if (target_q is None) == (target_pos is None):
            raise ValueError("query_arm_path requires exactly one of target_q or target_pos")
        if scene is None:
            return SkillResult(
                success=False, skill=self.name,
                details={"reason": "query_arm_path needs a scene (obstacles)"},
            )
        if target_q is None and kin is None:
            return SkillResult(
                success=False, skill=self.name,
                details={"reason": "EE target planning requires a kinematics backend"},
            )
        from r1pro_data_gen.methods.manipulation.arm_path_optimizer import optimize_arm_path
        from r1pro_data_gen.methods.manipulation.mplib_path import mplib_qpos_from_joint_positions

        # State-sync: read the current arm config and base pose from the sim.
        obs = adapter.read_observation(0.0)
        q_cur = np.array([obs.joint_positions[j] for j in ARM_JOINTS_BY_SIDE[side]])
        base_pose = obs.base_pose or (0.0, 0.0, 0.0)
        base_xy = (float(base_pose[0]), float(base_pose[1]))
        base_yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
        full_q_current = mplib_qpos_from_joint_positions(obs.joint_positions)
        if target_q is None:
            from r1pro_data_gen.skills.manipulation.arm import quat_from_z_axis

            if target_quat is not None and target_z_axis is not None:
                raise ValueError("query_arm_path accepts target_quat or target_z_axis, not both")
            quat = None if target_quat is None else np.asarray(target_quat, dtype=float)
            if quat is None and target_z_axis is not None:
                quat = quat_from_z_axis(np.asarray(target_z_axis, dtype=float))
            if hasattr(kin, "ik_candidates"):
                solutions = kin.ik_candidates(
                    np.asarray(target_pos, dtype=float),
                    quat,
                    q_cur,
                )
            else:
                sol = kin.ik(np.asarray(target_pos, dtype=float), quat, q_init=q_cur)
                solutions = [sol] if sol.success and sol.q_arm is not None else []
            if not solutions:
                return SkillResult(
                    False,
                    self.name,
                    details={"reason": "no online IK candidate reached the target"},
                )
        else:
            target = np.asarray(target_q, dtype=float)
            solutions = [
                SimpleNamespace(
                    q_arm=target,
                    position_error=0.0,
                    rotation_error=0.0,
                )
            ]
        live_scene = runtime_scene_snapshot(scene, adapter, tuple(exclude_objects or ()))

        planning = optimize_arm_path(
            planner,
            kin,
            q_cur,
            solutions,
            live_scene,
            base_xy=base_xy,
            base_yaw=base_yaw,
            full_q_current=full_q_current,
            planning_time=float(planning_time),
            local_radius_m=float(local_radius_m),
            speed_scale=0.42,
            side=side,
            attempts_per_candidate=2,
            fallback_attempts_per_candidate=1,
        )
        if not planning.success or planning.winner is None:
            return SkillResult(
                success=False,
                skill=self.name,
                details={
                    "reason": planning.reason,
                    "status": planning.status,
                    "optimality_scope": planning.optimality_scope,
                    "planner_seed_controlled": planning.planner_seed_controlled,
                    "request_hash": planning.request_hash,
                    "candidates": [
                        {
                            "candidate_id": item.candidate_id,
                            "attempt_id": item.attempt_id,
                            "fallback": item.fallback,
                            "status": item.planner_status,
                            "failure_stage": item.constraints.stage,
                            "reasons": list(item.constraints.reasons),
                        }
                        for item in planning.candidates
                    ],
                },
            )
        winner = planning.winner
        out = winner.output
        assert out is not None
        return SkillResult(
            success=True,
            skill=self.name,
            metrics={
                "waypoints": float(len(out["position"])),
                "duration_s": round(float(out["duration"]), 3),
                "dt": float(out.get("dt", 1.0 / 60.0)),
                "winding": round(float(out.get("winding", 0.0)), 4),
                "ee_winding": round(float(out.get("ee_winding", 1.0)), 4),
                "ee_path_length_m": float(winner.metrics["ee_path_length_m"]),
                "normalized_joint_path_length": float(
                    winner.metrics["normalized_joint_path_length"]
                ),
                "smoothness_cost": float(winner.metrics["smoothness_cost"]),
            },
            details={
                "trajectory": out["position"].round(4).tolist(),
                "velocity": out["velocity"].round(4).tolist() if out["velocity"] is not None else None,
                "acceleration": out["acceleration"].round(4).tolist() if out["acceleration"] is not None else None,
                "goal_q": np.asarray(winner.q_goal, dtype=float).round(4).tolist(),
                "winner_candidate_id": winner.candidate_id,
                "winner_attempt_id": winner.attempt_id,
                "candidate_count": len(planning.candidates),
                "optimality_scope": planning.optimality_scope,
                "planner_seed_controlled": planning.planner_seed_controlled,
                "request_hash": planning.request_hash,
            },
        )


class QueryBasePath:
    """Plan a 2D collision-free base path (grid A*) to a target pose."""

    name = "query_base_path"
    description = (
        "Plan a 2D collision-free base path (grid A*) to a world pose "
        "(x, y, yaw), avoiding the current scene obstacles. Returns the "
        "waypoint path. Does not move the robot."
    )
    parameters: dict[str, ParamSpec] = {
        "target": ParamSpec("array", "Target world pose (x, y, yaw)", required=True),
        "resolution": ParamSpec("number", "Grid cell size (m)", default=0.05),
        "footprint_radius": ParamSpec("number", "Optional footprint override; otherwise derived from the robot", default=None, minimum=0.05),
    }

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        target: list[float] = None,
        resolution: float = 0.05,
        footprint_radius: float | None = None,
        **_: Any,
    ) -> SkillResult:
        import math

        from r1pro_data_gen.methods import astar_path, occupancy_from_boxes, path_to_world_waypoints, simplify_grid_path
        from r1pro_data_gen.planning.navigation.contract import NAVIGATION_INFLATION_CLEARANCE_M
        from r1pro_data_gen.skills.mobility.base_motion import _footprint_radius

        if target is None:
            raise ValueError("query_base_path requires target (x, y, yaw)")
        obs = adapter.read_observation(0.0)
        bx, by = (float(obs.base_pose[0]), float(obs.base_pose[1])) if obs.base_pose else (0.0, 0.0)
        tx, ty, _ = float(target[0]), float(target[1]), float(target[2])

        pad = 1.5
        xmin, xmax = min(bx, tx) - pad, max(bx, tx) + pad
        ymin, ymax = min(by, ty) - pad, max(by, ty) + pad
        rows = max(2, int(math.ceil((ymax - ymin) / resolution)))
        cols = max(2, int(math.ceil((xmax - xmin) / resolution)))

        footprint_radius = _footprint_radius(adapter, scene) if footprint_radius is None else float(footprint_radius)
        live_scene = runtime_scene_snapshot(scene, adapter)
        boxes: list[tuple[float, float, float, float]] = []
        if live_scene is not None:
            for obj in live_scene.objects:
                if not obj.physics.collision_enabled:
                    continue
                hx = hy = 0.0
                if obj.type.value == "cuboid":
                    hx, hy, _ = obj.size
                    hx /= 2.0
                    hy /= 2.0
                else:
                    hx = hy = obj.radius
                inflate = footprint_radius + NAVIGATION_INFLATION_CLEARANCE_M
                boxes.append(
                    (obj.pos[0] - hx - inflate, obj.pos[1] - hy - inflate,
                     obj.pos[0] + hx + inflate, obj.pos[1] + hy + inflate)
                )
        grid = occupancy_from_boxes(boxes, xmin, ymin, resolution, (rows, cols))
        start = (int((by - ymin) / resolution), int((bx - xmin) / resolution))
        goal = (int((ty - ymin) / resolution), int((tx - xmin) / resolution))
        if grid[start]:
            return SkillResult(
                success=False,
                skill=self.name,
                details={
                    "reason": "start cell is inside an obstacle",
                    "target": [tx, ty, float(target[2])],
                    "footprint_radius_m": footprint_radius,
                },
            )
        if grid[goal]:
            return SkillResult(
                success=False,
                skill=self.name,
                details={
                    "reason": "goal cell is inside an obstacle",
                    "target": [tx, ty, float(target[2])],
                    "footprint_radius_m": footprint_radius,
                },
            )
        path = astar_path(grid, start, goal, allow_diagonal=True)
        if path is None:
            return SkillResult(
                success=False, skill=self.name,
                details={"reason": "no collision-free 2D path to target"},
            )
        path = simplify_grid_path(path, grid)
        waypoints = path_to_world_waypoints(path, xmin, ymin, resolution)
        return SkillResult(
            success=True,
            skill=self.name,
            metrics={"waypoints": float(len(waypoints))},
            details={
                "path": [[round(float(w[0]), 4), round(float(w[1]), 4)] for w in waypoints],
                "footprint_radius_m": footprint_radius,
                "target_yaw": float(target[2]),
            },
        )


__all__ = ["QueryArmPath", "QueryBasePath", "QueryIKSolution"]
