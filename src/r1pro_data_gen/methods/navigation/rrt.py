"""Joint-space RRT planner for collision-free arm paths.

Sampling-based planning in the 7-DOF arm configuration space. Each sampled
configuration is validated against an inflated obstacle scene (see
``methods.collision``); the result is a collision-free waypoint path from
start to goal. This is a *method* that the ``arm_move_to`` skill falls back
to when straight joint interpolation is not collision-free.
"""

from __future__ import annotations

import numpy as np

from ..collision import CollisionChecker, check_path


class RRTPlanner:
    """Basic RRT in joint space with goal biasing."""

    def __init__(
        self,
        kin,
        checker: CollisionChecker,
        step: float = 0.3,
        goal_bias: float = 0.2,
        goal_tol: float = 0.05,
        max_iters: int = 8000,
        seed: int = 0,
    ) -> None:
        self.kin = kin
        self.checker = checker
        self.step = step
        self.goal_bias = goal_bias
        self.goal_tol = goal_tol
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
    ) -> tuple[bool, list[np.ndarray], int]:
        """Plan from ``q_start`` to ``q_goal``; returns (success, path, nodes)."""
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        if not self.checker.is_collision_free(q_goal, base_xy, base_yaw):
            return False, [q_start], 0

        nodes = [q_start]
        parents: dict[int, int] = {}
        lower, upper = self.kin.lower, self.kin.upper

        for it in range(self.max_iters):
            if self.rng.random() < self.goal_bias:
                sample = q_goal
            else:
                sample = self.rng.uniform(lower, upper)

            # Nearest node.
            dists = np.linalg.norm(np.asarray(nodes) - sample, axis=1)
            nearest = int(np.argmin(dists))
            direction = sample - nodes[nearest]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            q_new = nodes[nearest] + self.step * direction / norm
            q_new = np.clip(q_new, lower, upper)

            if not self.checker.is_collision_free(q_new, base_xy, base_yaw):
                continue

            idx = len(nodes)
            nodes.append(q_new)
            parents[idx] = nearest

            if np.linalg.norm(q_new - q_goal) < self.goal_tol:
                path = self._reconstruct(parents, idx, nodes, q_goal)
                return True, path, len(nodes)

        return False, [q_start], len(nodes)

    def _reconstruct(
        self,
        parents: dict[int, int],
        idx: int,
        nodes: list[np.ndarray],
        q_goal: np.ndarray,
    ) -> list[np.ndarray]:
        path = [q_goal.copy()]
        while idx in parents:
            idx = parents[idx]
            path.append(nodes[idx])
        return list(reversed(path))


def plan_rrt_path(
    kin,
    checker: CollisionChecker,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_yaw: float = 0.0,
    **kwargs,
) -> tuple[bool, list[np.ndarray]]:
    """Convenience wrapper: run RRT and validate the full dense path."""
    planner = RRTPlanner(kin, checker, **kwargs)
    ok, path, _ = planner.plan(q_start, q_goal, base_xy, base_yaw)
    if ok:
        ok, _, _ = check_path(
            checker,
            path,
            base_xy,
            dense=10,
            base_yaw=base_yaw,
        )
    return ok, path


class RRTConnectPlanner:
    """Bidirectional RRT-Connect in the arm joint space.

    Two trees grow alternately (extend from one, greedily connect from the
    other); a join yields the whole path in one shot, so no goal bias is
    needed and convergence is far more reliable than the single-tree RRT
    above -- which matters because this planner is the controlled fallback
    for MPlib/OMPL timeouts observed only inside real GPU processes.
    """

    TRAPPED, ADVANCED, REACHED = 0, 1, 2

    def __init__(
        self,
        kin,
        checker: CollisionChecker,
        step: float = 0.20,
        max_iters: int = 3000,
        connect_depth_cap: int = 32,
        seed: int = 0,
    ) -> None:
        self.kin = kin
        self.checker = checker
        self.step = step
        self.max_iters = max_iters
        self.connect_depth_cap = connect_depth_cap
        self.rng = np.random.default_rng(seed)

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
    ) -> tuple[bool, list[np.ndarray], dict]:
        """Plan from ``q_start`` to ``q_goal``.

        Returns ``(success, path, stats)``; ``stats`` carries node/iteration
        counts so callers can distinguish "exhausted budget" from other
        failures.
        """
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        lower, upper = self.kin.lower, self.kin.upper
        if not self.checker.is_collision_free(q_start, base_xy, base_yaw):
            return False, [q_start], {"iterations": 0, "nodes_a": 1, "nodes_b": 1}
        if not self.checker.is_collision_free(q_goal, base_xy, base_yaw):
            return False, [q_start], {"iterations": 0, "nodes_a": 1, "nodes_b": 1}

        tree_a = [q_start]
        tree_b = [q_goal]
        parents_a: dict[int, int] = {}
        parents_b: dict[int, int] = {}
        iterations = 0
        for iterations in range(1, self.max_iters + 1):
            grow_nodes, grow_parents, connect_nodes, connect_parents = (
                (tree_a, parents_a, tree_b, parents_b)
                if iterations % 2 == 1
                else (tree_b, parents_b, tree_a, parents_a)
            )
            sample = self.rng.uniform(lower, upper)
            status, new_idx = self._extend(
                grow_nodes, grow_parents, sample, lower, upper, base_xy, base_yaw
            )
            if status != self.TRAPPED:
                bridge_status = self._connect(
                    connect_nodes,
                    connect_parents,
                    grow_nodes[new_idx],
                    lower,
                    upper,
                    base_xy,
                    base_yaw,
                )
                if bridge_status == self.REACHED:
                    # The connecting tree reached the newly added node.
                    path_from_grow = self._path_to_root(grow_parents, new_idx, grow_nodes)
                    path_from_connect = self._path_to_root(
                        connect_parents, len(connect_nodes) - 1, connect_nodes
                    )
                    if grow_nodes is tree_a:
                        # start-tree segment then goal-tree segment.
                        path = list(reversed(path_from_grow)) + path_from_connect
                    else:
                        path = list(reversed(path_from_connect)) + path_from_grow
                    return True, path, {
                        "iterations": iterations,
                        "nodes_a": len(tree_a),
                        "nodes_b": len(tree_b),
                    }
        return False, [q_start], {
            "iterations": self.max_iters,
            "nodes_a": len(tree_a),
            "nodes_b": len(tree_b),
        }

    def _nearest_index(self, nodes: list[np.ndarray], target: np.ndarray) -> int:
        distances = np.linalg.norm(np.asarray(nodes) - target, axis=1)
        return int(np.argmin(distances))

    def _extend(
        self,
        nodes: list[np.ndarray],
        parents: dict[int, int],
        target: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        base_xy: tuple[float, float],
        base_yaw: float,
    ) -> tuple[int, int]:
        nearest_idx = self._nearest_index(nodes, target)
        direction = target - nodes[nearest_idx]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return self.REACHED, nearest_idx
        q_new = np.clip(
            nodes[nearest_idx] + min(self.step, norm) * direction / norm,
            lower,
            upper,
        )
        if not self.checker.is_collision_free(q_new, base_xy, base_yaw):
            return self.TRAPPED, -1
        nodes.append(q_new)
        parents[len(nodes) - 1] = nearest_idx
        reached = norm <= self.step
        return (self.REACHED if reached else self.ADVANCED), len(nodes) - 1

    def _connect(
        self,
        nodes: list[np.ndarray],
        parents: dict[int, int],
        target: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        base_xy: tuple[float, float],
        base_yaw: float,
    ) -> int:
        for _ in range(self.connect_depth_cap):
            status, _ = self._extend(nodes, parents, target, lower, upper, base_xy, base_yaw)
            if status != self.ADVANCED:
                return status
        return self.TRAPPED

    def _path_to_root(
        self,
        parents: dict[int, int],
        idx: int,
        nodes: list[np.ndarray],
    ) -> list[np.ndarray]:
        path = [nodes[idx]]
        while idx in parents:
            idx = parents[idx]
            path.append(nodes[idx])
        return path


def plan_rrt_connect_path(
    kin,
    checker: CollisionChecker,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_yaw: float = 0.0,
    seed: int = 0,
    **kwargs,
) -> tuple[bool, list[np.ndarray], dict]:
    """Convenience wrapper: run RRT-Connect and densely validate the path."""
    planner = RRTConnectPlanner(kin, checker, seed=seed, **kwargs)
    ok, path, stats = planner.plan(q_start, q_goal, base_xy, base_yaw)
    if ok:
        ok, _, _ = check_path(checker, path, base_xy, dense=10, base_yaw=base_yaw)
    return ok, path, stats


__all__ = ["RRTPlanner", "plan_rrt_path", "RRTConnectPlanner", "plan_rrt_connect_path"]
