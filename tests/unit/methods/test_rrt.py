"""RRT-Connect unit tests on an analytic configuration-space checker."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from r1pro_data_gen.methods.navigation.rrt import RRTConnectPlanner


class _FakeKin:
    """Only ``lower``/``upper`` bounds are needed by the planner."""

    def __init__(self, lower, upper):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)


class _BallObstacleChecker:
    """Collision if the configuration enters a ball around a C-space center."""

    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def is_collision_free(self, q, base_xy=(0.0, 0.0), base_yaw=0.0) -> bool:
        return bool(np.linalg.norm(np.asarray(q, dtype=float) - self.center) > self.radius)


LOWER = np.full(7, -2.0)
UPPER = np.full(7, 2.0)


def test_rrt_connect_solves_free_space() -> None:
    kin = _FakeKin(LOWER, UPPER)
    checker = _BallObstacleChecker(center=np.full(7, 10.0), radius=0.1)  # effectively empty
    planner = RRTConnectPlanner(kin, checker, step=0.3, max_iters=2000, seed=11)
    start = np.full(7, -1.0)
    goal = np.full(7, 1.0)
    ok, path, stats = planner.plan(start, goal)
    assert ok
    assert np.allclose(path[0], start) and np.allclose(path[-1], goal)
    for a, b in zip(path[:-1], path[1:]):
        assert np.linalg.norm(b - a) <= 0.3 * np.sqrt(7) + 1e-9
        assert np.all(a >= LOWER - 1e-9) and np.all(a <= UPPER + 1e-9)
    assert stats["iterations"] > 0


def test_rrt_connect_routes_around_cspace_ball() -> None:
    kin = _FakeKin(LOWER, UPPER)
    checker = _BallObstacleChecker(center=np.zeros(7), radius=0.8)
    planner = RRTConnectPlanner(kin, checker, step=0.25, max_iters=4000, seed=23)
    start = np.full(7, -1.5)
    goal = np.full(7, 1.5)
    ok, path, _ = planner.plan(start, goal)
    assert ok
    dense = np.vstack(
        [
            np.linspace(a, b, 8)[:-1]
            for a, b in zip(path[:-1], path[1:])
        ]
        + [path[-1:]]
    )
    distances = np.linalg.norm(dense - checker.center, axis=1)
    assert np.all(distances > checker.radius)


def test_blocked_goal_fails_immediately() -> None:
    kin = _FakeKin(LOWER, UPPER)
    checker = _BallObstacleChecker(center=np.zeros(7), radius=0.5)
    planner = RRTConnectPlanner(kin, checker, max_iters=50, seed=37)
    ok, path, stats = planner.plan(np.full(7, 1.5), np.zeros(7))
    assert not ok
    assert len(path) == 1 and stats["iterations"] == 0


def test_deterministic_for_same_seed() -> None:
    kin = _FakeKin(LOWER, UPPER)
    checker = _BallObstacleChecker(center=np.zeros(7), radius=0.6)

    def run():
        planner = RRTConnectPlanner(kin, checker, step=0.25, max_iters=3000, seed=99)
        return planner.plan(np.full(7, -1.4), np.full(7, 1.4))

    ok_a, path_a, stats_a = run()
    ok_b, path_b, stats_b = run()
    assert ok_a and ok_b
    assert len(path_a) == len(path_b)
    assert all(np.allclose(x, y) for x, y in zip(path_a, path_b))
    assert stats_a == stats_b


def test_narrow_passage_bounded_runtime() -> None:
    kin = _FakeKin(LOWER, UPPER)
    checker = _BallObstacleChecker(center=np.array([0.0, 0, 0, 0, 0, 0, 0]), radius=0.95)
    planner = RRTConnectPlanner(kin, checker, step=0.2, max_iters=1500, seed=5)
    ok, _, stats = planner.plan(np.full(7, -1.6), np.full(7, 1.6))
    # Either it found a route through the annulus or it stopped cleanly.
    assert stats["iterations"] <= 1500
    if ok:
        assert isinstance(stats["nodes_a"], int)


def test_nearest_neighbor_matches_brute_force() -> None:
    kin = _FakeKin(LOWER, UPPER)
    checker = _BallObstacleChecker(center=np.full(7, 10.0), radius=0.1)
    planner = RRTConnectPlanner(kin, checker, seed=1)
    nodes = [np.full(7, -2.0), np.zeros(7), np.full(7, 2.0)]
    target = np.array([1.8, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0])
    brute = int(np.argmin([np.linalg.norm(n - target) for n in nodes]))
    assert planner._nearest_index(nodes, target) == brute


def test_extend_respects_joint_limits_and_step() -> None:
    kin = _FakeKin(LOWER, UPPER)
    checker = _BallObstacleChecker(center=np.full(7, 10.0), radius=0.1)
    planner = RRTConnectPlanner(kin, checker, step=0.2, seed=2)
    nodes = [np.zeros(7)]
    parents: dict[int, int] = {}
    status, idx = planner._extend(
        nodes, parents, np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]), LOWER, UPPER,
        (0.0, 0.0), 0.0,
    )
    assert status == RRTConnectPlanner.ADVANCED
    new = nodes[idx]
    assert np.all(new <= UPPER + 1e-9)
    moved = np.linalg.norm(new - nodes[parents[idx]])
    assert moved <= 0.2 * np.sqrt(7) + 1e-9
