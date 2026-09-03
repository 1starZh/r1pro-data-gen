"""Grid A* path planning tests (pure logic, no Isaac Sim)."""

from __future__ import annotations

import numpy as np

from r1pro_data_gen.methods import astar_path, clearance_cost_grid, occupancy_from_boxes, path_to_world_waypoints, simplify_grid_path


def _empty_grid(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=bool)


def test_astar_finds_straight_path() -> None:
    grid = _empty_grid(10, 10)
    path = astar_path(grid, (1, 1), (1, 8))
    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (1, 8)
    assert all(c == 1 for c, _ in path)  # straight row


def test_astar_returns_none_when_no_path() -> None:
    grid = _empty_grid(5, 5)
    grid[2, :] = True  # wall across the middle
    assert astar_path(grid, (0, 0), (4, 4)) is None


def test_astar_start_inside_obstacle() -> None:
    grid = _empty_grid(5, 5)
    grid[1, 1] = True
    assert astar_path(grid, (1, 1), (4, 4)) is None


def test_astar_corner_cutting_blocked() -> None:
    grid = _empty_grid(5, 5)
    grid[2, 2] = True
    grid[1, 2] = True
    grid[2, 1] = True
    # The only diagonal shortcut (1,1)->(2,2) is corner cutting; path must go around.
    path = astar_path(grid, (1, 1), (3, 3), allow_diagonal=True)
    assert path is not None
    assert all(not grid[r, c] for r, c in path)


def test_occupancy_from_boxes_inflates_cells() -> None:
    grid = occupancy_from_boxes(
        [(0.0, 0.0, 1.0, 1.0)],  # one 1m box at the origin
        origin_x=0.0,
        origin_y=0.0,
        resolution=0.5,
        shape=(4, 4),
    )
    # Box covers cols 0..1 and rows 0..1 (endpoint cell is conservatively full).
    assert grid[0, 0] and grid[1, 1]
    assert not grid[3, 3]  # far corner stays free


def test_path_to_world_waypoints_uses_cell_centers() -> None:
    waypoints = path_to_world_waypoints([(0, 0), (0, 1)], origin_x=1.0, origin_y=2.0, resolution=0.5)
    assert waypoints == [(1.25, 2.25), (1.75, 2.25)]


def test_simplify_grid_path_removes_straight_cells() -> None:
    grid = _empty_grid(8, 8)
    path = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    simplified = simplify_grid_path(path, grid)
    assert simplified[0] == path[0]
    assert simplified[-1] == path[-1]
    assert len(simplified) < len(path)


def test_clearance_cost_prefers_corridor_center() -> None:
    grid = _empty_grid(11, 15)
    grid[1, 2:13] = True
    grid[9, 2:13] = True
    soft = clearance_cost_grid(grid, clearance_cells=4.0, weight=5.0)
    path = astar_path(grid, (5, 1), (5, 13), traversal_cost=soft)
    assert path is not None
    # The equal-length alternatives near rows 2/8 are more expensive than the
    # corridor centre, so a planner must not hug either wall.
    assert max(abs(r - 5) for r, _ in path) <= 1


def test_simplifier_does_not_replace_centered_path_with_wall_hugging_line() -> None:
    grid = _empty_grid(9, 12)
    grid[1, 2:10] = True
    soft = clearance_cost_grid(grid, clearance_cells=4.0, weight=4.0)
    path = [(4, 1), (5, 3), (5, 8), (4, 10)]
    simplified = simplify_grid_path(path, grid, traversal_cost=soft)
    assert all(float(soft[cell]) <= max(float(soft[p]) for p in path) + 1e-9 for cell in simplified)
