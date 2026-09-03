"""2D grid A* path planning for base navigation.

Pure Python (numpy only). The base is a 3-wheel omni-directional chassis, so a
path is a sequence of (x, y) waypoints the base follows with closed-loop
control (see the ``base_navigate_to`` skill). Obstacles are projected to a
2D occupancy grid with an inflation radius (robot footprint) before planning.

The grid is row/column indexed: row = world +y, column = world +x, origin at
the grid's lower-left corner (min_x, min_y), resolution in meters per cell.
"""

from __future__ import annotations

import heapq

import numpy as np


def occupancy_from_boxes(
    box_xyxy: list[tuple[float, float, float, float]],
    origin_x: float,
    origin_y: float,
    resolution: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a 2D occupancy grid (True = blocked) from world-space boxes.

    Args:
        box_xyxy: axis-aligned 2D boxes as (xmin, ymin, xmax, ymax), each
            already inflated by the robot footprint / safety margin.
        origin_x / origin_y: world coordinates of the grid's lower-left corner.
        resolution: meters per grid cell.
        shape: (rows, cols) of the grid (rows = +y, cols = +x).
    """
    rows, cols = shape
    grid = np.zeros((rows, cols), dtype=bool)
    for xmin, ymin, xmax, ymax in box_xyxy:
        c0, c1 = _world_to_grid_cols(xmin, xmax, origin_x, resolution, cols)
        r0, r1 = _world_to_grid_rows(ymin, ymax, origin_y, resolution, rows)
        if c1 < c0 or r1 < r0:
            continue
        grid[r0 : r1 + 1, c0 : c1 + 1] = True
    return grid


def astar_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    allow_diagonal: bool = True,
    traversal_cost: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    """A* over ``grid`` (True = blocked) from start to goal (row, col).

    Returns the path as a list of (row, col) cells including both endpoints,
    or None if no path exists. When ``allow_diagonal`` is set, diagonal moves
    are allowed only when both orthogonal neighbours are free (no corner
    cutting).
    """
    rows, cols = grid.shape
    if traversal_cost is not None:
        traversal_cost = np.asarray(traversal_cost, dtype=float)
        if traversal_cost.shape != grid.shape:
            raise ValueError("traversal_cost must have the same shape as grid")
        if np.any(traversal_cost < 0.0) or not np.isfinite(traversal_cost).all():
            raise ValueError("traversal_cost must contain finite non-negative values")
    for r, c in (start, goal):
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"cell {(r, c)} is outside the {shape_grid(rows, cols)} grid")
        if grid[r, c]:
            return None  # start or goal inside an obstacle

    neighbors = _neighbors(allow_diagonal)
    open_heap: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, r, c = heapq.heappop(open_heap)
        current = (r, c)
        if current == goal:
            return _reconstruct(came_from, current)
        if current in closed:
            continue
        closed.add(current)
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc]:
                continue
            if allow_diagonal and dr != 0 and dc != 0:
                if grid[r + dr, c] or grid[r, c + dc]:
                    continue  # corner cutting through two blocked orthogonal cells
            neighbor = (nr, nc)
            step_cost = _step_cost(dr, dc)
            soft_cost = 0.0 if traversal_cost is None else float(traversal_cost[nr, nc])
            tentative = g_score[current] + step_cost * (1.0 + soft_cost)
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + _heuristic(neighbor, goal)
                heapq.heappush(open_heap, (priority, nr, nc))
    return None


def path_to_world_waypoints(
    path: list[tuple[int, int]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> list[tuple[float, float]]:
    """Convert grid-cell path to world (x, y) waypoints (cell centers)."""
    return [
        (
            origin_x + (col + 0.5) * resolution,
            origin_y + (row + 0.5) * resolution,
        )
        for row, col in path
    ]


def simplify_grid_path(
    path: list[tuple[int, int]],
    grid: np.ndarray,
    traversal_cost: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Remove unnecessary grid corners while preserving collision safety.

    A* returns one cell per step. Following every cell makes the omni base
    repeatedly stop and re-orient, which looks like spinning in place. This
    line-of-sight pass keeps only meaningful corners and checks every traversed
    cell, including diagonal corner-cutting constraints.
    """
    if len(path) <= 2:
        return list(path)
    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        chosen = anchor + 1
        for candidate in range(anchor + 2, len(path)):
            if _line_is_free(grid, path[anchor], path[candidate]) and _line_preserves_clearance(
                path, anchor, candidate, traversal_cost
            ):
                chosen = candidate
            else:
                break
        result.append(path[chosen])
        anchor = chosen
    return result


def clearance_cost_grid(
    grid: np.ndarray,
    clearance_cells: float,
    weight: float = 3.0,
) -> np.ndarray:
    """Return a soft traversal penalty that prefers the middle of free space.

    ``grid`` already contains the hard robot-footprint inflation.  This extra
    cost does not make a doorway impassable: it only breaks shortest-path ties
    in favour of cells farther from walls.  The quadratic profile reaches zero
    at ``clearance_cells``.
    """
    if clearance_cells <= 0.0 or weight <= 0.0 or not np.any(grid):
        return np.zeros_like(grid, dtype=float)
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~np.asarray(grid, dtype=bool))
    normalized = np.clip((float(clearance_cells) - distance) / float(clearance_cells), 0.0, 1.0)
    cost = float(weight) * normalized**2
    cost[grid] = 0.0
    return cost


def _line_preserves_clearance(
    path: list[tuple[int, int]],
    start_index: int,
    goal_index: int,
    traversal_cost: np.ndarray | None,
) -> bool:
    """Prevent line-of-sight simplification from cutting back toward a wall."""
    if traversal_cost is None:
        return True
    direct = _line_cells(path[start_index], path[goal_index])
    original = path[start_index : goal_index + 1]
    direct_peak = max(float(traversal_cost[cell]) for cell in direct)
    original_peak = max(float(traversal_cost[cell]) for cell in original)
    direct_mean = float(np.mean([traversal_cost[cell] for cell in direct]))
    original_mean = float(np.mean([traversal_cost[cell] for cell in original]))
    return direct_peak <= original_peak + 1e-9 and direct_mean <= original_mean + 0.05


def _line_cells(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = goal
    steps = max(abs(r1 - r0), abs(c1 - c0))
    return [
        (
            int(round(r0 + (r1 - r0) * i / max(steps, 1))),
            int(round(c0 + (c1 - c0) * i / max(steps, 1))),
        )
        for i in range(steps + 1)
    ]


def _line_is_free(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """Conservative integer line traversal used by :func:`simplify_grid_path`."""
    r0, c0 = start
    r1, c1 = goal
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for i in range(steps + 1):
        t = i / max(steps, 1)
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if grid[r, c]:
            return False
        if i:
            pr = int(round(r0 + (r1 - r0) * (i - 1) / max(steps, 1)))
            pc = int(round(c0 + (c1 - c0) * (i - 1) / max(steps, 1)))
            if r != pr and c != pc and (grid[pr, c] or grid[r, pc]):
                return False
    return True


def _step_cost(dr: int, dc: int) -> float:
    return 1.0 if dr == 0 or dc == 0 else 2.0**0.5


def _heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> float:
    dr = abs(cell[0] - goal[0])
    dc = abs(cell[1] - goal[1])
    return (dr + dc) + (2.0**0.5 - 2.0) * min(dr, dc)  # octile


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _neighbors(allow_diagonal: bool) -> list[tuple[int, int]]:
    if allow_diagonal:
        return [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
    return [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _world_to_grid_cols(
    xmin: float, xmax: float, origin_x: float, resolution: float, cols: int
) -> tuple[int, int]:
    c0 = int((xmin - origin_x) / resolution)
    c1 = int((xmax - origin_x) / resolution)
    if c1 < 0 or c0 >= cols:
        return 1, 0  # fully outside -> empty range
    return max(0, c0), min(cols - 1, c1)


def _world_to_grid_rows(
    ymin: float, ymax: float, origin_y: float, resolution: float, rows: int
) -> tuple[int, int]:
    r0 = int((ymin - origin_y) / resolution)
    r1 = int((ymax - origin_y) / resolution)
    if r1 < 0 or r0 >= rows:
        return 1, 0  # fully outside -> empty range
    return max(0, r0), min(rows - 1, r1)


def shape_grid(rows: int, cols: int) -> str:
    return f"{rows}x{cols}"


__all__ = [
    "astar_path",
    "clearance_cost_grid",
    "occupancy_from_boxes",
    "path_to_world_waypoints",
    "simplify_grid_path",
]
