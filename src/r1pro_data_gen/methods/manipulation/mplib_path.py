"""MPlib collision-aware trajectory planning for the R1Pro arm.

Wraps MPlib (OMPL + FCL) so skills can request a *collision-free, smoothed*
joint-space path to a target configuration -- replacing the raw RRT polyline +
linear interpolation that produced jerky, obstacle-blocked motion.

MPlib pipeline (verified on the R1Pro left arm):
    1. Load the open-galaxea planning URDF/SRDF (collision meshes + SRDF
       self-collision pairs; gripper finger limits aligned with the sim USDA).
    2. Update the planning world with the current scene obstacles (point cloud).
    3. ``plan_qpos`` to the goal configuration (fixed torso joints), which runs
       OMPL (RRT-Connect) for a collision-free path and TOPP for a
       time-optimal, velocity/acceleration-bounded trajectory.

State-sync contract (LLM re-planning prerequisite): callers must pass the
current robot configuration and the current obstacle geometry -- the planner
never caches world state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from r1pro_data_gen.domain import SceneModel
from r1pro_data_gen.robot.robot_config import (
    R1PRO_ARM_MIN_TRAJECTORY_S,
    R1PRO_ARM_VELOCITY_LIMITS,
    R1PRO_JOINT_LIMITS,
)

# Planning assets (copied from open-galaxea, finger limits fixed).
_PLAN_URDF = Path(__file__).resolve().parents[4] / "asset" / "r1pro" / "mplib" / "robot.urdf"
_PLAN_SRDF = Path(__file__).resolve().parents[4] / "asset" / "r1pro" / "mplib" / "robot_mplib.srdf"

# MPlib user-joint layout (22): torso[0:4], left_arm[4:11], left_grip[11:13],
# right_arm[13:20], right_grip[20:22]. Each planner owns torso + one arm.
_MOVE_GROUP_BY_SIDE = {side: f"{side}_gripper_link" for side in ("left", "right")}
_TORSO_IDX = list(range(4))
_ARM_SLICE_BY_SIDE = {"left": slice(4, 11), "right": slice(13, 20)}
_MOVE_GROUP_ARM_SLICE = slice(4, 11)
_USER_JOINT_NAMES = (
    *(f"torso_joint{i}" for i in range(1, 5)),
    *(f"left_arm_joint{i}" for i in range(1, 8)),
    "left_gripper_finger_joint1", "left_gripper_finger_joint2",
    *(f"right_arm_joint{i}" for i in range(1, 8)),
    "right_gripper_finger_joint1", "right_gripper_finger_joint2",
)

# Default planning / point-cloud resolution.
_DEFAULT_POINT_RES = 0.02
_DEFAULT_PLANNING_TIME = 3.0

# TOPP joint limits for the 11-DOF move group (torso 4 + left arm 7). Without
# these MPlib defaults to 1 rad/s / 1 rad/s^2, which squeezes the trajectory
# into a ~40 s crawl. Scale from the real actuator values: TOPP is
# time-optimal, so a mid-range fraction stays smooth while matching indoor
# human-like arm speed for WBC data.
_MOVE_GROUP_VEL_LIMITS = np.array(
    [0.5, 0.5, 0.5, 0.5, 7.12, 7.12, 8.3776, 8.3776, 10.472, 10.472, 10.472]
) * 0.55
# Low acceleration cap: at the shortcut corner waypoints the spline demands a
# fast joint direction change; the compliant PD (KP 800) cannot track acc
# spikes above ~2 rad/s^2 there (measured tracking error explodes to 0.4 rad).
_MOVE_GROUP_ACC_LIMITS = np.full(11, 2.0)

# MPlib's plan_qpos samples the TOPP trajectory at ``time_step`` (0.1 s).
# ``arm_trajectory_follow`` advances one 1/60 s simulation step per trajectory
# point, so the samples are re-sampled to the simulation dt before execution --
# playing 10 Hz samples at the 60 Hz cadence would run the motion 6x too fast
# and make the PD position/velocity references fight (the "weird" arm motion).
_SIM_DT = 1.0 / 60.0

_ARM_JOINT_NAMES_BY_SIDE = {
    side: tuple(f"{side}_arm_joint{i}" for i in range(1, 8))
    for side in ("left", "right")
}
_ARM_LIMITS_BY_SIDE = {
    side: (
        np.array([R1PRO_JOINT_LIMITS[n][0] if R1PRO_JOINT_LIMITS[n][0] is not None else -1e9 for n in names]),
        np.array([R1PRO_JOINT_LIMITS[n][1] if R1PRO_JOINT_LIMITS[n][1] is not None else 1e9 for n in names]),
    )
    for side, names in _ARM_JOINT_NAMES_BY_SIDE.items()
}
_ARM_LIMITS_LO, _ARM_LIMITS_HI = _ARM_LIMITS_BY_SIDE["left"]


def _require_side(side: str) -> str:
    if side not in _ARM_SLICE_BY_SIDE:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return side


def mplib_qpos_from_joint_positions(joint_positions: dict[str, float]) -> np.ndarray:
    """Build MPlib's 22-joint user vector from a live named observation.

    Keeping the non-planned arm at its measured configuration makes sequential
    left/right use safe: the selected arm is checked against the other arm's
    actual static geometry instead of an assumed all-zero posture.
    """
    return np.asarray([float(joint_positions.get(name, 0.0)) for name in _USER_JOINT_NAMES])


def resample_trajectory(
    position: np.ndarray,
    velocity: np.ndarray | None,
    acceleration: np.ndarray | None,
    dt_out: float,
    dt_in: float = 0.1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Resample a uniformly-sampled trajectory (``dt_in``) to ``dt_out``.

    The TOPP output from MPlib is a piecewise polynomial sampled at 10 Hz;
    this re-samples it at the simulation dt. Each interval is rebuilt as a
    cubic Hermite segment pinned to the sampled position *and* velocity
    endpoints -- exact for the TOPP piecewise-polynomial curve, with no
    overshoot at corners (a plain CubicSpline through the samples overshoots
    wherever the velocity profile has a sharp turn). The velocity derivative
    therefore comes out kinematically consistent with the position.

    Returns ``(position, velocity, acceleration, times)`` all at ``dt_out``;
    velocity/acceleration are None when the inputs are None.
    """
    if dt_out <= 0 or dt_in <= 0:
        raise ValueError("dt_out and dt_in must be positive")
    position = np.asarray(position, dtype=float)
    if position.ndim != 2:
        raise ValueError("position must be (n, dof)")
    n = position.shape[0]
    if n < 2:
        raise ValueError("resample_trajectory needs at least 2 points")
    t_in = np.linspace(0.0, (n - 1) * dt_in, n)
    t_out = np.arange(0.0, t_in[-1] + dt_out / 2.0, dt_out)
    if velocity is None:
        if n == 2:
            alpha = (t_out / max(t_in[-1], 1e-12))[:, None]
            pos_out = position[0] + alpha * (position[1] - position[0])
            return pos_out, None, None, t_out
        spline = CubicSpline(t_in, position, axis=0, bc_type="natural")
        pos_out = spline(t_out)
        return pos_out, None, None, t_out
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != position.shape:
        raise ValueError("velocity must match position shape")
    spline = CubicHermiteSpline(t_in, position, velocity, axis=0)
    pos_out = spline(t_out)
    vel_out = spline(t_out, 1)
    acc_out = None if acceleration is None else spline(t_out, 2)
    return pos_out, vel_out, acc_out, t_out


def _table_point_cloud(
    scene: SceneModel,
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_yaw: float = 0.0,
    local_radius_m: float | None = 2.0,
) -> np.ndarray:
    """Sample scene obstacles as a point cloud in the *base* frame.

    MPlib assumes the robot base is at the world origin, so world-frame obstacle
    points must be shifted into the base frame (subtract the base position).
    A non-zero base yaw is handled by rotating world points by ``-base_yaw``;
    planning in a translated-only frame was a latent bug after navigation.
    """
    bx, by = float(base_xy[0]), float(base_xy[1])
    c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))

    def to_base(points_world: np.ndarray) -> np.ndarray:
        shifted = points_world[:, :2] - np.array([bx, by])
        rotated = np.stack([c * shifted[:, 0] + s * shifted[:, 1], -s * shifted[:, 0] + c * shifted[:, 1]], axis=1)
        return np.column_stack([rotated, points_world[:, 2]])
    points: list[np.ndarray] = []
    res = _DEFAULT_POINT_RES
    for obj in scene.objects:
        if not obj.physics.collision_enabled:
            continue
        # The arm planner only needs geometry that can intersect the arm's
        # local workspace.  A room may contain metres of fencing that is
        # essential for base navigation but cannot be reached by a 1.5 m arm.
        # Culling distant AABBs keeps the MPlib/FCL world small without
        # weakening collision checking near the robot.
        if local_radius_m is not None:
            if obj.type.value == "cuboid":
                hx, hy = obj.size[0] / 2.0, obj.size[1] / 2.0
            else:
                hx = hy = float(obj.radius)
            dx = max(abs(float(obj.pos[0]) - bx) - hx, 0.0)
            dy = max(abs(float(obj.pos[1]) - by) - hy, 0.0)
            if float(np.hypot(dx, dy)) > float(local_radius_m):
                continue
        margin = float(obj.physics.planning_margin or 0.0)
        if obj.type.value == "cuboid":
            sx, sy, sz = obj.size
            x0, x1 = obj.pos[0] - sx / 2 - margin, obj.pos[0] + sx / 2 + margin
            y0, y1 = obj.pos[1] - sy / 2 - margin, obj.pos[1] + sy / 2 + margin
            z_bot, z_top = obj.pos[2] - sz / 2 - margin, obj.pos[2] + sz / 2 + margin
            # Top face, inflated upward by 5 cm so the planner keeps the whole
            # arm above the surface (a point-cloud at the exact surface lets a
            # thin link brush the edge).
            z_top_i = z_top + 0.05
            xs = np.arange(x0, x1 + 1e-6, res)
            ys = np.arange(y0, y1 + 1e-6, res)
            gx, gy = np.meshgrid(xs, ys)
            pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z_top_i)], axis=1)
            points.append(to_base(pts))
            # Sparse vertical skirts along the four edges: enough to stop the
            # arm cutting around the table horizontally (a full skirt turned the
            # obstacle into a solid box that blocked valid overhead paths).
            zs = np.linspace(z_bot, z_top, 4)
            for ex, ey in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
                # Vertical strip at each of the four corners.
                pts = np.stack(
                    [np.full(zs.size, ex), np.full(zs.size, ey), zs], axis=1
                )
                points.append(to_base(pts))
            # Horizontal edge skirts (the arm brushing the table edge at height).
            for ex in (x0, x1):
                pts = np.stack(
                        [np.full(ys.size, ex), ys, np.full(ys.size, z_top_i)], axis=1
                    )
                points.append(to_base(pts))
            for ey in (y0, y1):
                pts = np.stack(
                        [xs, np.full(xs.size, ey), np.full(xs.size, z_top_i)], axis=1
                    )
                points.append(to_base(pts))
        elif obj.type.value == "cylinder":
            r, h = obj.radius + margin, obj.height + 2.0 * margin
            res = _DEFAULT_POINT_RES
            n_ring = max(8, int(2 * np.pi * r / res))
            n_z = max(2, int(h / res))
            for iz in range(n_z):
                z = obj.pos[2] - h / 2 + (iz + 0.5) * (h / n_z)
                ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
                ring = np.stack(
                    [
                        obj.pos[0] + r * np.cos(ang),
                        obj.pos[1] + r * np.sin(ang),
                        np.full(n_ring, z),
                    ],
                    axis=1,
                )
                points.append(to_base(ring))
    if not points:
        return np.zeros((0, 3))
    return np.concatenate(points, axis=0)


def build_planner(side: str = "left"):
    """Construct the MPlib planner (cached per call; cheap, ~0.1s).

    Real TOPP joint limits are passed in: without them MPlib defaults to
    1 rad/s / 1 rad/s^2, which squeezes every trajectory into a crawl.
    """
    from mplib import Planner

    side = _require_side(side)
    return Planner(
        urdf=str(_PLAN_URDF),
        srdf=str(_PLAN_SRDF),
        move_group=_MOVE_GROUP_BY_SIDE[side],
        joint_vel_limits=_MOVE_GROUP_VEL_LIMITS,
        joint_acc_limits=_MOVE_GROUP_ACC_LIMITS,
        verbose=False,
    )


def plan_arm_path(
    planner,
    q_cur: np.ndarray,
    q_goal: np.ndarray,
    scene: SceneModel,
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_yaw: float = 0.0,
    planning_time: float = _DEFAULT_PLANNING_TIME,
    point_resolution: float = _DEFAULT_POINT_RES,
    kin: Any = None,
    min_ee_z: float | None = None,
    local_radius_m: float | None = 2.0,
    speed_scale: float = 0.12,
    mplib_attempts: int = 3,
    allow_rrt_fallback: bool = True,
    rrt_connect_mode: str = "fallback",
    side: str = "left",
    full_q_current: np.ndarray | None = None,
    shortcut_iterations: int = 96,
    direct_path: bool = True,
) -> dict:
    """Plan a collision-free, smoothed joint-space path from ``q_cur`` to
    ``q_goal`` (7-DOF left-arm configs) around ``scene`` obstacles.

    ``base_xy``/``base_yaw`` are the live base world pose; obstacle points are
    transformed into the planner's base frame (MPlib assumes the base at the
    origin).

    ``kin`` (optional Pinocchio kinematics) and ``min_ee_z`` (optional, base
    frame; defaults to the highest table top in the scene) add a Cartesian
    height floor to the path straightening -- see :func:`shortcut_path`.

    The OMPL path is straightened with verified shortcutting and re-TOP'ped;
    the returned trajectory is sampled directly on the 60 Hz simulation dt so
    the execution layer can play one point per simulation step.

    Returns:
        {
          "success": bool,
          "position": np.ndarray (N, 7) joint trajectory @ 60 Hz,
          "velocity": np.ndarray (N, 7) | None,
          "acceleration": np.ndarray (N, 7) | None,
          "times": np.ndarray (N,) sample times (s),
          "duration": float,
          "winding": float (joint-space path/straight ratio),
          "status": str (MPlib status),
          "reason": str | None,
        }
    """
    from mplib import Planner  # noqa: F401  (type hint)

    side = _require_side(side)
    arm_slice = _ARM_SLICE_BY_SIDE[side]
    q_cur = np.asarray(q_cur, dtype=np.float64)
    q_goal = np.asarray(q_goal, dtype=np.float64)
    if q_cur.shape != (7,) or q_goal.shape != (7,):
        raise ValueError("plan_arm_path expects a 7-DOF arm configuration")
    q_reference = np.zeros(22, dtype=np.float64) if full_q_current is None else np.asarray(full_q_current, dtype=np.float64).copy()
    if q_reference.shape != (22,):
        raise ValueError(f"full_q_current must have shape (22,), got {q_reference.shape}")

    # Full 22-dim qpos: torso at current, arm at goal.
    def _full(arm7: np.ndarray) -> np.ndarray:
        q = q_reference.copy()
        q[arm_slice] = arm7
        return q

    # Obstacle point cloud from the current scene (base frame).
    points = _table_point_cloud(scene, base_xy=base_xy, base_yaw=base_yaw, local_radius_m=local_radius_m)
    if len(points):
        planner.update_point_cloud(points, resolution=point_resolution, name="scene")
    else:
        planner.remove_point_cloud("scene")

    if min_ee_z is None:
        tables = [o.top_z for o in scene.objects if o.type.value == "cuboid"]
        # A hard floor slightly below the table top: joint-space paths around
        # the table legitimately dip the wrist below the top surface when
        # passing beside it; a floor at the exact top rejects every such path.
        min_ee_z = float(max(tables)) - 0.2 if tables else None
    if kin is not None and min_ee_z is not None:
        # The start/goal end-effector heights are part of the task: the floor
        # must not be above them, otherwise every candidate containing the
        # endpoints (all of them) is rejected.
        min_ee_z = min(
            min_ee_z, float(kin.fk(q_cur)[0][2]), float(kin.fk(q_goal)[0][2])
        )

    # A collision-free straight joint interpolation is the preferred path for
    # nearby closed-loop segments.  It is deterministic, minimum-motion, and
    # avoids invoking randomized OMPL for every 1 cm lift/descent step.  The
    # same MPlib environment and self-collision checks are used at a dense
    # resolution, so this is a proof-backed fast path rather than a collision
    # bypass.  Distant/nonlinear motions still fall through to MPlib.
    direct = np.linspace(q_cur, q_goal, max(12, int(np.max(np.abs(q_goal - q_cur)) / 0.02) + 1))
    from r1pro_data_gen.methods.collision import CollisionChecker, check_path, obstacles_from_scene
    direct_checker = CollisionChecker(
        kin,
        obstacles_from_scene(scene, exclude=(), include_ground=True),
    ) if kin is not None else None
    direct_free = (
        direct_checker is not None
        and all(
            direct_checker.is_collision_free(q, base_xy=base_xy, base_yaw=base_yaw)
            for q in direct
        )
        and path_collision_free(
            planner, direct, scene, base_xy=base_xy, base_yaw=base_yaw,
            dense=8, side=side, full_q_current=q_reference,
        )[0]
    )
    if direct_path and direct_free:
        if kin is None or min(
            float(kin.fk(q)[0][2]) for q in direct
        ) >= (min_ee_z if min_ee_z is not None else -np.inf):
            pos60, vel60, acc60 = _minimum_jerk_trajectory(direct, speed_scale=speed_scale, side=side)
            if (
                all(
                    direct_checker.is_collision_free(
                        q,
                        base_xy=base_xy,
                        base_yaw=base_yaw,
                    )
                    for q in pos60
                )
                and path_collision_free(
                    planner, pos60, scene, base_xy=base_xy, base_yaw=base_yaw,
                    dense=8, side=side, full_q_current=q_reference,
                )[0]
            ):
                times = np.arange(len(pos60), dtype=np.float64) * _SIM_DT
                return {
                    "success": True, "position": pos60, "velocity": vel60,
                    "acceleration": acc60, "times": times,
                    "duration": float(times[-1]) if len(times) else 0.0,
                    "dt": _SIM_DT, "winding": 1.0, "ee_winding": _ee_winding(pos60, kin),
                    "status": "DirectVerified", "reason": None,
                }

    last_failure = {
        "status": "NoAttempt",
        "reason": "no MPlib attempt was run",
        "failure_stage": "mplib_plan",
    }

    def _plan_once() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
        """One OMPL attempt: plan -> verified shortcut -> re-TOPP -> verify.

        Returns (position, velocity, acceleration, winding) at the 60 Hz
        grid, or None when the attempt is unusable (planning failed or the
        shortcut candidate collided).
        """
        nonlocal last_failure
        try:
            result = planner.plan_qpos(
                [_full(q_goal)],
                _full(q_cur),
                # Sample the TOPP trajectory directly on the simulation dt (as
                # the reference r1p project does).
                time_step=_SIM_DT,
                planning_time=planning_time,
                fixed_joint_indices=_TORSO_IDX,
                # Keep the raw OMPL path (no short-cutting): MPlib's
                # ``simplify=True`` shortcuts without dense verification and
                # its linear interpolation cuts through obstacles between the
                # sparse samples. We straighten the path ourselves below,
                # verifying every shortcut segment.
                simplify=False,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            last_failure = {
                "status": type(exc).__name__,
                "reason": str(exc) or "MPlib raised an exception",
                "failure_stage": "mplib_exception",
            }
            return None
        raw_status = str(result.get("status", "MissingStatus"))
        if raw_status != "Success":
            last_failure = {
                "status": raw_status,
                "reason": f"MPlib plan_qpos returned {raw_status}",
                "failure_stage": "mplib_plan",
            }
            return None
        # TOPP returns the selected 11-DOF move group (torso + chosen arm),
        # not the 22-DOF user-joint vector passed to plan_qpos.
        position = np.asarray(result["position"], dtype=np.float64)[:, _MOVE_GROUP_ARM_SLICE]
        # Straighten the winding RRT-Connect path with verified shortcutting.
        # The raw path wanders in joint space: the end-effector then sweeps a
        # large, winding arc in Cartesian space (measured: 2.45 m of EE motion
        # for a 0.78 m displacement) -- the "weird" motion. Every shortcut
        # segment is verified collision-free before it is accepted.
        shortened = shortcut_path(
            planner, position, iterations=max(0, int(shortcut_iterations)), kin=kin, ee_min_z=min_ee_z, side=side,
            full_q_current=q_reference,
        )
        if len(shortened) < 3:
            shortened = np.concatenate(
                [shortened[:1], (shortened[:1] + shortened[-1:]) / 2.0, shortened[-1:]]
            )
        candidate, qds, qdds = _minimum_jerk_trajectory(shortened, speed_scale=speed_scale, side=side)

        def _verify(cand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
            """Run the dense MPlib/hppfcl/height verification and score."""
            if not path_collision_free(planner, cand, scene, base_xy=base_xy, base_yaw=base_yaw, dense=20, side=side, full_q_current=q_reference)[0]:
                return None
            if direct_checker is not None and not check_path(
                direct_checker,
                list(cand),
                base_xy=base_xy,
                dense=4,
                base_yaw=base_yaw,
            )[0]:
                return None
            if kin is not None and min_ee_z is not None:
                for q in cand:
                    if float(kin.fk(q)[0][2]) < min_ee_z:
                        return None
            w = _joint_winding(cand)
            ee_w = _ee_winding(cand, kin)
            min_z = float(min(kin.fk(q)[0][2] for q in cand)) if kin is not None else 9.0
            height_penalty = max(0.0, 0.85 - min_z) * 4.0
            ee_penalty = max(0.0, ee_w - 1.25) * 0.7
            return cand, qds, qdds, w, ee_w, w + ee_penalty + height_penalty

        verified = _verify(candidate)
        if verified is None:
            # The cubic-spline min-jerk pass can overshoot between the verified
            # shortcut vertices and clip an obstacle (observed on tabletop
            # pregrasps).  The shortcut polyline itself is verified segment by
            # segment, so re-sample it piecewise-linearly (which cannot move off
            # the polyline) and only give up if that also fails verification.
            linear_candidate, linear_qds, linear_qdds = _linear_resample(
                shortened, speed_scale=speed_scale, side=side
            )
            qds, qdds = linear_qds, linear_qdds
            verified = _verify(linear_candidate)
            if verified is None:
                last_failure = {
                    "status": raw_status,
                    "reason": "smoothed and linear fallback trajectories both failed collision verification",
                    "failure_stage": "smoothed_hppfcl_collision",
                }
                return None
            candidate = linear_candidate
        return verified

    # OMPL is randomized: paths vary between straight and winding. Try several
    # seeds and keep the best-scoring verified candidate.
    best = None
    # Three independently seeded attempts retain OMPL robustness while
    # avoiding unbounded native planner churn in a live Isaac process.  The
    # verified shortcut and dense collision pass still reject every unsafe
    # candidate.
    rrt_connect_report: dict | None = None

    def _try_rrt_connect() -> dict | None:
        """Run the deterministic RRT-Connect fallback through all four gates.

        Returns the verified trajectory dict, or None after recording why the
        fallback itself failed (diagnosed via ``rrt_connect_report``).
        """
        nonlocal rrt_connect_report
        from r1pro_data_gen.methods.navigation.rrt import RRTConnectPlanner
        checker = CollisionChecker(
            kin, obstacles_from_scene(scene, exclude=(), include_ground=True)
        ) if kin is not None else None
        if checker is None:
            rrt_connect_report = {"status": "RRTConnectExhausted", "failure_stage": "rrt_connect_plan", "reason": "collision checker unavailable"}
            return None
        for seed in (11, 23, 37):
            planner_rrt = RRTConnectPlanner(
                kin, checker, step=0.20, max_iters=3000, seed=seed,
            )
            ok, rrt_path, stats = planner_rrt.plan(
                q_cur,
                q_goal,
                base_xy=base_xy,
                base_yaw=base_yaw,
            )
            if not ok:
                rrt_connect_report = {
                    "status": "RRTConnectExhausted",
                    "failure_stage": "rrt_connect_plan",
                    "reason": "no collision-free tree join within the iteration budget",
                    "seed": seed,
                    **stats,
                }
                continue
            ok, _, _ = check_path(
                checker,
                rrt_path,
                base_xy=base_xy,
                dense=20,
                base_yaw=base_yaw,
            )
            if not ok:
                rrt_connect_report = {"status": "RRTConnectExhausted", "failure_stage": "rrt_connect_collision", "reason": "raw tree path failed the dense hppfcl check", "seed": seed}
                continue
            shortened = shortcut_path(
                planner, np.asarray(rrt_path, dtype=np.float64), iterations=max(0, int(shortcut_iterations) * 2),
                kin=kin, ee_min_z=min_ee_z, side=side, full_q_current=q_reference,
            )
            # Densify the polyline before quintic smoothing: minimum-jerk
            # blending overshoots corners, and on a sparse waypoint list that
            # overshoot can clip an obstacle the polyline itself cleared
            # (observed as "smoothed path failed the MPlib dense check" on
            # carried-object retract edges).  Short segments make the
            # overshoot negligible while preserving the geometric path.
            dense_shortened = [shortened[0]]
            for prev, nxt in zip(shortened[:-1], shortened[1:]):
                span = float(np.linalg.norm(nxt - prev))
                steps = max(1, int(np.ceil(span / 0.02)))
                for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
                    dense_shortened.append(prev + (nxt - prev) * t)
            shortened = np.asarray(dense_shortened, dtype=np.float64)
            pos60, vel60, acc60 = _minimum_jerk_trajectory(shortened, speed_scale=speed_scale, side=side)
            ok, _, _ = check_path(
                checker,
                pos60,
                base_xy=base_xy,
                dense=20,
                base_yaw=base_yaw,
            )
            if not ok:
                rrt_connect_report = {"status": "RRTConnectExhausted", "failure_stage": "rrt_connect_collision", "reason": "smoothed path failed the dense hppfcl check", "seed": seed}
                continue
            if not path_collision_free(
                planner, pos60, scene, base_xy=base_xy, base_yaw=base_yaw,
                dense=12, side=side, full_q_current=q_reference,
            )[0]:
                rrt_connect_report = {"status": "RRTConnectExhausted", "failure_stage": "rrt_connect_collision", "reason": "smoothed path failed the MPlib dense check", "seed": seed}
                continue
            times = np.arange(len(pos60), dtype=np.float64) * _SIM_DT
            rrt_connect_report = None
            return {
                "success": True, "position": pos60, "velocity": vel60,
                "acceleration": acc60, "times": times,
                "duration": float(times[-1]) if len(times) else 0.0,
                "dt": _SIM_DT, "winding": _joint_winding(pos60),
                "ee_winding": _ee_winding(pos60, kin),
                "status": "RRTConnectVerified", "reason": None,
            }
        return None

    for _ in range(max(1, int(mplib_attempts))):
        attempt = _plan_once()
        if attempt is not None:
            if best is None or attempt[5] < best[5]:
                best = attempt
        elif rrt_connect_mode == "second_opinion" and allow_rrt_fallback:
            # Real GPU processes have shown OMPL failing on every candidate
            # while the same request succeeds elsewhere; waiting out the full
            # attempt budget adds no information.  Give the deterministic
            # RRT-Connect a chance right after each failed OMPL attempt.
            second = _try_rrt_connect()
            if second is not None:
                return second
    if best is None and allow_rrt_fallback and rrt_connect_mode != "off":
        # MPlib's OMPL backend can exhaust its randomized candidates even when
        # a narrow tabletop passage is reachable.  The simulator-independent
        # hppfcl checker bounds this fallback; every edge and the final dense
        # path are collision checked before the trajectory is returned.
        fallback = _try_rrt_connect()
        if fallback is not None:
            return fallback
    if best is None:
        failure = dict(last_failure)
        if rrt_connect_report is not None:
            # Surface the fallback's own verdict instead of leaving the stale
            # OMPL timeout as the only diagnostic.
            failure.update(rrt_connect_report)
        return {
            "success": False,
            "position": np.asarray([q_cur]),
            "velocity": None,
            "acceleration": None,
            "status": failure["status"],
            "reason": failure["reason"],
            "failure_stage": failure["failure_stage"],
        }

    position, velocity, acceleration, winding, ee_winding, _score = best
    # Enforce the same velocity/acceleration contract the sim joints have
    # (reference r1p project, _enforce_reference_limits): clip to joint limits,
    # then stretch time globally until the sampled velocity/acceleration stay
    # within limits. MPlib interpolation can produce small spikes at planner
    # segment seams or at a clipped limit; PhysX reacts to those with hard
    # clamping, which reads as arm jitter.
    # Keep the planner's requested speed contract through the final limiter.
    # Omitting ``speed_scale`` here silently widened the reference envelope to
    # the physical joint limits (the helper default is 1.0), so a waypoint
    # requested at 0.12 could still be emitted at full speed and saturate the
    # Isaac drive during gravity-loaded reaches.
    pos60, vel60, acc60 = _enforce_reference_limits(
        position, side=side, speed_scale=speed_scale
    )
    # Final dense collision verification on the executed trajectory.
    if not path_collision_free(planner, pos60, scene, base_xy=base_xy, base_yaw=base_yaw, dense=20, side=side, full_q_current=q_reference)[0]:
        return {
            "success": False,
            "position": pos60,
            "velocity": vel60,
            "acceleration": acc60,
            "status": "collision",
            "reason": "final trajectory collides after limit enforcement",
            "failure_stage": "final_collision",
        }
    # The limiter may stretch the trajectory in time (more samples); the grid
    # stays uniform at the simulation dt.
    times = np.arange(len(pos60), dtype=np.float64) * _SIM_DT
    return {
        "success": True,
        "position": pos60,
        "velocity": vel60,
        "acceleration": acc60,
        "times": times,
        "duration": float(times[-1]) if len(times) else 0.0,
        "dt": _SIM_DT,
        "winding": float(winding),
        "ee_winding": float(ee_winding),
        "status": "Success",
        "reason": None,
    }


def retime_and_validate_path(
    planner,
    geometric_path: np.ndarray,
    scene: SceneModel,
    *,
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_yaw: float = 0.0,
    kin: Any = None,
    speed_scale: float = 0.12,
    side: str = "left",
    full_q_current: np.ndarray | None = None,
) -> dict[str, Any]:
    """Retime a complete waypoint path and re-certify the executed samples.

    Segment planners deliberately stop at every endpoint.  Re-splining all
    adjacent segments that share collision semantics removes those artificial
    stops while preserving a zero-velocity boundary before contact motion.
    """
    side = _require_side(side)
    path = np.asarray(geometric_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
        raise ValueError("geometric_path must have shape (n>=2, 7)")
    position, velocity, acceleration = _minimum_jerk_trajectory(
        path,
        speed_scale=float(speed_scale),
        side=side,
    )
    reference_report = validate_reference_trajectory(
        position,
        side=side,
        dt=_SIM_DT,
        speed_scale=float(speed_scale),
    )
    if not reference_report["valid"]:
        return {
            "success": False,
            "status": "reference_limits",
            "reason": "; ".join(reference_report["reasons"]),
            "failure_stage": "sequence_reference_limits",
        }
    velocity = np.asarray(reference_report["velocity"], dtype=np.float64)
    acceleration = np.asarray(reference_report["acceleration"], dtype=np.float64)
    free, collision = path_collision_free(
        planner,
        position,
        scene,
        base_xy=base_xy,
        base_yaw=base_yaw,
        dense=20,
        side=side,
        full_q_current=full_q_current,
    )
    if not free:
        return {
            "success": False,
            "status": "collision",
            "reason": "retimed waypoint trajectory failed MPlib collision verification",
            "failure_stage": "sequence_mplib_collision",
            "collision": collision,
        }
    if kin is not None:
        from r1pro_data_gen.methods.collision import (
            CollisionChecker,
            check_path,
            obstacles_from_scene,
        )

        checker = CollisionChecker(
            kin,
            obstacles_from_scene(scene, exclude=(), include_ground=True),
        )
        valid, link, index = check_path(
            checker,
            list(position),
            base_xy=base_xy,
            base_yaw=base_yaw,
            dense=8,
        )
        if not valid:
            return {
                "success": False,
                "status": "collision",
                "reason": "retimed waypoint trajectory failed hpp-fcl collision verification",
                "failure_stage": "sequence_hppfcl_collision",
                "collision": link,
                "collision_index": index,
            }
    times = np.arange(len(position), dtype=np.float64) * _SIM_DT
    return {
        "success": True,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "times": times,
        "duration": float(times[-1]) if len(times) else 0.0,
        "dt": _SIM_DT,
        "winding": _joint_winding(position),
        "ee_winding": _ee_winding(position, kin),
        "status": "SequenceVerified",
        "reason": None,
    }


def _joint_winding(path: np.ndarray) -> float:
    """Joint-space path length / straight-line distance (1 = straight line)."""
    path = np.asarray(path, dtype=np.float64)
    seg = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(path[-1] - path[0]))
    return seg / max(straight, 1e-9)


def _ee_winding(path: np.ndarray, kin: Any | None) -> float:
    """End-effector path length / displacement ratio for candidate ranking."""
    if kin is None or len(path) < 2:
        return 1.0
    ee = np.asarray([kin.fk(np.asarray(q, dtype=float))[0] for q in path])
    length = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())
    displacement = float(np.linalg.norm(ee[-1] - ee[0]))
    return length / max(displacement, 1e-6)


# Velocity/acceleration limits for the time-stretch limiter (arm slice).
_ARM_VEL_LIMITS = np.asarray(R1PRO_ARM_VELOCITY_LIMITS, dtype=np.float64)
# Per-60 Hz-step normalized velocity change (reference project value): the
# velocity reference may change by at most this fraction of the limit per step.
_NORMALIZED_ACCEL_STEP = 0.10
_REFERENCE_LIMIT_TOL = 1e-4


def validate_reference_trajectory(
    position: np.ndarray,
    *,
    side: str = "left",
    dt: float = _SIM_DT,
    speed_scale: float = 1.0,
) -> dict[str, Any]:
    """Validate the exact reference samples that will be sent to the controller."""
    side = _require_side(side)
    samples = np.asarray(position, dtype=np.float64)
    reasons: list[str] = []
    if samples.ndim != 2 or samples.shape[1] != 7 or len(samples) < 2:
        return {
            "valid": False,
            "reasons": ("position_shape",),
            "velocity": np.empty((0, 7), dtype=np.float64),
            "acceleration": np.empty((0, 7), dtype=np.float64),
        }
    if not np.isfinite(dt) or dt <= 0.0:
        return {
            "valid": False,
            "reasons": ("invalid_dt",),
            "velocity": np.zeros_like(samples),
            "acceleration": np.zeros_like(samples),
        }
    if not np.all(np.isfinite(samples)):
        reasons.append("non_finite_position")
    lower, upper = _ARM_LIMITS_BY_SIDE[side]
    if np.any(samples < lower[None, :] - _REFERENCE_LIMIT_TOL) or np.any(
        samples > upper[None, :] + _REFERENCE_LIMIT_TOL
    ):
        reasons.append("joint_limit")
    velocity = np.zeros_like(samples)
    velocity[1:] = np.diff(samples, axis=0) / float(dt)
    acceleration = np.zeros_like(samples)
    acceleration[1:] = np.diff(velocity, axis=0) / float(dt)
    if not np.all(np.isfinite(velocity)):
        reasons.append("non_finite_velocity")
    if not np.all(np.isfinite(acceleration)):
        reasons.append("non_finite_acceleration")
    bounded_scale = max(float(speed_scale), 1e-6)
    max_velocity = _ARM_VEL_LIMITS * bounded_scale
    max_acceleration = _NORMALIZED_ACCEL_STEP * max_velocity / float(dt)
    if np.any(np.abs(velocity) > max_velocity[None, :] * (1.0 + _REFERENCE_LIMIT_TOL)):
        reasons.append("velocity_limit")
    if np.any(
        np.abs(acceleration) > max_acceleration[None, :] * (1.0 + _REFERENCE_LIMIT_TOL)
    ):
        reasons.append("acceleration_limit")
    return {
        "valid": not reasons,
        "reasons": tuple(reasons),
        "velocity": velocity,
        "acceleration": acceleration,
    }


def _enforce_reference_limits(
    position: np.ndarray,
    max_iterations: int = 12,
    side: str = "left",
    speed_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip to joint limits and stretch time until velocity/acceleration stay
    within the arm limits (reference r1p project, ``_enforce_reference_limits``).

    ``position`` is (n, 7) at the 60 Hz grid. Velocity/acceleration are derived
    from the *clipped* position by finite differences (so the reference and the
    integrator stay consistent), then global time stretching removes spikes at
    planner segment seams without changing the geometric path.  ``speed_scale``
    is applied to the velocity/acceleration ceilings so a scaled reference also
    passes the scaled validation gate.
    """
    lower, upper = _ARM_LIMITS_BY_SIDE[_require_side(side)]
    position = np.clip(position, lower[None, :], upper[None, :])
    max_vel = _ARM_VEL_LIMITS * max(float(speed_scale), 1e-6)
    max_acc = _NORMALIZED_ACCEL_STEP * max_vel / _SIM_DT
    for _ in range(max_iterations):
        vel = np.zeros_like(position)
        vel[1:] = np.diff(position, axis=0) / _SIM_DT
        acc = np.zeros_like(position)
        acc[1:] = np.diff(vel, axis=0) / _SIM_DT
        vel_ratio = float(np.max(np.abs(vel) / max_vel))
        acc_ratio = float(np.max(np.abs(acc) / max_acc))
        if max(vel_ratio, acc_ratio) <= 1.0001:
            return position, vel, acc
        time_scale = max(vel_ratio, np.sqrt(acc_ratio), 1.0)
        old = np.arange(position.shape[0], dtype=np.float64)
        new = np.linspace(
            0.0, old[-1], max(position.shape[0] + 1, int(np.ceil(old[-1] * time_scale)) + 1)
        )
        stretched = np.stack(
            [np.interp(new, old, position[:, j]) for j in range(position.shape[1])], axis=-1
        )
        position = np.clip(stretched, lower[None, :], upper[None, :])
    vel = np.zeros_like(position)
    vel[1:] = np.diff(position, axis=0) / _SIM_DT
    acc = np.zeros_like(position)
    acc[1:] = np.diff(vel, axis=0) / _SIM_DT
    return position, vel, acc


def _linear_resample(
    geometric_path: np.ndarray,
    speed_scale: float = 0.12,
    side: str = "left",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-sample a verified polyline on the 60 Hz grid by linear segments.

    Unlike the cubic-spline min-jerk pass, piecewise-linear interpolation stays
    exactly on the (already verified) shortcut segments, so it cannot overshoot
    into an obstacle between vertices.  Velocity/acceleration come from finite
    differences and are typically noisier than the spline; use this only as a
    collision-safe fallback when the smoothed candidate is rejected.
    """
    lower, upper = _ARM_LIMITS_BY_SIDE[_require_side(side)]
    path = np.asarray(geometric_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
        raise ValueError("geometric_path must have shape (n>=2, 7)")
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8]
    path = path[keep]
    if len(path) < 2:
        position = np.repeat(path[-1][None, :], 3, axis=0)
        return position, np.zeros_like(position), np.zeros_like(position)
    allowed_velocity = np.maximum(_ARM_VEL_LIMITS * float(speed_scale), 1e-4)
    joint_travel = np.sum(np.abs(np.diff(path, axis=0)), axis=0)
    duration = max(0.50, float(np.max(1.875 * joint_travel / allowed_velocity)))
    count = max(3, int(np.ceil(duration / _SIM_DT)) + 1)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    u = np.linspace(0.0, float(cumulative[-1]), count)
    position = np.column_stack(
        [np.interp(u, cumulative, path[:, i]) for i in range(7)]
    )
    # Piecewise-linear joints switch direction abruptly at the polyline
    # vertices; finite-difference velocity/acceleration spikes there would fail
    # the reference gate.  Stretch time so the derived signals stay within the
    # arm limits without changing the (already verified) geometric path.
    return _enforce_reference_limits(position, side=side, speed_scale=speed_scale)


def _minimum_jerk_trajectory(
    geometric_path: np.ndarray,
    speed_scale: float = 0.12,
    max_iterations: int = 10,
    side: str = "left",
    min_duration_s: float = R1PRO_ARM_MIN_TRAJECTORY_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a collision path into a C2, zero-velocity endpoint reference.

    A planner path is geometric: its vertices are not control-time commands.
    Feeding those vertices through TOPP preserves abrupt direction changes at
    shortcut corners.  Here a natural cubic spline rounds the geometric path,
    while a quintic minimum-jerk phase law gives zero velocity and acceleration
    at both ends. Callers must collision-check the returned curve because any
    geometric smoothing can move between the original polyline vertices.
    """
    lower, upper = _ARM_LIMITS_BY_SIDE[_require_side(side)]
    path = np.asarray(geometric_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
        raise ValueError("geometric_path must have shape (n>=2, 7)")
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8]
    path = path[keep]
    if len(path) < 2:
        # A closed-loop correction can legitimately ask for a target already
        # reached within numerical precision. CubicSpline requires strictly
        # increasing coordinates, so return a short stationary C2 reference
        # instead of constructing a zero-length spline.
        position = np.repeat(np.asarray(geometric_path[-1], dtype=np.float64)[None, :], 3, axis=0)
        return position, np.zeros_like(position), np.zeros_like(position)
    coordinate = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    coordinate /= max(float(coordinate[-1]), 1e-9)
    spline = CubicSpline(coordinate, path, axis=0, bc_type="natural")

    allowed_velocity = np.maximum(_ARM_VEL_LIMITS * float(speed_scale), 1e-4)
    allowed_acceleration = _NORMALIZED_ACCEL_STEP * allowed_velocity / _SIM_DT
    joint_travel = np.sum(np.abs(np.diff(path, axis=0)), axis=0)
    duration = max(
        max(3.0 * _SIM_DT, float(min_duration_s)),
        float(np.max(1.875 * joint_travel / allowed_velocity)),
    )
    for _ in range(max_iterations):
        count = max(3, int(np.ceil(duration / _SIM_DT)) + 1)
        time = np.linspace(0.0, duration, count)
        u = time / duration
        phase = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        position = np.asarray(spline(phase), dtype=np.float64)
        # A smoothed path outside a hard joint limit is rejected by returning
        # the original polyline through the legacy limiter; clipping a spline
        # would create a visible flat spot at the limit.
        if np.any(position < lower[None, :] - 1e-7) or np.any(position > upper[None, :] + 1e-7):
            return _enforce_reference_limits(
                _dense_interp(path), side=side, speed_scale=speed_scale
            )

        # The controller keeps the final position after this command stream.
        # Represent that first hold sample explicitly so finite differences see
        # the deceleration to zero.  Checking only ``np.gradient`` on the
        # unextended curve misses this boundary: its manually zeroed endpoint
        # derivative can pass while the next repeated command creates a one-step
        # acceleration spike when semantic trajectory groups are joined.
        executed_position = np.concatenate(
            [position, np.repeat(position[-1:], 2, axis=0)],
            axis=0,
        )
        velocity = np.zeros_like(executed_position)
        velocity[1:] = np.diff(executed_position, axis=0) / _SIM_DT
        acceleration = np.zeros_like(executed_position)
        acceleration[1:] = np.diff(velocity, axis=0) / _SIM_DT
        velocity_ratio = float(np.max(np.abs(velocity) / allowed_velocity))
        acceleration_ratio = float(
            np.max(np.abs(acceleration) / allowed_acceleration)
        )
        ratio = max(velocity_ratio, np.sqrt(acceleration_ratio), 1.0)
        if ratio <= 1.001:
            return executed_position, velocity, acceleration
        duration *= ratio * 1.02
    return executed_position, velocity, acceleration


def _dense_interp(traj: np.ndarray, dense_step: float = 0.02) -> np.ndarray:
    """Subdivide a joint trajectory so consecutive points are at most
    ``dense_step`` rad apart (prevents interpolation between sparse samples
    from cutting through an obstacle)."""
    if len(traj) < 2:
        return traj
    out: list[np.ndarray] = []
    for q0, q1 in zip(traj[:-1], traj[1:]):
        dist = float(np.max(np.abs(q1 - q0)))
        n = max(1, int(np.ceil(dist / dense_step)))
        for i in range(n):
            out.append(q0 + (q1 - q0) * (i / n))
    out.append(traj[-1])
    return np.asarray(out, dtype=np.float64)


def path_collision_free(
    planner, trajectory: np.ndarray, scene: SceneModel,
    base_xy: tuple[float, float] = (0.0, 0.0), base_yaw: float = 0.0, dense: int = 20,
    side: str = "left",
    full_q_current: np.ndarray | None = None,
) -> tuple[bool, str | None]:
    """Validate a joint trajectory with dense interpolation against the scene
    obstacle world. Returns (collision_free, colliding_link)."""
    arm_slice = _ARM_SLICE_BY_SIDE[_require_side(side)]
    q_reference = np.zeros(22) if full_q_current is None else np.asarray(full_q_current, dtype=float).copy()
    if q_reference.shape != (22,):
        raise ValueError(f"full_q_current must have shape (22,), got {q_reference.shape}")
    points = _table_point_cloud(scene, base_xy=base_xy, base_yaw=base_yaw, local_radius_m=2.0)
    if len(points):
        planner.update_point_cloud(points, resolution=_DEFAULT_POINT_RES, name="scene")
    else:
        planner.remove_point_cloud("scene")

    def _full(arm7: np.ndarray) -> np.ndarray:
        q = q_reference.copy()
        q[arm_slice] = arm7
        return q

    for i, (q0, q1) in enumerate(zip(trajectory[:-1], trajectory[1:])):
        for t in np.linspace(0, 1, dense + 1):
            q = _full(np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * t)
            if planner.check_for_env_collision(q) or planner.check_for_self_collision(q):
                return False, f"segment {i}"
    return True, None


def shortcut_path(
    planner,
    path: np.ndarray,
    iterations: int = 8,
    max_joint_step: float = 0.02,
    rng_seed: int = 0,
    kin: Any = None,
    ee_min_z: float | None = None,
    side: str = "left",
    full_q_current: np.ndarray | None = None,
) -> np.ndarray:
    """Straighten a collision-free joint path by verified shortcutting.

    Repeatedly replaces a random sub-path with the straight joint-space
    segment, but only when the segment is *verified* collision-free: the
    segment is sampled densely (at most ``max_joint_step`` apart) and every
    sample is checked against the environment and self-collision. This is the
    OMPL PathSimplifier pattern done on the dense, safe path -- MPlib's own
    ``simplify=True`` shortcuts without dense verification, and its linear
    interpolation then cuts through obstacles between the sparse samples.

    When ``kin`` and ``ee_min_z`` are given, each sample must additionally
    keep the end-effector at or above ``ee_min_z`` (base frame): joint-space
    straightening alone can produce visually weird low swoops (e.g. the wrist
    dipping under the table edge while still being collision-free), so the
    Cartesian height floor keeps the shortcut natural-looking.

    The input path must already be collision-free; the output stays
    collision-free and preserves the endpoints.
    """
    arm_slice = _ARM_SLICE_BY_SIDE[_require_side(side)]
    q_reference = np.zeros(22) if full_q_current is None else np.asarray(full_q_current, dtype=float).copy()
    if q_reference.shape != (22,):
        raise ValueError(f"full_q_current must have shape (22,), got {q_reference.shape}")
    path = np.asarray(path, dtype=np.float64).copy()
    rng = np.random.default_rng(rng_seed)
    n = len(path)
    if n < 4:
        return path

    def _full(arm7: np.ndarray) -> np.ndarray:
        q = q_reference.copy()
        q[arm_slice] = arm7
        return q

    def segment_free(q0: np.ndarray, q1: np.ndarray) -> bool:
        dist = float(np.max(np.abs(q1 - q0)))
        steps = max(2, int(np.ceil(dist / max_joint_step)))
        for t in np.linspace(0.0, 1.0, steps + 1):
            q = _full(q0 + (q1 - q0) * t)
            if planner.check_for_env_collision(q) or planner.check_for_self_collision(q):
                return False
            if kin is not None and ee_min_z is not None:
                ee = kin.fk(q0 + (q1 - q0) * t)[0]
                if float(ee[2]) < ee_min_z:
                    return False
        return True

    # First perform a deterministic farthest-visible pass. Random pair
    # shortcutting can miss the one useful table-edge waypoint even after many
    # iterations, leaving a visibly winding RRT path. Starting at each retained
    # waypoint and connecting to the furthest collision-free future sample
    # produces a small, ordered polyline around the obstacle (typically
    # start -> clear-table-edge -> goal) without changing either endpoint.
    greedy = [path[0]]
    index = 0
    while index < len(path) - 1:
        remaining = len(path) - index - 1
        if remaining == 1:
            next_index = len(path) - 1
        else:
            candidate_count = min(48, remaining)
            candidates = np.unique(
                np.linspace(index + 1, len(path) - 1, candidate_count, dtype=int)
            )[::-1]
            next_index = index + 1
            for candidate_index in candidates:
                if segment_free(path[index], path[candidate_index]):
                    next_index = int(candidate_index)
                    break
        greedy.append(path[next_index])
        index = next_index
    path = np.asarray(greedy, dtype=np.float64)

    for _ in range(iterations):
        if len(path) < 4:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        if not segment_free(path[i], path[j]):
            continue
        path = np.concatenate([path[: i + 1], path[j:]])
    return path


__all__ = [
    "build_planner",
    "mplib_qpos_from_joint_positions",
    "path_collision_free",
    "plan_arm_path",
    "resample_trajectory",
]
