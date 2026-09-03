"""Unit tests for the MPlib path wrapper's pure-logic pieces.

Only the resampling helper (and other non-Isaac logic) lives here: MPlib
planning itself is CPU-only but needs the planning URDF/SRDF assets, so it is
exercised on GPU verification runs rather than in the pure-logic suite.

Root cause context: MPlib's ``plan_qpos`` samples the TOPP trajectory at its
own ``time_step`` (0.1 s), while ``arm_trajectory_follow`` advances one
1/60 s simulation step per trajectory point. Playing the 10 Hz samples at the
60 Hz cadence plays the motion 6x too fast and makes the PD position/velocity
references fight each other -- the "weird" arm motion. ``resample_trajectory``
re-samples the TOPP output to the simulation dt so the 1-point-per-step
execution semantics reproduce the planned time parameterization.
"""

from __future__ import annotations

import numpy as np
import pytest

from r1pro_data_gen.methods.manipulation.mplib_path import (
    _enforce_reference_limits,
    _minimum_jerk_trajectory,
    resample_trajectory,
    validate_reference_trajectory,
)


def test_resample_preserves_duration_and_length():
    """60 Hz resample of a 10 Hz trajectory keeps the duration and matches the
    expected point count."""
    t10 = np.linspace(0.0, 10.0, 101)  # 100 intervals at 0.1 s
    # Smooth analytic trajectory: q(t) = sin(0.5 t), v = 0.5 cos(0.5 t)
    pos = np.column_stack([np.sin(0.5 * t10) * i for i in range(1, 8)])
    vel = np.column_stack([0.5 * np.cos(0.5 * t10) * i for i in range(1, 8)])
    acc = np.column_stack([-0.25 * np.sin(0.5 * t10) * i for i in range(1, 8)])

    pos60, vel60, acc60, t60 = resample_trajectory(pos, vel, acc, dt_out=1.0 / 60.0, dt_in=0.1)

    assert len(t60) == 601  # 10 s * 60 + 1
    assert pos60.shape == (601, 7)
    assert vel60.shape == pos60.shape
    assert acc60.shape == pos60.shape
    # Duration preserved (first/last sample coincide with the input times).
    assert abs(t60[0] - 0.0) < 1e-9
    assert abs(t60[-1] - 10.0) < 1e-9
    assert np.allclose(pos60[0], pos[0])
    assert np.allclose(pos60[-1], pos[-1])


def test_resample_matches_analytic_trajectory():
    """Resampled position/velocity track the analytic curve within tolerance."""
    t10 = np.linspace(0.0, 5.0, 51)
    pos = np.column_stack([np.sin(0.5 * t10) * i for i in range(1, 8)])
    vel = np.column_stack([0.5 * np.cos(0.5 * t10) * i for i in range(1, 8)])
    acc = np.column_stack([-0.25 * np.sin(0.5 * t10) * i for i in range(1, 8)])

    pos60, vel60, acc60, t60 = resample_trajectory(pos, vel, acc, dt_out=1.0 / 60.0, dt_in=0.1)

    exact_pos = np.column_stack([np.sin(0.5 * t60) * i for i in range(1, 8)])
    exact_vel = np.column_stack([0.5 * np.cos(0.5 * t60) * i for i in range(1, 8)])
    # Cubic resample of a smooth 10 Hz sample set: sub-milliradian error.
    assert np.abs(pos60 - exact_pos).max() < 1e-3
    assert np.abs(vel60 - exact_vel).max() < 1e-3
    # Velocity continuity: the analytic per-step velocity change is
    # |a| / 60 = 1.75 / 60 = 0.0292; the resample must not add any jump.
    step = np.abs(np.diff(vel60, axis=0)).max()
    assert step < 0.031


def test_resample_stationary_endpoints():
    """Endpoints of a start/stop trajectory have zero velocity after resample."""
    # Piecewise-linear q(t): ramp up 0 -> 0.1 (0..0.4 s), hold (0.4..1.7 s),
    # ramp down 0.1 -> 0 (1.7..2.2 s). Position samples are the integral of the
    # velocity samples, so the Hermite rebuild reproduces the ramps exactly.
    t10 = np.linspace(0.0, 2.2, 23)

    def q_of(t):
        return np.where(t < 0.4, 0.25 * t, np.where(t < 1.7, 0.1, 0.1 - 0.25 * (t - 1.7)))

    def v_of(t):
        return np.where(t < 0.4, 0.25, np.where(t < 1.7, 0.0, -0.25))

    pos = np.column_stack([q_of(t10) * (i / 7.0) for i in range(1, 8)])
    vel = np.column_stack([v_of(t10) * (i / 7.0) for i in range(1, 8)])
    acc = np.zeros_like(pos)

    pos60, vel60, _, _ = resample_trajectory(pos, vel, acc, dt_out=1.0 / 60.0, dt_in=0.1)
    # Endpoint velocities are reproduced exactly (the Hermite rebuild pins the
    # sampled derivative at every sample, including the first/last one).
    assert np.allclose(vel60[0], vel[0])
    assert np.allclose(vel60[-1], vel[-1])
    # Joint displacement per step is bounded (smooth execution at 60 Hz):
    # ramps move 0.25 rad/s -> ~0.004 rad per step, Hermite corners peak at
    # ~0.33 rad/s -> ~0.006 rad per step.
    step = np.abs(np.diff(pos60, axis=0)).max()
    assert step < 0.01


def test_resample_requires_positive_dt():
    t10 = np.linspace(0.0, 1.0, 11)
    pos = np.zeros((11, 7))
    with pytest.raises(ValueError):
        resample_trajectory(pos, None, None, dt_out=0.0, dt_in=0.1)
    with pytest.raises(ValueError):
        resample_trajectory(pos, None, None, dt_out=1.0 / 60.0, dt_in=0.0)


def test_linear_resample_stays_on_polyline_and_reaches_endpoints():
    """Linear fallback re-samples exactly along the verified shortcut path.

    The cubic-spline min-jerk pass can overshoot between verified shortcut
    vertices; the piecewise-linear fallback must stay on the polyline so it
    cannot clip an obstacle the shortcut already cleared.
    """
    from r1pro_data_gen.methods.manipulation.mplib_path import _linear_resample

    path = np.zeros((4, 7), dtype=float)
    path[1] = [-0.15, -0.03, -0.08, -0.25, 0.03, -0.04, 0.02]
    path[2] = [-0.32, -0.08, -0.12, -0.62, 0.08, -0.13, 0.05]
    path[3] = [-0.48, -0.10, -0.17, -0.91, 0.11, -0.20, 0.08]

    position, velocity, acceleration = _linear_resample(path, speed_scale=0.12)

    assert position.shape[0] > 60
    assert np.allclose(position[0], path[0])
    assert np.allclose(position[-1], path[-1])
    for index in range(7):
        assert position[:, index].min() >= path[:, index].min() - 1e-9
        assert position[:, index].max() <= path[:, index].max() + 1e-9
    assert validate_reference_trajectory(position, speed_scale=0.12)["valid"]


def test_minimum_jerk_default_floor_is_short_enough_for_humanlike_motion():
    from r1pro_data_gen.robot.robot_config import R1PRO_ARM_MIN_TRAJECTORY_S
    from r1pro_data_gen.methods.manipulation.mplib_path import _SIM_DT

    path = np.zeros((2, 7), dtype=float)
    path[1, 3] = -0.05
    position, _, _ = _minimum_jerk_trajectory(path, speed_scale=0.36)
    duration_s = (len(position) - 1) * _SIM_DT
    assert duration_s <= max(0.35, R1PRO_ARM_MIN_TRAJECTORY_S + 0.08)


def test_minimum_jerk_short_segment_can_drop_two_second_floor():
    path = np.zeros((2, 7), dtype=float)
    path[1, 3] = -0.05
    slow, _, _ = _minimum_jerk_trajectory(path, speed_scale=0.22, min_duration_s=2.0)
    fast, _, _ = _minimum_jerk_trajectory(path, speed_scale=0.22, min_duration_s=0.50)
    assert slow.shape[0] >= 120
    assert fast.shape[0] < 50


def test_minimum_jerk_trajectory_is_continuous_and_stationary_at_endpoints():
    """A geometric detour becomes one smooth 60 Hz command stream."""
    path = np.zeros((4, 7), dtype=float)
    path[1] = [-0.15, -0.03, -0.08, -0.25, 0.03, -0.04, 0.02]
    path[2] = [-0.32, -0.08, -0.12, -0.62, 0.08, -0.13, 0.05]
    path[3] = [-0.48, -0.10, -0.17, -0.91, 0.11, -0.20, 0.08]

    position, velocity, acceleration = _minimum_jerk_trajectory(path, speed_scale=0.12)
    assert position.shape[0] > 80
    assert np.allclose(position[0], path[0])
    assert np.allclose(position[-1], path[-1])
    assert np.allclose(velocity[[0, -1]], 0.0)
    assert np.allclose(acceleration[[0, -1]], 0.0)
    assert np.allclose(position[-3:], path[-1])
    assert validate_reference_trajectory(position, speed_scale=0.12)["valid"]
    # No command discontinuity or stop-start plateau at internal waypoints.
    assert np.max(np.abs(np.diff(position, axis=0))) < 0.02
    internal_speed = np.linalg.norm(np.diff(position, axis=0), axis=1)
    assert np.count_nonzero(internal_speed[10:-10] < 1e-6) == 0


def test_reference_validation_rejects_non_finite_and_limit_violations():
    position = np.zeros((3, 7), dtype=float)
    position[1, 0] = np.nan
    assert not validate_reference_trajectory(position)["valid"]
    assert "non_finite_position" in validate_reference_trajectory(position)["reasons"]

    position = np.zeros((3, 7), dtype=float)
    position[1, 0] = 100.0
    report = validate_reference_trajectory(position)
    assert not report["valid"]
    assert "joint_limit" in report["reasons"]


def test_reference_validation_rejects_velocity_and_acceleration_spikes():
    position = np.zeros((4, 7), dtype=float)
    position[1:, 0] = 0.5
    report = validate_reference_trajectory(position, speed_scale=0.1)

    assert not report["valid"]
    assert "velocity_limit" in report["reasons"]
    assert "acceleration_limit" in report["reasons"]


def test_reference_limiter_honors_requested_speed_scale():
    """The final planner limiter must preserve the semantic speed budget.

    This guards the generic planner/executor boundary: a slow waypoint must
    not silently become a full-joint-limit command just because it came from
    the MPlib branch rather than the direct branch.
    """
    path = np.zeros((3, 7), dtype=float)
    path[-1, 3] = -1.0
    position, velocity, _ = _enforce_reference_limits(
        path, speed_scale=0.12
    )
    assert len(position) > len(path)
    report = validate_reference_trajectory(position, speed_scale=0.12)
    assert report["valid"]
    assert np.max(np.abs(velocity), axis=0)[3] <= 8.3776 * 0.12 * 1.0001


def test_mplib_live_qpos_preserves_both_arms() -> None:
    from r1pro_data_gen.methods.manipulation.mplib_path import mplib_qpos_from_joint_positions

    positions = {
        "torso_joint1": 0.1,
        "left_arm_joint1": -0.2,
        "left_gripper_finger_joint1": 0.03,
        "right_arm_joint1": -0.7,
        "right_gripper_finger_joint2": -0.04,
    }
    q = mplib_qpos_from_joint_positions(positions)
    assert q.shape == (22,)
    assert q[0] == 0.1
    assert q[4] == -0.2
    assert q[11] == 0.03
    assert q[13] == -0.7
    assert q[21] == -0.04


def test_ee_winding_scores_cartesian_sweep():
    """The planner's quality score must see a large EE arc, not only q length."""
    from r1pro_data_gen.methods.manipulation.mplib_path import _ee_winding

    class Kin:
        def fk(self, q):
            q = np.asarray(q)
            return np.array([q[0], 0.6 * np.sin(4.0 * np.pi * q[0]), 1.0]), np.array([1.0, 0.0, 0.0, 0.0])

    path = np.zeros((41, 7))
    path[:, 0] = np.linspace(0.0, 1.0, 41)
    assert _ee_winding(path, Kin()) > 1.5


def test_ee_winding_is_one_for_straight_motion():
    from r1pro_data_gen.methods.manipulation.mplib_path import _ee_winding

    class Kin:
        def fk(self, q):
            return np.asarray(q[:3]), np.array([1.0, 0.0, 0.0, 0.0])

    path = np.zeros((5, 7))
    path[:, 0] = np.linspace(0.0, 1.0, 5)
    assert abs(_ee_winding(path, Kin()) - 1.0) < 1e-9


def test_empty_scene_clears_stale_point_cloud_and_preserves_raw_status():
    from types import SimpleNamespace

    from r1pro_data_gen.methods.manipulation.mplib_path import plan_arm_path

    class Planner:
        def __init__(self):
            self.removed = []

        def remove_point_cloud(self, name):
            self.removed.append(name)
            return True

        def plan_qpos(self, *_args, **_kwargs):
            return {"status": "RRTConnect Failed"}

    planner = Planner()
    scene = SimpleNamespace(objects=())
    out = plan_arm_path(
        planner,
        np.zeros(7),
        np.ones(7) * 0.1,
        scene,
        kin=None,
        mplib_attempts=1,
        allow_rrt_fallback=False,
    )

    assert planner.removed == ["scene"]
    assert not out["success"]
    assert out["status"] == "RRTConnect Failed"
    assert out["failure_stage"] == "mplib_plan"


def test_mplib_exception_keeps_failure_stage():
    from types import SimpleNamespace

    from r1pro_data_gen.methods.manipulation.mplib_path import plan_arm_path

    class Planner:
        def remove_point_cloud(self, _name):
            return True

        def plan_qpos(self, *_args, **_kwargs):
            raise RuntimeError("backend unavailable")

    out = plan_arm_path(
        Planner(),
        np.zeros(7),
        np.ones(7) * 0.1,
        SimpleNamespace(objects=()),
        kin=None,
        mplib_attempts=1,
        allow_rrt_fallback=False,
    )

    assert not out["success"]
    assert out["status"] == "RuntimeError"
    assert out["reason"] == "backend unavailable"
    assert out["failure_stage"] == "mplib_exception"


def test_collision_checker_world_position_applies_base_yaw():
    from r1pro_data_gen.methods.collision import CollisionChecker

    world = CollisionChecker._world_position(
        np.array([1.0, 0.0, 0.5]),
        base_xy=(2.0, 3.0),
        base_yaw=np.pi / 2.0,
    )

    assert np.allclose(world, [2.0, 4.0, 0.5])


# ---------------------------------------------------------------------------
# Path shortcutting (OMPL path straightening with collision verification)
# ---------------------------------------------------------------------------

class _FakePlanner:
    """Planner stub: a 2-D blocked box in (joint_a0 x joint_a1)."""

    def __init__(self, a0: int, lo0: float, hi0: float, a1: int, lo1: float, hi1: float):
        self.a0, self.lo0, self.hi0 = a0, lo0, hi0
        self.a1, self.lo1, self.hi1 = a1, lo1, hi1

    def check_for_env_collision(self, q: np.ndarray) -> bool:
        # The arm slice is at indices 4:11 in the padded qpos.
        q = np.asarray(q)
        if q.ndim == 1 and len(q) == 22:
            q = q[4:11]
        in0 = bool(self.lo0 < q[self.a0] < self.hi0)
        in1 = bool(self.lo1 < q[self.a1] < self.hi1)
        return in0 and in1

    def check_for_self_collision(self, q: np.ndarray) -> bool:
        return False


def test_shortcut_rejects_paths_through_obstacle():
    """A shortcut segment crossing the blocked region must be rejected: the
    shortcut must never cut through a collision."""
    from r1pro_data_gen.methods.manipulation.mplib_path import shortcut_path

    # Blocked box: joint0 in [0.4, 0.6] AND joint1 in [-0.2, 0.2]. The input
    # path routes joint1 up to 0.5 while joint0 crosses the box -- a collision-
    # free detour. Straightening the detour would put joint1 back into the
    # blocked band, so the shortcut must reject it.
    n = 200
    t = np.linspace(-1.0, 1.0, n)
    path = np.zeros((n, 7))
    path[:, 0] = t
    path[:, 1] = np.where((t > 0.3) & (t < 0.7), 0.5, 0.0)
    planner = _FakePlanner(0, 0.4, 0.6, 1, -0.2, 0.2)
    out = shortcut_path(planner, path, iterations=150, rng_seed=0)
    # Verify every segment of the output with dense interpolation: none may
    # cross the blocked box (the shortcut must route above it, not through it).
    for q0, q1 in zip(out[:-1], out[1:]):
        for t in np.linspace(0.0, 1.0, 50):
            q = q0 + (q1 - q0) * t
            in0 = 0.4 < q[0] < 0.6
            in1 = -0.2 < q[1] < 0.2
            assert not (in0 and in1), f"segment crosses the blocked box at {q}"
    # The straight-through line (joint1 == 0) is blocked, so the output must
    # keep a detour (more than the two endpoints).
    assert len(out) >= 3
    # Endpoints preserved.
    assert np.allclose(out[0], path[0])
    assert np.allclose(out[-1], path[-1])


def test_shortcut_straightens_collision_free_path():
    """With no obstacles the shortcut collapses the path to (almost) the two
    endpoints -- the straight segment is collision-free."""
    from r1pro_data_gen.methods.manipulation.mplib_path import shortcut_path

    n = 200
    path = np.zeros((n, 7))
    path[:, 0] = np.linspace(-1.0, 1.0, n)
    path[:, 1] = 0.5 * np.sin(np.linspace(0, 2 * np.pi, n))  # a winding detour
    planner = _FakePlanner(0, 100, 101, 1, 100, 101)  # no obstacle anywhere
    out = shortcut_path(planner, path, iterations=300, rng_seed=0)
    # Path much shorter than the original winding path.
    assert len(out) < n // 4
    assert np.allclose(out[0], path[0])
    assert np.allclose(out[-1], path[-1])


def test_shortcut_deterministically_keeps_only_required_detour():
    """The farthest-visible pass should retain a compact obstacle bypass."""
    from r1pro_data_gen.methods.manipulation.mplib_path import shortcut_path

    path = np.zeros((101, 7))
    path[:, 0] = np.linspace(-1.0, 1.0, 101)
    path[:, 1] = 0.55 * np.sin(np.linspace(0.0, np.pi, 101))
    planner = _FakePlanner(0, 0.35, 0.65, 1, -0.2, 0.2)
    out = shortcut_path(planner, path, iterations=0)
    assert 3 <= len(out) <= 6
    assert np.allclose(out[0], path[0])
    assert np.allclose(out[-1], path[-1])
    for q0, q1 in zip(out[:-1], out[1:]):
        for alpha in np.linspace(0.0, 1.0, 100):
            q = q0 + alpha * (q1 - q0)
            assert not (0.35 < q[0] < 0.65 and -0.2 < q[1] < 0.2)


class _FakeKin:
    """Kinematics stub whose end-effector z depends on joint1 only."""

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q)
        z = 1.3 - 0.5 * abs(q[1])  # high when joint1 ~ 0, low when |joint1| large
        return np.array([0.0, 0.0, z]), np.array([1.0, 0.0, 0.0, 0.0])


def test_shortcut_respects_min_ee_height():
    """A shortcut segment dipping the end-effector below the minimum height
    must be rejected -- joint-space straightening must not produce visually
    weird low swoops under the table."""
    from r1pro_data_gen.methods.manipulation.mplib_path import shortcut_path

    n = 100
    path = np.zeros((n, 7))
    path[:, 0] = np.linspace(-1.0, 1.0, n)
    path[:, 1] = 0.0  # EE stays at z = 1.3 - 0.5 * 0 = 1.3
    kin = _FakeKin()
    planner = _FakePlanner(0, 100, 101, 1, 100, 101)  # no env obstacles
    # Height floor below the path's EE height: straightening still works.
    free_out = shortcut_path(planner, path, iterations=50, rng_seed=0,
                             kin=kin, ee_min_z=1.2)
    assert len(free_out) < n // 4
    # Height floor ABOVE the path's EE height: no segment may be accepted,
    # the output stays the full input path.
    constrained = shortcut_path(planner, path, iterations=50, rng_seed=0,
                                kin=kin, ee_min_z=1.35)
    assert len(constrained) == n
    assert np.allclose(constrained, path)


# ---------------------------------------------------------------------------
# RRT-Connect second-opinion fallback (component B of the planner upgrade)
# ---------------------------------------------------------------------------

def _rrt_connect_fixtures(monkeypatch):
    """Shared fakes: a C-space ball obstacle seen by BOTH checker layers."""
    from types import SimpleNamespace
    import sys

    import r1pro_data_gen.methods.collision as collision_module

    lower = np.full(7, -2.0)
    upper = np.full(7, 2.0)
    center = np.zeros(7)

    class FakeKin:
        lower = np.full(7, -2.0)
        upper = np.full(7, 2.0)

        def fk(self, q):
            q = np.asarray(q, dtype=float)
            return np.asarray([0.5 * q[0], 0.1 * q[1], 1.2]), np.array([1.0, 0.0, 0.0, 0.0])

    def in_ball(arm7):
        return bool(np.linalg.norm(np.asarray(arm7, dtype=float) - center) <= 0.9)

    class FakeChecker:
        """hppfcl-side stand-in applying the same C-space ball."""

        def __init__(self, *args, **kwargs):
            pass

        def is_collision_free(self, q, base_xy=(0.0, 0.0), base_yaw=0.0):
            return not in_ball(np.asarray(q, dtype=float)[:7])

        def first_collision_link(self, q, base_xy=(0.0, 0.0), base_yaw=0.0):
            return None if self.is_collision_free(q, base_xy, base_yaw) else "fake"

    # plan_arm_path imports CollisionChecker inside its body from this module.
    monkeypatch.setattr(collision_module, "CollisionChecker", FakeChecker)
    assert "r1pro_data_gen.methods.manipulation.mplib_path" in sys.modules

    class FakePlanner:
        def remove_point_cloud(self, name):
            return True

        def update_point_cloud(self, points, resolution=None, name="scene"):
            return True

        def plan_qpos(self, *_args, **_kwargs):
            return {"status": "RRTConnect Failed. Timeout"}

        def check_for_env_collision(self, q):
            q = np.asarray(q)
            arm = q[4:11] if q.ndim == 1 and len(q) == 22 else q[:7]
            return in_ball(arm)

        def check_for_self_collision(self, q):
            return False

    scene = SimpleNamespace(objects=(), world=SimpleNamespace(ground=False))
    return FakeKin(), FakePlanner(), scene


def test_rrt_connect_fallback_success_status_and_gates(monkeypatch):
    """OMPL times out on every attempt; the deterministic fallback routes
    around the C-space ball through all four verification gates."""
    kin, planner, scene = _rrt_connect_fixtures(monkeypatch)
    from r1pro_data_gen.methods.manipulation.mplib_path import plan_arm_path

    out = plan_arm_path(
        planner,
        np.full(7, -1.5),
        np.full(7, 1.5),
        scene,
        kin=kin,
        mplib_attempts=1,
        allow_rrt_fallback=True,
        rrt_connect_mode="fallback",
        speed_scale=0.12,
    )

    assert out["success"]
    assert out["status"] == "RRTConnectVerified"
    pos60 = np.asarray(out["position"])
    distances = np.linalg.norm(pos60, axis=1)
    assert np.all(distances > 0.9)  # the detour really avoided the ball


def test_fallback_exhaustion_records_own_stage(monkeypatch):
    """When even the fallback cannot run, the failure names rrt_connect_plan
    instead of leaving the stale OMPL timeout as the only diagnostic."""
    kin, planner, scene = _rrt_connect_fixtures(monkeypatch)
    # A checker that rejects everything makes the fallback bail out
    # immediately while still passing through the recording path.
    import r1pro_data_gen.methods.collision as collision_module

    class ExhaustedChecker:
        def __init__(self, *args, **kwargs):
            pass

        def is_collision_free(self, q, base_xy=(0.0, 0.0), base_yaw=0.0):
            return False  # everything collides: no tree can grow

    monkeypatch.setattr(collision_module, "CollisionChecker", ExhaustedChecker)
    from r1pro_data_gen.methods.manipulation.mplib_path import plan_arm_path

    out = plan_arm_path(
        planner,
        np.full(7, -1.5),
        np.full(7, 1.5),
        scene,
        kin=kin,
        mplib_attempts=1,
        allow_rrt_fallback=True,
        rrt_connect_mode="fallback",
    )

    assert not out["success"]
    assert out["failure_stage"] == "rrt_connect_plan"
    assert out["status"] == "RRTConnectExhausted"
