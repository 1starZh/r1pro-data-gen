"""Pure base-motion skill behavior and safety clamps."""

from __future__ import annotations

import pytest

from r1pro_data_gen.skills import BaseFollowPath, BaseVelocitySet
from r1pro_data_gen.skills.mobility.base_motion import (
    BaseMoveTo,
    _clamp_velocity,
    _forward_tracking_command,
    _slew,
)

from tests.support import FakeAdapter


def test_slew_limits_one_step_acceleration():
    assert _slew(0.0, 0.5, max_accel=0.8, dt=1.0 / 60.0) == 0.8 / 60.0
    assert _slew(0.49, 0.50, max_accel=0.8, dt=1.0 / 60.0) == 0.50


def test_clamp_velocity_limits_magnitude():
    vx, vy, omega = _clamp_velocity(10.0, -5.0, 3.0, v_max=0.1, omega_max=0.2)
    assert vx == 0.1 and vy == -0.1 and omega == 0.2


def test_base_follow_path_drives_through_waypoints():
    result = BaseFollowPath().execute(
        FakeAdapter(), None, path=[[0.0, 0.0], [0.1, 0.0]], target_yaw=0.0,
        arrive_tol=0.5, max_steps=2,
    )
    assert result.success
    assert result.metrics["waypoints"] == 2.0


def test_base_velocity_set_requires_no_target():
    result = BaseVelocitySet().execute(
        FakeAdapter(), None, vx=0.05, vy=0.0, omega=0.0, duration=0.05,
    )
    assert result.success
    assert result.metrics["steps"] >= 1


def test_navigation_forward_tracker_never_commands_lateral_velocity():
    vx, vy, omega, heading_error = _forward_tracking_command(
        dx_world=1.0, dy_world=1.0, current_yaw=0.0,
        v_max=0.2, omega_max=0.4,
    )
    assert vy == 0.0
    assert vx > 0.0
    assert omega > 0.0
    assert heading_error > 0.0


def test_navigation_forward_tracker_turns_before_target_behind_robot():
    vx, vy, omega, _ = _forward_tracking_command(
        dx_world=-1.0, dy_world=0.0, current_yaw=0.0,
        v_max=0.2, omega_max=0.4,
    )
    assert vx == 0.0
    assert vy == 0.0
    assert abs(omega) == 0.4


def test_forward_tracker_keeps_speed_through_ninety_degree_corner():
    import math

    vx, vy, omega, heading = _forward_tracking_command(
        dx_world=0.0, dy_world=1.0, current_yaw=0.0,
        v_max=0.2, omega_max=0.4,
    )
    assert heading == pytest.approx(math.pi / 2.0)
    assert vx > 0.0
    assert vy == 0.0
    assert omega > 0.0


def test_drive_path_aborts_when_pose_stops_progressing():
    from r1pro_data_gen.skills.mobility.base_motion import (
        _STALL_WINDOW_STEPS,
        _drive_path,
    )

    adapter = FakeAdapter()
    metrics = _drive_path(
        adapter,
        [(0.0, 0.0), (1.0, 0.0)],
        v_max=0.2,
        omega_max=0.4,
        arrive_tol=0.02,
        max_steps=2000,
        step_hook=None,
    )
    assert metrics["success"] == 0.0
    assert metrics["steps"] <= _STALL_WINDOW_STEPS
    assert adapter.steps <= _STALL_WINDOW_STEPS


def test_base_navigate_skips_motion_when_already_at_target():
    from r1pro_data_gen.skills import BaseNavigateTo

    result = BaseNavigateTo().execute(
        FakeAdapter(), None, target=[0.01, 0.02, 0.0]
    )
    assert result.success
    assert result.metrics["steps"] == 0.0
    assert result.details["reason"] == "already at navigation target"


def test_lookahead_aims_past_a_corner_not_at_the_vertex():
    from r1pro_data_gen.skills.mobility.base_motion import _lookahead_point

    carrot = _lookahead_point(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        0.95,
        0.0,
        lookahead_m=0.5,
    )
    assert carrot[0] == pytest.approx(1.0, abs=0.05)
    assert carrot[1] > 0.2


def test_base_move_to_closes_position_before_heading(monkeypatch):
    import r1pro_data_gen.skills.mobility.base_motion as base_motion

    calls = []

    monkeypatch.setattr(base_motion, "_read_base", lambda _adapter: (0.0, 0.0, 1.0))

    def drive(_adapter, target, *_args, **kwargs):
        calls.append(("position", target, kwargs.get("position_only")))
        return {
            "success": 1.0,
            "steps": 3.0,
            "final_x": 0.2,
            "final_y": -0.1,
            "final_yaw": 1.0,
            "arrival_error_m": 0.0,
            "yaw_error_rad": 0.0,
        }

    def rotate(_adapter, target_yaw, *_args, **_kwargs):
        calls.append(("rotation", target_yaw))
        return {"success": 1.0, "steps": 2.0, "final_yaw": target_yaw, "yaw_error_rad": 0.0}

    monkeypatch.setattr(base_motion, "_drive_to", drive)
    monkeypatch.setattr(base_motion, "_rotate_in_place", rotate)
    result = BaseMoveTo().execute(FakeAdapter(), target=[0.2, -0.1, -0.5])

    assert calls == [
        ("position", (0.2, -0.1, 1.0), True),
        ("rotation", -0.5),
    ]
    assert result.success
    assert result.metrics["steps"] == 5.0


class _WheelLockRecordingAdapter(FakeAdapter):
    """Records wheel-lock releases; navigation must clear skill locks."""

    def __init__(self):
        super().__init__()
        self._wheels_locked = True
        self.joint_mask_locked = False
        self.unlock_calls = 0

    def unlock_wheels(self):
        self.unlock_calls += 1
        self._wheels_locked = False


def test_base_rotate_to_releases_residual_skill_wheel_lock():
    from r1pro_data_gen.skills import BaseRotateTo

    adapter = _WheelLockRecordingAdapter()
    result = BaseRotateTo(max_steps=2).execute(adapter, None, target_yaw=0.0)
    assert adapter.unlock_calls == 1
    assert result.skill == "base_rotate_to"


def test_base_rotate_to_keeps_task_joint_mask():
    from r1pro_data_gen.skills import BaseRotateTo

    adapter = _WheelLockRecordingAdapter()
    adapter.joint_mask_locked = True
    BaseRotateTo(max_steps=2).execute(adapter, None, target_yaw=0.0)
    assert adapter.unlock_calls == 0


def test_rotate_in_place_stops_at_chassis_yaw_band(monkeypatch):
    import r1pro_data_gen.skills.mobility.base_motion as base_motion

    monkeypatch.setattr(base_motion, "_read_base", lambda _adapter: (0.0, 0.0, 0.03))
    adapter = FakeAdapter()
    metrics = base_motion._rotate_in_place(
        adapter, 0.0, 0.2, 0.10, 2000, 0, None
    )
    assert metrics["success"] == 1.0
    assert metrics["steps"] == 0.0
    assert adapter.steps == 0


def test_brake_until_stopped_is_noop_without_velocity():
    from r1pro_data_gen.skills.mobility.base_motion import _brake_until_stopped

    adapter = FakeAdapter()
    assert _brake_until_stopped(adapter) == 0.0
    assert adapter.steps == 0


def test_brake_until_stopped_waits_while_chassis_is_spinning():
    from r1pro_data_gen.domain import Observation
    from r1pro_data_gen.skills.mobility.base_motion import _brake_until_stopped

    class _Spinning(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.yaw_rate = 0.20

        def read_observation(self, timestamp):
            del timestamp
            if self.steps >= 5:
                self.yaw_rate = 0.01
            return Observation(
                timestamp=0.0,
                joint_positions=dict(self._joint_positions),
                base_velocity=(0.0, 0.0, self.yaw_rate),
            )

    adapter = _Spinning()
    steps = _brake_until_stopped(adapter, max_steps=20)
    assert steps == 5.0
    assert adapter.steps == 5


def test_arm_ready_targets_step_toward_rest_pose():
    from r1pro_data_gen.robot.robot_config import R1PRO_ARM_READY_Q_BY_SIDE
    from r1pro_data_gen.skills.manipulation.arm import ARM_JOINTS_BY_SIDE
    from r1pro_data_gen.skills.mobility.base_motion import _ARM_READY_STEP_RAD, _arm_ready_targets

    joints = {name: 0.0 for name in ARM_JOINTS_BY_SIDE["left"]}
    adapter = FakeAdapter(joint_positions=joints)
    extra = _arm_ready_targets(adapter)
    ready_j2 = float(R1PRO_ARM_READY_Q_BY_SIDE["left"][1])
    assert extra["left_arm_joint2"] == _ARM_READY_STEP_RAD
    assert extra["left_arm_joint2"] < ready_j2
    assert extra["left_gripper_finger_joint1"] > 0.0


def test_steer_drive_scale_is_one_when_modules_are_aligned_or_unmeasured():
    from r1pro_data_gen.robot.chassis import STEER_JOINTS, wheel_commands
    from r1pro_data_gen.skills.mobility.base_motion import _steer_drive_scale

    cmds = wheel_commands(0.2, 0.0, 0.0)
    aligned = {name: cmds[name].steer_angle for name in STEER_JOINTS}
    assert _steer_drive_scale(aligned, cmds, STEER_JOINTS) == 1.0
    assert _steer_drive_scale({}, cmds, STEER_JOINTS) == 1.0


def test_steer_drive_scale_kills_wheel_speed_at_ninety_degrees():
    from r1pro_data_gen.robot.chassis import STEER_JOINTS, wheel_commands
    from r1pro_data_gen.skills.mobility.base_motion import _steer_drive_scale

    cmds = wheel_commands(0.0, 0.2, 0.0)
    parked = {name: 0.0 for name in STEER_JOINTS}
    assert _steer_drive_scale(parked, cmds, STEER_JOINTS) == 0.0


def test_stop_wheels_holds_measured_steer_angles():
    from r1pro_data_gen.robot.chassis import STEER_JOINTS, WHEEL_JOINTS
    from r1pro_data_gen.skills.mobility.base_motion import _stop_wheels_hold_steer

    class _Recording(FakeAdapter):
        def __init__(self):
            super().__init__(
                joint_positions={
                    "steer_motor_joint1": 0.5,
                    "steer_motor_joint2": -0.4,
                    "steer_motor_joint3": 0.1,
                }
            )
            self.commands = []

        def set_targets(self, position, velocity=None):
            self.commands.append((dict(position), dict(velocity or {})))
            super().set_targets(position, velocity)

    adapter = _Recording()
    _stop_wheels_hold_steer(adapter, STEER_JOINTS, WHEEL_JOINTS)
    position, velocity = adapter.commands[-1]
    assert position["steer_motor_joint1"] == 0.5
    assert position["steer_motor_joint2"] == -0.4
    assert position["steer_motor_joint3"] == 0.1
    assert all(value == 0.0 for value in velocity.values())
