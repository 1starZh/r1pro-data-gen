"""Chassis kinematics tests (pure logic, no Isaac Sim)."""

from __future__ import annotations

import math

import pytest

from r1pro_data_gen.robot import (
    WHEEL_RADIUS,
    base_velocity_for_drive,
    shortest_steer_command,
    wheel_commands,
)


def test_forward_drive_all_wheels_forward() -> None:
    cmds = wheel_commands(vx=0.5, vy=0.0, omega=0.0)
    for steer, cmd in cmds.items():
        assert cmd.steer_angle == pytest.approx(0.0, abs=1e-9)
        assert cmd.wheel_speed == pytest.approx(0.5 / WHEEL_RADIUS, rel=1e-9)


def test_rotation_about_center() -> None:
    omega = 1.0
    cmds = wheel_commands(vx=0.0, vy=0.0, omega=omega)
    # Wheel 1 at (0.169, 0.28): ground velocity = (-omega*yi, omega*xi)
    front_right = cmds["steer_motor_joint1"]
    assert front_right.steer_angle == pytest.approx(math.atan2(0.169, -0.28), rel=1e-9)
    assert front_right.wheel_speed == pytest.approx(
        math.hypot(0.28, 0.169) / WHEEL_RADIUS, rel=1e-9
    )
    # Wheel 3 at (-0.327, 0): velocity = (0, omega * xi) -> direction -Y
    # for a counter-clockwise turn (points behind the center move downward).
    rear = cmds["steer_motor_joint3"]
    assert rear.steer_angle == pytest.approx(-math.pi / 2, abs=1e-9)


def test_inverse_model_round_trip() -> None:
    vx, vy, omega = 0.3, 0.1, -0.4
    cmds = wheel_commands(vx=vx, vy=vy, omega=omega)
    speeds = {s: c.wheel_speed for s, c in cmds.items()}
    angles = {s: c.steer_angle for s, c in cmds.items()}
    est = base_velocity_for_drive(speeds, angles)
    assert est[0] == pytest.approx(vx, rel=1e-6)
    assert est[1] == pytest.approx(vy, rel=1e-6)
    assert est[2] == pytest.approx(omega, rel=1e-6)


def test_shortest_steer_prefers_reverse_over_a_half_turn() -> None:
    command = shortest_steer_command(0.0, 3.0, 1.0)
    assert abs(command.steer_angle) < 0.3
    assert command.wheel_speed == pytest.approx(-1.0)


def test_shortest_steer_does_not_wrap_across_pi() -> None:
    command = shortest_steer_command(3.0, -3.0, 1.0)
    assert command.steer_angle > 2.8
    assert command.wheel_speed > 0.0
