"""R1Pro chassis kinematics: 3-wheel steer-and-drive motion model.

Pure Python. The R1Pro base has three independently steered, independently
driven wheels (verified from the USDA on 2026-08-08):

    wheel 1 (front-right): steer at (0.169,  0.28), wheel axis Y
    wheel 2 (front-left):  steer at (0.169, -0.28), wheel axis Y
    wheel 3 (rear):        steer at (-0.327,  0.0), wheel axis Y

Steer joints rotate around Z; wheel joints drive around Y. Wheel radius 0.07 m
(mesh extent x/z = +/-0.07). For a base velocity (vx, vy, omega) in the
base_link frame, the ground-contact velocity of wheel i at (xi, yi) is

    v_ground = (vx - omega * yi, vy + omega * xi)

so the steer angle is the direction of v_ground and the wheel speed is its
projection on that direction divided by the wheel radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WHEEL_RADIUS = 0.07  # m
# (name, x, y) in base_link frame, from USDA physics:localPos0 of steer joints.
STEER_JOINTS = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
WHEEL_JOINTS = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
# Steer position (x, y) per steer joint, in meters.
STEER_POSITIONS: dict[str, tuple[float, float]] = {
    "steer_motor_joint1": (0.169, 0.28),
    "steer_motor_joint2": (0.169, -0.28),
    "steer_motor_joint3": (-0.327, 0.0),
}
# steer i drives wheel i.
STEER_TO_WHEEL = {
    "steer_motor_joint1": "wheel_motor_joint1",
    "steer_motor_joint2": "wheel_motor_joint2",
    "steer_motor_joint3": "wheel_motor_joint3",
}


def default_footprint_radius_m() -> float:
    """Return the conservative circumscribed radius of the R1Pro chassis."""
    half_x = max(abs(x) for x, _ in STEER_POSITIONS.values()) + 0.06
    half_y = max(abs(y) for _, y in STEER_POSITIONS.values()) + 0.06
    return math.hypot(half_x, half_y)


@dataclass(frozen=True, slots=True)
class WheelCommand:
    """Per-wheel steer angle (rad) and wheel speed (rad/s)."""

    steer_angle: float
    wheel_speed: float


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def shortest_steer_command(
    current_angle: float,
    desired_angle: float,
    wheel_speed: float,
) -> WheelCommand:
    """Map a desired heading onto the shorter of two equivalent steers.

    Driving at ``desired`` and driving at ``desired + π`` with reversed wheel
    speed are the same ground velocity. Choosing the smaller joint move stops
    a module from spinning almost a full turn when a few degrees (or a
    reverse) would do.
    """
    current = _wrap_pi(current_angle)
    direct = _wrap_pi(float(desired_angle) - current)
    flipped = _wrap_pi(float(desired_angle) + math.pi - current)
    if abs(flipped) < abs(direct) - 1.0e-9:
        delta = flipped
        speed = -float(wheel_speed)
    else:
        delta = direct
        speed = float(wheel_speed)
    # Keep the command on the same side of ±π as the measured joint. Wrapping
    # 3.1 to -3.1 would make the limited revolute steer spin the long way.
    commanded = max(-math.pi, min(math.pi, current + delta))
    return WheelCommand(steer_angle=commanded, wheel_speed=speed)


def wheel_commands(
    vx: float,
    vy: float,
    omega: float,
    current_steer: dict[str, float] | None = None,
) -> dict[str, WheelCommand]:
    """Steer/wheel commands for a base velocity (vx, vy, omega).

    vx/vy in m/s and omega in rad/s, expressed in the base_link frame
    (x forward, y left, z up). Returns a dict keyed by steer joint name.
    When ``current_steer`` is provided, each module uses the shortest
    equivalent steer relative to its measured angle.
    """
    commands: dict[str, WheelCommand] = {}
    measured = current_steer or {}
    for steer_name, (xi, yi) in STEER_POSITIONS.items():
        gx = vx - omega * yi
        gy = vy + omega * xi
        steer_angle = math.atan2(gy, gx)
        speed = math.hypot(gx, gy) / WHEEL_RADIUS
        if steer_name in measured:
            commands[steer_name] = shortest_steer_command(
                measured[steer_name], steer_angle, speed
            )
        else:
            commands[steer_name] = WheelCommand(steer_angle=steer_angle, wheel_speed=speed)
    return commands


def base_velocity_for_drive(
    wheel_speeds: dict[str, float],
    steer_angles: dict[str, float],
) -> tuple[float, float, float]:
    """Inverse model: base velocity from measured wheel speeds/steer angles.

    Solves the least-squares system for (vx, vy, omega) given the three wheel
    ground-velocity equations. Useful for verifying that commanded motion
    actually happened.
    """
    import numpy as np

    rows: list[list[float]] = []
    rhs: list[float] = []
    for steer_name, speed in wheel_speeds.items():
        xi, yi = STEER_POSITIONS[steer_name]
        theta = steer_angles[steer_name]
        # speed * r = gx*cos + gy*sin = (vx - w*yi)*cos + (vy + w*xi)*sin
        rows.append([math.cos(theta), math.sin(theta), -yi * math.cos(theta) + xi * math.sin(theta)])
        rhs.append(speed * WHEEL_RADIUS)
    solution, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    return (float(solution[0]), float(solution[1]), float(solution[2]))
