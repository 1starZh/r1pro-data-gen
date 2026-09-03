"""Robot articulation and control-chain contracts (no Isaac Sim)."""

from __future__ import annotations

import re

import pytest

from r1pro_data_gen.control import CommandRouter, ControllerConfig, JointGroup
from r1pro_data_gen.domain import ControlMode, Trajectory, TrajectoryPoint
from r1pro_data_gen.robot import R1PRO_JOINT_GROUP_EXPR, R1PRO_JOINT_LIMITS
from r1pro_data_gen.robot.joints import JointMapping

# 28 joints in the articulation's reported USD order.
R1PRO_JOINT_NAMES = (
    "steer_motor_joint1",
    "steer_motor_joint2",
    "steer_motor_joint3",
    "torso_joint1",
    "wheel_motor_joint1",
    "wheel_motor_joint2",
    "wheel_motor_joint3",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
    "left_arm_joint1",
    "right_arm_joint1",
    "left_arm_joint2",
    "right_arm_joint2",
    "left_arm_joint3",
    "right_arm_joint3",
    "left_arm_joint4",
    "right_arm_joint4",
    "left_arm_joint5",
    "right_arm_joint5",
    "left_arm_joint6",
    "right_arm_joint6",
    "left_arm_joint7",
    "right_arm_joint7",
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
)


def test_mapping_validates_against_real_articulation_names() -> None:
    mapping = JointMapping(
        joint_names=R1PRO_JOINT_NAMES,
        group_exprs=dict(R1PRO_JOINT_GROUP_EXPR),
    )
    mapping.validate()  # must not raise
    assert mapping.indices_of("wheel") == (4, 5, 6)
    assert mapping.names_of("left_arm") == tuple(
        n for n in R1PRO_JOINT_NAMES if n.startswith("left_arm_joint")
    )
    assert len(mapping.names_of("left_arm")) == 7


def test_mapping_rejects_overlapping_groups() -> None:
    mapping = JointMapping(
        joint_names=("torso_joint1",),
        group_exprs={"torso": "torso_joint.*", "all": ".*"},
    )
    with pytest.raises(ValueError, match="multiple groups"):
        mapping.validate()


def test_mapping_rejects_dead_group() -> None:
    mapping = JointMapping(
        joint_names=("torso_joint1",),
        group_exprs={"torso": "torso_joint.*", "wheel": "wheel_motor_joint.*"},
    )
    with pytest.raises(ValueError, match="matches no joints"):
        mapping.validate()


def test_mapping_rejects_unknown_group_lookup() -> None:
    mapping = JointMapping(joint_names=("torso_joint1",), group_exprs={"torso": "torso_joint.*"})
    with pytest.raises(KeyError):
        mapping.indices_of("missing")


def test_limits_cover_all_articulation_joints() -> None:
    missing = [n for n in R1PRO_JOINT_NAMES if n not in R1PRO_JOINT_LIMITS]
    assert not missing, f"missing limits: {missing}"


def test_safe_pose_within_limits() -> None:
    safe_pose = {
        "torso_joint1": 0.2,
        "left_arm_joint1": -0.5,
        "left_arm_joint2": 0.3,
    }
    for name, value in safe_pose.items():
        lower, upper = R1PRO_JOINT_LIMITS[name]
        assert lower is None or value >= lower, f"{name}={value} < {lower}"
        assert upper is None or value <= upper, f"{name}={value} > {upper}"


def test_workspace_pose_within_limits() -> None:
    # Workspace arm pose (backward swing, away from the table).
    # joint2=-0.3 exceeds the authored lower limit and must never be used.
    workspace_pose = {
        "left_arm_joint1": 0.5,
        "left_arm_joint2": -0.15,
    }
    for name, value in workspace_pose.items():
        lower, upper = R1PRO_JOINT_LIMITS[name]
        assert lower is None or value >= lower, f"{name}={value} < {lower}"
        assert upper is None or value <= upper, f"{name}={value} > {upper}"


def test_command_router_full_chain_with_group_config() -> None:
    # All groups are position-controlled in this pure control-chain contract.
    groups = tuple(
        JointGroup(
            name,
            tuple(n for n in R1PRO_JOINT_NAMES if re.match(expr, n)),
            ControlMode.POSITION,
        )
        for name, expr in R1PRO_JOINT_GROUP_EXPR.items()
    )
    router = CommandRouter(ControllerConfig(groups=groups))
    point = TrajectoryPoint(
        timestamp=1.0,
        joint_positions={n: 0.0 for n in R1PRO_JOINT_NAMES},
        stage="move_to_safe_pose",
    )
    command = router.command(point, observation=None, timestamp=1.0)  # type: ignore[arg-type]
    assert set(command.position_targets) == set(R1PRO_JOINT_NAMES)
    assert command.velocity_targets == {}
