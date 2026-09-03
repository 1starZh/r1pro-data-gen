"""Contract tests for reusable joint-mask phase skills."""

from types import SimpleNamespace

import pytest

from r1pro_data_gen.robot.joints import JointMapping
from r1pro_data_gen.simulation.isaac_sim.adapter import R1ProSimAdapter
from r1pro_data_gen.skills.posture.joint_mask import JointMaskLock, JointMaskUnlock
from tests.support import TensorStub


class _MaskAdapter:
    def __init__(self):
        self.calls = []
        self.steps = 0

    def lock_joint_mask(self, **kwargs):
        self.calls.append(("lock", kwargs))
        return {"locked_joint_names": ["torso_joint1"], "active_joint_names": ["left_arm_joint1"]}

    def unlock_joint_mask(self):
        self.calls.append(("unlock", {}))

    def joint_lock_metrics(self):
        return {
            "locked_joint_count": 1.0,
            "max_locked_joint_error": 0.002,
            "max_root_tilt_rad": 0.01,
            "joint_lock_torso_current_error_rad": 0.001,
            "joint_lock_torso_max_error_rad": 0.002,
        }

    def joint_lock_diagnostics(self):
        return {"joint_lock_max_error_joints": {"torso": "torso_joint1"}}

    def step(self):
        self.steps += 1


def test_joint_mask_allow_mode_is_forwarded_and_settled():
    adapter = _MaskAdapter()
    result = JointMaskLock().execute(
        adapter,
        mask_mode="allow",
        joint_groups=["left_arm", "left_gripper"],
        lock_root=True,
        settle_steps=3,
    )
    assert result.success
    assert adapter.steps == 3
    assert adapter.calls[0][1]["mask_mode"] == "allow"
    assert adapter.calls[0][1]["joint_groups"] == ("left_arm", "left_gripper")
    assert result.metrics["max_locked_joint_error"] == 0.002
    assert result.metrics["joint_lock_torso_current_error_rad"] == 0.001
    assert result.metrics["joint_lock_torso_max_error_rad"] == 0.002
    assert result.details["joint_lock_max_error_joints"] == {"torso": "torso_joint1"}


def test_joint_mask_unlock_delegates_to_adapter():
    adapter = _MaskAdapter()
    result = JointMaskUnlock().execute(adapter)
    assert result.success
    assert adapter.calls == [("unlock", {})]


def test_adapter_reports_current_and_phase_max_errors_by_group():
    adapter = object.__new__(R1ProSimAdapter)
    adapter.mapping = JointMapping(
        joint_names=("wheel_joint1", "wheel_joint2", "torso_joint1"),
        group_exprs={"wheel": r"^wheel_", "torso": r"^torso_"},
    )
    adapter._joint_lock_targets = {
        "wheel_joint1": 0.0,
        "wheel_joint2": 0.0,
        "torso_joint1": 0.0,
    }
    adapter._joint_lock_groups = ("wheel", "torso")
    adapter._joint_lock_max_error = 0.0
    adapter._joint_lock_max_error_by_group = {}
    adapter._joint_lock_max_joint_by_group = {}
    adapter._joint_lock_max_root_tilt = 0.0
    data = SimpleNamespace(
        joint_pos=[TensorStub([0.01, 0.03, 0.02])],
        root_quat_w=[TensorStub([1.0, 0.0, 0.0, 0.0])],
    )
    adapter.robot = SimpleNamespace(data=data)

    adapter._update_joint_lock_metrics()
    data.joint_pos = [TensorStub([0.005, 0.01, 0.025])]
    metrics = adapter.joint_lock_metrics()
    diagnostics = adapter.joint_lock_diagnostics()

    assert metrics["current_locked_joint_error"] == pytest.approx(0.025)
    assert metrics["max_locked_joint_error"] == pytest.approx(0.03)
    assert metrics["joint_lock_wheel_current_error_rad"] == pytest.approx(0.01)
    assert metrics["joint_lock_wheel_max_error_rad"] == pytest.approx(0.03)
    assert metrics["joint_lock_torso_current_error_rad"] == pytest.approx(0.025)
    assert metrics["joint_lock_torso_max_error_rad"] == pytest.approx(0.025)
def test_adapter_attachment_state_reports_live_constraints():
    adapter = object.__new__(R1ProSimAdapter)
    adapter._grasp_joints = {
        "item": {"body_name": "left_gripper_finger_midpoint"},
    }

    assert adapter.attachment_state() == {
        "item": "left_gripper_finger_midpoint",
    }
