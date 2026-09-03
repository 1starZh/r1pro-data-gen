"""Pure arm interpolation, orientation and gripper behavior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from r1pro_data_gen.domain import ContactEvent
from r1pro_data_gen.skills import GripperGrasp
from r1pro_data_gen.skills.manipulation.arm_motion import ArmMoveDirectional, direction_steps, rotate_quat_about_axis
from r1pro_data_gen.skills.core.base import SkillResult, stabilize_base

from tests.support import FakeAdapter, load_fixture_scene


def test_rotate_quat_about_world_x_rotates_z_axis():
    q = rotate_quat_about_axis(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), math.pi / 2)
    from scipy.spatial.transform import Rotation

    z_new = Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply([0.0, 0.0, 1.0])
    assert np.allclose(z_new, [0.0, -1.0, 0.0], atol=1e-6)


def test_direction_steps_linear_interpolation():
    steps = direction_steps(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), distance=0.3, step=0.1)
    assert len(steps) == 3
    assert np.allclose(steps[0], [0.0, 0.1, 1.0], atol=1e-9)
    assert np.allclose(steps[-1], [0.0, 0.3, 1.0], atol=1e-9)


def test_direction_steps_rejects_zero_direction():
    with pytest.raises(ValueError, match="non-zero"):
        direction_steps(np.zeros(3), np.zeros(3), distance=0.1, step=0.01)


@pytest.mark.parametrize(
    ("distance", "step"),
    ((0.0, 0.01), (-0.1, 0.01), (0.1, 0.0), (0.1, -0.01)),
)
def test_directional_rejects_non_positive_distance_or_step(distance, step):
    skill = ArmMoveDirectional(object(), np.ones(7), object())

    with pytest.raises(ValueError, match="positive"):
        skill.execute(
            object(),
            direction=[0.0, 0.0, -1.0],
            distance=distance,
            step=step,
        )


def test_directional_contact_mode_fails_without_progress_or_contact(monkeypatch):
    class Kin:
        def fk(self, q_arm):
            return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])

        def ik(self, target, quat, q_init):
            return type(
                "IKResult",
                (),
                {
                    "success": True,
                    "q_arm": np.asarray(q_init),
                    "position_error": 0.0,
                    "rotation_error": 0.0,
                },
            )()

    class Adapter:
        joint_mask_locked = True

        def __init__(self):
            self.joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}

        def read_observation(self, timestamp):
            return type("Obs", (), {"joint_positions": dict(self.joints)})()

        def set_targets(self, position, velocity):
            pass

        def step(self):
            pass

        def finger_contact_forces(self, side="left"):
            return (0.0, 0.0)

        def contact_events(self):
            return ()

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path._minimum_jerk_trajectory",
        lambda geometric, speed_scale, side: (np.asarray(geometric), None, None),
    )
    result = ArmMoveDirectional(Kin(), np.ones(7), object()).execute(
        Adapter(),
        direction=[0.0, 0.0, -1.0],
        distance=0.04,
        step=0.01,
        until_contact=True,
        object_name="item",
    )

    assert not result.success
    assert result.metrics["requested_distance_m"] == pytest.approx(0.04)
    assert result.metrics["actual_displacement_m"] == pytest.approx(0.0)
    assert result.metrics["contact_detected"] is False
    assert result.metrics["contact_object"] is None
    assert result.metrics["failure_code"] == "no_progress"


def test_directional_contact_mode_rejects_contact_with_wrong_object(monkeypatch):
    class Kin:
        def fk(self, q_arm):
            q = np.asarray(q_arm)
            return np.array([0.0, 0.0, 1.0 - q[0]]), np.array([1.0, 0.0, 0.0, 0.0])

        def ik(self, target, quat, q_init):
            q = np.asarray(q_init).copy()
            q[0] = 1.0 - float(target[2])
            return type(
                "IKResult",
                (),
                {
                    "success": True,
                    "q_arm": q,
                    "position_error": 0.0,
                    "rotation_error": 0.0,
                },
            )()

    class Adapter:
        joint_mask_locked = True

        def __init__(self):
            self.joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}

        def read_observation(self, timestamp):
            return type("Obs", (), {"joint_positions": dict(self.joints)})()

        def set_targets(self, position, velocity):
            self.joints.update(position)

        def step(self):
            pass

        def finger_contact_forces(self, side="left"):
            return (3.0, 3.0)

        def contact_events(self):
            from r1pro_data_gen.domain import ContactEvent

            return (ContactEvent(0.1, "left_finger", "support", 3.0),)

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path._minimum_jerk_trajectory",
        lambda geometric, speed_scale, side: (np.asarray(geometric), None, None),
    )
    result = ArmMoveDirectional(Kin(), np.ones(7), object()).execute(
        Adapter(),
        direction=[0.0, 0.0, -1.0],
        distance=0.04,
        step=0.01,
        until_contact=True,
        object_name="item",
    )

    assert not result.success
    assert result.metrics["contact_detected"] is False
    assert result.metrics["contact_object"] == "support"
    assert result.metrics["failure_code"] == "contact_not_established"


def test_directional_finite_move_can_use_measured_gripper_midpoint(monkeypatch):
    class Kin:
        base_calibration_frames = ("link1", "link2", "link3")

        def calibrated_base_transform(self, q_arm, measured_world_positions, frame_names=None):
            return np.eye(3), np.zeros(3), 0.0

        def fk(self, q_arm):
            return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])

    class Adapter:
        def __init__(self):
            self.midpoint = np.array([0.0, 0.0, 1.0])

        def read_observation(self, timestamp):
            return type("Obs", (), {
                "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
            })()

        def body_position(self, name):
            return (0.0, 0.0, 0.0)

        def gripper_object_alignment(self, object_name, side="left"):
            return {"finger_midpoint": self.midpoint.tolist()}

        def finger_contact_forces(self, side="left"):
            return (0.0, 0.0)

    calls = []

    class FakeArmMoveTo:
        def __init__(self, kin, vel_limits, planner):
            pass

        def execute(self, adapter, **kwargs):
            calls.append(kwargs)
            adapter.midpoint[2] -= 0.01
            return SkillResult(
                False,
                "arm_move_to",
                metrics={"final_position_error_m": 0.032},
                details={"reason": "final target-frame tolerance failed"},
            )

    monkeypatch.setattr("r1pro_data_gen.skills.manipulation.arm_motion.ArmMoveTo", FakeArmMoveTo)
    result = ArmMoveDirectional(Kin(), np.ones(7), object()).execute(
        Adapter(), object(), direction=[0.0, 0.0, -1.0], distance=0.01,
        until_contact=False, object_name="cylinder", support_surface_name="table",
    )

    assert result.success
    assert result.details["motion_reference"] == "measured_gripper_midpoint"
    assert result.metrics["underlying_success"] == 0.0
    assert result.metrics["moved_m"] == pytest.approx(0.01)
    assert result.metrics["requested_distance_m"] == pytest.approx(0.01)
    assert result.metrics["actual_displacement_m"] == pytest.approx(0.01)
    assert result.metrics["endpoint_error_m"] == pytest.approx(0.0)
    assert result.metrics["contact_detected"] is False
    assert result.metrics["contact_object"] is None
    assert result.metrics["failure_code"] is None
    assert calls[0]["scene"] is not None
    assert calls[0]["target_pos"] == pytest.approx([0.0, 0.0, 0.99])
    assert calls[0]["exclude_objects"] == ["cylinder", "table"]


def test_quaternion_error_is_zero_for_same_nontrivial_orientation():
    from r1pro_data_gen.skills.manipulation.arm_motion import _quat_error

    q = np.array([0.70710678, 0.0, -0.70710678, 0.0])
    assert np.allclose(_quat_error(q, q), 0.0, atol=1e-8)


class _GraspScene:
    def object(self, name):
        if name != "item":
            raise KeyError(name)
        return object()


class _TargetContactAdapter(FakeAdapter):
    def __init__(self, *, contacts, side="left"):
        super().__init__(
            joint_positions={f"{side}_gripper_finger_joint1": 0.05},
            contacts=contacts,
        )
        self.side = side
        self.attached = {}

    def contact_events(self):
        return tuple(
            ContactEvent(
                0.1,
                f"{self.side}_gripper_finger_link{index}",
                "item",
                float(force),
            )
            for index, force in enumerate(self._contacts, start=1)
            if force > 0.0
        )

    def attach_object(self, object_name, body_name):
        self.attached[object_name] = body_name
        return True

    def attachment_state(self):
        return dict(self.attached)

    def grasp_attachment_error(self, object_name):
        return 0.0

    def detach_object(self, object_name):
        self.attached.pop(object_name, None)


_GRASP_SCENE = _GraspScene()


def test_gripper_grasp_requires_resolvable_object_identity():
    adapter = FakeAdapter(joint_positions={"left_gripper_finger_joint1": 0.05})

    with pytest.raises(ValueError, match="object_name"):
        GripperGrasp().execute(adapter, _GRASP_SCENE, side="left")

    with pytest.raises(ValueError, match="not present"):
        GripperGrasp().execute(adapter, _GRASP_SCENE, side="left", object_name="missing")


def test_gripper_grasp_without_contact_fails():
    adapter = _TargetContactAdapter(contacts=(0.0, 0.0))
    result = GripperGrasp().execute(
        adapter,
        _GRASP_SCENE,
        side="left",
        object_name="item",
        max_close=0.05,
        step=0.01,
    )
    assert not result.success
    assert result.metrics["final_finger_pos_m"] == 0.0


def test_gripper_grasp_max_close_limits_actual_finger_travel():
    adapter = _TargetContactAdapter(contacts=(0.0, 0.0))
    result = GripperGrasp().execute(
        adapter,
        _GRASP_SCENE,
        side="left",
        object_name="item",
        max_close=0.02,
        step=0.01,
    )

    assert not result.success
    assert result.metrics["final_finger_pos_m"] == pytest.approx(0.03)
    assert result.metrics["failure_code"] == "target_contact_not_established"


def test_gripper_grasp_closes_each_finger_from_its_measured_opening():
    adapter = _TargetContactAdapter(contacts=(0.0, 0.0))
    adapter._joint_positions.update(
        {
            "left_gripper_finger_joint1": 0.05,
            "left_gripper_finger_joint2": -0.03,
        }
    )

    result = GripperGrasp(hold_steps=1).execute(
        adapter,
        _GRASP_SCENE,
        side="left",
        object_name="item",
        max_close=0.02,
        step=0.01,
    )

    assert not result.success
    assert adapter.targets["left_gripper_finger_joint1"] == pytest.approx(0.03)
    assert adapter.targets["left_gripper_finger_joint2"] == pytest.approx(-0.01)


def test_verified_pinch_window_allows_bounded_contact_offset_centering():
    from types import SimpleNamespace

    class Scene:
        def object(self, name):
            if name != "item":
                raise KeyError(name)
            return SimpleNamespace(
                radius=0.025,
                physics=SimpleNamespace(planning_margin=0.04, contact_offset=0.008),
            )

    class CenteringAdapter(_TargetContactAdapter):
        def __init__(self):
            super().__init__(contacts=(3.0, 3.0))
            self.position = np.zeros(3)

        def object_position(self, name):
            assert name == "item"
            return tuple(self.position)

        def gripper_object_alignment(self, object_name, side="left"):
            assert object_name == "item"
            assert side == "left"
            return {"between_fingers": True}

        def step(self):
            self.position[0] = 0.0035

    result = GripperGrasp(hold_steps=1).execute(
        CenteringAdapter(),
        Scene(),
        side="left",
        object_name="item",
    )

    assert result.success


def test_gripper_grasp_allows_settling_after_two_sided_contact():
    class SettlingAdapter(_TargetContactAdapter):
        def __init__(self):
            super().__init__(contacts=(0.0, 3.0))
            self.position = np.zeros(3)

        def object_position(self, name):
            assert name == "item"
            return tuple(self.position)

        def step(self):
            self.steps += 1
            if self.steps == 3:
                # The second finger establishes target contact on this frame.
                self.position[0] = 0.002
            elif self.steps >= 4:
                # This is contact settling, not a one-sided pre-grasp push.
                self.position[0] = 0.008

        def finger_contact_forces(self, side="left"):
            return (3.0, 3.0) if self.steps >= 3 else (0.0, 3.0)

        def contact_events(self):
            contacts = self.finger_contact_forces()
            return tuple(
                ContactEvent(
                    0.1,
                    f"left_gripper_finger_link{index}",
                    "item",
                    float(force),
                )
                for index, force in enumerate(contacts, start=1)
                if force > 0.0
            )

    result = GripperGrasp(hold_steps=2).execute(
        SettlingAdapter(),
        _GRASP_SCENE,
        side="left",
        object_name="item",
        step=0.01,
    )

    assert result.success
    assert result.metrics["contact_bodies"] == ["item", "item"]


def test_gripper_grasp_rejects_double_contact_with_wrong_object():
    class WrongObjectAdapter(FakeAdapter):
        def contact_events(self):
            return (
                ContactEvent(0.1, "left_gripper_finger_link1", "support", 3.0),
                ContactEvent(0.1, "left_gripper_finger_link2", "support", 3.0),
            )

    adapter = WrongObjectAdapter(
        joint_positions={"left_gripper_finger_joint1": 0.05},
        contacts=(3.0, 3.0),
    )
    result = GripperGrasp(hold_steps=1).execute(
        adapter,
        _GRASP_SCENE,
        side="left",
        object_name="item",
    )

    assert not result.success
    assert result.metrics["both_fingers"] is False
    assert result.metrics["contact_bodies"] == ["support", "support"]
    assert result.metrics["failure_code"] == "target_contact_not_established"


def test_gripper_grasp_rejects_single_finger_target_contact():
    class SingleFingerAdapter(FakeAdapter):
        def contact_events(self):
            return (
                ContactEvent(0.1, "left_gripper_finger_link1", "item", 3.0),
            )

    adapter = SingleFingerAdapter(
        joint_positions={"left_gripper_finger_joint1": 0.05},
        contacts=(3.0, 3.0),
    )
    result = GripperGrasp(hold_steps=1).execute(
        adapter,
        _GRASP_SCENE,
        side="left",
        object_name="item",
    )

    assert not result.success
    assert result.metrics["contact_bodies"] == ["item", None]
    assert result.metrics["both_fingers"] is False
    assert result.metrics["failure_code"] == "target_contact_not_established"


def test_gripper_grasp_rejects_unstable_attachment():
    class UnstableAttachmentAdapter(FakeAdapter):
        def __init__(self):
            super().__init__(
                joint_positions={"left_gripper_finger_joint1": 0.05},
                contacts=(3.0, 3.0),
            )
            self.errors = iter((0.01, 0.05))
            self.detached = False

        def contact_events(self):
            return (
                ContactEvent(0.1, "left_gripper_finger_link1", "item", 3.0),
                ContactEvent(0.1, "left_gripper_finger_link2", "item", 3.0),
            )

        def attach_object(self, object_name, body_name):
            return True

        def grasp_attachment_error(self, object_name):
            return next(self.errors, 0.05)

        def detach_object(self, object_name):
            self.detached = True

    adapter = UnstableAttachmentAdapter()
    result = GripperGrasp(hold_steps=2).execute(
        adapter,
        _GRASP_SCENE,
        side="left",
        object_name="item",
    )

    assert not result.success
    assert result.metrics["both_fingers"] is True
    assert result.metrics["attachment_stable"] is False
    assert result.metrics["failure_code"] == "attachment_unstable"
    assert adapter.detached is True


def test_gripper_grasp_detects_two_sided_contact():
    one_sided = _TargetContactAdapter(contacts=(0.0, 3.0))
    assert not GripperGrasp().execute(
        one_sided,
        _GRASP_SCENE,
        side="left",
        object_name="item",
        max_close=0.05,
        step=0.01,
    ).success

    both_sides = _TargetContactAdapter(contacts=(3.0, 3.0))
    assert GripperGrasp().execute(
        both_sides,
        _GRASP_SCENE,
        side="left",
        object_name="item",
        max_close=0.05,
        step=0.01,
    ).success


def test_gripper_grasp_recovers_a_transient_contact_drop():
    class FlickeringContactAdapter(_TargetContactAdapter):
        def __init__(self):
            super().__init__(contacts=(3.0, 3.0))
            self.steps = 0

        def step(self):
            self.steps += 1

        def _current_contacts(self):
            # One-sided contact for a single settling frame, then both fingers
            # recover. The skill must re-close only the missing finger and
            # require a stable final tail.
            return (0.0, 3.0) if self.steps == 2 else (3.0, 3.0)

        def finger_contact_forces(self, side="left"):
            return self._current_contacts()

        def contact_events(self):
            return tuple(
                ContactEvent(
                    0.1,
                    f"left_gripper_finger_link{index}",
                    "item",
                    float(force),
                )
                for index, force in enumerate(self._current_contacts(), start=1)
                if force > 0.0
            )

    result = GripperGrasp(hold_steps=5).execute(
        FlickeringContactAdapter(),
        _GRASP_SCENE,
        side="left",
        object_name="item",
    )
    assert result.success
    assert result.metrics["both_fingers"] is True
    assert result.metrics["failure_code"] is None


def test_stabilize_base_locks_wheels_and_torso_without_root_override():
    class RecordingAdapter:
        joint_mask_locked = False

        def __init__(self):
            self.calls = []

        def lock_joint_mask(self, **kwargs):
            self.calls.append(kwargs)

    adapter = RecordingAdapter()
    stabilize_base(adapter)

    assert adapter.calls == [
        {
            "mask_mode": "lock",
            "joint_groups": ("steer", "wheel", "torso"),
            "lock_root": False,
            "gain_overrides": {"wheel": (500.0, 100.0)},
        }
    ]


def test_stabilize_base_can_hold_wheels_while_leaving_torso_active():
    class RecordingAdapter:
        joint_mask_locked = False

        def __init__(self):
            self.calls = []

        def lock_joint_mask(self, **kwargs):
            self.calls.append(kwargs)

    adapter = RecordingAdapter()
    stabilize_base(adapter, lock_torso=False)

    assert adapter.calls[0]["joint_groups"] == ("steer", "wheel")
    assert adapter.calls[0]["lock_root"] is False


def test_stabilize_base_preserves_existing_task_mask():
    class RecordingAdapter:
        joint_mask_locked = True

        def __init__(self):
            self.calls = []

        def lock_joint_mask(self, **kwargs):
            self.calls.append(kwargs)

    adapter = RecordingAdapter()
    stabilize_base(adapter)

    assert adapter.calls == []


def test_stabilize_base_can_upgrade_its_wheel_only_phase_mask():
    class RecordingAdapter:
        joint_mask_locked = True
        joint_lock_groups = ("steer", "wheel")

        def __init__(self):
            self.calls = []

        def unlock_joint_mask(self):
            self.joint_mask_locked = False
            self.joint_lock_groups = ()
            self.calls.append(("unlock", {}))

        def lock_joint_mask(self, **kwargs):
            self.calls.append(("lock", kwargs))

    adapter = RecordingAdapter()
    stabilize_base(adapter, replace_wheel_only=True)

    assert adapter.calls[0] == ("unlock", {})
    assert adapter.calls[1][0] == "lock"
    assert adapter.calls[1][1]["joint_groups"] == ("steer", "wheel", "torso")


def test_stabilize_base_prefers_atomic_mask_extension_when_available():
    class RecordingAdapter:
        joint_mask_locked = True
        joint_lock_groups = ("steer", "wheel")

        def __init__(self):
            self.calls = []

        def extend_joint_mask(self, **kwargs):
            self.calls.append(("extend", kwargs))

        def unlock_joint_mask(self):
            self.calls.append(("unlock", {}))

        def lock_joint_mask(self, **kwargs):
            self.calls.append(("lock", kwargs))

    adapter = RecordingAdapter()
    stabilize_base(adapter, replace_wheel_only=True)

    assert adapter.calls == [
        (
            "extend",
            {
                "joint_groups": ("torso",),
                "gain_overrides": {"wheel": (500.0, 100.0)},
            },
        )
    ]


def test_gripper_grasp_stabilizes_base_unless_task_mask_active():
    class RecordingAdapter(_TargetContactAdapter):
        def __init__(self):
            super().__init__(contacts=(3.0, 3.0))
            self.lock_calls = 0
            self.joint_mask_locked = False

        def lock_wheels(self):
            self.lock_calls += 1

    adapter = RecordingAdapter()
    assert GripperGrasp().execute(
        adapter,
        _GRASP_SCENE,
        side="left",
        object_name="item",
        max_close=0.05,
        step=0.01,
    ).success
    assert adapter.lock_calls == 1

    masked = RecordingAdapter()
    masked.joint_mask_locked = True
    assert GripperGrasp().execute(
        masked,
        _GRASP_SCENE,
        side="left",
        object_name="item",
        max_close=0.05,
        step=0.01,
    ).success
    assert masked.lock_calls == 0


def test_gripper_grasp_uses_right_joint_names() -> None:
    adapter = _TargetContactAdapter(contacts=(3.0, 3.0), side="right")
    result = GripperGrasp().execute(
        adapter,
        _GRASP_SCENE,
        side="right",
        object_name="item",
        max_close=0.05,
        step=0.01,
    )
    assert result.success


def test_gripper_set_reports_detach_and_separation_facts():
    from r1pro_data_gen.skills import GripperSet

    class ReleaseAdapter(FakeAdapter):
        def __init__(self):
            super().__init__(
                joint_positions={"left_gripper_finger_joint1": 0.05}
            )
            self.attached = {"item": "left_gripper_finger_midpoint"}
            self.object = np.array([0.0, 0.0, 0.0])

        def detach_object(self, object_name):
            self.attached.pop(object_name, None)
            return True

        def attachment_state(self):
            return dict(self.attached)

        def object_position(self, object_name):
            return tuple(self.object)

        def end_effector_poses(self):
            return {"left_ee": (0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)}

    result = GripperSet(hold_steps=1).execute(
        ReleaseAdapter(),
        _GRASP_SCENE,
        open_value=0.05,
        side="left",
        object_name="item",
    )

    assert result.success
    assert result.metrics["detached"] is True
    assert result.metrics["separation_m"] == pytest.approx(0.1)
    assert "placed" not in result.metrics
    assert "placement" not in result.details


def test_trapezoid_profile_starts_and_stops_at_zero():
    from r1pro_data_gen.skills.manipulation.arm import trapezoid_progress, trapezoid_scale

    assert trapezoid_scale(0.0) == 0.0
    assert trapezoid_scale(1.0) == 0.0
    assert trapezoid_scale(0.5) == 1.0
    assert trapezoid_progress(0.0) == 0.0
    assert trapezoid_progress(1.0) == 1.0
    us = np.linspace(0.0, 1.0, 101)
    progress = np.array([trapezoid_progress(u) for u in us])
    scales = np.array([trapezoid_scale(u) for u in us])
    assert np.all(np.diff(progress) >= 0.0)
    assert abs(scales.max() - 1.0) < 1e-9
    assert abs(progress[-1] - 1.0) < 1e-9


def test_trapezoid_progress_matches_integral_of_scale():
    from r1pro_data_gen.skills.manipulation.arm import trapezoid_progress, trapezoid_scale

    us = np.linspace(0.0, 1.0, 1001)
    scales = np.array([trapezoid_scale(u) for u in us])
    integral = np.cumsum(scales) / scales.sum()
    progress = np.array([trapezoid_progress(u) for u in us])
    assert np.abs(progress - integral).max() < 0.01


def test_torso_move_commands_zero_velocity_at_start_and_end():
    from r1pro_data_gen.skills.posture.torso import TORSO_JOINTS, TorsoMoveTo

    class _Recording(FakeAdapter):
        def __init__(self):
            super().__init__(joint_positions={name: 0.0 for name in TORSO_JOINTS})
            self.commands = []

        def set_targets(self, position, velocity=None):
            self.commands.append((dict(position), dict(velocity or {})))
            super().set_targets(position, velocity)

    adapter = _Recording()
    TorsoMoveTo(hold_steps=0).execute(
        adapter, target_q=(0.4, 0.0, 0.0, 0.0), speed_scale=0.5
    )
    first_velocity = adapter.commands[0][1]["torso_joint1"]
    last_velocity = adapter.commands[-1][1]["torso_joint1"]
    assert abs(first_velocity) < 0.08
    assert last_velocity == 0.0


def test_arm_move_to_recovers_oneshot_ik_miss_with_cartesian_substeps(monkeypatch):
    """A long descend can be unreachable as one IK solve but reachable in chain.

    cuRobo and the waypoint planner already seed through Cartesian substeps
    when the terminal pose is outside the current DLS basin. arm_move_to must
    do the same so place-descend is not declared workspace-unreachable.
    """
    from r1pro_data_gen.robot.kinematics import IKSolution
    from r1pro_data_gen.skills.manipulation.arm_motion import ArmMoveTo

    arm_joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}

    class Kin:
        upper = np.full(7, 2.0)
        lower = np.full(7, -2.0)

        def fk(self, q_arm):
            q_arm = np.asarray(q_arm, dtype=float)
            return q_arm[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0])

        def ik_candidates(self, target, quat, q_init, max_candidates):
            del quat, max_candidates
            target = np.asarray(target, dtype=float)
            current = np.asarray(q_init, dtype=float)
            if float(np.linalg.norm(target - current[:3])) > 0.04:
                return []
            q = current.copy()
            q[:3] = target
            return [IKSolution(True, q, 0.0, 0.0, 1, "ok")]

        def _ik_once(self, target, quat, q_init, **_kwargs):
            del quat
            q = np.asarray(q_init, dtype=float).copy()
            q[:3] = np.asarray(target, dtype=float)
            return IKSolution(True, q, 0.0, 0.0, 1, "ok")

        def ik(self, target, quat, q_init=None):
            return self._ik_once(target, quat, np.zeros(7) if q_init is None else q_init)

    captured = {}

    def fake_optimize(*args, **kwargs):
        captured["solutions"] = list(args[3])
        winner = type(
            "W",
            (),
            {
                "candidate_id": 0,
                "attempt_id": 0,
                "q_goal": tuple(args[3][0].q_arm),
                "output": {
                    "position": np.zeros((3, 7)),
                    "velocity": None,
                    "duration": 0.1,
                    "status": "TaskSpaceVerified",
                    "ee_winding": 1.0,
                    "winding": 1.0,
                },
                "metrics": {},
            },
        )()
        return type(
            "R",
            (),
            {
                "success": True,
                "winner": winner,
                "candidates": [],
                "optimality_scope": "test",
                "planner_seed_controlled": True,
                "request_hash": "test",
                "status": "success",
                "reason": "certified Cartesian interpolant",
            },
        )()

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.arm_path_optimizer.optimize_arm_path",
        fake_optimize,
    )
    adapter = FakeAdapter(joint_positions=arm_joints)
    skill = ArmMoveTo(Kin(), np.ones(7), object())
    result = skill.execute(
        adapter,
        scene=load_fixture_scene("bare"),
        target_pos=[0.20, 0.0, 0.0],
        target_quat=[1.0, 0.0, 0.0, 0.0],
        side="left",
    )

    assert captured.get("solutions")
    assert float(np.linalg.norm(np.asarray(captured["solutions"][0].q_arm)[:3] - [0.20, 0.0, 0.0])) < 1e-9
    assert result.details.get("planner_status") == "TaskSpaceVerified"
    assert "outside the arm workspace" not in str(result.details.get("reason"))


def test_arm_move_to_ik_failure_publishes_paired_tolerance_and_position_diagnosis():
    """IK failure must distinguish pose-unreachable from position-unreachable.

    The factual feedback loop can only turn an error into a usable discrepancy
    when the skill publishes a paired tolerance; the planner also needs to know
    whether moving the base would help (position unreachable) or whether only
    the commanded orientation needs relaxing (pose unreachable).
    """
    from r1pro_data_gen.robot.kinematics import IKSolution
    from r1pro_data_gen.skills.manipulation.arm_motion import ArmMoveTo

    arm_joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}

    class Kin:
        upper = np.full(7, 2.0)
        lower = np.full(7, -2.0)

        def fk(self, q_arm):
            return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])

        def ik_candidates(self, target, quat, q_init, max_candidates):
            return []

        def ik(self, target, quat, q_init=None):
            if quat is None:
                return IKSolution(True, np.zeros(7), 0.0, 0.0, 1, "ok")
            return IKSolution(False, None, 0.288, 0.127, 50, "no progress")

    adapter = FakeAdapter(joint_positions=arm_joints)
    skill = ArmMoveTo(Kin(), np.ones(7), object())
    result = skill.execute(
        adapter,
        scene=object(),
        target_pos=[0.5, 0.7, 1.1],
        target_quat=[0.70710678, 0.0, -0.70710678, 0.0],
        side="left",
    )

    assert result.success is False
    assert result.metrics["ik_error_m"] == pytest.approx(0.288)
    assert result.metrics["ik_tolerance_m"] == pytest.approx(0.03)
    assert result.metrics["rotation_error_rad"] == pytest.approx(0.127)
    assert result.metrics["rotation_tolerance_rad"] == pytest.approx(0.10)
    assert result.metrics["position_reachable_without_orientation"] is True
    assert result.details["base_pose_world"] == [0.0, 0.0, 0.0]
    assert "only without the commanded orientation" in result.details["reason"]


def test_arm_move_to_grasp_center_falls_back_to_position_only_when_orientation_unreachable(monkeypatch):
    """A grasp-center target reachable only position-first defers orientation.

    Navigation residuals and IK nulls can make the commanded orientation
    unreachable while the position itself is fine; the skill must move
    position-first and mark the orientation deferred for a later measured
    align stage instead of failing the whole plan.
    """
    from r1pro_data_gen.robot.kinematics import IKSolution
    from r1pro_data_gen.skills.manipulation.arm_motion import ArmMoveTo

    arm_joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}

    class Kin:
        upper = np.full(7, 2.0)
        lower = np.full(7, -2.0)

        def fk(self, q_arm):
            return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])

        def grasp_center_fk(self, q_arm):
            return np.array([0.0, 0.6, 1.2]), np.array([1.0, 0.0, 0.0, 0.0])

        def ee_target_from_grasp_center(self, target, quat):
            return target

        def ik_candidates(self, target, quat, q_init, max_candidates):
            if quat is not None:
                return []
            return [IKSolution(True, np.zeros(7), 0.0, 0.0, 1, "ok")]

        def ik(self, target, quat, q_init=None):
            if quat is None:
                return IKSolution(True, np.zeros(7), 0.0, 0.0, 1, "ok")
            return IKSolution(False, None, 0.3, 0.2, 10, "fail")

    # Short-circuit MPlib planning: position-only should reach and execute.
    class Planner:
        pass

    import r1pro_data_gen.methods.manipulation.arm_path_optimizer as apo
    monkeypatch.setattr("r1pro_data_gen.skills.manipulation.arm_motion._arm_move_to_margin_best", lambda *a, **k: None)
    captured = {}

    def fake_optimize(*args, **kwargs):
        captured["quat_deferred"] = True
        class W:
            success = True
            winner = type("W", (), {
                "candidate_id": 0,
                "attempt_id": 0,
                "output": {"position": np.zeros((3, 7)), "velocity": None, "duration": 0.1,
                           "status": "DirectVerified", "ee_winding": 1.0, "winding": 1.0},
                "metrics": {},
            })()
            candidates = []
            optimality_scope = "test"
            planner_seed_controlled = True
            request_hash = "test"
        return W()

    monkeypatch.setattr(apo, "optimize_arm_path", fake_optimize)

    class Adapter:
        class Robot:
            class Data:
                root_pos_w = []
                root_quat_w = []
            data = Data()
        robot = Robot()
        def read_observation(self, timestamp):
            return type("Obs", (), {"joint_positions": dict(arm_joints), "base_pose": (0.0, 0.0, 0.0)})()
        def step(self): pass
        def set_targets(self, position, velocity=None): pass

    adapter = Adapter()
    scene = load_fixture_scene("bare")
    skill = ArmMoveTo(Kin(), np.ones(7), Planner())
    result = skill.execute(
        adapter,
        scene=scene,
        target_pos=[0.0, 0.6, 1.2],
        target_frame="grasp_center",
        target_quat=[0.70710678, 0.0, -0.70710678, 0.0],
        side="left",
        exclude_objects=[],
    )

    assert captured.get("quat_deferred") is True
    assert result.details.get("orientation_deferred") is True
    assert result.success


def test_measured_gripper_correction_uses_base_pose_rotation():
    """World-frame correction maps to the model frame via the base yaw.

    The online URDF/USD calibration drifts at deep pre-grasp postures; the
    alignment correction must use the deterministic base-pose rotation so a
    world displacement moves the EE the same way in base frame.
    """
    from r1pro_data_gen.skills.manipulation.arm_motion import _measured_gripper_correction

    class Kin:
        def fk(self, q_arm):
            return np.array([0.0, 0.5, 1.2]), np.array([1.0, 0.0, 0.0, 0.0])

    class Adapter:
        class Robot:
            class Data:
                root_pos_w = []
                root_quat_w = []
            data = Data()
        robot = Robot()
        def read_observation(self, timestamp):
            joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
            return type("Obs", (), {"joint_positions": joints, "base_pose": (0.0, 0.0, math.pi / 2)})()

    # base yaw = pi/2: world +x maps to base -y, world +y maps to base +x.
    target_ee, target_quat = _measured_gripper_correction(
        Kin(), Adapter(), "left", np.array([0.1, 0.0, 0.0])
    )
    assert target_ee is not None
    assert target_ee == pytest.approx([0.0, 0.5 - 0.1, 1.2])
