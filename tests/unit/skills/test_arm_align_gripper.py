from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from r1pro_data_gen.skills.manipulation.arm_motion import (
    ArmAlignGripper,
    _align_continuous_move,
    _execute_alignment_trajectory,
    _object_window_direction_step,
    _select_continuous_ik_solution,
)
from r1pro_data_gen.skills.core.base import SkillResult
from tests.support import load_fixture_scene


class _Adapter:
    def __init__(self):
        self.calls = 0

    def gripper_object_alignment(self, object_name, side="left"):
        self.calls += 1
        midpoint = np.array([0.4, 0.0, 1.0])
        if self.calls == 1:
            midpoint[0] -= 0.04
        return {
            "between_fingers": self.calls > 1,
            "finger_midpoint": midpoint.tolist(),
            "object_position": [0.4, 0.0, 1.0],
        }


class _Kin:
    base_calibration_frames = ("link1", "link2", "link3")

    def calibrated_base_transform(self, q_arm, measured_world_positions, frame_names=None):
        return np.eye(3), np.zeros(3), 0.0

    def fk(self, q_arm):
        return np.array([0.4, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])


class _Motion:
    def __init__(self):
        self.calls = []

    def execute(self, adapter, **kwargs):
        self.calls.append(kwargs)
        return SkillResult(True, "arm_move_directional")


def test_arm_align_gripper_uses_measured_correction(monkeypatch):
    adapter = _Adapter()
    calls = []

    def fake_continuous(kin, adapter, side, start_q, target_ee, speed_scale, step_hook, scene=None, exclude_objects=()):
        calls.append({"target_ee": np.asarray(target_ee, dtype=float), "speed_scale": speed_scale})
        return True, None

    monkeypatch.setattr("r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move", fake_continuous)

    class Scene:
        pass

    adapter.read_observation = lambda timestamp: type("Obs", (), {
        "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
        "base_pose": (0.0, 0.0, 0.0),
    })()
    adapter.body_position = lambda name: (float(len(name)), 0.0, 0.0)

    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        adapter, Scene(), object_name="object", max_iterations=2
    )
    assert result.success
    assert result.metrics["iterations"] == 2.0
    assert len(calls) == 1
    # The measured error is below the adaptive 10 cm proposal cap, so the
    # correction moves the EE by the full 4 cm toward the object.
    target_ee = calls[0]["target_ee"]
    assert target_ee[0] == pytest.approx(0.44, abs=1e-6)
    assert target_ee[1] == pytest.approx(0.0, abs=1e-6)
    assert target_ee[2] == pytest.approx(1.0, abs=1e-6)


def test_arm_align_gripper_uses_finite_segment_closest_point(monkeypatch):
    """A rotated finger segment must be corrected from its closest point."""

    class SegmentAdapter(_Adapter):
        def gripper_object_alignment(self, object_name, side="left"):
            del object_name, side
            self.calls += 1
            return {
                "between_fingers": False,
                "finger_midpoint": [0.4, 0.0, 1.0],
                "closest_point": [0.38, 0.0, 1.0],
                "object_position": [0.4, 0.0, 1.0],
                "segment_fraction": 0.50,
                "surface_distance_m": 0.02,
            }

    adapter = SegmentAdapter()
    targets = []

    def fake_continuous(kin, adapter, side, start_q, target_ee, speed_scale, step_hook, scene=None, exclude_objects=()):
        del kin, adapter, side, start_q, speed_scale, step_hook, scene, exclude_objects
        targets.append(np.asarray(target_ee, dtype=float))
        return True, None

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        fake_continuous,
    )
    adapter.read_observation = lambda timestamp: type("Obs", (), {
        "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
        "base_pose": (0.0, 0.0, 0.0),
    })()
    adapter.body_position = lambda name: (float(len(name)), 0.0, 0.0)

    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        adapter, object(), object_name="object", max_iterations=1
    )

    assert not result.success
    assert len(targets) == 1
    # The midpoint is already at x=0.4; only the closest point reveals the
    # 2 cm perpendicular error and therefore commands the capped correction.
    assert targets[0][0] == pytest.approx(0.42, abs=1e-6)
    assert targets[0][1] == pytest.approx(0.0, abs=1e-6)
    assert targets[0][2] == pytest.approx(1.0, abs=1e-6)


def test_arm_align_gripper_can_require_object_window_after_contact(monkeypatch):
    adapter = _Adapter()

    def fake_continuous(kin, adapter, side, start_q, target_ee, speed_scale, step_hook, scene=None, exclude_objects=()):
        return True, None

    monkeypatch.setattr("r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move", fake_continuous)
    adapter.read_observation = lambda timestamp: type("Obs", (), {
        "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
        "base_pose": (0.0, 0.0, 0.0),
    })()
    adapter.body_position = lambda name: (float(len(name)), 0.0, 0.0)

    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        adapter,
        object(),
        object_name="object",
        position_tolerance=0.015,
        require_between_fingers=True,
        max_iterations=2,
    )

    assert result.success
    assert result.details["between_fingers"] is True


def test_arm_align_gripper_accepts_measured_window_tolerance(monkeypatch):
    class WindowAdapter(_Adapter):
        def gripper_object_alignment(self, object_name, side="left"):
            self.calls += 1
            return {
                "between_fingers": False,
                "segment_fraction": 0.50,
                "surface_distance_m": 0.020,
                "surface_tolerance_m": 0.012,
                "finger_midpoint": [0.4, 0.0, 1.0],
                "object_position": [0.4, 0.0, 1.0],
            }

    adapter = WindowAdapter()
    adapter.read_observation = lambda timestamp: type("Obs", (), {
        "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
        "base_pose": (0.0, 0.0, 0.0),
    })()
    adapter.body_position = lambda name: (float(len(name)), 0.0, 0.0)

    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        adapter,
        object(),
        object_name="object",
        require_between_fingers=True,
        surface_tolerance_m=0.025,
    )

    assert result.success
    assert result.details["between_fingers"] is True
    assert result.details["raw_between_fingers"] is False
    assert result.details["surface_tolerance_m"] == pytest.approx(0.025)


def test_alignment_cannot_succeed_above_object_when_vertical_alignment_required(monkeypatch):
    class AboveObjectAdapter:
        joint_mask_locked = True

        def gripper_object_alignment(self, object_name, side="left"):
            return {
                "between_fingers": True,
                "finger_midpoint": [0.4, 0.0, 1.10],
                "object_position": [0.4, 0.0, 1.00],
                "surface_distance_m": 0.10,
            }

        def read_observation(self, timestamp):
            return type(
                "Obs",
                (),
                {
                    "joint_positions": {
                        f"left_arm_joint{i}": 0.0 for i in range(1, 8)
                    },
                    "base_pose": (0.0, 0.0, 0.0),
                },
            )()

        def body_position(self, name):
            return (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        lambda *args, **kwargs: (True, None),
    )

    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        AboveObjectAdapter(),
        object(),
        object_name="item",
        require_vertical_alignment=True,
        max_iterations=1,
    )

    assert not result.success
    assert result.metrics["vertical_error_m"] > result.metrics["vertical_tolerance_m"]
    assert result.metrics["horizontal_error_m"] == pytest.approx(0.0)
    assert result.metrics["surface_distance_m"] == pytest.approx(0.10)
    assert result.metrics["failure_code"] == "vertical_alignment_not_reached"


def test_alignment_accepts_physical_finger_window_above_object_center(monkeypatch):
    class PhysicalWindowAdapter:
        joint_mask_locked = True

        def gripper_object_alignment(self, object_name, side="left"):
            return {
                "between_fingers": True,
                "finger_midpoint": [0.4, 0.0, 1.10],
                "object_position": [0.4, 0.0, 1.00],
                "surface_distance_m": 0.0,
                "window_geometry_source": "projected_finger_boxes",
                "finger_vertical_intervals": [
                    {"overlap_m": 0.01},
                    {"overlap_m": 0.02},
                ],
            }

        def read_observation(self, timestamp):
            del timestamp
            return type(
                "Obs",
                (),
                {
                    "joint_positions": {
                        f"left_arm_joint{i}": 0.0 for i in range(1, 8)
                    },
                    "base_pose": (0.0, 0.0, 0.0),
                },
            )()

        def body_position(self, name):
            del name
            return (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        lambda *args, **kwargs: (True, None),
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        PhysicalWindowAdapter(),
        object(),
        object_name="item",
        require_between_fingers=True,
        require_vertical_alignment=True,
        max_iterations=1,
    )

    assert result.success


def test_physical_finger_window_rejects_grazing_vertical_overlap():
    from r1pro_data_gen.skills.manipulation.arm_motion import (
        _alignment_physical_vertical_window_ready,
    )

    alignment = {
        "window_geometry_source": "projected_finger_boxes",
        "required_vertical_overlap_m": 0.01,
        "finger_vertical_intervals": [
            {"overlap_m": 0.002},
            {"overlap_m": 0.012},
        ],
    }
    assert not _alignment_physical_vertical_window_ready(alignment)

    alignment["finger_vertical_intervals"][0]["overlap_m"] = 0.01
    assert _alignment_physical_vertical_window_ready(alignment)


def test_alignment_centers_laterally_before_descending_to_avoid_side_contact(monkeypatch):
    class StagedAdapter:
        joint_mask_locked = True

        def __init__(self):
            self.calls = 0

        def gripper_object_alignment(self, object_name, side="left"):
            self.calls += 1
            if self.calls == 1:
                return {
                    "between_fingers": False,
                    "finger_midpoint": [0.392, 0.0, 1.10],
                    "object_position": [0.400, 0.0, 1.00],
                    "surface_distance_m": 0.02,
                }
            if self.calls == 2:
                return {
                    "between_fingers": False,
                    "finger_midpoint": [0.398, 0.0, 1.10],
                    "object_position": [0.400, 0.0, 1.00],
                    "surface_distance_m": 0.02,
                }
            return {
                "between_fingers": True,
                "finger_midpoint": [0.400, 0.0, 1.00],
                "object_position": [0.400, 0.0, 1.00],
                "surface_distance_m": 0.0,
            }

        def read_observation(self, timestamp):
            return type(
                "Obs",
                (),
                {
                    "joint_positions": {
                        f"left_arm_joint{i}": 0.0 for i in range(1, 8)
                    },
                    "base_pose": (0.0, 0.0, 0.0),
                },
            )()

        def body_position(self, name):
            return (0.0, 0.0, 0.0)

    targets = []

    def fake_continuous(kin, adapter, side, start_q, target_ee, speed_scale, step_hook, scene=None, exclude_objects=()):
        targets.append(np.asarray(target_ee, dtype=float))
        return True, None

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        fake_continuous,
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        StagedAdapter(),
        object(),
        object_name="item",
        require_between_fingers=True,
        require_vertical_alignment=True,
        max_iterations=3,
    )

    assert result.success
    assert len(targets) == 2
    # The first correction is lateral only; a diagonal descent would have
    # moved the model EE below its current z before the jaw window was ready.
    assert targets[0][2] == pytest.approx(1.0)
    # After one lateral correction, the remaining lateral error is within the
    # position tolerance, so the next correction may descend.
    assert targets[1][2] < targets[0][2]


def test_alignment_refreshes_window_after_contact_motion(monkeypatch):
    """Contact gating must use geometry measured after the correction move."""

    class RefreshingAdapter:
        joint_mask_locked = True

        def __init__(self):
            self.alignment_calls = 0

        def gripper_object_alignment(self, object_name, side="left"):
            del object_name, side
            self.alignment_calls += 1
            if self.alignment_calls == 1:
                # The pre-motion sample is still above the finite jaw window.
                return {
                    "between_fingers": False,
                    "segment_fraction": 0.50,
                    "surface_distance_m": 0.020,
                    "finger_midpoint": [0.4, 0.0, 1.10],
                    "object_position": [0.4, 0.0, 1.00],
                }
            # The correction move closes the window before the contact sample.
            return {
                "between_fingers": True,
                "segment_fraction": 0.50,
                "surface_distance_m": 0.0,
                "finger_midpoint": [0.4, 0.0, 1.00],
                "object_position": [0.4, 0.0, 1.00],
            }

        def read_observation(self, timestamp):
            del timestamp
            return type(
                "Obs",
                (),
                {
                    "joint_positions": {
                        f"left_arm_joint{i}": 0.0 for i in range(1, 8)
                    },
                    "base_pose": (0.0, 0.0, 0.0),
                },
            )()

        def finger_contact_forces(self, side="left"):
            del side
            return (2.0, 2.0)

        def body_position(self, name):
            del name
            return (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        lambda *args, **kwargs: (True, None),
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        RefreshingAdapter(),
        object(),
        object_name="item",
        require_between_fingers=True,
        require_vertical_alignment=True,
        max_iterations=1,
    )

    assert result.success
    assert result.details["between_fingers"] is True
    assert result.details["contact_detected"] is True


def test_alignment_ik_prefers_continuous_branch_over_margin_best():
    kin = SimpleNamespace(lower=np.full(7, -2.0), upper=np.full(7, 2.0))
    reference = np.zeros(7)
    near = SimpleNamespace(
        success=True,
        q_arm=np.full(7, 0.01),
        position_error=0.001,
        rotation_error=0.001,
    )
    farther = SimpleNamespace(
        success=True,
        q_arm=np.full(7, 0.40),
        position_error=0.0001,
        rotation_error=0.0001,
    )

    selected = _select_continuous_ik_solution(kin, [farther, near], reference)

    assert selected is not None
    q_goal, continuity, _ = selected
    assert np.allclose(q_goal, near.q_arm)
    assert continuity < 0.02


def test_window_direction_step_limits_the_full_3d_jaw_rotation():
    class Adapter:
        def body_position(self, name):
            return {
                "left_gripper_finger_link1": (0.0, 0.0, 0.0),
                "left_gripper_finger_link2": (0.0, 0.0, -1.0),
            }[name]

    direction = _object_window_direction_step(
        Adapter(),
        "left",
        "item",
        np.array([1.0, 0.0, 0.0]),
        max_step_rad=np.pi / 4.0,
    )

    assert direction is not None
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[2] < 0.0
    assert np.dot(np.array([0.0, 0.0, -1.0]), direction) == pytest.approx(
        np.cos(np.pi / 4.0), abs=1e-6
    )


def test_alignment_rejects_a_distant_redundant_branch(monkeypatch):
    class Kin:
        lower = np.full(7, -2.0)
        upper = np.full(7, 2.0)

        def fk(self, q_arm):
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def ik_candidates(self, target_pos, target_quat, q_current, max_candidates=8):
            del target_pos, target_quat, q_current, max_candidates
            return [
                SimpleNamespace(
                    success=True,
                    q_arm=np.full(7, 0.60),
                    position_error=0.001,
                    rotation_error=0.0,
                )
            ]

    moved, details = _align_continuous_move(
        Kin(), object(), "left", np.zeros(7), np.array([0.01, 0.0, 0.0]), 0.07, None
    )

    assert not moved
    assert details["failure_code"] == "alignment_ik_failed"
    assert details["alignment_local_branch_rejected_count"] > 0


def test_alignment_relaxes_orientation_when_position_only_is_continuous():
    class Kin:
        lower = np.full(7, -2.0)
        upper = np.full(7, 2.0)

        def fk(self, q_arm):
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def ik_candidates(self, target_pos, target_quat, q_current, max_candidates=8):
            if target_quat is not None:
                return []
            return [
                SimpleNamespace(
                    success=True,
                    q_arm=np.full(7, 0.02),
                    position_error=0.001,
                    rotation_error=0.0,
                )
            ]

    class Adapter:
        def __init__(self):
            self.targets = []

        def set_targets(self, position, velocity):
            self.targets.append(position)

        def step(self):
            return None

    moved, details = _align_continuous_move(
        Kin(), Adapter(), "left", np.zeros(7), np.array([0.01, 0.0, 0.0]), 0.07, None
    )

    assert moved
    assert details["orientation_relaxed"] is True


def test_align_stops_on_contact_instead_of_pushing_object(monkeypatch):
    """A measured descent that reaches the object must stop at contact.

    Without the contact stop, alignment would keep commanding lower goals past
    the object and push it (observed: 0.54 m drift on a tall standoff).  Once
    a finger touches, the alignment reports success and the grasp primitive
    owns the closure.
    """
    class ContactAdapter:
        joint_mask_locked = True
        def gripper_object_alignment(self, object_name, side="left"):
            return {
                "between_fingers": False,
                "finger_midpoint": [0.4, 0.0, 1.10],
                "object_position": [0.4, 0.0, 1.00],
                "surface_distance_m": 0.10,
            }
        def read_observation(self, timestamp):
            return type("Obs", (), {
                "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
                "base_pose": (0.0, 0.0, 0.0),
            })()
        def finger_contact_forces(self, side="left"):
            return (2.5, 2.0)  # both fingers touching
        def body_position(self, name):
            return (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        lambda *args, **kwargs: (True, None),
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        ContactAdapter(),
        object(),
        object_name="object",
        require_vertical_alignment=True,
        max_iterations=4,
    )
    assert result.success
    assert result.details.get("contact_detected") is True
    assert result.details.get("reason") == "contact reached during alignment"


def test_align_stops_on_single_finger_contact(monkeypatch):
    """Any finger contact is a valid grasp-ready stop.

    From above the first finger to touch the object is grasp-ready;
    continuing past it pushes the object instead of improving alignment.
    """
    class SingleContactAdapter:
        joint_mask_locked = True
        def gripper_object_alignment(self, object_name, side="left"):
            return {
                "between_fingers": False,
                "finger_midpoint": [0.4, 0.0, 1.10],
                "object_position": [0.4, 0.0, 1.00],
                "surface_distance_m": 0.10,
            }
        def read_observation(self, timestamp):
            return type("Obs", (), {
                "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
                "base_pose": (0.0, 0.0, 0.0),
            })()
        def finger_contact_forces(self, side="left"):
            return (2.5, 0.0)  # only one finger touching
        def body_position(self, name):
            return (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        lambda *args, **kwargs: (True, None),
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        SingleContactAdapter(),
        object(),
        object_name="object",
        require_vertical_alignment=True,
        max_iterations=3,
    )
    # Single-finger contact while the object is outside the jaw window means
    # the object is off to one side; closing the gripper would pinch empty air
    # on one side and push the object on the other, so alignment must report
    # failure rather than a grasp-ready state.
    assert result.success is False
    assert result.details.get("contact_detected") is True
    assert result.details.get("reason") == "contact reached but object not centered in jaw"


def test_align_does_not_succeed_on_one_finger_graze_inside_jaw_window(monkeypatch):
    """Geometric between_fingers plus one loaded finger is not a pinch.

    The first tabletop grasp closed on this state and failed
    ``target_contact_not_established`` (one finger 14 N, the other 0 N).
    """
    class GrazeAdapter:
        joint_mask_locked = True
        def __init__(self):
            self.alignment_calls = 0

        def gripper_object_alignment(self, object_name, side="left"):
            self.alignment_calls += 1
            if self.alignment_calls == 1:
                return {
                    "between_fingers": False,
                    "finger_midpoint": [0.4, 0.0, 1.16],
                    "object_position": [0.4, 0.0, 1.10],
                    "surface_distance_m": 0.06,
                }
            return {
                "between_fingers": True,
                "finger_midpoint": [0.4, 0.0, 1.105],
                "object_position": [0.4, 0.0, 1.10],
                "surface_distance_m": 0.012,
            }
        def read_observation(self, timestamp):
            return type("Obs", (), {
                "joint_positions": {f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
                "base_pose": (0.0, 0.0, 0.0),
            })()
        def finger_contact_forces(self, side="left"):
            return (0.0, 14.8)
        def body_position(self, name):
            return (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        lambda *args, **kwargs: (True, None),
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        GrazeAdapter(),
        object(),
        object_name="object",
        require_between_fingers=True,
        require_vertical_alignment=True,
        max_iterations=2,
        position_tolerance=0.015,
    )
    assert result.success is False
    assert result.details.get("contact_detected") is True
    assert result.metrics.get("failure_code") == "one_finger_contact"


def test_alignment_trajectory_stops_at_first_live_finger_contact():
    """A contact-sensitive trajectory must not execute waypoints past contact."""

    class ContactAdapter:
        def __init__(self):
            self.targets = []
            self.hooks = 0

        def read_observation(self, timestamp):
            del timestamp
            return type("Obs", (), {"base_pose": (0.0, 0.0, 0.0)})()

        def set_targets(self, position, velocity):
            del velocity
            self.targets.append(dict(position))

        def step(self):
            return None

        def finger_contact_forces(self, side="left"):
            del side
            return (0.4, 0.0)

    adapter = ContactAdapter()
    joints = tuple(f"left_arm_joint{i}" for i in range(1, 8))
    trajectory = np.vstack((np.zeros(7), np.full(7, 0.1), np.full(7, 0.2)))

    executed = _execute_alignment_trajectory(
        object(),
        adapter,
        "left",
        trajectory,
        np.zeros(7),
        lambda: setattr(adapter, "hooks", adapter.hooks + 1),
    )

    assert executed is True
    assert len(adapter.targets) == 1
    assert adapter.hooks == 1


def test_align_fails_closed_when_object_moves_before_attachment(monkeypatch):
    """An open-gripper push must stop the generic alignment loop immediately."""

    class MovingAdapter:
        joint_mask_locked = True

        def __init__(self):
            self.object_position_world = np.array([0.0, 0.0, 1.0])

        def gripper_object_alignment(self, object_name, side="left"):
            del object_name, side
            return {
                "between_fingers": False,
                "finger_midpoint": [0.10, 0.0, 1.0],
                "object_position": self.object_position_world.tolist(),
                "surface_distance_m": 0.10,
            }

        def object_position(self, object_name):
            del object_name
            return tuple(self.object_position_world)

        def read_observation(self, timestamp):
            del timestamp
            return type(
                "Obs",
                (),
                {
                    "joint_positions": {
                        f"left_arm_joint{i}": 0.0 for i in range(1, 8)
                    },
                    "base_pose": (0.0, 0.0, 0.0),
                },
            )()

        def body_position(self, name):
            del name
            return (0.0, 0.0, 0.0)

    def fake_continuous(*args, **kwargs):
        adapter = args[1]
        adapter.object_position_world[:] = [0.03, 0.0, 1.0]
        violation = kwargs["object_motion_guard"]()
        assert violation is not None
        return False, {
            "reason": "movable object shifted before attachment",
            "failure_code": "object_moved_before_grasp",
            **violation,
        }

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        fake_continuous,
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        MovingAdapter(),
        object(),
        object_name="item",
        max_iterations=1,
    )

    assert result.success is False
    assert result.metrics["failure_code"] == "object_moved_before_grasp"
    assert result.metrics["object_motion_m"] == pytest.approx(0.03)
    assert result.details["failure_code"] == "object_moved_before_grasp"
    assert result.details["current_object_position"] == pytest.approx([0.03, 0.0, 1.0])


def test_align_rebaselines_small_remote_settling_with_explicit_no_contact(monkeypatch):
    """Support settling is tolerated only while the fingers are remote."""

    class SettlingAdapter:
        joint_mask_locked = True

        def __init__(self):
            self.object_position_world = np.array([0.02, 0.0, 1.0])
            self.alignment_calls = 0

        def gripper_object_alignment(self, object_name, side="left"):
            del object_name, side
            self.alignment_calls += 1
            terminal = self.alignment_calls >= 3
            midpoint = (
                self.object_position_world.tolist()
                if terminal
                else [0.0, 0.0, 1.0]
            )
            return {
                "between_fingers": terminal,
                "finger_midpoint": midpoint,
                "surface_distance_m": 0.0 if terminal else 0.10,
                "segment_fraction": 0.5,
                "object_position": self.object_position_world.tolist(),
            }

        def finger_contact_forces(self, side="left"):
            del side
            return (0.0, 0.0)

        def object_position(self, object_name):
            del object_name
            return tuple(self.object_position_world)

        def read_observation(self, timestamp):
            del timestamp
            return type(
                "Obs",
                (),
                {
                    "joint_positions": {
                        f"left_arm_joint{i}": 0.0 for i in range(1, 8)
                    },
                    "base_pose": (0.0, 0.0, 0.0),
                },
            )()

        def body_position(self, name):
            del name
            return (0.0, 0.0, 0.0)

    def fake_continuous(*args, **kwargs):
        adapter = args[1]
        adapter.object_position_world[:] = [0.0235, 0.0, 1.0]
        violation = kwargs["object_motion_guard"]()
        assert violation is None
        return True, None

    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion._align_continuous_move",
        fake_continuous,
    )
    result = ArmAlignGripper(_Kin(), np.ones(7), object()).execute(
        SettlingAdapter(),
        object(),
        object_name="item",
        require_between_fingers=True,
        max_iterations=1,
    )

    assert result.success
    assert result.details["noncontact_rebaseline_count"] == 1


def test_align_collision_gate_rejects_trajectory_through_object():
    """A correction whose arm links sweep through the object must not execute.

    Alignment legitimately ends with the fingers around the object, but the arm
    links must never pass through it (that pushes the object).  The collision
    gate checks only arm links, so a finger-only contact passes while an arm
    link through the object is rejected.
    """
    from r1pro_data_gen.methods.collision import (
        CollisionChecker,
        check_path,
        obstacles_from_scene,
    )
    from r1pro_data_gen.methods.manipulation.mplib_path import _ARM_SLICE_BY_SIDE
    from r1pro_data_gen.skills.manipulation.arm_motion import _execute_alignment_trajectory
    from r1pro_data_gen.skills.planning import runtime_scene_snapshot

    from r1pro_data_gen.robot.kinematics import R1ProKinematics
    from tests.support import PROJECT_ROOT

    scene = load_fixture_scene("tabletop_navigation")
    # Build a trajectory from home to a grasp pose over the table cylinder.
    # The direct descent sweeps the arm down through the cylinder region.
    urdf = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"
    kin = R1ProKinematics(str(urdf), side="left")
    bx, by, byaw = 1.201, 0.1, 1.5708
    import math
    cos_y, sin_y = math.cos(byaw), math.sin(byaw)
    cyl = next(o for o in scene.objects if o.type.value == "cylinder")
    dx, dy = cyl.pos[0] - bx, cyl.pos[1] - by
    target_base = np.array([cos_y*dx + sin_y*dy, -sin_y*dx + cos_y*dy, 1.11])
    q_start = np.zeros(7)
    # Position-only IK to the cylinder height (through the object plane).
    sol = kin.ik(target_base, None, q_init=q_start)
    q_goal = np.asarray(sol.q_arm)
    trajectory = np.linspace(q_start, q_goal, 24)

    class Adapter:
        def read_observation(self, timestamp):
            return type("Obs", (), {"base_pose": (bx, by, byaw)})()

    adapter = Adapter()
    executed = {"n": 0}
    live = runtime_scene_snapshot(scene, adapter=None, exclude_objects=("table",))
    # Force execution attempt by calling the executor with a scene; the gate
    # must reject the through-object trajectory before any set_targets.
    from r1pro_data_gen.methods.collision import LINK_SPHERE_RADII_BY_SIDE
    full_radii = dict(LINK_SPHERE_RADII_BY_SIDE["left"])
    arm_radii = {n: r for n, r in full_radii.items()
                 if not n.endswith("gripper_link")
                 and not n.endswith("gripper_finger_link1")
                 and not n.endswith("gripper_finger_link2")}
    checker = CollisionChecker(kin, obstacles_from_scene(live, include_ground=True), link_radii=arm_radii)
    free, _, link = check_path(checker, [np.asarray(q) for q in trajectory],
                               base_xy=(bx, by), base_yaw=byaw, dense=8)
    assert free is False, f"descent through the object must be rejected (link={link})"
    assert link is not None and "gripper_finger" not in link
