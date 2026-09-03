from __future__ import annotations

from types import SimpleNamespace

import pytest

from r1pro_data_gen.skills import SkillResult
from r1pro_data_gen.skills.manipulation.arm import ARM_JOINTS_BY_SIDE
from r1pro_data_gen.skills.manipulation.grasp import GraspObject, _needs_ground_posture
from r1pro_data_gen.skills.posture.torso import TORSO_JOINTS


class _Adapter:
    def __init__(
        self,
        hanging: bool = False,
        ee_z: float | None = None,
        torso_q: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ):
        self.object_pos = (1.0, 0.2, 1.05)
        joints = ARM_JOINTS_BY_SIDE["left"]
        if hanging:
            self.joint_positions = {name: 0.0 for name in joints}
        else:
            self.joint_positions = {name: 0.8 for name in joints}
        self.joint_positions.update(
            {name: float(value) for name, value in zip(TORSO_JOINTS, torso_q)}
        )
        self.ee_z = ee_z

    def read_observation(self, timestamp):
        del timestamp
        return SimpleNamespace(base_pose=(0.0, 0.0, 0.0), joint_positions=self.joint_positions)

    def object_position(self, name):
        del name
        return self.object_pos

    def lock_joint_mask(self, **kwargs):
        del kwargs

    def end_effector_poses(self):
        if self.ee_z is None:
            return {}
        return {"left_ee": (0.4, 0.2, self.ee_z, 1.0, 0.0, 0.0, 0.0)}


class _Scene:
    def object(self, name):
        if name != "item":
            raise KeyError(name)
        return SimpleNamespace(name=name)


class _Open:
    def execute(self, adapter, **params):
        del adapter
        assert params["open_value"] > 0
        assert params["side"] == "left"
        return SkillResult(True, "gripper_set")


class _Move:
    def __init__(self):
        self.targets = []
        self.planning_times = []

    def execute(self, adapter, **params):
        del adapter
        self.targets.append((tuple(params["target_pos"]), params.get("target_frame")))
        self.planning_times.append(params.get("planning_time"))
        return SkillResult(True, "arm_move_to")


class _Align:
    def __init__(self):
        self.calls = 0

    def execute(self, adapter, **params):
        del adapter
        self.calls += 1
        assert params["require_between_fingers"] is True
        assert params["require_vertical_alignment"] is True
        if self.calls == 1:
            return SkillResult(
                False,
                "arm_align_gripper",
                details={"failure_code": "contact_not_centered", "between_fingers": False},
            )
        return SkillResult(True, "arm_align_gripper", details={"between_fingers": True})


class _Grasp:
    def execute(self, adapter, **params):
        del adapter
        assert params["object_name"] == "item"
        return SkillResult(True, "gripper_grasp", details={"attached": True})


class _Joint:
    def __init__(self):
        self.calls = 0

    def execute(self, adapter, **params):
        del adapter
        self.calls += 1
        assert len(params["target_q"]) == 7
        return SkillResult(True, "arm_joint_to")


def test_standoff_is_computed_in_the_live_base_frame() -> None:
    from r1pro_data_gen.skills.manipulation.grasp import _standoff_target

    class _PoseAdapter:
        def __init__(self, base_pose):
            self.base_pose = base_pose

        def read_observation(self, timestamp):
            del timestamp
            return SimpleNamespace(base_pose=self.base_pose, joint_positions={})

        def object_position(self, name):
            del name
            return (1.9, 2.1, 1.05)

    attempt = {"height_m": 0.16, "yaw_rad": 0.0, "nudge_m": 0.0}
    facing_east = _standoff_target(_PoseAdapter((1.35, 2.12, 0.0)), "item", attempt)
    facing_north = _standoff_target(_PoseAdapter((1.35, 2.12, 1.57)), "item", attempt)
    assert facing_east is not None and facing_north is not None
    assert facing_east[2] == pytest.approx(facing_north[2])
    assert facing_east[0] != pytest.approx(facing_north[0], abs=0.05)


def test_grasp_object_opens_then_retries_standoff_after_contact_not_centered() -> None:
    move = _Move()
    align = _Align()
    joint = _Joint()
    skill = GraspObject(_Open(), move, align, _Grasp(), joint)
    result = skill.execute(_Adapter(), scene=_Scene(), object_name="item", side="left")
    assert result.success
    assert joint.calls == 0
    assert align.calls == 2
    assert move.targets[0][1] == "grasp_center"
    # Failed align retreats to the same Cartesian standoff instead of ready.
    assert len(move.targets) >= 3
    assert move.targets[0][0] == move.targets[1][0]
    assert all(time == pytest.approx(0.4) for time in move.planning_times)
    assert result.details["failure_code"] is None


class _AlwaysAlign:
    def __init__(self):
        self.calls = 0

    def execute(self, adapter, **params):
        del adapter, params
        self.calls += 1
        return SkillResult(True, "arm_align_gripper", details={"between_fingers": True})


class _FailThenGrasp:
    def __init__(self):
        self.calls = 0

    def execute(self, adapter, **params):
        del adapter, params
        self.calls += 1
        if self.calls == 1:
            return SkillResult(
                False,
                "gripper_grasp",
                details={"failure_code": "target_contact_not_established"},
            )
        return SkillResult(True, "gripper_grasp", details={"attached": True})


def test_grasp_object_retries_next_standoff_after_close_fails() -> None:
    move = _Move()
    align = _AlwaysAlign()
    grasp = _FailThenGrasp()
    skill = GraspObject(_Open(), move, align, grasp, _Joint())
    result = skill.execute(_Adapter(), scene=_Scene(), object_name="item", side="left")
    assert result.success
    assert grasp.calls == 2
    assert align.calls == 2
    assert result.metrics["attempts"] == 2.0


def test_grasp_object_raises_hanging_arm_to_ready_once() -> None:
    move = _Move()
    align = _Align()
    joint = _Joint()
    skill = GraspObject(_Open(), move, align, _Grasp(), joint)
    result = skill.execute(_Adapter(hanging=True), scene=_Scene(), object_name="item", side="left")
    assert result.success
    assert joint.calls == 1
    assert align.calls == 2
    assert move.targets


class _FailThenMove(_Move):
    def execute(self, adapter, **params):
        if not self.targets:
            self.targets.append((tuple(params["target_pos"]), params.get("target_frame")))
            self.planning_times.append(params.get("planning_time"))
            return SkillResult(
                False,
                "arm_move_to",
                details={"failure_code": "no_complete_waypoint_path"},
            )
        return super().execute(adapter, **params)


def test_grasp_object_hanging_arm_uses_ready_after_approach_failure() -> None:
    move = _FailThenMove()
    align = _Align()
    joint = _Joint()
    skill = GraspObject(_Open(), move, align, _Grasp(), joint)
    result = skill.execute(_Adapter(hanging=True), scene=_Scene(), object_name="item", side="left")
    assert result.success
    assert joint.calls == 1
    assert align.calls == 2


class _Torso:
    def __init__(self):
        self.called = False
        self.target = None

    def execute(self, adapter, **params):
        del adapter
        self.called = True
        self.target = list(params["target_q"])
        return SkillResult(True, "torso_move_to")


class _WholeBody:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, adapter, **params):
        del adapter, params
        self.calls += 1
        return SkillResult(True, "whole_body_pregrasp_transition")


def test_grasp_object_low_object_fails_when_workspace_is_not_prepared() -> None:
    adapter = _Adapter()
    adapter.object_pos = (1.0, 0.2, 0.05)
    move = _Move()
    joint = _Joint()
    torso = _Torso()
    whole_body = _WholeBody()
    skill = GraspObject(
        _Open(),
        move,
        _Align(),
        _Grasp(),
        joint,
        torso_move_to=torso,
        whole_body_pregrasp=whole_body,
    )
    result = skill.execute(adapter, scene=_Scene(), object_name="item", side="left")
    assert not result.success
    assert result.details["failure_code"] == "workspace_not_prepared"
    assert not joint.calls
    assert not torso.called
    assert whole_body.calls == 0
    assert move.targets == []


def test_grasp_object_does_not_force_ready_or_crouch_when_unprepared() -> None:
    adapter = _Adapter(hanging=True)
    adapter.object_pos = (1.0, 0.2, 0.05)
    move = _Move()
    joint = _Joint()
    torso = _Torso()
    whole_body = _WholeBody()
    skill = GraspObject(
        _Open(),
        move,
        _Align(),
        _Grasp(),
        joint,
        torso_move_to=torso,
        whole_body_pregrasp=whole_body,
    )

    result = skill.execute(adapter, scene=_Scene(), object_name="item", side="left")

    assert not result.success
    assert result.details["failure_code"] == "workspace_not_prepared"
    assert joint.calls == 0
    assert not torso.called
    assert whole_body.calls == 0
    assert move.targets == []


def test_grasp_object_low_object_proceeds_after_prepare_workspace() -> None:
    adapter = _Adapter(torso_q=(0.0, 0.80, 0.0, 0.0))
    adapter.object_pos = (1.0, 0.2, 0.05)
    move = _Move()
    joint = _Joint()
    torso = _Torso()
    whole_body = _WholeBody()
    skill = GraspObject(
        _Open(),
        move,
        _Align(),
        _Grasp(),
        joint,
        torso_move_to=torso,
        whole_body_pregrasp=whole_body,
    )
    result = skill.execute(adapter, scene=_Scene(), object_name="item", side="left")
    assert result.success
    assert whole_body.calls == 0
    assert not torso.called
    assert joint.calls == 0
    assert move.targets


def test_ground_posture_uses_measured_height_even_when_floor_has_a_name() -> None:
    assert _needs_ground_posture((0.0, 0.0, 0.05), "floor") is True
    assert _needs_ground_posture((0.0, 0.0, 0.05), "low_platform") is True
    assert _needs_ground_posture((0.0, 0.0, 0.60), "work_table") is False


def test_low_grasp_does_not_retry_whole_body_inside_grasp_object() -> None:
    adapter = _Adapter()
    adapter.object_pos = (1.0, 0.2, 0.05)

    class _Pregrasp:
        def __init__(self):
            self.targets = []

        def execute(self, adapter, **params):
            del adapter
            self.targets.append(list(params.get("target_center_world") or []))
            return SkillResult(True, "whole_body_pregrasp_transition")

    pregrasp = _Pregrasp()
    skill = GraspObject(_Open(), _Move(), _Align(), _Grasp(), whole_body_pregrasp=pregrasp)
    result = skill.execute(adapter, scene=_Scene(), object_name="item", side="left")

    assert not result.success
    assert result.details["failure_code"] == "workspace_not_prepared"
    assert pregrasp.targets == []


def test_grasp_object_reports_unreachable_from_base() -> None:
    class _Unreachable:
        def execute(self, adapter, **params):
            del adapter, params
            return SkillResult(
                False,
                "arm_move_to",
                details={"position_reachable_without_orientation": False, "ik_error_m": 0.25},
            )

    result = GraspObject(_Open(), _Unreachable(), _Align(), _Grasp(), _Joint()).execute(
        _Adapter(), scene=_Scene(), object_name="item", side="left"
    )
    assert not result.success
    assert result.details["failure_code"] == "unreachable_from_base"
