from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from r1pro_data_gen.domain import ObjectType, PhysicsProps, ObjectModel
from r1pro_data_gen.skills import (
    SkillResult,
    SupportAwareGraspObject,
    derive_support_aware_pregrasp,
    derive_support_aware_pregrasp_candidates,
    pregrasp_motion_tolerance,
    world_point_to_base,
)


class _Adapter:
    def __init__(self, base_pose=(0.0, 0.0, 0.0), object_position=(1.0, 0.5, 0.05)):
        self.base_pose = base_pose
        self.position = np.asarray(object_position, dtype=float)

    def read_observation(self, timestamp):
        del timestamp
        return SimpleNamespace(base_pose=self.base_pose)

    def object_position(self, name):
        del name
        return tuple(float(value) for value in self.position)


class _MeasuredGripperAdapter(_Adapter):
    def end_effector_poses(self):
        midpoint = tuple(float(value) for value in self.position)
        return {"left_gripper_finger_midpoint": (*midpoint, 1.0, 0.0, 0.0, 0.0)}

    def body_position(self, name):
        midpoint = self.position.copy()
        offsets = {
            "left_gripper_link": np.array([0.05, 0.0, 0.0]),
            "left_gripper_finger_link1": np.array([0.0, 0.012, 0.0]),
            "left_gripper_finger_link2": np.array([0.0, -0.012, 0.0]),
        }
        return tuple(float(value) for value in midpoint + offsets[name])


class _Scene:
    world = SimpleNamespace(ground=True)

    def __init__(self, objects):
        self.objects = tuple(objects)

    def object(self, name):
        return next(item for item in self.objects if item.name == name)


def _cylinder(name="item", pos=(1.0, 0.5, 0.05), radius=0.025, height=0.1):
    return ObjectModel(
        name=name,
        type=ObjectType.CYLINDER,
        pos=pos,
        radius=radius,
        height=height,
        physics=PhysicsProps(
            planning_margin=0.04,
            contact_offset=0.008,
        ),
    )


def test_support_aware_target_is_derived_from_robot_object_geometry() -> None:
    adapter = _Adapter()
    scene = _Scene([_cylinder()])
    target, details = derive_support_aware_pregrasp(
        adapter,
        scene,
        "item",
        scene.object("item"),
        adapter.position,
        support_name=None,
        side="left",
    )

    assert target is not None
    direction = np.asarray([1.0, 0.5]) / np.linalg.norm([1.0, 0.5])
    assert target[2] == pytest.approx(details["target_grasp_height_m"])
    assert target[2] > adapter.position[2]
    assert np.allclose(
        adapter.position[:2] - target[:2],
        direction * details["standoff_m"],
    )
    assert details["gripper_collision_envelope_m"] >= 0.075
    assert details["gripper_envelope_source"] == "robot_profile"
    assert details["side"] == "left"


def test_support_aware_candidates_are_bounded_and_share_live_geometry() -> None:
    adapter = _Adapter()
    scene = _Scene([_cylinder()])
    candidates = derive_support_aware_pregrasp_candidates(
        adapter,
        scene,
        "item",
        scene.object("item"),
        adapter.position,
        support_name=None,
    )

    assert len(candidates) == 4
    assert [item[1]["approach_offset_rad"] for item in candidates] == pytest.approx(
        [0.0, np.pi / 2.0, -np.pi / 2.0, np.pi]
    )
    from r1pro_data_gen.robot.robot_config import (
        R1PRO_GRIPPER_FINGER_SUPPORT_ENVELOPE_Z_M,
        R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M,
    )

    expected_height = (
        R1PRO_GRIPPER_FINGER_SUPPORT_ENVELOPE_Z_M
        + 0.008
        + 0.015
        + R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M
    )
    assert all(item[1]["target_grasp_height_m"] == pytest.approx(expected_height) for item in candidates)
    assert all(item[1]["standoff_m"] == pytest.approx(candidates[0][1]["standoff_m"]) for item in candidates)


def test_support_aware_grasp_passes_all_live_directions_to_pregrasp_solver() -> None:
    adapter = _Adapter()
    scene = _Scene([_cylinder()])
    skill = SupportAwareGraspObject(None, None, None, None)

    parameters = skill._whole_body_pregrasp_parameter_candidates(
        adapter,
        scene=scene,
        object_name="item",
        object_model=scene.object("item"),
        object_world=adapter.position,
        support_name=None,
        low_object=True,
        side="left",
    )

    assert len(parameters) == 4
    assert all("target_center_world" in item for item in parameters)
    assert len({tuple(item["target_center_world"]) for item in parameters}) == 4


def test_support_aware_target_respects_declared_support_plane() -> None:
    item = _cylinder(pos=(1.0, 0.5, 0.25), height=0.10)
    support = ObjectModel(
        name="support",
        type=ObjectType.CUBOID,
        pos=(1.0, 0.5, 0.10),
        size=(0.8, 0.8, 0.20),
        physics=PhysicsProps(kinematic=True),
    )
    scene = _Scene([item, support])
    target, details = derive_support_aware_pregrasp(
        _Adapter(object_position=item.pos),
        scene,
        "item",
        item,
        item.pos,
        support_name="support",
    )

    assert target is not None
    assert details["support_top_z_m"] == pytest.approx(0.20)
    assert target[2] >= 0.20 + 0.02 + 0.008 + 0.015


def test_support_aware_target_uses_live_gripper_collision_envelope() -> None:
    from r1pro_data_gen.methods.collision import LINK_SPHERE_RADII_BY_SIDE

    item = _cylinder()
    scene = _Scene([item])
    adapter = _MeasuredGripperAdapter(object_position=item.pos)
    target, details = derive_support_aware_pregrasp(
        adapter,
        scene,
        "item",
        item,
        item.pos,
        support_name=None,
    )

    assert target is not None
    assert details["gripper_envelope_source"] == "runtime_and_profile"
    expected = 0.05 + LINK_SPHERE_RADII_BY_SIDE["left"]["left_gripper_link"]
    assert details["gripper_collision_envelope_m"] == pytest.approx(expected)
    assert details["gripper_envelope_by_link_m"]["left_gripper_link"] == pytest.approx(expected)


def test_support_aware_target_handles_rotated_base_without_mixing_frames() -> None:
    base_pose = (0.4, -0.2, np.pi / 2.0)
    target = world_point_to_base((0.4, 0.8, 0.25), base_pose)
    assert target == pytest.approx([1.0, 0.0, 0.25])


def test_pregrasp_motion_tolerance_is_smaller_than_object_footprint() -> None:
    item = _cylinder()
    tolerance = pregrasp_motion_tolerance(item)
    assert 0.0 < tolerance <= 0.012
    assert tolerance < item.radius
    # Remote PhysX settling of a few millimetres after navigation must not
    # abort an otherwise open-gripper approach.
    assert tolerance >= 0.003


def test_support_aware_grasp_inserts_a_non_contact_pregrasp_for_low_objects() -> None:
    item = _cylinder()
    scene = _Scene([item])
    adapter = _Adapter(object_position=item.pos)
    calls = []

    class _Probe(SupportAwareGraspObject):
        def _approach(self, adapter, **params):
            del adapter
            calls.append(params)
            return SkillResult(True, self.name, details={})

    skill = _Probe(None, None, None, None)
    result = skill._prepare_alignment_standoff(
        adapter,
        scene=scene,
        object_name="item",
        object_model=item,
        object_world=item.pos,
        support_name=None,
        low_object=True,
        side="left",
    )

    assert result is not None and result.success
    assert len(calls) == 2
    assert calls[0]["exclude"] == []
    assert calls[0]["target"][2] > calls[1]["target"][2]
    assert result.details["approach_mode"] == "plane_parallel_non_contact"
    assert result.details["target_world"][2] > item.pos[2]


def test_support_aware_grasp_retries_a_geometric_direction_after_ik_failure() -> None:
    item = _cylinder()
    scene = _Scene([item])
    adapter = _Adapter(object_position=item.pos)
    calls = []

    class _DirectionalProbe(SupportAwareGraspObject):
        def _approach(self, adapter, **params):
            del adapter
            calls.append(params)
            if len(calls) == 1:
                return SkillResult(
                    False,
                    self.name,
                    details={"failure_code": "measured_center_ik_failed"},
                )
            return SkillResult(True, self.name, details={})

    result = _DirectionalProbe(None, None, None, None)._prepare_alignment_standoff(
        adapter,
        scene=scene,
        object_name="item",
        object_model=item,
        object_world=item.pos,
        support_name=None,
        low_object=True,
        side="left",
    )

    assert result is not None and result.success
    assert len(calls) == 3
    assert calls[0]["target"] != calls[1]["target"]
    assert result.details["approach_candidate_index"] == 1


def test_support_aware_grasp_aborts_when_pregrasp_moves_object() -> None:
    item = _cylinder()
    scene = _Scene([item])
    adapter = _Adapter(object_position=item.pos)

    class _Moving(SupportAwareGraspObject):
        def _approach(self, adapter, **params):
            del params
            adapter.position = adapter.position + np.array([0.0, 0.0, 0.02])
            return SkillResult(True, self.name, details={})

    result = _Moving(None, None, None, None)._prepare_alignment_standoff(
        adapter,
        scene=scene,
        object_name="item",
        object_model=item,
        object_world=item.pos,
        support_name=None,
        low_object=True,
        side="left",
    )

    assert result is not None and not result.success
    assert result.details["failure_code"] == "object_moved_before_grasp"
