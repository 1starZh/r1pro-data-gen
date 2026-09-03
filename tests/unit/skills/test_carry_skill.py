"""Pure contracts for live carried-object waypoint generation."""

from types import SimpleNamespace

import numpy as np
import pytest

from r1pro_data_gen.domain import GraspContext
from r1pro_data_gen.skills import SkillResult
from r1pro_data_gen.skills.manipulation.carry import (
    ArmCarryObjectTo,
    _infer_source_support_surface,
    _infer_support_surface_below_target,
)


class _Kin:
    def calibrated_base_transform(self, q_arm, measured, frame_names):
        del q_arm, measured, frame_names
        return np.eye(3), np.zeros(3), 0.0

    def fk(self, q_arm):
        del q_arm
        return np.array([0.4, 0.0, 1.2]), np.array([1.0, 0.0, 0.0, 0.0])

    def ee_target_from_grasp_center(self, target_center, target_quat):
        del target_quat
        return np.asarray(target_center, dtype=float)


class _Adapter:
    def __init__(self):
        self.phase = "held"
        self.captured = None
        self.placed = None
        self.context = GraspContext(
            object_name="held_object",
            side="left",
            attached=True,
            object_position_world=(1.70, 2.30, 1.25),
            grasp_center_world=(1.70, 2.30, 1.335),
            object_to_grasp_center_world=(0.0, 0.0, 0.085),
            attachment_error_m=0.001,
        )

    def read_observation(self, timestamp):
        del timestamp
        return SimpleNamespace(
            base_pose=(0.0, 0.0, 0.0),
            joint_positions={f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
        )

    def body_position(self, name):
        del name
        return (0.0, 0.0, 0.0)

    def get_grasp_context(self, object_name, side="left"):
        assert object_name == self.context.object_name
        assert side == self.context.side
        return self.context

    def object_position(self, name):
        if name == "held_object":
            if self.phase == "placed":
                return tuple(self.placed)
            return self.context.object_position_world
        if name == "target_region":
            return (1.90, 2.05, 1.056)
        raise KeyError(name)


class _MoveThrough:
    def __init__(self, adapter):
        self.adapter = adapter

    def execute(self, adapter, **params):
        assert adapter is self.adapter
        self.adapter.captured = params
        return SkillResult(True, "arm_move_through", details={"planning_status": "fake"})


class _MoveTo:
    def __init__(self, adapter=None):
        self.adapter = adapter
        self.calls: list[dict[str, object]] = []

    def execute(self, adapter, **kwargs):
        self.calls.append(kwargs)
        host = self.adapter if self.adapter is not None else adapter
        if kwargs.get("target_frame") not in {"grasp_center", "ee"}:
            raise AssertionError("test fixture should already be above carry height")
        direction = -np.asarray([1.9, 2.05]) / np.linalg.norm([1.9, 2.05])
        host.phase = "placed"
        host.placed = [1.9 + 0.05 * direction[0], 2.05 + 0.05 * direction[1], 1.10]
        return SkillResult(True, "arm_move_to", details={"planning_status": "local"})


def _scene():
    objects = {
        "held_object": SimpleNamespace(radius=0.025, height=0.10),
        "target_region": SimpleNamespace(pos=(1.90, 2.05, 1.056), size=(0.16, 0.16, 0.01)),
        "support_surface": SimpleNamespace(top_z=1.05),
    }
    return SimpleNamespace(object=lambda name: objects[name])


def test_carry_waypoints_use_live_grasp_context_not_reset_object_pose():
    adapter = _Adapter()
    move_through = _MoveThrough(adapter)
    move_to = _MoveTo(adapter)
    skill = ArmCarryObjectTo(_Kin(), np.ones(7), object(), move_through, move_to)

    result = skill.execute(
        adapter,
        scene=_scene(),
        object_name="held_object",
        target_region_name="target_region",
        support_surface_name="support_surface",
    )

    assert result.success
    assert adapter.captured["beam_width"] == 3
    assert adapter.captured["max_planned_edges"] == 72
    waypoints = adapter.captured["waypoints"]
    assert [item["name"] for item in waypoints] == [
        "carry_retract", "carry_traverse", "carry_extend"
    ]
    assert move_to.calls and move_to.calls[0]["target_frame"] == "ee"
    assert "held_object" in move_to.calls[0]["exclude_objects"]
    assert "support_surface" in move_to.calls[0]["exclude_objects"]
    assert move_to.calls[0]["prefer_local_certified_path"] is False
    assert move_to.calls[0]["target_quat"] is not None
    assert np.allclose(move_to.calls[0]["target_quat"], [1.0, 0.0, 0.0, 0.0])
    # The live held-object x coordinate is 1.70, not the target object's reset
    # coordinate 1.90. The first waypoint must therefore be based near 1.58.
    first_position = np.asarray(waypoints[0]["poses"][0]["position"])
    assert first_position[0] < 1.70
    assert result.metrics["object_xy_error_m"] < 0.015
    assert result.details["grasp_context"]["attached"] is True


def test_short_same_support_carry_skips_radial_retract():
    adapter = _Adapter()
    adapter.context = GraspContext(
        object_name="held_object",
        side="left",
        attached=True,
        object_position_world=(1.88, 2.06, 1.25),
        grasp_center_world=(1.88, 2.06, 1.335),
        object_to_grasp_center_world=(0.0, 0.0, 0.085),
        attachment_error_m=0.001,
    )
    move_through = _MoveThrough(adapter)
    move_to = _MoveTo(adapter)
    skill = ArmCarryObjectTo(_Kin(), np.ones(7), object(), move_through, move_to)
    held = SimpleNamespace(name="held_object", height=0.10, radius=0.025, size=None)
    support = SimpleNamespace(
        name="support_surface",
        size=(0.80, 0.80, 0.10),
        pos=(1.90, 2.10, 1.00),
        top_z=1.05,
    )
    target = SimpleNamespace(
        name="target_region",
        pos=(1.90, 2.05, 1.056),
        size=(0.16, 0.16, 0.01),
        top_z=1.061,
    )
    objects = {"held_object": held, "target_region": target, "support_surface": support}
    scene = SimpleNamespace(
        object=lambda name: objects[name],
        objects=(held, support, target),
    )

    result = skill.execute(
        adapter,
        scene=scene,
        object_name="held_object",
        target_region_name="target_region",
        support_surface_name="support_surface",
    )

    assert result.success
    assert [item["name"] for item in adapter.captured["waypoints"]] == ["carry_extend"]


def test_carry_succeeds_when_object_is_in_region_even_if_descend_tracking_fails():
    adapter = _Adapter()
    move_through = _MoveThrough(adapter)

    class _MissedDescend:
        def execute(self, adapter, **kwargs):
            del kwargs
            adapter.phase = "placed"
            adapter.placed = [1.90, 2.05, 1.10]
            return SkillResult(
                False,
                "arm_move_to",
                details={"failure_code": "final target-frame tolerance failed"},
            )

    skill = ArmCarryObjectTo(_Kin(), np.ones(7), object(), move_through, _MissedDescend())
    result = skill.execute(
        adapter,
        scene=_scene(),
        object_name="held_object",
        target_region_name="target_region",
        support_surface_name="support_surface",
    )

    assert result.success
    assert result.details["reason"] == "carried object reached verified release pose"
    assert result.metrics["object_z_error_m"] <= 0.05


def test_carry_rejects_missing_attachment_before_planning():
    adapter = _Adapter()
    adapter.context = GraspContext(
        object_name="held_object",
        side="left",
        attached=False,
        object_position_world=(1.70, 2.30, 1.25),
        grasp_center_world=(1.70, 2.30, 1.335),
        object_to_grasp_center_world=(0.0, 0.0, 0.085),
    )
    move_through = _MoveThrough(adapter)
    skill = ArmCarryObjectTo(_Kin(), np.ones(7), object(), move_through, _MoveTo(adapter))

    result = skill.execute(
        adapter,
        scene=_scene(),
        object_name="held_object",
        target_region_name="target_region",
        support_surface_name="support_surface",
    )

    assert not result.success
    assert result.details["reason"] == "object is not attached"
    assert adapter.captured is None


def test_source_support_is_inferred_from_live_object_geometry_not_destination():
    held = SimpleNamespace(
        name="held_object",
        height=0.10,
        radius=0.025,
        size=None,
    )
    source = SimpleNamespace(
        name="source_table",
        size=(0.40, 0.80, 0.10),
        pos=(1.90, 2.10, 1.00),
        top_z=1.05,
    )
    destination = SimpleNamespace(
        name="destination_target",
        size=(0.16, 0.16, 0.01),
        pos=(1.90, 2.05, 1.056),
        top_z=1.061,
    )
    scene = SimpleNamespace(objects=(held, source, destination))

    assert _infer_source_support_surface(
        scene, held, (1.90, 2.25, 1.10)
    ) == "source_table"


def test_carry_uses_cuboid_size_for_place_height_not_cylinder_height():
    adapter = _Adapter()
    adapter.context = GraspContext(
        object_name="held_object",
        side="left",
        attached=True,
        object_position_world=(1.70, 2.30, 1.25),
        grasp_center_world=(1.70, 2.30, 1.335),
        object_to_grasp_center_world=(0.0, 0.0, 0.085),
        attachment_error_m=0.001,
    )
    move_through = _MoveThrough(adapter)
    move_to = _MoveTo(adapter)
    skill = ArmCarryObjectTo(_Kin(), np.ones(7), object(), move_through, move_to)
    objects = {
        "held_object": SimpleNamespace(radius=None, height=None, size=(0.04, 0.04, 0.10)),
        "target_region": SimpleNamespace(pos=(1.90, 2.05, 1.056), size=(0.16, 0.16, 0.01)),
        "support_surface": SimpleNamespace(top_z=1.05),
    }
    scene = SimpleNamespace(object=lambda name: objects[name])

    result = skill.execute(
        adapter,
        scene=scene,
        object_name="held_object",
        target_region_name="target_region",
        support_surface_name="support_surface",
    )

    assert result.success
    assert result.details["target_center_z"] == pytest.approx(1.10)
    # Vertical set-down: live EE z plus the object-center delta from cuboid height.
    assert move_to.calls[0]["target_pos"][2] == pytest.approx(1.2 + (1.10 - 1.25), abs=1e-6)


def test_source_support_uses_cuboid_vertical_extent():
    held = SimpleNamespace(
        name="held_object",
        height=None,
        radius=None,
        size=(0.04, 0.04, 0.10),
    )
    source = SimpleNamespace(
        name="source_table",
        size=(0.40, 0.80, 0.10),
        pos=(1.90, 2.10, 1.00),
        top_z=1.05,
    )
    scene = SimpleNamespace(objects=(held, source))

    assert _infer_source_support_surface(
        scene, held, (1.90, 2.25, 1.10)
    ) == "source_table"


def test_destination_backing_support_is_inferred_for_local_release_descent():
    marker = SimpleNamespace(
        name="place_target",
        size=(0.16, 0.16, 0.01),
        pos=(1.90, 2.05, 1.056),
        top_z=1.061,
    )
    table = SimpleNamespace(
        name="work_table",
        size=(0.40, 0.80, 0.10),
        pos=(1.90, 2.10, 1.00),
        top_z=1.05,
    )
    scene = SimpleNamespace(objects=(marker, table))

    assert _infer_support_surface_below_target(scene, marker, marker) == "work_table"
