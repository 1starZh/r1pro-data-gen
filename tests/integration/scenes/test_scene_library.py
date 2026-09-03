"""Scene model tests: YAML loading, validation, geometry helpers (no Isaac Sim)."""

from __future__ import annotations

import pytest

from r1pro_data_gen.domain import ObjectType, SceneModel
from r1pro_data_gen.data.scenes import load_scene_data
from r1pro_data_gen.tasks import load_task_spec
from tests.support import load_fixture_scene


def test_load_tabletop_fixture() -> None:
    scene = load_fixture_scene("tabletop_basic")
    assert scene.name == "tabletop_basic_fixture"
    assert [o.name for o in scene.objects] == ["table", "cylinder"]
    table = scene.object("table")
    assert table.type is ObjectType.CUBOID
    assert table.size == (0.5, 0.8, 0.1)
    assert table.top_z == pytest.approx(1.05)
    cylinder = scene.object("cylinder")
    assert cylinder.type is ObjectType.CYLINDER
    assert (cylinder.radius, cylinder.height) == (0.03, 0.12)
    assert cylinder.physics.mass == 0.1
    assert cylinder.physics.friction_combine == "max"
    assert scene.world.dt == pytest.approx(1.0 / 60.0, abs=1e-3)
    assert scene.robot.asset.endswith("r1pro.usda")
    assert len(scene.robot.home_joint_pos) == 28


def test_task_spec_holdout_uses_a_box_not_the_cylinder() -> None:
    scene = load_scene_data(load_task_spec("pickplace.holdout_prism_on_slate").scene)
    box = scene.object("grasp_prism")
    cylinder_names = {obj.name for obj in scene.objects if obj.type is ObjectType.CYLINDER}
    assert box.type is ObjectType.CUBOID
    assert box.size == (0.04, 0.04, 0.10)
    assert box.vertical_extent_m == pytest.approx(0.10)
    assert not cylinder_names
    assert {sensor.body for sensor in scene.contact_sensors} == {
        "left_gripper_finger_link1",
        "left_gripper_finger_link2",
        "right_gripper_finger_link1",
        "right_gripper_finger_link2",
    }


def test_task_spec_floor_to_table_places_cylinder_on_the_ground() -> None:
    scene = load_scene_data(load_task_spec("pickplace.floor_to_table_complete").scene)
    cylinder = scene.object("pick_cylinder")
    table = scene.object("work_table")
    assert cylinder.pos[2] < 0.2
    assert table.pos[2] > cylinder.pos[2]
    assert table.pos[2] - cylinder.pos[2] > 0.8


def test_navigation_scene_has_far_start_and_blocking_obstacle() -> None:
    scene = load_fixture_scene("navigation_obstacle")
    assert scene.robot.init_pose == (-1.5, 0.0, 0.0)
    crate = scene.object("crate")
    assert crate.pos == (0.0, 0.0, 0.4)
    assert crate.physics.kinematic


def test_navigation_showcase_is_closed_and_has_internal_gates() -> None:
    scene = load_scene_data(load_task_spec("navigation.arena_route").scene)
    names = {obj.name for obj in scene.objects}
    assert {"fence_north", "fence_south", "fence_west", "fence_east"} <= names
    assert {"gate1_left", "gate1_right", "gate2_left", "gate2_right"} <= names
    assert scene.robot.init_pose == (-3.8, -2.4, 0.0)
    # The fence is longer than the interior and all navigation geometry is
    # kinematic, so the route must be solved inside the arena.
    assert scene.object("fence_north").size[0] >= 12.0
    assert all(obj.physics.kinematic for obj in scene.objects)


def test_contact_filters_resolve_to_absolute_object_prim_paths() -> None:
    from r1pro_data_gen.simulation.isaac_sim.adapter import _contact_filter_prim_paths

    assert _contact_filter_prim_paths(("cylinder", "crate")) == [
        "/World/Cylinder",
        "/World/Crate",
    ]


def test_gripper_fixture_object_fits_fingertip_region() -> None:
    scene = load_fixture_scene("gripper_fixture")
    fixture = scene.object("cylinder")
    assert fixture.type is ObjectType.CUBOID
    assert max(fixture.size) <= 0.04
    assert len(scene.contact_sensors) == 4
    assert {sensor.body for sensor in scene.contact_sensors} == {
        "left_gripper_finger_link1", "left_gripper_finger_link2",
        "right_gripper_finger_link1", "right_gripper_finger_link2",
    }


def test_scene_object_names_must_be_unique() -> None:
    base = SceneModel.from_dict({"name": "x", "robot": {"asset": "a"}, "objects": []})
    with pytest.raises(ValueError, match="unique"):
        SceneModel(
            name="dup",
            world=base.world,
            robot=base.robot,
            objects=(_cuboid("a"), _cuboid("a")),
        )


def test_cuboid_requires_size() -> None:
    with pytest.raises(ValueError, match="cuboid requires size"):
        _cuboid("t", size=None)


def test_cylinder_requires_radius_and_height() -> None:
    with pytest.raises(ValueError, match="radius"):
        SceneModel.from_dict(
            {
                "name": "s",
                "robot": {"asset": "a"},
                "objects": [{"name": "cyl", "type": "cylinder", "pos": [0, 0, 0]}],
            }
        )


def test_from_dict_requires_name_and_robot_asset() -> None:
    with pytest.raises(ValueError, match="name"):
        SceneModel.from_dict({"robot": {"asset": "a"}, "objects": []})
    with pytest.raises(ValueError, match="robot.asset"):
        SceneModel.from_dict({"name": "s", "robot": {}, "objects": []})


def test_cylinder_top_z() -> None:
    cyl = _cylinder("c", pos=(0.5, 0.15, 1.115), height=0.12)
    assert cyl.top_z == pytest.approx(1.175)


def _cuboid(name: str, size=(0.5, 0.8, 0.1)) -> object:
    from r1pro_data_gen.domain import ObjectModel, PhysicsProps, VisualProps

    return ObjectModel(
        name=name,
        type=ObjectType.CUBOID,
        pos=(0.5, 0.3, 1.0),
        size=size,
        physics=PhysicsProps(kinematic=True),
        visual=VisualProps(color=(0.55, 0.40, 0.15)),
    )


def _cylinder(name: str, pos=(0.5, 0.15, 1.115), height=0.12):
    from r1pro_data_gen.domain import ObjectModel

    return ObjectModel(
        name=name,
        type=ObjectType.CYLINDER,
        pos=pos,
        radius=0.03,
        height=height,
    )
