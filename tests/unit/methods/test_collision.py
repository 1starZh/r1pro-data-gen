from __future__ import annotations

import hppfcl

from r1pro_data_gen.domain import ObjectModel, ObjectType
from r1pro_data_gen.methods.collision import object_obstacle
from r1pro_data_gen.methods.collision import LINK_SPHERE_OFFSETS_BY_SIDE, LINK_SPHERE_RADII_BY_SIDE
from r1pro_data_gen.robot.robot_config import (
    R1PRO_GRIPPER_LINK_COLLISION_CENTER_LOCAL,
    R1PRO_GRIPPER_LINK_COLLISION_RADIUS_M,
)


def test_cylinder_obstacle_inflates_full_height_once() -> None:
    object_model = ObjectModel(
        name="cylinder",
        type=ObjectType.CYLINDER,
        pos=(0.0, 0.0, 1.0),
        radius=0.025,
        height=0.10,
    )

    obstacle = object_obstacle(object_model, margin=0.008)

    assert isinstance(obstacle.shape, hppfcl.Cylinder)
    assert obstacle.shape.radius == 0.033
    assert obstacle.shape.halfLength == 0.058


def test_gripper_proxy_is_centered_on_the_collision_mesh() -> None:
    assert LINK_SPHERE_RADII_BY_SIDE["left"]["left_gripper_link"] == (
        R1PRO_GRIPPER_LINK_COLLISION_RADIUS_M
    )
    assert LINK_SPHERE_OFFSETS_BY_SIDE["left"]["left_gripper_link"] == (
        R1PRO_GRIPPER_LINK_COLLISION_CENTER_LOCAL
    )
