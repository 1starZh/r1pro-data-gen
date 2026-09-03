from types import SimpleNamespace

import pytest

from r1pro_data_gen.domain import (
    ObjectModel,
    ObjectType,
    object_xy_half_extents_m,
    object_vertical_extent_m,
    object_xy_radius_m,
)


def test_cylinder_and_cuboid_share_vertical_extent() -> None:
    cylinder = ObjectModel(
        name="cyl",
        type=ObjectType.CYLINDER,
        pos=(0.0, 0.0, 0.1),
        radius=0.025,
        height=0.10,
    )
    cuboid = ObjectModel(
        name="box",
        type=ObjectType.CUBOID,
        pos=(0.0, 0.0, 0.1),
        size=(0.04, 0.08, 0.10),
    )
    assert object_vertical_extent_m(cylinder) == pytest.approx(0.10)
    assert object_vertical_extent_m(cuboid) == pytest.approx(0.10)
    assert object_xy_radius_m(cylinder) == pytest.approx(0.025)
    assert object_xy_radius_m(cuboid) == pytest.approx(0.02)


def test_extent_helpers_accept_plain_namespace_models() -> None:
    box = SimpleNamespace(height=None, radius=None, size=(0.04, 0.04, 0.12))
    assert object_vertical_extent_m(box) == pytest.approx(0.12)
    assert object_xy_radius_m(box) == pytest.approx(0.02)


def test_rotated_cuboid_uses_conservative_world_xy_projection() -> None:
    box = ObjectModel(
        name="rotated",
        type=ObjectType.CUBOID,
        pos=(0.0, 0.0, 0.2),
        quat=(2**-0.5, 0.0, 0.0, 2**-0.5),
        size=(1.0, 0.2, 0.4),
    )
    half_x, half_y = object_xy_half_extents_m(box)
    assert half_x == pytest.approx(0.1, abs=1e-6)
    assert half_y == pytest.approx(0.5, abs=1e-6)


def test_cuboid_finger_window_uses_surface_not_center_distance() -> None:
    from r1pro_data_gen.domain import object_surface_distance_m

    box = ObjectModel(
        name="box",
        type=ObjectType.CUBOID,
        pos=(0.0, 0.0, 0.1),
        size=(0.04, 0.04, 0.10),
    )
    cylinder = ObjectModel(
        name="cyl",
        type=ObjectType.CYLINDER,
        pos=(0.0, 0.0, 0.1),
        radius=0.025,
        height=0.10,
    )
    # A point on the XY face is 2 cm from the cuboid center but on the surface.
    assert object_surface_distance_m((0.0, 0.0, 0.1), (0.02, 0.0, 0.1), box) == pytest.approx(0.0)
    assert object_surface_distance_m((0.0, 0.0, 0.1), (0.025, 0.0, 0.1), cylinder) == pytest.approx(0.0)
    # The old center-radius gate of 1.2 cm would reject a valid cuboid pinch.
    assert object_surface_distance_m((0.0, 0.0, 0.1), (0.02, 0.0, 0.1), box) < 0.012
