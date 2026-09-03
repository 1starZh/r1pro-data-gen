from __future__ import annotations

import pytest

from r1pro_data_gen.domain import ObjectCapability, ObjectModel, ObjectType, RobotModel, SceneModel, WorldModel
from r1pro_data_gen.skills.manipulation.push import (
    PushObjectTo,
    _base_support_radius,
    _object_support_radius,
)


def _scene(capabilities=()):
    return SceneModel(
        name="push_scene",
        world=WorldModel(),
        robot=RobotModel(asset="robot.usd"),
        objects=(
            ObjectModel(
                name="box",
                type=ObjectType.CUBOID,
                pos=(0.0, 0.0, 0.06),
                size=(0.12, 0.12, 0.12),
                capabilities=tuple(capabilities),
            ),
        ),
    )


class _Adapter:
    def object_position(self, name):
        assert name == "box"
        return (0.0, 0.0, 0.06)


def test_push_requires_explicit_scene_capability():
    result = PushObjectTo().execute(
        _Adapter(),
        scene=_scene((ObjectCapability.MOVABLE,)),
        object_name="box",
        target_pose=(0.5, 0.0, 0.06),
    )

    assert not result.success
    assert result.details["failure_code"] == "object_not_pushable"


def test_push_requires_movable_even_when_pushable_is_declared():
    result = PushObjectTo().execute(
        _Adapter(),
        scene=_scene((ObjectCapability.PUSHABLE,)),
        object_name="box",
        target_pose=(0.5, 0.0, 0.06),
    )

    assert not result.success
    assert result.details["failure_code"] == "object_not_movable"


class _FootprintAdapter:
    def base_footprint(self):
        return {
            "half_length_m": 2.0,
            "half_width_m": 1.0,
            "circumscribed_radius_m": 2.2360679,
        }


class _AsymmetricFootprintAdapter:
    def base_footprint(self):
        return {
            "half_length_m": 2.0,
            "half_width_m": 1.0,
            "front_extent_m": 0.5,
            "rear_extent_m": 2.0,
            "left_extent_m": 1.0,
            "right_extent_m": 0.75,
            "circumscribed_radius_m": 2.2360679,
        }


def test_push_uses_directional_chassis_support_instead_of_circumscribed_radius():
    adapter = _FootprintAdapter()

    assert _base_support_radius(adapter, (1.0, 0.0), 0.0) == pytest.approx(2.0)
    assert _base_support_radius(adapter, (0.0, 1.0), 0.0) == pytest.approx(1.0)


def test_push_prefers_asymmetric_front_and_rear_support_when_available():
    adapter = _AsymmetricFootprintAdapter()

    assert _base_support_radius(adapter, (1.0, 0.0), 0.0) == pytest.approx(0.5)
    assert _base_support_radius(adapter, (-1.0, 0.0), 0.0) == pytest.approx(2.0)


def test_push_uses_object_support_along_direction():
    object_model = ObjectModel(
        name="rectangular_box",
        type=ObjectType.CUBOID,
        pos=(0.0, 0.0, 0.1),
        size=(0.12, 0.30, 0.20),
    )

    assert _object_support_radius(object_model, (1.0, 0.0)) == pytest.approx(0.06)
    assert _object_support_radius(object_model, (0.0, 1.0)) == pytest.approx(0.15)
