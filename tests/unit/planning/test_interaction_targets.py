from __future__ import annotations

from types import SimpleNamespace

import pytest

from r1pro_data_gen.domain import (
    ObjectType,
    ObjectModel,
    RegionModel,
    RobotModel,
    SceneModel,
    WorldModel,
)
from r1pro_data_gen.planning.context.interaction_targets import (
    InteractionTargetError,
    resolve_interaction_target,
)


def _scene() -> SceneModel:
    return SceneModel(
        name="interaction_scene",
        world=WorldModel(),
        robot=RobotModel(asset="robot.usd"),
        objects=(
            ObjectModel(
                name="marker",
                type=ObjectType.CUBOID,
                pos=(1.0, 2.0, 0.1),
                size=(0.4, 0.4, 0.2),
                regions=(
                    RegionModel(
                        name="goal",
                        shape=ObjectType.CUBOID,
                        center=(0.1, 0.0, 0.2),
                        size=(0.2, 0.2, 0.1),
                    ),
                ),
            ),
        ),
    )


class _Adapter:
    def object_state(self, name):
        assert name == "marker"
        return SimpleNamespace(
            position=(2.0, 3.0, 0.5),
            quaternion=(0.70710678, 0.0, 0.0, 0.70710678),
        )


def test_region_reference_uses_live_object_pose_and_local_region_center():
    result = resolve_interaction_target(_scene(), _Adapter(), target_ref="scene://marker/goal")

    # A 90-degree z rotation maps local +x to world +y.
    assert result.position == pytest.approx((2.0, 3.1, 0.7), abs=1e-5)
    assert result.source == "live_object_region"
    assert result.to_details()["target_region_name"] == "goal"


def test_target_pose_is_a_task_neutral_explicit_fallback():
    result = resolve_interaction_target(
        _scene(), _Adapter(), target_pose=(0.4, -0.2, 0.0)
    )
    assert result.position == (0.4, -0.2, 0.0)
    assert result.source == "explicit_world_pose"


def test_legacy_region_parameter_is_normalized_to_scene_reference():
    result = resolve_interaction_target(
        _scene(), _Adapter(), target_region_name="marker/goal"
    )
    assert result.reference == "scene://marker/goal"
    assert result.position == pytest.approx((2.0, 3.1, 0.7), abs=1e-5)


def test_target_resolution_rejects_ambiguous_or_unknown_references():
    with pytest.raises(InteractionTargetError, match="exactly one"):
        resolve_interaction_target(
            _scene(), _Adapter(), target_ref="scene://marker", target_pose=(0.0, 0.0, 0.0)
        )
    with pytest.raises(InteractionTargetError, match="not declared"):
        resolve_interaction_target(_scene(), _Adapter(), target_ref="scene://marker/missing")
