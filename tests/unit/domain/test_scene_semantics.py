from __future__ import annotations

import math

import pytest

from r1pro_data_gen.domain import (
    ObjectCapability,
    ObjectType,
    SceneModel,
)
from r1pro_data_gen.planning.context.facts import scene_to_facts


def _scene_dict() -> dict[str, object]:
    return {
        "name": "semantic_scene",
        "world": {"dt": 1.0 / 60.0},
        "robot": {"asset": "asset/robot.usda"},
        "objects": [
            {
                "name": "movable_item",
                "type": "cylinder",
                "pos": [0.0, 0.0, 0.1],
                "radius": 0.03,
                "height": 0.2,
                "semantic_class": "household_object",
                "aliases": ["the red item"],
                "capabilities": ["movable", "graspable"],
                "rigid_object": True,
            },
            {
                "name": "support_unit",
                "type": "cuboid",
                "pos": [1.0, 0.0, 0.4],
                "size": [0.8, 0.6, 0.8],
                "semantic_class": "furniture",
                "aliases": ["work surface"],
                "capabilities": ["supports_objects"],
                "regions": [
                    {
                        "name": "placement_area",
                        "shape": "cuboid",
                        "center": [0.0, 0.0, 0.42],
                        "size": [0.4, 0.3, 0.04],
                    }
                ],
                "surfaces": [
                    {
                        "name": "top",
                        "center": [0.0, 0.0, 0.4],
                        "normal": [0.0, 0.0, 1.0],
                        "size": [0.8, 0.6],
                    }
                ],
                "kinematic": True,
            },
        ],
        "cameras": [],
        "contact_sensors": [
            {
                "name": "left_contact",
                "body": "left_finger",
                "filter": ["movable_item"],
            }
        ],
        "collision_sensors": [],
    }


def test_scene_rejects_unknown_object_field() -> None:
    data = _scene_dict()
    data["objects"][0]["suport_surface"] = True

    with pytest.raises(ValueError, match="unknown object fields"):
        SceneModel.from_dict(data)


def test_scene_rejects_unknown_top_level_field() -> None:
    data = _scene_dict()
    data["task_policy"] = {"sequence": ["grasp", "place"]}

    with pytest.raises(ValueError, match="unknown scene fields"):
        SceneModel.from_dict(data)


def test_scene_rejects_unknown_capability() -> None:
    data = _scene_dict()
    data["objects"][0]["capabilities"] = ["pick_target"]

    with pytest.raises(ValueError, match="unknown capability"):
        SceneModel.from_dict(data)


def test_scene_rejects_sensor_filter_for_unknown_object() -> None:
    data = _scene_dict()
    data["contact_sensors"][0]["filter"] = ["missing"]

    with pytest.raises(ValueError, match="unknown objects"):
        SceneModel.from_dict(data)


def test_scene_rejects_non_finite_and_non_positive_geometry() -> None:
    for key, value in (("radius", 0.0), ("height", math.inf)):
        data = _scene_dict()
        data["objects"][0][key] = value

        with pytest.raises(ValueError, match="finite and positive"):
            SceneModel.from_dict(data)


def test_scene_rejects_unknown_collision_sensor_field() -> None:
    data = _scene_dict()
    data["collision_sensors"] = [{"name": "world", "body": "robot", "extra": True}]

    with pytest.raises(ValueError, match="unknown collision sensor fields"):
        SceneModel.from_dict(data)


def test_scene_rejects_collision_sensor_name_collision() -> None:
    data = _scene_dict()
    data["collision_sensors"] = [
        {"name": "left_contact", "body": "robot", "filter": ["support_unit"]}
    ]

    with pytest.raises(ValueError, match="sensor names must be unique"):
        SceneModel.from_dict(data)


def test_scene_rejects_unfiltered_collision_sensor() -> None:
    data = _scene_dict()
    data["collision_sensors"] = [{"name": "world", "body": "robot"}]

    with pytest.raises(ValueError, match="collision sensor.*filter"):
        SceneModel.from_dict(data)


    data = _scene_dict()
    data["collision_sensors"] = [
        {"name": "world", "body": "robot", "filter": ["support_unit"]}
    ]
    scene = SceneModel.from_dict(data)
    facts = scene_to_facts(scene)

    assert facts["collision_sensors"] == [
        {"name": "world", "body": "robot", "filter": ["support_unit"]}
    ]


    scene = SceneModel.from_dict(_scene_dict())
    facts = scene_to_facts(scene)
    item, support = facts["objects"]

    assert scene.objects[0].capabilities == (
        ObjectCapability.MOVABLE,
        ObjectCapability.GRASPABLE,
    )
    assert item["semantic_class"] == "household_object"
    assert item["aliases"] == ["the red item"]
    assert item["capabilities"] == ["movable", "graspable"]
    assert support["regions"][0]["name"] == "placement_area"
    assert support["surfaces"][0]["name"] == "top"
    assert "source_object" not in str(facts)
    assert "target_object" not in str(facts)
    assert "pick_sequence" not in str(facts)
