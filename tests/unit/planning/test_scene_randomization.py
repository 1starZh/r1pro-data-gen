from __future__ import annotations

import random

import pytest

from r1pro_data_gen.data.randomization import (
    SceneRandomizationError,
    check_scene_feasibility,
    randomize_scene_data,
)


def _scene() -> dict:
    return {
        "name": "randomization_fixture",
        "world": {
            "ground": True,
            "ground_size": [8.0, 8.0],
        },
        "robot": {
            "asset": "asset/r1pro/r1pro.usda",
            "init_pose": [-2.0, -2.0, 0.0],
            "navigation_footprint_radius_m": 0.25,
        },
        "objects": [
            {
                "name": "support",
                "type": "cuboid",
                "size": [1.2, 1.0, 0.2],
                "pos": [1.0, 1.0, 0.1],
                "kinematic": True,
                "capabilities": ["supports_objects"],
            },
            {
                "name": "object",
                "type": "cuboid",
                "size": [0.2, 0.2, 0.2],
                "pos": [1.0, 1.0, 0.3],
                "rigid_object": True,
                "capabilities": ["movable", "graspable"],
                "mass": 0.2,
                "static_friction": 0.8,
                "dynamic_friction": 0.6,
                "contact_offset": 0.008,
            },
            {
                "name": "target",
                "type": "cuboid",
                "size": [0.5, 0.5, 0.01],
                "pos": [2.0, 1.5, 0.005],
                "kinematic": True,
                "collision_enabled": False,
                "capabilities": ["contains_objects"],
                "regions": [
                    {
                        "name": "goal",
                        "shape": "cuboid",
                        "center": [0.0, 0.0, 0.1],
                        "size": [0.5, 0.5, 0.2],
                    }
                ],
            },
            {
                "name": "obstacle",
                "type": "cuboid",
                "size": [0.4, 0.4, 0.8],
                "pos": [-0.5, 0.8, 0.4],
                "kinematic": True,
            },
        ],
    }


def _spec() -> dict:
    return {
        "schema_version": "scene_randomization.v1",
        "max_attempts": 32,
        "preserve_relations": True,
        "robot": {"xy_radius_m": 0.25, "yaw_range_rad": 0.4},
        "objects": [
            {"match": {"role": "support"}, "xy_radius_m": 0.10, "yaw_range_rad": 0.1},
            {"match": {"role": "object"}, "xy_radius_m": 0.05, "yaw_range_rad": 0.2},
            {"match": {"role": "target"}, "xy_radius_m": 0.10, "yaw_range_rad": 0.2},
            {"match": {"role": "obstacle"}, "xy_radius_m": 0.05, "yaw_range_rad": 0.1},
        ],
        "physics": [
            {
                "match": {"role": "object"},
                "mass_scale": [0.9, 1.1],
                "friction_scale": [0.9, 1.1],
                "contact_offset_scale": [0.95, 1.05],
            }
        ],
    }


def test_randomization_is_deterministic_and_preserves_support_relation() -> None:
    first, first_meta = randomize_scene_data(_scene(), random.Random(7), _spec())
    second, second_meta = randomize_scene_data(_scene(), random.Random(7), _spec())

    assert first == second
    assert first_meta == second_meta
    assert first_meta["feasibility"]["valid"] is True
    assert {tuple(item.values()) for item in first_meta["support_relations"]} == {
        ("support", "object")
    }
    support = next(item for item in first["objects"] if item["name"] == "support")
    obj = next(item for item in first["objects"] if item["name"] == "object")
    assert obj["pos"][2] - support["pos"][2] == pytest.approx(0.2, abs=1e-5)
    assert first["objects"][1]["mass"] != _scene()["objects"][1]["mass"]


def test_feasibility_rejects_collision_and_robot_overlap() -> None:
    data = _scene()
    data["objects"][3]["pos"] = [-2.0, -2.0, 0.4]
    data["objects"][3]["size"] = [0.8, 0.8, 0.8]
    report = check_scene_feasibility(data)
    assert not report.valid
    assert any("robot initial pose overlaps" in reason for reason in report.reasons)


def test_feasibility_rejects_pose_inside_runtime_navigation_clearance() -> None:
    data = _scene()
    # The obstacle is outside the old centre-point overlap test, but the
    # runtime A* grid would still mark the robot's initial cell occupied after
    # chassis-footprint inflation and rasterization.
    data["objects"][3]["pos"] = [-2.0, -1.5, 0.4]
    report = check_scene_feasibility(data)
    assert not report.valid
    assert any("lacks navigation clearance" in reason for reason in report.reasons)


def test_randomization_fails_closed_when_robot_budget_is_infeasible() -> None:
    spec = {
        "max_attempts": 2,
        "robot": {"xy_radius_m": 20.0, "yaw_range_rad": 0.0},
    }
    with pytest.raises(SceneRandomizationError, match="could not sample"):
        randomize_scene_data(_scene(), random.Random(1), spec)
