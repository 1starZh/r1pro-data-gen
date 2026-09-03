from __future__ import annotations

import math

import numpy as np
import pytest

from r1pro_data_gen.methods import occupancy_from_boxes
from r1pro_data_gen.planning.navigation.contract import NAVIGATION_GRID_RESOLUTION_M
from r1pro_data_gen.robot.kinematics import R1ProKinematics
from tests.support import PROJECT_ROOT, load_fixture_scene


URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"
GRASP_QUAT = np.array([0.70710678, 0.0, -0.70710678, 0.0])


def test_collision_free_approach_is_rejected_when_interaction_target_is_unreachable():
    """A generic candidate contract must include robot-level reachability."""
    try:
        from r1pro_data_gen.planning.navigation.reachability import assess_interaction_target
    except ImportError as exc:
        pytest.fail(f"generic reachability contract is missing: {exc}")

    scene = load_fixture_scene("tabletop_navigation")
    facts = __import__(
        "r1pro_data_gen.planning.context.facts", fromlist=["scene_to_facts"]
    ).scene_to_facts(scene)
    support = next(
        obj
        for obj in scene.objects
        if obj.type.value == "cuboid" and obj.physics.collision_enabled
    )
    target = next(obj for obj in scene.objects if obj.type.value == "cylinder")
    candidate = next(
        item
        for item in facts["navigation"]["approach_candidates"]
        if item["obstacle_name"] == support.name
        and item["side"] == "south"
        and abs(item["pose"][0] - support.pos[0]) < 1e-6
    )

    radius = facts["navigation"]["footprint_radius_m"]
    inflate = radius + facts["navigation"]["inflation_clearance_m"]
    bx, by, _ = scene.robot.init_pose
    tx, ty, _ = candidate["pose"]
    pad = 1.5
    origin_x = min(bx, tx) - pad
    origin_y = min(by, ty) - pad
    rows = max(
        2,
        int(
            math.ceil(
                (max(by, ty) + pad - origin_y) / NAVIGATION_GRID_RESOLUTION_M
            )
        ),
    )
    cols = max(
        2,
        int(
            math.ceil(
                (max(bx, tx) + pad - origin_x) / NAVIGATION_GRID_RESOLUTION_M
            )
        ),
    )
    grid = occupancy_from_boxes(
        [
            (
                obj.pos[0] - obj.size[0] / 2.0 - inflate,
                obj.pos[1] - obj.size[1] / 2.0 - inflate,
                obj.pos[0] + obj.size[0] / 2.0 + inflate,
                obj.pos[1] + obj.size[1] / 2.0 + inflate,
            )
            for obj in scene.objects
            if obj.type.value == "cuboid" and obj.physics.collision_enabled
        ],
        origin_x,
        origin_y,
        NAVIGATION_GRID_RESOLUTION_M,
        (rows, cols),
    )
    candidate_cell = (
        int((ty - origin_y) / NAVIGATION_GRID_RESOLUTION_M),
        int((tx - origin_x) / NAVIGATION_GRID_RESOLUTION_M),
    )
    assert not grid[candidate_cell]

    report = assess_interaction_target(
        candidate_pose_world=candidate["pose"],
        target_position_world=target.pos,
        target_quaternion=GRASP_QUAT,
        target_frame="grasp_center",
        kinematics=R1ProKinematics(str(URDF), side="left"),
    )

    assert report.navigation_free is True
    assert report.target_reachable is False
    assert report.accepted is False
    assert report.reason == "interaction target is not reachable"
