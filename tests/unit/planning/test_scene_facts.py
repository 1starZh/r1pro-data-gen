from __future__ import annotations

import math

import pytest

from r1pro_data_gen.planning.context.facts import object_names, scene_facts_from_mapping, scene_to_facts
from r1pro_data_gen.methods import occupancy_from_boxes
from r1pro_data_gen.robot.chassis import default_footprint_radius_m
from r1pro_data_gen.data.scenes import load_scene_data
from r1pro_data_gen.tasks import load_task_spec
from tests.support import PROJECT_ROOT, load_fixture_scene


URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"


def _validated_room_scene():
    return load_scene_data(load_task_spec("pickplace.tabletop").scene)


def test_scene_facts_are_canonical_and_json_compatible():
    scene = load_fixture_scene("tabletop_navigation")
    facts = scene_to_facts(scene)
    assert facts["name"] == "tabletop_navigation_fixture"
    assert object_names(facts) == ("table", "cylinder", "crate")
    assert facts["objects"][1]["type"] == "cylinder"
    assert facts["objects"][1]["top_z"] == 1.175
    assert "asset" in facts["robot"]


def test_scene_facts_export_implicit_top_surface_for_cuboids():
    scene = load_fixture_scene("tabletop_navigation")
    facts = scene_to_facts(scene)
    table = next(item for item in facts["objects"] if item["name"] == "table")

    assert table["surfaces"] == [
        {
            "name": "top",
            "center": [0.0, 0.0, 0.05],
            "normal": [0.0, 0.0, 1.0],
            "size": [0.5, 0.8],
        }
    ]


def test_navigation_facts_export_free_space_candidates_for_collision_cuboids():
    scene = load_fixture_scene("tabletop_navigation")
    facts = scene_to_facts(scene)

    navigation = facts["navigation"]
    assert navigation["footprint_radius_m"] == default_footprint_radius_m()
    assert navigation["footprint_radius_source"] == "robot_default"
    assert any(
        item["obstacle_name"] == "table"
        for item in navigation["approach_candidates"]
    )


    scene = _validated_room_scene()
    facts = scene_to_facts(scene)

    assert facts["navigation"]["footprint_radius_m"] == 0.25
    assert facts["navigation"]["footprint_radius_source"] == "scene"
    candidates = facts["navigation"]["approach_candidates"]
    assert any(
        item["obstacle_name"] == "work_table"
        and item["side"] == "west"
        and item["pose"][0] < 1.35
        for item in candidates
    )


def test_navigation_candidates_are_free_of_other_collision_cuboids():
    scene = load_fixture_scene("tabletop_navigation")
    facts = scene_to_facts(scene)
    navigation = facts["navigation"]
    inflate = navigation["footprint_radius_m"] + navigation["inflation_clearance_m"]
    obstacles = {
        obj.name: (
            obj.pos[0] - obj.size[0] / 2.0 - inflate,
            obj.pos[1] - obj.size[1] / 2.0 - inflate,
            obj.pos[0] + obj.size[0] / 2.0 + inflate,
            obj.pos[1] + obj.size[1] / 2.0 + inflate,
        )
        for obj in scene.objects
        if obj.type.value == "cuboid" and obj.physics.collision_enabled
    }

    for candidate in navigation["approach_candidates"]:
        x, y, _ = candidate["pose"]
        for obstacle_name, (xmin, ymin, xmax, ymax) in obstacles.items():
            if obstacle_name == candidate["obstacle_name"]:
                continue
            assert not (xmin <= x <= xmax and ymin <= y <= ymax), (
                f"candidate {candidate} is inside another inflated obstacle "
                f"{obstacle_name}"
            )


def test_navigation_candidates_are_free_of_runtime_occupancy_cells():
    scene = load_fixture_scene("tabletop_navigation")
    facts = scene_to_facts(scene)
    navigation = facts["navigation"]
    resolution = 0.05
    inflate = navigation["footprint_radius_m"] + navigation["inflation_clearance_m"]
    bx, by, _ = scene.robot.init_pose

    for candidate in navigation["approach_candidates"]:
        tx, ty, _ = candidate["pose"]
        pad = 1.5
        origin_x = min(bx, tx) - pad
        origin_y = min(by, ty) - pad
        rows = max(2, int(math.ceil((max(by, ty) + pad - origin_y) / resolution)))
        cols = max(2, int(math.ceil((max(bx, tx) + pad - origin_x) / resolution)))
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
            resolution,
            (rows, cols),
        )
        row = int((ty - origin_y) / resolution)
        col = int((tx - origin_x) / resolution)
        assert not grid[row, col], (
            f"candidate is occupied in the runtime grid: {candidate}"
        )


def test_every_approach_candidate_faces_its_obstacle():
    """Each candidate pose must aim the base heading toward the obstacle."""
    scene = _validated_room_scene()
    facts = scene_to_facts(scene)
    obstacles = {item["name"]: item for item in facts["objects"]}
    for item in facts["navigation"]["approach_candidates"]:
        obstacle = obstacles[item["obstacle_name"]]
        cand_x, cand_y, cand_yaw = item["pose"]
        dx = obstacle["pos"][0] - cand_x
        dy = obstacle["pos"][1] - cand_y
        norm = math.hypot(dx, dy)
        assert norm > 1e-9, f"candidate {item} sits on its obstacle"
        facing = (math.cos(cand_yaw), math.sin(cand_yaw))
        toward = (dx / norm, dy / norm)
        alignment = facing[0] * toward[0] + facing[1] * toward[1]
        assert alignment > 0.5, (
            f"candidate side={item['side']} pose={item['pose']} faces away from "
            f"obstacle {item['obstacle_name']} (alignment={alignment:.3f})"
        )
        declared = item["facing"]
        assert isinstance(declared, list) and len(declared) == 2
        assert abs(declared[0] - round(math.cos(cand_yaw), 4)) < 1e-6
        assert abs(declared[1] - round(math.sin(cand_yaw), 4)) < 1e-6


def test_unreachable_candidate_shrinks_into_reachable_nav_free_cell():
    """Kinematics-backed candidates pull toward the obstacle until reachable."""
    from r1pro_data_gen.robot.kinematics import R1ProKinematics

    scene = _validated_room_scene()
    kin = R1ProKinematics(str(URDF))
    facts = scene_to_facts(scene, kinematics=kin)
    table_candidates = [
        item for item in facts["navigation"]["approach_candidates"]
        if item["obstacle_name"] == "work_table"
    ]
    reachable = [
        item for item in table_candidates
        if any(
            entry.get("name") == "pick_cylinder" and entry.get("reachable")
            for entry in item.get("ik_reachability", [])
        )
    ]
    assert reachable, "room_v1 must expose at least one IK-reachable approach candidate"
    # The shrunk pose must still sit in a navigation-free cell (grid check).
    from r1pro_data_gen.methods import occupancy_from_boxes

    footprint = facts["navigation"]["footprint_radius_m"]
    for item in reachable:
        pose = item["pose"]
        start = scene.robot.init_pose
        pad = 1.5
        res = 0.05
        xmin = min(start[0], pose[0]) - pad
        xmax = max(start[0], pose[0]) + pad
        ymin = min(start[1], pose[1]) - pad
        ymax = max(start[1], pose[1]) + pad
        rows = max(2, int(math.ceil((ymax - ymin) / res)))
        cols = max(2, int(math.ceil((xmax - xmin) / res)))
        boxes = []
        inflate = footprint + 0.05
        for obj in scene.objects:
            if not obj.physics.collision_enabled:
                continue
            hx = hy = 0.0
            if obj.type.value == "cuboid":
                hx, hy, _ = obj.size
                hx /= 2.0
                hy /= 2.0
            else:
                hx = hy = obj.radius
            boxes.append((obj.pos[0] - hx - inflate, obj.pos[1] - hy - inflate,
                          obj.pos[0] + hx + inflate, obj.pos[1] + hy + inflate))
        grid = occupancy_from_boxes(boxes, xmin, ymin, res, (rows, cols))
        row = int((pose[1] - ymin) / res)
        col = int((pose[0] - xmin) / res)
        assert 0 <= row < rows and 0 <= col < cols and not grid[row, col], (
            f"shrunk candidate {item} is not navigation-free"
        )


def test_shrink_skips_candidates_that_do_not_approach_dynamic_target():
    """Fence/wall candidates keep their authored pose when the approach axis
    moves away from the interaction target (no wasted shrink budget)."""
    from r1pro_data_gen.robot.kinematics import R1ProKinematics

    scene = _validated_room_scene()
    kin = R1ProKinematics(str(URDF))
    facts = scene_to_facts(scene, kinematics=kin)
    authored = scene_to_facts(scene)
    for candidate, authored_candidate in zip(
        facts["navigation"]["approach_candidates"],
        authored["navigation"]["approach_candidates"],
    ):
        if candidate["obstacle_name"].startswith("fence") or candidate["obstacle_name"].startswith("room_wall"):
            assert candidate["pose"] == authored_candidate["pose"], (
                f"{candidate['obstacle_name']} {candidate['side']} drifted: "
                f"{authored_candidate['pose']} -> {candidate['pose']}"
            )
