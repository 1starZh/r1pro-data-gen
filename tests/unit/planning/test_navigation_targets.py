"""Semantic navigation target compilation tests."""

from __future__ import annotations

import math

import pytest

from r1pro_data_gen.domain import Observation
from r1pro_data_gen.planning.navigation.targets import (
    NavigationTargetError,
    resolve_navigation_target,
)
from r1pro_data_gen.data.scenes import load_scene_data
from r1pro_data_gen.skills import BaseNavigateTo
from r1pro_data_gen.tasks import load_task_spec
from tests.support import FakeAdapter, PROJECT_ROOT, TensorStub


ROOM = load_task_spec("pickplace.tabletop").scene
FLOOR = load_task_spec("pickplace.floor_to_table_complete").scene
URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"


def test_movable_target_resolves_to_support_approach_not_object_center():
    scene = load_scene_data(ROOM)

    result = resolve_navigation_target(scene, "scene://pick_cylinder", purpose="pregrasp")

    assert result.source == "scene_approach_candidate"
    assert result.approach_side in {"west", "east", "south", "north"}
    assert result.candidate_count >= 1
    assert result.resolved_pose != pytest.approx((1.90, 2.25, 0.0))
    assert result.clearance_m is not None and result.clearance_m > 0.0


def test_preferred_pose_only_ranks_safe_candidates():
    scene = load_scene_data(ROOM)

    result = resolve_navigation_target(
        scene,
        "scene://pick_cylinder",
        purpose="pregrasp",
        preferred_pose=(2.45, 2.30, math.pi),
    )

    assert result.resolved_pose[0] > 2.0
    assert result.resolved_pose[2] == pytest.approx(math.pi, abs=1e-4)


def test_approach_side_is_a_constraint_not_a_coordinate_repair():
    scene = load_scene_data(ROOM)

    result = resolve_navigation_target(
        scene,
        "scene://pick_cylinder",
        purpose="pregrasp",
        approach_side="west",
    )

    assert result.approach_side == "west"
    assert result.resolved_pose[0] < 1.90


def test_static_target_reuses_its_geometry_candidate_outside_inflated_boundary():
    scene = load_scene_data(ROOM)

    result = resolve_navigation_target(
        scene,
        "scene://work_table",
        purpose="pregrasp",
        approach_side="west",
    )

    # The table is itself the semantic target.  Its published candidate is
    # one grid-cell beyond the inflated footprint; the resolver must not fall
    # back to the exact boundary pose (1.40 m for this scene).
    assert result.source == "scene_approach_candidate"
    assert result.resolved_pose[0] < 1.39
    assert result.candidate_count >= 1


def test_floor_object_without_support_uses_object_geometry():
    scene = load_scene_data(FLOOR)

    result = resolve_navigation_target(scene, "scene://pick_cylinder", purpose="pregrasp")

    assert result.source in {"scene_approach_candidate", "geometry_candidate"}
    object_pos = scene.object("pick_cylinder").pos
    offset = math.hypot(
        result.resolved_pose[0] - object_pos[0],
        result.resolved_pose[1] - object_pos[1],
    )
    assert 0.30 < offset < 0.50
    assert result.approach_side in {"west", "east", "south", "north"}


@pytest.mark.skipif(not URDF.exists(), reason="R1Pro URDF is required for IK-annotated facts")
def test_floor_object_kinematics_do_not_inherit_wall_annotations():
    from r1pro_data_gen.robot.kinematics import R1ProKinematics

    scene = load_scene_data(FLOOR)
    result = resolve_navigation_target(
        scene,
        "scene://pick_cylinder",
        purpose="pregrasp",
        kinematics=R1ProKinematics(str(URDF)),
    )

    assert result.source == "geometry_candidate"
    object_pos = scene.object("pick_cylinder").pos
    offset = math.hypot(
        result.resolved_pose[0] - object_pos[0],
        result.resolved_pose[1] - object_pos[1],
    )
    assert 0.30 < offset < 0.50


def test_unknown_or_non_scene_target_is_rejected_before_execution():
    scene = load_scene_data(ROOM)

    with pytest.raises(NavigationTargetError, match="scene://"):
        resolve_navigation_target(scene, "pick_cylinder")
    with pytest.raises(KeyError):
        resolve_navigation_target(scene, "scene://missing_object")


def test_navigation_skill_records_compiled_target(monkeypatch):
    scene = load_scene_data(ROOM)

    def drive(_adapter, _target, _v_max, _omega_max, _tol, _steps, _hook, **_kwargs):
        return {
            "success": True,
            "steps": 1.0,
            "arrival_error_m": 0.0,
            "max_lateral_command_mps": 0.0,
        }

    def rotate(_adapter, _yaw, _omega_max, _tol, _steps, _hold, _hook, **_kwargs):
        return {"success": True, "steps": 1.0, "yaw_error_rad": 0.0}

    monkeypatch.setattr("r1pro_data_gen.skills.mobility.base_motion._drive_forward_to", drive)
    monkeypatch.setattr("r1pro_data_gen.skills.mobility.base_motion._rotate_in_place", rotate)
    result = BaseNavigateTo().execute(
        FakeAdapter(),
        scene,
        target_ref="scene://pick_cylinder",
        purpose="pregrasp",
    )

    assert result.success
    assert result.details["target_ref"] == "scene://pick_cylinder"
    assert result.details["resolution_source"] == "scene_approach_candidate"
    assert result.details["resolved_target"][0] < 1.90


def test_navigation_skill_resolves_floor_object_geometry(monkeypatch):
    scene = load_scene_data(FLOOR)

    class FloorStartAdapter(FakeAdapter):
        class Robot:
            class Data:
                root_pos_w = [TensorStub([-0.8, -0.2, 0.0])]
                root_quat_w = [TensorStub([1.0, 0.0, 0.0, 0.0])]

            data = Data()

        robot = Robot()

        def read_observation(self, timestamp):
            return Observation(timestamp=timestamp, base_pose=(-0.8, -0.2, 0.04))

    def drive(_adapter, _target, _v_max, _omega_max, _tol, _steps, _hook, **_kwargs):
        return {
            "success": True,
            "steps": 1.0,
            "arrival_error_m": 0.0,
            "max_lateral_command_mps": 0.0,
        }

    def rotate(_adapter, _yaw, _omega_max, _tol, _steps, _hold, _hook, **_kwargs):
        return {"success": True, "steps": 1.0, "yaw_error_rad": 0.0}

    monkeypatch.setattr("r1pro_data_gen.skills.mobility.base_motion._drive_forward_to", drive)
    monkeypatch.setattr("r1pro_data_gen.skills.mobility.base_motion._rotate_in_place", rotate)
    result = BaseNavigateTo().execute(
        FloorStartAdapter(),
        scene,
        target_ref="scene://pick_cylinder",
        purpose="pregrasp",
    )

    assert result.success
    assert result.details["resolution_source"] in {
        "scene_approach_candidate",
        "geometry_candidate",
    }
    assert result.details["failure_code"] is None if "failure_code" in result.details else True
