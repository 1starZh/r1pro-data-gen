"""Pure IK selection and base-path query behavior."""

from __future__ import annotations

import numpy as np

from r1pro_data_gen.domain import Observation
from r1pro_data_gen.skills import (
    ArmMoveThrough,
    QueryArmPath,
    QueryBasePath,
    QueryIKSolution,
    SkillResult,
)

from tests.support import FakeAdapter, FakeKinematics, TensorStub, load_fixture_scene


def test_runtime_snapshot_exclusions_prune_sensor_filters():
    """Excluded planning objects must not leave invalid sensor references."""
    from r1pro_data_gen.skills.planning import runtime_scene_snapshot

    scene = load_fixture_scene("tabletop_basic")
    excluded = {"pick_cylinder"}
    snapshot = runtime_scene_snapshot(scene, adapter=None, exclude_objects=tuple(excluded))

    assert all(obj.name not in excluded for obj in snapshot.objects)
    for sensor in (*snapshot.contact_sensors, *snapshot.collision_sensors):
        assert excluded.isdisjoint(sensor.filter)


def test_query_ik_solution_returns_joint_config():
    class SolvableKinematics(FakeKinematics):
        def ik(self, target_pos, target_quat, q_init=None):
            from r1pro_data_gen.robot.kinematics import IKSolution

            return IKSolution(True, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), 0.01, 0.01, 5, "ok")

    result = QueryIKSolution(SolvableKinematics()).execute(FakeAdapter(), None, target_pos=[0.4, 0.0, 1.2])
    assert result.success
    assert len(result.details["q_arm"]) == 7


def test_query_ik_solution_reports_unreachable():
    class UnreachableKinematics(FakeKinematics):
        def ik(self, target_pos, target_quat, q_init=None):
            from r1pro_data_gen.robot.kinematics import IKSolution

            return IKSolution(False, None, 0.3, 0.2, 500, "no solution")

    result = QueryIKSolution(UnreachableKinematics()).execute(FakeAdapter(), None, target_pos=[9.0, 9.0, 9.0])
    assert not result.success
    assert result.metrics["ik_error_m"] == 0.3


def test_query_ik_solution_selects_right_backend():
    class SideKinematics(FakeKinematics):
        def __init__(self, marker):
            self.marker = marker

        def ik(self, target_pos, target_quat, q_init=None):
            from r1pro_data_gen.robot.kinematics import IKSolution

            return IKSolution(True, np.full(7, self.marker), 0.0, 0.0, 1, "ok")

    skill = QueryIKSolution({"left": SideKinematics(0.1), "right": SideKinematics(0.7)})
    result = skill.execute(FakeAdapter(), None, target_pos=[0.4, 0.0, 1.2], side="right")
    assert result.success
    assert result.details["q_arm"] == [0.7] * 7


def test_pick_min_motion_solution_selects_smallest_motion():
    from r1pro_data_gen.robot.kinematics import IKSolution, pick_min_motion_solution

    q_ref = np.zeros(7)
    big = IKSolution(True, np.ones(7), 0.01, 0.01, 5, "ok")
    small = IKSolution(True, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]), 0.01, 0.01, 8, "ok")
    assert pick_min_motion_solution([big, small], q_ref) is small
    tie = IKSolution(True, np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.01, 0.01, 6, "ok")
    assert pick_min_motion_solution([small, tie], q_ref) is small


def test_query_arm_path_uses_shared_optimizer(monkeypatch):
    from types import SimpleNamespace

    from r1pro_data_gen.methods.manipulation.contracts import (
        ArmPlanningResult,
        ConstraintReport,
        PathCandidate,
    )
    q_goal = tuple([0.1] * 7)
    output = {
        "position": np.stack([np.zeros(7), np.full(7, 0.1)]),
        "velocity": np.zeros((2, 7)),
        "acceleration": np.zeros((2, 7)),
        "duration": 1.0,
        "dt": 1.0 / 60.0,
        "winding": 1.0,
        "ee_winding": 2.2,
    }
    winner = PathCandidate(
        candidate_id=0,
        attempt_id=0,
        fallback=False,
        q_goal=q_goal,
        planner_status="Success",
        constraints=ConstraintReport(True, "verified"),
        metrics={
            "ee_path_length_m": 0.2,
            "normalized_joint_path_length": 0.1,
            "smoothness_cost": 0.0,
        },
        score=(0,),
        output=output,
    )
    captured = {}

    def optimize(*args, **kwargs):
        captured["solutions"] = args[3]
        captured["full_q_current"] = kwargs["full_q_current"]
        return ArmPlanningResult(
            True,
            "success",
            "selected",
            "request-hash",
            (winner,),
            winner,
        )

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.arm_path_optimizer.optimize_arm_path",
        optimize,
    )
    joints = {
        **{f"left_arm_joint{i}": 0.0 for i in range(1, 8)},
        "right_arm_joint1": 0.7,
    }
    result = QueryArmPath(object(), FakeKinematics()).execute(
        FakeAdapter(joint_positions=joints),
        load_fixture_scene("bare"),
        target_q=list(q_goal),
    )

    assert result.success
    assert result.metrics["ee_winding"] == 2.2
    assert result.details["optimality_scope"] == "best_verified_candidate_within_budget"
    assert result.details["request_hash"] == "request-hash"
    assert captured["solutions"][0].q_arm.tolist() == list(q_goal)
    assert captured["full_q_current"][13] == 0.7


def test_query_arm_path_reports_candidate_failure(monkeypatch):
    from r1pro_data_gen.methods.manipulation.contracts import (
        ArmPlanningResult,
        ConstraintReport,
        PathCandidate,
    )
    report = PathCandidate(
        candidate_id=0,
        attempt_id=0,
        fallback=False,
        q_goal=tuple([0.1] * 7),
        planner_status="TimedOut",
        constraints=ConstraintReport(False, "mplib_plan", ("time budget exhausted",)),
    )
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.arm_path_optimizer.optimize_arm_path",
        lambda *_args, **_kwargs: ArmPlanningResult(
            False,
            "no_collision_free_path",
            "no verified path",
            "request-hash",
            (report,),
        ),
    )
    joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
    result = QueryArmPath(object(), FakeKinematics()).execute(
        FakeAdapter(joint_positions=joints),
        load_fixture_scene("bare"),
        target_q=[0.1] * 7,
    )

    assert not result.success
    assert result.details["status"] == "no_collision_free_path"
    assert result.details["candidates"][0]["failure_stage"] == "mplib_plan"


def _sequence_planning_result(success=True):
    from r1pro_data_gen.methods.manipulation.contracts import (
        ArmSequenceCandidate,
        ArmSequencePlanningResult,
        ConstraintReport,
        WaypointIKCandidate,
    )

    if not success:
        return ArmSequencePlanningResult(
            False,
            "no_complete_waypoint_path",
            "no verified path",
            "sequence-hash",
            (),
        )
    waypoint_candidate = WaypointIKCandidate(
        waypoint_id=0,
        candidate_id=0,
        orientation_id=0,
        q_goal=tuple([0.1] * 7),
        position_error_m=0.001,
        rotation_error_rad=0.002,
        continuity_cost=0.1,
        posture_cost=0.1,
        minimum_limit_margin=0.2,
        wrist_motion=0.05,
        minimum_singular_value=0.1,
        score=(0,),
    )
    output = {
        "position": np.stack([np.zeros(7), np.full(7, 0.1)]),
        "velocity": np.zeros((2, 7)),
        "acceleration": np.zeros((2, 7)),
        "duration": 1.0,
        "dt": 1.0 / 60.0,
    }
    winner = ArmSequenceCandidate(
        sequence_id=0,
        waypoint_candidates=(waypoint_candidate,),
        segment_reports=(),
        constraints=ConstraintReport(True, "verified"),
        metrics={"naturalness_cost": 0.1},
        score=(0,),
        output=output,
    )
    return ArmSequencePlanningResult(
        True,
        "success",
        "selected",
        "sequence-hash",
        (winner,),
        winner,
    )


def _arm_move_through_waypoints():
    return [
        {
            "name": "target",
            "poses": [
                {
                    "position": [0.4, 0.0, 1.2],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                }
            ],
        }
    ]


def test_arm_move_through_executes_one_complete_trajectory(monkeypatch):
    optimize_calls = []
    follow_calls = []
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.arm_path_optimizer.optimize_arm_waypoint_path",
        lambda *args, **kwargs: optimize_calls.append((args, kwargs))
        or _sequence_planning_result(),
    )
    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion.ArmTrajectoryFollow.execute",
        lambda _self, _adapter, **kwargs: follow_calls.append(kwargs)
        or SkillResult(True, "arm_trajectory_follow"),
    )
    joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
    result = ArmMoveThrough(FakeKinematics(), np.ones(7), object()).execute(
        FakeAdapter(joint_positions=joints),
        load_fixture_scene("bare"),
        waypoints=_arm_move_through_waypoints(),
    )

    assert result.success
    assert len(optimize_calls) == 1
    assert len(follow_calls) == 1
    assert len(follow_calls[0]["trajectory"]) == 2


def test_arm_move_through_fails_when_carried_object_detaches_during_execution(monkeypatch):
    from r1pro_data_gen.domain import GraspContext
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.arm_path_optimizer.optimize_arm_waypoint_path",
        lambda *args, **kwargs: _sequence_planning_result(),
    )
    monkeypatch.setattr(
        "r1pro_data_gen.methods.collision.carried_object_path_free",
        lambda *args, **kwargs: (True, {}),
    )

    class DetachingAdapter(FakeAdapter):
        def __init__(self):
            super().__init__(
                joint_positions={f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
            )
            self.attached = True
            self.steps_after_start = 0

        def attachment_state(self):
            return (
                {"item": "left_gripper_finger_midpoint"}
                if self.attached
                else {}
            )

        def step(self, render=True):
            self.steps_after_start += 1
            if self.steps_after_start >= 1:
                self.attached = False

    context = GraspContext(
        object_name="item",
        side="left",
        attached=True,
        object_position_world=(0.0, 0.0, 1.0),
        grasp_center_world=(0.0, 0.0, 1.0),
        object_to_grasp_center_world=(0.0, 0.0, 0.0),
        attachment_error_m=0.0,
    )
    result = ArmMoveThrough(FakeKinematics(), np.ones(7), object()).execute(
        DetachingAdapter(),
        load_fixture_scene("bare"),
        waypoints=_arm_move_through_waypoints(),
        carried_context=context,
    )

    assert not result.success
    assert result.metrics["held_context_verified"] is False
    assert result.metrics["failure_code"] == "attachment_lost"


def test_arm_move_through_planning_failure_executes_nothing(monkeypatch):
    follow_calls = []
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.arm_path_optimizer.optimize_arm_waypoint_path",
        lambda *_args, **_kwargs: _sequence_planning_result(False),
    )
    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion.ArmTrajectoryFollow.execute",
        lambda *_args, **_kwargs: follow_calls.append(True),
    )
    joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
    result = ArmMoveThrough(FakeKinematics(), np.ones(7), object()).execute(
        FakeAdapter(joint_positions=joints),
        load_fixture_scene("bare"),
        waypoints=_arm_move_through_waypoints(),
    )

    assert not result.success
    assert follow_calls == []


def test_arm_move_through_freezes_scene_and_rejects_stale_object(monkeypatch):
    scene = load_fixture_scene("tabletop_basic")
    calls = {obj.name: 0 for obj in scene.objects}

    class MovingObjectAdapter(FakeAdapter):
        def object_position(self, name):
            calls[name] += 1
            position = np.asarray(scene.object(name).pos, dtype=float)
            if calls[name] >= 2 and name == scene.objects[0].name:
                position[0] += 0.01
            return tuple(position)

    def optimize(*_args, **kwargs):
        kwargs["scene_for_exclusions"](())
        kwargs["scene_for_exclusions"](())
        return _sequence_planning_result()

    follow_calls = []
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.arm_path_optimizer.optimize_arm_waypoint_path",
        optimize,
    )
    monkeypatch.setattr(
        "r1pro_data_gen.skills.manipulation.arm_motion.ArmTrajectoryFollow.execute",
        lambda *_args, **_kwargs: follow_calls.append(True),
    )
    joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
    result = ArmMoveThrough(FakeKinematics(), np.ones(7), object()).execute(
        MovingObjectAdapter(joint_positions=joints),
        scene,
        waypoints=_arm_move_through_waypoints(),
    )

    assert not result.success
    assert result.details["planning_status"] == "stale_scene"
    assert result.details["scene_changed"] is True
    assert follow_calls == []
    assert all(count == 2 for count in calls.values())


def test_query_base_path_rejects_goal_inside_inflated_obstacle_with_diagnostics():
    class FarStartAdapter(FakeAdapter):
        class Robot:
            class Data:
                root_pos_w = [TensorStub([-1.5, 0.0, 0.0])]
                root_quat_w = [TensorStub([1.0, 0.0, 0.0, 0.0])]

            data = Data()

        robot = Robot()

        def read_observation(self, timestamp):
            return Observation(timestamp=timestamp, base_pose=(-1.5, 0.0, 0.0))

    result = QueryBasePath().execute(
        FarStartAdapter(), load_fixture_scene("tabletop_navigation"), target=[0.5, 0.3, 0.0], resolution=0.05
    )

    assert not result.success
    assert result.details["reason"] == "goal cell is inside an obstacle"
    assert result.details["target"] == [0.5, 0.3, 0.0]
    assert result.details["footprint_radius_m"] > 0.0


def test_base_navigate_uses_scene_footprint_when_no_override():
    from r1pro_data_gen.data.scenes import load_scene_data
    from r1pro_data_gen.skills import BaseNavigateTo
    from r1pro_data_gen.tasks import load_task_spec

    scene = load_scene_data(load_task_spec("pickplace.tabletop").scene)
    result = BaseNavigateTo().execute(
        FakeAdapter(), scene, target=[-0.6, 0.3, 0.0], resolution=0.2
    )

    assert result.details["footprint_radius_m"] == 0.25


    from r1pro_data_gen.skills import BaseNavigateTo

    class FarStartAdapter(FakeAdapter):
        class Robot:
            class Data:
                root_pos_w = [TensorStub([-1.5, 0.0, 0.0])]
                root_quat_w = [TensorStub([1.0, 0.0, 0.0, 0.0])]

            data = Data()

        robot = Robot()

        def read_observation(self, timestamp):
            return Observation(timestamp=timestamp, base_pose=(-1.5, 0.0, 0.0))

    result = BaseNavigateTo().execute(
        FarStartAdapter(), load_fixture_scene("tabletop_navigation"), target=[0.5, 0.3, 0.0], resolution=0.05
    )

    assert not result.success
    assert result.details["reason"] == "goal cell is inside an obstacle"
    assert result.details["target"] == [0.5, 0.3, 0.0]
    assert result.details["footprint_radius_m"] > 0.0


def test_query_base_path_uses_scene_footprint_when_no_override():
    from r1pro_data_gen.data.scenes import load_scene_data
    from r1pro_data_gen.tasks import load_task_spec

    scene = load_scene_data(load_task_spec("pickplace.tabletop").scene)
    result = QueryBasePath().execute(
        FakeAdapter(), scene, target=[-0.6, 0.3, 0.0], resolution=0.2
    )

    assert result.success
    assert result.details["footprint_radius_m"] == 0.25


    result = QueryBasePath().execute(
        FakeAdapter(), load_fixture_scene("tabletop_basic"), target=[-0.6, 0.3, 0.0], resolution=0.2, footprint_radius=0.1
    )
    assert result.success
    assert len(result.details["path"]) >= 2
