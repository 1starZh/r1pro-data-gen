"""Contract tests for finite-budget arm path candidate selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from r1pro_data_gen.methods.manipulation.arm_path_optimizer import (
    optimize_arm_path,
    optimize_arm_waypoint_path,
)
from r1pro_data_gen.methods.manipulation.contracts import ArmWaypoint


class _Kin:
    lower = np.full(7, -2.0)
    upper = np.full(7, 2.0)

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        return q[:3], np.array([1.0, 0.0, 0.0, 0.0])

    def posture_score(self, q, current):
        return float(np.linalg.norm(np.asarray(q) - np.asarray(current)))

    def minimum_singular_value(self, q):
        return 0.1

    def ik_candidates(self, target_pos, target_quat, q_current, max_candidates=4):
        del target_quat, max_candidates
        target = float(np.asarray(target_pos)[0])
        if np.isclose(target, 1.0):
            values = (0.1, 0.8)
        else:
            values = (0.9,)
        return [_solution(value) for value in values]


def _solution(value: float):
    return SimpleNamespace(
        q_arm=np.full(7, value),
        position_error=0.001,
        rotation_error=0.002,
    )


def _scene(x: float = 1.0):
    physics = SimpleNamespace(collision_enabled=True, planning_margin=0.05)
    obj = SimpleNamespace(
        name="table",
        type=SimpleNamespace(value="cuboid"),
        pos=(x, 0.0, 0.5),
        size=(1.0, 1.0, 0.1),
        radius=None,
        height=None,
        physics=physics,
    )
    return SimpleNamespace(objects=(obj,))


def _success(q_goal, *, ee_offset=0.0, winding=1.0, ee_winding=1.0):
    start = np.zeros(7)
    middle = (start + q_goal) / 2.0
    middle[1] += ee_offset
    position = np.stack([start, middle, q_goal])
    return {
        "success": True,
        "position": position,
        "velocity": np.zeros_like(position),
        "acceleration": np.zeros_like(position),
        "duration": 1.0,
        "dt": 0.5,
        "winding": winding,
        "ee_winding": ee_winding,
        "status": "Success",
        "reason": None,
    }


def _run(monkeypatch, solutions, backend, **kwargs):
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path",
        backend,
    )
    return optimize_arm_path(
        object(),
        _Kin(),
        np.zeros(7),
        solutions,
        _scene(),
        base_xy=(0.0, 0.0),
        base_yaw=0.0,
        full_q_current=np.zeros(22),
        planning_time=1.0,
        local_radius_m=2.0,
        speed_scale=0.1,
        side="left",
        **kwargs,
    )


def test_round_robin_gives_every_ik_equal_attempts(monkeypatch):
    calls = []

    def backend(_planner, _q_cur, q_goal, _scene, **kwargs):
        calls.append((round(float(q_goal[0]), 2), kwargs["allow_rrt_fallback"]))
        return {
            "success": False,
            "status": "TimedOut",
            "failure_stage": "mplib_plan",
            "reason": "time budget exhausted",
        }

    result = _run(
        monkeypatch,
        [_solution(0.1), _solution(0.2)],
        backend,
        attempts_per_candidate=2,
        fallback_attempts_per_candidate=1,
    )

    assert calls == [
        (0.1, False), (0.2, False),
        (0.1, False), (0.2, False),
        (0.1, True), (0.2, True),
    ]
    assert not result.success
    assert [item.attempt_id for item in result.candidates] == [0, 0, 1, 1, 2, 2]
    assert all(item.constraints.stage == "mplib_plan" for item in result.candidates)


def test_default_winding_is_diagnostic_not_rejection(monkeypatch):
    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        return _success(q_goal, winding=3.0, ee_winding=2.5)

    result = _run(
        monkeypatch,
        [_solution(0.2)],
        backend,
        attempts_per_candidate=1,
        fallback_attempts_per_candidate=0,
    )

    assert result.success
    assert result.winner is not None
    assert result.winner.metrics["ee_winding"] == 2.5


def test_explicit_winding_limit_is_reported_as_task_quality(monkeypatch):
    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        return _success(q_goal, winding=1.2, ee_winding=2.5)

    result = _run(
        monkeypatch,
        [_solution(0.2)],
        backend,
        attempts_per_candidate=1,
        fallback_attempts_per_candidate=0,
        max_ee_winding=2.0,
    )

    assert not result.success
    report = result.candidates[0]
    assert report.constraints.stage == "task_quality_limit"
    assert report.constraints.reasons == ("task_ee_winding_limit",)


def test_unique_winner_uses_absolute_path_quality_then_stable_tie_break(monkeypatch):
    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        offset = 0.3 if np.isclose(q_goal[0], 0.1) else 0.0
        return _success(q_goal, ee_offset=offset)

    result = _run(
        monkeypatch,
        [_solution(0.1), _solution(0.2)],
        backend,
        attempts_per_candidate=2,
        fallback_attempts_per_candidate=0,
    )

    assert result.success
    assert result.winner is not None
    assert result.winner.candidate_id == 1
    assert result.winner.attempt_id == 0
    assert sum(candidate.valid for candidate in result.candidates) == 4


def test_request_hash_is_stable_and_tracks_live_scene(monkeypatch):
    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        return _success(q_goal)

    first = _run(
        monkeypatch,
        [_solution(0.2)],
        backend,
        attempts_per_candidate=1,
        fallback_attempts_per_candidate=0,
    )
    second = _run(
        monkeypatch,
        [_solution(0.2)],
        backend,
        attempts_per_candidate=1,
        fallback_attempts_per_candidate=0,
    )
    assert first.request_hash == second.request_hash

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path",
        backend,
    )
    moved = optimize_arm_path(
        object(), _Kin(), np.zeros(7), [_solution(0.2)], _scene(1.1),
        base_xy=(0.0, 0.0), base_yaw=0.0,
        full_q_current=np.zeros(22), planning_time=1.0,
        local_radius_m=2.0, speed_scale=0.1, side="left",
        attempts_per_candidate=1, fallback_attempts_per_candidate=0,
    )
    assert first.request_hash != moved.request_hash


def _allow_reference(monkeypatch):
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.validate_reference_trajectory",
        lambda position, **_kwargs: {
            "valid": True,
            "reasons": (),
            "velocity": np.zeros_like(position),
            "acceleration": np.zeros_like(position),
        },
    )


def _waypoint(name, x, exclusions=(), contact=False):
    return ArmWaypoint(
        name=name,
        poses=(
            (
                (float(x), 0.0, 1.0),
                (1.0, 0.0, 0.0, 0.0),
            ),
        ),
        exclude_objects=tuple(exclusions),
        contact=contact,
    )


def test_waypoint_optimizer_selects_complete_continuous_branch(monkeypatch):
    _allow_reference(monkeypatch)
    edge_calls = []

    def backend(_planner, q_cur, q_goal, _scene, **_kwargs):
        edge_calls.append((round(float(q_cur[0]), 1), round(float(q_goal[0]), 1)))
        if np.isclose(q_cur[0], 0.1) and np.isclose(q_goal[0], 0.9):
            return {
                "success": False,
                "status": "blocked",
                "failure_stage": "mplib_plan",
                "reason": "branch cannot continue",
            }
        return _success(q_goal)

    def certify(_planner, path, _scene, **_kwargs):
        path = np.asarray(path, dtype=float)
        return {
            "success": True,
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "duration": float(len(path) - 1),
            "dt": 1.0,
            "status": "SequenceVerified",
            "reason": None,
        }

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path",
        backend,
    )
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.retime_and_validate_path",
        certify,
    )
    result = optimize_arm_waypoint_path(
        object(),
        _Kin(),
        np.zeros(7),
        (_waypoint("retract", 1.0), _waypoint("target", 2.0)),
        _scene(),
        scene_for_exclusions=lambda _items: _scene(),
        base_xy=(0.0, 0.0),
        base_yaw=0.0,
        full_q_current=np.zeros(22),
        planning_time=0.1,
        local_radius_m=2.0,
        speed_scale=0.2,
        side="left",
        beam_width=2,
    )

    assert result.success
    assert result.winner is not None
    assert [round(item.q_goal[0], 1) for item in result.winner.waypoint_candidates] == [0.8, 0.9]
    assert (0.1, 0.9) in edge_calls


def test_waypoint_optimizer_recovers_endpoint_ik_through_cartesian_substeps(monkeypatch):
    _allow_reference(monkeypatch)

    class InterpolatedKin(_Kin):
        def ik_candidates(self, target_pos, target_quat, q_current, max_candidates=4):
            del target_quat, max_candidates
            target = float(np.asarray(target_pos)[0])
            current = float(np.asarray(q_current)[0])
            # The endpoint is reachable, but a direct solve from the home
            # branch is outside the numerical convergence basin.
            if target > 0.6 and current < 0.6:
                return []
            return [_solution(target)]

        def _ik_once(self, target_pos, target_quat, q_init, **_kwargs):
            del target_quat, q_init
            return _solution(float(np.asarray(target_pos)[0]))

    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        return _success(q_goal)

    def certify(_planner, path, _scene, **_kwargs):
        path = np.asarray(path, dtype=float)
        return {
            "success": True,
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "duration": float(len(path) - 1),
            "dt": 1.0,
            "status": "SequenceVerified",
            "reason": None,
        }

    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path", backend)
    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.retime_and_validate_path", certify)
    result = optimize_arm_waypoint_path(
        object(), InterpolatedKin(), np.zeros(7), (_waypoint("descend", 1.0),),
        _scene(), scene_for_exclusions=lambda _items: _scene(),
        base_xy=(0.0, 0.0), base_yaw=0.0, full_q_current=np.zeros(22),
        planning_time=0.1, local_radius_m=2.0, speed_scale=0.2,
        side="left", beam_width=1,
    )

    assert result.success
    assert result.winner is not None
    assert np.allclose(result.winner.waypoint_candidates[0].q_goal, np.ones(7))


def test_waypoint_candidate_limit_applies_across_all_orientations(monkeypatch):
    _allow_reference(monkeypatch)
    planned = []

    class ManyPoseKin(_Kin):
        def ik_candidates(self, target_pos, target_quat, q_current, max_candidates=4):
            del target_pos, q_current
            orientation_marker = round(float(np.asarray(target_quat)[3]), 1)
            offset = 0.0 if orientation_marker == 0.0 else 0.5
            return [_solution(offset + value) for value in (0.1, 0.2, 0.3)]

    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        planned.append(round(float(q_goal[0]), 1))
        return _success(q_goal)

    def certify(_planner, path, _scene, **_kwargs):
        path = np.asarray(path, dtype=float)
        return {
            "success": True,
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "duration": float(len(path) - 1),
            "dt": 1.0,
            "status": "SequenceVerified",
            "reason": None,
        }

    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path", backend)
    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.retime_and_validate_path", certify)
    waypoint = ArmWaypoint(
        name="many_poses",
        poses=(
            ((1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)),
            ((1.0, 0.0, 1.0), (0.995, 0.0, 0.0, 0.1)),
        ),
    )
    result = optimize_arm_waypoint_path(
        object(), ManyPoseKin(), np.zeros(7), (waypoint,), _scene(),
        scene_for_exclusions=lambda _items: _scene(),
        base_xy=(0.0, 0.0), base_yaw=0.0,
        full_q_current=np.zeros(22), planning_time=0.1,
        local_radius_m=2.0, speed_scale=0.2, side="left",
        ik_candidates_per_waypoint=2, beam_width=2, max_planned_edges=2,
    )

    assert result.success
    assert planned == [0.1, 0.2]


def test_waypoint_optimizer_keeps_zero_velocity_boundary_between_semantic_groups(monkeypatch):
    _allow_reference(monkeypatch)

    def backend(_planner, q_cur, q_goal, _scene, **_kwargs):
        output = _success(q_goal)
        output["position"][0] = np.asarray(q_cur, dtype=float)
        return output

    def certify(_planner, path, _scene, **_kwargs):
        path = np.asarray(path, dtype=float)
        return {
            "success": True,
            "position": np.stack([path[0], path[-1]]),
            "velocity": np.zeros((2, 7)),
            "acceleration": np.zeros((2, 7)),
            "duration": 1.0,
            "dt": 1.0,
            "status": "SequenceVerified",
            "reason": None,
        }

    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path", backend)
    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.retime_and_validate_path", certify)
    result = optimize_arm_waypoint_path(
        object(), _Kin(), np.zeros(7),
        (
            _waypoint("carry", 1.0, ("pick",)),
            _waypoint("place", 2.0, ("pick", "table"), contact=True),
        ),
        _scene(), scene_for_exclusions=lambda _items: _scene(),
        base_xy=(0.0, 0.0), base_yaw=0.0,
        full_q_current=np.zeros(22), planning_time=0.1,
        local_radius_m=2.0, speed_scale=0.2, side="left",
        beam_width=1,
    )

    assert result.success
    position = np.asarray(result.winner.output["position"])
    assert len(position) == 4
    assert np.allclose(position[1], position[2])


def test_waypoint_optimizer_recertifies_each_collision_semantic_group(monkeypatch):
    _allow_reference(monkeypatch)
    scenes = []

    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        return _success(q_goal)

    def certify(_planner, path, scene, **_kwargs):
        scenes.append(tuple(obj.name for obj in scene.objects))
        path = np.asarray(path, dtype=float)
        return {
            "success": True,
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "duration": float(len(path) - 1),
            "dt": 1.0,
            "status": "SequenceVerified",
            "reason": None,
        }

    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path", backend)
    monkeypatch.setattr("r1pro_data_gen.methods.manipulation.mplib_path.retime_and_validate_path", certify)

    def snapshot(exclusions):
        return SimpleNamespace(
            objects=tuple(
                obj
                for obj in _scene().objects
                if obj.name not in set(exclusions)
            )
        )

    result = optimize_arm_waypoint_path(
        object(), _Kin(), np.zeros(7),
        (
            _waypoint("carry", 1.0, ("pick",)),
            _waypoint("place", 2.0, ("pick", "table"), contact=True),
        ),
        _scene(), scene_for_exclusions=snapshot,
        base_xy=(0.0, 0.0), base_yaw=0.0,
        full_q_current=np.zeros(22), planning_time=0.1,
        local_radius_m=2.0, speed_scale=0.2, side="left",
        beam_width=1,
    )

    assert result.success
    assert scenes == [("table",), ()]


def test_optimize_arm_path_prefers_certified_task_space(monkeypatch):
    calls = []

    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        calls.append(tuple(np.round(q_goal, 2)))
        return _success(q_goal)

    def certified(*_args, **_kwargs):
        position = np.stack([np.zeros(7), np.full(7, 0.05)])
        return {
            "success": True,
            "position": position,
            "velocity": np.zeros_like(position),
            "acceleration": np.zeros_like(position),
            "duration": 0.5,
            "dt": 0.25,
            "winding": 1.0,
            "ee_winding": 1.0,
            "status": "TaskSpaceVerified",
            "reason": None,
        }

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path",
        backend,
    )
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.taskspace.plan_certified_task_path",
        certified,
    )
    result = optimize_arm_path(
        object(),
        _Kin(),
        np.zeros(7),
        [_solution(0.1), _solution(1.8)],
        _scene(),
        base_xy=(0.0, 0.0),
        base_yaw=0.0,
        full_q_current=np.zeros(22),
        planning_time=1.0,
        local_radius_m=2.0,
        speed_scale=0.1,
        side="left",
        attempts_per_candidate=1,
        fallback_attempts_per_candidate=0,
        target_pos=np.array([0.2, 0.0, 0.8]),
        target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert result.success
    assert result.winner is not None
    assert result.winner.planner_status == "TaskSpaceVerified"
    assert result.winner.candidate_id == -1
    assert calls == []


def test_optimize_arm_path_falls_back_when_task_space_collides(monkeypatch):
    calls = []

    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        calls.append(round(float(q_goal[0]), 2))
        return _success(q_goal)

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.plan_arm_path",
        backend,
    )
    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.taskspace.plan_certified_task_path",
        lambda *_args, **_kwargs: {
            "success": False,
            "status": "collision",
            "reason": "table edge",
            "failure_stage": "sequence_hppfcl_collision",
        },
    )
    result = _run(
        monkeypatch,
        [_solution(0.1), _solution(0.2)],
        backend,
        attempts_per_candidate=1,
        fallback_attempts_per_candidate=0,
        target_pos=np.array([0.2, 0.0, 0.8]),
        target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert result.success
    assert calls == [0.1, 0.2]
    assert result.candidates[0].planner_status == "collision"


def test_optimize_arm_path_skips_discontinuous_ik_when_a_live_branch_exists(monkeypatch):
    calls = []

    def backend(_planner, _q_cur, q_goal, _scene, **_kwargs):
        calls.append(round(float(q_goal[0]), 2))
        return _success(q_goal)

    result = _run(
        monkeypatch,
        [_solution(0.1), _solution(1.8)],
        backend,
        attempts_per_candidate=1,
        fallback_attempts_per_candidate=0,
    )

    assert result.success
    assert calls == [0.1]
    assert any(item.planner_status == "DiscontinuousIK" for item in result.candidates)
