from __future__ import annotations

import json

import pytest

from r1pro_data_gen.control import CommandRouter, ControllerConfig, JointGroup
from r1pro_data_gen.data import RunProvenance, write_provenance
from r1pro_data_gen.domain import (
    ControlMode,
    FailureEvidence,
    Observation,
    Plan,
    PlanStage,
    TaskResult,
    TaskStatus,
    Trajectory,
    TrajectoryPoint,
)
from r1pro_data_gen.planning import PlannerRequest
from r1pro_data_gen.planning.backends import ReplayPlanner


def make_plan() -> Plan:
    return Plan(
        task_name="r1pro_pickplace",
        stages=(
            PlanStage(name="approach", goal="approach object"),
            PlanStage(name="grasp", goal="grasp object", depends_on=("approach",)),
        ),
    )


def test_plan_rejects_unknown_dependency() -> None:
    with pytest.raises(ValueError, match="unknown stages"):
        Plan(
            task_name="task",
            stages=(PlanStage(name="grasp", goal="grasp", depends_on=("approach",)),),
        )


def test_trajectory_validates_time_and_samples_latest_point() -> None:
    trajectory = Trajectory(
        joint_names=("left_elbow",),
        points=(
            TrajectoryPoint(timestamp=0.0, joint_positions={"left_elbow": 0.0}),
            TrajectoryPoint(timestamp=1.0, joint_positions={"left_elbow": 1.0}),
        ),
    )
    assert trajectory.duration == 1.0
    assert trajectory.sample(0.5).joint_positions["left_elbow"] == 0.0
    assert trajectory.sample(1.5).joint_positions["left_elbow"] == 1.0

    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(
            joint_names=("left_elbow",),
            points=(TrajectoryPoint(timestamp=0.0), TrajectoryPoint(timestamp=0.0)),
        )


def test_replay_planner_obeys_planner_contract() -> None:
    plan = make_plan()
    observation = Observation(timestamp=0.0)
    trajectory = Trajectory(
        joint_names=(),
        points=(TrajectoryPoint(timestamp=0.0, stage="approach"),),
        planner="replay",
    )
    result = ReplayPlanner(trajectory).plan(
        PlannerRequest(plan=plan, stage_name="approach", observation=observation)
    )
    assert result.feasible
    assert result.trajectory == trajectory


def test_command_router_separates_position_and_velocity_groups() -> None:
    router = CommandRouter(
        ControllerConfig(
            groups=(
                JointGroup("torso", ("torso_joint",), ControlMode.POSITION),
                JointGroup("base_drive", ("wheel_left",), ControlMode.VELOCITY),
            )
        )
    )
    command = router.command(
        TrajectoryPoint(
            timestamp=0.0,
            joint_positions={"torso_joint": 0.2},
            joint_velocities={"wheel_left": 0.4},
        ),
        Observation(timestamp=0.0),
        timestamp=0.0,
    )
    assert command.position_targets == {"torso_joint": 0.2}
    assert command.velocity_targets == {"wheel_left": 0.4}
    assert command.mode_by_group["torso"] is ControlMode.POSITION


def test_command_router_rejects_wrong_reference_mode() -> None:
    router = CommandRouter(
        ControllerConfig(
            groups=(JointGroup("torso", ("torso_joint",), ControlMode.POSITION),)
        )
    )
    with pytest.raises(ValueError, match="velocity reference"):
        router.command(
            TrajectoryPoint(timestamp=0.0, joint_velocities={"torso_joint": 0.1}),
            Observation(timestamp=0.0),
            timestamp=0.0,
        )


def test_failed_task_requires_evidence() -> None:
    evidence = FailureEvidence(
        category="planning",
        stage="approach",
        reason="no feasible trajectory",
    )
    result = TaskResult(
        status=TaskStatus.FAILED,
        task_name="r1pro_pickplace",
        failure=evidence,
    )
    assert result.failure == evidence

    with pytest.raises(ValueError, match="failure evidence"):
        TaskResult(status=TaskStatus.FAILED, task_name="task")


def test_provenance_is_written_as_sorted_json(tmp_path) -> None:
    output = tmp_path / "manifest.json"
    write_provenance(
        RunProvenance(
            run_id="phase-00-test",
            task="r1pro_pickplace",
            seed=7,
            project_version="0.1.0",
            planner="replay",
            controller="command-router",
        ),
        output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert list(payload) == sorted(payload)
