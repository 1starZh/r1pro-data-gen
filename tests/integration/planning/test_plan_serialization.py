"""Plan serialization tests: Plan is data, round-trips losslessly (no Isaac Sim)."""

from __future__ import annotations

import pytest

from r1pro_data_gen.data.plan_io import (
    load_plan,
    plan_from_dict,
    plan_from_json,
    plan_to_dict,
    plan_to_json,
    save_plan,
)
from r1pro_data_gen.domain import Plan, PlanStage


def make_plan() -> Plan:
    return Plan(
        task_name="r1pro_pickplace",
        metadata={"scene": "pickplace", "author": "test"},
        stages=(
            PlanStage(
                name="navigate_to_work",
                goal="drive to the work position",
                parameters={"skill": "base_move_to", "target": [0.05, 0.15, 0.0]},
            ),
            PlanStage(
                name="open_gripper",
                goal="open the gripper",
                parameters={"skill": "gripper_set", "open_value": 0.05},
            ),
        ),
    )


def test_plan_dict_round_trip() -> None:
    plan = make_plan()
    rebuilt = plan_from_dict(plan_to_dict(plan))
    assert rebuilt == plan
    assert rebuilt.stage_names == ("navigate_to_work", "open_gripper")
    assert rebuilt.stages[0].parameters["target"] == [0.05, 0.15, 0.0]


def test_plan_json_round_trip() -> None:
    plan = make_plan()
    assert plan_from_json(plan_to_json(plan)) == plan


def test_save_and_load_plan(tmp_path) -> None:
    path = tmp_path / "plan.json"
    save_plan(make_plan(), path)
    assert load_plan(path) == make_plan()


def test_tuple_parameters_become_json_lists() -> None:
    plan = Plan(
        task_name="t",
        stages=(
            PlanStage(
                name="go",
                goal="g",
                parameters={"skill": "x", "target": (0.1, 0.2, 0.3)},
            ),
        ),
    )
    data = plan_to_dict(plan)
    assert data["stages"][0]["parameters"]["target"] == [0.1, 0.2, 0.3]
    rebuilt = plan_from_dict(data)
    assert rebuilt.stages[0].parameters["target"] == [0.1, 0.2, 0.3]


def test_non_jsonable_parameter_rejected() -> None:
    plan = Plan(
        task_name="t",
        stages=(
            PlanStage(
                name="bad",
                goal="g",
                parameters={"skill": "x", "weird": {1, 2, 3}},
            ),
        ),
    )
    with pytest.raises(ValueError, match="not JSON-serializable"):
        plan_from_dict(plan_to_dict(plan))
