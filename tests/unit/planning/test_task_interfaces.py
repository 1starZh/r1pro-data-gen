from __future__ import annotations

import pytest

from r1pro_data_gen.planning.task.interfaces import TaskPlanningRequest


def test_planning_request_rejects_goal_hash_without_goal_spec() -> None:
    with pytest.raises(ValueError, match="goal_spec_hash"):
        TaskPlanningRequest(
            task_description="move an object",
            scene_facts={},
            skill_catalog=(),
            goal_spec_hash="abc",
        )


def test_planning_request_carries_frozen_goal_spec_and_hash() -> None:
    request = TaskPlanningRequest(
        task_description="move an object",
        scene_facts={},
        skill_catalog=(),
        goal_spec={"schema_version": 1, "required": []},
        goal_spec_hash="abc123",
    )

    assert request.goal_spec_hash == "abc123"
    assert request.goal_spec["schema_version"] == 1
