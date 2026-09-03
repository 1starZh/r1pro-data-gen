from __future__ import annotations

import json

from r1pro_data_gen.domain import (
    ObjectModel,
    ObjectType,
    RegionModel,
    RobotModel,
    SceneModel,
    WorldModel,
)
from r1pro_data_gen.planning.goals.planner import GoalPlanner, GoalPlanningRequest
from r1pro_data_gen.planning.llm.providers.protocol import ProviderResponse


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, response: dict[str, object]):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> ProviderResponse:
        self.calls.append((system, user))
        return ProviderResponse(
            text=json.dumps(self.response), provider=self.name, model=self.model
        )


def _scene() -> SceneModel:
    return SceneModel(
        name="scene",
        world=WorldModel(),
        robot=RobotModel(asset="robot.usd"),
        objects=(
            ObjectModel(
                name="item",
                type=ObjectType.CUBOID,
                pos=(0.0, 0.0, 0.1),
                size=(0.1, 0.1, 0.1),
                regions=(
                    RegionModel(
                        name="stable_region",
                        shape=ObjectType.CUBOID,
                        center=(0.0, 0.0, 0.05),
                        size=(0.1, 0.1, 0.1),
                    ),
                ),
            ),
        ),
    )


def test_goal_planner_returns_grounded_goal_without_action_fields() -> None:
    provider = FakeProvider(
        {
            "schema_version": 1,
            "bindings": {"subject": "scene://item"},
            "required": [
                {"predicate": "settled", "arguments": {"subject": "subject"}}
            ],
            "invariants": [],
        }
    )
    result = GoalPlanner(provider).plan(
        GoalPlanningRequest(
            task_description="Make the item stable.",
            scene_facts={"objects": [{"name": "item"}]},
            scene=_scene(),
        )
    )

    assert result.status == "planned"
    assert result.goal_spec is not None
    assert result.goal_spec.bindings["subject"] == "scene://item"
    assert result.goal_spec_hash
    assert "skill" not in result.raw_response


def test_goal_planner_prompt_freezes_predicate_json_shape() -> None:
    provider = FakeProvider(
        {
            "schema_version": 1,
            "bindings": {"subject": "scene://item"},
            "required": [
                {"predicate": "settled", "arguments": {"subject": "subject"}}
            ],
            "invariants": [],
        }
    )
    GoalPlanner(provider).plan(
        GoalPlanningRequest(
            task_description="Make the item stable.",
            scene_facts={"objects": [{"name": "item"}]},
            scene=_scene(),
        )
    )

    system, user = provider.calls[0]
    assert 'exactly the keys "predicate" and "arguments"' in system
    assert 'never use the key "args"' in system
    assert "on_support surface must be an object" in system
    assert "Never substitute object for subject" in system
    assert "never use the slash-separated string robot/base/mobile_base" in system
    assert "Never use robot, base, or mobile_base as an effector" in system
    assert "Every explicit completion clause" in system
    assert "pick_cylinder" not in system
    assert "pick_cylinder" not in user
    user_payload = json.loads(user)
    assert user_payload["canonical_regions"] == {
        "item": [
            {
                "name": "stable_region",
                "shape": "cuboid",
                "center": [0.0, 0.0, 0.05],
                "size": [0.1, 0.1, 0.1],
            }
        ]
    }
    rules = user_payload["output_rules"]
    assert 'Each predicate object must contain exactly the keys "predicate" and "arguments"; never use "args".' in rules
    assert "The arguments value must be a JSON object, never an array; schema_version must be the integer 1." in rules
    assert any(
        rule.startswith("on_support requires subject, support, surface, and subject_half_height_m")
        for rule in rules
    )
    assert any(rule.startswith("For inside_region, copy the exact region geometry") for rule in rules)
    assert "never use object instead of subject" in rules
    assert any("robot/base/mobile_base are invalid there" in rule for rule in rules)
    assert any("never use the slash-separated value robot/base/mobile_base" in rule for rule in rules)


def test_goal_planner_rejects_a_valid_but_incomplete_transfer_contract() -> None:
    """A terminal region predicate must not silently replace the grasp/release chain."""
    provider = FakeProvider(
        {
            "schema_version": 1,
            "bindings": {"subject": "scene://item"},
            "required": [
                {
                    "predicate": "inside_region",
                    "arguments": {
                        "subject": "subject",
                        "reference": "subject",
                        "region": {
                            "shape": "cuboid",
                            "center": [0.0, 0.0, 0.05],
                            "size": [0.1, 0.1, 0.1],
                        },
                    },
                }
            ],
            "invariants": [],
        }
    )

    result = GoalPlanner(provider, max_attempts=1).plan(
        GoalPlanningRequest(
            task_description=(
                "Grasp and carry the item into its declared region, release it, "
                "and leave it settled there."
            ),
            scene_facts={"objects": [{"name": "item"}]},
            scene=_scene(),
        )
    )

    assert result.status == "failed"
    assert "GoalSpec lacks released" in result.reason
    assert "GoalSpec lacks attached" in result.reason
    assert "GoalSpec lacks settled" in result.reason


def test_goal_planner_rejects_unknown_predicate_and_bounded_retry() -> None:
    provider = FakeProvider(
        {
            "schema_version": 1,
            "bindings": {"subject": "scene://item"},
            "required": [
                {"predicate": "task_specific_success", "arguments": {"subject": "subject"}}
            ],
            "invariants": [],
        }
    )
    result = GoalPlanner(provider, max_attempts=1).plan(
        GoalPlanningRequest(
            task_description="Make the item stable.",
            scene_facts={"objects": [{"name": "item"}]},
            scene=_scene(),
        )
    )

    assert result.status == "failed"
    assert "unknown predicate" in result.reason


def test_goal_planner_repair_prompt_exposes_exact_declared_region_geometry() -> None:
    provider = FakeProvider(
        {
            "schema_version": 1,
            "bindings": {"subject": "scene://item"},
            "required": [
                {
                    "predicate": "inside_region",
                    "arguments": {
                        "subject": "subject",
                        "reference": "subject",
                        "region": {
                            "shape": "cuboid",
                            "center": [0.1, 0.0, 0.05],
                            "size": [0.1, 0.1, 0.1],
                        },
                    },
                }
            ],
            "invariants": [],
        }
    )
    result = GoalPlanner(provider, max_attempts=2).plan(
        GoalPlanningRequest(
            task_description="Keep the item inside its declared region.",
            scene_facts={"objects": [{"name": "item"}]},
            scene=_scene(),
        )
    )

    assert result.status == "failed"
    assert len(provider.calls) == 2
    repair_payload = json.loads(provider.calls[1][1])
    assert repair_payload["canonical_regions"] == {
        "item": [
            {
                "name": "stable_region",
                "shape": "cuboid",
                "center": [0.0, 0.0, 0.05],
                "size": [0.1, 0.1, 0.1],
            }
        ]
    }
    assert "copy the exact shape/center/size" in repair_payload["schema_correction"]
