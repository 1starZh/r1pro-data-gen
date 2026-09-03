from __future__ import annotations

from r1pro_data_gen.domain import GoalPredicate, GoalSpec
from r1pro_data_gen.agent.skeleton import build_semantic_plan_skeleton


def test_skeleton_is_semantic_and_keeps_multiple_goal_families_open():
    spec = GoalSpec(
        schema_version=1,
        bindings={"item": "scene://box", "goal": "scene://target"},
        required=(
            GoalPredicate("inside_region", {"subject": "item", "reference": "goal", "region": {"shape": "cuboid", "center": [0, 0, 0], "size": [1, 1, 1]}}),
            GoalPredicate("settled", {"subject": "item"}),
        ),
    )
    skeleton = build_semantic_plan_skeleton(
        spec,
        skill_catalogue=[{"name": "push_object_to"}, {"name": "release_object"}],
    )

    assert skeleton["execution_policy"]["one_skill_per_step"] is True
    assert skeleton["steps"][0]["entities"]["subject"] == "box"
    assert skeleton["steps"][0]["candidate_skills"] == ["release_object", "push_object_to"]
    assert skeleton["steps"][1]["candidate_skills"] == []


def test_skeleton_recommends_grasp_then_carry_from_public_catalogue() -> None:
    spec = GoalSpec(
        schema_version=1,
        bindings={"item": "scene://pick_cylinder", "goal": "scene://place_target"},
        required=(
            GoalPredicate("attached", {"subject": "item"}),
            GoalPredicate(
                "inside_region",
                {
                    "subject": "item",
                    "reference": "goal",
                    "region": {"shape": "cuboid", "center": [0, 0, 0], "size": [1, 1, 1]},
                },
            ),
        ),
    )
    catalogue = [
        {"name": "base_navigate_to"},
        {"name": "prepare_workspace"},
        {"name": "grasp_object"},
        {"name": "arm_carry_object_to"},
        {"name": "release_object"},
        {"name": "push_object_to"},
        {"name": "support_aware_grasp_object"},
        {"name": "whole_body_transfer_object_between_supports"},
    ]
    skeleton = build_semantic_plan_skeleton(spec, skill_catalogue=catalogue)
    assert skeleton["steps"][0]["candidate_skills"] == ["grasp_object", "prepare_workspace"]
    assert skeleton["steps"][1]["candidate_skills"][0] == "arm_carry_object_to"
    assert "whole_body_transfer_object_between_supports" not in skeleton["steps"][0]["candidate_skills"]
    assert "support_aware_grasp_object" not in skeleton["steps"][0]["candidate_skills"]
