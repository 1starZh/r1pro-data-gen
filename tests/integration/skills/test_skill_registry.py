"""Skill registry tests: declaration validation, lookup, plan-parameter checks."""

from __future__ import annotations

import pytest

from r1pro_data_gen.skills import ParamSpec, SkillResult, SkillRegistry
from r1pro_data_gen.skills.core.base import Skill


class _FakeSkill:
    name = "fake_move"
    description = "A fake skill for tests."
    parameters: dict[str, ParamSpec] = {
        "target": ParamSpec("array", "Target", required=True),
        "speed": ParamSpec("number", "Speed", default=1.0),
    }

    def execute(self, adapter, scene=None, **params):
        return SkillResult(success=True, skill=self.name, metrics={"target": len(params.get("target", []))})


def test_registry_lists_and_describes_skills() -> None:
    reg = SkillRegistry([_FakeSkill()])
    assert reg.names == ("fake_move",)
    desc = reg.descriptions()[0]
    assert desc["name"] == "fake_move"
    assert desc["parameters"]["target"]["required"] is True
    assert desc["parameters"]["speed"]["default"] == 1.0


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SkillRegistry([_FakeSkill(), _FakeSkill()])


def test_registry_validate_plan_params() -> None:
    reg = SkillRegistry([_FakeSkill()])
    reg.validate_plan_params("fake_move", {"target": [0, 0, 0]})  # ok
    with pytest.raises(ValueError, match="missing required"):
        reg.validate_plan_params("fake_move", {"speed": 1.0})
    with pytest.raises(KeyError, match="unknown"):
        reg.validate_plan_params("nope", {})


def test_registry_execute_returns_skill_result() -> None:
    reg = SkillRegistry([_FakeSkill()])
    result = reg.execute("fake_move", adapter=None, target=[1.0, 2.0])
    assert result.success and result.skill == "fake_move"


def test_registry_rejects_bad_shape_and_unknown_parameters() -> None:
    reg = SkillRegistry([
        type("StrictSkill", (), {
            "name": "strict_move",
            "description": "strict",
            "parameters": {"target": ParamSpec("array", "xyz", required=True, shape=(3,))},
            "execute": lambda self, adapter, scene=None, **params: SkillResult(True, self.name),
        })()
    ])
    with pytest.raises(ValueError, match="shape"):
        reg.validate_plan_params("strict_move", {"target": [0.0, 1.0]})
    with pytest.raises(ValueError, match="unknown parameters"):
        reg.validate_plan_params("strict_move", {"target": [0.0, 1.0, 2.0], "oops": 1})


def test_build_default_registry_contains_the_generic_skill_library() -> None:
    from r1pro_data_gen.agent.contracts import AGENT_PUBLIC_SKILLS
    from r1pro_data_gen.skills import build_default_registry

    reg = build_default_registry(kin=None, vel_limits=[0.1] * 7)
    assert len(reg.names) >= 30
    expected = {
        "arm_joint_to", "arm_move_directional", "arm_move_through", "arm_move_to", "arm_carry_object_to",
        "arm_align_gripper",        "arm_rotate_ee", "arm_trajectory_follow", "base_follow_path", "base_lock_wheels",
        "base_move_to", "base_navigate_to", "base_rotate_to", "base_unlock_wheels",
        "base_velocity_set", "gripper_grasp", "gripper_set", "grasp_object", "release_object",
        "transfer_object_between_supports",
        "whole_body_transfer_object_between_supports",
        "whole_body_pregrasp_transition",
        "whole_body_hold_transition",
        "support_aware_grasp_object",
        "push_object_to",
        "prepare_workspace",
        "query_arm_path",
        "query_base_path", "query_contacts", "query_ee_pose", "query_ik_solution",
        "query_joint_pos", "query_object_pose", "torso_move_to",
        "joint_mask_lock", "joint_mask_unlock",
    }
    assert set(reg.names) == expected
    assert "arm_move_to" in reg
    llm_names = {item["name"] for item in reg.llm_descriptions()}
    # Low-level arm/base/joint primitives remain trusted internal backends;
    # the external LLM receives semantic task capabilities only.
    assert llm_names == set(AGENT_PUBLIC_SKILLS)
    assert llm_names == {
        "base_navigate_to",
        "prepare_workspace",
        "grasp_object",
        "arm_carry_object_to",
        "release_object",
        "push_object_to",
    }
    assert "arm_move_to" not in llm_names
    assert "support_aware_grasp_object" not in llm_names
    assert "whole_body_transfer_object_between_supports" not in llm_names
    assert "query_object_pose" not in llm_names
    assert "torso_move_to" not in llm_names
    assert "arm_move_through" not in llm_names
    assert "arm_move_directional" not in llm_names
    assert "run_registered_task" not in reg
    assert "arm_trajectory_follow" not in llm_names
    assert "base_velocity_set" not in llm_names
    agent_names = {item["name"] for item in reg.agent_descriptions()}
    assert agent_names == set(AGENT_PUBLIC_SKILLS)
    assert "whole_body_transfer_object_between_supports" not in agent_names
    assert "arm_move_through" not in agent_names
    assert "arm_move_directional" not in agent_names
    assert "whole_body_hold_transition" not in agent_names
    assert "arm_move_to" not in agent_names
    assert "arm_align_gripper" not in agent_names
    assert "gripper_grasp" not in agent_names
    assert "query_object_pose" not in agent_names
    assert "prepare_workspace" in agent_names
    for name in (
        "arm_joint_to", "arm_trajectory_follow", "arm_move_to", "arm_move_through",
        "arm_move_directional", "arm_rotate_ee", "gripper_set",
        "gripper_grasp", "query_contacts", "query_ee_pose",
        "query_ik_solution", "query_arm_path",
    ):
        assert reg[name].parameters["side"].enum == ("left", "right")
    for name in (
        "arm_carry_object_to", "grasp_object", "release_object",
        "support_aware_grasp_object", "transfer_object_between_supports",
        "whole_body_transfer_object_between_supports",
    ):
        assert reg[name].parameters["side"].enum == ("auto", "left", "right")
