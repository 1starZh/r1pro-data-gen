from __future__ import annotations

import pytest

from r1pro_data_gen.agent.contracts import (
    AGENT_SCHEMA_VERSION,
    AgentActionValidationError,
    validate_action_envelope,
)


CATALOG = [
    {
        "name": "grasp_object",
        "parameters": {
            "object_name": {"type": "string", "required": True},
            "side": {"type": "string", "required": False, "enum": ["left", "right"]},
        },
    },
    {
        "name": "base_navigate_to",
        "parameters": {
            "target_ref": {"type": "string", "required": False},
            "purpose": {"type": "string", "required": False},
            "approach_side": {"type": "string", "required": False},
        },
    },
    {
        "name": "push_object_to",
        "parameters": {
            "object_name": {"type": "string", "required": True},
            "target_ref": {"type": "string", "required": False},
            "target_region_name": {"type": "string", "required": False},
            "target_pose": {"type": "array", "required": False},
        },
    },
    {
        "name": "prepare_workspace",
        "parameters": {
            "profile": {
                "type": "string",
                "required": True,
                "enum": ["tabletop", "floor", "carry", "travel"],
            },
        },
    },
]


def _table_scene():
    from types import SimpleNamespace

    return SimpleNamespace(
        objects=(
            SimpleNamespace(
                name="work_table",
                capabilities=("supports_objects",),
                size=(1.2, 0.8, 0.75),
                pos=(1.0, 0.0, 0.375),
                quat=(1.0, 0.0, 0.0, 0.0),
                surfaces=("top",),
            ),
            SimpleNamespace(
                name="other_table",
                capabilities=("supports_objects",),
                size=(1.2, 0.8, 0.75),
                pos=(3.0, 0.0, 0.375),
                quat=(1.0, 0.0, 0.0, 0.0),
                surfaces=("top",),
            ),
            SimpleNamespace(
                name="pick_cylinder",
                capabilities=("movable", "graspable"),
                radius=0.03,
                height=0.10,
                pos=(1.0, 0.0, 0.80),
                quat=(1.0, 0.0, 0.0, 0.0),
                surfaces=(),
            ),
        )
    )


def _act(skill: str, **parameters) -> dict[str, object]:
    return {
        "schema_version": AGENT_SCHEMA_VERSION,
        "status": "act",
        "reason": "",
        "action": {"skill": skill, "parameters": parameters},
    }


def test_valid_grasp_action_is_accepted() -> None:
    action = validate_action_envelope(
        _act("grasp_object", object_name="cylinder", side="left"),
        skill_catalog=CATALOG,
        scene_object_names=("cylinder", "table"),
    )
    assert action is not None
    assert action.skill == "grasp_object"
    assert action.parameters["object_name"] == "cylinder"


def test_agent_policy_rejects_micro_skill() -> None:
    with pytest.raises(AgentActionValidationError, match="outside the agent policy"):
        validate_action_envelope(
            _act("arm_align_gripper", object_name="cylinder"),
            skill_catalog=CATALOG,
            scene_object_names=("cylinder",),
        )
    with pytest.raises(AgentActionValidationError, match="outside the agent policy"):
        validate_action_envelope(
            _act("arm_move_through", waypoints=[]),
            skill_catalog=CATALOG,
            scene_object_names=("cylinder",),
        )


def test_navigation_requires_one_target_form() -> None:
    with pytest.raises(AgentActionValidationError, match="exactly one"):
        validate_action_envelope(
            _act("base_navigate_to"),
            skill_catalog=CATALOG,
            scene_object_names=("table",),
        )
    action = validate_action_envelope(
        _act("base_navigate_to", target=[1.0, 2.0, 0.0]),
        skill_catalog=CATALOG,
        scene_object_names=("table",),
    )
    assert action is not None
    assert action.parameters["target"] == [1.0, 2.0, 0.0]
    action = validate_action_envelope(
        _act(
            "base_navigate_to",
            target_ref="scene://table",
            purpose="pregrasp",
            approach_side="west",
        ),
        skill_catalog=CATALOG,
        scene_object_names=("table",),
    )
    assert action is not None
    assert action.parameters["target_ref"] == "scene://table"


def test_nested_region_name_is_rejected() -> None:
    with pytest.raises(AgentActionValidationError, match="top-level scene object"):
        validate_action_envelope(
            _act("grasp_object", object_name="place_region"),
            skill_catalog=CATALOG,
            scene_object_names=("cylinder", "place_target"),
        )


def test_push_accepts_one_semantic_region_target() -> None:
    action = validate_action_envelope(
        _act(
            "push_object_to",
            object_name="box",
            target_region_name="goal/region",
        ),
        skill_catalog=CATALOG,
        scene_object_names=("box", "goal"),
    )
    assert action is not None
    assert action.parameters["target_region_name"] == "goal/region"


def test_push_rejects_ambiguous_target_forms() -> None:
    with pytest.raises(AgentActionValidationError, match="exactly one"):
        validate_action_envelope(
            _act(
                "push_object_to",
                object_name="box",
                target_ref="scene://goal",
                target_pose=[1.0, 0.0, 0.0],
            ),
            skill_catalog=CATALOG,
            scene_object_names=("box", "goal"),
        )


def test_policy_rejects_query_and_composite_skills() -> None:
    for skill in (
        "query_object_pose",
        "torso_move_to",
        "support_aware_grasp_object",
        "transfer_object_between_supports",
        "whole_body_transfer_object_between_supports",
        "base_move_to",
    ):
        with pytest.raises(AgentActionValidationError, match="outside the agent policy"):
            validate_action_envelope(
                _act(skill, object_name="pick_cylinder"),
                skill_catalog=CATALOG,
                scene_object_names=("pick_cylinder", "work_table"),
            )


def test_prepare_workspace_is_accepted() -> None:
    action = validate_action_envelope(
        _act("prepare_workspace", profile="tabletop"),
        skill_catalog=CATALOG,
        scene_object_names=("pick_cylinder", "work_table"),
    )
    assert action is not None
    assert action.skill == "prepare_workspace"
    assert action.parameters["profile"] == "tabletop"


def test_attached_same_support_navigation_is_rejected() -> None:
    scene = _table_scene()
    with pytest.raises(AgentActionValidationError, match="same_support_navigation_forbidden"):
        validate_action_envelope(
            _act(
                "base_navigate_to",
                target_ref="scene://work_table",
                purpose="dropoff",
            ),
            skill_catalog=CATALOG,
            scene_object_names=("pick_cylinder", "work_table", "other_table"),
            scene=scene,
            attachments={"pick_cylinder": "left_gripper"},
            object_positions={"pick_cylinder": [1.0, 0.0, 0.95], "work_table": [1.0, 0.0, 0.375]},
        )


def test_attached_cross_support_dropoff_is_allowed() -> None:
    scene = _table_scene()
    action = validate_action_envelope(
        _act(
            "base_navigate_to",
            target_ref="scene://other_table",
            purpose="dropoff",
        ),
        skill_catalog=CATALOG,
        scene_object_names=("pick_cylinder", "work_table", "other_table"),
        scene=scene,
        attachments={"pick_cylinder": "left_gripper"},
        object_positions={
            "pick_cylinder": [1.0, 0.0, 0.95],
            "work_table": [1.0, 0.0, 0.375],
            "other_table": [3.0, 0.0, 0.375],
        },
    )
    assert action is not None
    assert action.parameters["target_ref"] == "scene://other_table"


def test_unsupported_requires_reason_and_no_action() -> None:
    with pytest.raises(AgentActionValidationError, match="reason"):
        validate_action_envelope(
            {
                "schema_version": AGENT_SCHEMA_VERSION,
                "status": "unsupported",
                "reason": "",
                "action": None,
            },
            skill_catalog=CATALOG,
        )
    assert (
        validate_action_envelope(
            {
                "schema_version": AGENT_SCHEMA_VERSION,
                "status": "unsupported",
                "reason": "cannot represent the task",
                "action": None,
            },
            skill_catalog=CATALOG,
        )
        is None
    )
