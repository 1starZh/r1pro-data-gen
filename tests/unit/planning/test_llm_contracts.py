from __future__ import annotations

import pytest

from r1pro_data_gen.planning.llm.contracts import (
    LLMPlanValidationError,
    parse_json_object,
    validate_envelope,
)


CATALOG = [
    {
        "name": "base_navigate_to",
        "parameters": {
            "target": {"type": "array", "required": True, "shape": [3]},
            "motion_mode": {
                "type": "string",
                "required": False,
                "enum": ["forward", "holonomic"],
            },
        },
    }
]


def envelope(skill: str = "base_navigate_to") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": "navigate",
            "stages": [
                {
                    "name": "go",
                    "goal": "reach the work area",
                    "depends_on": [],
                    "parameters": {"skill": skill, "target": [1.0, 0.2, 0.0]},
                }
            ],
            "metadata": {"source": "external_llm"},
        },
    }


def test_valid_external_envelope_is_normalized_to_plan():
    plan = validate_envelope(envelope(), skill_catalog=CATALOG)
    assert plan is not None
    assert plan.task_name == "navigate"
    assert plan.stages[0].parameters["target"] == [1.0, 0.2, 0.0]


def test_external_policy_rejects_hidden_backend_skill():
    with pytest.raises(LLMPlanValidationError, match="outside the external LLM policy"):
        validate_envelope(envelope("base_follow_path"), skill_catalog=CATALOG)


def test_external_policy_rejects_forward_dependency_and_bad_shape():
    data = envelope()
    data["plan"]["stages"][0]["depends_on"] = ["later"]
    with pytest.raises(LLMPlanValidationError, match="earlier stages"):
        validate_envelope(data, skill_catalog=CATALOG)

    data = envelope()
    data["plan"]["stages"][0]["parameters"]["target"] = [1.0, 0.2]
    with pytest.raises(LLMPlanValidationError, match="shape"):
        validate_envelope(data, skill_catalog=CATALOG)


def test_semantic_navigation_target_ref_can_replace_literal_target():
    catalog = [
        {
            "name": "base_navigate_to",
            "parameters": {
                "target": {"type": "array", "required": False, "shape": [3]},
                "target_ref": {"type": "string", "required": False},
                "purpose": {
                    "type": "string",
                    "required": False,
                    "enum": ["navigation", "pregrasp", "dropoff", "staging", "observe"],
                },
            },
        }
    ]
    data = envelope()
    parameters = data["plan"]["stages"][0]["parameters"]
    parameters.pop("target")
    parameters["target_ref"] = "scene://pick_cylinder"
    parameters["purpose"] = "pregrasp"

    plan = validate_envelope(
        data,
        skill_catalog=catalog,
        scene_object_names=("pick_cylinder",),
    )

    assert plan is not None
    assert plan.stages[0].parameters["target_ref"] == "scene://pick_cylinder"


def test_semantic_navigation_target_ref_must_name_a_scene_object():
    catalog = [
        {
            "name": "base_navigate_to",
            "parameters": {
                "target_ref": {"type": "string", "required": False},
            },
        }
    ]
    data = envelope()
    parameters = data["plan"]["stages"][0]["parameters"]
    parameters.pop("target")
    parameters["target_ref"] = "scene://missing"

    with pytest.raises(LLMPlanValidationError, match="unknown scene object") as exc_info:
        validate_envelope(data, skill_catalog=catalog, scene_object_names=("pick_cylinder",))
    assert "pick_cylinder" in str(exc_info.value)


def test_unknown_skill_parameter_error_lists_allowed_parameters():
    data = envelope()
    data["plan"]["stages"][0]["parameters"]["typo"] = 1

    with pytest.raises(LLMPlanValidationError, match="unknown parameters") as exc_info:
        validate_envelope(data, skill_catalog=CATALOG)
    assert "allowed parameters" in str(exc_info.value)
    assert "motion_mode" in str(exc_info.value)


def test_unsupported_response_is_explicit():
    result = validate_envelope(
        {
            "schema_version": "1.0",
            "status": "unsupported",
            "reason": "requires closed-loop grasp feedback",
            "plan": None,
        },
        skill_catalog=CATALOG,
    )
    assert result is None


def test_json_parser_strips_markdown_fence_and_rejects_nonfinite():
    # A Markdown-wrapped payload is recoverable and must not be rejected.
    assert parse_json_object('```json\n{"status": "planned", "plan": {"task_name": "t", "stages": []}}\n```') == {
        "status": "planned",
        "plan": {"task_name": "t", "stages": []},
    }
    with pytest.raises(LLMPlanValidationError):
        parse_json_object('{"value": NaN}')


def test_json_parser_accepts_only_object():
    assert parse_json_object('{"status":"unsupported"}') == {"status": "unsupported"}
    with pytest.raises(LLMPlanValidationError):
        parse_json_object("[]")


def test_json_parser_ignores_trailing_content_after_first_object():
    # deepseek-chat has appended a second object / stray text after the plan;
    # the first balanced JSON object must still be parsed.
    payload = '{"status": "planned", "plan": {"task_name": "t", "stages": [{"name": "go", "goal": "g", "depends_on": [], "parameters": {"skill": "base_navigate_to", "target": [1.0, 0.2, 0.0]}}], "metadata": {}}}'
    assert parse_json_object(f"{payload}{payload}")["status"] == "planned"
    assert parse_json_object(f"{payload}   \n```\ntrailing note```")["status"] == "planned"
    assert parse_json_object(f"  {payload}\nrandom trailing words")["plan"]["task_name"] == "t"
    # A brace inside a string literal must not terminate the scan early.
    nested = '{"status": "planned", "plan": {"note": "a { brace inside", "task_name": "t", "stages": []}}'
    assert parse_json_object(nested)["plan"]["task_name"] == "t"


def test_direct_plan_envelope_variant_is_strictly_normalized():
    plan = validate_envelope(
        {
            "envelope_schema_version": "1.0",
            "task_name": "navigate",
            "stages": envelope()["plan"]["stages"],
            "metadata": {},
        },
        skill_catalog=CATALOG,
    )
    assert plan is not None
    assert plan.task_name == "navigate"


def test_direct_plan_envelope_rejects_extra_fields():
    data = {
        "envelope_schema_version": "1.0",
        "task_name": "navigate",
        "stages": envelope()["plan"]["stages"],
        "unsafe": "must reject",
    }
    with pytest.raises(LLMPlanValidationError, match="unknown fields"):
        validate_envelope(data, skill_catalog=CATALOG)


def test_external_policy_rejects_low_level_grasp_motion():
    """The LLM selects semantic grasp capabilities, never Cartesian joints."""
    data = envelope("arm_move_to")
    data["plan"]["stages"][0]["parameters"] = {
        "skill": "arm_move_to",
        "target_pos": [1.0, 0.2, 0.8],
        "target_frame": "grasp_center",
    }
    with pytest.raises(LLMPlanValidationError, match="outside the external LLM policy"):
        validate_envelope(data, skill_catalog=[{"name": "arm_move_to", "parameters": {}}])


def test_public_grasp_contract_accepts_auto_side():
    catalog = [
        {
            "name": "grasp_object",
            "parameters": {
                "object_name": {"type": "string", "required": True},
                "side": {"type": "string", "required": False, "enum": ["auto", "left", "right"]},
            },
        }
    ]
    data = envelope("grasp_object")
    data["plan"]["stages"][0]["parameters"] = {
        "skill": "grasp_object",
        "object_name": "pick_cylinder",
        "side": "auto",
    }
    plan = validate_envelope(data, skill_catalog=catalog, scene_object_names=("pick_cylinder",))
    assert plan is not None
    assert plan.stages[0].parameters["side"] == "auto"


def test_external_policy_rejects_standalone_alignment_micro_skill():
    catalog = [{"name": "arm_align_gripper", "parameters": {"object_name": {"type": "string"}}}]
    data = envelope("arm_align_gripper")
    data["plan"]["stages"][0]["parameters"] = {
        "skill": "arm_align_gripper",
        "object_name": "pick_cylinder",
    }
    with pytest.raises(LLMPlanValidationError, match="outside the external LLM policy"):
        validate_envelope(data, skill_catalog=catalog, scene_object_names=("pick_cylinder",))


def _held_object_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": "grasp_object",
            "parameters": {
                "object_name": {"type": "string", "required": True},
                "side": {"type": "string", "required": False},
            },
        },
        {
            "name": "arm_carry_object_to",
            "parameters": {
                "object_name": {"type": "string", "required": True},
                "target_region_name": {"type": "string", "required": True},
                "support_surface_name": {"type": "string", "required": True},
                "side": {"type": "string", "required": False},
            },
        },
    ]


def _held_plan(stage_parameters: list[dict[str, object]]) -> dict[str, object]:
    stages = []
    for index, parameters in enumerate(stage_parameters):
        stages.append(
            {
                "name": f"stage_{index}",
                "goal": "perform the next manipulation action",
                "depends_on": [f"stage_{index - 1}"] if index else [],
                "parameters": parameters,
            }
        )
    return {
        "schema_version": "1.0",
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": "generic_manipulation",
            "stages": stages,
            "metadata": {"source": "external_llm"},
        },
    }


def test_external_policy_rejects_low_level_held_object_sequence():
    data = _held_plan(
        [
            {"skill": "joint_mask_lock"},
            {"skill": "gripper_grasp", "object_name": "pick", "side": "left"},
            {
                "skill": "arm_move_to",
                "target_pos": {
                    "ref": "scene.object.pick.position",
                    "value_type": "array",
                    "shape": [3],
                    "frame": "base",
                },
            },
        ]
    )
    with pytest.raises(LLMPlanValidationError, match="outside the external LLM policy"):
        validate_envelope(
            data,
            skill_catalog=_held_object_catalog(),
            scene_object_names=["pick", "target", "table"],
        )


def test_carry_without_prior_lock_or_grasp_still_validates():
    """Lock/grasp-before-carry is prompt doctrine; the runtime enforces it.

    The generic carry skill verifies live grasp state at execution time and
    fails safely when nothing is held, so the validator no longer duplicates
    that orchestration rule statically.
    """
    data = _held_plan(
        [
            {
                "skill": "arm_carry_object_to",
                "object_name": "pick",
                "target_region_name": "target",
                "support_surface_name": "table",
                "side": "left",
            }
        ]
    )
    plan = validate_envelope(
        data,
        skill_catalog=_held_object_catalog(),
        scene_object_names=["pick", "target", "table"],
    )
    assert plan is not None
    assert plan.stages[-1].parameters["skill"] == "arm_carry_object_to"


def test_carry_with_full_doctrine_prefix_still_validates():
    """The doctrinally correct sequence remains a valid plan (no regression)."""
    data = _held_plan(
        [
            {"skill": "grasp_object", "object_name": "pick", "side": "auto"},
            {
                "skill": "arm_carry_object_to",
                "object_name": "pick",
                "target_region_name": "target",
                "support_surface_name": "table",
                "side": "auto",
            },
        ]
    )
    plan = validate_envelope(
        data,
        skill_catalog=_held_object_catalog(),
        scene_object_names=["pick", "target", "table"],
    )
    assert plan is not None
    assert plan.stages[-1].parameters["skill"] == "arm_carry_object_to"


def test_external_policy_rejects_world_frame_arm_backend():
    """Cartesian frame details stay inside the trusted semantic backend."""
    catalog = [
        {
            "name": "arm_move_to",
            "parameters": {
                "target_pos": {"type": "array", "required": True, "shape": [3]},
                "target_frame": {"type": "string", "required": False},
            },
        }
    ]
    data = envelope("arm_move_to")
    data["plan"]["stages"][0]["parameters"] = {
        "skill": "arm_move_to",
        "target_pos": {
            "ref": "scene.object.pick.position",
            "value_type": "array",
            "shape": [3],
            "frame": "world",
        },
    }
    with pytest.raises(LLMPlanValidationError, match="outside the external LLM policy"):
        validate_envelope(data, skill_catalog=catalog, scene_object_names=("pick",))

    data["plan"]["stages"][0]["parameters"]["target_pos"]["frame"] = "base"
    with pytest.raises(LLMPlanValidationError, match="outside the external LLM policy"):
        validate_envelope(data, skill_catalog=catalog, scene_object_names=("pick",))
