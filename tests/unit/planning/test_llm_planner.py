from __future__ import annotations

import json

import pytest

from r1pro_data_gen.planning.task.interfaces import TaskPlanningRequest
from r1pro_data_gen.domain import Plan, PlanStage
from r1pro_data_gen.planning.llm.contracts import LLMPlanValidationError
from r1pro_data_gen.planning.task.planner import (
    LLMTaskPlanner,
    _validate_gripper_alignment_precondition,
    _validate_runtime_repair,
    _repair_prompt,
    _system_prompt,
    _user_prompt,
)
from r1pro_data_gen.planning.llm.providers.protocol import ProviderResponse


CATALOG = (
    {
        "name": "base_navigate_to",
        "parameters": {
            "target": {"type": "array", "required": True, "shape": [3]},
        },
    },
)


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, system, user):
        self.calls.append((system, user))
        return ProviderResponse(
            text=self.responses.pop(0), provider=self.name, model=self.model
        )


def request():
    return TaskPlanningRequest(
        task_description="navigate to the work area",
        scene_facts={"objects": [{"name": "crate"}]},
        skill_catalog=CATALOG,
    )


def request():
    return TaskPlanningRequest(
        task_description="navigate to the work area",
        scene_facts={"objects": [{"name": "crate"}]},
        skill_catalog=CATALOG,
    )


def generic_request(*, goal_hash: str = "a" * 64, contract_hash: str | None = None):
    return TaskPlanningRequest(
        task_description="navigate to the work area",
        scene_facts={"objects": [{"name": "crate"}]},
        skill_catalog=CATALOG,
        goal_spec={
            "schema_version": 1,
            "bindings": {"subject": "scene://crate"},
            "required": [],
            "invariants": [],
        },
        goal_spec_hash=goal_hash,
        goal_contract_hash=contract_hash,
    )


def test_user_prompt_includes_frozen_goal_spec_and_hash():
    prompt = _user_prompt(generic_request())
    payload = json.loads(prompt)
    assert payload["goal_spec_hash"] == "a" * 64
    assert payload["goal_spec"]["bindings"]["subject"] == "scene://crate"


def test_repair_prompt_spells_out_base_reachability_action():
    prompt = _repair_prompt(
        generic_request(),
        "runtime repair after position_reachable_without_orientation=false must "
        "change the base approach",
    )
    assert "change the existing base_navigate_to approach_side" in prompt
    assert "Changing only arm side or IK budget" in prompt
    payload = json.loads(prompt)
    assert payload["runtime_repair_contract"]["kind"] == "base_approach_change"
    assert "arm side" in payload["runtime_repair_contract"]["forbidden_as_sole_change"]
    assert "standoff offset" not in payload["runtime_repair_contract"]["forbidden_as_sole_change"]


def test_repair_prompt_copies_fact_supported_navigation_candidates():
    request_with_navigation = TaskPlanningRequest(
        task_description="reach the object",
        scene_facts={
            "navigation": {
                "approach_candidates": [
                    {
                        "side": "west",
                        "pose": [1.0, 2.0, 0.0],
                        "obstacle_name": "support",
                        "ik_reachability": [{"reachable": True, "distance_m": 0.4}],
                    },
                    {
                        "side": "east",
                        "pose": [2.0, 2.0, 3.14],
                        "obstacle_name": "support",
                        "ik_reachability": [{"reachable": False, "distance_m": 0.8}],
                    },
                ]
            }
        },
        skill_catalog=CATALOG,
    )
    payload = json.loads(
        _repair_prompt(
            request_with_navigation,
            "runtime repair after position_reachable_without_orientation=false must change the base approach",
        )
    )
    assert payload["runtime_repair_contract"]["fact_supported_candidates"][0]["preferred_pose"] == [1.0, 2.0, 0.0]


def test_user_prompt_includes_frozen_goal_contract_hash():
    contract_hash = "c" * 64
    payload = json.loads(_user_prompt(generic_request(contract_hash=contract_hash)))
    assert payload["goal_contract_hash"] == contract_hash
    assert any("goal_contract_hash" in rule for rule in payload["output_rules"])


def test_generic_plan_rejects_changed_goal_contract_hash():
    response = {
        "schema_version": "1.0",
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": "navigate",
            "stages": [
                {
                    "name": "go",
                    "goal": "navigate",
                    "depends_on": [],
                    "parameters": {
                        "skill": "base_navigate_to",
                        "target": [1.0, 0.2, 0.0],
                    },
                }
            ],
            "metadata": {
                "goal_spec_hash": "a" * 64,
                "goal_contract_hash": "b" * 64,
            },
        },
    }
    result = LLMTaskPlanner(
        FakeProvider([json.dumps(response)]), max_attempts=1
    ).plan(generic_request(contract_hash="c" * 64))
    assert result.status == "failed"
    assert "goal_contract_hash" in result.reason


def test_generic_plan_requires_matching_goal_spec_hash():
    response = {
        "schema_version": "1.0",
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": "navigate",
            "stages": [
                {
                    "name": "go",
                    "goal": "navigate",
                    "depends_on": [],
                    "parameters": {
                        "skill": "base_navigate_to",
                        "target": [1.0, 0.2, 0.0],
                    },
                }
            ],
            "metadata": {"goal_spec_hash": "b" * 64},
        },
    }
    provider = FakeProvider([json.dumps(response)])
    result = LLMTaskPlanner(provider, max_attempts=1).plan(generic_request())
    assert result.status == "failed"
    assert "goal_spec_hash" in result.reason


def test_generic_plan_preserves_matching_goal_spec_hash():
    goal_hash = "a" * 64
    response = {
        "schema_version": "1.0",
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": "navigate",
            "stages": [
                {
                    "name": "go",
                    "goal": "navigate",
                    "depends_on": [],
                    "parameters": {
                        "skill": "base_navigate_to",
                        "target": [1.0, 0.2, 0.0],
                    },
                }
            ],
            "metadata": {"goal_spec_hash": goal_hash},
        },
    }
    provider = FakeProvider([json.dumps(response)])
    result = LLMTaskPlanner(provider, max_attempts=1).plan(generic_request(goal_hash=goal_hash))
    assert result.status == "planned"
    assert result.plan is not None
    assert result.plan.metadata["goal_spec_hash"] == goal_hash


def test_legacy_plan_does_not_require_goal_spec_hash():
    response = {
        "schema_version": "1.0",
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": "navigate",
            "stages": [
                {
                    "name": "go",
                    "goal": "navigate",
                    "depends_on": [],
                    "parameters": {
                        "skill": "base_navigate_to",
                        "target": [1.0, 0.2, 0.0],
                    },
                }
            ],
            "metadata": {},
        },
    }
    provider = FakeProvider([json.dumps(response)])
    result = LLMTaskPlanner(provider, max_attempts=1).plan(request())
    assert result.status == "planned"


    response = {
        "schema_version": "1.0",
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": "navigate",
            "stages": [
                {
                    "name": "go",
                    "goal": "navigate",
                    "depends_on": [],
                    "parameters": {
                        "skill": "base_navigate_to",
                        "target": [1.0, 0.2, 0.0],
                    },
                }
            ],
            "metadata": {},
        },
    }
    provider = FakeProvider([json.dumps(response)])
    result = LLMTaskPlanner(provider, max_attempts=1).plan(request())
    assert result.status == "planned"
    assert result.plan is not None
    assert result.plan.metadata["source"] == "external_llm"
    assert len(provider.calls) == 1


def test_invalid_first_response_gets_one_bounded_repair_attempt():
    provider = FakeProvider(["not-json", '{"schema_version":"1.0","status":"unsupported","reason":"closed loop","plan":null}'])
    result = LLMTaskPlanner(provider, max_attempts=2).plan(request())
    assert result.status == "unsupported"
    assert len(provider.calls) == 2
    assert "Validation error" in provider.calls[1][1]
    assert "depends_on" in provider.calls[1][1]
    assert "query_ee_after_grasp" not in provider.calls[1][1]


def test_unsupported_task_is_not_fabricated_into_a_plan():
    provider = FakeProvider(['{"schema_version":"1.0","status":"unsupported","reason":"requires dynamic feedback","plan":null}'])
    result = LLMTaskPlanner(provider, max_attempts=1).plan(request())
    assert result.status == "unsupported"
    assert result.plan is None


def test_system_prompt_uses_factual_feedback_without_task_recipe():
    prompt = _system_prompt(CATALOG)
    assert "identify the relevant entities" in prompt
    assert "coordinate frames explicit" in prompt
    assert "fact_feedback.v1" in prompt
    assert "request, observations, discrepancies, and completed_prefix" in prompt
    assert "family recovery principles" in prompt
    assert "do not copy a scene-specific repair" in prompt
    assert "do not follow a scene-specific recipe" in prompt
    assert "failure_feedback.v2" not in prompt
    assert "root_cause_hypotheses" not in prompt
    assert "required_repairs" not in prompt
    assert "do_not_repeat" not in prompt
    assert "non-contact pre-grasp" not in prompt
    assert "joint_mask_lock" not in prompt
    assert "measured descent" not in prompt
    assert "planning_time=" not in prompt
    assert "ik_candidates=" not in prompt
    assert "footprint_radius" not in prompt
    assert "approach_candidates" not in prompt
    assert "[1.35" not in prompt
    assert "constraints.previous_plan" in prompt
    assert "arm_align_gripper" not in prompt
    assert "gripper_set" not in prompt
    assert "require_between_fingers" not in prompt
    assert "ordinary tabletop pick-and-place" not in prompt
    assert "must not be the source support" not in prompt
    assert "carry on the current support" in prompt
    assert "observe object size and pose" in prompt
    assert "support_surface_name identifies the physical object supporting that destination" in prompt
    assert "position_reachable_without_orientation=false" in prompt
    assert "changing only arm side" in prompt


def test_user_prompt_preserves_previous_plan_context():
    request_with_previous = TaskPlanningRequest(
        task_description="navigate to the work area",
        scene_facts={"objects": [{"name": "crate"}]},
        skill_catalog=CATALOG,
        constraints={
            "previous_plan": {
                "task_name": "navigate",
                "stages": [{"name": "approach", "parameters": {"skill": "base_navigate_to"}}],
            }
        },
    )
    payload = json.loads(_user_prompt(request_with_previous))
    assert payload["constraints"]["previous_plan"]["stages"][0]["name"] == "approach"


def _runtime_repair_request(previous_plan):
    feedback = {
        "schema_version": "fact_feedback.v1",
        "observations": {
            "failure_code": "contact_not_centered",
            "vertical_error_m": 0.071,
            "vertical_tolerance_m": 0.015,
        },
    }
    return TaskPlanningRequest(
        task_description="pick and place an observed object",
        scene_facts={"objects": [{"name": "item"}]},
        skill_catalog=CATALOG,
        constraints={
            "active_runtime_feedback": [json.dumps(feedback)],
            "previous_plan": previous_plan,
        },
    )


def _approach_plan(*, approach_side="west", offset_z=0.15):
    return {
        "task_name": "pick",
        "stages": [
            {
                "name": "navigate",
                "parameters": {
                    "skill": "base_navigate_to",
                    "approach_side": approach_side,
                    "target_ref": "scene://support",
                },
            },
            {
                "name": "approach",
                "parameters": {
                    "skill": "arm_move_to",
                    "side": "left",
                    "target_frame": "grasp_center",
                    "target_pos": {
                        "ref": "stage.observe.details.position",
                        "offset": [0.0, 0.0, offset_z],
                    },
                },
            },
            {
                "name": "align",
                "parameters": {
                    "skill": "arm_align_gripper",
                    "side": "left",
                },
            },
        ],
    }


def test_runtime_repair_rejects_cosmetic_vertical_retry():
    previous = _approach_plan(offset_z=0.15)
    current = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={**stage["parameters"]},
            )
            for stage in _approach_plan(offset_z=0.08)["stages"]
        ),
    )
    with pytest.raises(LLMPlanValidationError, match="independently observable approach"):
        _validate_runtime_repair(current, _runtime_repair_request(previous))


def test_measured_pinch_alignment_requires_explicit_open_gripper():
    plan = Plan(
        task_name="pick",
        stages=(
            PlanStage(
                "align",
                "align the gripper",
                parameters={
                    "skill": "arm_align_gripper",
                    "side": "left",
                    "require_between_fingers": True,
                },
            ),
        ),
    )
    with pytest.raises(LLMPlanValidationError, match="preceded by gripper_set"):
        _validate_gripper_alignment_precondition(plan)


def test_measured_pinch_alignment_accepts_same_side_open_command():
    plan = Plan(
        task_name="pick",
        stages=(
            PlanStage(
                "open",
                "open the gripper",
                parameters={
                    "skill": "gripper_set",
                    "side": "left",
                    "open_value": 0.05,
                },
            ),
            PlanStage(
                "align",
                "align the gripper",
                depends_on=("open",),
                parameters={
                    "skill": "arm_align_gripper",
                    "side": "left",
                    "require_between_fingers": True,
                },
            ),
        ),
    )
    _validate_gripper_alignment_precondition(plan)


def test_runtime_repair_accepts_different_reachable_approach_side():
    previous = _approach_plan(approach_side="west")
    current = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={**stage["parameters"]},
            )
            for stage in _approach_plan(approach_side="east")["stages"]
        ),
    )
    _validate_runtime_repair(current, _runtime_repair_request(previous))


def test_validator_feedback_cannot_hide_unresolved_physical_contact_gate():
    previous = _approach_plan(approach_side="west", offset_z=0.20)
    current = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={**stage["parameters"]},
            )
            for stage in _approach_plan(approach_side="west", offset_z=0.15)["stages"]
        ),
    )
    contact_feedback = {
        "schema_version": "fact_feedback.v1",
        "skill": "arm_align_gripper",
        "observations": {
            "failure_type": "gpu",
            "failure_code": "contact_not_centered",
            "vertical_error_m": 0.071,
            "vertical_tolerance_m": 0.015,
        },
    }
    validator_feedback = {
        "schema_version": "fact_feedback.v1",
        "observations": {
            "failure_type": "validator",
            "raw_error": "target reference frame must be base",
        },
    }
    request = TaskPlanningRequest(
        task_description="pick and place an observed object",
        scene_facts={"objects": [{"name": "item"}]},
        skill_catalog=CATALOG,
        constraints={
            "active_runtime_feedback": [
                json.dumps(contact_feedback),
                json.dumps(validator_feedback),
            ],
            "previous_plan": previous,
        },
    )
    with pytest.raises(LLMPlanValidationError, match="independently observable approach"):
        _validate_runtime_repair(current, request)


def _runtime_standoff_repair_request(previous_plan):
    feedback = {
        "schema_version": "fact_feedback.v1",
        "failed_stage": "approach_standoff",
        "skill": "arm_move_to",
        "request": {"target_frame": "grasp_center"},
        "observations": {
            "failure_type": "gpu",
            "stage_details": {"planning_status": "no_collision_free_path"},
        },
    }
    return TaskPlanningRequest(
        task_description="pick and place an observed object",
        scene_facts={"objects": [{"name": "item"}]},
        skill_catalog=CATALOG,
        constraints={
            "active_runtime_feedback": [json.dumps(feedback)],
            "previous_plan": previous_plan,
        },
    )


def test_standoff_repair_rejects_lower_clearance_or_budget_only_retry():
    previous = _approach_plan(offset_z=0.10)
    lower = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={**stage["parameters"]},
            )
            for stage in _approach_plan(offset_z=0.05)["stages"]
        ),
    )
    with pytest.raises(LLMPlanValidationError, match="no_collision_free_path"):
        _validate_runtime_repair(lower, _runtime_standoff_repair_request(previous))


def test_standoff_repair_accepts_materially_higher_clearance():
    previous = _approach_plan(offset_z=0.10)
    higher = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={**stage["parameters"]},
            )
            for stage in _approach_plan(offset_z=0.15)["stages"]
        ),
    )
    _validate_runtime_repair(higher, _runtime_standoff_repair_request(previous))


def _runtime_base_repair_request(previous_plan):
    feedback = {
        "schema_version": "fact_feedback.v1",
        "failed_stage": "move_to_standoff",
        "skill": "arm_move_to",
        "request": {"target_frame": "grasp_center"},
        "observations": {
            "failure_type": "gpu",
            "position_reachable_without_orientation": False,
        },
    }
    return TaskPlanningRequest(
        task_description="pick and place an observed object",
        scene_facts={"objects": [{"name": "item"}]},
        skill_catalog=CATALOG,
        constraints={
            "active_runtime_feedback": [json.dumps(feedback)],
            "previous_plan": previous_plan,
        },
    )


def test_position_unreachable_repair_requires_base_approach_change():
    previous = _approach_plan(approach_side="west")
    arm_only = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={
                    **stage["parameters"],
                    **({"side": "right"} if stage["name"] == "align" else {}),
                },
            )
            for stage in _approach_plan(approach_side="west")["stages"]
        ),
    )
    with pytest.raises(LLMPlanValidationError, match="base approach"):
        _validate_runtime_repair(arm_only, _runtime_base_repair_request(previous))


def test_position_unreachable_repair_accepts_changed_base_approach():
    previous = _approach_plan(approach_side="west")
    changed_base = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={**stage["parameters"]},
            )
            for stage in _approach_plan(approach_side="east")["stages"]
        ),
    )
    _validate_runtime_repair(changed_base, _runtime_base_repair_request(previous))


def test_position_unreachable_repair_accepts_fact_supported_preferred_pose_change():
    previous = _approach_plan(approach_side="west")
    changed_pose = _approach_plan(approach_side="west")
    changed_pose["stages"][0]["parameters"]["preferred_pose"] = [1.349, 2.1, 0.0]
    current = Plan(
        task_name="pick",
        stages=tuple(
            PlanStage(
                stage["name"],
                stage["name"],
                parameters={**stage["parameters"]},
            )
            for stage in changed_pose["stages"]
        ),
    )
    _validate_runtime_repair(current, _runtime_base_repair_request(previous))


def test_runtime_repair_is_bounded_inside_planner_call():
    goal_hash = "a" * 64
    previous = _approach_plan(approach_side="west")
    previous["stages"][0]["parameters"]["target"] = [1.0, 0.0, 0.0]

    def envelope(approach_side: str) -> dict:
        return {
            "schema_version": "1.0",
            "status": "planned",
            "reason": "",
            "plan": {
                "task_name": "pick",
                "stages": [
                    {
                        "name": "navigate",
                        "goal": "approach the observed object",
                        "depends_on": [],
                        "parameters": {
                            "skill": "base_navigate_to",
                            "approach_side": approach_side,
                            "target": [1.0, 0.0, 0.0],
                        },
                    }
                ],
                "metadata": {"goal_spec_hash": goal_hash},
            },
        }

    provider = FakeProvider(
        [
            json.dumps(envelope("west")),
            json.dumps(envelope("east")),
        ]
    )
    request_with_feedback = TaskPlanningRequest(
        task_description="pick and place an observed object",
        scene_facts={"objects": [{"name": "item"}]},
        skill_catalog=(
            {
                "name": "base_navigate_to",
                "parameters": {
                    "approach_side": {"type": "string", "required": False},
                    "target": {"type": "array", "required": True, "shape": [3]},
                },
            },
        ),
        goal_spec={
            "schema_version": 1,
            "bindings": {"subject": "scene://item"},
            "required": [],
            "invariants": [],
        },
        goal_spec_hash=goal_hash,
        constraints={
            "active_runtime_feedback": [
                json.dumps(
                    {
                        "schema_version": "fact_feedback.v1",
                        "observations": {
                            "failure_code": "contact_not_centered",
                            "vertical_error_m": 0.071,
                            "vertical_tolerance_m": 0.015,
                        },
                    }
                )
            ],
            "previous_plan": previous,
        },
    )
    result = LLMTaskPlanner(provider, max_attempts=2).plan(request_with_feedback)
    assert result.status == "planned"
    assert result.plan is not None
    assert result.plan.stages[0].parameters["approach_side"] == "east"
    assert len(provider.calls) == 2
    assert "changing only search budget" in provider.calls[1][1]
