"""Contract tests for pure factual feedback extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from r1pro_data_gen.domain import Plan, PlanStage
from r1pro_data_gen.execution import PlanExecution, StageCall
from r1pro_data_gen.agent import (
    FactFeedback,
    Feedback as ExportedFeedback,
    extract_failure_feedback as exported_extract_failure_feedback,
)
from r1pro_data_gen.agent.feedback import (
    Discrepancy,
    Feedback,
    extract_failure_feedback,
)


def _minimal_plan() -> Plan:
    return Plan(
        task_name="generic_task",
        stages=(
            PlanStage(
                "observe",
                "observe an entity",
                parameters={"skill": "query_object_pose", "object_name": "item"},
            ),
            PlanStage(
                "move",
                "move the end effector",
                depends_on=("observe",),
                parameters={
                    "skill": "arm_move_to",
                    "target_pos": [1.0, 2.0, 1.2],
                    "side": "left",
                },
            ),
        ),
    )


def _failed_execution() -> PlanExecution:
    return PlanExecution(
        stage_calls={
            "move": StageCall(
                "arm_move_to",
                {"target_pos": [1.0, 2.0, 1.2], "side": "left"},
            )
        },
        completed=("observe",),
        failed="move",
        failure_reason="no verified path was found",
    )


def test_gpu_feedback_preserves_request_observation_and_prefix():
    feedback = extract_failure_feedback(
        "move the item",
        {"robot": {"init_pose": [0.0, 0.0, 0.0]}},
        _minimal_plan(),
        _failed_execution(),
        {},
        {"status": "failed"},
        attempt=1,
        goal_spec_hash="A" * 64,
        evidence_refs=("evidence.json",),
    )

    payload = feedback.to_planner_payload()
    assert payload["schema_version"] == "fact_feedback.v1"
    assert set(payload) == {
        "schema_version",
        "attempt",
        "failed_stage",
        "skill",
        "request",
        "observations",
        "discrepancies",
        "completed_prefix",
        "goal_spec_hash",
        "evidence_refs",
    }
    assert payload["failed_stage"] == "move"
    assert payload["skill"] == "arm_move_to"
    assert payload["request"] == {
        "target_pos": [1.0, 2.0, 1.2],
        "side": "left",
    }
    assert payload["observations"]["failure_type"] == "gpu"
    assert payload["observations"]["reason"] == "no verified path was found"
    assert payload["completed_prefix"] == ["observe"]
    assert payload["goal_spec_hash"] == "a" * 64
    assert payload["evidence_refs"] == ["evidence.json"]


def test_feedback_types_preserve_provider_validator_and_unsupported_facts():
    validator = extract_failure_feedback(
        "task", {}, None, None, {}, {}, attempt=1,
        validator_error="stage move uses unknown skill",
    )
    provider = extract_failure_feedback(
        "task", {}, None, None, {}, {}, attempt=2,
        provider_error="transport timeout",
    )
    unsupported = extract_failure_feedback(
        "task", {}, None, None, {},
        {"status": "unsupported", "reason": "not representable"}, attempt=3,
    )

    assert validator.to_json()["observations"] == {
        "failure_type": "validator",
        "raw_error": "stage move uses unknown skill",
    }
    assert provider.to_json()["observations"] == {
        "failure_type": "provider",
        "raw_error": "transport timeout",
    }
    assert unsupported.to_json()["observations"] == {
        "failure_type": "unsupported",
        "reason": "not representable",
        "raw_error": "not representable",
        "evaluation": {"status": "unsupported", "reason": "not representable"},
    }


def test_feedback_rejects_invalid_inputs_and_hashes():
    with pytest.raises(ValueError, match="attempt"):
        FactFeedback(0, None, None, {}, {}, (), ())
    with pytest.raises(ValueError, match="SHA-256"):
        FactFeedback(1, None, None, {}, {}, (), (), goal_spec_hash="bad")
    with pytest.raises(ValueError, match="task_description"):
        extract_failure_feedback(" ", {}, None, None, {}, {}, attempt=1)
    with pytest.raises(TypeError, match="measurements"):
        extract_failure_feedback("task", {}, None, None, [], {}, attempt=1)
    with pytest.raises(TypeError, match="evaluation"):
        extract_failure_feedback("task", {}, None, None, {}, [], attempt=1)
    with pytest.raises(ValueError, match="no failure signal"):
        extract_failure_feedback("task", {}, None, None, {}, {}, attempt=1)


def test_mapping_execution_extracts_nested_call_and_stage_facts():
    execution = {
        "stage_results": {
            "observe": {"success": True},
            "move": {
                "success": False,
                "skill": "arm_move_to",
                "details": {"reason": "collision"},
                "metrics": {"horizontal_error_m": 0.2, "horizontal_tolerance_m": 0.05},
                "call": {"skill": "arm_move_to", "raw_parameters": {"side": "left"}},
            },
        },
        "completed": ["observe"],
    }
    feedback = extract_failure_feedback(
        "move item", {}, None, execution, {}, {"status": "failed"}, attempt=1
    )

    payload = feedback.to_json()
    assert payload["failed_stage"] == "move"
    assert payload["skill"] == "arm_move_to"
    assert payload["request"] == {"side": "left"}
    assert payload["observations"]["reason"] == "collision"
    assert payload["observations"]["horizontal_error_m"] == 0.2
    assert payload["discrepancies"] == [
        {
            "field": "horizontal",
            "requested": None,
            "observed": 0.2,
            "tolerance": 0.05,
        }
    ]


def test_discrepancies_include_paired_radian_error_and_tolerance():
    """Radian-unit errors pair with their tolerance like meter ones do."""
    execution = {
        "stage_results": {
            "move": {
                "success": False,
                "skill": "arm_move_to",
                "metrics": {
                    "rotation_error_rad": 0.127,
                    "rotation_tolerance_rad": 0.10,
                    "ik_error_m": 0.288,
                    "ik_tolerance_m": 0.03,
                },
                "call": {"skill": "arm_move_to", "raw_parameters": {}},
            },
        },
        "completed": [],
    }
    feedback = extract_failure_feedback(
        "move item", {}, None, execution, {}, {"status": "failed"}, attempt=1
    )

    by_field = {item["field"]: item for item in feedback.to_json()["discrepancies"]}
    assert by_field["rotation"] == {
        "field": "rotation",
        "requested": None,
        "observed": 0.127,
        "tolerance": 0.10,
    }
    assert by_field["ik"] == {
        "field": "ik",
        "requested": None,
        "observed": 0.288,
        "tolerance": 0.03,
    }


def test_feedback_normalizes_dataclasses_and_redacts_secrets():
    @dataclass(frozen=True)
    class Measurement:
        distance: float
        token: str

    feedback = Feedback(
        attempt=1,
        failed_stage="move token=sk-test-secret-value-12345",
        skill="arm_move_to",
        request={"samples": (1, 2)},
        observations={
            "measurement": Measurement(0.4, "TOPSECRET"),
            "nested": {"access_token": "TOPSECRET"},
            "raw_error": "authorization: Bearer TOPSECRET",
        },
        discrepancies=(),
        completed_prefix=(),
    )

    payload = feedback.to_json()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "TOPSECRET" not in encoded
    assert "sk-test-secret-value-12345" not in encoded
    assert payload["failed_stage"] == "move token=[REDACTED]"
    assert payload["observations"]["measurement"] == {
        "distance": 0.4,
        "token": "[REDACTED]",
    }
    assert payload["observations"]["nested"]["access_token"] == "[REDACTED]"


def test_feedback_is_immutable_and_discrepancy_validates_time():
    feedback = Feedback(
        attempt=1,
        failed_stage=None,
        skill=None,
        request={"nested": {"values": [1, 2]}},
        observations={},
        discrepancies=(Discrepancy("distance", observed=0.2, tolerance=0.1),),
        completed_prefix=(),
    )
    with pytest.raises(TypeError):
        feedback.request["new"] = "value"
    with pytest.raises(TypeError):
        feedback.request["nested"]["new"] = "value"
    with pytest.raises(ValueError, match="first_violation_time"):
        Discrepancy("distance", first_violation_time=-1.0)


def test_feedback_round_trips_json_and_markdown_contains_only_contract_fields():
    feedback = Feedback(
        attempt=4,
        failed_stage="stage",
        skill="generic_skill",
        request={"value": 1},
        observations={"failure_type": "gpu", "reason": "failed"},
        discrepancies=(),
        completed_prefix=("previous",),
        goal_spec_hash="b" * 64,
        evidence_refs=("evidence.json",),
    )
    restored = FactFeedback.from_json(feedback.to_json())

    assert restored == feedback
    markdown = feedback.to_markdown()
    assert "factual feedback" in markdown
    assert "Request" in markdown
    assert "Observations" in markdown
    assert "required_repairs" not in markdown
    assert "root_cause_hypotheses" not in markdown
    assert "do_not_repeat" not in markdown


def test_feedback_has_no_planner_recipe_or_task_specific_guidance():
    execution = {
        "stage_calls": {
            "action": {
                "skill": "generic_action",
                "resolved_parameters": {"require_vertical_alignment": True},
            }
        },
        "stage_results": {
            "action": {
                "success": False,
                "metrics": {
                    "actual_displacement_m": 0.0,
                    "vertical_error_m": 0.08,
                    "vertical_tolerance_m": 0.015,
                },
            }
        },
        "failed": "action",
        "failure_reason": "tolerance not reached",
    }
    payload = extract_failure_feedback(
        "perform a generic action", {}, None, execution, {}, {"status": "failed"}, 1
    ).to_planner_payload()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert payload["request"]["require_vertical_alignment"] is True
    assert payload["observations"]["actual_displacement_m"] == 0.0
    assert "root_cause_hypotheses" not in encoded
    assert "required_repairs" not in encoded
    assert "do_not_repeat" not in encoded
    assert "arm_move_directional" not in encoded
    assert "base_navigate_to" not in encoded
    assert "planning_time" not in encoded
    assert "ik_candidates" not in encoded
    assert '"require_vertical_alignment": false' not in encoded.lower()


def test_navigation_resolution_facts_survive_feedback_boundary():
    execution = {
        "stage_calls": {
            "navigate": {
                "skill": "base_navigate_to",
                "raw_parameters": {
                    "target_ref": "scene://target",
                    "purpose": "pregrasp",
                },
                "resolved_parameters": {
                    "target": [1.35, 2.3, 0.0],
                },
            }
        },
        "stage_results": {
            "navigate": {
                "success": False,
                "skill": "base_navigate_to",
                "details": {
                    "reason": "goal cell is inside an obstacle",
                    "error_code": "NO_SAFE_APPROACH",
                    "target_ref": "scene://target",
                    "resolved_target": [1.35, 2.3, 0.0],
                    "alternative_targets": [[2.45, 2.3, 3.1416]],
                },
            }
        },
        "failed": "navigate",
        "failure_reason": "goal cell is inside an obstacle",
    }
    feedback = extract_failure_feedback(
        "reach the target for manipulation", {}, None, execution, {}, {"status": "failed"}, 1
    )

    payload = feedback.to_planner_payload()
    assert payload["observations"]["stage_details"]["target_ref"] == "scene://target"
    assert payload["observations"]["stage_details"]["alternative_targets"]
    assert "required_repairs" not in json.dumps(payload)


def test_public_exports_use_fact_feedback_contract():
    assert ExportedFeedback is FactFeedback
    assert exported_extract_failure_feedback is extract_failure_feedback


def test_feedback_bounds_large_inputs():
    execution = {
        "failed": "stage",
        "failure_reason": "failure",
        "completed": [f"stage-{i}" for i in range(50)],
        "stage_results": {
            "stage": {
                "success": False,
                "details": {"log": "x" * 5000},
                "metrics": {"samples": list(range(50))},
            }
        },
    }
    feedback = extract_failure_feedback(
        "task", {}, None, execution,
        {"samples": list(range(50))}, {"status": "failed"}, attempt=1,
    )
    payload = feedback.to_json()
    assert len(payload["completed_prefix"]) <= 10
    assert len(payload["observations"]["measurements"]["samples"]) <= 10
    assert len(payload["observations"]["stage_details"]["log"]) <= 1000
    assert len(payload["observations"]["stage_metrics"]["samples"]) <= 10


@pytest.mark.parametrize("failure_field", ["failed", "failed_stage"])
def test_false_execution_failure_fields_do_not_create_signal(failure_field):
    with pytest.raises(ValueError, match="no failure signal"):
        extract_failure_feedback(
            "task", {}, None, {failure_field: False}, {}, {}, attempt=1
        )
