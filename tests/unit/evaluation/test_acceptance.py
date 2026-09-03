from __future__ import annotations

from r1pro_data_gen.evaluation import (
    VerificationReport,
    VerificationStatus,
    evaluate_acceptance,
    finalize_result_payload,
)


def _report(status: VerificationStatus, *, evidence_complete: bool = True) -> VerificationReport:
    return VerificationReport(
        status=status,
        predicates=(),
        evidence_complete=evidence_complete,
    )


def test_physical_goal_can_pass_with_a_failed_intermediate_stage() -> None:
    decision = evaluate_acceptance(
        _report(VerificationStatus.SUCCEEDED),
        evidence_coverage_complete=True,
        stage_success_complete=False,
        artifact_valid=True,
    )

    assert decision.accepted
    assert not decision.stage_success_complete


def test_acceptance_rejects_missing_evidence_or_artifact() -> None:
    decision = evaluate_acceptance(
        _report(VerificationStatus.SUCCEEDED, evidence_complete=False),
        evidence_coverage_complete=False,
        stage_success_complete=True,
        artifact_valid=False,
    )

    assert not decision.accepted
    assert "evidence coverage is incomplete" in decision.reasons
    assert "required output artifact is invalid" in decision.reasons


def test_result_finalization_requires_video_and_frozen_hashes() -> None:
    payload = finalize_result_payload(
        {
            "goal_spec_hash": "a" * 64,
            "goal_contract_hash": "b" * 64,
            "evaluation": {
                "status": "succeeded",
                "evidence_complete": True,
                "stage_success_complete": False,
            },
            "video_rgb_valid": 1.0,
            "video_frame_count": 10.0,
            "video_duration_s": 1.0,
        },
        expected_goal_spec_hash="a" * 64,
        expected_contract_hash="b" * 64,
        artifact_valid=True,
    )

    assert payload["result"] == "passed"
    assert payload["status"] == "succeeded"
    assert payload["acceptance"]["status"] == "accepted"
    assert payload["acceptance"]["stage_success_complete"] is False
