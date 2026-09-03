"""One acceptance gate shared by product and replay entry points."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .predicates import VerificationReport, VerificationStatus


ACCEPTANCE_SCHEMA_VERSION = 2


class AcceptanceStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    """Auditable final gate for one episode.

    Goal satisfaction, evidence coverage, stage outcomes, and artifact
    validity are intentionally separate.  A failed intermediate skill is
    retained as a diagnostic fact and does not veto a physically satisfied
    goal by itself.
    """

    status: AcceptanceStatus
    goal_status: VerificationStatus
    goal_satisfied: bool
    evidence_coverage_complete: bool
    stage_success_complete: bool
    artifact_valid: bool
    hashes_match: bool
    reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status is AcceptanceStatus.ACCEPTED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ACCEPTANCE_SCHEMA_VERSION,
            "status": self.status.value,
            "goal_status": self.goal_status.value,
            "goal_satisfied": self.goal_satisfied,
            "evidence_coverage_complete": self.evidence_coverage_complete,
            "stage_success_complete": self.stage_success_complete,
            "artifact_valid": self.artifact_valid,
            "hashes_match": self.hashes_match,
            "reasons": list(self.reasons),
        }


def evaluate_acceptance(
    report: VerificationReport,
    *,
    evidence_coverage_complete: bool,
    stage_success_complete: bool,
    artifact_valid: bool,
    actual_goal_spec_hash: str | None = None,
    expected_goal_spec_hash: str | None = None,
    actual_contract_hash: str | None = None,
    expected_contract_hash: str | None = None,
) -> AcceptanceDecision:
    """Apply the common final acceptance gate.

    Hash equality is required only when an expected hash is supplied.  This
    keeps the helper useful for pure unit tests while making every frozen
    product/replay run validate its immutable GoalSpec and GoalContract.
    """
    goal_satisfied = report.status is VerificationStatus.SUCCEEDED
    coverage = bool(evidence_coverage_complete and report.evidence_complete)
    artifact = bool(artifact_valid)
    hashes_match = _hashes_match(
        actual_goal_spec_hash,
        expected_goal_spec_hash,
        actual_contract_hash,
        expected_contract_hash,
    )
    reasons: list[str] = []
    if not goal_satisfied:
        reasons.append(f"goal verification is {report.status.value}")
    if not coverage:
        reasons.append("evidence coverage is incomplete")
    if not artifact:
        reasons.append("required output artifact is invalid")
    if not hashes_match:
        reasons.append("frozen GoalSpec or GoalContract hash mismatch")
    status = AcceptanceStatus.ACCEPTED if not reasons else AcceptanceStatus.REJECTED
    return AcceptanceDecision(
        status=status,
        goal_status=report.status,
        goal_satisfied=goal_satisfied,
        evidence_coverage_complete=coverage,
        stage_success_complete=bool(stage_success_complete),
        artifact_valid=artifact,
        hashes_match=hashes_match,
        reasons=tuple(reasons),
    )


def finalize_result_payload(
    payload: Mapping[str, Any],
    *,
    expected_goal_spec_hash: str | None = None,
    expected_contract_hash: str | None = None,
    artifact_valid: bool | None = None,
) -> dict[str, Any]:
    """Attach acceptance fields and normalize the public result status.

    Entry points may build their detailed predicate report independently;
    this adapter keeps their final JSON shape and gate identical.
    """
    result = {str(key): value for key, value in payload.items()}
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("result payload must contain an evaluation object")
    report_status = _verification_status(evaluation.get("status", "incomplete"))
    evidence_complete = bool(
        evaluation.get(
            "evidence_coverage_complete",
            evaluation.get("evidence_complete", False),
        )
    )
    stage_success_complete = bool(evaluation.get("stage_success_complete", False))
    if artifact_valid is None:
        artifact_valid = _video_artifact_valid(result)
    decision = evaluate_acceptance(
        _report_stub(report_status, evidence_complete),
        evidence_coverage_complete=evidence_complete,
        stage_success_complete=stage_success_complete,
        artifact_valid=bool(artifact_valid),
        actual_goal_spec_hash=_optional_string(result.get("goal_spec_hash")),
        expected_goal_spec_hash=expected_goal_spec_hash,
        actual_contract_hash=_optional_string(
            result.get("goal_contract_hash", result.get("contract_hash"))
        ),
        expected_contract_hash=expected_contract_hash,
    )
    evaluation_copy = dict(evaluation)
    evaluation_copy["evidence_coverage_complete"] = decision.evidence_coverage_complete
    evaluation_copy["stage_success_complete"] = decision.stage_success_complete
    result["evaluation"] = evaluation_copy
    result["acceptance"] = decision.to_dict()
    result["result"] = "passed" if decision.accepted else "failed"
    result["status"] = "succeeded" if decision.accepted else "failed"
    result["reason"] = None if decision.accepted else "; ".join(decision.reasons)
    return result


def _hashes_match(
    actual_goal: str | None,
    expected_goal: str | None,
    actual_contract: str | None,
    expected_contract: str | None,
) -> bool:
    if expected_goal is not None and actual_goal != expected_goal:
        return False
    if expected_contract is not None and actual_contract != expected_contract:
        return False
    return True


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _verification_status(value: object) -> VerificationStatus:
    """Normalize serialized status without letting malformed output crash the gate."""
    try:
        return VerificationStatus(str(value))
    except ValueError:
        return VerificationStatus.INCOMPLETE


def _video_artifact_valid(payload: Mapping[str, Any]) -> bool:
    """Validate the minimum RGB/video artifact contract shared by entrypoints."""
    try:
        return bool(
            float(payload.get("video_rgb_valid", 0.0)) > 0.0
            and float(payload.get("video_bytes", 0.0)) > 0.0
            and float(payload.get("video_frame_count", 0.0)) > 0.0
            and float(payload.get("video_duration_s", 0.0)) > 0.0
        )
    except (TypeError, ValueError):
        return False


def _report_stub(status: VerificationStatus, evidence_complete: bool) -> VerificationReport:
    """Create the minimal report needed when finalizing serialized JSON."""
    return VerificationReport(
        status=status,
        predicates=(),
        evidence_complete=evidence_complete,
    )


__all__ = [
    "ACCEPTANCE_SCHEMA_VERSION",
    "AcceptanceDecision",
    "AcceptanceStatus",
    "evaluate_acceptance",
    "finalize_result_payload",
]
