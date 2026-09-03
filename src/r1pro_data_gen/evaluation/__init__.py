"""Deterministic task-goal evaluation over generic execution evidence."""

from .acceptance import (
    ACCEPTANCE_SCHEMA_VERSION,
    AcceptanceDecision,
    AcceptanceStatus,
    evaluate_acceptance,
    finalize_result_payload,
)
from .policy import VERIFICATION_POLICY_VERSION, VerificationPolicy
from .predicates import (
    PredicateEvaluation,
    PredicateStatus,
    VerificationReport,
    VerificationStatus,
)
from .verifier import PredicateVerifier

__all__ = [
    "ACCEPTANCE_SCHEMA_VERSION",
    "AcceptanceDecision",
    "AcceptanceStatus",
    "PredicateEvaluation",
    "PredicateStatus",
    "PredicateVerifier",
    "VERIFICATION_POLICY_VERSION",
    "VerificationPolicy",
    "VerificationReport",
    "VerificationStatus",
    "evaluate_acceptance",
    "finalize_result_payload",
]
