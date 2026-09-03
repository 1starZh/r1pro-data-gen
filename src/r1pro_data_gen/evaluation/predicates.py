"""Predicate and report result contracts for deterministic verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class PredicateStatus(StrEnum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class VerificationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class PredicateEvaluation:
    predicate: str
    status: PredicateStatus
    requested: Mapping[str, object] = field(default_factory=dict)
    observed: Mapping[str, object] = field(default_factory=dict)
    error: Mapping[str, float] = field(default_factory=dict)
    tolerance: Mapping[str, float] = field(default_factory=dict)
    evidence_range: tuple[float, float] | None = None
    reason: str | None = None
    invariant: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", MappingProxyType(dict(self.requested)))
        object.__setattr__(self, "observed", MappingProxyType(dict(self.observed)))
        object.__setattr__(self, "error", MappingProxyType(dict(self.error)))
        object.__setattr__(self, "tolerance", MappingProxyType(dict(self.tolerance)))


@dataclass(frozen=True, slots=True)
class VerificationReport:
    status: VerificationStatus
    predicates: tuple[PredicateEvaluation, ...]
    evidence_complete: bool
    failure_reason: str | None = None


__all__ = [
    "PredicateEvaluation",
    "PredicateStatus",
    "VerificationReport",
    "VerificationStatus",
]
