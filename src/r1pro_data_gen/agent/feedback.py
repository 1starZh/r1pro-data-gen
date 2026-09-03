"""Bounded feedback for plan-only replanning.

The payload records what was requested and what was observed.  Task-family
recovery (change base when unreachable, retry grasp while still in reach,
carry on the same support) lives in the planner/agent prompts, not as a
scene-specific stage list inside this record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from r1pro_data_gen.domain import Plan
    from r1pro_data_gen.execution import PlanExecution


_FAILURE_TYPES = frozenset({"gpu", "validator", "provider", "unsupported"})
_FEEDBACK_SCHEMA_VERSION = "fact_feedback.v1"
_MAX_STRING = 1000
_MAX_ITEMS = 10
_MAX_KEYS = 32
_MAX_DEPTH = 16
_MAX_CHARS = 16000
_SECRET_TOKEN_RE = re.compile(r"(?i)\b(?:sk|rk)-[a-z0-9_-]{12,}\b")
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_ -]?key|access[_ -]?token|access[_ -]?key(?:[_ -]?id)?|"
    r"auth[_ -]?token|auth[_ -]?key|authorization|client[_ -]?secret|"
    r"credential|password|private[_ -]?key|secret[_ -]?key|token)\b"
    r"\s*[:=]\s*)((?!bearer\b)[^\s,;]+)"
)
_QUOTED_SECRET_ASSIGNMENT_RE = re.compile(
    r'''(?i)(["']?\b(?:api[_ -]?key|access[_ -]?token|access[_ -]?key(?:[_ -]?id)?|'''
    r'''auth[_ -]?token|auth[_ -]?key|authorization|client[_ -]?secret|'''
    r'''credential|password|private[_ -]?key|secret[_ -]?key|secret|token)\b'''
    r'''["']?\s*[:=]\s*["'])([^"']*)(["'])'''
)


@dataclass(frozen=True, slots=True)
class Discrepancy:
    """One measured request/observation difference.

    Values are intentionally untyped JSON values.  The originating skill or
    GoalSpec owns the meaning of a field; this record only preserves the
    reported request, observation, tolerance, and optional evidence time.
    """

    field: str
    requested: object = None
    observed: object = None
    tolerance: object = None
    first_violation_time: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("discrepancy field must not be blank")
        if self.first_violation_time is not None:
            value = _finite_number(self.first_violation_time)
            if value is None or value < 0.0:
                raise ValueError("first_violation_time must be finite and non-negative")
            object.__setattr__(self, "first_violation_time", value)
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise TypeError("discrepancy reason must be a string or None")
            object.__setattr__(self, "reason", _redact(self.reason)[:_MAX_STRING])

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "field": self.field,
            "requested": _json_safe(self.requested),
            "observed": _json_safe(self.observed),
            "tolerance": _json_safe(self.tolerance),
        }
        if self.first_violation_time is not None:
            result["first_violation_time"] = self.first_violation_time
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True, slots=True)
class FactFeedback:
    """Immutable factual feedback sent across the planner boundary."""

    attempt: int
    failed_stage: str | None
    skill: str | None
    request: Mapping[str, object]
    observations: Mapping[str, object]
    discrepancies: tuple[Discrepancy, ...]
    completed_prefix: tuple[str, ...]
    goal_spec_hash: str = ""
    evidence_refs: tuple[str, ...] = ()
    goal_contract_hash: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int):
            raise TypeError("attempt must be an integer")
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        for name, value in (("failed_stage", self.failed_stage), ("skill", self.skill)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
            if value is not None:
                object.__setattr__(self, name, _redact(value)[:_MAX_STRING])
        for name in ("request", "observations"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            normalized = _bounded_json(value)
            if not isinstance(normalized, dict):
                raise TypeError(f"{name} must normalize to a JSON object")
            object.__setattr__(self, name, _freeze_json(_limit_total(normalized)))
        discrepancies = tuple(self.discrepancies)
        if any(not isinstance(item, Discrepancy) for item in discrepancies):
            raise TypeError("discrepancies must contain Discrepancy values")
        object.__setattr__(self, "discrepancies", discrepancies[:_MAX_ITEMS])
        prefix = tuple(self.completed_prefix)
        if any(not isinstance(item, str) or not item.strip() for item in prefix):
            raise TypeError("completed_prefix must contain non-empty strings")
        object.__setattr__(self, "completed_prefix", tuple(_redact(item)[:_MAX_STRING] for item in prefix[:_MAX_ITEMS]))
        if not isinstance(self.goal_spec_hash, str):
            raise TypeError("goal_spec_hash must be a string")
        if self.goal_spec_hash and not re.fullmatch(r"[0-9a-fA-F]{64}", self.goal_spec_hash):
            raise ValueError("goal_spec_hash must be a SHA-256 hex digest or empty")
        object.__setattr__(self, "goal_spec_hash", self.goal_spec_hash.lower())
        if not isinstance(self.goal_contract_hash, str):
            raise TypeError("goal_contract_hash must be a string")
        if self.goal_contract_hash and not re.fullmatch(r"[0-9a-fA-F]{64}", self.goal_contract_hash):
            raise ValueError("goal_contract_hash must be a SHA-256 hex digest or empty")
        object.__setattr__(self, "goal_contract_hash", self.goal_contract_hash.lower())
        refs = tuple(self.evidence_refs)
        if any(not isinstance(item, str) or not item.strip() for item in refs):
            raise TypeError("evidence_refs must contain non-empty strings")
        object.__setattr__(self, "evidence_refs", tuple(_redact(item)[:_MAX_STRING] for item in refs[:_MAX_ITEMS]))

    @property
    def failure_type(self) -> str:
        value = self.observations.get("failure_type")
        return value if isinstance(value, str) and value in _FAILURE_TYPES else "gpu"

    @property
    def reason(self) -> str | None:
        value = self.observations.get("reason")
        return value if isinstance(value, str) else None

    @property
    def raw_error(self) -> str | None:
        value = self.observations.get("raw_error")
        return value if isinstance(value, str) else None

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Compatibility view; factual feedback never manufactures diagnostics."""
        return ()

    @property
    def evidence(self) -> Mapping[str, object]:
        """Compatibility view over observations for artifact consumers."""
        return self.observations

    def to_json(self) -> dict[str, Any]:
        return self.to_planner_payload()

    def to_planner_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": _FEEDBACK_SCHEMA_VERSION,
            "attempt": self.attempt,
            "failed_stage": self.failed_stage,
            "skill": self.skill,
            "request": _json_safe(self.request),
            "observations": _json_safe(self.observations),
            "discrepancies": [item.to_json() for item in self.discrepancies],
            "completed_prefix": list(self.completed_prefix),
            "goal_spec_hash": self.goal_spec_hash,
            "evidence_refs": list(self.evidence_refs),
        }
        if self.goal_contract_hash:
            payload["goal_contract_hash"] = self.goal_contract_hash
        return payload

    def to_markdown(self) -> str:
        payload = self.to_planner_payload()
        return "\n".join(
            (
                f"## Attempt {self.attempt} factual feedback",
                f"- failed_stage: {_markdown_scalar(self.failed_stage)}",
                f"- skill: {_markdown_scalar(self.skill)}",
                "",
                "### Request",
                json.dumps(payload["request"], ensure_ascii=False, sort_keys=True, allow_nan=False),
                "",
                "### Observations",
                json.dumps(payload["observations"], ensure_ascii=False, sort_keys=True, allow_nan=False),
                "",
                "### Discrepancies",
                json.dumps(payload["discrepancies"], ensure_ascii=False, sort_keys=True, allow_nan=False),
                "",
                "### Completed Prefix",
                json.dumps(payload["completed_prefix"], ensure_ascii=False),
                "",
                "### Evidence References",
                json.dumps(payload["evidence_refs"], ensure_ascii=False),
            )
        )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "FactFeedback":
        if data.get("schema_version") != _FEEDBACK_SCHEMA_VERSION:
            raise ValueError("initial feedback must use fact_feedback.v1")
        raw_discrepancies = data.get("discrepancies", ())
        if not isinstance(raw_discrepancies, Sequence) or isinstance(raw_discrepancies, (str, bytes)):
            raise TypeError("discrepancies must be an array")
        discrepancies = tuple(Discrepancy(**item) for item in raw_discrepancies if isinstance(item, Mapping))
        return cls(
            attempt=data["attempt"],
            failed_stage=data.get("failed_stage"),
            skill=data.get("skill"),
            request=data.get("request", {}),
            observations=data.get("observations", {}),
            discrepancies=discrepancies,
            completed_prefix=data.get("completed_prefix", ()),
            goal_spec_hash=data.get("goal_spec_hash", ""),
            evidence_refs=data.get("evidence_refs", ()),
            goal_contract_hash=data.get("goal_contract_hash", ""),
        )


# Keep the public import name while making the new contract the only type.
Feedback = FactFeedback


def extract_failure_feedback(
    task_description: str,
    scene_facts: Mapping[str, Any],
    plan: Plan | None,
    execution: PlanExecution | Mapping[str, Any] | None,
    measurements: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    attempt: int,
    provider_error: str | None = None,
    validator_error: str | None = None,
    failure_type: str | None = None,
    goal_spec_hash: str = "",
    goal_contract_hash: str = "",
    evidence_refs: Sequence[str] = (),
) -> FactFeedback:
    """Extract bounded request/observation facts without planning guidance."""
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("task_description must not be empty")
    if not isinstance(scene_facts, Mapping):
        raise TypeError("scene_facts must be a mapping")
    if measurements is not None and not isinstance(measurements, Mapping):
        raise TypeError("measurements must be a mapping or None")
    if evaluation is not None and not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping or None")
    if not _has_failure_signal(execution, evaluation, provider_error, validator_error, failure_type):
        raise ValueError("no failure signal was provided")
    evaluations = evaluation if isinstance(evaluation, Mapping) else {}
    resolved_type = _infer_failure_type(failure_type, provider_error, validator_error, evaluations, execution)
    stage, skill, reason, request, details, metrics, completed = _failed_stage_evidence(plan, execution, evaluations)
    raw_error = _clean_text(provider_error or validator_error) if resolved_type in {"provider", "validator"} else None
    if resolved_type == "unsupported":
        raw_error = _evaluation_reason(evaluations) or _clean_text(validator_error or provider_error)
    observations: dict[str, Any] = {}
    observations["failure_type"] = resolved_type
    if reason is not None:
        observations["reason"] = reason
    if raw_error is not None:
        observations["raw_error"] = raw_error
    if stage is not None:
        observations["failed_stage"] = stage
    if skill is not None:
        observations["skill"] = skill
    _merge_facts(observations, details, "stage_details")
    _merge_facts(observations, metrics, "stage_metrics", flatten=True)
    _merge_facts(observations, measurements or {}, "measurements")
    _merge_facts(observations, _bounded_evaluation(evaluations), "evaluation")
    discrepancies = _generic_discrepancies(observations)
    return FactFeedback(
        attempt=attempt,
        failed_stage=stage,
        skill=skill,
        request=request,
        observations=observations,
        discrepancies=discrepancies,
        completed_prefix=completed,
        goal_spec_hash=goal_spec_hash,
        goal_contract_hash=goal_contract_hash,
        evidence_refs=tuple(evidence_refs),
    )


def _failed_stage_evidence(plan: Any, execution: Any, evaluation: Mapping[str, Any]) -> tuple[Any, ...]:
    failed = _text(_get(execution, "failed")) or _text(_get(execution, "failed_stage")) or _evaluation_stage(evaluation)
    results = _get(execution, "stage_results", {})
    calls = _get(execution, "stage_calls", {})
    if failed is None and isinstance(results, Mapping):
        for name, result in results.items():
            if _get(result, "success", True) is False:
                failed = _text(name)
                break
    call = calls.get(failed) if failed is not None and isinstance(calls, Mapping) else None
    result = results.get(failed) if failed is not None and isinstance(results, Mapping) else None
    if call is None:
        nested_call = _get(result, "call")
        if isinstance(nested_call, Mapping):
            call = nested_call
    plan_stage = _plan_stage(plan, failed)
    plan_params = _get(plan_stage, "parameters", {})
    if not isinstance(plan_params, Mapping):
        plan_params = {}
    skill = _text(_get(call, "skill")) or _text(_get(result, "skill")) or _text(plan_params.get("skill"))
    request: dict[str, Any] = {}
    for candidate in (
        _get(call, "resolved_parameters", {}),
        _get(call, "parameters", {}),
        _get(call, "raw_parameters", {}),
        plan_params,
    ):
        if isinstance(candidate, Mapping):
            for key, value in candidate.items():
                if key != "skill" and key not in request:
                    request[str(key)] = value
    details = _get(result, "details", {})
    details = details if isinstance(details, Mapping) else {}
    metrics = _get(result, "metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    if not metrics and isinstance(result, Mapping):
        excluded = {"success", "skill", "details", "call", "error", "failure_reason"}
        metrics = {str(key): value for key, value in result.items() if key not in excluded}
    reason = _text(_get(execution, "failure_reason")) or _text(details.get("reason")) or _text(_get(result, "failure_reason")) or _text(_get(result, "error")) or _evaluation_reason(evaluation)
    completed_value = _get(execution, "completed", ())
    completed = tuple(item for item in completed_value if isinstance(item, str) and item.strip()) if isinstance(completed_value, Sequence) and not isinstance(completed_value, (str, bytes)) else ()
    return failed, skill, reason, request, details, metrics, completed


def _merge_facts(target: dict[str, Any], value: Mapping[str, Any], key: str, *, flatten: bool = False) -> None:
    if not value:
        return
    safe = _bounded_json(value)
    if not isinstance(safe, dict):
        return
    if flatten:
        target.update(safe)
    target[key] = safe


def _generic_discrepancies(observations: Mapping[str, Any]) -> tuple[Discrepancy, ...]:
    """Copy explicitly paired error/tolerance facts, preserving their units."""
    result: list[Discrepancy] = []
    for key, observed in observations.items():
        if key.endswith("_error"):
            prefix = key[:-6]
            tolerance_key = f"{prefix}_tolerance"
        elif key.endswith("_error_m"):
            prefix = key[:-8]
            tolerance_key = f"{prefix}_tolerance_m"
        elif key.endswith("_error_rad"):
            prefix = key[:-10]
            tolerance_key = f"{prefix}_tolerance_rad"
        else:
            continue
        tolerance = observations.get(tolerance_key)
        if tolerance is None:
            continue
        result.append(Discrepancy(field=prefix, observed=observed, tolerance=tolerance))
    return tuple(result[:_MAX_ITEMS])


def _has_failure_signal(execution: Any, evaluation: Mapping[str, Any] | None, provider_error: Any, validator_error: Any, failure_type: Any) -> bool:
    if failure_type is not None or _clean_text(provider_error) or _clean_text(validator_error):
        return True
    if isinstance(evaluation, Mapping):
        status = _text(evaluation.get("status"))
        if status in {"failed", "failure", "error", "unsupported"} or evaluation.get("success") is False or evaluation.get("failure") is not None:
            return True
    if execution is None:
        return False
    if _text(_get(execution, "failed")) or _text(_get(execution, "failed_stage")) or _get(execution, "success", True) is False:
        return True
    results = _get(execution, "stage_results", {})
    return isinstance(results, Mapping) and any(_get(item, "success", True) is False for item in results.values())


def _infer_failure_type(explicit: str | None, provider: Any, validator: Any, evaluation: Mapping[str, Any], execution: Any) -> str:
    if explicit is not None:
        if explicit not in _FAILURE_TYPES:
            raise ValueError(f"unsupported failure_type: {explicit!r}")
        return explicit
    if _clean_text(provider):
        return "provider"
    if _clean_text(validator):
        return "validator"
    if _text(evaluation.get("status")) == "unsupported":
        return "unsupported"
    return "gpu"


def _bounded_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return _bounded_json(evaluation) if isinstance(_bounded_json(evaluation), dict) else {}


def _evaluation_stage(evaluation: Mapping[str, Any]) -> str | None:
    failure = evaluation.get("failure") if isinstance(evaluation, Mapping) else None
    if isinstance(failure, Mapping):
        return _text(failure.get("stage"))
    return _text(evaluation.get("stage")) if isinstance(evaluation, Mapping) else None


def _evaluation_reason(evaluation: Mapping[str, Any]) -> str | None:
    nodes = [evaluation]
    failure = evaluation.get("failure") if isinstance(evaluation, Mapping) else None
    if isinstance(failure, Mapping):
        nodes.insert(0, failure)
    for node in nodes:
        for key in ("reason", "message", "error", "failure_reason"):
            value = _text(node.get(key))
            if value:
                return value
    return None


def _plan_stage(plan: Any, name: str | None) -> Any:
    if plan is None or name is None:
        return None
    stages = _get(plan, "stages", ())
    for stage in stages if isinstance(stages, (list, tuple)) else ():
        if _get(stage, "name") == name:
            return stage
    return None


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default) if value is not None else default


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = _redact(value if isinstance(value, str) else str(value)).strip()[:_MAX_STRING]
    return text or None


def _clean_text(value: Any) -> str | None:
    return _text(value)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(marker in normalized for marker in ("apikey", "token", "password", "secret", "credential", "authorization", "privatekey"))


def _redact(value: str) -> str:
    value = _BEARER_RE.sub(r"\1[REDACTED]", value)
    value = _QUOTED_SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]\3", value)
    value = _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", value)
    return _SECRET_TOKEN_RE.sub("[REDACTED]", value)


def _bounded_json(value: Any, *, depth: int = 0, key: str | None = None, seen: set[int] | None = None) -> Any:
    if depth > _MAX_DEPTH:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact(value)[:_MAX_STRING]
    if isinstance(value, Enum):
        return _bounded_json(value.value, depth=depth + 1, key=key, seen=seen)
    if seen is None:
        seen = set()
    identity = id(value)
    container = isinstance(value, (Mapping, list, tuple, set)) or is_dataclass(value)
    if container and identity in seen:
        return "<recursive>"
    if container:
        seen.add(identity)
    try:
        if is_dataclass(value):
            return {field.name: ("[REDACTED]" if _sensitive_key(field.name) else _bounded_json(getattr(value, field.name), depth=depth + 1, key=field.name, seen=seen)) for field in fields(value)}
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for index, (raw_key, item) in enumerate(value.items()):
                if index >= _MAX_KEYS:
                    break
                text_key = str(raw_key)
                result[_redact(text_key)[:_MAX_STRING]] = "[REDACTED]" if _sensitive_key(text_key) else _bounded_json(item, depth=depth + 1, key=text_key, seen=seen)
            return result
        if isinstance(value, (list, tuple, set)):
            items = list(value)[:_MAX_ITEMS]
            return [_bounded_json(item, depth=depth + 1, key=key, seen=seen) for item in items]
        for method_name in ("item", "tolist"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    converted = method()
                except Exception:
                    continue
                if converted is not value:
                    return _bounded_json(converted, depth=depth + 1, key=key, seen=seen)
        return _redact(str(value))[:_MAX_STRING]
    finally:
        if container:
            seen.discard(identity)


def _limit_total(value: dict[str, Any]) -> dict[str, Any]:
    encoded = lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded(value)) <= _MAX_CHARS:
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        candidate = {**result, key: item}
        if len(encoded(candidate)) <= _MAX_CHARS:
            result = candidate
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Discrepancy):
        return value.to_json()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _markdown_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return _redact(value).replace("\r", " ").replace("\n", " ")
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)


__all__ = ["Discrepancy", "FactFeedback", "Feedback", "extract_failure_feedback"]
