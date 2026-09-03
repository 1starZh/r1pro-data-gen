"""Plan serialization: a :class:`Plan` is data.

Plans are skill-call sequences (stage name -> skill name + JSON-serializable
parameters). Making them data means the same format is produced by the
Claude planner today, the LLM planner later, and the deterministic template
planner -- and replayed through the same ``run_plan`` entrypoint.

Stage parameters must be JSON-serializable (numbers, strings, booleans,
None, lists, dicts). Tuples are converted to lists on export so the JSON
round-trip is lossless in JSON terms; skills must accept lists (they already
pass geometry through ``numpy.asarray``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from r1pro_data_gen.domain import Plan, PlanStage

_JSON_TYPES = (str, int, float, bool, type(None))


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    """Serialize a Plan to a plain dict with JSON-compatible values."""
    return {
        "task_name": plan.task_name,
        "stages": [
            {
                "name": stage.name,
                "goal": stage.goal,
                "depends_on": list(stage.depends_on),
                "parameters": _to_jsonable(stage.parameters),
                "outputs": list(stage.outputs),
                "preconditions": _to_jsonable(stage.preconditions),
                "postconditions": _to_jsonable(stage.postconditions),
            }
            for stage in plan.stages
        ],
        "metadata": _to_jsonable(plan.metadata),
    }


def plan_from_dict(data: Mapping[str, Any]) -> Plan:
    """Rebuild a Plan from ``plan_to_dict`` output (or parsed JSON)."""
    if "task_name" not in data or "stages" not in data:
        raise ValueError("plan data requires task_name and stages")
    stages = []
    for stage in data["stages"]:
        if "name" not in stage or "goal" not in stage:
            raise ValueError("each plan stage requires name and goal")
        stages.append(
            PlanStage(
                name=stage["name"],
                goal=stage["goal"],
                depends_on=tuple(stage.get("depends_on", []) or []),
                parameters=dict(stage.get("parameters", {}) or {}),
                outputs=tuple(stage.get("outputs", []) or []),
                preconditions=tuple(stage.get("preconditions", []) or []),
                postconditions=tuple(stage.get("postconditions", []) or []),
            )
        )
    plan = Plan(
        task_name=data["task_name"],
        stages=tuple(stages),
        metadata=dict(data.get("metadata", {}) or {}),
    )
    _validate_plan(plan)
    return plan


def plan_to_json(plan: Plan) -> str:
    """Serialize a Plan to a JSON string."""
    return json.dumps(plan_to_dict(plan), indent=2, sort_keys=True, ensure_ascii=False)


def plan_from_json(text: str) -> Plan:
    """Parse a Plan from a JSON string."""
    return plan_from_dict(json.loads(text))


def save_plan(plan: Plan, path: str | Path) -> None:
    """Write a Plan to a JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan_to_json(plan) + "\n", encoding="utf-8")


def load_plan(path: str | Path) -> Plan:
    """Read a Plan from a JSON file."""
    return plan_from_json(Path(path).read_text(encoding="utf-8"))


def _validate_plan(plan: Plan) -> None:
    """Reject stage parameters that are not JSON-serializable."""
    for stage in plan.stages:
        bad = _first_non_jsonable(stage.parameters)
        if bad is not None:
            raise ValueError(
                f"stage {stage.name!r} parameter {bad!r} is not JSON-serializable"
            )
    bad = _first_non_jsonable(plan.metadata)
    if bad is not None:
        raise ValueError(
            f"plan metadata value {bad!r} is not JSON-serializable"
        )


def _to_jsonable(value: Any) -> Any:
    """Convert tuples recursively to lists; keep everything else JSON-able."""
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _first_non_jsonable(value: Any) -> Any:
    """Return the first non-JSON-serializable element, or None if all good."""
    if isinstance(value, _JSON_TYPES):
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            bad = _first_non_jsonable(item)
            if bad is not None:
                return bad
        return None
    if isinstance(value, dict):
        for item in value.values():
            bad = _first_non_jsonable(item)
            if bad is not None:
                return bad
        return None
    return value


__all__ = [
    "load_plan",
    "plan_from_dict",
    "plan_from_json",
    "plan_to_dict",
    "plan_to_json",
    "save_plan",
]
