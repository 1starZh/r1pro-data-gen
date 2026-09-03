"""Debug-only semantic skeletons derived from a frozen GoalSpec.

This mapping is an operator artifact (``plan_skeleton.json``). It must not be
sent to the agent or task-planning LLM: listing candidate skills in GoalSpec
order is a method leak. The live agent chooses from observation + GoalSpec +
the public catalogue only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from r1pro_data_gen.domain import GoalSpec


_INTENT_BY_PREDICATE: dict[str, tuple[str, tuple[str, ...]]] = {
    "base_at_pose": ("reach_base_goal", ("base_navigate_to",)),
    "object_at_pose": (
        "move_entity_to_pose",
        ("push_object_to", "arm_carry_object_to"),
    ),
    "within_tolerance": ("measure_entity_state", ()),
    "inside_region": (
        "move_entity_into_region",
        ("arm_carry_object_to", "release_object", "push_object_to"),
    ),
    "on_support": (
        "place_entity_on_support",
        ("arm_carry_object_to", "release_object", "push_object_to"),
    ),
    "attached": (
        "establish_attachment",
        ("grasp_object", "prepare_workspace"),
    ),
    "lifted": ("lift_or_transport_entity", ("arm_carry_object_to",)),
    "released": ("release_entity", ("release_object",)),
    "settled": ("observe_settled_entity", ()),
    "contact": ("establish_or_measure_contact", ("grasp_object", "push_object_to")),
    "collision_free": ("check_safety_invariant", ()),
}


def build_semantic_plan_skeleton(
    goal_spec: GoalSpec | None,
    *,
    skill_catalogue: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a bounded, non-executable semantic skeleton from GoalSpec."""
    if goal_spec is None:
        return {"schema_version": "semantic_skeleton.v1", "steps": [], "invariants": []}
    available = {
        item.get("name")
        for item in skill_catalogue
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    steps = [
        _step(index, predicate, goal_spec.bindings, available)
        for index, predicate in enumerate(goal_spec.required, start=1)
    ]
    invariants = [
        {
            "predicate": item.predicate,
            "role": "invariant",
            "arguments": {
                key: _binding_name(value, goal_spec.bindings)
                for key, value in item.arguments.items()
            },
        }
        for item in goal_spec.invariants
    ]
    return {
        "schema_version": "semantic_skeleton.v1",
        "source": "frozen_goal_spec",
        "steps": steps,
        "invariants": invariants,
        "execution_policy": {
            "one_skill_per_step": True,
            "verify_after_each_skill": True,
            "replan_suffix_on_observed_failure": True,
        },
    }


def _step(
    index: int,
    predicate: Any,
    bindings: Mapping[str, str],
    available: set[object],
) -> dict[str, Any]:
    intent, candidates = _INTENT_BY_PREDICATE.get(
        predicate.predicate,
        ("satisfy_goal_predicate", ()),
    )
    compatible = [name for name in candidates if not available or name in available]
    entities = {
        key: _binding_name(value, bindings)
        for key, value in predicate.arguments.items()
        if key in {"subject", "support", "reference", "entity", "entity_a", "entity_b"}
    }
    return {
        "index": index,
        "intent": intent,
        "goal_predicate": predicate.predicate,
        "entities": entities,
        "candidate_skills": compatible,
        "status": "pending",
    }


def _binding_name(value: object, bindings: Mapping[str, str]) -> object:
    if not isinstance(value, str):
        return value
    root, _, suffix = value.partition(".")
    reference = bindings.get(root)
    if reference is None:
        return value
    name = reference.removeprefix("scene://")
    return name if not suffix else f"{name}.{suffix}"


__all__ = ["build_semantic_plan_skeleton"]
