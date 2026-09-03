"""Constrained natural-language goal to frozen GoalSpec planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from collections.abc import Mapping
from typing import Any

from r1pro_data_gen.domain import (
    GoalSpec,
    SceneModel,
    goal_spec_sha256,
    goal_spec_to_dict,
    parse_goal_spec,
)
from ..llm.contracts import parse_json_object
from ..llm.providers.protocol import ProviderError, TaskPlanningProvider
from .compiler import GoalCompileError, GoalCompiler
from .completeness import goal_spec_completeness_errors


@dataclass(frozen=True, slots=True)
class GoalPlanningRequest:
    task_description: str
    scene_facts: Mapping[str, Any]
    scene: SceneModel

    def __post_init__(self) -> None:
        if not self.task_description.strip():
            raise ValueError("task_description must not be empty")
        if not isinstance(self.scene_facts, Mapping):
            raise TypeError("scene_facts must be a mapping")
        if not isinstance(self.scene, SceneModel):
            raise TypeError("scene must be a SceneModel")


@dataclass(frozen=True, slots=True)
class GoalPlanningResult:
    status: str
    goal_spec: GoalSpec | None = None
    goal_spec_hash: str | None = None
    goal_contract_hash: str | None = None
    reason: str = ""
    provider: str = ""
    model: str = ""
    raw_response: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"planned", "failed"}:
            raise ValueError(f"unsupported goal planning status: {self.status!r}")
        if self.status == "planned":
            if self.goal_spec is None or not self.goal_spec_hash:
                raise ValueError("planned goal result requires spec and hash")
        elif not self.reason.strip():
            raise ValueError("failed goal result requires reason")


class GoalPlanner:
    """Generate and strictly ground GoalSpec; never emits execution actions."""

    name = "goal_planner"

    def __init__(self, provider: TaskPlanningProvider, *, max_attempts: int = 2) -> None:
        if max_attempts < 1 or max_attempts > 2:
            raise ValueError("max_attempts must be 1 or 2")
        self.provider = provider
        self.max_attempts = max_attempts
        self.model = provider.model

    def plan(self, request: GoalPlanningRequest) -> GoalPlanningResult:
        system = _system_prompt()
        user = _user_prompt(request)
        last_error = ""
        raw_response: Mapping[str, Any] | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.provider.complete(system=system, user=user)
                payload = parse_json_object(response.text)
                raw_response = payload
                spec = parse_goal_spec(payload, request.scene)
                completeness_errors = goal_spec_completeness_errors(
                    spec,
                    request.task_description,
                    request.scene,
                )
                if completeness_errors:
                    raise GoalCompileError(
                        "INCOMPLETE_GOAL_SPEC",
                        "; ".join(completeness_errors),
                        details={"errors": list(completeness_errors)},
                    )
                # Compile before freezing: geometry and observation coverage
                # are deterministic scene facts, not fields the LLM may guess.
                compiled = GoalCompiler().compile(spec, request.scene)
                return GoalPlanningResult(
                    status="planned",
                    goal_spec=spec,
                    goal_spec_hash=goal_spec_sha256(spec),
                    goal_contract_hash=compiled.contract_hash,
                    provider=response.provider,
                    model=response.model,
                    raw_response=payload,
                    usage=response.usage,
                )
            except (ProviderError, TypeError, ValueError, GoalCompileError) as exc:
                last_error = str(exc)
                if attempt + 1 < self.max_attempts:
                    user = _repair_prompt(request, last_error, raw_response)
        return GoalPlanningResult(
            status="failed",
            reason=last_error or "goal planner failed",
            provider=getattr(self.provider, "name", "unknown"),
            model=getattr(self.provider, "model", "unknown"),
            raw_response=raw_response,
        )


def _system_prompt() -> str:
    return (
        "You are a constrained goal-specification planner. Convert the natural-language "
        "goal and structured scene facts into exactly one bare JSON object with only "
        "schema_version, bindings, required, and invariants. Each predicate object "
        "must have exactly the keys \"predicate\" and \"arguments\"; never use the "
        "key \"args\". The arguments value must always be a JSON object, never an array; "
        "schema_version must be the integer 1. Bind aliases only to "
        "existing scene:// object names. Predicates must use the closed vocabulary "
        "object_at_pose, within_tolerance, inside_region, on_support, contact, "
        "attached, lifted, released, settled, base_at_pose, collision_free. "
        "Only emit contact when scene_facts.contact_sensors contains a pair "
        "covering both referenced entities; only emit collision_free when "
        "scene_facts.collision_sensors explicitly covers the referenced subject. "
        "If a predicate has no declared sensor or history coverage, omit it "
        "rather than guessing telemetry or making the goal unverifiable. "
        "Arguments must use the canonical predicate contracts: subject for the target "
        "entity; on_support requires subject, support, surface, and "
        "subject_half_height_m. The on_support surface must be an object copied from "
        "the support object's scene_facts.surfaces entry, containing exactly center "
        "(three numbers) and size (two positive numbers). Do not use a surface name "
        "string such as top. released optionally accepts effector, but the "
        "effector, when supplied, must identify a concrete end-effector; omit it when the "
        "release is agnostic to which gripper performed it. Never use robot, "
        "base, or mobile_base as an effector. contact uses entity_a and "
        "entity_b; attached uses subject and optionally a concrete end-effector; "
        "contact may use a bound scene object plus the robot/base/mobile_base semantic "
        "endpoint, or the exact robot body name listed in scene_facts.contact_sensors; "
        "inside_region uses subject, reference, and region where region is an "
        "object with shape (\"cuboid\" or \"cylinder\"), center (three numbers), "
        "and either size (three positive numbers) for cuboid or radius and "
        "height (positive numbers) for cylinder -- never a region name string; "
        "base_at_pose requires pose and may optionally include subject as one "
        "of these three exact strings: robot, base, or mobile_base; never use "
        "the slash-separated string robot/base/mobile_base; omit subject when "
        "the single robot base is implicit. "
        "Never substitute object for subject. Numeric geometry must come from the "
        "provided scene facts. "
        "Scene:// URIs appear only as the values of bindings (alias to scene://name). "
        "Inside a predicate argument, reference an entity by its bare binding alias "
        "(for example \"subject\": \"object\" when bindings maps object to "
        "scene://<top_level_object_name>) -- never repeat the scene:// URI as the "
        "argument value. Never copy example names; use only names from scene_facts. "
        "Every explicit completion clause in the task description must be represented "
        "by observable GoalSpec predicates: grasp/pick/carry requires attached, "
        "release requires released, stable/settled/stop requires settled, and every/all/each "
        "named movable object must have its own terminal placement predicate. Do not emit "
        "attached for an instruction that explicitly says without/no grasping. "
    )


def _user_prompt(request: GoalPlanningRequest) -> str:
    return json.dumps(
        {
            "task_description": request.task_description,
            "scene_facts": request.scene_facts,
            # Keep the small, exact region table beside the large scene dump.
            # Providers often lose a local-frame center or round a size when
            # they have to recover it from the full facts object.  This is
            # still grounding-only: the table is copied from SceneModel and
            # does not invent a target or alter the goal semantics.
            "canonical_regions": _canonical_regions(request),
            "output_rules": [
                "Return only schema_version, bindings, required, invariants.",
                "scene:// URIs appear only as the values of bindings; bind each alias to an existing scene:// object name.",
                "Predicate arguments reference entities by their bare binding alias (e.g. \"subject\": \"object\"), never by the scene:// URI itself.",
                "Each predicate object must contain exactly the keys \"predicate\" and \"arguments\"; never use \"args\".",
                "The arguments value must be a JSON object, never an array; schema_version must be the integer 1.",
                "Use canonical predicate argument names: subject for target entities; attached and released optionally accept a concrete effector identifier but should omit it when the physical end-effector is agnostic; robot/base/mobile_base are invalid there as effectors; base_at_pose optionally accepts subject only as the exact string robot, base, or mobile_base; never use the slash-separated value robot/base/mobile_base.",
                "inside_region.region must be an object (shape/center/size or radius/height), never a name string.",
                "For inside_region, copy the exact region geometry from canonical_regions for the bound reference object; do not round, translate, or invent it.",
                "never use object instead of subject",
                "on_support requires subject, support, surface, and subject_half_height_m; "
                "surface must be an object copied from scene_facts.surfaces with exactly "
                "center and size fields, never a name string.",
                "Use numeric geometry only from scene_facts; do not invent surface or tolerance values.",
                "Use contact only for pairs covered by scene_facts.contact_sensors; use collision_free only for subjects covered by scene_facts.collision_sensors; omit predicates without observable coverage.",
                "For an explicit physical-contact or push clause, emit a contact predicate involving the named movable object and the matching contact-sensor endpoint; the robot/base/mobile_base semantic endpoint or the exact sensor body name is allowed.",
                "Represent every explicit completion clause: grasp/pick/carry -> attached; release -> released; stable/settled/stop -> settled; every/all/each named movable object -> its own terminal placement predicate. Do not emit attached for an explicit without/no-grasp instruction.",
                "Do not emit skill, action, plan, order, evaluator, or repair fields.",
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _repair_prompt(
    request: GoalPlanningRequest,
    error: str,
    previous: Mapping[str, Any] | None,
) -> str:
    # Region geometry is authored in the scene model and must be copied
    # exactly into an inside_region predicate.  Providers sometimes return a
    # region label or round/translate the local-frame geometry even after the
    # first schema correction.  Expose a compact, generated grounding table
    # in the bounded retry so the provider can repair its JSON without us
    # inventing a task-specific target or silently changing the frozen goal.
    return json.dumps(
        {
            "task_description": request.task_description,
            "scene_facts": request.scene_facts,
            "canonical_regions": _canonical_regions(request),
            "previous_response": previous,
                "schema_correction": (
                "Return only the four GoalSpec fields from the system prompt. "
                "Every predicate object must use exactly \"predicate\" and \"arguments\" "
                "keys (never \"args\"); arguments must be a JSON object, never an array; "
                "schema_version must be the integer 1. "
                "Only use contact/collision_free when the corresponding "
                "scene_facts sensors cover the referenced entities; omit any "
                "predicate without observable coverage. "
                "For inside_region, copy the exact shape/center/size (or "
                "radius/height) object from canonical_regions for the bound "
                "reference object; the region is in the reference object's "
                "local frame, and the region name is not a valid replacement "
                "for that geometry. "
                "For released/attached, an optional effector value must identify a concrete end-effector; robot/base/mobile_base are invalid there, and omit it when the physical end-effector is agnostic. For base_at_pose, pose is required and subject is optional only when its value is exactly robot, base, or mobile_base; never emit the slash-separated value robot/base/mobile_base. "
                "The task description's explicit completion clauses must all be present: grasp/pick/carry requires attached, release requires released, stable/settled/stop requires settled, and every/all/each named movable object requires its own terminal placement predicate. Do not add attached when the instruction says without/no grasping. "
                "An explicit physical-contact or push clause also requires an observable contact predicate involving the named movable object. "
                "Correct the local validation error without adding an action or repair recipe: "
                + error[:1000]
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _canonical_regions(request: GoalPlanningRequest) -> dict[str, list[dict[str, Any]]]:
    """Return the exact declared region geometry in a compact provider table."""
    canonical_regions: dict[str, list[dict[str, Any]]] = {}
    for obj in request.scene.objects:
        regions: list[dict[str, Any]] = []
        for region in obj.regions:
            item: dict[str, Any] = {
                "name": region.name,
                "shape": region.shape.value,
                "center": list(region.center),
            }
            if region.shape.value == "cuboid":
                item["size"] = list(region.size or ())
            else:
                item["radius"] = region.radius
                item["height"] = region.height
            regions.append(item)
        if regions:
            canonical_regions[obj.name] = regions
    return canonical_regions


__all__ = ["GoalPlanner", "GoalPlanningRequest", "GoalPlanningResult"]
