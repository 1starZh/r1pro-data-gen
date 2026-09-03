"""Prompting and validation orchestration for an external task-planning LLM."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.contracts import (
    LLM_SCHEMA_VERSION,
    LLMPlanValidationError,
    parse_json_object,
    validate_envelope,
)
from ..context.facts import object_names
from .interfaces import TaskPlanningRequest, TaskPlanningResult
from ..llm.providers.protocol import ProviderError, TaskPlanningProvider
from r1pro_data_gen.domain import Plan


class LLMTaskPlanner:
    """Generate a semantic Plan, never a trajectory or executable function."""

    name = "llm_task_planner"

    def __init__(
        self,
        provider: TaskPlanningProvider,
        *,
        registry: Any = None,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1 or max_attempts > 2:
            raise ValueError("max_attempts must be 1 or 2")
        self.provider = provider
        self.registry = registry
        self.max_attempts = max_attempts
        self.model = provider.model

    def plan(self, request: TaskPlanningRequest) -> TaskPlanningResult:
        """Call the provider with canonical facts and validate the response."""
        system = _system_prompt(request.skill_catalog)
        user = _user_prompt(request)
        last_error = ""
        previous_envelope: Mapping[str, Any] | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.provider.complete(system=system, user=user)
                envelope = parse_json_object(response.text)
                previous_envelope = envelope
                plan = validate_envelope(
                    envelope,
                    skill_catalog=request.skill_catalog,
                    registry=self.registry,
                    scene_object_names=object_names(request.scene_facts),
                )
                if plan is None:
                    return TaskPlanningResult(
                        status="unsupported",
                        reason=str(envelope.get("reason", "unsupported")),
                        provider=response.provider,
                        model=response.model,
                        raw_response=envelope,
                        usage=response.usage,
                    )
                _validate_goal_spec_binding(plan, request)
                _validate_gripper_alignment_precondition(plan)
                _validate_runtime_repair(plan, request)
                _validate_complete_transfer_plan(plan, request)
                plan = Plan(
                    task_name=plan.task_name,
                    stages=plan.stages,
                    metadata={
                        **plan.metadata,
                        "source": "external_llm",
                        "provider": response.provider,
                        "model": response.model,
                        "schema_version": LLM_SCHEMA_VERSION,
                        "task_description": request.task_description,
                        **(
                            {"goal_contract_hash": request.goal_contract_hash}
                            if request.goal_contract_hash is not None
                            else {}
                        ),
                    },
                )
                return TaskPlanningResult(
                    status="planned",
                    plan=plan,
                    provider=response.provider,
                    model=response.model,
                    raw_response=envelope,
                    usage=response.usage,
                )
            except (ProviderError, LLMPlanValidationError, ValueError, TypeError) as exc:
                last_error = str(exc)
                if attempt + 1 >= self.max_attempts:
                    break
                # The second attempt is still bounded and receives no execution
                # feedback, so it cannot become an uncontrolled re-planning loop.
                # Include the rejected envelope and exact dependency repairs: a
                # generic reminder about depends_on was repeatedly ignored by
                # deepseek-chat when it regenerated an otherwise valid plan.
                user = _repair_prompt(
                    request,
                    last_error,
                    previous_envelope=previous_envelope,
                )
        return TaskPlanningResult(
            status="failed",
            reason=last_error or "LLM planner failed",
            provider=getattr(self.provider, "name", "unknown"),
            model=getattr(self.provider, "model", "unknown"),
        )


def _system_prompt(skill_catalog: Sequence[Mapping[str, Any]]) -> str:
    catalog = json.dumps(
        list(skill_catalog), ensure_ascii=False, separators=(",", ":")
    )
    example = json.dumps(
        {
            "schema_version": LLM_SCHEMA_VERSION,
            "status": "planned",
            "reason": "",
            "plan": {
                "task_name": "short_task_name",
                "stages": [
                    {
                        "name": "approach_target",
                        "goal": "reach a pregrasp stance",
                        "depends_on": [],
                        "parameters": {
                            "skill": "base_navigate_to",
                            "target_ref": "scene://object_name_from_scene",
                            "purpose": "pregrasp",
                        },
                        "outputs": [],
                        "preconditions": [],
                        "postconditions": [],
                    },
                    {
                        "name": "grasp_target",
                        "goal": "attach the named object",
                        "depends_on": ["approach_target"],
                        "parameters": {
                            "skill": "grasp_object",
                            "object_name": "object_name_from_scene",
                            "side": "auto",
                        },
                        "outputs": [],
                        "preconditions": [],
                        "postconditions": [],
                    },
                ],
                "metadata": {},
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        # --- Section 1: output envelope contract ---------------------------
        "You are a constrained robot task planner for a mobile dual-arm "
        "robot. Convert the natural-language task and the structured scene "
        "facts into an explicit multi-stage plan of public skill calls. "
        "Return exactly one bare JSON object and nothing else: no Markdown "
        "fences, commentary, alternate schema, or extra top-level fields. The "
        "only accepted planned-response shape is schema_version, status, "
        "reason, and plan; plan contains task_name, stages, and metadata. Each "
        "stage contains name, goal, depends_on, parameters, and may contain "
        "outputs, preconditions, and postconditions. outputs is an array of "
        "strings from {position, quaternion, contact_forces, joint_positions}; "
        "preconditions and postconditions, when present, are arrays of objects "
        "each shaped exactly {\"predicate\": \"contact_detected\"|"
        "\"reference_available\"|\"within_tolerance\", \"parameters\": {}} -- "
        "never plain strings; prefer [] over invented shapes. The skill name MUST be "
        "parameters.skill, never a stage-level skill field; put every skill "
        "argument in parameters and never use args or dependencies. Stage "
        "dependencies may reference earlier stages only. Use only skills and "
        "parameter names listed in the catalogue. Decompose the natural-language "
        "task into multiple explicit semantic stages whenever it needs more "
        "than one action; every stage has a concrete goal and one public "
        "generic skill. Stay within the validator's 16-stage limit; prefer the "
        "minimum complete sequence without redundant query or bookkeeping "
        "stages. The JSON template below is schema-only and is not an "
        "execution recipe. "
        # --- Section 2: typed references and value flow ---------------------
        "A typed runtime reference is the parameter value itself as one JSON "
        "object; never wrap it in a one-element array and never nest it under "
        "an extra key such as 'reference'. Canonical shape: {\"ref\": "
        "\"stage.observe_object.details.position\", \"value_type\": \"array\", "
        "\"shape\": [3], \"frame\": \"base\", \"offset\": [0.0, 0.0, 0.1]} "
        "where value_type is one of \"number\"|\"array\"|\"object\" and frame "
        "is one of \"world\"|\"base\". References are "
        "data-only: no expressions, attribute escapes, indexing, or calls. "
        "Reference sources are limited to stage.<name>.details.<declared_output>, "
        "observation.base_pose, and scene.object.<name>.position/quaternion; "
        "other scene-facts fields such as navigation candidate lists are not "
        "reference paths, so copy a chosen fact as a literal parameter instead. "
        "Skill parameters that name scene entities (object_name, "
        "target_region_name, support_surface_name) must use the top-level object "
        "names from scene facts.objects; names nested under an object's regions "
        "or surfaces are goal-predicate geometry and are never valid skill "
        "parameters. "
        "For arm_carry_object_to, target_region_name identifies the destination "
        "marker/object and support_surface_name identifies the physical object "
        "supporting that destination; it must not be the source support from "
        "which the object was picked, because the carry skill infers source "
        "support internally and uses this parameter to compute destination "
        "height. "
        "For pick-and-place, emit separate public skills in order: "
        "base_navigate_to(purpose=pregrasp), optionally prepare_workspace when "
        "the grasp height is wrong or the object is on the floor, grasp_object, "
        "arm_carry_object_to, release_object. There is one public grasp name. "
        "Do not emit transfer_object_between_supports, "
        "whole_body_transfer_object_between_supports, support_aware_grasp_object, "
        "torso_move_to, or query skills; those are internal. After attachment, "
        "do not navigate to the current support; use arm_carry_object_to. "
        "purpose=dropoff is only for a different support. "
        "Keep coordinate frames explicit: never copy a "
        "world-frame coordinate directly into a base-frame parameter; declare "
        "the reference with its frame and let the runtime resolver convert. "
        # --- Section 3: planning methodology --------------------------------
        "First identify the relevant entities, support surfaces, destination "
        "regions, and goal state from the supplied task and scene facts. Live "
        "object poses and contacts are already in the observation; do not call "
        "query skills. Structure physical interaction as "
        "action, verification, and recovery stages according to "
        "the public skill contracts. Choose safe, collision-aware, reachable "
        "poses from declared geometry and keep coordinate frames explicit. Use "
        "only values supported by the task, scene facts, observations, and "
        "skill catalogue; do not invent task-specific recipes, calibration "
        "constants, or tuning numbers. Prefer a semantic public skill over "
        "manual backend details and let references read live state rather than "
        "stale copied coordinates. For base navigation to a scene entity, use "
        "base_navigate_to with target_ref='scene://<object>' and a purpose such "
        "as navigation, pregrasp, dropoff, or staging. The runtime resolves "
        "that semantic request into a collision-free, reachable approach pose. "
        "A literal target pose may be supplied only as a preferred pose or when "
        "the instruction explicitly requires an exact world pose; do not treat "
        "a guessed coordinate as the task goal. "
        # --- Section 4: physical interaction doctrine ------------------------
        "Plan physical interaction in phases that keep the robot safe and the "
        "target reachable: approach an object on a support surface from a "
        "non-contact standoff clearly above it before descending (a small "
        "offset lets the gripper collide with the object or its support, and a "
        "single long move straight to the object can cross the support edge "
        "and fail planning), prefer measured alignment (arm_align_gripper) "
        "when the object's exact grasp point matters, then establish contact, "
        "verify the grasp holds, and only then carry the object clear of its "
        "surfaces. When a measured alignment precedes a grasp, always set "
        "require_between_fingers=true and require_vertical_alignment=true so "
        "the object is brought into the jaw window and down to grasp height in "
        "one measured loop; alignment success without the object between the "
        "fingers cannot support a pinch. Never command a grasped object "
        "through its support surface "
        "or hold it at an unreachable height while moving. A measured "
        "alignment or short local descent onto a support surface should "
        "exclude that support surface (and the target object itself) from its "
        "obstacle set -- these are controlled local corrections, not "
        "traversals, and keeping the support as an obstacle makes the "
        "short descent's IK fail right above the surface. The pre-grasp "
        "standoff approach itself is not a local contact correction: keep the "
        "support surface and other physical obstacles in its obstacle set, "
        "and reserve those exclusions for the subsequent measured alignment "
        "or contact descent stage. "
        "Before arm_align_gripper is used with require_between_fingers=true, "
        "explicitly open the same gripper side with gripper_set (open_value "
        "> 0) earlier in the plan; do not rely on reset state or an implicit "
        "opening. This is a generic gripper-state precondition, and the local "
        "validator rejects a measured pinch alignment that has no preceding "
        "open command. "
        "standoff must be high enough that the grasp-pose target is "
        "collision-free, but not so high that the measured alignment cannot "
        "descend to the object within its iteration budget; a standoff of a "
        "few tens of centimetres is the usual range, and request enough "
        "alignment iterations when the standoff is large. When a standoff "
        "precedes arm_align_gripper and its target is expressed relative to "
        "the observed object, write target_frame=grasp_center explicitly; "
        "never rely on arm_move_to's default ee frame for a measured grasp "
        "alignment, because the model EE origin and the live finger midpoint "
        "are not interchangeable. "
        # --- Section 5: interpreting execution feedback ----------------------
        "If constraints.failure_feedback or constraints.active_runtime_feedback "
        "is present, parse each item as the structured fact_feedback.v1 contract. "
        "The contract contains request, observations, discrepancies, and "
        "completed_prefix, plus the immutable GoalSpec hash and evidence refs. "
        "Treat these fields as informational evidence only: they describe what "
        "was requested and observed, and must not be treated as an action "
        "prescription. Do not infer unverified causes as facts, do not emit a "
        "repair recipe, and do not select a next skill because feedback names "
        "one. Re-plan from the unchanged GoalSpec, scene facts, task text, and "
        "public skill catalogue. A paired discrepancy compares an observed "
        "error against the skill's declared tolerance; an observation with "
        "position_reachable_without_orientation=true means the base pose is "
        "workable and only the commanded orientation must be relaxed or "
        "rotated, while position_reachable_without_orientation=false means "
        "the base is too far away and must move "
        "closer or select another reachable approach before the arm can work; "
        "changing only arm side, wrist orientation, or IK budget is not a base "
        "repair. "
        "A previous stage is not valid merely because its skill returned "
        "success: if a later measured arm_align_gripper result reports "
        "contact_not_centered, object_window_not_reached, or a vertical "
        "alignment error outside tolerance, treat the preceding non-contact "
        "standoff as unresolved and re-observe/re-resolve it. Preserve the "
        "goal and collision semantics, but do not blindly reuse a standoff "
        "target_frame or stale offset that the measured alignment disproves; "
        "a grasp-center standoff is the generic position-first choice when a "
        "measured gripper alignment follows, and make that frame explicit in "
        "the stage parameters. "
        "When vertical_error_m is still materially above its declared "
        "vertical_tolerance_m, a small cosmetic offset change is not evidence "
        "of repair: use the observed direction and magnitude to choose a "
        "materially closer non-contact approach or another bounded local "
        "motion that remains collision-safe, then let the alignment gate "
        "re-measure it. Do not treat a one-sided contact force as a successful "
        "grasp and do not claim convergence until between_fingers and all "
        "requested alignment gates are observed. If one-sided contact repeats "
        "after a materially different standoff, do not keep perturbing the "
        "same vertical target: re-observe the live geometry and choose a "
        "different collision-free reachable approach pose or arm side only "
        "when the scene facts support it; retain fail-closed contact gates. "
        "When the first contact_not_centered failure leaves vertical error "
        "materially above tolerance, the next plan must make at least one "
        "independently observable approach change (a different reachable "
        "approach pose or direction, arm side, or freshly measured "
        "non-contact grasp-center target); changing only search budget or a "
        "cosmetic vertical offset is not a new approach. Preserve the same "
        "safety and alignment predicates while testing that change. "
        "If a non-contact arm_move_to approach with target_frame=grasp_center "
        "reports planning_status=no_collision_free_path, treat that target as "
        "unverified: do not lower its standoff, remove support/obstacle "
        "exclusions, or claim that a larger planning budget repaired it. "
        "Preserve the grasp-center frame and choose a materially higher "
        "collision-free standoff or another fact-supported reachable approach "
        "pose/arm/base direction, then let the runtime planner re-certify the "
        "path. "
        "If a carry or waypoint stage reports no IK candidate at the final "
        "descent, do not assume that more search budget alone repairs it: use "
        "the reported target diagnostics to choose another reachable point "
        "inside the declared destination region, a supported orientation, or "
        "a different arm/approach, then let the collision and GoalSpec gates "
        "re-check the placement. Never release above the support or weaken the "
        "final placement predicates merely to make IK succeed. "
        # --- Section 5: bounded re-planning discipline -------------------------
        "When re-planning after a failure, preserve completed semantic work when "
        "it remains valid, re-observe live state before using new coordinates, "
        "and change only what is justified by the unchanged goal and available "
        "facts. If constraints.previous_plan is present, it is the exact last "
        "validated Plan; copy its valid stages and parameters first, then make "
        "the smallest evidence-backed edit. Do not delete a previously valid "
        "approach, observation, or standoff stage merely to shorten the plan. "
        "If the reported failure is trajectory tracking or execution after a "
        "safe target was planned, preserve safety-critical semantics such as "
        "target_frame, collision exclusions, measured-alignment gates, and a "
        "non-contact standoff; adjust only evidence-supported planning budget, "
        "IK branch count, speed, or a fresh observation. Do not lower a "
        "standoff or switch target frames just to hide a tracking failure. "
        "Never weaken safety, validation, or final completion criteria. "
        # --- Section 6: boundaries ---------------------------------------------
        "Do not use run_registered_task, task_id, variant, task policy names, "
        "or any task-specific executor. You produce semantic Plan stages only "
        "- never Python, adapter calls, joint trajectories, velocity commands, "
        "or hidden backend skills. Do not emit the optional direct-plan "
        "compatibility form; always emit the envelope below. If the task "
        "cannot be represented safely using the public catalogue and supported "
        "references, return status unsupported with plan null and a reason; "
        "unsupported is not a substitute for attempting the task.\n"
        f"Required JSON template (replace placeholder values):\n{example}\n"
        f"The required envelope schema version is {LLM_SCHEMA_VERSION}.\n"
        "Skill catalogue:\n"
        f"{catalog}"
    )


def _user_prompt(request: TaskPlanningRequest) -> str:
    payload = {
        "task_description": request.task_description,
        "scene_facts": request.scene_facts,
        "constraints": request.constraints,
        "metadata": request.metadata,
        "output_rules": [
            "Return the exact envelope shape from the system prompt.",
            "Put the selected skill in stage.parameters.skill.",
            "Put all skill arguments beside skill in stage.parameters.",
            "Represent a typed reference as one object at the parameter value, never as a one-element array.",
            "Use only stage.<name>.details.<declared_output>, observation.base_pose, or scene.object.<name>.position/quaternion as reference sources.",
            "Use [] for depends_on on the first stage.",
            "Do not invent fields such as args, dependencies, or stage.skill.",
            "If constraints.previous_plan is present, preserve its valid stages and parameters and edit only what the feedback justifies.",
            "Before arm_align_gripper with require_between_fingers=true, include a preceding gripper_set with open_value > 0 for the same side.",
        ],
    }
    if request.goal_spec is not None:
        payload["goal_spec"] = request.goal_spec
        payload["goal_spec_hash"] = request.goal_spec_hash
        payload["output_rules"].append(
            "Preserve the frozen goal_spec_hash exactly in plan.metadata.goal_spec_hash."
        )
        if request.goal_contract_hash is not None:
            payload["goal_contract_hash"] = request.goal_contract_hash
            payload["output_rules"].append(
                "Preserve the frozen goal_contract_hash exactly in plan.metadata.goal_contract_hash."
            )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _validate_goal_spec_binding(
    plan: Plan,
    request: TaskPlanningRequest,
) -> None:
    """Require generic plans to remain bound to the frozen completion contract."""
    if request.goal_spec is None:
        return
    expected = request.goal_spec_hash
    actual = plan.metadata.get("goal_spec_hash")
    if not isinstance(expected, str) or not expected:
        raise LLMPlanValidationError("generic planning requires a non-empty goal_spec_hash")
    if actual != expected:
        raise LLMPlanValidationError(
            "plan.metadata.goal_spec_hash must match the frozen goal_spec_hash"
        )
    expected_contract = request.goal_contract_hash
    if expected_contract is not None and plan.metadata.get("goal_contract_hash") != expected_contract:
        raise LLMPlanValidationError(
            "plan.metadata.goal_contract_hash must match the frozen goal_contract_hash"
        )


def _validate_complete_transfer_plan(
    plan: Plan,
    request: TaskPlanningRequest,
) -> None:
    """Reject a partial task Plan when the frozen goal is a full transfer.

    The hosted closed-loop agent may legitimately emit one action at a time
    and replan after each observation.  This validator is for the separate
    task-level Plan contract: such a Plan must either call the complete
    transfer skill or explicitly contain grasp, carry/place, and release.
    """
    goal = request.goal_spec
    if not isinstance(goal, Mapping):
        return
    required = goal.get("required", ())
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return
    predicates = {
        item.get("predicate")
        for item in required
        if isinstance(item, Mapping)
    }
    is_full_transfer = bool(
        {"attached", "released"}.issubset(predicates)
        and ("inside_region" in predicates or "on_support" in predicates)
    )
    if not is_full_transfer:
        return
    skills = [skill for skill, _ in _plan_stage_records(plan)]
    if not ({"grasp_object"} & set(skills)):
        raise LLMPlanValidationError(
            "full transfer GoalSpec requires a grasp_object phase"
        )
    if "arm_carry_object_to" not in skills:
        raise LLMPlanValidationError(
            "full transfer GoalSpec requires an arm_carry_object_to phase"
        )
    if "release_object" not in skills:
        raise LLMPlanValidationError(
            "full transfer GoalSpec requires a release_object phase"
        )


def _validate_gripper_alignment_precondition(plan: Plan) -> None:
    """Require an explicit open command before a measured pinch alignment.

    ``arm_align_gripper`` measures the live object window and is intended to
    position an object between an open pair of fingers.  Reset state is not a
    portable precondition across scenes or retries, so an external LLM plan
    must establish the opening for the same arm side first.  This is a
    capability-level contract; it contains no object name, coordinate, or
    pick-and-place recipe.
    """
    open_sides: set[str] = set()
    for skill, parameters in _plan_stage_records(plan):
        side = parameters.get("side", "left")
        if not isinstance(side, str):
            continue
        if skill == "arm_align_gripper" and bool(parameters.get("require_between_fingers", False)):
            if side not in open_sides:
                raise LLMPlanValidationError(
                    "arm_align_gripper with require_between_fingers=true must be "
                    "preceded by gripper_set with open_value > 0 for the same side"
                )
        elif skill == "gripper_set":
            try:
                open_value = float(parameters.get("open_value", 0.05))
            except (TypeError, ValueError):
                open_value = 0.0
            if math.isfinite(open_value) and open_value > 0.0:
                open_sides.add(side)
            else:
                open_sides.discard(side)
        elif skill == "gripper_grasp":
            open_sides.discard(side)


def _validate_runtime_repair(plan: Plan, request: TaskPlanningRequest) -> None:
    """Reject cosmetic retries after a measured one-sided contact.

    This is a generic plan contract, not a task recipe: when the last physical
    evidence says the gripper touched one side while vertical error remains
    materially outside tolerance, a new plan must test an independently
    observable approach change.  A few centimetres of z tuning or extra search
    budget alone cannot establish that the approach changed.  The LLM receives
    this local validation error and gets one bounded repair call.
    """
    constraints = request.constraints
    if not isinstance(constraints, Mapping):
        return
    previous = constraints.get("previous_plan")
    if not isinstance(previous, Mapping):
        return
    feedback = _latest_feedback(constraints)
    if feedback is None:
        return
    if _requires_approach_change(feedback):
        if _has_independent_approach_change(previous, plan):
            return
        raise LLMPlanValidationError(
            "runtime repair after contact_not_centered with vertical error above "
            "tolerance must change an independently observable approach (reachable "
            "approach pose or direction, arm side, or a fresh non-contact target); "
            "changing only search budget or a cosmetic vertical offset is invalid"
        )
    if _requires_standoff_repair(feedback):
        if _has_standoff_repair_change(previous, plan):
            return
        raise LLMPlanValidationError(
            "runtime repair after a grasp_center arm_move_to with "
            "no_collision_free_path must preserve or increase non-contact clearance "
            "or change an independently observable approach; lowering the standoff "
            "or changing only search budget is invalid"
        )
    if _requires_base_repair(feedback):
        if _has_base_approach_change(previous, plan):
            return
        raise LLMPlanValidationError(
            "runtime repair after position_reachable_without_orientation=false "
            "must change the base approach or navigation candidate before retrying "
            "the arm; changing only arm side, orientation, or IK budget is invalid"
        )


def _latest_feedback(constraints: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Read bounded feedback, retaining any unresolved physical repair gate.

    A schema/reference rejection can follow a GPU failure before a valid Plan
    is ever replayed.  Treating that validator item as the sole "latest"
    feedback would let a cosmetic retry bypass the earlier physical gate.
    Prefer the newest feedback that still requires a runtime repair, then fall
    back to the newest parsed item for ordinary prompt context.
    """
    values: list[Any] = []
    for key in ("failure_feedback", "active_runtime_feedback"):
        raw = constraints.get(key, ())
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(raw)
    parsed_values: list[Mapping[str, Any]] = []
    for value in reversed(values):
        if isinstance(value, Mapping):
            parsed_values.append(value)
            continue
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, Mapping):
            parsed_values.append(parsed)
    for feedback in parsed_values:
        if (
            _requires_approach_change(feedback)
            or _requires_standoff_repair(feedback)
            or _requires_base_repair(feedback)
        ):
            return feedback
    return parsed_values[0] if parsed_values else None


def _requires_approach_change(feedback: Mapping[str, Any]) -> bool:
    observations = feedback.get("observations")
    if not isinstance(observations, Mapping):
        return False
    if observations.get("failure_code") != "contact_not_centered":
        return False
    try:
        vertical = float(observations.get("vertical_error_m"))
        tolerance = float(observations.get("vertical_tolerance_m"))
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(vertical) and math.isfinite(tolerance) and tolerance > 0.0):
        return False
    # "Materially" is deliberately expressed relative to the skill contract,
    # not a scene/object constant.  Either a 2 cm absolute miss or 1.5x the
    # declared tolerance is enough to rule out a cosmetic retry.
    return vertical > max(tolerance + 0.02, tolerance * 1.5)


def _requires_standoff_repair(feedback: Mapping[str, Any]) -> bool:
    """Recognize a failed, non-contact grasp-center path certification."""
    if feedback.get("skill") != "arm_move_to":
        return False
    request = feedback.get("request")
    if not isinstance(request, Mapping) or request.get("target_frame") != "grasp_center":
        return False
    observations = feedback.get("observations")
    if not isinstance(observations, Mapping):
        return False
    if observations.get("failure_type") != "gpu":
        return False
    details = observations.get("stage_details")
    if not isinstance(details, Mapping):
        return False
    return details.get("planning_status") == "no_collision_free_path"


def _requires_base_repair(feedback: Mapping[str, Any]) -> bool:
    """Recognize an arm target whose position is unreachable from the base pose."""
    if feedback.get("skill") != "arm_move_to":
        return False
    observations = feedback.get("observations")
    if not isinstance(observations, Mapping):
        return False
    return observations.get("position_reachable_without_orientation") is False


def _plan_stage_records(plan: Any) -> list[tuple[str, Mapping[str, Any]]]:
    raw_stages = plan.get("stages", ()) if isinstance(plan, Mapping) else getattr(plan, "stages", ())
    records: list[tuple[str, Mapping[str, Any]]] = []
    for stage in raw_stages if isinstance(raw_stages, Sequence) else ():
        if isinstance(stage, Mapping):
            name = stage.get("name")
            parameters = stage.get("parameters", {})
        else:
            name = getattr(stage, "name", None)
            parameters = getattr(stage, "parameters", {})
        if isinstance(name, str) and isinstance(parameters, Mapping):
            skill = parameters.get("skill")
            if isinstance(skill, str):
                records.append((skill, parameters))
    return records


def _has_independent_approach_change(previous: Any, current: Plan) -> bool:
    previous_records = _plan_stage_records(previous)
    current_records = _plan_stage_records(current)
    previous_nav = next((p for s, p in previous_records if s == "base_navigate_to"), None)
    current_nav = next((p for s, p in current_records if s == "base_navigate_to"), None)
    if previous_nav is not None and current_nav is not None:
        for key in ("approach_side", "target_ref", "target", "preferred_pose", "purpose"):
            if key in current_nav and current_nav.get(key) is not None and (
                key not in previous_nav or current_nav[key] != previous_nav.get(key)
            ):
                return True
    previous_align = next((p for s, p in previous_records if s == "arm_align_gripper"), None)
    current_align = next((p for s, p in current_records if s == "arm_align_gripper"), None)
    if previous_align is not None and current_align is not None:
        if (
            "side" in previous_align
            and "side" in current_align
            and current_align["side"] != previous_align["side"]
        ):
            return True
    previous_moves = [p for s, p in previous_records if s == "arm_move_to"]
    current_moves = [p for s, p in current_records if s == "arm_move_to"]
    for old, new in zip(previous_moves, current_moves):
        if (
            "side" in old
            and "side" in new
            and new["side"] != old["side"]
        ):
            return True
        if (
            "target_frame" in old
            and "target_frame" in new
            and new["target_frame"] != old["target_frame"]
        ):
            return True
        old_target = old.get("target_pos")
        new_target = new.get("target_pos")
        if old_target is None or new_target is None:
            if old_target is None and new_target is not None:
                return True
            continue
        if isinstance(old_target, Mapping) and isinstance(new_target, Mapping):
            if (
                "ref" in old_target
                and "ref" in new_target
                and new_target["ref"] != old_target["ref"]
            ):
                return True
            old_offset = old_target.get("offset")
            new_offset = new_target.get("offset")
            if old_offset is not None and new_offset is not None and _xy_offset_changed(old_offset, new_offset):
                return True
            if old_offset is None and new_offset is not None:
                return True
        elif old_target != new_target:
            return True
    # Adding a distinct directional/rotation approach stage is independently
    # observable even when the original navigation stage is retained.
    previous_approach_skills = {s for s, _ in previous_records if s in {"base_rotate_to", "base_move_to", "arm_move_directional"}}
    current_approach_skills = {s for s, _ in current_records if s in {"base_rotate_to", "base_move_to", "arm_move_directional"}}
    return bool(current_approach_skills - previous_approach_skills)


def _has_standoff_repair_change(previous: Any, current: Plan) -> bool:
    """Accept an independently changed approach or a materially higher standoff."""
    if _has_independent_approach_change(previous, current):
        return True
    previous_moves = [p for s, p in _plan_stage_records(previous) if s == "arm_move_to"]
    current_moves = [p for s, p in _plan_stage_records(current) if s == "arm_move_to"]
    for old, new in zip(previous_moves, current_moves):
        if old.get("target_frame") != "grasp_center" or new.get("target_frame") != "grasp_center":
            continue
        old_z = _target_z(old.get("target_pos"))
        new_z = _target_z(new.get("target_pos"))
        if old_z is not None and new_z is not None and new_z - old_z >= 0.02:
            return True
    return False


def _has_base_approach_change(previous: Any, current: Plan) -> bool:
    """Require a navigation/base-direction change for a position-unreachable arm."""
    previous_records = _plan_stage_records(previous)
    current_records = _plan_stage_records(current)
    previous_nav = next((p for s, p in previous_records if s == "base_navigate_to"), None)
    current_nav = next((p for s, p in current_records if s == "base_navigate_to"), None)
    if previous_nav is not None and current_nav is not None:
        for key in ("approach_side", "target_ref", "target", "preferred_pose", "purpose"):
            if key in current_nav and current_nav.get(key) is not None and (
                key not in previous_nav or current_nav[key] != previous_nav.get(key)
            ):
                return True
    previous_base_skills = {s for s, _ in previous_records if s in {"base_navigate_to", "base_move_to", "base_rotate_to"}}
    current_base_skills = {s for s, _ in current_records if s in {"base_navigate_to", "base_move_to", "base_rotate_to"}}
    return bool(current_base_skills - previous_base_skills)


def _target_z(target: Any) -> float | None:
    if isinstance(target, Mapping):
        offset = target.get("offset")
        if isinstance(offset, Sequence) and not isinstance(offset, (str, bytes)) and len(offset) >= 3:
            try:
                return float(offset[2])
            except (TypeError, ValueError):
                return None
        value = target.get("value")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
            try:
                return float(value[2])
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)) and len(target) >= 3:
        try:
            return float(target[2])
        except (TypeError, ValueError):
            return None
    return None


def _xy_offset_changed(old: Any, new: Any, *, threshold: float = 0.02) -> bool:
    if not isinstance(old, Sequence) or isinstance(old, (str, bytes)):
        return old != new
    if not isinstance(new, Sequence) or isinstance(new, (str, bytes)):
        return True
    if len(old) < 2 or len(new) < 2:
        return old != new
    try:
        return any(abs(float(new[i]) - float(old[i])) > threshold for i in (0, 1))
    except (TypeError, ValueError):
        return old != new


def _repair_prompt(
    request: TaskPlanningRequest,
    error: str,
    *,
    previous_envelope: Mapping[str, Any] | None = None,
) -> str:
    repair_actions = _reference_dependency_repairs(previous_envelope)
    runtime_repair_contract = _runtime_repair_contract(request, error)
    runtime_repair_guidance = ""
    if "position_reachable_without_orientation=false" in error:
        runtime_repair_guidance = (
            " This is a base-position reachability repair: change the existing "
            "base_navigate_to approach_side, target/target_ref/preferred_pose, or purpose, or "
            "add a fact-supported base_move_to/base_rotate_to stage before the "
            "arm stage. Changing only side, wrist orientation, standoff, or IK "
            "budget is not a base approach change."
        )
    elif "contact_not_centered" in error or "independently observable approach" in error:
        runtime_repair_guidance = (
            " This is a measured approach repair: change a reachable approach "
            "pose/direction, arm side, or freshly measured non-contact target; "
            "do not only change search budget or make a cosmetic z edit."
        )
    elif "no_collision_free_path" in error:
        runtime_repair_guidance = (
            " This is a non-contact clearance repair: preserve the grasp_center "
            "frame and increase clearance or choose another fact-supported "
            "reachable approach; do not lower the standoff or only increase "
            "search budget."
        )
    payload = {
        "task_description": request.task_description,
        "scene_facts": request.scene_facts,
        "constraints": request.constraints,
        "repair_instruction": (
            "The previous response was rejected by local validation. Return only "
            "a new bare JSON object using the exact envelope contract from the "
            "system prompt; do not return a direct plan. Each stage must use "
            "name, goal, depends_on, parameters, and may use outputs, "
            "preconditions, and postconditions; the selected generic skill must "
            "be inside parameters.skill. Do not use args, dependencies, "
            "stage.skill, run_registered_task, task_id, variant, or task-specific "
            "policy names. Every typed reference source must be limited to "
            "stage.<name>.details.<declared_output>, observation.base_pose, or "
            "scene.object.<name>.position/quaternion, must use only output names "
            "declared by the producing skill (omit an uncertain optional "
            "stage.outputs field instead of inventing a name), and must appear "
            "exactly in that stage's depends_on list: for every parameter "
            "reference of the form stage.S.details.F, add the exact stage name S "
            "to depends_on. A typed reference is one object at the parameter "
            "value, never a one-element array and never nested under an extra "
            "key; its keys are exactly ref, value_type (number|array|object), "
            "shape, frame (world|base), offset (optional). Make the smallest repair to the "
            "previous plan: preserve valid stages, stay at or below 16 stages, "
            "and do not add unrelated actions, recipes, calibration constants, "
            "or numeric tuning. If constraints.previous_plan is present, copy "
            "its valid stages and parameters and repair only the field named by "
            "the validation error; do not drop previously valid approach or "
            "observation stages. Return status unsupported if the task cannot be "
            "represented safely. If the validation error concerns "
            "arm_align_gripper with require_between_fingers=true, add a prior "
            "gripper_set with open_value > 0 for that same side and preserve "
            "the measured-alignment safety gates. Validation error: "
            f"{error[:1000]}"
            f"{runtime_repair_guidance}"
        ),
        "exact_repair_actions": repair_actions,
        # Keep runtime repairs machine-readable as well as prose-readable.
        # The validator remains the authority; this contract only makes the
        # required semantic diff explicit to providers that otherwise tend to
        # change a nearby arm parameter while leaving the failed base pose
        # untouched.
        "runtime_repair_contract": runtime_repair_contract,
        "previous_rejected_response": previous_envelope,
        "required_output_shape": {
            "schema_version": LLM_SCHEMA_VERSION,
            "status": "planned|unsupported",
            "reason": "string",
            "plan": "object|null",
        },
    }
    if request.goal_spec is not None:
        payload["goal_spec"] = request.goal_spec
        payload["goal_spec_hash"] = request.goal_spec_hash
        payload["repair_instruction"] += (
            " Preserve the unchanged goal_spec_hash exactly in "
            "plan.metadata.goal_spec_hash."
        )
        if request.goal_contract_hash is not None:
            payload["goal_contract_hash"] = request.goal_contract_hash
            payload["repair_instruction"] += (
                " Preserve the unchanged goal_contract_hash exactly in "
                "plan.metadata.goal_contract_hash."
            )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _runtime_repair_contract(
    request: TaskPlanningRequest,
    error: str,
) -> Mapping[str, Any] | None:
    """Describe the required semantic diff for a bounded runtime retry.

    This is deliberately a contract, not a task-specific repair recipe.  It
    names the plan-level field that must change and the fields that do not
    satisfy the gate; scene-specific candidates remain in ``scene_facts``.
    """
    constraints = request.constraints
    previous_plan = constraints.get("previous_plan") if isinstance(constraints, Mapping) else None
    previous_navigation: Mapping[str, Any] | None = None
    if isinstance(previous_plan, Mapping):
        for skill, parameters in _plan_stage_records(previous_plan):
            if skill == "base_navigate_to":
                previous_navigation = dict(parameters)
                break

    if "position_reachable_without_orientation=false" in error:
        return {
            "kind": "base_approach_change",
            "required_change": [
                "change at least one existing base_navigate_to parameter among approach_side, target_ref, target, preferred_pose, purpose",
                "or add a fact-supported base_move_to/base_rotate_to stage before the failing arm stage",
                "when a fact_supported_candidates list is present, copy one candidate pose verbatim into preferred_pose",
            ],
            "forbidden_as_sole_change": [
                "arm side",
                "wrist orientation",
                "standoff offset",
                "IK/search budget",
            ],
            "candidate_source": "scene_facts.navigation.approach_candidates",
            "fact_supported_candidates": _fact_supported_navigation_candidates(request.scene_facts),
            "previous_navigation": previous_navigation,
        }
    if "contact_not_centered" in error or "independently observable approach" in error:
        return {
            "kind": "independently_observable_approach_change",
            "required_change": [
                "change a reachable approach pose or direction, arm side, or freshly measured non-contact target",
            ],
            "forbidden_as_sole_change": ["search budget", "cosmetic vertical offset"],
            "candidate_source": "scene_facts and declared observation outputs",
        }
    if "no_collision_free_path" in error:
        return {
            "kind": "non_contact_clearance_change",
            "required_change": [
                "preserve grasp_center and increase non-contact clearance, or choose another fact-supported reachable approach",
            ],
            "forbidden_as_sole_change": ["lowering standoff", "search budget"],
            "candidate_source": "scene_facts and declared observation outputs",
        }
    return None


def _fact_supported_navigation_candidates(
    scene_facts: Mapping[str, Any],
    *,
    limit: int = 8,
) -> list[Mapping[str, Any]]:
    """Return a compact, fact-backed candidate list for a base repair prompt.

    The full navigation fact table can contain many obstacle-boundary samples.
    A bounded summary keeps the repair actionable without inventing a pose: a
    candidate is copied verbatim from the scene facts and is only a suggestion
    for the LLM's semantic Plan, never an executor-side repair.
    """
    navigation = scene_facts.get("navigation") if isinstance(scene_facts, Mapping) else None
    raw = navigation.get("approach_candidates") if isinstance(navigation, Mapping) else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    candidates: list[tuple[bool, float, Mapping[str, Any]]] = []
    for candidate in raw:
        if not isinstance(candidate, Mapping):
            continue
        pose = candidate.get("pose")
        side = candidate.get("side")
        if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)) or len(pose) != 3:
            continue
        if not isinstance(side, str):
            continue
        probes = candidate.get("ik_reachability")
        reachable = False
        distance = float("inf")
        if isinstance(probes, Sequence) and not isinstance(probes, (str, bytes)):
            for probe in probes:
                if not isinstance(probe, Mapping):
                    continue
                reachable = reachable or probe.get("reachable") is True
                try:
                    distance = min(distance, float(probe.get("distance_m", float("inf"))))
                except (TypeError, ValueError):
                    pass
        summary = {
            "side": side,
            "preferred_pose": [float(value) for value in pose],
            "obstacle_name": candidate.get("obstacle_name"),
            "ik_reachable": bool(reachable),
        }
        candidates.append((not reachable, distance, summary))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [summary for _, _, summary in candidates[: max(1, int(limit))]]



def _reference_dependency_repairs(
    envelope: Mapping[str, Any] | None,
) -> list[str]:
    """Describe deterministic dependency edits for a rejected plan.

    The validator intentionally remains strict.  This helper only makes its
    structural error actionable for the bounded provider retry; it never
    changes the plan or executes a repair locally.
    """
    if not isinstance(envelope, Mapping):
        return []
    plan = envelope.get("plan")
    stages = plan.get("stages") if isinstance(plan, Mapping) else None
    if not isinstance(stages, list):
        return []
    actions: list[str] = []
    for stage in stages:
        if not isinstance(stage, Mapping) or not isinstance(stage.get("name"), str):
            continue
        stage_name = stage["name"]
        depends_on = stage.get("depends_on", [])
        existing = set(depends_on) if isinstance(depends_on, list) else set()
        refs: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                ref = value.get("ref")
                if isinstance(ref, str):
                    parts = ref.split(".")
                    if len(parts) >= 4 and parts[0] == "stage" and parts[2] == "details":
                        refs.add(parts[1])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(stage.get("parameters"))
        for source in sorted(refs - existing):
            actions.append(
                f"In stage {stage_name!r}, append {source!r} to depends_on; "
                f"the parameter reference stage.{source}.details.* requires this exact dependency."
            )
    return actions


__all__ = ["LLMTaskPlanner"]
