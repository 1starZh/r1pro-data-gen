"""Contracts and validation for plans proposed by an external LLM.

The existing :class:`Plan` model is intentionally permissive enough for trusted
replay plans. LLM output is untrusted input, so this module adds a stricter,
policy-aware validation boundary before a plan can reach simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from r1pro_data_gen.data.plan_io import plan_from_dict, plan_to_dict
from r1pro_data_gen.domain import Plan

LLM_SCHEMA_VERSION = "1.0"

# This is an execution policy, not merely a prompt catalogue. Keep it explicit
# so a caller cannot turn a hidden backend skill into an LLM action by changing
# the descriptions sent to the model.
LLM_PUBLIC_SKILLS = frozenset(
    {
        "base_navigate_to",
        "prepare_workspace",
        "grasp_object",
        "arm_carry_object_to",
        "release_object",
        "push_object_to",
    }
)

_ALLOWED_ENVELOPE_KEYS = frozenset({"schema_version", "status", "reason", "plan"})
_ALLOWED_PLAN_KEYS = frozenset({"task_name", "stages", "metadata"})
_ALLOWED_STAGE_KEYS = frozenset(
    {"name", "goal", "depends_on", "parameters", "outputs", "preconditions", "postconditions"}
)
_ALLOWED_OUTPUTS = frozenset({"position", "quaternion", "contact_forces", "joint_positions"})
_ALLOWED_REFERENCE_KEYS = frozenset({"ref", "value_type", "shape", "frame", "offset"})
_ALLOWED_METADATA_KEYS = frozenset(
    {
        "source",
        "provider",
        "model",
        "schema_version",
        "scene",
        "task_description",
        "prompt_version",
        "notes",
        "goal_spec_hash",
        "goal_contract_hash",
        "stages_subset",
    }
)


@dataclass(frozen=True, slots=True)
class PlanLimits:
    """Hard limits applied to an untrusted LLM plan."""

    max_json_bytes: int = 64 * 1024
    # Typed references nested inside waypoint poses need more than the
    # ordinary Plan fields, while this remains a finite hostile-input bound.
    max_depth: int = 16
    max_stages: int = 16
    max_motion_stages: int = 14
    max_arm_motion_stages: int = 10
    max_base_motion_stages: int = 2
    max_waypoints: int = 6
    max_string_length: int = 2048


DEFAULT_PLAN_LIMITS = PlanLimits()


class LLMPlanValidationError(ValueError):
    """Raised when an external LLM plan violates its data or execution policy."""


def validate_envelope(
    data: Mapping[str, Any],
    *,
    skill_catalog: Sequence[Mapping[str, Any]],
    registry: Any = None,
    scene_object_names: Sequence[str] = (),
    limits: PlanLimits = DEFAULT_PLAN_LIMITS,
) -> Plan | None:
    """Validate an LLM response envelope and return its executable Plan.

    ``None`` is returned for a deliberate ``unsupported`` response. All other
    invalid responses raise :class:`LLMPlanValidationError` and must not be
    executed.
    """
    if not isinstance(data, Mapping):
        raise LLMPlanValidationError("LLM response must be a JSON object")
    data = _normalize_direct_plan_envelope(data)
    _check_keys(data, _ALLOWED_ENVELOPE_KEYS, "LLM response")
    if data.get("schema_version") != LLM_SCHEMA_VERSION:
        raise LLMPlanValidationError(
            f"unsupported LLM schema_version: {data.get('schema_version')!r}"
        )
    status = data.get("status")
    if status not in {"planned", "unsupported"}:
        raise LLMPlanValidationError("LLM status must be 'planned' or 'unsupported'")
    reason = data.get("reason", "")
    if not isinstance(reason, str) or len(reason) > limits.max_string_length:
        raise LLMPlanValidationError("LLM reason must be a bounded string")
    if status == "unsupported":
        if data.get("plan") is not None:
            raise LLMPlanValidationError("unsupported response must not contain a plan")
        if not reason.strip():
            raise LLMPlanValidationError("unsupported response requires a reason")
        return None
    if not isinstance(data.get("plan"), Mapping):
        raise LLMPlanValidationError("planned response requires a plan object")
    return validate_plan_dict(
        data["plan"],
        skill_catalog=skill_catalog,
        registry=registry,
        scene_object_names=scene_object_names,
        limits=limits,
    )


def _normalize_direct_plan_envelope(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize one strict direct-Plan variant emitted by JSON-mode models."""
    direct_keys = {"envelope_schema_version", "task_name", "stages"}
    if not (set(data) & {"envelope_schema_version", "task_name", "stages"}):
        return data
    if set(data) - direct_keys - {"metadata"}:
        raise LLMPlanValidationError("direct Plan response contains unknown fields")
    if set(data) != direct_keys and set(data) != direct_keys | {"metadata"}:
        raise LLMPlanValidationError("direct Plan response is incomplete")
    if data.get("envelope_schema_version") != LLM_SCHEMA_VERSION:
        raise LLMPlanValidationError(
            f"unsupported direct Plan schema version: {data.get('envelope_schema_version')!r}"
        )
    return {
        "schema_version": LLM_SCHEMA_VERSION,
        "status": "planned",
        "reason": "",
        "plan": {
            "task_name": data.get("task_name"),
            "stages": data.get("stages"),
            "metadata": data.get("metadata", {}),
        },
    }


def validate_plan_dict(
    data: Mapping[str, Any],
    *,
    skill_catalog: Sequence[Mapping[str, Any]],
    registry: Any = None,
    scene_object_names: Sequence[str] = (),
    limits: PlanLimits = DEFAULT_PLAN_LIMITS,
) -> Plan:
    """Validate and construct a Plan from untrusted JSON-compatible data."""
    _ensure_json_tree(data, "plan", limits.max_depth, limits.max_string_length)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > limits.max_json_bytes:
        raise LLMPlanValidationError("plan exceeds the maximum JSON size")
    _check_keys(data, _ALLOWED_PLAN_KEYS, "plan")
    task_name = data.get("task_name")
    if not _bounded_string(task_name, limits.max_string_length):
        raise LLMPlanValidationError("plan.task_name must be a bounded non-empty string")
    stages = data.get("stages")
    if not isinstance(stages, list):
        raise LLMPlanValidationError("plan.stages must be an array")
    if not stages or len(stages) > limits.max_stages:
        raise LLMPlanValidationError(
            f"plan.stages must contain 1..{limits.max_stages} stages"
        )
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise LLMPlanValidationError("plan.metadata must be an object")
    _check_keys(metadata, _ALLOWED_METADATA_KEYS, "plan.metadata")

    known_names: list[str] = []
    declared_outputs: dict[str, set[str]] = {}
    motion_count = 0
    arm_motion_count = 0
    base_motion_count = 0
    catalog = {str(item.get("name")): item for item in skill_catalog}
    for index, stage in enumerate(stages):
        prefix = f"plan.stages[{index}]"
        if not isinstance(stage, Mapping):
            raise LLMPlanValidationError(f"{prefix} must be an object")
        _check_keys(stage, _ALLOWED_STAGE_KEYS, prefix)
        name = stage.get("name")
        goal = stage.get("goal")
        if not _bounded_string(name, limits.max_string_length):
            raise LLMPlanValidationError(f"{prefix}.name must be a bounded non-empty string")
        if name in known_names:
            raise LLMPlanValidationError(f"duplicate stage name: {name!r}")
        if not _bounded_string(goal, limits.max_string_length):
            raise LLMPlanValidationError(f"{prefix}.goal must be a bounded non-empty string")
        depends_on = stage.get("depends_on", [])
        if not isinstance(depends_on, list) or any(not isinstance(item, str) for item in depends_on):
            raise LLMPlanValidationError(f"{prefix}.depends_on must be an array of strings")
        missing_or_forward = [item for item in depends_on if item not in known_names]
        if missing_or_forward:
            raise LLMPlanValidationError(
                f"{prefix}.depends_on must reference earlier stages: {missing_or_forward}"
            )
        parameters = stage.get("parameters")
        if not isinstance(parameters, Mapping):
            raise LLMPlanValidationError(f"{prefix}.parameters must be an object")
        outputs = stage.get("outputs", [])
        if not isinstance(outputs, list) or any(item not in _ALLOWED_OUTPUTS for item in outputs):
            raise LLMPlanValidationError(
                f"{prefix}.outputs must contain only {_ALLOWED_OUTPUTS}"
            )
        if len(set(outputs)) != len(outputs):
            raise LLMPlanValidationError(f"{prefix}.outputs must be unique")
        declared_outputs[name] = set(outputs)
        for condition_name in ("preconditions", "postconditions"):
            conditions = stage.get(condition_name, [])
            if not isinstance(conditions, list) or any(not isinstance(item, Mapping) for item in conditions):
                raise LLMPlanValidationError(f"{prefix}.{condition_name} must be an array of objects")
            for condition in conditions:
                _check_keys(condition, {"predicate", "parameters"}, f"{prefix}.{condition_name}")
                predicate = condition.get("predicate")
                if not _bounded_string(predicate, limits.max_string_length):
                    raise LLMPlanValidationError(f"{prefix}.{condition_name}.predicate must be a bounded string")
                if predicate not in _ALLOWED_PREDICATES:
                    raise LLMPlanValidationError(f"{prefix}.{condition_name} uses an unsupported predicate: {predicate!r}")
                condition_params = condition.get("parameters", {})
                if not isinstance(condition_params, Mapping):
                    raise LLMPlanValidationError(f"{prefix}.{condition_name}.parameters must be an object")
                _validate_reference_tree(condition_params, prefix, limits)
                _validate_reference_sources(
                    condition_params,
                    f"{prefix}.{condition_name}",
                    current_stage=name,
                    prior_stages=set(known_names),
                    declared_outputs=declared_outputs,
                    depends_on=depends_on,
                )
        skill_name = parameters.get("skill")
        if not isinstance(skill_name, str) or skill_name not in LLM_PUBLIC_SKILLS:
            raise LLMPlanValidationError(
                f"{prefix} uses a skill outside the external LLM policy: {skill_name!r}"
            )
        spec = catalog.get(skill_name)
        if spec is None:
            raise LLMPlanValidationError(f"skill catalogue does not describe {skill_name!r}")
        _validate_skill_parameters(
            skill_name,
            parameters,
            spec,
            registry=registry,
            scene_object_names=set(scene_object_names),
            limits=limits,
        )
        _validate_reference_tree(parameters, prefix, limits)
        if "ref" in parameters:
            raise LLMPlanValidationError(f"{prefix}.parameters itself cannot be a reference")
        _validate_reference_sources(
            parameters,
            prefix,
            current_stage=name,
            prior_stages=set(known_names),
            declared_outputs=declared_outputs,
            depends_on=depends_on,
        )
        if _is_motion_skill(skill_name):
            motion_count += 1
        if skill_name.startswith("arm_"):
            arm_motion_count += 1
        if skill_name.startswith("base_"):
            base_motion_count += 1
        known_names.append(name)

    if motion_count > limits.max_motion_stages:
        raise LLMPlanValidationError("plan exceeds the motion-stage budget")
    if arm_motion_count > limits.max_arm_motion_stages:
        raise LLMPlanValidationError("plan exceeds the arm-motion budget")
    if base_motion_count > limits.max_base_motion_stages:
        raise LLMPlanValidationError("plan exceeds the base-motion budget")

    try:
        return plan_from_dict(
            {
                "task_name": task_name,
                "stages": stages,
                "metadata": dict(metadata),
            }
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise LLMPlanValidationError(f"invalid normalized Plan: {exc}") from exc


def validate_plan(
    plan: Plan,
    *,
    skill_catalog: Sequence[Mapping[str, Any]],
    registry: Any = None,
    scene_object_names: Sequence[str] = (),
    limits: PlanLimits = DEFAULT_PLAN_LIMITS,
) -> Plan:
    """Validate an already constructed external Plan before execution."""
    return validate_plan_dict(
        plan_to_dict(plan),
        skill_catalog=skill_catalog,
        registry=registry,
        scene_object_names=scene_object_names,
        limits=limits,
    )


def parse_json_object(text: str, *, max_bytes: int = DEFAULT_PLAN_LIMITS.max_json_bytes) -> Mapping[str, Any]:
    """Parse the first JSON object in a bounded provider response.

    Providers occasionally wrap JSON in one Markdown fence or append a
    second object/explanation.  Recovering the first balanced object keeps the
    parser tolerant to that transport noise while retaining an object-only,
    finite-number contract.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMPlanValidationError("LLM response text is empty")
    if len(text.encode("utf-8")) > max_bytes:
        raise LLMPlanValidationError("LLM response exceeds the maximum size")
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline < 0:
            raise LLMPlanValidationError("LLM response Markdown fence is incomplete")
        stripped = stripped[first_newline + 1 :]
        closing = stripped.find("```")
        if closing >= 0:
            stripped = stripped[:closing]
    start = stripped.find("{")
    if start < 0:
        raise LLMPlanValidationError("LLM response must contain a JSON object")
    end = _first_balanced_object_end(stripped, start)
    if end is None:
        raise LLMPlanValidationError("LLM response contains an incomplete JSON object")
    try:
        value = json.loads(stripped[start:end], parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMPlanValidationError(f"invalid LLM JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LLMPlanValidationError("LLM response must decode to a JSON object")
    return value


def _first_balanced_object_end(text: str, start: int) -> int | None:
    """Return the exclusive end of the first balanced JSON object."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def _validate_skill_parameters(
    skill_name: str,
    params: Mapping[str, Any],
    skill_description: Mapping[str, Any],
    *,
    registry: Any,
    scene_object_names: set[str],
    limits: PlanLimits,
) -> None:
    if "skill" not in params:
        raise LLMPlanValidationError(f"skill parameter missing for {skill_name!r}")
    if registry is not None and not _contains_reference(params):
        try:
            registry.validate_plan_params(skill_name, params)
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMPlanValidationError(f"invalid parameters for {skill_name!r}: {exc}") from exc
    declared = skill_description.get("parameters", {})
    if not isinstance(declared, Mapping):
        raise LLMPlanValidationError(f"invalid catalogue entry for {skill_name!r}")
    unknown = set(params) - {"skill"} - set(declared)
    if unknown:
        allowed = sorted(str(name) for name in declared)
        raise LLMPlanValidationError(
            f"unknown parameters for {skill_name!r}: {sorted(unknown)}; "
            f"allowed parameters: {allowed}"
        )
    for pname, spec in declared.items():
        if not isinstance(spec, Mapping):
            raise LLMPlanValidationError(f"invalid parameter declaration {skill_name}.{pname}")
        if spec.get("required") and pname not in params:
            raise LLMPlanValidationError(f"missing required parameter {skill_name}.{pname}")
        if pname not in params:
            continue
        _validate_declared_value(skill_name, pname, params[pname], spec)

    for pname in ("object_name", "target_region_name", "support_surface_name"):
        value = params.get(pname)
        if value is not None and scene_object_names and value not in scene_object_names:
            raise LLMPlanValidationError(
                f"unknown scene object {value!r} in {skill_name}.{pname}; "
                f"available objects: {sorted(scene_object_names)}"
            )
    target_ref = params.get("target_ref")
    if skill_name == "base_navigate_to" and isinstance(target_ref, str):
        if target_ref.startswith("scene://"):
            target_name = target_ref[len("scene://"):].split("/", 1)[0]
            if scene_object_names and target_name not in scene_object_names:
                raise LLMPlanValidationError(
                    f"unknown scene object {target_name!r} in {skill_name}.target_ref; "
                    f"available objects: {sorted(scene_object_names)}"
                )
    exclusions = params.get("exclude_objects", [])
    if isinstance(exclusions, list) and scene_object_names:
        unknown_objects = set(exclusions) - scene_object_names
        if unknown_objects:
            raise LLMPlanValidationError(
                f"unknown excluded scene objects for {skill_name}: {sorted(unknown_objects)}"
            )
    if skill_name == "arm_move_through":
        waypoints = params.get("waypoints", [])
        if not isinstance(waypoints, list) or len(waypoints) > limits.max_waypoints:
            raise LLMPlanValidationError("arm_move_through exceeds waypoint budget")
        _validate_arm_waypoints(waypoints, scene_object_names, limits)


def _validate_arm_waypoints(
    waypoints: list[Any],
    scene_object_names: set[str],
    limits: PlanLimits,
) -> None:
    for index, waypoint in enumerate(waypoints):
        prefix = f"arm_move_through.waypoints[{index}]"
        if not isinstance(waypoint, Mapping):
            raise LLMPlanValidationError(f"{prefix} must be an object")
        allowed = {"name", "poses", "exclude_objects", "contact", "speed_scale"}
        _check_keys(waypoint, allowed, prefix)
        if not _bounded_string(waypoint.get("name"), limits.max_string_length):
            raise LLMPlanValidationError(f"{prefix}.name must be a bounded non-empty string")
        poses = waypoint.get("poses")
        if not isinstance(poses, list) or not poses or len(poses) > limits.max_waypoints:
            raise LLMPlanValidationError(
                f"{prefix}.poses must contain 1..{limits.max_waypoints} poses"
            )
        for pose_index, pose in enumerate(poses):
            pose_path = f"{prefix}.poses[{pose_index}]"
            if not isinstance(pose, Mapping):
                raise LLMPlanValidationError(f"{pose_path} must be an object")
            _check_keys(pose, {"position", "orientation"}, pose_path)
            position = pose.get("position")
            orientation = pose.get("orientation")
            if not isinstance(position, list) or len(position) != 3:
                raise LLMPlanValidationError(f"{pose_path}.position must have shape (3,)")
            if not isinstance(orientation, list) or len(orientation) != 4:
                raise LLMPlanValidationError(f"{pose_path}.orientation must have shape (4,)")
            if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in position + orientation):
                raise LLMPlanValidationError(f"{pose_path} contains a non-finite numeric value")
        exclusions = waypoint.get("exclude_objects", [])
        if not isinstance(exclusions, list) or any(not isinstance(item, str) for item in exclusions):
            raise LLMPlanValidationError(f"{prefix}.exclude_objects must be an array of strings")
        unknown = set(exclusions) - scene_object_names
        if scene_object_names and unknown:
            raise LLMPlanValidationError(f"{prefix} names unknown excluded objects: {sorted(unknown)}")
        contact = waypoint.get("contact", False)
        if not isinstance(contact, bool):
            raise LLMPlanValidationError(f"{prefix}.contact must be boolean")
        if "speed_scale" in waypoint:
            speed_scale = waypoint["speed_scale"]
            if (
                isinstance(speed_scale, bool)
                or not isinstance(speed_scale, Real)
                or not math.isfinite(float(speed_scale))
                or speed_scale <= 0
            ):
                raise LLMPlanValidationError(
                    f"{prefix}.speed_scale must be a finite positive number"
                )


_ALLOWED_PREDICATES = frozenset(
    {
        "contact_detected",
        "reference_available",
        "within_tolerance",
    }
)


def _validate_declared_value(
    skill: str,
    name: str,
    value: Any,
    spec: Mapping[str, Any],
) -> None:
    """Validate one catalogue value while allowing typed references."""
    if _contains_reference(value):
        return
    kind = spec.get("type")
    if value is None:
        if spec.get("required"):
            raise LLMPlanValidationError(f"required parameter {skill}.{name} cannot be null")
        return
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise LLMPlanValidationError(f"{skill}.{name} must be a finite number")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise LLMPlanValidationError(f"{skill}.{name} must be an integer")
    elif kind == "string" and not isinstance(value, str):
        raise LLMPlanValidationError(f"{skill}.{name} must be a string")
    elif kind == "boolean" and not isinstance(value, bool):
        raise LLMPlanValidationError(f"{skill}.{name} must be a boolean")
    elif kind == "array":
        if not isinstance(value, list):
            raise LLMPlanValidationError(f"{skill}.{name} must be an array")
        minimum = spec.get("min_items")
        maximum = spec.get("max_items")
        if minimum is not None and len(value) < minimum:
            raise LLMPlanValidationError(f"{skill}.{name} has too few items")
        if maximum is not None and len(value) > maximum:
            raise LLMPlanValidationError(f"{skill}.{name} has too many items")
        shape = spec.get("shape")
        if shape is not None and _shape_of(value) != tuple(shape):
            raise LLMPlanValidationError(
                f"{skill}.{name} must have shape {tuple(shape)}, got {_shape_of(value)}"
            )
        _ensure_json_tree(value, f"{skill}.{name}", 6, 2048)
    elif kind == "object" and not isinstance(value, Mapping):
        raise LLMPlanValidationError(f"{skill}.{name} must be an object")
    elif kind not in {"number", "integer", "string", "boolean", "array", "object"}:
        raise LLMPlanValidationError(f"unsupported declared type {kind!r}")
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if minimum is not None and value < minimum:
            raise LLMPlanValidationError(f"{skill}.{name} is below its minimum")
        if maximum is not None and value > maximum:
            raise LLMPlanValidationError(f"{skill}.{name} is above its maximum")
    enum = spec.get("enum") or []
    if enum and value not in enum:
        raise LLMPlanValidationError(f"{skill}.{name} must be one of {list(enum)!r}")


def _validate_reference_tree(value: Any, path: str, limits: PlanLimits, depth: int = 6) -> None:
    """Validate declarative references without evaluating them."""
    if depth < 0:
        raise LLMPlanValidationError(f"{path} reference nesting is too deep")
    if isinstance(value, Mapping):
        if "ref" in value:
            _check_keys(value, _ALLOWED_REFERENCE_KEYS, f"{path}.reference")
            ref = value.get("ref")
            if not isinstance(ref, str) or not ref or any(
                part in {"", "__dict__", "__class__"} for part in ref.split(".")
            ):
                raise LLMPlanValidationError(f"{path}.reference has an invalid source path")
            if not ref.startswith(("stage.", "observation.", "scene.object.")):
                raise LLMPlanValidationError(f"{path}.reference has an unsupported source")
            value_type = value.get("value_type")
            if value_type is not None and value_type not in {"number", "array", "object"}:
                raise LLMPlanValidationError(f"{path}.reference has an unsupported value_type")
            frame = value.get("frame")
            if frame is not None and frame not in {"world", "base"}:
                raise LLMPlanValidationError(f"{path}.reference has an unsupported frame")
            shape = value.get("shape")
            if shape is not None and (
                not isinstance(shape, list)
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
            ):
                raise LLMPlanValidationError(f"{path}.reference.shape is invalid")
            offset = value.get("offset")
            if offset is not None and (
                not isinstance(offset, list)
                or len(offset) != 3
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, Real)
                    or not math.isfinite(float(item))
                    for item in offset
                )
            ):
                raise LLMPlanValidationError(f"{path}.reference.offset must be a finite 3-vector")
            return
        for key, item in value.items():
            if not isinstance(key, str):
                raise LLMPlanValidationError(f"{path} contains a non-string key")
            _validate_reference_tree(item, f"{path}.{key}", limits, depth - 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_reference_tree(item, f"{path}[{index}]", limits, depth - 1)


def _validate_reference_sources(
    value: Any,
    path: str,
    *,
    current_stage: str,
    prior_stages: set[str],
    declared_outputs: Mapping[str, set[str]],
    depends_on: Sequence[str] = (),
) -> None:
    """Check that stage references use declared, completed dependencies."""
    if isinstance(value, Mapping):
        if "ref" in value:
            ref = value["ref"]
            if not isinstance(ref, str) or not ref.startswith("stage."):
                return
            parts = ref.split(".")
            if len(parts) != 4 or parts[2] != "details":
                raise LLMPlanValidationError(
                    f"{path}.reference stage source must be stage.<name>.details.<field>"
                )
            source, field = parts[1], parts[3]
            if source == current_stage:
                raise LLMPlanValidationError(f"{path}.reference cannot reference its current stage")
            if source not in prior_stages:
                raise LLMPlanValidationError(f"{path}.reference must use an earlier stage: {source!r}")
            if source not in depends_on:
                raise LLMPlanValidationError(
                    f"{path}.reference source {source!r} must be listed in depends_on"
                )
            if field not in _ALLOWED_OUTPUTS:
                raise LLMPlanValidationError(f"{path}.reference uses an unsupported output: {field!r}")
            if field not in declared_outputs.get(source, set()):
                raise LLMPlanValidationError(
                    f"{path}.reference field {field!r} was not declared by stage {source!r}"
                )
            return
        for key, item in value.items():
            _validate_reference_sources(
                item,
                f"{path}.{key}",
                current_stage=current_stage,
                prior_stages=prior_stages,
                declared_outputs=declared_outputs,
                depends_on=depends_on,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_reference_sources(
                item,
                f"{path}[{index}]",
                current_stage=current_stage,
                prior_stages=prior_stages,
                declared_outputs=declared_outputs,
                depends_on=depends_on,
            )


def _contains_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "ref" in value or any(_contains_reference(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_reference(item) for item in value)
    return False


def _ensure_json_tree(value: Any, path: str, depth: int, max_string_length: int) -> None:
    if depth < 0:
        raise LLMPlanValidationError(f"{path} exceeds maximum nesting depth")
    if isinstance(value, str):
        if len(value) > max_string_length:
            raise LLMPlanValidationError(f"{path} contains an oversized string")
        return
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise LLMPlanValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LLMPlanValidationError(f"{path} contains a non-string key")
            _ensure_json_tree(item, f"{path}.{key}", depth - 1, max_string_length)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _ensure_json_tree(item, f"{path}[{index}]", depth - 1, max_string_length)
        return
    raise LLMPlanValidationError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _shape_of(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return (len(value),) + (_shape_of(value[0]) if value else ())


def _check_keys(value: Mapping[str, Any], allowed: set[str] | frozenset[str], path: str) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise LLMPlanValidationError(f"{path} contains unknown fields: {sorted(unknown)}")


def _bounded_string(value: Any, maximum: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _is_motion_skill(name: str) -> bool:
    return name.startswith(("base_", "arm_", "torso_", "gripper_", "joint_mask_"))


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


__all__ = [
    "DEFAULT_PLAN_LIMITS",
    "LLMPlanValidationError",
    "LLM_PUBLIC_SKILLS",
    "LLM_SCHEMA_VERSION",
    "PlanLimits",
    "parse_json_object",
    "validate_envelope",
    "validate_plan",
    "validate_plan_dict",
]
