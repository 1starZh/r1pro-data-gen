"""Task-goal contracts independent from plans and execution backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from .scene import SceneModel

GOAL_SPEC_SCHEMA_VERSION = 1
ALLOWED_GOAL_PREDICATES = frozenset(
    {
        "object_at_pose",
        "within_tolerance",
        "inside_region",
        "on_support",
        "contact",
        "attached",
        "lifted",
        "released",
        "settled",
        "base_at_pose",
        "collision_free",
    }
)

_GOAL_SPEC_FIELDS = frozenset(
    {"schema_version", "bindings", "required", "invariants"}
)
_PREDICATE_FIELDS = frozenset({"predicate", "arguments"})
_BASE_ENTITY_TERMS = frozenset({"robot", "base", "mobile_base"})

# Predicate arguments are a public semantic contract.  Keeping this table next
# to the closed vocabulary makes malformed goals fail before verification.
_PREDICATE_ARGUMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "object_at_pose": (frozenset({"subject", "position"}), frozenset({"quaternion"})),
    "within_tolerance": (
        frozenset({"subject", "field", "target", "tolerance"}),
        frozenset(),
    ),
    "inside_region": (frozenset({"subject", "reference", "region"}), frozenset()),
    "on_support": (
        frozenset({"subject", "support", "surface", "subject_half_height_m"}),
        frozenset(),
    ),
    "contact": (frozenset({"entity_a", "entity_b"}), frozenset()),
    "attached": (frozenset({"subject"}), frozenset({"effector"})),
    "lifted": (frozenset({"subject"}), frozenset()),
    "released": (frozenset({"subject"}), frozenset({"effector"})),
    "settled": (frozenset({"subject"}), frozenset()),
    # ``pose`` is sufficient for the current single-base robot.  ``subject``
    # is an optional, explicitly scoped spelling because providers naturally
    # apply the entity-oriented vocabulary to a robot navigation goal.  It is
    # accepted only for the robot base; arbitrary object subjects would make
    # the predicate ambiguous and are rejected below.
    "base_at_pose": (frozenset({"pose"}), frozenset({"subject"})),
    "collision_free": (frozenset({"subject"}), frozenset()),
}


@dataclass(frozen=True, slots=True)
class GoalPredicate:
    """One closed-vocabulary predicate in a task completion contract."""

    predicate: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, str) or not self.predicate.strip():
            raise ValueError("predicate must be a non-empty string")
        if self.predicate not in ALLOWED_GOAL_PREDICATES:
            raise ValueError(f"unknown predicate: {self.predicate!r}")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("predicate arguments must be a mapping")
        normalized = {
            str(key): _normalize_json_value(value, f"arguments.{key}")
            for key, value in self.arguments.items()
        }
        if any(not key.strip() for key in normalized):
            raise ValueError("predicate argument names must not be empty")
        _validate_predicate_arguments(self.predicate, set(normalized))
        if self.predicate in {"attached", "released"} and "effector" in normalized:
            _validate_effector_reference(normalized["effector"], self.predicate)
        if self.predicate == "base_at_pose" and "subject" in normalized:
            subject = normalized["subject"]
            if (
                not isinstance(subject, str)
                or subject.split(".", 1)[0] not in _BASE_ENTITY_TERMS
            ):
                raise ValueError(
                    "base_at_pose subject must identify the robot base"
                )
        if self.predicate == "on_support":
            _validate_surface_contract(normalized["surface"])
        object.__setattr__(self, "arguments", _freeze_mapping(normalized))


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """Frozen statement of task success, independent from any action plan."""

    schema_version: int
    bindings: Mapping[str, str]
    required: tuple[GoalPredicate, ...]
    invariants: tuple[GoalPredicate, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != GOAL_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"goal schema_version must be {GOAL_SPEC_SCHEMA_VERSION}"
            )
        if not isinstance(self.bindings, Mapping):
            raise TypeError("goal bindings must be a mapping")
        normalized_bindings: dict[str, str] = {}
        for alias, reference in self.bindings.items():
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("goal binding aliases must be non-empty strings")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("goal binding references must be non-empty strings")
            normalized_bindings[alias] = reference
        if not self.required:
            raise ValueError("goal spec requires at least one required predicates entry")
        if any(not isinstance(item, GoalPredicate) for item in self.required):
            raise TypeError("required predicates must contain GoalPredicate values")
        if any(not isinstance(item, GoalPredicate) for item in self.invariants):
            raise TypeError("invariants must contain GoalPredicate values")
        object.__setattr__(self, "bindings", _freeze_mapping(normalized_bindings))
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "invariants", tuple(self.invariants))


def parse_goal_spec(data: Mapping[str, object], scene: SceneModel) -> GoalSpec:
    """Parse, ground, and validate a closed GoalSpec mapping."""
    if not isinstance(data, Mapping):
        raise TypeError("goal spec must be a mapping")
    unknown = set(data) - _GOAL_SPEC_FIELDS
    missing = _GOAL_SPEC_FIELDS - set(data)
    if unknown:
        raise ValueError(f"goal spec contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"goal spec is missing fields: {sorted(missing)}")

    bindings_raw = data["bindings"]
    if not isinstance(bindings_raw, Mapping):
        raise TypeError("goal bindings must be a mapping")
    bindings: dict[str, str] = {}
    scene_names = {obj.name for obj in scene.objects}
    for alias, raw_reference in bindings_raw.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("goal binding aliases must be non-empty strings")
        if not isinstance(raw_reference, str) or not raw_reference.startswith("scene://"):
            raise ValueError(
                f"goal binding {alias!r} must use a scene:// object reference"
            )
        object_name = raw_reference.removeprefix("scene://")
        # Some providers qualify an otherwise valid top-level object with the
        # scene name (``scene://<scene>/<object>``).  The public scene
        # reference contract is top-level ``scene://<object>``, but accepting
        # this unambiguous qualified spelling at the grounding boundary keeps
        # provider formatting noise from aborting GoalSpec compilation.  It is
        # normalized before hashing so the frozen contract remains stable.
        qualified_prefix = f"{scene.name}/"
        if object_name.startswith(qualified_prefix):
            candidate = object_name[len(qualified_prefix) :]
            if candidate in scene_names:
                object_name = candidate
                raw_reference = f"scene://{candidate}"
        if not object_name or object_name not in scene_names:
            raise ValueError(
                f"goal binding {alias!r} references unknown scene object {object_name!r}"
            )
        bindings[alias] = raw_reference

    required = _parse_predicates(data["required"], "required")
    invariants = _parse_predicates(data["invariants"], "invariants")
    spec = GoalSpec(
        schema_version=_require_schema_version(data["schema_version"]),
        bindings=bindings,
        required=required,
        invariants=invariants,
    )
    _validate_grounded_arguments(spec)
    _validate_region_contract(spec)
    return spec


def _validate_region_contract(spec: GoalSpec) -> None:
    """Reject inside_region predicates whose region is not a geometric object.

    A provider occasionally emits a region name string (for example ``"top"``)
    when it cannot find a region object in the scene facts.  Such a goal can
    never be verified, so rejecting it at parse time lets the goal planner
    repair the response instead of silently producing an UNKNOWN predicate.
    """
    for item in (*spec.required, *spec.invariants):
        if item.predicate != "inside_region":
            continue
        region = item.arguments.get("region")
        if not isinstance(region, Mapping):
            raise ValueError(
                f"inside_region region must be an object, got "
                f"{type(region).__name__}"
            )
        shape = region.get("shape")
        if shape not in {"cuboid", "cylinder"}:
            raise ValueError(
                f"inside_region region.shape must be 'cuboid' or 'cylinder'"
            )
        center = region.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            raise ValueError("inside_region region center must be a 3-vector")
        if shape == "cuboid":
            size = region.get("size")
            if not isinstance(size, (list, tuple)) or len(size) != 3:
                raise ValueError("inside_region cuboid region size must be a 3-vector")
        else:
            if "radius" not in region or "height" not in region:
                raise ValueError(
                    "inside_region cylinder region requires radius and height"
                )


def goal_spec_to_dict(spec: GoalSpec) -> dict[str, object]:
    """Return the public JSON shape for a GoalSpec."""
    if not isinstance(spec, GoalSpec):
        raise TypeError("spec must be a GoalSpec")
    return {
        "schema_version": spec.schema_version,
        "bindings": dict(spec.bindings),
        "required": [_predicate_to_dict(item) for item in spec.required],
        "invariants": [_predicate_to_dict(item) for item in spec.invariants],
    }


def goal_spec_sha256(spec: GoalSpec) -> str:
    """Hash a GoalSpec through a deterministic canonical JSON encoding."""
    canonical = json.dumps(
        goal_spec_to_dict(spec),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_predicate_arguments(
    predicate: str,
    argument_names: set[str],
) -> None:
    required, optional = _PREDICATE_ARGUMENTS[predicate]
    allowed = required | optional
    missing = required - argument_names
    unknown = argument_names - allowed
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(
            f"predicate {predicate!r} has invalid arguments: {'; '.join(details)}"
        )


def _require_schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("goal schema_version must be an integer")
    return value


def _validate_surface_contract(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("surface must be an object")
    required = {"center", "size"}
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"surface has invalid fields: {'; '.join(details)}")
    _validate_finite_vector(value["center"], 3, "surface.center")
    _validate_finite_vector(value["size"], 2, "surface.size", positive=True)


def _validate_finite_vector(
    value: object,
    length: int,
    path: str,
    *,
    positive: bool = False,
) -> None:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a numeric array")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be a numeric array") from exc
    if len(values) != length or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{path} must contain {length} finite values")
    if positive and any(item <= 0.0 for item in values):
        raise ValueError(f"{path} must contain positive values")


def _parse_predicates(value: object, field_name: str) -> tuple[GoalPredicate, ...]:
    if not isinstance(value, list):
        raise TypeError(f"goal {field_name} must be an array")
    parsed: list[GoalPredicate] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"goal {field_name}[{index}] must be an object")
        unknown = set(raw) - _PREDICATE_FIELDS
        missing = _PREDICATE_FIELDS - set(raw)
        if unknown:
            raise ValueError(
                f"goal {field_name}[{index}] contains unknown fields: {sorted(unknown)}"
            )
        if missing:
            raise ValueError(
                f"goal {field_name}[{index}] is missing fields: {sorted(missing)}"
            )
        predicate = raw["predicate"]
        if not isinstance(predicate, str):
            raise TypeError(f"goal {field_name}[{index}].predicate must be a string")
        arguments = raw["arguments"]
        if not isinstance(arguments, Mapping):
            raise TypeError(f"goal {field_name}[{index}].arguments must be an object")
        parsed.append(GoalPredicate(predicate=predicate, arguments=arguments))
    return tuple(parsed)


def _validate_grounded_arguments(spec: GoalSpec) -> None:
    aliases = set(spec.bindings)
    # Direct scene:// references are also valid when they name a bound object:
    # the planner prompt asks for bare aliases, but a provider occasionally
    # repeats the full URI and that is still an unambiguous entity reference.
    bound_scene_objects = {
        reference.removeprefix("scene://")
        for reference in spec.bindings.values()
        if reference.startswith("scene://")
    }
    robot_terms = {
        "robot",
        "base",
        "mobile_base",
        "left_gripper",
        "right_gripper",
        "left_ee",
        "right_ee",
    }
    entity_keys = {
        "subject",
        "support",
        "entity",
        "entity_a",
        "entity_b",
        "object",
        "effector",
        "reference",
    }
    for item in (*spec.required, *spec.invariants):
        for key, raw_value in item.arguments.items():
            if key not in entity_keys or not isinstance(raw_value, str):
                continue
            root = raw_value.split(".", 1)[0]
            if (
                key == "effector"
                and item.predicate in {"attached", "released"}
                and root in _BASE_ENTITY_TERMS
            ):
                raise ValueError(
                    f"predicate {item.predicate!r} effector must identify an "
                    "end-effector, not the robot base; omit it for an "
                    "effector-agnostic release"
                )
            if root in aliases or root in robot_terms:
                continue
            # Contact predicates may name the concrete robot body attached to
            # a declared contact sensor (for example ``base_link``).  The
            # scene-aware GoalCompiler validates that endpoint against the
            # sensor table; keeping it open here supports robots whose body
            # naming is not part of the generic GoalSpec vocabulary.
            if item.predicate == "contact" and key in {"entity_a", "entity_b"}:
                continue
            if root.startswith("scene://") and root.removeprefix("scene://") in bound_scene_objects:
                continue
            raise ValueError(
                f"predicate {item.predicate!r} argument {key!r} references "
                f"unknown binding {raw_value!r}"
            )


def _validate_effector_reference(value: object, predicate: str) -> None:
    """Reject base-level vocabulary where an attachment endpoint is required.

    Attachment evidence connects an entity to a concrete end-effector.  The
    robot/base/mobile_base terms are intentionally reserved for
    ``base_at_pose`` and other base-level predicates; accepting one here would
    create an apparently valid but unverifiable GoalSpec.  Concrete aliases
    remain open so a different robot adapter can expose its own endpoint
    names.
    """
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{predicate} effector must be a non-empty string")
    root = value.split(".", 1)[0]
    if root in _BASE_ENTITY_TERMS:
        raise ValueError(
            f"{predicate} effector must identify an end-effector, not the "
            "robot base; omit it for an effector-agnostic release"
        )


def _predicate_to_dict(predicate: GoalPredicate) -> dict[str, object]:
    return {
        "predicate": predicate.predicate,
        "arguments": _thaw_json(predicate.arguments),
    }


def _normalize_json_value(value: object, path: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} keys must be non-empty strings")
            normalized[key] = _normalize_json_value(item, f"{path}.{key}")
        return _freeze_mapping(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(
            _normalize_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} is not JSON-compatible: {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "ALLOWED_GOAL_PREDICATES",
    "GOAL_SPEC_SCHEMA_VERSION",
    "GoalPredicate",
    "GoalSpec",
    "goal_spec_sha256",
    "goal_spec_to_dict",
    "parse_goal_spec",
]
