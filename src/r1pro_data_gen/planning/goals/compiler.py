"""Deterministic compilation of semantic goals into observable contracts.

The LLM supplies semantic predicates.  ``GoalCompiler`` binds those
predicates to SceneModel geometry and the simulator's observation surface. It
does not infer missing geometry or invent tolerances, and it never calls an
LLM.  A compile failure is safe to feed back to the goal planner for a bounded
repair attempt before Isaac Sim starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from r1pro_data_gen.domain import GoalPredicate, GoalSpec, ObjectType, SceneModel
from r1pro_data_gen.evaluation import VERIFICATION_POLICY_VERSION
from r1pro_data_gen.planning.context.facts import scene_to_facts


_BASE_ENTITY_TERMS = frozenset({"robot", "base", "mobile_base"})


class GoalCompileError(ValueError):
    """Raised when a semantic goal cannot be grounded or observed safely."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class CompiledGoalContract:
    """Immutable goal contract consumed by execution and verification."""

    goal_spec: GoalSpec
    goal_spec_hash: str
    contract_hash: str
    scene_name: str
    required_observations: tuple[str, ...]
    predicate_contracts: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "goal_spec_hash": self.goal_spec_hash,
            "contract_hash": self.contract_hash,
            "scene_name": self.scene_name,
            "required_observations": list(self.required_observations),
            "predicate_contracts": [dict(item) for item in self.predicate_contracts],
        }


class GoalCompiler:
    """Compile a GoalSpec against one immutable SceneModel."""

    def compile(self, goal_spec: GoalSpec, scene: SceneModel) -> CompiledGoalContract:
        if not isinstance(goal_spec, GoalSpec):
            raise TypeError("goal_spec must be a GoalSpec")
        if not isinstance(scene, SceneModel):
            raise TypeError("scene must be a SceneModel")

        observations: set[str] = set()
        contracts: list[Mapping[str, Any]] = []
        for invariant in goal_spec.invariants:
            contracts.append(
                self._compile_predicate(
                    invariant,
                    goal_spec,
                    scene,
                    observations,
                    invariant=True,
                )
            )
        for required in goal_spec.required:
            contracts.append(
                self._compile_predicate(
                    required,
                    goal_spec,
                    scene,
                    observations,
                    invariant=False,
                )
            )

        goal_hash = _goal_hash(goal_spec)
        scene_hash = _canonical_hash(scene_to_facts(scene))
        contract_payload = {
            "goal_spec_hash": goal_hash,
            "scene_hash": scene_hash,
            "verification_policy_version": VERIFICATION_POLICY_VERSION,
            "required_observations": sorted(observations),
            "predicate_contracts": contracts,
        }
        contract_hash = _canonical_hash(contract_payload)
        return CompiledGoalContract(
            goal_spec=goal_spec,
            goal_spec_hash=goal_hash,
            contract_hash=contract_hash,
            scene_name=scene.name,
            required_observations=tuple(sorted(observations)),
            predicate_contracts=tuple(contracts),
        )

    def _compile_predicate(
        self,
        predicate: GoalPredicate,
        spec: GoalSpec,
        scene: SceneModel,
        observations: set[str],
        *,
        invariant: bool,
    ) -> Mapping[str, Any]:
        args = predicate.arguments
        source: dict[str, Any] = {
            "predicate": predicate.predicate,
            "invariant": invariant,
        }
        if predicate.predicate == "inside_region":
            reference = _bound_object(spec, scene, args["reference"])
            region = _mapping(args["region"], "inside_region.region")
            matched = _match_region(reference, region)
            if matched is None:
                raise GoalCompileError(
                    "REGION_GEOMETRY_MISMATCH",
                    f"inside_region does not match any region declared by {reference.name!r}",
                    details={"reference": reference.name, "region": dict(region)},
                )
            observations.add("entity_states")
            source.update(
                {
                    "reference": reference.name,
                    "geometry_source": f"scene://{reference.name}.regions.{matched}",
                    "evaluation": "full_footprint_in_reference_frame",
                }
            )
        elif predicate.predicate == "on_support":
            support = _bound_object(spec, scene, args["support"])
            subject = _bound_object(spec, scene, args["subject"])
            surface = _mapping(args["surface"], "on_support.surface")
            matched = _match_surface(support, surface)
            if matched is None:
                raise GoalCompileError(
                    "SUPPORT_GEOMETRY_MISMATCH",
                    f"on_support surface does not match geometry declared by {support.name!r}",
                    details={"support": support.name, "surface": dict(surface)},
                )
            expected_half_height = _half_height(subject)
            actual_half_height = _positive_number(
                args.get("subject_half_height_m"),
                "subject_half_height_m",
            )
            if abs(expected_half_height - actual_half_height) > 1e-6:
                raise GoalCompileError(
                    "SUBJECT_GEOMETRY_MISMATCH",
                    f"on_support subject height does not match scene object {subject.name!r}",
                    details={
                        "subject": subject.name,
                        "expected_half_height_m": expected_half_height,
                        "provided_half_height_m": actual_half_height,
                    },
                )
            observations.update({"entity_states", "settled_windows"})
            # Contact is an optional strengthening signal.  Geometry + a
            # verified settled window remains a valid support proof when a
            # scene did not declare pairwise contact sensors.
            if _has_pair_sensor(scene.contact_sensors, subject.name, support.name):
                observations.add("contact_events")
                contact_mode = "contact_plus_geometry"
            else:
                contact_mode = "geometry_plus_settled"
            source.update(
                {
                    "support": support.name,
                    "surface_source": f"scene://{support.name}.surfaces.{matched}",
                    "evaluation": contact_mode,
                }
            )
        elif predicate.predicate == "collision_free":
            subject = _bound_object(spec, scene, args["subject"])
            if not _has_collision_sensor(scene, subject.name):
                raise GoalCompileError(
                    "COLLISION_OBSERVATION_UNAVAILABLE",
                    f"collision_free for {subject.name!r} has no declared collision telemetry",
                    details={"subject": subject.name},
                )
            observations.add("collision_events")
            source.update({"subject": subject.name, "evaluation": "scoped_collision_telemetry"})
        elif predicate.predicate in {"contact"}:
            first = _contact_endpoint(spec, scene, args["entity_a"])
            second = _contact_endpoint(spec, scene, args["entity_b"])
            if not _has_pair_sensor(scene.contact_sensors, first, second):
                raise GoalCompileError(
                    "CONTACT_OBSERVATION_UNAVAILABLE",
                    f"contact predicate has no sensor coverage for {first!r}/{second!r}",
                )
            observations.add("contact_events")
            source.update({"entity_a": first, "entity_b": second})
        elif predicate.predicate in {"attached", "released"}:
            observations.update({"entity_states", "attachment_events", "end_effector_poses"})
            source["evaluation"] = "attachment_and_state_history"
        elif predicate.predicate in {"settled", "lifted", "object_at_pose", "within_tolerance"}:
            observations.add("entity_states")
            if predicate.predicate in {"settled", "lifted"}:
                observations.add("state_history")
            source["evaluation"] = "state_history"
        elif predicate.predicate == "base_at_pose":
            observations.add("base_state")
            source["evaluation"] = "base_pose_history"
            if "subject" in args:
                # The domain contract restricts this optional field to the
                # robot-base vocabulary. Preserve it in the compiled audit
                # record without changing the single-base verifier semantics.
                source["subject"] = args["subject"]
        else:
            source["evaluation"] = "deterministic_predicate"
        return source


def _bound_object(spec: GoalSpec, scene: SceneModel, alias: object) -> Any:
    if not isinstance(alias, str):
        raise GoalCompileError("INVALID_ENTITY_REFERENCE", "goal entity reference must be a string")
    reference = spec.bindings.get(alias)
    if reference is None:
        raise GoalCompileError("INVALID_ENTITY_REFERENCE", f"unknown goal binding {alias!r}")
    name = reference.removeprefix("scene://")
    try:
        return scene.object(name)
    except KeyError as exc:
        raise GoalCompileError("INVALID_ENTITY_REFERENCE", str(exc)) from exc


def _contact_endpoint(spec: GoalSpec, scene: SceneModel, value: object) -> str:
    """Resolve a contact endpoint to an object name or measured robot body.

    Contact sensors are attached to robot bodies, while the other endpoint is
    normally a bound scene object.  Keeping this distinction in the compiler
    allows a second robot to expose a different base-body name without adding
    a task-specific predicate or binding it as a fake scene object.
    """
    if not isinstance(value, str) or not value.strip():
        raise GoalCompileError("INVALID_ENTITY_REFERENCE", "contact endpoint must be a string")
    root = value.split(".", 1)[0]
    reference = spec.bindings.get(root)
    if reference is not None:
        return reference.removeprefix("scene://")
    if root.startswith("scene://"):
        return root.removeprefix("scene://")
    if root in _BASE_ENTITY_TERMS:
        base_bodies = [
            sensor.body
            for sensor in scene.contact_sensors
            if _looks_like_base_body(sensor.body)
        ]
        if not base_bodies:
            raise GoalCompileError(
                "CONTACT_OBSERVATION_UNAVAILABLE",
                "contact predicate names the robot base but no base-body contact sensor is declared",
            )
        return base_bodies[0]
    if any(sensor.body == root for sensor in scene.contact_sensors):
        return root
    raise GoalCompileError(
        "INVALID_ENTITY_REFERENCE",
        f"unknown contact endpoint {value!r}",
    )


def _match_region(obj: Any, region: Mapping[str, Any]) -> str | None:
    for declared in obj.regions:
        if declared.shape.value != region.get("shape"):
            continue
        if not _same_vector(declared.center, region.get("center"), 3):
            continue
        if declared.shape is ObjectType.CUBOID:
            if _same_vector(declared.size, region.get("size"), 3):
                return declared.name
        elif (
            _close(declared.radius, region.get("radius"))
            and _close(declared.height, region.get("height"))
        ):
            return declared.name
    return None


def _match_surface(obj: Any, surface: Mapping[str, Any]) -> str | None:
    declared = list(obj.surfaces)
    if obj.type is ObjectType.CUBOID and not any(item.name == "top" for item in declared):
        half_x, half_y, half_z = (float(value) / 2.0 for value in obj.size)
        from r1pro_data_gen.domain.scene import SurfaceModel

        declared.append(
            SurfaceModel(
                name="top",
                center=(0.0, 0.0, half_z),
                normal=(0.0, 0.0, 1.0),
                size=(2.0 * half_x, 2.0 * half_y),
            )
        )
    for item in declared:
        if _same_vector(item.center, surface.get("center"), 3) and _same_vector(item.size, surface.get("size"), 2):
            return item.name
    return None


def _has_pair_sensor(sensors: Any, first: str, second: str) -> bool:
    return any(
        (sensor.body == first and second in sensor.filter)
        or (sensor.body == second and first in sensor.filter)
        for sensor in sensors
    )


def _looks_like_base_body(body: str) -> bool:
    normalized = body.casefold()
    if any(token in normalized for token in ("arm", "gripper", "finger", "wheel", "steer")):
        return False
    return normalized in {"base", "base_link", "mobile_base", "chassis"}


def _has_collision_sensor(scene: SceneModel, subject: str) -> bool:
    return any(subject in sensor.filter for sensor in scene.collision_sensors)


def _half_height(obj: Any) -> float:
    return float(obj.height / 2.0 if obj.type is ObjectType.CYLINDER else obj.size[2] / 2.0)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoalCompileError("INVALID_GOAL_GEOMETRY", f"{field} must be an object")
    return value


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise GoalCompileError("INVALID_GOAL_GEOMETRY", f"{field} must be positive and finite")
    return float(value)


def _same_vector(expected: Any, actual: object, length: int) -> bool:
    if not isinstance(actual, (list, tuple)) or len(actual) != length:
        return False
    return all(_close(left, right) for left, right in zip(expected, actual))


def _close(left: object, right: object, tolerance: float = 1e-6) -> bool:
    try:
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _goal_hash(spec: GoalSpec) -> str:
    from r1pro_data_gen.domain import goal_spec_sha256

    return goal_spec_sha256(spec)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["CompiledGoalContract", "GoalCompileError", "GoalCompiler"]
