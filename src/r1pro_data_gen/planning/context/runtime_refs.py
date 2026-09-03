"""Safe resolution of declarative runtime references in semantic Plans."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


class RuntimeReferenceError(ValueError):
    """Raised when a declarative runtime reference is invalid or unavailable."""


_ALLOWED_DETAIL_KEYS = frozenset({"position", "quaternion", "contact_forces", "joint_positions"})
_ALLOWED_VALUE_TYPES = frozenset({"number", "array", "object"})
_ALLOWED_FRAMES = frozenset({"world", "base"})


def resolve_parameters(
    value: Any,
    *,
    stage_results: Mapping[str, Any],
    observation: Any,
    scene: Any,
    current_stage: str,
    stage_outputs: Mapping[str, Iterable[str]] | None = None,
    stage_dependencies: Iterable[str] | None = None,
    frame_converter: Callable[[Any, str, str], Any] | None = None,
) -> Any:
    """Resolve all typed reference objects in a JSON-compatible parameter tree.

    References are deliberately data-only.  No attribute access, expression,
    indexing, function call, or string interpolation is accepted.

    ``frame_converter`` (optional) lets the caller supply a calibrated
    world<->base converter (e.g. one based on live simulator link geometry)
    that supersedes the raw ``observation.base_pose`` fallback in
    :func:`_convert_frame`.
    """
    if isinstance(value, Mapping):
        if "ref" in value:
            return resolve_reference(
                value,
                stage_results=stage_results,
                observation=observation,
                scene=scene,
                current_stage=current_stage,
                stage_outputs=stage_outputs,
                stage_dependencies=stage_dependencies,
                frame_converter=frame_converter,
            )
        return {
            str(key): resolve_parameters(
                item,
                stage_results=stage_results,
                observation=observation,
                scene=scene,
                current_stage=current_stage,
                stage_outputs=stage_outputs,
                stage_dependencies=stage_dependencies,
                frame_converter=frame_converter,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_parameters(
                item,
                stage_results=stage_results,
                observation=observation,
                scene=scene,
                current_stage=current_stage,
                stage_outputs=stage_outputs,
                stage_dependencies=stage_dependencies,
                frame_converter=frame_converter,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            resolve_parameters(
                item,
                stage_results=stage_results,
                observation=observation,
                scene=scene,
                current_stage=current_stage,
                stage_outputs=stage_outputs,
                stage_dependencies=stage_dependencies,
                frame_converter=frame_converter,
            )
            for item in value
        ]
    return value


def resolve_reference(
    reference: Mapping[str, Any],
    *,
    stage_results: Mapping[str, Any],
    observation: Any,
    scene: Any,
    current_stage: str,
    stage_outputs: Mapping[str, Iterable[str]] | None = None,
    stage_dependencies: Iterable[str] | None = None,
    frame_converter: Callable[[Any, str, str], Any] | None = None,
) -> Any:
    """Resolve one reference against already completed runtime state."""
    allowed = {"ref", "value_type", "shape", "frame", "offset"}
    unknown = set(reference) - allowed
    if unknown:
        raise RuntimeReferenceError(f"reference contains unknown fields: {sorted(unknown)}")
    path = reference.get("ref")
    if not isinstance(path, str) or not path:
        raise RuntimeReferenceError("reference.ref must be a non-empty string")
    source_frame, value = _read_source(
        path,
        stage_results=stage_results,
        observation=observation,
        scene=scene,
        current_stage=current_stage,
        stage_outputs=stage_outputs,
        stage_dependencies=stage_dependencies,
    )
    target_frame = reference.get("frame")
    if target_frame is not None:
        if target_frame not in _ALLOWED_FRAMES:
            raise RuntimeReferenceError(f"unsupported reference frame: {target_frame!r}")
        value = _convert_frame(value, source_frame, target_frame, observation, frame_converter=frame_converter)
    if "offset" in reference:
        offset = reference["offset"]
        if not isinstance(offset, list) or len(offset) != 3:
            raise RuntimeReferenceError("reference.offset must have shape (3,)")
        if not all(_finite_number(item) for item in offset):
            raise RuntimeReferenceError("reference.offset must contain finite numbers")
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise RuntimeReferenceError("reference.offset requires a 3-vector value")
        value = [float(a) + float(b) for a, b in zip(value, offset)]
    _validate_value(value, reference)
    return value


def _read_source(
    path: str,
    *,
    stage_results: Mapping[str, Any],
    observation: Any,
    scene: Any,
    current_stage: str,
    stage_outputs: Mapping[str, Iterable[str]] | None = None,
    stage_dependencies: Iterable[str] | None = None,
) -> tuple[str, Any]:
    parts = path.split(".")
    if any(not part or part in {"__dict__", "__class__"} for part in parts):
        raise RuntimeReferenceError("reference path contains an invalid segment")
    if parts[0] == "stage":
        if len(parts) != 4 or parts[2] != "details":
            raise RuntimeReferenceError("stage references must be stage.<name>.details.<field>")
        stage_name, field = parts[1], parts[3]
        if stage_name == current_stage:
            raise RuntimeReferenceError("a stage cannot reference its own output")
        if stage_name not in stage_results:
            raise RuntimeReferenceError(f"referenced stage has not completed: {stage_name!r}")
        if stage_dependencies is not None and stage_name not in set(stage_dependencies):
            raise RuntimeReferenceError(
                f"referenced stage is not a dependency of {current_stage!r}: {stage_name!r}"
            )
        if stage_outputs is not None and field not in set(stage_outputs.get(stage_name, ())):
            raise RuntimeReferenceError(
                f"stage {stage_name!r} did not declare output {field!r}"
            )
        result = stage_results[stage_name]
        details = result.details if hasattr(result, "details") else result.get("details", {})
        if not isinstance(details, Mapping) or field not in details:
            raise RuntimeReferenceError(f"stage {stage_name!r} has no detail {field!r}")
        result_skill = getattr(result, "skill", None)
        source_frame = "base" if result_skill == "query_ee_pose" else "world"
        return (source_frame if field in {"position", "quaternion"} else "none", details[field])
    if parts == ["observation", "base_pose"]:
        value = getattr(observation, "base_pose", None)
        if value is None:
            raise RuntimeReferenceError("observation has no base_pose")
        return "world", list(value)
    if len(parts) == 4 and parts[:2] == ["scene", "object"] and parts[3] in {"position", "quaternion"}:
        name, field = parts[2], parts[3]
        try:
            obj = scene.object(name)
        except (AttributeError, KeyError, ValueError) as exc:
            raise RuntimeReferenceError(f"unknown scene object: {name!r}") from exc
        value = obj.pos if field == "position" else obj.quat
        return "world", list(value)
    raise RuntimeReferenceError(f"unsupported reference source: {path!r}")


def _convert_frame(
    value: Any,
    source: str,
    target: str,
    observation: Any,
    *,
    frame_converter: Callable[[Any, str, str], Any] | None = None,
) -> Any:
    if source == target or source == "none":
        return value
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RuntimeReferenceError("world/base conversion requires a 3-vector")
    # A caller-provided converter (e.g. online link calibration) wins over the
    # raw base_pose fallback, which does not account for the URDF/USD origin
    # offset that appears once the arm leaves the neutral home.
    if frame_converter is not None:
        converted = frame_converter(value, source, target)
        if converted is not None:
            return converted
    pose = getattr(observation, "base_pose", None)
    if pose is None or len(pose) != 3:
        raise RuntimeReferenceError("world/base conversion requires observation.base_pose")
    x, y, yaw = (float(item) for item in pose)
    vx, vy, vz = (float(item) for item in value)
    c, s = math.cos(yaw), math.sin(yaw)
    if source == "world" and target == "base":
        return [c * (vx - x) + s * (vy - y), -s * (vx - x) + c * (vy - y), vz]
    if source == "base" and target == "world":
        return [x + c * vx - s * vy, y + s * vx + c * vy, vz]
    raise RuntimeReferenceError(f"cannot convert frame {source!r} to {target!r}")


def _validate_value(value: Any, reference: Mapping[str, Any]) -> None:
    value_type = reference.get("value_type")
    if value_type is not None and value_type not in _ALLOWED_VALUE_TYPES:
        raise RuntimeReferenceError(f"unsupported reference value_type: {value_type!r}")
    shape = reference.get("shape")
    if shape is not None:
        if not isinstance(shape, list) or any(not isinstance(item, int) or item < 0 for item in shape):
            raise RuntimeReferenceError("reference.shape must be a non-negative integer array")
        actual = _shape(value)
        if actual != tuple(shape):
            raise RuntimeReferenceError(f"reference value shape {actual} does not match {tuple(shape)}")
    if value_type == "number" and not _finite_number(value):
        raise RuntimeReferenceError("reference value must be a finite number")
    if value_type == "array" and not isinstance(value, list):
        raise RuntimeReferenceError("reference value must be an array")
    _ensure_finite(value)


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return (len(value),) + (_shape(value[0]) if value else ())


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _ensure_finite(value: Any) -> None:
    if _finite_number(value) or value is None or isinstance(value, str) or isinstance(value, bool):
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _ensure_finite(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _ensure_finite(item)
        return
    if isinstance(value, np.generic):
        if not _finite_number(value.item()):
            raise RuntimeReferenceError("reference result contains a non-finite value")
        return
    raise RuntimeReferenceError("reference result is not JSON-compatible")


__all__ = ["RuntimeReferenceError", "resolve_parameters", "resolve_reference"]
