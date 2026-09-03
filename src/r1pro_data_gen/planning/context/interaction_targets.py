"""Resolve task-neutral scene references to live world-space interaction points.

The planner names an object, marker, or named region.  It does not copy a
reset-pose coordinate into a skill call.  This resolver is shared by future
interaction skills (push, place, probe, and articulated-object actions) so
each skill does not grow its own task-specific target convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from r1pro_data_gen.domain import SceneModel


class InteractionTargetError(ValueError):
    """Raised when a semantic interaction target cannot be resolved."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class InteractionTargetResolution:
    """Auditable semantic-to-world target resolution."""

    reference: str
    position: tuple[float, float, float]
    source: str
    object_name: str
    region_name: str | None = None

    def to_details(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target_reference": self.reference,
            "target_position_world": [round(float(value), 5) for value in self.position],
            "target_resolution_source": self.source,
            "target_object_name": self.object_name,
        }
        if self.region_name is not None:
            result["target_region_name"] = self.region_name
        return result


def resolve_interaction_target(
    scene: SceneModel,
    adapter: Any,
    *,
    target_ref: str | None = None,
    target_region_name: str | None = None,
    target_pose: Sequence[float] | None = None,
) -> InteractionTargetResolution:
    """Resolve one semantic target from live object state and scene geometry.

    Exactly one of ``target_ref``, ``target_region_name``, and ``target_pose``
    is accepted.  A reference may be ``scene://object`` or
    ``scene://object/region``; the latter transforms a region's local center
    by the object's live pose when the adapter exposes it.
    """
    provided = sum(value is not None for value in (target_ref, target_region_name, target_pose))
    if provided != 1:
        raise InteractionTargetError(
            "INVALID_TARGET_SPEC",
            "provide exactly one of target_ref, target_region_name, or target_pose",
        )
    if not isinstance(scene, SceneModel):
        raise InteractionTargetError("INVALID_SCENE", "interaction target needs a SceneModel")
    if target_pose is not None:
        position = _finite_vector(target_pose, "target_pose")
        return InteractionTargetResolution(
            reference="world://pose",
            position=position,
            source="explicit_world_pose",
            object_name="",
        )

    raw_reference = target_ref if target_ref is not None else target_region_name
    if target_ref is None and isinstance(raw_reference, str) and not raw_reference.startswith("scene://"):
        # ``target_region_name`` is a legacy-friendly parameter name. Accept
        # ``object/region`` there, but normalize it to the same canonical
        # scene reference used by the agent and replay contracts.
        raw_reference = f"scene://{raw_reference}"
    reference, object_name, region_name = _parse_reference(raw_reference)
    try:
        target_object = scene.object(object_name)
    except KeyError as exc:
        raise InteractionTargetError(
            "UNKNOWN_TARGET_OBJECT",
            f"interaction target object {object_name!r} is not in the scene",
        ) from exc
    if region_name is not None:
        region = next(
            (item for item in target_object.regions if item.name == region_name),
            None,
        )
        if region is None:
            raise InteractionTargetError(
                "UNKNOWN_TARGET_REGION",
                f"region {region_name!r} is not declared on object {object_name!r}",
            )
        local = np.asarray(region.center, dtype=float)
        source = "live_object_region"
    else:
        local = np.zeros(3, dtype=float)
        source = "live_object_center"
    world_position, quaternion = _live_object_pose(adapter, target_object)
    if region_name is not None:
        world_position = world_position + _quat_rotate(quaternion, local)
    return InteractionTargetResolution(
        reference=reference,
        position=tuple(float(value) for value in world_position),
        source=source,
        object_name=object_name,
        region_name=region_name,
    )


def _parse_reference(value: object) -> tuple[str, str, str | None]:
    if not isinstance(value, str) or not value.strip():
        raise InteractionTargetError("INVALID_TARGET_REF", "target reference must be non-empty")
    reference = value.strip()
    if not reference.startswith("scene://"):
        raise InteractionTargetError(
            "INVALID_TARGET_REF",
            "target reference must use scene://<object>[/<region>]",
        )
    path = reference[len("scene://") :]
    parts = path.split("/")
    if len(parts) not in {1, 2} or any(not part for part in parts):
        raise InteractionTargetError(
            "INVALID_TARGET_REF",
            "target reference must use scene://<object>[/<region>]",
        )
    object_name = parts[0]
    region_name = parts[1] if len(parts) == 2 else None
    return reference, object_name, region_name


def _finite_vector(value: Sequence[float], field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise InteractionTargetError("INVALID_TARGET_POSE", f"{field} must have shape (3,)")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InteractionTargetError("INVALID_TARGET_POSE", f"{field} must contain numbers") from exc
    if not all(math.isfinite(item) for item in vector):
        raise InteractionTargetError("INVALID_TARGET_POSE", f"{field} must contain finite numbers")
    return vector


def _live_object_pose(adapter: Any, object_model: Any) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(object_model.pos, dtype=float)
    quaternion = np.asarray(object_model.quat, dtype=float)
    if hasattr(adapter, "object_state"):
        try:
            state = adapter.object_state(object_model.name)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            state = None
        if state is not None:
            position = np.asarray(state.position, dtype=float)
            quaternion = np.asarray(state.quaternion, dtype=float)
    return position, _quat_normalize(quaternion)


def _quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise InteractionTargetError("INVALID_OBJECT_POSE", "target object quaternion is invalid")
    return quaternion / norm


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quaternion)
    q_vector = np.asarray([0.0, *vector], dtype=float)
    conjugate = np.asarray([w, -x, -y, -z], dtype=float)
    return _quat_multiply(_quat_multiply(np.asarray([w, x, y, z]), q_vector), conjugate)[1:]


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


__all__ = [
    "InteractionTargetError",
    "InteractionTargetResolution",
    "resolve_interaction_target",
]
