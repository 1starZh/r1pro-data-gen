"""Semantic transfer of a movable object between scene support surfaces.

This is a reusable task-level capability, not a task policy.  The planner
provides the object and destination roles; the skill composes the existing
generic grasp, carry, and release primitives.  Source support is inferred from
the live object pose, so the same capability covers a floor object, a tabletop
object, or an object on another declared support.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.base import ParamSpec, SkillResult
from r1pro_data_gen.execution.contracts import PhysicalSafetyViolation
from ..core.sides import require_side, resolve_side


class TransferObjectBetweenSupports:
    """Complete a semantic pick-carry-place-release transition.

    The operation is intentionally one public skill so a closed-loop agent can
    select the complete manipulation unit after the navigation phase has
    already been accepted.  The child results remain in the diagnostic record,
    while final task acceptance is still decided by the physical GoalSpec
    verifier (attachment, destination placement, release, and settling).
    """

    name = "transfer_object_between_supports"
    tier = "semantic"
    exposed = False
    description = (
        "Move a movable graspable object from its current support to a named "
        "destination support: perform a geometry-aware low/high grasp, verify "
        "attachment, lift and carry clear of surfaces, place inside the target "
        "region, release, and leave the object to settle. Source support and "
        "all Cartesian/joint details are inferred from live scene geometry."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec(
            "string", "Top-level scene object to transfer", required=True
        ),
        "target_region_name": ParamSpec(
            "string",
            "Top-level scene marker/object defining the destination region",
            required=True,
        ),
        "support_surface_name": ParamSpec(
            "string",
            "Top-level scene object physically supporting the destination",
            required=True,
        ),
        "side": ParamSpec(
            "string",
            "Arm side used for the transfer; auto selects one arm from live geometry",
            default="auto",
            enum=("auto", "left", "right"),
        ),
        "settle_steps": ParamSpec(
            "integer",
            "Physics steps to hold after release",
            default=12,
            minimum=1,
            maximum=240,
        ),
    }

    def __init__(
        self,
        grasp: Any,
        carry: Any,
        release: Any,
        handoff: Any = None,
        base_reposition: Any = None,
    ) -> None:
        self.grasp = grasp
        self.carry = carry
        self.release = release
        self.handoff = handoff
        self.base_reposition = base_reposition

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        target_region_name: str | None = None,
        support_surface_name: str | None = None,
        side: str = "auto",
        settle_steps: int = 12,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if scene is None or not hasattr(scene, "object"):
            return _failure("missing_scene", "transfer requires a scene")
        if not object_name or not target_region_name or not support_surface_name:
            raise ValueError(
                "transfer_object_between_supports requires object, target region, "
                "and support surface names"
            )
        requested_side = require_side(side, allow_auto=True)
        try:
            object_model = scene.object(object_name)
            target_model = scene.object(target_region_name)
            support_model = scene.object(support_surface_name)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return _failure(
                "invalid_scene_roles",
                "transfer scene roles could not be resolved",
                error=str(exc),
            )
        object_capabilities = _capability_names(object_model)
        if object_capabilities and "graspable" not in object_capabilities:
            return _failure(
                "object_not_graspable",
                "transfer source does not declare graspable capability",
                object_name=object_name,
            )
        support_capabilities = _capability_names(support_model)
        if support_capabilities and "supports_objects" not in support_capabilities:
            return _failure(
                "destination_not_support",
                "transfer destination support does not declare supports_objects capability",
                support_surface_name=support_surface_name,
            )
        if not getattr(target_model, "regions", ()) and getattr(target_model, "size", None) is None:
            return _failure(
                "destination_geometry_unavailable",
                "transfer destination has no region or placement geometry",
                target_region_name=target_region_name,
            )

        source_support_name = _source_support_name(adapter, scene, object_model, object_name)
        source_level = _source_support_level(scene, object_model, object_name, adapter, source_support_name)
        destination_level = _support_top_level(support_model)
        object_height = _vertical_extent(object_model)
        handoff_needed = bool(
            self.handoff is not None
            and destination_level - source_level > _WHOLE_BODY_HANDOFF_ELEVATION_M
        )
        phases: list[dict[str, Any]] = []

        grasp_result = self.grasp.execute(
            adapter,
            scene=scene,
            object_name=object_name,
            side=requested_side,
            step_hook=step_hook,
        )
        phases.append(_phase("grasp", grasp_result))
        if not grasp_result.success:
            return _failure(
                "grasp_phase_failed",
                "transfer grasp phase failed",
                phases=phases,
                object_name=object_name,
                target_region_name=target_region_name,
                support_surface_name=support_surface_name,
            )

        # The grasp skill owns auto-side selection. Carry/release must use the
        # concrete side that actually established the measured attachment.
        selected_side = grasp_result.details.get("side")
        if not isinstance(selected_side, str) or selected_side not in {"left", "right"}:
            selected_side = resolve_side(requested_side, adapter, object_name=object_name)
        side = selected_side

        if handoff_needed:
            target_height = destination_level + object_height / 2.0 + _TRANSFER_LIFT_CLEARANCE_M
            handoff_result = self.handoff.execute(
                adapter,
                scene=scene,
                object_name=object_name,
                target_posture="carry",
                target_height_m=target_height,
                source_support_name=source_support_name,
                side=side,
                step_hook=step_hook,
            )
            phases.append(_phase("whole_body_handoff", handoff_result))
            if not handoff_result.success:
                return _failure(
                    "whole_body_handoff_failed",
                    "transfer could not complete the lift and whole-body posture handoff",
                    phases=phases,
                    object_name=object_name,
                    target_region_name=target_region_name,
                    support_surface_name=support_surface_name,
                    source_support_name=source_support_name,
                )

        carry_params = {
            "adapter": adapter,
            "scene": scene,
            "object_name": object_name,
            "target_region_name": target_region_name,
            "support_surface_name": support_surface_name,
            "side": side,
            "step_hook": step_hook,
        }
        if handoff_needed:
            # The handoff already performed the lift.  Carry must refresh the
            # live grasp context and start its retract/traverse/extend path
            # from the post-transition posture rather than lifting twice.
            carry_params["skip_lift"] = True
        carry_result = self.carry.execute(
            **carry_params,
        )
        phases.append(_phase("carry_and_place", carry_result))
        if (
            not carry_result.success
            and handoff_needed
            and self.base_reposition is not None
            and _carry_still_attached(carry_result)
        ):
            # A low-source transfer can require two distinct manipulation
            # stances: a rearward/low grasp stance and a destination-side
            # upright stance.  Repositioning here is an internal transport
            # phase of the complete transfer, not a separate navigation
            # acceptance unit.  Candidate poses are resolved from the named
            # support's geometry, never from task coordinates.
            for index, approach_side in enumerate(("west", "south", "east", "north"), start=1):
                _unlock_internal_hold(adapter)
                try:
                    reposition = self.base_reposition.execute(
                        adapter,
                        scene=scene,
                        target_ref=f"scene://{support_surface_name}",
                        purpose="dropoff",
                        approach_side=approach_side,
                        motion_mode="holonomic",
                        step_hook=step_hook,
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                    if isinstance(exc, PhysicalSafetyViolation):
                        raise
                    reposition = SkillResult(False, "base_navigate_to", details={"reason": str(exc)})
                phases.append(_phase(f"base_reposition_for_carry_{index}", reposition))
                if not reposition.success:
                    continue
                carry_retry = self.carry.execute(**carry_params)
                phases.append(_phase(f"carry_and_place_retry_{index}", carry_retry))
                carry_result = carry_retry
                if carry_result.success:
                    break
        if not carry_result.success:
            return _failure(
                "carry_phase_failed",
                "transfer carry/place phase failed",
                phases=phases,
                object_name=object_name,
                target_region_name=target_region_name,
                support_surface_name=support_surface_name,
            )

        release_result = self.release.execute(
            adapter,
            scene=scene,
            object_name=object_name,
            side=side,
            settle_steps=settle_steps,
            step_hook=step_hook,
        )
        phases.append(_phase("release_and_settle", release_result))
        if not release_result.success:
            return _failure(
                "release_phase_failed",
                "transfer release phase failed",
                phases=phases,
                object_name=object_name,
                target_region_name=target_region_name,
                support_surface_name=support_surface_name,
            )

        return SkillResult(
            True,
            self.name,
            metrics={
                "completed_phases": float(len(phases)),
                "settle_steps": float(settle_steps),
            },
            details={
                "reason": "complete support-to-support transfer executed",
                "object_name": object_name,
                "target_region_name": target_region_name,
                "support_surface_name": support_surface_name,
                "side": side,
                "phases": phases,
                "source_support_inferred": True,
                "whole_body_handoff": handoff_needed,
                "source_support_name": source_support_name,
            },
        )


def _phase(name: str, result: SkillResult) -> dict[str, Any]:
    """Keep nested diagnostics JSON-safe without changing acceptance semantics."""
    return {
        "name": name,
        "skill": result.skill,
        "success": bool(result.success),
        "metrics": _json_safe(result.metrics),
        "details": _json_safe(result.details),
    }


def _failure(code: str, reason: str, **details: Any) -> SkillResult:
    return SkillResult(
        False,
        "transfer_object_between_supports",
        details={"reason": reason, "failure_code": code, **_json_safe(details)},
    )


def _capability_names(model: Any) -> set[str]:
    """Normalize optional legacy/enum capability declarations."""
    values = getattr(model, "capabilities", ()) or ()
    return {
        str(getattr(value, "value", value))
        for value in values
    }


_WHOLE_BODY_HANDOFF_ELEVATION_M = 0.20
_TRANSFER_LIFT_CLEARANCE_M = 0.13


def _source_support_name(adapter: Any, scene: Any, object_model: Any, object_name: str) -> str | None:
    """Infer the source support before grasp changes the live object pose."""
    try:
        world = adapter.object_position(object_name)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        world = getattr(object_model, "pos", None)
    if world is None:
        return None
    from .carry import _infer_source_support_surface

    try:
        return _infer_source_support_surface(scene, object_model, tuple(float(v) for v in world))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _source_support_level(
    scene: Any,
    object_model: Any,
    object_name: str,
    adapter: Any,
    source_support_name: str | None,
) -> float:
    if source_support_name is not None:
        try:
            return float(scene.object(source_support_name).top_z)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    try:
        position = adapter.object_position(object_name)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        position = getattr(object_model, "pos", (0.0, 0.0, 0.0))
    return float(position[2]) - _vertical_extent(object_model) / 2.0


def _support_top_level(model: Any) -> float:
    top = getattr(model, "top_z", None)
    if top is not None:
        try:
            return float(top)
        except (TypeError, ValueError):
            pass
    position = getattr(model, "pos", (0.0, 0.0, 0.0))
    size = getattr(model, "size", None)
    height = float(size[2]) if size is not None and len(size) >= 3 else 0.0
    return float(position[2]) + height / 2.0


def _vertical_extent(model: Any) -> float:
    from r1pro_data_gen.domain import object_vertical_extent_m

    try:
        return max(0.0, float(object_vertical_extent_m(model)))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return value


def _carry_still_attached(result: SkillResult) -> bool:
    context = result.details.get("grasp_context") if isinstance(result.details, dict) else None
    if isinstance(context, dict) and "attached" in context:
        return bool(context["attached"])
    # A carry backend may not return a context on a pre-planning failure.  The
    # composite may safely ask its live backend to re-read the context on the
    # next retry; callers that do report an explicit detach must not retry.
    return result.details.get("reason") not in {"object is not attached", "attachment lost"}


def _unlock_internal_hold(adapter: Any) -> None:
    """Unlock only the standard wheel/torso hold created by a skill."""
    if not getattr(adapter, "joint_mask_locked", False) or not hasattr(adapter, "unlock_joint_mask"):
        return
    groups = set(getattr(adapter, "joint_lock_groups", getattr(adapter, "_joint_lock_groups", ())))
    if groups.issubset({"steer", "wheel", "torso"}):
        adapter.unlock_joint_mask()


__all__ = ["TransferObjectBetweenSupports"]
