"""Semantic planar pushing for scene-authored movable objects."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from r1pro_data_gen.domain import ObjectCapability, object_xy_half_extents_m, object_xy_radius_m
from r1pro_data_gen.planning.context.interaction_targets import (
    InteractionTargetError,
    resolve_interaction_target,
)
from r1pro_data_gen.planning.navigation.contract import (
    NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M,
    NAVIGATION_INFLATION_CLEARANCE_M,
)
from r1pro_data_gen.robot.chassis import default_footprint_radius_m
from r1pro_data_gen.robot import wheel_commands

from ..core.base import ParamSpec, SkillResult
from ..mobility.base_motion import (
    BaseNavigateTo,
    _brake_until_stopped,
    _forward_tracking_command,
    _footprint_radius as _navigation_footprint_radius,
    _read_base,
    _set_drive_targets,
    _wrap_pi,
)


class PushObjectTo:
    """Push a scene-authored pushable object toward a semantic target.

    The skill owns the generic state transition: resolve the live object and
    target, approach from the opposite side, and drive in the target direction
    while re-reading the object pose. It does not contain object names, task
    recipes, or reset-pose coordinates.
    """

    name = "push_object_to"
    tier = "semantic"
    exposed = True
    description = (
        "Push a named pushable object toward a semantic target without "
        "grasping. Use when the goal forbids grasping or asks to push, and "
        "live capabilities include pushable. Do not combine with grasp or "
        "carry, and do not use when the goal requires attachment. Give exactly "
        "one of target_ref, target_region_name, or target_pose."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Pushable scene object", required=True),
        "target_ref": ParamSpec(
            "string",
            "Semantic target such as scene://goal or scene://table/region",
            default=None,
        ),
        "target_region_name": ParamSpec(
            "string",
            "Compatibility form for a scene object or object/region target",
            default=None,
        ),
        "target_pose": ParamSpec(
            "array",
            "Explicit world target (x, y, z) when no scene reference is available",
            default=None,
            shape=(3,),
        ),
        "position_tolerance_m": ParamSpec(
            "number",
            "Final planar object position tolerance (m)",
            default=0.05,
            minimum=0.005,
            exposed=False,
        ),
        "contact_clearance_m": ParamSpec(
            "number",
            "Extra base-to-object contact clearance (m)",
            default=0.03,
            minimum=0.0,
            exposed=False,
        ),
        "v_max": ParamSpec("number", "Maximum forward push speed (m/s)", default=0.05, minimum=0.005, exposed=False),
        "omega_max": ParamSpec("number", "Maximum heading rate (rad/s)", default=0.20, minimum=0.02, exposed=False),
        "max_steps": ParamSpec("integer", "Physics-step budget for approach and push", default=900, minimum=30, exposed=False),
        "stall_steps": ParamSpec("integer", "Steps without object progress before failure", default=90, minimum=10, exposed=False),
        "settle_steps": ParamSpec(
            "integer",
            "Physics steps to brake the base and observe the object after reaching the target",
            default=60,
            minimum=1,
            maximum=240,
            exposed=False,
        ),
    }

    def __init__(self, navigator: Any | None = None) -> None:
        self.navigator = navigator or BaseNavigateTo()

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        target_ref: str | None = None,
        target_region_name: str | None = None,
        target_pose: Sequence[float] | None = None,
        position_tolerance_m: float = 0.05,
        contact_clearance_m: float = 0.03,
        v_max: float = 0.05,
        omega_max: float = 0.20,
        max_steps: int = 900,
        stall_steps: int = 90,
        settle_steps: int = 60,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if scene is None or not hasattr(scene, "object"):
            return self._failure("missing_scene", "push_object_to requires a SceneModel")
        if not object_name:
            raise ValueError("push_object_to requires object_name")
        try:
            object_model = scene.object(object_name)
        except KeyError:
            return self._failure("unknown_object", f"object {object_name!r} is not in the scene")
        capabilities = set(object_model.capabilities)
        if ObjectCapability.PUSHABLE not in capabilities:
            return self._failure(
                "object_not_pushable",
                f"object {object_name!r} is not authored with the pushable capability",
            )
        if ObjectCapability.MOVABLE not in capabilities:
            return self._failure(
                "object_not_movable",
                f"object {object_name!r} is not authored with the movable capability",
            )
        if isinstance(max_steps, bool) or int(max_steps) < 1:
            raise ValueError("max_steps must be positive")
        if isinstance(stall_steps, bool) or int(stall_steps) < 1:
            raise ValueError("stall_steps must be positive")
        if isinstance(settle_steps, bool) or not 1 <= int(settle_steps) <= 240:
            raise ValueError("settle_steps must be between 1 and 240")

        try:
            target = resolve_interaction_target(
                scene,
                adapter,
                target_ref=target_ref,
                target_region_name=target_region_name,
                target_pose=target_pose,
            )
            object_position = np.asarray(adapter.object_position(object_name), dtype=float)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, InteractionTargetError) as exc:
            return self._failure("target_unavailable", str(exc))
        if target.object_name == object_name:
            return self._failure("self_target", "push target must not be the pushed object")
        target_xy = np.asarray(target.position[:2], dtype=float)
        object_xy = object_position[:2]
        initial_distance = float(np.linalg.norm(target_xy - object_xy))
        tolerance = max(0.005, float(position_tolerance_m))
        details = target.to_details()
        details.update(
            {
                "object_name": object_name,
                "initial_object_position": [float(v) for v in object_position],
                "initial_target_distance_m": initial_distance,
            }
        )
        if initial_distance <= tolerance:
            settled_steps = _settle_after_push(
                adapter,
                settle_steps=int(settle_steps),
                step_hook=step_hook,
            )
            return SkillResult(
                True,
                self.name,
                metrics={
                    "steps": float(settled_steps),
                    "object_distance_m": initial_distance,
                    "settle_steps": float(settled_steps),
                },
                details={**details, "settle_steps": settled_steps},
            )
        direction = (target_xy - object_xy) / initial_distance
        approach_yaw = math.atan2(float(direction[1]), float(direction[0]))
        object_support_radius = _object_support_radius(object_model, direction)
        standoff = _contact_standoff(
            adapter,
            direction=direction,
            yaw=approach_yaw,
            object_support_radius=object_support_radius,
            contact_clearance_m=contact_clearance_m,
        )
        navigation_standoff = max(
            standoff + NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M,
            _navigation_standoff(adapter, scene, object_model),
        )
        # Navigation must end outside the obstacle-inflated object.  The
        # push controller then closes the remaining gap using live geometry.
        approach_xy = object_xy - direction * navigation_standoff
        approach = self.navigator.execute(
            adapter,
            scene=scene,
            target=[float(approach_xy[0]), float(approach_xy[1]), approach_yaw],
            motion_mode="forward",
            step_hook=step_hook,
        )
        if not approach.success:
            details.update(
                {
                    "failure_code": "push_approach_failed",
                    "approach": approach.details,
                    "approach_metrics": approach.metrics,
                }
            )
            return SkillResult(False, self.name, metrics=approach.metrics, details=details)

        push_metrics = _push_forward(
            adapter,
            object_name=object_name,
            target_xy=target_xy,
            direction=direction,
            standoff=standoff,
            object_support_radius=object_support_radius,
            contact_clearance_m=contact_clearance_m,
            tolerance=tolerance,
            v_max=max(0.005, float(v_max)),
            omega_max=max(0.02, float(omega_max)),
            max_steps=int(max_steps),
            stall_steps=int(stall_steps),
            step_hook=step_hook,
        )
        post_push_settle_steps = _settle_after_push(
            adapter,
            settle_steps=int(settle_steps),
            step_hook=step_hook,
        )
        final_position = np.asarray(adapter.object_position(object_name), dtype=float)
        final_distance = float(np.linalg.norm(target_xy - final_position[:2]))
        success = bool(final_distance <= tolerance)
        details.update(
            {
                "approach": approach.details,
                "approach_metrics": approach.metrics,
                "approach_pose": [float(approach_xy[0]), float(approach_xy[1]), approach_yaw],
                "standoff_m": standoff,
                "navigation_standoff_m": navigation_standoff,
                "settle_steps": post_push_settle_steps,
                "initial_push_direction": [float(v) for v in direction],
                "final_object_position": [float(v) for v in final_position],
                "failure_code": None if success else push_metrics.get("failure_code", "target_not_reached"),
            }
        )
        return SkillResult(
            success,
            self.name,
            metrics={
                "steps": (
                    float(approach.metrics.get("steps", 0.0))
                    + float(push_metrics["steps"])
                    + float(post_push_settle_steps)
                ),
                "object_distance_m": final_distance,
                "object_displacement_m": float(push_metrics["object_displacement_m"]),
                "stall_steps": float(push_metrics["stall_steps"]),
                "settle_steps": float(post_push_settle_steps),
            },
            details=details,
        )

    def _failure(self, code: str, reason: str) -> SkillResult:
        return SkillResult(False, self.name, details={"failure_code": code, "reason": reason})


def _push_forward(
    adapter: Any,
    *,
    object_name: str,
    target_xy: np.ndarray,
    direction: np.ndarray,
    standoff: float,
    tolerance: float,
    object_support_radius: float | None = None,
    contact_clearance_m: float = 0.03,
    v_max: float,
    omega_max: float,
    max_steps: int,
    stall_steps: int,
    step_hook: Callable[[], None] | None,
) -> dict[str, float | str]:
    start = np.asarray(adapter.object_position(object_name), dtype=float)[:2]
    previous_distance = float(np.linalg.norm(target_xy - start))
    best_distance = previous_distance
    stalled = 0
    last_position = start.copy()
    for index in range(max_steps):
        current = np.asarray(adapter.object_position(object_name), dtype=float)[:2]
        remaining = float(np.linalg.norm(target_xy - current))
        if remaining <= tolerance:
            return {
                "steps": float(index),
                "object_distance_m": remaining,
                "object_displacement_m": float(np.linalg.norm(current - start)),
                "stall_steps": float(stalled),
            }
        bx, by, yaw = _read_base(adapter)
        current_standoff = standoff
        if object_support_radius is not None:
            current_standoff = _contact_standoff(
                adapter,
                direction=direction,
                yaw=yaw,
                object_support_radius=object_support_radius,
                contact_clearance_m=contact_clearance_m,
            )
        contact = current - direction * current_standoff
        contact_error = np.asarray([contact[0] - bx, contact[1] - by], dtype=float)
        contact_distance = float(np.linalg.norm(contact_error))
        desired_yaw = math.atan2(float(direction[1]), float(direction[0]))
        heading_error = _wrap_pi(desired_yaw - yaw)
        if contact_distance > 0.075:
            vx, _, omega, _ = _forward_tracking_command(
                float(contact_error[0]),
                float(contact_error[1]),
                yaw,
                v_max,
                omega_max,
            )
        else:
            # Keep a small positive drive while in contact. The object pose,
            # not a fixed base endpoint, is the moving controller target.
            vx = min(v_max, max(0.01, 1.5 * remaining))
            omega = max(-omega_max, min(omega_max, 2.0 * heading_error))
        cmds = wheel_commands(vx=vx, vy=0.0, omega=omega)
        _set_drive_targets(
            adapter,
            cmds,
            ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3"),
            ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3"),
            None,
        )
        adapter.step()
        if step_hook is not None:
            step_hook()
        updated = np.asarray(adapter.object_position(object_name), dtype=float)[:2]
        updated_distance = float(np.linalg.norm(target_xy - updated))
        progress = previous_distance - updated_distance
        if updated_distance < best_distance - 1e-4:
            best_distance = updated_distance
            stalled = 0
        elif progress <= 1e-5:
            stalled += 1
        else:
            stalled = 0
        if stalled >= stall_steps:
            return {
                "steps": float(index + 1),
                "object_distance_m": updated_distance,
                "object_displacement_m": float(np.linalg.norm(updated - start)),
                "stall_steps": float(stalled),
                "failure_code": "object_not_moving",
            }
        previous_distance = updated_distance
        last_position = updated
    final_distance = float(np.linalg.norm(target_xy - last_position))
    return {
        "steps": float(max_steps),
        "object_distance_m": final_distance,
        "object_displacement_m": float(np.linalg.norm(last_position - start)),
        "stall_steps": float(stalled),
        "failure_code": "push_action_budget_exhausted",
    }


def _settle_after_push(
    adapter: Any,
    *,
    settle_steps: int,
    step_hook: Callable[[], None] | None,
) -> int:
    """Brake the base and retain a stable post-push evidence tail."""
    steps = max(1, min(240, int(settle_steps)))
    brake_steps = int(_brake_until_stopped(adapter, max_steps=min(60, steps), step_hook=step_hook))
    remaining = max(0, steps - brake_steps)
    steer_joints = ("steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3")
    wheel_joints = ("wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3")
    for _ in range(remaining):
        if hasattr(adapter, "set_targets"):
            adapter.set_targets(
                position={name: 0.0 for name in steer_joints},
                velocity={name: 0.0 for name in wheel_joints},
            )
        adapter.step()
        if step_hook is not None:
            step_hook()
    return brake_steps + remaining


def _footprint_radius(adapter: Any) -> float:
    if hasattr(adapter, "base_footprint"):
        try:
            value = float(adapter.base_footprint()["circumscribed_radius_m"])
            if math.isfinite(value) and value > 0.0:
                return value
        except (KeyError, TypeError, ValueError, RuntimeError):
            pass
    return float(default_footprint_radius_m())


def _object_support_radius(model: Any, direction: np.ndarray) -> float:
    """Return a conservative object support distance along ``direction``.

    ``object_xy_radius_m`` is useful for isotropic objects, but using the
    smaller cuboid side as a push clearance can underestimate a rotated or
    rectangular object.  The scene geometry helper returns world-XY half
    extents, so its support function is safe for all primitive shapes in the
    current scene schema and remains independent of object names.
    """
    vector = np.asarray(direction, dtype=float)[:2]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not math.isfinite(norm):
        return float(object_xy_radius_m(model))
    vector = vector / norm
    try:
        half_x, half_y = object_xy_half_extents_m(model)
        support = abs(float(vector[0])) * float(half_x) + abs(float(vector[1])) * float(half_y)
        if math.isfinite(support) and support > 0.0:
            return support
    except (AttributeError, TypeError, ValueError):
        pass
    return float(object_xy_radius_m(model))


def _base_support_radius(adapter: Any, direction: np.ndarray, yaw: float) -> float:
    """Return the chassis support distance in a world push direction.

    Navigation uses the circumscribed radius because it is orientation-free.
    Pushing is different: the base is intentionally aligned with the push
    direction, so the contact point should use the directional support of the
    authored footprint.  Falling back to the circumscribed radius keeps custom
    adapters safe when they do not expose rectangular footprint dimensions.
    """
    vector = np.asarray(direction, dtype=float)[:2]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not math.isfinite(norm):
        return _footprint_radius(adapter)
    try:
        footprint = adapter.base_footprint()
        extents = None
        try:
            extents = {
                name: float(footprint[name])
                for name in ("front_extent_m", "rear_extent_m", "left_extent_m", "right_extent_m")
            }
        except (KeyError, TypeError, ValueError):
            pass
        if extents is not None and all(
            math.isfinite(value) and value > 0.0 for value in extents.values()
        ):
            heading = math.atan2(float(vector[1]), float(vector[0]))
            relative = _wrap_pi(heading - float(yaw))
            local_x = math.cos(relative)
            local_y = math.sin(relative)
            return (
                max(local_x, 0.0) * extents["front_extent_m"]
                + max(-local_x, 0.0) * extents["rear_extent_m"]
                + max(local_y, 0.0) * extents["left_extent_m"]
                + max(-local_y, 0.0) * extents["right_extent_m"]
            )
        half_length = float(footprint["half_length_m"])
        half_width = float(footprint["half_width_m"])
        if (
            math.isfinite(half_length)
            and math.isfinite(half_width)
            and half_length > 0.0
            and half_width > 0.0
        ):
            heading = math.atan2(float(vector[1]), float(vector[0]))
            relative = _wrap_pi(heading - float(yaw))
            return abs(math.cos(relative)) * half_length + abs(math.sin(relative)) * half_width
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        pass
    return _footprint_radius(adapter)


def _navigation_standoff(
    adapter: Any,
    scene: Any,
    model: Any,
) -> float:
    """Return a free navigation distance before the physical push closure."""
    try:
        half_x, half_y = object_xy_half_extents_m(model)
        object_bound = math.hypot(float(half_x), float(half_y))
    except (AttributeError, TypeError, ValueError):
        object_bound = object_xy_radius_m(model)
    # BaseNavigateTo inflates every collidable object by its planning
    # footprint and the shared hard clearance.  Add one grid cell so the
    # snapped target does not land on the occupied boundary.
    nav_footprint = _navigation_footprint_radius(adapter, scene)
    return (
        float(nav_footprint)
        + float(object_bound)
        + NAVIGATION_INFLATION_CLEARANCE_M
        + NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M
    )


def _contact_standoff(
    adapter: Any,
    *,
    direction: np.ndarray,
    yaw: float,
    object_support_radius: float,
    contact_clearance_m: float,
) -> float:
    """Compute the center-to-center distance needed for physical contact."""
    return (
        _base_support_radius(adapter, direction, yaw)
        + max(0.0, float(object_support_radius))
        + max(0.0, float(contact_clearance_m))
    )


__all__ = ["PushObjectTo"]
