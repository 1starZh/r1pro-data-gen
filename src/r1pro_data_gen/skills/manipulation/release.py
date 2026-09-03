"""Semantic release skill: open a gripper, lift the hand, and settle."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from r1pro_data_gen.robot.robot_config import R1PRO_RELEASE_LIFT_SPEED_SCALE

from ..core.base import ParamSpec, SkillResult, stabilize_base
from .gripper import GRIPPER_OPEN
from ..core.sides import resolve_side, require_side

_LIFT_M = 0.10
_LIFT_PLANNING_TIME_S = 0.4
_LIFT_SPEED_SCALE = R1PRO_RELEASE_LIFT_SPEED_SCALE


class ReleaseObject:
    """Open the gripper holding a named object, lift clear, then settle.

    Success means the open command and optional detach completed. A Cartesian
    lift is best-effort: placement predicates remain a GoalSpec/Verifier
    concern, but the hand should leave the object instead of lingering in the
    release pose.
    """

    name = "release_object"
    tier = "semantic"
    exposed = True
    description = (
        "Release a named attached object after it is at the destination. Use "
        "when GoalSpec still needs released or settled and the object is "
        "already in the target region or on the target support. Do not release "
        "to substitute for carrying the object into the region, and do not "
        "call this if the object is not attached."
    )
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Attached object to release", required=True),
        "side": ParamSpec(
            "string",
            "Gripper side; auto resolves the side holding the live attachment",
            default="auto",
            enum=("auto", "left", "right"),
        ),
        "settle_steps": ParamSpec("integer", "Physics steps to hold after opening", default=12, minimum=1, maximum=240, exposed=False),
    }

    def __init__(
        self,
        gripper_set: Any,
        arm_move_to: Any = None,
        arm_move_directional: Any = None,
    ) -> None:
        self.gripper_set = gripper_set
        self.arm_move_to = arm_move_to
        self.arm_move_directional = arm_move_directional

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        object_name: str | None = None,
        side: str = "auto",
        settle_steps: int = 12,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        if not object_name:
            raise ValueError("release_object requires object_name")
        side = resolve_side(
            require_side(side, allow_auto=True),
            adapter,
            object_name=object_name,
        )
        stabilize_base(adapter)
        opened = self.gripper_set.execute(
            adapter,
            scene=scene,
            open_value=GRIPPER_OPEN,
            side=side,
            object_name=object_name,
            step_hook=step_hook,
        )
        if not opened.success:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "gripper did not open to release the object",
                    "failure_code": opened.details.get("failure_code")
                    or opened.metrics.get("failure_code")
                    or "opening_not_reached",
                    "open": opened.details,
                },
            )
        lift_target = _lift_target_base(adapter, object_name, side)
        lift_result: SkillResult | None = None
        # A 10 cm world-+z lift after opening must not go through MPlib: the
        # fingers are still intersecting the just-released object/table, so a
        # collision-checked start is often invalid and the hand never leaves.
        # Seeded Cartesian IK from the live joints keeps the current branch.
        if self.arm_move_directional is not None:
            lift_result = self.arm_move_directional.execute(
                adapter,
                scene=scene,
                direction=[0.0, 0.0, 1.0],
                distance=_LIFT_M,
                until_contact=False,
                speed_scale=_LIFT_SPEED_SCALE,
                side=side,
                step_hook=step_hook,
            )
        if (
            (lift_result is None or not lift_result.success)
            and self.arm_move_to is not None
            and lift_target is not None
            and scene is not None
        ):
            lift_result = self.arm_move_to.execute(
                adapter,
                scene=scene,
                target_pos=lift_target,
                target_frame="grasp_center",
                side=side,
                exclude_objects=_lift_exclude_objects(adapter, scene, object_name),
                planning_time=_LIFT_PLANNING_TIME_S,
                trajectory_speed_scale=_LIFT_SPEED_SCALE,
                step_hook=step_hook,
            )
        held = max(1, int(settle_steps))
        for _ in range(held):
            adapter.step()
            if step_hook is not None:
                step_hook()
        lifted = bool(lift_result is not None and lift_result.success)
        return SkillResult(
            True,
            self.name,
            metrics={
                "settle_steps": float(held),
                "detached": float(opened.metrics.get("detached", 0.0) or 0.0),
                "lifted": float(lifted),
            },
            details={
                "object_name": object_name,
                "side": side,
                "open": opened.details,
                "lift_target": lift_target,
                "lift": None if lift_result is None else lift_result.details,
                "failure_code": None,
            },
        )


def _lift_exclude_objects(adapter: Any, scene: Any, object_name: str) -> list[str]:
    names = [object_name]
    world = None
    if hasattr(adapter, "object_position"):
        try:
            position = adapter.object_position(object_name)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            position = None
        if position is not None and len(position) >= 3:
            world = (float(position[0]), float(position[1]), float(position[2]))
    if world is None or scene is None or not hasattr(scene, "object"):
        return names
    try:
        from r1pro_data_gen.skills.manipulation.carry import _infer_source_support_surface

        support = _infer_source_support_surface(scene, scene.object(object_name), world)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        support = None
    if support:
        names.append(support)
    return names


def _lift_target_base(adapter: Any, object_name: str, side: str) -> list[float] | None:
    world = _current_grasp_center_world(adapter, object_name, side)
    if world is None:
        return None
    observation = adapter.read_observation(0.0)
    base_pose = getattr(observation, "base_pose", None) or (0.0, 0.0, 0.0)
    local = _world_xy_to_base(world, base_pose)
    return [float(local[0]), float(local[1]), float(local[2] + _LIFT_M)]


def _current_grasp_center_world(
    adapter: Any,
    object_name: str,
    side: str,
) -> tuple[float, float, float] | None:
    if hasattr(adapter, "gripper_object_alignment"):
        try:
            alignment = adapter.gripper_object_alignment(object_name, side=side)
            midpoint = alignment.get("finger_midpoint") if isinstance(alignment, dict) else None
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            midpoint = None
        if midpoint is not None and len(midpoint) >= 3:
            return (float(midpoint[0]), float(midpoint[1]), float(midpoint[2]))
    if hasattr(adapter, "end_effector_poses"):
        try:
            poses = adapter.end_effector_poses() or {}
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            poses = {}
        ee = poses.get(f"{side}_ee")
        if ee is not None and len(ee) >= 3:
            return (float(ee[0]), float(ee[1]), float(ee[2]))
    return None


def _world_xy_to_base(
    world: tuple[float, float, float],
    base_pose: Sequence[float],
) -> list[float]:
    import math

    dx = float(world[0]) - float(base_pose[0])
    dy = float(world[1]) - float(base_pose[1])
    yaw = float(base_pose[2]) if len(base_pose) > 2 else 0.0
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return [cosine * dx + sine * dy, -sine * dx + cosine * dy, float(world[2])]


__all__ = ["ReleaseObject"]
