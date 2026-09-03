"""Skill contracts: what a capability advertises and how it is invoked.

A skill is a task-agnostic, reusable robot capability (e.g. "move the base to
a pose", "move the arm to a target pose avoiding obstacles", "set the gripper
opening"). Beyond execution, a skill *declares* its capability: a name, a
human-readable description, and a parameter schema. That declaration is the
"tool catalogue" the planner (Claude today, an LLM later) uses to choose and
validate skill calls.

Skills take semantic goals (positions, poses, joint configurations, opening
values) as parameters -- never task constants. Obstacle/geometry knowledge
comes from the runtime :class:`SceneModel` passed at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One skill parameter: JSON-schema-like declaration for the planner."""

    type: str  # "number" | "string" | "boolean" | "array" | "object"
    description: str
    required: bool = False
    default: object = None
    enum: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    min_items: int | None = None
    max_items: int | None = None
    shape: tuple[int, ...] | None = None
    # False hides this parameter from LLM/agent catalogues. Execution still
    # accepts it from trusted replay; the model must not see tuning knobs.
    exposed: bool = True


@dataclass(frozen=True, slots=True)
class SkillResult:
    """Outcome of one skill execution."""

    success: bool
    skill: str
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class Skill(Protocol):
    """A robot capability executable against a simulation adapter."""

    name: str
    description: str
    parameters: dict[str, ParamSpec]

    def execute(self, adapter: Any, scene: Any = None, **params: Any) -> SkillResult:
        """Run the skill with the given semantic parameters.

        ``adapter`` is the simulation adapter (``simulation.isaac_sim``);
        ``scene`` is the runtime :class:`SceneModel` (may be None for skills
        that do not need environment geometry). Skills must not depend on
        task-specific logic or constants.
        """
        ...


def stabilize_base(
    adapter: Any,
    *,
    lock_torso: bool = True,
    replace_wheel_only: bool = False,
) -> None:
    """Hold the mobile base joints and articulated torso during manipulation.

    Steering and wheel position targets keep the chassis parked through real
    joint drives. The floating base remains fully physical; no root parking
    brake, external wrench, or pose write is used. The
    torso joints must also retain immutable measured targets: otherwise sparse
    arm commands copy their load-induced deflection back as the next target and
    the upper body progressively folds during a long reach. An explicit
    task-level joint mask remains authoritative and is never replaced.
    ``replace_wheel_only`` is an explicit phase transition for a
    manipulation skill that temporarily held only the wheels while moving
    the torso; it upgrades exactly that internally-created mask after the
    torso motion has finished.  Physical adapters may extend that mask
    atomically so the old wheel targets and actuator buffers are not cleared
    for one uncontrolled physics frame.
    """
    if getattr(adapter, "joint_mask_locked", False):
        groups = tuple(
            getattr(
                adapter,
                "joint_lock_groups",
                getattr(adapter, "_joint_lock_groups", ()),
            )
        )
        if not (
            replace_wheel_only
            and lock_torso
            and {"steer", "wheel"}.issubset(set(groups))
            and "torso" not in set(groups)
            and hasattr(adapter, "unlock_joint_mask")
        ):
            return
        if hasattr(adapter, "extend_joint_mask"):
            adapter.extend_joint_mask(
                joint_groups=("torso",),
                gain_overrides={"wheel": (500.0, 100.0)},
            )
            return
        adapter.unlock_joint_mask()
    if hasattr(adapter, "lock_joint_mask"):
        adapter.lock_joint_mask(
            mask_mode="lock",
            joint_groups=("steer", "wheel", "torso") if lock_torso else ("steer", "wheel"),
            lock_root=False,
            # A torso pitch transfers a large reaction through the three
            # continuous wheel joints.  The velocity-mode actuator is
            # upgraded to this position hold only for the manipulation phase;
            # navigation keeps its own drive gains after the mask is released.
            gain_overrides={"wheel": (500.0, 100.0)},
        )
    elif hasattr(adapter, "lock_wheels"):
        adapter.lock_wheels()


def release_skill_wheel_lock(adapter: Any) -> None:
    """Release a wheel lock left behind by a prior manipulation skill.

    Navigation skills drive the wheels, so they must clear any residual skill
    lock first; a task-level joint mask is authoritative and stays untouched.
    """
    if (
        hasattr(adapter, "unlock_wheels")
        and getattr(adapter, "_wheels_locked", False)
        and not getattr(adapter, "joint_mask_locked", False)
    ):
        adapter.unlock_wheels()


__all__ = ["ParamSpec", "Skill", "SkillResult", "stabilize_base", "release_skill_wheel_lock"]
