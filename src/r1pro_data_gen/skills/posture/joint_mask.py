"""Reusable phase-level joint mask skills."""

from __future__ import annotations

from typing import Any, Callable

from ..core.base import ParamSpec, SkillResult


class JointMaskLock:
    """Freeze selected joints, or freeze everything except selected joints."""

    name = "joint_mask_lock"
    tier = "semantic"
    exposed = True
    description = (
        "Capture measured joint positions and continuously lock a joint mask. "
        "Use mask_mode='allow' to leave only specified groups/joints movable "
        "during a manipulation phase. The floating base remains physical."
    )
    parameters: dict[str, ParamSpec] = {
        "mask_mode": ParamSpec(
            "string", "Whether selected entries are locked or are the only entries allowed to move",
            default="lock", enum=("lock", "allow"),
        ),
        "joint_groups": ParamSpec(
            "array", "Joint groups selected by the mask (steer, wheel, torso, left_arm, right_arm, left_gripper, right_gripper)",
            default=[],
        ),
        "joint_names": ParamSpec("array", "Additional explicit joint names selected by the mask", default=[]),
        "lock_root": ParamSpec("boolean", "Deprecated compatibility flag; root pose is never externally locked", default=False),
        "stiffness_scale": ParamSpec("number", "Multiplier on validated minimum hold gains", default=1.0, minimum=0.1, maximum=3.0),
        "settle_steps": ParamSpec("integer", "Physics steps after applying the mask", default=30, minimum=0),
    }

    def execute(
        self,
        adapter: Any,
        scene: Any = None,
        mask_mode: str = "lock",
        joint_groups: list[str] | None = None,
        joint_names: list[str] | None = None,
        lock_root: bool = False,
        stiffness_scale: float = 1.0,
        settle_steps: int = 30,
        step_hook: Callable[[], None] | None = None,
        **_: Any,
    ) -> SkillResult:
        del scene
        try:
            details = adapter.lock_joint_mask(
                mask_mode=mask_mode,
                joint_groups=tuple(joint_groups or ()),
                joint_names=tuple(joint_names or ()),
                lock_root=bool(lock_root),
                stiffness_scale=float(stiffness_scale),
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            return SkillResult(False, self.name, details={"reason": str(exc)})
        steps = max(0, int(settle_steps))
        for _ in range(steps):
            adapter.step()
            if step_hook is not None:
                step_hook()
        metrics = {"settle_steps": float(steps)}
        if hasattr(adapter, "joint_lock_metrics"):
            metrics.update(adapter.joint_lock_metrics())
        if hasattr(adapter, "joint_lock_diagnostics"):
            details.update(adapter.joint_lock_diagnostics())
        return SkillResult(True, self.name, metrics=metrics, details=details)


class JointMaskUnlock:
    """Release a phase-level joint mask and restore previous actuator gains."""

    name = "joint_mask_unlock"
    tier = "semantic"
    exposed = True
    description = "Release the active joint mask and restore the actuator gains used before it was applied."
    parameters: dict[str, ParamSpec] = {}

    def execute(self, adapter: Any, scene: Any = None, **_: Any) -> SkillResult:
        del scene
        adapter.unlock_joint_mask()
        return SkillResult(True, self.name)


__all__ = ["JointMaskLock", "JointMaskUnlock"]
