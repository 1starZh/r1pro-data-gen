"""Task-agnostic live grasp state shared by skills and execution evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class GraspContext:
    """Measured relationship between a held object and a gripper anchor.

    The context deliberately stores the object-to-anchor translation in world
    coordinates rather than a reset-pose target.  A carry skill can therefore
    regenerate end-effector goals from the current physical grasp after every
    preceding motion.
    """

    object_name: str
    side: str
    attached: bool
    object_position_world: tuple[float, float, float]
    grasp_center_world: tuple[float, float, float]
    object_to_grasp_center_world: tuple[float, float, float]
    attachment_error_m: float | None = None

    def __post_init__(self) -> None:
        if not self.object_name.strip():
            raise ValueError("grasp context object_name must not be empty")
        if self.side not in {"left", "right"}:
            raise ValueError("grasp context side must be 'left' or 'right'")
        for name, value in (
            ("object_position_world", self.object_position_world),
            ("grasp_center_world", self.grasp_center_world),
            ("object_to_grasp_center_world", self.object_to_grasp_center_world),
        ):
            if len(value) != 3 or not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must be a finite 3-vector")
        if self.attachment_error_m is not None and (
            not math.isfinite(float(self.attachment_error_m))
            or self.attachment_error_m < 0.0
        ):
            raise ValueError("attachment_error_m must be a finite non-negative number")

    def to_dict(self) -> dict[str, object]:
        """Return bounded JSON-compatible evidence for a skill result."""
        return {
            "object_name": self.object_name,
            "side": self.side,
            "attached": self.attached,
            "object_position_world": list(self.object_position_world),
            "grasp_center_world": list(self.grasp_center_world),
            "object_to_grasp_center_world": list(self.object_to_grasp_center_world),
            "attachment_error_m": self.attachment_error_m,
        }


__all__ = ["GraspContext"]
