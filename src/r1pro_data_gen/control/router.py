"""Simple group-aware command router."""

from __future__ import annotations

from r1pro_data_gen.domain import ControlCommand, Observation, TrajectoryPoint
from .interfaces import ControllerConfig


class CommandRouter:
    """Route trajectory references to position or velocity command fields."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._joint_modes = {
            joint: group.mode for group in config.groups for joint in group.joints
        }
        if len(self._joint_modes) != sum(len(group.joints) for group in config.groups):
            raise ValueError("a joint cannot belong to multiple groups")

    def command(
        self,
        point: TrajectoryPoint,
        observation: Observation,
        timestamp: float,
    ) -> ControlCommand:
        """Create a command; ``observation`` is accepted for feedback-aware APIs."""
        del observation
        positions: dict[str, float] = {}
        velocities: dict[str, float] = {}
        for joint, value in point.joint_positions.items():
            mode = self._joint_modes.get(joint)
            if mode is None:
                raise ValueError(f"joint is not assigned to a control group: {joint}")
            if mode.value == "position":
                positions[joint] = value
            else:
                raise ValueError(f"position reference for velocity-controlled joint: {joint}")
        for joint, value in point.joint_velocities.items():
            mode = self._joint_modes.get(joint)
            if mode is None:
                raise ValueError(f"joint is not assigned to a control group: {joint}")
            if mode.value == "velocity":
                velocities[joint] = value
            else:
                raise ValueError(f"velocity reference for position-controlled joint: {joint}")
        return ControlCommand(
            timestamp=timestamp,
            mode_by_group={group.name: group.mode for group in self._config.groups},
            position_targets=positions,
            velocity_targets=velocities,
            gripper_target=point.gripper,
        )
