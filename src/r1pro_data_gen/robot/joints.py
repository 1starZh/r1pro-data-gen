"""Joint mapping model: logical joint names to simulation indices.

Pure Python. The mapping is built once per simulation run from the actual
articulation joint names (never from assumed USD index order) and then used
by every read/write path.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class JointMapping:
    """Name-to-index mapping validated against the simulated articulation.

    Args:
        joint_names: names in simulation order (len N).
        group_exprs: group name -> regex matching that group's joints.
    """

    joint_names: tuple[str, ...]
    group_exprs: dict[str, str]

    def __post_init__(self) -> None:
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        if not self.group_exprs:
            raise ValueError("group_exprs must not be empty")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def group_of(self, joint: str) -> str | None:
        """Return the group name owning ``joint``, or None if ungrouped."""
        for group, expr in self.group_exprs.items():
            if re.match(expr, joint):
                return group
        return None

    def indices_of(self, group: str) -> tuple[int, ...]:
        """Indices of all joints in ``group`` (simulation order)."""
        expr = self.group_exprs.get(group)
        if expr is None:
            raise KeyError(f"unknown joint group: {group}")
        return tuple(
            i for i, name in enumerate(self.joint_names) if re.match(expr, name)
        )

    def names_of(self, group: str) -> tuple[str, ...]:
        """Joint names in ``group`` (simulation order)."""
        return tuple(self.joint_names[i] for i in self.indices_of(group))

    def validate(self) -> None:
        """Raise if joints are ungrouped, groups overlap, or a group is empty."""
        for name in self.joint_names:
            groups = [g for g in self.group_exprs if re.match(self.group_exprs[g], name)]
            if not groups:
                raise ValueError(f"joint is not assigned to any group: {name}")
            if len(groups) > 1:
                raise ValueError(f"joint matches multiple groups {groups}: {name}")
        for group, expr in self.group_exprs.items():
            if not any(re.match(expr, n) for n in self.joint_names):
                raise ValueError(f"group {group!r} matches no joints: {expr}")
