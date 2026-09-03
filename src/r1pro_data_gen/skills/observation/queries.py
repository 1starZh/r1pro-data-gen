"""Query skills: read robot/scene state as structured results.

These make observation a first-class capability of the skill library -- the
planner (Claude today, an LLM later) can call them to understand the current
state mid-task, exactly like action skills. All read-only; no simulation state
is modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.base import ParamSpec, SkillResult
from ..core.sides import for_side, require_side

ARM_JOINTS_BY_SIDE = {
    "left": tuple(f"left_arm_joint{i}" for i in range(1, 8)),
    "right": tuple(f"right_arm_joint{i}" for i in range(1, 8)),
}


class QueryObjectPose:
    """World position of a dynamic scene object."""

    name = "query_object_pose"
    description = "Query the world position (x, y, z) of a dynamic scene object by name."
    parameters: dict[str, ParamSpec] = {
        "object_name": ParamSpec("string", "Object name in the scene", required=True),
    }

    def execute(self, adapter: Any, scene: Any = None, object_name: str = None, **_: Any) -> SkillResult:
        del scene
        if object_name is None:
            raise ValueError("query_object_pose requires object_name")
        try:
            pos = adapter.object_position(object_name)
        except RuntimeError as exc:
            return SkillResult(success=False, skill=self.name, details={"reason": str(exc)})
        return SkillResult(success=True, skill=self.name, details={"position": list(pos)})


class QueryContacts:
    """Net contact forces on the scene contact sensors."""

    name = "query_contacts"
    description = "Query the net contact forces (N) on the scene's contact sensors."
    parameters: dict[str, ParamSpec] = {
        "side": ParamSpec("string", "Which gripper's finger sensors to query", default="left", enum=("left", "right")),
    }

    def execute(self, adapter: Any, scene: Any = None, side: str = "left", **_: Any) -> SkillResult:
        del scene
        side = require_side(side)
        try:
            forces = adapter.finger_contact_forces(side=side)
        except TypeError:  # legacy/test adapters
            forces = adapter.finger_contact_forces()
        return SkillResult(success=True, skill=self.name, details={"contact_forces": list(forces)})


class QueryEEPose:
    """End-effector pose (base frame) via FK of the current arm joints."""

    name = "query_ee_pose"
    description = "Query the arm end-effector pose (position + quaternion, base frame) from the current joints."
    parameters: dict[str, ParamSpec] = {
        "side": ParamSpec("string", "Which arm ('left' or 'right')", default="left", enum=("left", "right")),
    }

    def __init__(self, kin: Any):
        self.kin = kin

    def execute(self, adapter: Any, scene: Any = None, side: str = "left", **_: Any) -> SkillResult:
        del scene
        side = require_side(side)
        kin = for_side(self.kin, side)
        obs = adapter.read_observation(0.0)
        q = np.array([obs.joint_positions[j] for j in ARM_JOINTS_BY_SIDE[side]])
        pos, quat = kin.fk(q)
        return SkillResult(
            success=True,
            skill=self.name,
            details={"position": [float(v) for v in pos], "quaternion": [float(v) for v in quat]},
        )


class QueryJointPos:
    """Current positions of selected joints."""

    name = "query_joint_pos"
    description = "Query the current positions of specified joints (empty list = all joints)."
    parameters: dict[str, ParamSpec] = {
        "joints": ParamSpec("array", "Joint names to query", default=[]),
    }

    def execute(self, adapter: Any, scene: Any = None, joints: list[str] = None, **_: Any) -> SkillResult:
        del scene
        obs = adapter.read_observation(0.0)
        names = joints or list(obs.joint_positions)
        return SkillResult(
            success=True,
            skill=self.name,
            details={"joint_positions": {n: obs.joint_positions[n] for n in names}},
        )


__all__ = [
    "ARM_JOINTS_BY_SIDE",
    "QueryContacts",
    "QueryEEPose",
    "QueryJointPos",
    "QueryObjectPose",
]
