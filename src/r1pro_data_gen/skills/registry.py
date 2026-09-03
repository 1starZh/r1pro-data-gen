"""Skill registry: the capability catalogue the planner selects from.

A registry holds named skill instances, validates their declarations once at
construction, and offers the description set (name + description + parameter
schema) that the planner uses to choose skills and to validate a plan before
execution. This is also the exact "tools" surface a future LLM planner will
see.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, Iterable, Mapping

from .core.base import ParamSpec, Skill, SkillResult


class SkillRegistry:
    """Named skill instances with schema validation and lookup."""

    def __init__(self, skills: Iterable[Skill]) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if not skill.name.strip():
                raise ValueError("skill name must not be empty")
            if skill.name in self._skills:
                raise ValueError(f"duplicate skill name: {skill.name!r}")
            _validate_declaration(skill)
            self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def __getitem__(self, name: str) -> Skill:
        return self._skills[name]

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def descriptions(
        self,
        *,
        public_only: bool = False,
        for_llm: bool = False,
    ) -> list[dict[str, object]]:
        """Catalogue for the planner: name, description, parameter schema.

        Low-level actuator skills remain registered for replay and diagnostics,
        but a future LLM should normally receive ``llm_descriptions()`` so it
        chooses semantic actions such as ``arm_move_to`` instead of assembling
        an unsafe raw position command by itself. ``for_llm=True`` omits
        parameters marked ``exposed=False`` so tuning knobs stay off the model.
        """
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "tier": getattr(skill, "tier", "public"),
                "parameters": {
                    pname: _llm_parameter_description(spec)
                    for pname, spec in skill.parameters.items()
                    if not for_llm or getattr(spec, "exposed", True)
                },
            }
            for skill in self._skills.values()
            if not public_only or getattr(skill, "exposed", True)
        ]

    def llm_descriptions(self) -> list[dict[str, object]]:
        """Return only high-level semantic actions intended for an LLM."""
        from r1pro_data_gen.planning.llm.contracts import LLM_PUBLIC_SKILLS

        return [
            item
            for item in self.descriptions(public_only=False, for_llm=True)
            if item["name"] in LLM_PUBLIC_SKILLS
        ]

    def agent_descriptions(self) -> list[dict[str, object]]:
        """Return the closed-loop agent catalogue (high-level skills only)."""
        from r1pro_data_gen.agent.contracts import AGENT_PUBLIC_SKILLS

        return [
            item
            for item in self.descriptions(public_only=False, for_llm=True)
            if item["name"] in AGENT_PUBLIC_SKILLS
        ]

    def validate_plan_params(self, name: str, params: Mapping[str, Any]) -> None:
        """Raise if a plan stage calls ``name`` with missing required params."""
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"unknown skill: {name}")
        # Plan JSON stores the selected skill beside its parameters. The
        # orchestrator strips it before execution, and validation accepts the
        # same representation for pre-flight checks.
        params = {key: value for key, value in params.items() if key != "skill"}
        missing = [
            pname for pname, spec in skill.parameters.items()
            if spec.required and pname not in params
        ]
        if missing:
            raise ValueError(f"skill {name!r} missing required parameters: {sorted(missing)}")
        unknown = sorted(set(params) - set(skill.parameters))
        if unknown:
            raise ValueError(f"skill {name!r} received unknown parameters: {unknown}")
        for pname, value in params.items():
            _validate_value(name, pname, skill.parameters[pname], value)

    def execute(
        self,
        name: str,
        adapter: Any,
        scene: Any = None,
        step_hook: Any = None,
        **params: Any,
    ) -> SkillResult:
        """Validate and execute a skill call (used by the orchestrator)."""
        self.validate_plan_params(name, params)
        skill = self._skills[name]
        return skill.execute(adapter, scene=scene, step_hook=step_hook, **params)


def _llm_parameter_description(spec: ParamSpec) -> dict[str, object]:
    """Expose the literal schema and the declarative-reference form together."""
    description: dict[str, object] = {
        "type": spec.type,
        "description": spec.description,
        "required": spec.required,
        "default": spec.default,
        "enum": list(spec.enum),
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "min_items": spec.min_items,
        "max_items": spec.max_items,
        "shape": list(spec.shape) if spec.shape is not None else None,
    }
    if spec.type == "array":
        description["accepts_typed_reference"] = True
        description["typed_reference_form"] = (
            "Use one JSON object {ref,value_type,shape,frame,offset} at this parameter; "
            "do not wrap the reference object in an array."
        )
    return description


def _validate_declaration(skill: Skill) -> None:
    allowed_types = {"number", "integer", "string", "boolean", "array", "object"}
    for pname, spec in skill.parameters.items():
        if not isinstance(spec, ParamSpec):
            raise ValueError(f"skill {skill.name!r} parameter {pname!r} is not a ParamSpec")
        if not spec.type or not spec.description:
            raise ValueError(f"skill {skill.name!r} parameter {pname!r} needs type and description")
        if spec.type not in allowed_types:
            raise ValueError(f"skill {skill.name!r} parameter {pname!r} has unsupported type {spec.type!r}")
        if spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum:
            raise ValueError(f"skill {skill.name!r} parameter {pname!r} has invalid numeric bounds")
        if spec.min_items is not None and spec.min_items < 0:
            raise ValueError(f"skill {skill.name!r} parameter {pname!r} has invalid min_items")


def _validate_value(skill_name: str, pname: str, spec: ParamSpec, value: Any) -> None:
    """Validate the subset of JSON schema needed by skill calls.

    This deliberately runs before simulation starts. A malformed target should
    be a planner error, never a half-executed robot motion.
    """
    if value is None and not spec.required:
        return
    if spec.type == "number" and (not isinstance(value, Real) or isinstance(value, bool)):
        raise ValueError(f"skill {skill_name!r} parameter {pname!r} must be a number")
    if spec.type == "integer" and (not isinstance(value, Integral) or isinstance(value, bool)):
        raise ValueError(f"skill {skill_name!r} parameter {pname!r} must be an integer")
    if spec.type == "string" and not isinstance(value, str):
        raise ValueError(f"skill {skill_name!r} parameter {pname!r} must be a string")
    if spec.type == "boolean" and not isinstance(value, bool):
        raise ValueError(f"skill {skill_name!r} parameter {pname!r} must be a boolean")
    if spec.type == "array":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"skill {skill_name!r} parameter {pname!r} must be an array")
        if spec.min_items is not None and len(value) < spec.min_items:
            raise ValueError(f"skill {skill_name!r} parameter {pname!r} has too few items")
        if spec.max_items is not None and len(value) > spec.max_items:
            raise ValueError(f"skill {skill_name!r} parameter {pname!r} has too many items")
        if spec.shape is not None:
            shape = _shape_of(value)
            if shape != spec.shape:
                raise ValueError(
                    f"skill {skill_name!r} parameter {pname!r} must have shape {spec.shape}, got {shape}"
                )
    if spec.type == "object" and not isinstance(value, Mapping):
        raise ValueError(f"skill {skill_name!r} parameter {pname!r} must be an object")
    if spec.type in {"number", "integer"}:
        numeric = float(value)
        if spec.minimum is not None and numeric < spec.minimum:
            raise ValueError(f"skill {skill_name!r} parameter {pname!r} is below minimum")
        if spec.maximum is not None and numeric > spec.maximum:
            raise ValueError(f"skill {skill_name!r} parameter {pname!r} is above maximum")
    if spec.enum and value not in spec.enum:
        raise ValueError(f"skill {skill_name!r} parameter {pname!r} must be one of {spec.enum}")


def _shape_of(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return (len(value),) + (_shape_of(value[0]) if value else ())


def build_default_registry(kin: Any, vel_limits: Any) -> SkillRegistry:
    """The full R1Pro skill library (v2: solve/plan/execute pipeline).

    ``kin`` may be one legacy kinematics object or a ``{"left": ..., "right":
    ...}`` mapping. A real left-arm :class:`R1ProKinematics` is automatically
    paired with a right-arm model from the same URDF. ``vel_limits`` accepts
    the same singleton/mapping forms. Every skill is registered here; tasks
    pick the subset they need from the registry.
    """
    from .manipulation.arm import ArmJointTo
    from .manipulation.arm_motion import (
        ArmAlignGripper,
        ArmMoveDirectional,
        ArmMoveTo,
        ArmMoveThrough,
        ArmRotateEE,
        ArmTrajectoryFollow,
    )
    from .manipulation.carry import ArmCarryObjectTo
    from .manipulation.grasp import GraspObject
    from .manipulation.release import ReleaseObject
    from .manipulation.push import PushObjectTo
    from .manipulation.support_aware_grasp import SupportAwareGraspObject
    from .mobility.base_motion import (
        BaseFollowPath,
        BaseLockWheels,
        BaseMoveTo,
        BaseNavigateTo,
        BaseRotateTo,
        BaseUnlockWheels,
        BaseVelocitySet,
    )
    from .manipulation.gripper import GripperGrasp, GripperSet
    from .posture.joint_mask import JointMaskLock, JointMaskUnlock
    from .planning.queries import QueryArmPath, QueryBasePath, QueryIKSolution
    from .observation.queries import QueryContacts, QueryEEPose, QueryJointPos, QueryObjectPose
    from .manipulation.transfer import TransferObjectBetweenSupports
    from .posture.torso import TorsoMoveTo
    from .posture.workspace import PrepareWorkspace
    from .manipulation.whole_body import (
        WholeBodyHoldTransition,
        WholeBodyPregraspTransition,
        WholeBodyTransferObjectBetweenSupports,
    )

    def _make_planner(side: str):
        from r1pro_data_gen.methods.manipulation.mplib_path import build_planner

        return build_planner(side=side)

    if isinstance(kin, Mapping):
        kinematics = dict(kin)
    elif kin is not None and hasattr(kin, "urdf_path"):
        from r1pro_data_gen.robot.kinematics import R1ProKinematics

        kinematics = {
            side: kin if getattr(kin, "side", "left") == side else R1ProKinematics(kin.urdf_path, side=side)
            for side in ("left", "right")
        }
    else:
        # Pure tests and third-party backends may deliberately provide one
        # side-neutral implementation. Runtime entrypoints use the mapping.
        kinematics = {"left": kin, "right": kin}
    limits = dict(vel_limits) if isinstance(vel_limits, Mapping) else {"left": vel_limits, "right": vel_limits}
    planners = {side: _make_planner(side) for side in ("left", "right")}
    arm_move_to = ArmMoveTo(kinematics, limits, planners)
    arm_move_through = ArmMoveThrough(kinematics, limits, planners)
    arm_align = ArmAlignGripper(kinematics, limits, planners)
    arm_move_directional = ArmMoveDirectional(kinematics, limits, planners)
    gripper_set = GripperSet()
    gripper_grasp = GripperGrasp()
    arm_joint_to = ArmJointTo(kinematics, limits)
    torso_move_to = TorsoMoveTo()
    base_navigate_to = BaseNavigateTo(kinematics)
    whole_body_pregrasp = WholeBodyPregraspTransition(
        kinematics,
        base_staging=base_navigate_to,
    )
    carry_object = ArmCarryObjectTo(
        kinematics, limits, planners, arm_move_through, arm_move_to
    )
    grasp_object = GraspObject(
        gripper_set,
        arm_move_to,
        arm_align,
        gripper_grasp,
        arm_joint_to,
        torso_move_to=torso_move_to,
        whole_body_pregrasp=whole_body_pregrasp,
    )
    support_aware_grasp_object = SupportAwareGraspObject(
        gripper_set,
        arm_move_to,
        arm_align,
        gripper_grasp,
        arm_joint_to,
        torso_move_to=torso_move_to,
        whole_body_pregrasp=whole_body_pregrasp,
    )
    release_object = ReleaseObject(
        gripper_set, arm_move_to, arm_move_directional
    )
    whole_body_hold = WholeBodyHoldTransition(
        kinematics,
        arm_move_directional=arm_move_directional,
    )
    prepare_workspace = PrepareWorkspace(
        torso_move_to,
        whole_body_pregrasp=whole_body_pregrasp,
    )
    registry = SkillRegistry(
        [
            BaseMoveTo(),
            base_navigate_to,
            BaseRotateTo(),
            BaseFollowPath(),
            BaseVelocitySet(),
            BaseLockWheels(),
            BaseUnlockWheels(),
            JointMaskLock(),
            JointMaskUnlock(),
            torso_move_to,
            prepare_workspace,
            arm_joint_to,
            ArmTrajectoryFollow(kinematics, limits),
            arm_move_to,
            arm_align,
            arm_move_through,
            carry_object,
            grasp_object,
            release_object,
            support_aware_grasp_object,
            TransferObjectBetweenSupports(support_aware_grasp_object, carry_object, release_object),
            WholeBodyTransferObjectBetweenSupports(
                support_aware_grasp_object,
                carry_object,
                release_object,
                whole_body_hold,
                base_reposition=base_navigate_to,
            ),
            whole_body_pregrasp,
            whole_body_hold,
            PushObjectTo(base_navigate_to),
            arm_move_directional,
            ArmRotateEE(kinematics, limits),
            gripper_set,
            gripper_grasp,
            QueryObjectPose(),
            QueryContacts(),
            QueryEEPose(kinematics),
            QueryJointPos(),
            QueryIKSolution(kinematics),
            QueryArmPath(planners, kinematics),
            QueryBasePath(),
        ]
    )
    return registry


__all__ = ["SkillRegistry", "build_default_registry"]
