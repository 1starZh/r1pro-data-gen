"""Task-agnostic capability library.

The subpackages mirror the capability surface used by the agent:

``core``
    Skill protocol, result type, and common side helpers.
``mobility``
    Mobile-base motion.
``manipulation``
    Arm, gripper, transfer, push, and whole-body skills.
``planning``
    Read-only IK/path query skills.
``observation``
    Read-only state query skills.
``posture``
    Torso and joint-mask skills.

Importing from this package remains the concise public form.  Deep imports
should follow the capability subpackage rather than placing new modules at
this root.
"""

from .core import ParamSpec, Skill, SkillResult
from .manipulation import (
    ArmAlignGripper,
    ArmCarryObjectTo,
    ArmJointTo,
    ArmMoveDirectional,
    ArmMoveThrough,
    ArmMoveTo,
    ArmRotateEE,
    ArmTrajectoryFollow,
    GraspObject,
    GripperGrasp,
    GripperSet,
    PushObjectTo,
    ReleaseObject,
    SupportAwareGraspObject,
    TransferObjectBetweenSupports,
    WholeBodyHoldTransition,
    WholeBodyPregraspTransition,
    WholeBodyTransferObjectBetweenSupports,
    derive_support_aware_pregrasp,
    derive_support_aware_pregrasp_candidates,
    pregrasp_motion_tolerance,
    support_aware_orientation_candidates,
    world_point_to_base,
)
from .mobility import (
    BaseFollowPath,
    BaseLockWheels,
    BaseMoveTo,
    BaseNavigateTo,
    BaseRotateTo,
    BaseUnlockWheels,
    BaseVelocitySet,
)
from .observation import QueryContacts, QueryEEPose, QueryJointPos, QueryObjectPose
from .planning import QueryArmPath, QueryBasePath, QueryIKSolution
from .posture import JointMaskLock, JointMaskUnlock, PrepareWorkspace, TorsoMoveTo
from .registry import SkillRegistry, build_default_registry

__all__ = [
    "ArmAlignGripper",
    "ArmCarryObjectTo",
    "ArmJointTo",
    "ArmMoveDirectional",
    "ArmMoveThrough",
    "ArmMoveTo",
    "ArmRotateEE",
    "ArmTrajectoryFollow",
    "BaseFollowPath",
    "BaseLockWheels",
    "BaseMoveTo",
    "BaseNavigateTo",
    "BaseRotateTo",
    "BaseUnlockWheels",
    "BaseVelocitySet",
    "GraspObject",
    "GripperGrasp",
    "GripperSet",
    "JointMaskLock",
    "JointMaskUnlock",
    "ParamSpec",
    "PrepareWorkspace",
    "PushObjectTo",
    "QueryArmPath",
    "QueryBasePath",
    "QueryContacts",
    "QueryEEPose",
    "QueryIKSolution",
    "QueryJointPos",
    "QueryObjectPose",
    "ReleaseObject",
    "Skill",
    "SkillRegistry",
    "SkillResult",
    "SupportAwareGraspObject",
    "TorsoMoveTo",
    "TransferObjectBetweenSupports",
    "WholeBodyHoldTransition",
    "WholeBodyPregraspTransition",
    "WholeBodyTransferObjectBetweenSupports",
    "build_default_registry",
    "derive_support_aware_pregrasp",
    "derive_support_aware_pregrasp_candidates",
    "pregrasp_motion_tolerance",
    "support_aware_orientation_candidates",
    "world_point_to_base",
]
