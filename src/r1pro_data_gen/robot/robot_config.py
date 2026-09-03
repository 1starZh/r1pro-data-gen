"""R1Pro robot asset facts: joint groups, physical limits, actuator gains.

Pure data module (no isaaclab imports). Limits were read from
``asset/r1pro/r1pro.usda`` on 2026-08-08 (revolute values converted from
degrees to radians, None = unbounded). Actuator gains are the GPU-verified
values from the reference r1pro_datagen project (same USDA asset).
"""

from __future__ import annotations

import math
from pathlib import Path

# Default robot asset relative to the repository root.
R1PRO_USDA_RELPATH = Path("asset") / "r1pro" / "r1pro.usda"

# Joint group regex expressions, compatible with Articulation.find_joints().
R1PRO_JOINT_GROUP_EXPR = {
    "steer": "steer_motor_joint.*",
    "wheel": "wheel_motor_joint.*",
    "torso": "torso_joint.*",
    "left_arm": "left_arm_joint.*",
    "right_arm": "right_arm_joint.*",
    "left_gripper": "left_gripper_finger_joint.*",
    "right_gripper": "right_gripper_finger_joint.*",
}

# Joint position limits in radians (prismatic in meters), authored in USDA.
# Mirrored groups (left/right) share the same joint names with swapped bounds.
R1PRO_JOINT_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "steer_motor_joint1": (-3.1416, 3.1416),
    "steer_motor_joint2": (-3.1416, 3.1416),
    "steer_motor_joint3": (-3.1416, 3.1416),
    "wheel_motor_joint1": (None, None),  # continuous rotation
    "wheel_motor_joint2": (None, None),
    "wheel_motor_joint3": (None, None),
    "torso_joint1": (-1.1345, 1.8326),
    "torso_joint2": (-2.7925, 2.5307),
    "torso_joint3": (-1.8326, 1.5708),
    "torso_joint4": (-3.0543, 3.0543),
    "left_arm_joint1": (-4.4506, 1.309),
    "left_arm_joint2": (-0.1745, 3.1416),
    "left_arm_joint3": (-2.356196, 2.356196),
    "left_arm_joint4": (-1.7453, 0.3491),
    "left_arm_joint5": (-2.356196, 2.356196),
    "left_arm_joint6": (-1.047198, 1.047198),
    "left_arm_joint7": (-1.5708, 1.5708),
    "right_arm_joint1": (-4.4506, 1.309),
    "right_arm_joint2": (-3.1416, 0.1745),
    "right_arm_joint3": (-2.356196, 2.356196),
    "right_arm_joint4": (-1.7453, 0.3491),
    "right_arm_joint5": (-2.356196, 2.356196),
    "right_arm_joint6": (-1.047198, 1.047198),
    "right_arm_joint7": (-1.5708, 1.5708),
    "left_gripper_finger_joint1": (0.0, 0.05),  # meters (prismatic)
    "left_gripper_finger_joint2": (-0.05, 0.0),
    "right_gripper_finger_joint1": (0.0, 0.05),
    "right_gripper_finger_joint2": (-0.05, 0.0),
}

# Per-joint torque/velocity limits from the reference project (same asset).
R1PRO_ARM_TORQUE_LIMITS = [55.0, 55.0, 25.0, 25.0, 18.0, 18.0, 18.0]  # N·m, joint1-7
R1PRO_ARM_VELOCITY_LIMITS = [7.12, 7.12, 8.3776, 8.3776, 10.472, 10.472, 10.472]  # rad/s
R1PRO_TORSO_EFFORT_LIMIT = 100.0  # N·m, authored USD/URDF limit
R1PRO_TORSO_VELOCITY_LIMITS = (0.5, 0.5, 0.5, 0.5)  # rad/s, conservative runtime profile
R1PRO_EFFORT_PLANNING_UTILIZATION = 0.80
R1PRO_RUNTIME_EFFORT_ABORT_UTILIZATION = 0.95
# A single saturated PhysX drive sample is a normal transient when the
# floating base accelerates or an elbow lifts against gravity. Keep recording
# that peak, but require the *current* reserve crossing to persist before
# aborting. A recovered spike must not poison later skills.
R1PRO_RUNTIME_EFFORT_ABORT_PERSISTENCE_S = 0.40
R1PRO_SUPPORT_POLYGON_MARGIN_M = 0.03
R1PRO_ROOT_TILT_ABORT_RAD = 0.05235987755982989  # 3 degrees
R1PRO_ROOT_HEIGHT_RISE_ABORT_M = 0.015
# Three steered wheels can lose one contact for ~0.27 s while the steer
# modules reconfigure, with tilt still well under 1 deg. 0.40 s still fails
# a real lift-off; tilt/height gates remain the tip-over authority.
R1PRO_WHEEL_CONTACT_LOSS_TIMEOUT_S = 0.40
R1PRO_GRIPPER_EFFORT_LIMIT = 100.0  # N
R1PRO_GRIPPER_VELOCITY_LIMIT = 0.25  # m/s
# Indoor human-like motion for WBC training data. Contact phases stay slower
# than free transport, but no longer pad every short segment to a crawl.
R1PRO_BASE_V_MAX = 0.32  # m/s; ~3x the old crawl, below the wheel-hop cliff
R1PRO_BASE_OMEGA_MAX = 0.65  # rad/s
# Limit chassis jerk so cruise speed does not lift a wheel on the first step.
R1PRO_BASE_LIN_ACCEL_MAX = 0.50  # m/s^2
R1PRO_BASE_ANG_ACCEL_MAX = 1.20  # rad/s^2
R1PRO_FREE_ARM_SPEED_SCALE = 0.42
R1PRO_LOADED_ARM_SPEED_SCALE = 0.36
R1PRO_CONTACT_ARM_SPEED_SCALE = 0.18
R1PRO_ALIGNMENT_SPEED_SCALE = 0.12
R1PRO_ALIGNMENT_MIN_PHASE_S = 0.28
R1PRO_ARM_MIN_TRAJECTORY_S = 0.18
R1PRO_LOCAL_ARM_MIN_TRAJECTORY_S = 0.22
R1PRO_GRASP_APPROACH_SPEED_SCALE = 0.22
R1PRO_RELEASE_LIFT_SPEED_SCALE = 0.35
# A measured grasp-window correction is a local manipulation primitive.  The
# candidate IK branch may not change any arm joint by more than this amount in
# one correction, even when another redundant solution reaches the same
# Cartesian target.  This is a robot/controller continuity capability, not a
# task waypoint; callers should split a larger correction into more measured
# chunks instead of allowing a branch jump.
R1PRO_ALIGNMENT_MAX_LOCAL_JOINT_STEP_RAD = 0.45

# Robot-level grasp-frame calibration.  This is a capability of the R1Pro
# parallel gripper, not a task pose: when a semantic grasp-center motion has
# an underconstrained orientation, the motion skill may use this mirrored
# left/right default and still let the caller override it with a full pose.
R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE = {
    "left": (0.7071067811865476, 0.0, -0.7071067811865476, 0.0),
    "right": (0.7071067811865476, 0.0, 0.7071067811865476, 0.0),
}

# Robot-intrinsic height bounds used only to classify support-relative geometry.
# They are not posture targets.  A valid low-workspace configuration must be
# solved from the live object/support/base geometry and pass the physical
# stability certificate; no fixed torso vector is a valid fallback.
R1PRO_GROUND_INTERACTION_CENTER_Z_M = 0.40
R1PRO_SAFE_APPROACH_CENTER_Z_M = 0.55
R1PRO_GROUND_INTERACTION_MAX_STAGED_EE_Z_M = 0.80
R1PRO_GROUND_INTERACTION_MAX_STAGED_JOINT_SPEED_RADPS = 0.15

# Whole-body manipulation profiles.  These are robot capabilities rather than
# task poses: a support-to-support transfer may need to lift a held object,
# change the torso configuration, and keep the live grasp frame fixed while
# the arm is re-solved.  Keeping the profiles here lets the same semantic
# skill operate on arbitrary scene objects and support surfaces.
R1PRO_TRANSFER_TORSO_Q = (0.0, 0.0, 0.0, 0.0)
# Standing torso used by the public prepare_workspace profiles that do not
# require a certified low-support sweep. Floor remains a whole-body solve.
R1PRO_WORKSPACE_TORSO_Q = {
    "tabletop": R1PRO_TRANSFER_TORSO_Q,
    "carry": R1PRO_TRANSFER_TORSO_Q,
    "travel": R1PRO_TRANSFER_TORSO_Q,
}
R1PRO_TRANSFER_TORSO_REACHED_TOL_RAD = 0.12
R1PRO_TRANSFER_HOLD_CENTER_TOL_M = 0.08
R1PRO_TRANSFER_MAX_ARM_STEP_RAD = 0.24
# Maximum whole-body transition speed currently verified with the authored
# R1Pro position-drive gains and the runtime effort reserve. This is a
# robot-level capability ceiling; semantic callers may request a slower
# profile, but a planner cannot bypass the physical limit by emitting a
# larger speed value.  The limit is deliberately derived from the physical
# controller validation, not from any task or scene coordinate.
R1PRO_WHOLE_BODY_MAX_SPEED_SCALE = 0.16
# Maximum speed fraction for the unloaded arm transition from the hanging
# home posture to the robot's ready posture.  0.28 saturates left_arm_joint4
# (25 N·m) for >0.25 s while the elbow lifts the forearm; 0.18 still raises
# the arm in about two seconds without holding the clamp.
R1PRO_READY_POSE_SPEED_SCALE = 0.18
# Maximum change in a commanded arm target during a loaded whole-body
# transition.  This is a drive/effort capability of the supplied R1Pro
# actuator profile, not a task waypoint: with the configured position-drive
# stiffness it keeps the first-order PD demand inside the runtime torque
# reserve while the live effort gate still remains authoritative.
R1PRO_WHOLE_BODY_MAX_TARGET_STEP_RAD = 0.02
R1PRO_TRANSFER_TRACK_TOL_RAD = 0.10
# A 0.02-rad arm increment can take roughly 0.8--1.2 s to settle under the
# authored implicit drive while carrying the gravity-loaded torso.  Thirty
# 60-Hz samples made the controller abort during a still-converging, low-
# effort transient (~0.012 rad), even though the next certified increment was
# safe.  Keep the fail-closed tolerance, but allow the physical drive enough
# time to converge; the effort/tilt/contact gates remain authoritative.
R1PRO_TRANSFER_MAX_TRACK_STEPS = 90
R1PRO_TRANSFER_LIFT_CLEARANCE_M = 0.13

# Gripper collision-envelope facts used by support-aware grasp acquisition.
# These are dimensions of the supplied R1Pro gripper collision primitives, not
# task/object coordinates.  Keeping them in the robot capability profile lets
# the approach planner scale its clearance for cylinders, cuboids, and future
# primitive objects without embedding a scene-specific standoff.
R1PRO_GRIPPER_FINGER_HALF_LENGTH_M = 0.03
R1PRO_GRIPPER_FINGER_HALF_HEIGHT_M = 0.02
R1PRO_GRIPPER_PREGRASP_CLEARANCE_M = 0.015
# Minimum effective overlap of each physical finger box with the object's
# vertical band before a pose is considered close-ready.  A merely positive
# convex-box overlap can be a grazing edge intersection: on the supplied
# R1Pro asset, about 2 mm passed the geometric gate but produced no second-
# finger contact.  Keep this robot calibration independent of task poses and
# scale it down for objects shorter than the normal finger envelope.
R1PRO_GRIPPER_MIN_FINGER_VERTICAL_OVERLAP_M = 0.01


def gripper_min_vertical_overlap_m(object_height_m: float) -> float:
    """Return the two-finger terminal overlap required for an object height."""
    height = float(object_height_m)
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("object_height_m must be finite and positive")
    return min(R1PRO_GRIPPER_MIN_FINGER_VERTICAL_OVERLAP_M, 0.25 * height)


# The gripper-link collision mesh is authored above its frame origin.  The
# sphere proxy used by the lightweight path checker is therefore centered at
# this measured local mesh centroid, not at the joint/frame origin.
R1PRO_GRIPPER_LINK_COLLISION_CENTER_LOCAL = (0.0, 0.0, 0.0304551)
R1PRO_GRIPPER_LINK_COLLISION_RADIUS_M = 0.037
# Local axis of the two mirrored prismatic finger joints.  A live open-jaw
# pose can therefore be projected to the fully-closed pose without issuing a
# command; the grasp window uses that projection only as a contact-feasibility
# certificate.
R1PRO_GRIPPER_FINGER_PRISMATIC_AXIS_LOCAL = (0.0, 1.0, 0.0)
# Conservative fallback for the planar collision envelope of the gripper
# link/finger origins around the physical finger midpoint. Runtime adapters
# may replace this lower bound with a measured envelope; this value remains a
# robot capability calibration, never a task or scene coordinate.
R1PRO_GRIPPER_COLLISION_ENVELOPE_M = 0.075
# Axis-aligned bounds of the collision meshes in each finger link frame.  The
# bounds are measured from the supplied USD meshes
# ``usd/{left,right}_gripper_finger_link{1,2}.usd`` (the mirrored assets have
# the same dimensions).  They are deliberately kept as a robot capability
# description, so acquisition code can use the actual finger envelope for
# arbitrary primitive objects instead of treating the two link origins as a
# zero-thickness jaw.
R1PRO_GRIPPER_FINGER_COLLISION_BOXES_LOCAL = {
    "finger_link1": {
        "center": (0.0046750, -0.0300707, 0.0045148),
        "half_extents": (0.0187913, 0.0348479, 0.0459799),
    },
    "finger_link2": {
        "center": (-0.0046753, 0.0300706, 0.0045148),
        "half_extents": (0.0187914, 0.0348478, 0.0459798),
    },
}
# Conservative vertical envelope from the physical grasp midpoint to the
# lowest point of either open-finger collision box over the supported arm
# orientations.  It is used only when deriving a non-contact pregrasp above a
# source plane; the exact live box pose remains the runtime authority.
R1PRO_GRIPPER_FINGER_SUPPORT_ENVELOPE_Z_M = 0.08
# Reserve for the measured tracking error of the whole-body position drive
# when a low-support target is reached.  This is not a scene/object height:
# it prevents a compliant endpoint that is a few millimetres below its
# geometry-derived target from consuming the entire source-plane clearance.
# The exact live finger-box certificate remains authoritative at execution.
R1PRO_GRIPPER_PREGRASP_SUPPORT_RESERVE_M = 0.012
# Bounded in-plane orientation samples for support-aware acquisition.  They
# are robot capability redundancies around the support normal, not task or
# object poses; the live collision/IK gate chooses whether any is usable.
R1PRO_SUPPORT_AWARE_YAW_OFFSETS_RAD = (0.0, 1.5707963267948966, -1.5707963267948966, 3.141592653589793)
# Planar approach-direction redundancy for low-clearance acquisition. The
# support-aware skill tries the radial direction first, then tangent/opposite
# directions only when the robot's live workspace or collision certificate
# rejects it. These are capability-level search offsets, not task waypoints.
R1PRO_SUPPORT_AWARE_APPROACH_OFFSETS_RAD = (0.0, 1.5707963267948966, -1.5707963267948966, 3.141592653589793)

# Robot-intrinsic manipulation rest posture. Home (all-zero) hangs the
# forearm along base +y and collides with nearby tabletops. The previous
# ready pose pitched J1 to -90 deg with only a shallow elbow, which parked
# the gripper at about (x=-0.03, y=+/-0.88, z=1.71): high, lateral, slightly
# behind the shoulder. That is an IK seed, not a rest pose.
#
# This vector is R1Pro's published untucked default (OmniGibson R1Pro /
# GalaxeaManipSim homepage demos such as R1ProBlocksStackHard): shoulders
# abducted, elbows at 90 deg, wrists rotated so the parallel-jaw grippers
# point down in front of the chest. Full 90 deg shoulder abduction (J2 =
# +/-pi/2) parks the EE near z=1.45 m, which reads as too high against the
# torso. Keep the same elbow/wrist, drop J2 to +/-1.10 rad so FK left EE is
# about (0.34, 0.53, 1.31) with the default grasp quaternion. Not a task pose.
R1PRO_ARM_READY_Q_BY_SIDE = {
    "left": (0.0, 1.10, 0.0, -1.5708, 1.5708, 0.0, 0.0),
    "right": (0.0, -1.10, 0.0, -1.5708, -1.5708, 0.0, 0.0),
}


def arm_torque_by_joint(side: str) -> dict[str, float]:
    """Torque limit dict for one arm, keyed by joint name."""
    return {
        f"{side}_arm_joint{i}": v
        for i, v in enumerate(R1PRO_ARM_TORQUE_LIMITS, 1)
    }


def arm_velocity_by_joint(side: str) -> dict[str, float]:
    """Velocity limit dict for one arm, keyed by joint name."""
    return {
        f"{side}_arm_joint{i}": v
        for i, v in enumerate(R1PRO_ARM_VELOCITY_LIMITS, 1)
    }
