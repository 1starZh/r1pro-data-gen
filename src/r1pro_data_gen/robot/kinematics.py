"""R1Pro kinematics: Pinocchio FK and damped-least-squares IK for either arm.

Pure logic (pinocchio only, no isaaclab). Loads the URDF model, exposes the
selected arm joint chain and end-effector frame, and provides FK + DLS IK with
joint-limit clamping. The arm is treated as a 7-DOF chain; all other joints
are frozen at zero (the neutral home).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pinocchio as pin

ARM_JOINTS_BY_SIDE = {
    side: tuple(f"{side}_arm_joint{i}" for i in range(1, 8))
    for side in ("left", "right")
}
EE_FRAME_BY_SIDE = {side: f"{side}_gripper_link" for side in ("left", "right")}
BASE_CALIBRATION_FRAMES_BY_SIDE = {
    side: tuple(f"{side}_arm_link{i}" for i in range(1, 8)) + (EE_FRAME_BY_SIDE[side],)
    for side in ("left", "right")
}
# Backward-compatible left-arm aliases used by existing task code/tests.
ARM_JOINTS = ARM_JOINTS_BY_SIDE["left"]
EE_FRAME = EE_FRAME_BY_SIDE["left"]
BASE_CALIBRATION_FRAMES = BASE_CALIBRATION_FRAMES_BY_SIDE["left"]
# DLS parameters (manipulation-ik skill guidance, tuned for reachable
# targets near the workspace boundary: larger delta + more iterations).
DLS_DAMPING = 0.05
MAX_DELTA = 0.05  # rad per iteration
MAX_ITERS = 500
POS_TOL = 0.01  # m
ROT_TOL = 0.05  # rad
IK_RESTARTS = 5  # restart attempts from perturbed seeds

# Pink-QP bounded IK (see ik_qp): measured-stable parameters on this hardware
# model.  The QP path exists because fixed-damping DLS structurally stalls at
# joint-limit boundaries (the clip pins dq while the error stays large);
# box-constrained QP keeps iterating along the constraint instead.
QP_IK_DT = 4e-3
QP_IK_LM_DAMPING = 0.05
QP_IK_MAX_ITERS = 150
QP_IK_POSTURE_COST = 1e-2
QP_IK_LIMIT_GUARD_RAD = 0.002  # inset so soft limit damping still yields in-bounds solutions
QP_FALLBACK_TO_DLS = True  # per-seed DLS retry when the QP path fails

# The physical grasp anchor is the midpoint between the two finger bodies,
# not the gripper-link origin.  A pose IK that fixes the wrist orientation can
# therefore be infeasible even though the same object-centre position is
# reachable with a different redundant-arm posture.  These bounds belong to
# the robot kinematics capability; callers still provide the live target.
GRASP_CENTER_IK_DAMPING = 0.04
GRASP_CENTER_IK_MAX_DELTA = 0.08
GRASP_CENTER_IK_MAX_ITERS = 180
GRASP_CENTER_IK_POS_TOL = 0.008

# The parallel-jaw acquisition variable is weaker than a full 6-D EE pose:
# the finger midpoint must reach the object centre while the jaw span should
# be normal to the live approach direction.  A full orientation target can be
# outside the low-workspace arm manifold even when this physically sufficient
# grasp geometry is reachable.  These are solver scales/tolerances, not task
# coordinates or a stored grasp pose.  The horizontal direction remains the
# primary acquisition variable; a separate robot-level tilt gate prevents a
# nearly vertical jaw from hiding behind a short, apparently correct XY
# projection.
GRASP_WINDOW_IK_MAX_NFEV = 260
# Contact-sensitive alignment must also level the physical jaw span. A loose
# support-plane-only tolerance allowed millimetres of finger-height skew, so
# one finger could displace an object before the other entered its contact
# band. Two degrees remains above tracking noise while requiring the bounded
# grasp-window IK to remove that one-sided-contact geometry before descent.
GRASP_WINDOW_IK_DIRECTION_TOL = math.radians(2.0)
GRASP_WINDOW_IK_POSITION_SCALE_M = 0.003
GRASP_WINDOW_IK_DIRECTION_SCALE_RAD = 0.12
# A parallel-jaw acquisition is evaluated in the support plane, but the two
# physical finger boxes should also stay at a similar height.  This is a soft
# redundancy preference rather than a hard 3-D orientation constraint: some
# low-support arm configurations cannot make the link origins exactly level,
# while the runtime box-overlap certificate remains the physical authority.
GRASP_WINDOW_IK_VERTICAL_SPAN_SCALE_M = 0.04
GRASP_WINDOW_IK_VERTICAL_SPAN_WEIGHT = 0.50
# Keep this gate permissive enough for the measured link-origin geometry (some
# reachable postures are around 57 degrees tilted), while rejecting the
# near-vertical degeneracy whose XY projection cannot form a floor window.
GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD = math.radians(65.0)

# A robot-level posture prior, not a task-specific cached IK.  It describes a
# relaxed forward reach with a bent elbow and neutral wrist. Online IK remains
# responsible for satisfying every live target pose.
NATURAL_REACH_Q_BY_SIDE = {
    "left": np.array([-0.45, -0.09, -0.06, -0.94, -0.06, -0.18, -0.05], dtype=float),
    # The shoulder lateral joint has mirrored bounds. Distal signs are kept
    # neutral so online IK, rather than a task pose, determines the wrist.
    "right": np.array([-0.45, 0.09, 0.06, -0.94, 0.06, -0.18, 0.05], dtype=float),
}
NATURAL_REACH_Q = NATURAL_REACH_Q_BY_SIDE["left"]


@dataclass(frozen=True, slots=True)
class IKSolution:
    success: bool
    q_arm: np.ndarray | None
    position_error: float
    rotation_error: float
    iterations: int
    reason: str = ""


@dataclass(frozen=True, slots=True)
class WholeBodyIKSolution:
    """A bounded position solution for the torso plus one arm.

    The interaction variable is the physical finger midpoint.  The torso
    configuration is part of the solution rather than an authored pose, so a
    low object can be reached by a coordinated whole-body posture.  Collision
    and support/effort checks deliberately remain the responsibility of the
    caller because they depend on the live scene and base state.
    """

    success: bool
    q_arm: np.ndarray | None
    q_torso: np.ndarray | None
    position_error: float
    iterations: int
    reason: str = ""


def _quat_error_rotation_vector(q_ref: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Rotation vector from current quaternion to reference (wxyz quats)."""
    q_ref = q_ref / np.linalg.norm(q_ref)
    q_cur = q_cur / np.linalg.norm(q_cur)
    # dq = q_ref * conj(q_cur)
    w1, x1, y1, z1 = q_ref
    w2, x2, y2, z2 = q_cur
    dq = np.array(
        [
            w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
            -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2,
            -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2,
            -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2,
        ]
    )
    dq = dq / np.linalg.norm(dq)
    # Canonicalize the sign: an antipodal quaternion is the SAME rotation, so
    # without this the accepted-error check can report ~2*pi for zero error.
    if dq[0] < 0.0:
        dq = -dq
    angle = 2.0 * math.atan2(np.linalg.norm(dq[1:]), dq[0])
    if angle < 1e-12:
        return np.zeros(3)
    return dq[1:] / np.linalg.norm(dq[1:]) * angle


class R1ProKinematics:
    """Pinocchio model wrapper for one selected R1Pro arm chain."""

    def __init__(self, urdf_path: str, side: str = "left") -> None:
        if side not in ARM_JOINTS_BY_SIDE:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.urdf_path = str(urdf_path)
        self.side = side
        self.arm_joints = ARM_JOINTS_BY_SIDE[side]
        self.ee_frame = EE_FRAME_BY_SIDE[side]
        self.base_calibration_frames = BASE_CALIBRATION_FRAMES_BY_SIDE[side]
        self.natural_reach_q = NATURAL_REACH_Q_BY_SIDE[side].copy()
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.arm_joint_ids = [self.model.getJointId(name) for name in self.arm_joints]
        self.torso_joints = tuple(f"torso_joint{i}" for i in range(1, 5))
        self.torso_joint_ids = [self.model.getJointId(name) for name in self.torso_joints]
        # Joint ids are NOT q indices: fixed joints occupy nq dimensions with
        # no nv counterpart, so the mapping is not identity (this asset has 3
        # fixed joints). q_offset gives the correct q index; using joint ids
        # directly silently reads/writes the wrong configuration.
        self.arm_q_indices = [self.model.joints[jid].idx_q for jid in self.arm_joint_ids]
        self.arm_nv_indices = [self.model.joints[jid].idx_v for jid in self.arm_joint_ids]
        self.torso_q_indices = [self.model.joints[jid].idx_q for jid in self.torso_joint_ids]
        self.torso_nv_indices = [self.model.joints[jid].idx_v for jid in self.torso_joint_ids]
        self.ee_frame_id = self.model.getFrameId(self.ee_frame)
        # Override limits with the authoritative USDA values (same values
        # Isaac Sim enforces) to keep IK consistent with the physics scene.
        from r1pro_data_gen.robot.robot_config import R1PRO_JOINT_LIMITS

        for name, qidx in zip(self.arm_joints, self.arm_q_indices):
            lower, upper = R1PRO_JOINT_LIMITS[name]
            if lower is not None:
                self.model.lowerPositionLimit[qidx] = lower
            if upper is not None:
                self.model.upperPositionLimit[qidx] = upper
        self.lower = self.model.lowerPositionLimit[self.arm_q_indices]
        self.upper = self.model.upperPositionLimit[self.arm_q_indices]
        self._grasp_center_offset_local = self._measure_grasp_center_offset_local()

    def fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics of the EE frame for a 7-DOF arm config.

        Returns (position (3,), quaternion (w, x, y, z)).
        """
        q = self._full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_frame_id]
        # Copy: pose.translation aliases the internal data buffer, which the
        # next forwardKinematics call overwrites in place.
        pos = np.array(pose.translation, dtype=float)
        # Pinocchio Quaternion.coeffs() returns (x, y, z, w) -- roll to (w, x,
        # y, z) to match the convention used everywhere else in this module.
        coeffs = np.array(pin.Quaternion(pose.rotation).coeffs(), dtype=float)
        quat = np.roll(coeffs, 1)
        return pos, quat

    def _measure_grasp_center_offset_local(self) -> np.ndarray:
        """Measure the finger-body midpoint in the gripper-link frame."""
        q = self._full_q(np.zeros(7, dtype=float))
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        ee = self.data.oMf[self.ee_frame_id]
        p1 = np.asarray(self.data.oMf[self.model.getFrameId(f"{self.side}_gripper_finger_link1")].translation)
        p2 = np.asarray(self.data.oMf[self.model.getFrameId(f"{self.side}_gripper_finger_link2")].translation)
        return np.asarray(ee.rotation).T @ ((p1 + p2) * 0.5 - np.asarray(ee.translation))

    @property
    def grasp_center_offset_local(self) -> np.ndarray:
        return self._grasp_center_offset_local.copy()

    def grasp_center_fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics of the midpoint between the two finger bodies."""
        ee_pos, ee_quat = self.fk(q_arm)
        rotation = pin.Quaternion(
            float(ee_quat[0]), float(ee_quat[1]), float(ee_quat[2]), float(ee_quat[3])
        ).matrix()
        return ee_pos + rotation @ self._grasp_center_offset_local, ee_quat

    def grasp_center_jacobian(self, q_arm: np.ndarray) -> np.ndarray:
        """Numerical position Jacobian of the physical grasp-center anchor.

        The URDF arm chain exposes the gripper-link frame, while interaction
        skills reason about the midpoint of the two finger bodies.  The
        midpoint offset rotates with the wrist, so reusing the gripper-link
        Jacobian for a position-only grasp goal introduces a configuration
        dependent error.  A small central difference keeps this capability
        correct for the reduced model and for both mirrored arms without
        making the semantic skills know the robot's link layout.
        """
        q_arm = np.asarray(q_arm, dtype=float)
        if q_arm.shape != (7,) or not np.all(np.isfinite(q_arm)):
            raise ValueError("q_arm must be a finite 7-vector")
        jacobian = np.zeros((3, 7), dtype=float)
        for index in range(7):
            # Keep the finite-difference pair inside the authoritative joint
            # limits.  The step is large enough to avoid Pinocchio roundoff
            # and small enough to describe the local anchor motion.
            step = 2.0e-4
            q_plus = q_arm.copy()
            q_minus = q_arm.copy()
            q_plus[index] = min(float(self.upper[index]), q_arm[index] + step)
            q_minus[index] = max(float(self.lower[index]), q_arm[index] - step)
            denominator = q_plus[index] - q_minus[index]
            if denominator <= 1.0e-12:
                continue
            plus = self.grasp_center_fk(q_plus)[0]
            minus = self.grasp_center_fk(q_minus)[0]
            jacobian[:, index] = (plus - minus) / denominator
        return jacobian

    def finger_frame_fk(
        self, q_arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return finger-link origins and rotations in the model frame.

        The reduced arm model freezes the two prismatic opening joints at
        zero, which is sufficient for arm IK.  Their opening displacement is
        supplied by the live adapter when a physical collision envelope is
        reconstructed.  Returning the link rotations here lets that envelope
        follow the actual arm posture instead of using an axis-aligned sphere
        or a task-specific jaw pose.
        """
        q = self._full_q(np.asarray(q_arm, dtype=float))
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError("q_arm must form a finite model configuration")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        poses = []
        for index in (1, 2):
            pose = self.data.oMf[
                self.model.getFrameId(f"{self.side}_gripper_finger_link{index}")
            ]
            poses.append(
                (
                    np.asarray(pose.translation, dtype=float).copy(),
                    np.asarray(pose.rotation, dtype=float).copy(),
                )
            )
        return poses[0][0], poses[0][1], poses[1][0], poses[1][1]

    def finger_geometry_fk(
        self,
        q_arm: np.ndarray,
        opening_offsets: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the measured/open-finger geometry predicted by arm joints.

        The reduced arm model freezes the two prismatic finger joints at zero.
        During an open-jaw alignment their measured displacements are held
        constant in each finger-link frame.  Applying those offsets here keeps
        the window IK objective identical to the physical finger geometry used
        by the runtime collision certificate.
        """
        p1, r1, p2, r2 = self.finger_frame_fk(q_arm)
        if opening_offsets is not None:
            if len(opening_offsets) != 2:
                raise ValueError("opening_offsets must contain two vectors")
            offsets = []
            for offset in opening_offsets:
                value = np.asarray(offset, dtype=float)
                if value.shape != (3,) or not np.all(np.isfinite(value)):
                    raise ValueError("opening offsets must be finite 3-vectors")
                offsets.append(value)
            p1 = np.asarray(p1, dtype=float) + np.asarray(r1, dtype=float) @ offsets[0]
            p2 = np.asarray(p2, dtype=float) + np.asarray(r2, dtype=float) @ offsets[1]
        center = 0.5 * (np.asarray(p1, dtype=float) + np.asarray(p2, dtype=float))
        return (
            np.asarray(p1, dtype=float),
            np.asarray(r1, dtype=float),
            np.asarray(p2, dtype=float),
            np.asarray(r2, dtype=float),
            center,
        )

    def finger_span_fk(
        self,
        q_arm: np.ndarray,
        opening_offsets: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the two finger-link origins and their span in model frame.

        The parallel finger prismatic joints translate along the finger-span
        axis, so the reduced seven-DOF model can use the link-origin span to
        constrain jaw direction without pretending that finger opening is an
        arm orientation variable.  The physical adapter supplies the live
        opening for the final finite-segment/contact checks.
        """
        p1, _, p2, _, _ = self.finger_geometry_fk(q_arm, opening_offsets)
        return p1, p2, p2 - p1

    def ik_grasp_window_candidates(
        self,
        target_center: np.ndarray,
        desired_span: np.ndarray,
        q_current: np.ndarray,
        *,
        max_candidates: int = 6,
        position_tolerance: float = GRASP_CENTER_IK_POS_TOL,
        direction_tolerance: float = GRASP_WINDOW_IK_DIRECTION_TOL,
        opening_offsets: tuple[np.ndarray, np.ndarray] | None = None,
        span_to_constraint_rotation: np.ndarray | None = None,
    ) -> list[IKSolution]:
        """Solve the physically sufficient parallel-jaw acquisition geometry.

        The target is the live finger midpoint plus a desired jaw-span
        direction.  It deliberately does not require an arbitrary full
        wrist orientation: the R1Pro can reach some low-support positions only
        when its redundant wrist/upper-arm posture is allowed to vary.  The
        caller still certifies the joint path, live collision envelope,
        effort, and attachment before treating any result as a grasp.

        ``desired_span`` is expressed in the same frame used for the direction
        constraint.  By default that is the kinematics model frame.  When
        ``span_to_constraint_rotation`` is provided, the model-predicted
        finger span is first rotated into that frame before its horizontal
        direction is compared.  This is needed when a measured model-to-world
        calibration contains pitch/roll: projecting a world-horizontal target
        into the model frame and then dropping its model Z component changes
        the physical jaw direction.  ``opening_offsets`` are optional fixed
        prismatic displacements measured in each finger-link frame; when
        supplied, the physical open-jaw center is solved together with the
        horizontal jaw span. The vertical separation of the two real finger
        bodies is left to the collision-box/window certificate: an exactly
        horizontal 3-D link-origin line can reject a valid floor grasp even
        when both finger contact envelopes overlap the object vertically.
        """
        target_center = np.asarray(target_center, dtype=float)
        desired_span = np.asarray(desired_span, dtype=float)
        q_current = np.asarray(q_current, dtype=float)
        if (
            target_center.shape != (3,)
            or desired_span.shape != (3,)
            or q_current.shape != (7,)
            or not np.all(np.isfinite(target_center))
            or not np.all(np.isfinite(desired_span))
            or not np.all(np.isfinite(q_current))
        ):
            raise ValueError("window IK inputs must be finite target/span/q vectors")
        if span_to_constraint_rotation is not None:
            span_to_constraint_rotation = np.asarray(
                span_to_constraint_rotation, dtype=float
            )
            if (
                span_to_constraint_rotation.shape != (3, 3)
                or not np.all(np.isfinite(span_to_constraint_rotation))
            ):
                raise ValueError(
                    "span_to_constraint_rotation must be a finite 3x3 matrix"
                )
        # A parallel jaw acquires a floor object through the horizontal
        # projection of its finite opening. The physical vertical placement
        # and overlap of both finger boxes are checked by the adapter after
        # this reduced IK solve; making link origins exactly level here
        # over-constrains the low-support arm configuration.
        desired_xy_norm = float(np.linalg.norm(desired_span[:2]))
        if desired_xy_norm <= 1.0e-8:
            raise ValueError("desired_span must have a non-zero horizontal component")
        desired_direction = desired_span[:2] / desired_xy_norm
        desired_span_norm = float(np.linalg.norm(desired_span))
        # The runtime floor/table acquisition target is world-horizontal, so
        # the tilt gate is meaningful in that branch.  A caller may instead
        # provide a deliberately tilted constraint frame (for example a
        # calibrated test or a sloped support); in that case the target's own
        # inclination is the capability frame and this world-Z gate must not
        # reject it.
        tilt_gate_active = (
            abs(float(desired_span[2])) / max(desired_span_norm, 1.0e-12)
            <= math.sin(GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD)
        )
        q_current = np.clip(q_current, self.lower, self.upper)
        _, _, current_span = self.finger_span_fk(q_current, opening_offsets)
        if span_to_constraint_rotation is not None:
            current_span = span_to_constraint_rotation @ current_span
        current_xy_norm = float(np.linalg.norm(current_span[:2]))
        if current_xy_norm <= 1.0e-8:
            return []
        current_direction = current_span[:2] / current_xy_norm
        if float(np.dot(current_direction, desired_direction)) < 0.0:
            # The two fingers are interchangeable. Keep the sign closest to
            # the measured horizontal branch; the physical box certificate
            # handles the height relation at the actual contact envelopes.
            desired_direction = -desired_direction

        try:
            from scipy.optimize import least_squares
        except ImportError:
            return []

        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        span_limits = np.maximum(upper - lower, 1.0e-9)

        def direction_vector(q_arm: np.ndarray) -> np.ndarray:
            _, _, span = self.finger_span_fk(q_arm, opening_offsets)
            if span_to_constraint_rotation is not None:
                span = span_to_constraint_rotation @ span
            span_norm = float(np.linalg.norm(span[:2]))
            if span_norm <= 1.0e-8:
                return np.full(2, np.inf, dtype=float)
            return span[:2] / span_norm

        def direction_error(q_arm: np.ndarray) -> float:
            direction = direction_vector(q_arm)
            if not np.all(np.isfinite(direction)):
                return math.pi
            # The reduced IK direction objective is the support-plane yaw. A
            # separate tilt residual/gate below prevents a degenerate vertical
            # span without turning the link-origin height into a hard pose.
            residual = float(np.linalg.norm(direction - desired_direction))
            return float(2.0 * math.asin(min(1.0, 0.5 * residual)))

        def constraint_span(q_arm: np.ndarray) -> np.ndarray:
            """Return the predicted finger span in the direction frame."""
            span = self.finger_span_fk(q_arm, opening_offsets)[2]
            if span_to_constraint_rotation is not None:
                span = span_to_constraint_rotation @ span
            return np.asarray(span, dtype=float)

        def residual(q_arm: np.ndarray) -> np.ndarray:
            center = np.asarray(
                self.finger_geometry_fk(q_arm, opening_offsets)[4],
                dtype=float,
            )
            span = constraint_span(q_arm)
            return np.concatenate(
                (
                    (center - target_center) / GRASP_WINDOW_IK_POSITION_SCALE_M,
                    (direction_vector(q_arm) - desired_direction)
                    / GRASP_WINDOW_IK_DIRECTION_SCALE_RAD,
                    np.asarray(
                        [
                            GRASP_WINDOW_IK_VERTICAL_SPAN_WEIGHT
                            * (span[2] - float(desired_span[2]))
                            / GRASP_WINDOW_IK_VERTICAL_SPAN_SCALE_M
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        [
                            max(
                                0.0,
                                math.asin(
                                    min(
                                        1.0,
                                        abs(float(span[2]))
                                        / max(float(np.linalg.norm(span)), 1.0e-12),
                                    )
                                )
                                - GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD,
                            )
                            / GRASP_WINDOW_IK_DIRECTION_SCALE_RAD
                        ],
                        dtype=float,
                    )
                    if tilt_gate_active
                    else np.zeros(1, dtype=float),
                    # Keep the selected local branch close without making
                    # continuity a substitute for the physical geometry.
                    0.01 * (q_arm - q_current) / span_limits,
                )
            )

        rng = np.random.default_rng(43)
        seeds: list[np.ndarray] = [q_current.copy(), self.natural_reach_q.copy()]
        for scale in (0.18, 0.35, 0.60):
            seeds.append(
                np.clip(
                    q_current + rng.normal(0.0, scale, size=7), lower, upper
                )
            )
            seeds.append(
                np.clip(
                    self.natural_reach_q + rng.normal(0.0, scale, size=7),
                    lower,
                    upper,
                )
            )

        found: dict[tuple[int, ...], IKSolution] = {}
        for seed in seeds:
            try:
                solved = least_squares(
                    residual,
                    np.clip(seed, lower, upper),
                    bounds=(lower, upper),
                    max_nfev=GRASP_WINDOW_IK_MAX_NFEV,
                    ftol=1.0e-9,
                    xtol=1.0e-9,
                    gtol=1.0e-9,
                    x_scale="jac",
                )
                q_solution = np.clip(np.asarray(solved.x, dtype=float), lower, upper)
                center_error = float(
                    np.linalg.norm(
                        self.finger_geometry_fk(q_solution, opening_offsets)[4]
                        - target_center
                    )
                )
                angle_error = abs(direction_error(q_solution))
                solved_span = constraint_span(q_solution)
                solved_span_norm = float(np.linalg.norm(solved_span))
                solved_tilt = (
                    math.asin(
                        min(
                            1.0,
                            abs(float(solved_span[2]))
                            / max(solved_span_norm, 1.0e-12),
                        )
                    )
                    if solved_span_norm > 1.0e-8
                    else math.pi / 2.0
                )
            except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                continue
            if (
                q_solution.shape != (7,)
                or not np.all(np.isfinite(q_solution))
                or not np.isfinite(center_error)
                or not np.isfinite(angle_error)
                or not np.isfinite(solved_tilt)
                or center_error > float(position_tolerance)
                or angle_error > float(direction_tolerance)
                or (
                    tilt_gate_active
                    and solved_tilt > GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD
                )
            ):
                continue
            solution = IKSolution(
                success=True,
                q_arm=q_solution.copy(),
                position_error=center_error,
                rotation_error=angle_error,
                iterations=int(getattr(solved, "nfev", 0)),
                reason="grasp-center and jaw-span direction reached",
            )
            key = tuple(int(round(float(value) / 0.035)) for value in q_solution)
            previous = found.get(key)
            if previous is None or (
                solution.position_error + solution.rotation_error
                < previous.position_error + previous.rotation_error
            ):
                found[key] = solution

        solutions = list(found.values())
        solutions.sort(
            key=lambda item: (
                round(float(np.linalg.norm((item.q_arm - q_current) / span_limits)), 10),
                round(float(item.position_error), 10),
                round(float(item.rotation_error), 10),
                *(int(round(float(value) * 100_000.0)) for value in item.q_arm),
            )
        )
        return solutions[: max(1, int(max_candidates))]

    def _grasp_center_ik_seed(
        self,
        target_center: np.ndarray,
        seed: np.ndarray,
        *,
        position_tolerance: float = GRASP_CENTER_IK_POS_TOL,
        max_iters: int = GRASP_CENTER_IK_MAX_ITERS,
    ) -> IKSolution:
        """Solve a bounded position-only IK problem for the grasp anchor."""
        target_center = np.asarray(target_center, dtype=float)
        q_arm = np.asarray(seed, dtype=float).copy()
        if target_center.shape != (3,) or not np.all(np.isfinite(target_center)):
            raise ValueError("target_center must be a finite 3-vector")
        if q_arm.shape != (7,) or not np.all(np.isfinite(q_arm)):
            raise ValueError("seed must be a finite 7-vector")
        q_arm = np.clip(q_arm, self.lower, self.upper)
        last_error = float("inf")
        for iteration in range(1, max(1, int(max_iters)) + 1):
            center, _ = self.grasp_center_fk(q_arm)
            error = target_center - center
            error_norm = float(np.linalg.norm(error))
            if error_norm <= float(position_tolerance):
                return IKSolution(
                    success=True,
                    q_arm=q_arm.copy(),
                    position_error=error_norm,
                    rotation_error=0.0,
                    iterations=iteration,
                    reason="grasp-center position reached",
                )
            jacobian = self.grasp_center_jacobian(q_arm)
            normal = jacobian @ jacobian.T
            try:
                delta = jacobian.T @ np.linalg.solve(
                    normal + GRASP_CENTER_IK_DAMPING**2 * np.eye(3),
                    error,
                )
            except np.linalg.LinAlgError:
                break
            delta = np.clip(delta, -GRASP_CENTER_IK_MAX_DELTA, GRASP_CENTER_IK_MAX_DELTA)
            if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) <= 1.0e-9:
                break

            # A bounded backtracking step prevents a near-singular local
            # update from jumping to a worse configuration.  It does not use
            # scene geometry; collision certification remains the caller's
            # responsibility after a kinematic solution is found.
            accepted = False
            scale = 1.0
            for _ in range(6):
                candidate = np.clip(q_arm + scale * delta, self.lower, self.upper)
                candidate_error = float(
                    np.linalg.norm(target_center - self.grasp_center_fk(candidate)[0])
                )
                if candidate_error < error_norm:
                    q_arm = candidate
                    last_error = candidate_error
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
        center, _ = self.grasp_center_fk(q_arm)
        return IKSolution(
            success=False,
            q_arm=q_arm.copy(),
            position_error=float(np.linalg.norm(target_center - center)),
            rotation_error=0.0,
            iterations=max(1, int(max_iters)),
            reason="grasp-center max iterations or no progress",
        )

    def ik_grasp_center_candidates(
        self,
        target_center: np.ndarray,
        q_current: np.ndarray,
        *,
        max_candidates: int = 6,
        position_tolerance: float = GRASP_CENTER_IK_POS_TOL,
    ) -> list[IKSolution]:
        """Return continuous bounded IK branches for a grasp-center target.

        This is intentionally distinct from :meth:`ik_candidates`: the
        latter solves a gripper-link pose and can silently move the physical
        finger midpoint when orientation is relaxed.  The center solver
        treats the interaction anchor as the task variable, then leaves any
        orientation preference and obstacle/path checks to higher-level
        manipulation skills.
        """
        target_center = np.asarray(target_center, dtype=float)
        q_current = np.asarray(q_current, dtype=float)
        if target_center.shape != (3,) or not np.all(np.isfinite(target_center)):
            raise ValueError("target_center must be a finite 3-vector")
        if q_current.shape != (7,) or not np.all(np.isfinite(q_current)):
            raise ValueError("q_current must be a finite 7-vector")
        q_current = np.clip(q_current, self.lower, self.upper)
        rng = np.random.default_rng(31)
        seeds: list[np.ndarray] = [q_current, self.natural_reach_q.copy()]
        # Keep the deterministic current branch first.  Perturbed branches
        # are only alternate redundancy choices when the local basin is
        # blocked or produces a path that the caller later rejects.
        for base in (q_current, self.natural_reach_q):
            for scale in (0.20, 0.45, 0.75):
                seeds.append(np.clip(base + rng.normal(0.0, scale, size=7), self.lower, self.upper))

        unique: dict[tuple[int, ...], IKSolution] = {}
        for seed in seeds:
            solution = self._grasp_center_ik_seed(
                target_center,
                seed,
                position_tolerance=position_tolerance,
            )
            if not solution.success or solution.q_arm is None:
                continue
            key = tuple(int(round(float(value) / 0.035)) for value in solution.q_arm)
            previous = unique.get(key)
            if previous is None or solution.position_error < previous.position_error:
                unique[key] = solution
        span = np.maximum(self.upper - self.lower, 1.0e-9)
        solutions = list(unique.values())
        solutions.sort(
            key=lambda item: (
                round(float(np.linalg.norm((item.q_arm - q_current) / span)), 10),
                round(float(item.position_error), 10),
                *(int(round(float(value) * 100_000.0)) for value in item.q_arm),
            )
        )
        return solutions[: max(1, int(max_candidates))]

    def whole_body_grasp_center_candidates(
        self,
        target_center: np.ndarray,
        q_arm_current: np.ndarray,
        q_torso_current: np.ndarray,
        *,
        max_candidates: int = 12,
        position_tolerance: float = GRASP_CENTER_IK_POS_TOL,
        desired_span: np.ndarray | None = None,
        span_to_constraint_rotation: np.ndarray | None = None,
        direction_tolerance: float = GRASP_WINDOW_IK_DIRECTION_TOL,
        budget_check: Callable[[], None] | None = None,
    ) -> list[WholeBodyIKSolution]:
        """Solve bounded whole-body grasp geometry over torso and arm DOFs.

        Fixed-torso arm IK is insufficient for low-support interactions: the
        target may be outside every arm-only workspace even though a stable
        coordinated torso/arm posture can reach it.  This method searches
        that robot-level workspace from deterministic current, neutral, and
        bounded random seeds.  It does not choose a task pose or a scene
        coordinate; callers provide the live target and later certify the
        resulting path against collision, support margin, and effort.

        ``desired_span`` is optional.  When present, the solver constrains the
        horizontal direction of the physical finger-link span in the frame
        selected by ``span_to_constraint_rotation`` while still solving the
        finger midpoint.  This is the generic side-acquisition capability:
        the caller derives the direction from the live object/support
        geometry, rather than supplying a task-specific wrist pose.  Keeping
        the argument optional preserves the position-only whole-body mode for
        tasks whose interaction geometry does not require a parallel-jaw
        window.

        The implementation uses bounded nonlinear least squares rather than a
        hand-authored torso vector.  Position and, when requested, jaw
        direction are the task variables; redundancy is resolved by
        deterministic candidate ranking in joint space, leaving scene-specific
        feasibility to the higher-level planner.
        """
        target_center = np.asarray(target_center, dtype=float)
        q_arm_current = np.asarray(q_arm_current, dtype=float)
        q_torso_current = np.asarray(q_torso_current, dtype=float)
        if target_center.shape != (3,) or not np.all(np.isfinite(target_center)):
            raise ValueError("target_center must be a finite 3-vector")
        if q_arm_current.shape != (7,) or not np.all(np.isfinite(q_arm_current)):
            raise ValueError("q_arm_current must be a finite 7-vector")
        if q_torso_current.shape != (4,) or not np.all(np.isfinite(q_torso_current)):
            raise ValueError("q_torso_current must be a finite 4-vector")
        tolerance = float(position_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("position_tolerance must be finite and positive")
        if not np.isfinite(float(direction_tolerance)) or float(direction_tolerance) <= 0.0:
            raise ValueError("direction_tolerance must be finite and positive")
        if span_to_constraint_rotation is not None:
            span_to_constraint_rotation = np.asarray(
                span_to_constraint_rotation,
                dtype=float,
            )
            if (
                span_to_constraint_rotation.shape != (3, 3)
                or not np.all(np.isfinite(span_to_constraint_rotation))
            ):
                raise ValueError(
                    "span_to_constraint_rotation must be a finite 3x3 matrix"
                )
        if desired_span is not None:
            desired_span = np.asarray(desired_span, dtype=float)
            if desired_span.shape != (3,) or not np.all(np.isfinite(desired_span)):
                raise ValueError("desired_span must be a finite 3-vector")
            if float(np.linalg.norm(desired_span[:2])) <= 1.0e-8:
                raise ValueError("desired_span must have a non-zero horizontal component")

        # Keep the torso limits in the same authoritative robot profile as
        # the simulator adapter.  The arm limits are already loaded into the
        # Pinocchio model during construction.
        from r1pro_data_gen.robot.robot_config import R1PRO_JOINT_LIMITS

        torso_names = tuple(f"torso_joint{index}" for index in range(1, 5))
        torso_lower = np.asarray(
            [R1PRO_JOINT_LIMITS[name][0] for name in torso_names],
            dtype=float,
        )
        torso_upper = np.asarray(
            [R1PRO_JOINT_LIMITS[name][1] for name in torso_names],
            dtype=float,
        )
        if (
            torso_lower.shape != (4,)
            or torso_upper.shape != (4,)
            or not np.all(np.isfinite(torso_lower))
            or not np.all(np.isfinite(torso_upper))
            or np.any(torso_lower >= torso_upper)
        ):
            raise ValueError("torso joint limits are invalid")
        q_arm_current = np.clip(q_arm_current, self.lower, self.upper)
        q_torso_current = np.clip(q_torso_current, torso_lower, torso_upper)
        lower = np.concatenate((torso_lower, self.lower))
        upper = np.concatenate((torso_upper, self.upper))

        # The direction constraint is sign-invariant because the two parallel
        # fingers are interchangeable.  Select the sign closest to the live
        # current branch so the solver does not introduce an unnecessary
        # 180-degree wrist turn merely because the caller chose the opposite
        # finger ordering.
        desired_direction: np.ndarray | None = None
        tilt_gate_active = False
        if desired_span is not None:
            desired_span_norm = float(np.linalg.norm(desired_span))
            desired_direction = desired_span[:2] / float(
                np.linalg.norm(desired_span[:2])
            )
            tilt_gate_active = (
                abs(float(desired_span[2])) / max(desired_span_norm, 1.0e-12)
                <= math.sin(GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD)
            )
            self.set_auxiliary_q(
                {
                    name: float(value)
                    for name, value in zip(torso_names, q_torso_current)
                }
            )
            _, _, current_span = self.finger_span_fk(q_arm_current)
            if span_to_constraint_rotation is not None:
                current_span = span_to_constraint_rotation @ current_span
            current_xy_norm = float(np.linalg.norm(current_span[:2]))
            if current_xy_norm <= 1.0e-8:
                desired_direction = None
            elif float(np.dot(current_span[:2] / current_xy_norm, desired_direction)) < 0.0:
                desired_direction = -desired_direction

        try:
            from scipy.optimize import least_squares
        except ImportError:
            # A real Isaac environment has SciPy (support-aware geometry
            # already depends on it). Third-party minimal backends should
            # fail closed instead of silently using a non-reaching posture.
            return []

        previous_auxiliary = dict(getattr(self, "_auxiliary_q", {}))
        torso_neutral = np.zeros(4, dtype=float)
        torso_neutral = np.clip(torso_neutral, torso_lower, torso_upper)
        arm_posture = np.clip(self.natural_reach_q, self.lower, self.upper)
        rng = np.random.default_rng(113)
        seeds: list[np.ndarray] = [
            np.concatenate((q_torso_current, q_arm_current)),
            np.concatenate((q_torso_current, arm_posture)),
            np.concatenate((torso_neutral, q_arm_current)),
            np.concatenate((torso_neutral, arm_posture)),
        ]
        # A small bounded multi-start set is important near joint limits and
        # self-motion branches. The seed set is deterministic for reproducible
        # evidence while remaining independent of any benchmark scene.
        for _ in range(56):
            if len(seeds) >= 60:
                break
            torso_seed = rng.uniform(torso_lower, torso_upper)
            arm_seed = rng.uniform(self.lower, self.upper)
            seeds.append(np.concatenate((torso_seed, arm_seed)))

        def center_for(configuration: np.ndarray) -> np.ndarray:
            if budget_check is not None:
                budget_check()
            torso = np.asarray(configuration[:4], dtype=float)
            arm = np.asarray(configuration[4:], dtype=float)
            self.set_auxiliary_q(
                {name: float(value) for name, value in zip(torso_names, torso)}
            )
            return np.asarray(self.grasp_center_fk(arm)[0], dtype=float)

        def span_for(configuration: np.ndarray) -> np.ndarray:
            if budget_check is not None:
                budget_check()
            torso = np.asarray(configuration[:4], dtype=float)
            arm = np.asarray(configuration[4:], dtype=float)
            self.set_auxiliary_q(
                {name: float(value) for name, value in zip(torso_names, torso)}
            )
            span = np.asarray(self.finger_span_fk(arm)[2], dtype=float)
            if span_to_constraint_rotation is not None:
                span = span_to_constraint_rotation @ span
            return span

        def direction_for(configuration: np.ndarray) -> np.ndarray:
            span = span_for(configuration)
            norm = float(np.linalg.norm(span[:2]))
            if norm <= 1.0e-8:
                return np.full(2, 1.0e3, dtype=float)
            return span[:2] / norm

        def residual(configuration: np.ndarray) -> np.ndarray:
            # Position is scaled to millimetres so least_squares does not
            # treat the radian-sized joint values as the task objective. No
            # scene/task constants enter this residual.
            values = [
                (center_for(configuration) - target_center) / 0.002,
            ]
            if desired_direction is not None and desired_span is not None:
                span = span_for(configuration)
                values.extend(
                    [
                        (direction_for(configuration) - desired_direction)
                        / GRASP_WINDOW_IK_DIRECTION_SCALE_RAD,
                        np.asarray(
                            [
                                GRASP_WINDOW_IK_VERTICAL_SPAN_WEIGHT
                                * (span[2] - float(desired_span[2]))
                                / GRASP_WINDOW_IK_VERTICAL_SPAN_SCALE_M,
                            ],
                            dtype=float,
                        ),
                        np.asarray(
                            [
                                max(
                                    0.0,
                                    math.asin(
                                        min(
                                            1.0,
                                            abs(float(span[2]))
                                            / max(
                                                float(np.linalg.norm(span)),
                                                1.0e-12,
                                            ),
                                        )
                                    )
                                    - GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD,
                                )
                                / GRASP_WINDOW_IK_DIRECTION_SCALE_RAD,
                            ],
                            dtype=float,
                        )
                        if tilt_gate_active
                        else np.zeros(1, dtype=float),
                    ]
                )
            return np.concatenate(values)

        found: dict[tuple[int, ...], WholeBodyIKSolution] = {}
        try:
            for seed in seeds:
                if budget_check is not None:
                    budget_check()
                try:
                    solved = least_squares(
                        residual,
                        np.clip(seed, lower, upper),
                        bounds=(lower, upper),
                        max_nfev=180,
                        ftol=1.0e-8,
                        xtol=1.0e-8,
                        gtol=1.0e-8,
                        x_scale="jac",
                    )
                    configuration = np.clip(np.asarray(solved.x, dtype=float), lower, upper)
                    error = float(np.linalg.norm(center_for(configuration) - target_center))
                    direction_error = 0.0
                    solved_tilt = 0.0
                    if desired_direction is not None and desired_span is not None:
                        solved_span = span_for(configuration)
                        solved_direction = direction_for(configuration)
                        if not np.all(np.isfinite(solved_direction)):
                            direction_error = math.pi
                        else:
                            direction_error = math.acos(
                                float(
                                    np.clip(
                                        abs(
                                            float(
                                                np.dot(
                                                    solved_direction,
                                                    desired_direction,
                                                )
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        span_norm = float(np.linalg.norm(solved_span))
                        solved_tilt = (
                            math.asin(
                                min(
                                    1.0,
                                    abs(float(solved_span[2]))
                                    / max(span_norm, 1.0e-12),
                                )
                            )
                            if span_norm > 1.0e-8
                            else math.pi / 2.0
                        )
                except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                    continue
                if (
                    configuration.shape != (11,)
                    or not np.all(np.isfinite(configuration))
                    or not np.isfinite(error)
                    or error > tolerance
                    or not np.isfinite(direction_error)
                    or direction_error > float(direction_tolerance)
                    or (
                        tilt_gate_active
                        and solved_tilt > GRASP_WINDOW_IK_MAX_SPAN_TILT_RAD
                    )
                ):
                    continue
                solution = WholeBodyIKSolution(
                    success=True,
                    q_arm=configuration[4:].copy(),
                    q_torso=configuration[:4].copy(),
                    position_error=error,
                    iterations=int(getattr(solved, "nfev", 0)),
                    reason=(
                        "whole-body grasp-center and jaw-span direction reached"
                        if desired_direction is not None
                        else "whole-body grasp-center position reached"
                    ),
                )
                key = tuple(int(round(float(value) / 0.04)) for value in configuration)
                previous = found.get(key)
                if previous is None or solution.position_error < previous.position_error:
                    found[key] = solution
        finally:
            self.set_auxiliary_q(previous_auxiliary)

        torso_span = np.maximum(torso_upper - torso_lower, 1.0e-6)
        arm_span = np.maximum(self.upper - self.lower, 1.0e-6)

        def score(solution: WholeBodyIKSolution) -> tuple[float, ...]:
            assert solution.q_arm is not None and solution.q_torso is not None
            continuity = float(
                np.linalg.norm((solution.q_torso - q_torso_current) / torso_span)
                + np.linalg.norm((solution.q_arm - q_arm_current) / arm_span)
            )
            relaxed = float(
                np.linalg.norm(solution.q_torso / torso_span)
                + np.linalg.norm((solution.q_arm - arm_posture) / arm_span)
            )
            return (
                round(0.55 * continuity + 0.45 * relaxed, 10),
                round(float(solution.position_error), 10),
                *(int(round(float(value) * 100_000.0)) for value in solution.q_torso),
                *(int(round(float(value) * 100_000.0)) for value in solution.q_arm),
            )

        solutions = sorted(found.values(), key=score)
        return solutions[: max(1, int(max_candidates))]

    def frame_positions(
        self,
        q_arm: np.ndarray,
        frame_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Return planning-model frame origins for online model registration."""
        frame_names = self.base_calibration_frames if frame_names is None else frame_names
        q = self._full_q(np.asarray(q_arm, dtype=float))
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.asarray(
            [np.asarray(self.data.oMf[self.model.getFrameId(name)].translation).copy() for name in frame_names],
            dtype=float,
        )

    def calibrated_base_transform(
        self,
        q_arm: np.ndarray,
        measured_world_positions: np.ndarray,
        frame_names: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Estimate ``p_world = R @ p_model + t`` from live link positions.

        The PhysX articulation root frame in the USD is not the URDF model
        frame used by Pinocchio. Registering several corresponding link
        origins recovers the complete transform online and naturally includes
        mobile-base translation, yaw, and small chassis attitude changes.
        """
        frame_names = self.base_calibration_frames if frame_names is None else frame_names
        model = self.frame_positions(q_arm, frame_names)
        world = np.asarray(measured_world_positions, dtype=float)
        if world.shape != model.shape or len(world) < 3:
            raise ValueError(f"measured_world_positions must have shape {model.shape}, got {world.shape}")
        model_center = model.mean(axis=0)
        world_center = world.mean(axis=0)
        u, _, vt = np.linalg.svd((model - model_center).T @ (world - world_center))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1] *= -1.0
            rotation = vt.T @ u.T
        translation = world_center - rotation @ model_center
        fitted = (rotation @ model.T).T + translation
        rms_error = float(np.sqrt(np.mean(np.sum((fitted - world) ** 2, axis=1))))
        return rotation, translation, rms_error

    def ee_target_from_grasp_center(
        self, target_center: np.ndarray, target_quat: np.ndarray
    ) -> np.ndarray:
        """Convert a desired finger midpoint pose to the IK link position."""
        quat = np.asarray(target_quat, dtype=float)
        quat = quat / np.linalg.norm(quat)
        rotation = pin.Quaternion(
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        ).matrix()
        return np.asarray(target_center, dtype=float) - rotation @ self._grasp_center_offset_local

    def ik_candidates(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None,
        q_current: np.ndarray,
        posture_reference: np.ndarray | None = None,
        max_candidates: int = 6,
    ) -> list[IKSolution]:
        """Return online IK branches ranked by continuity and relaxed posture."""
        q_current = np.asarray(q_current, dtype=float)
        posture = self.natural_reach_q if posture_reference is None else np.asarray(posture_reference, dtype=float)
        rng = np.random.default_rng(17)
        seeds = [q_current, posture]
        for base in (q_current, posture):
            for scale in (0.18, 0.35, 0.60, 0.85):
                seeds.append(np.clip(base + rng.normal(0.0, scale, size=7), self.lower, self.upper))
                seeds.append(np.clip(base - rng.normal(0.0, scale, size=7), self.lower, self.upper))
        solutions_by_key: dict[tuple[int, ...], IKSolution] = {}
        for seed in seeds:
            solution = self._solve_seed(target_pos, target_quat, seed)
            if not solution.success or solution.q_arm is None:
                continue
            key = tuple(int(round(float(q) / 0.035)) for q in solution.q_arm)
            previous = solutions_by_key.get(key)
            if previous is None:
                solutions_by_key[key] = solution
                continue
            previous_error = previous.position_error + previous.rotation_error
            current_error = solution.position_error + solution.rotation_error
            if current_error < previous_error:
                solutions_by_key[key] = solution
        solutions = list(solutions_by_key.values())
        span = np.maximum(self.upper - self.lower, 1e-9)
        solutions.sort(
            key=lambda item: (
                round(self.posture_score(item.q_arm, q_current, posture), 10),
                round(float(np.linalg.norm((item.q_arm - q_current) / span)), 10),
                *(int(round(float(q) * 100_000.0)) for q in item.q_arm),
            )
        )
        return solutions[: max(1, int(max_candidates))]

    def posture_score(
        self,
        q_arm: np.ndarray,
        q_current: np.ndarray,
        posture_reference: np.ndarray | None = None,
    ) -> float:
        """Dimensionless naturalness cost for one redundant-arm solution."""
        q = np.asarray(q_arm, dtype=float)
        current = np.asarray(q_current, dtype=float)
        posture = self.natural_reach_q if posture_reference is None else np.asarray(posture_reference, dtype=float)
        span = np.maximum(self.upper - self.lower, 1e-6)
        weights = np.asarray([1.0, 1.2, 0.65, 1.15, 0.45, 0.55, 0.35])
        continuity = float(np.linalg.norm(weights * (q - current) / span))
        relaxed = float(np.linalg.norm(weights * (q - posture) / span))
        margin = np.minimum(q - self.lower, self.upper - q) / span
        limit_penalty = float(np.square(np.clip((0.04 - margin) / 0.04, 0.0, None)).sum())
        wrist_penalty = float(np.linalg.norm(q[4:] / span[4:]))
        return 0.30 * continuity + 0.60 * relaxed + 2.5 * limit_penalty + 0.10 * wrist_penalty

    def arm_jacobian(self, q_arm: np.ndarray) -> np.ndarray:
        """Return the world-aligned 6x7 EE Jacobian for quality diagnostics."""
        q = self._full_q(np.asarray(q_arm, dtype=float))
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
        return np.asarray(jacobian[:, self.arm_nv_indices], dtype=float)

    def minimum_singular_value(self, q_arm: np.ndarray) -> float:
        """Smallest EE Jacobian singular value; zero denotes a singular pose."""
        singular_values = np.linalg.svd(
            self.arm_jacobian(q_arm),
            compute_uv=False,
        )
        return float(singular_values[-1]) if len(singular_values) else 0.0

    def ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None,
        q_init: np.ndarray | None = None,
    ) -> IKSolution:
        """DLS IK for the selected arm, with restarts from perturbed seeds.

        ``target_quat`` may be None to solve for position only (orientation
        free); the rotation error is then ignored.

        Among all successful seed solutions the one with the smallest joint
        motion relative to ``q_init`` is returned: the 7-DOF arm has a
        redundant DOF, and returning the first successful seed's solution
        makes small end-effector targets come out as large whole-arm swings
        (e.g. a 45 deg end-effector rotation moving every joint ~0.7 rad).
        The min-motion solution concentrates the movement in the wrist --
        natural-looking -- and keeps successive per-step IK chains continuous.
        """
        q_ref = np.zeros(7) if q_init is None else np.asarray(q_init, dtype=float).copy()
        best: IKSolution | None = None
        rng = np.random.default_rng(0)
        q0 = np.zeros(7) if q_init is None else q_ref
        # The neutral-home seed (q=0) is a singular configuration for the
        # hanging arm; the reference-project forward-reach pose is a robust
        # non-singular seed for position IK toward the table.
        shoulder_lateral = 1.48355 if self.side == "left" else -1.48355
        fwd = np.array([-1.5708, shoulder_lateral, 0.0, -0.6981, 0.0, 0.0, 0.0])
        seeds = [q0, fwd]
        # DLS is numerically sensitive near the table surface: whether a
        # position-only goal converges depends on the seed (the fwd pose alone
        # misses some reachable goals). Perturb BOTH bases so a random walk from
        # the forward-reach pose has a fair chance, and keep the best-scoring
        # partial solution.
        for base in (q0, fwd):
            for _ in range(IK_RESTARTS):
                seeds.append(
                    np.clip(base + rng.uniform(-0.8, 0.8, size=7), self.lower, self.upper)
                )
        solutions: list[IKSolution] = []
        for seed in seeds:
            sol = self._solve_seed(target_pos, target_quat, seed)
            if sol.success:
                solutions.append(sol)
                continue
            score = sol.position_error + (sol.rotation_error if target_quat is not None else 0.0)
            if best is None or score < (best.position_error + (best.rotation_error if target_quat is not None else 0.0)):
                best = sol
        if solutions:
            return pick_min_motion_solution(solutions, q_ref)
        return best  # type: ignore[return-value]



    def _ik_once(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None,
        q_init: np.ndarray,
        pos_tol: float = POS_TOL,
        rot_tol: float = ROT_TOL,
    ) -> IKSolution:
        """Single DLS run from one seed."""
        q_arm = q_init.copy()
        prev = q_arm.copy()
        for it in range(MAX_ITERS):
            pos, quat = self.fk(q_arm)
            pos_err = target_pos - pos
            if target_quat is None:
                rot_err = np.zeros(3)
            else:
                rot_err = _quat_error_rotation_vector(target_quat, quat)
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                return IKSolution(
                    success=True, q_arm=q_arm,
                    position_error=float(np.linalg.norm(pos_err)),
                    rotation_error=float(np.linalg.norm(rot_err)),
                    iterations=it,
                )
            # Jacobian of the EE frame w.r.t. the 7 arm joints.
            q = self._full_q(q_arm)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            jac = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            j_arm = jac[:, self.arm_nv_indices]  # 6x7 (nv indices)
            jj = j_arm @ j_arm.T
            dq = j_arm.T @ np.linalg.solve(jj + DLS_DAMPING**2 * np.eye(6), err)
            dq = np.clip(dq, -MAX_DELTA, MAX_DELTA)
            q_arm = np.clip(q_arm + dq, self.lower, self.upper)
            if np.allclose(q_arm, prev, atol=1e-9):
                break
            prev = q_arm.copy()
        pos, quat = self.fk(q_arm)
        if target_quat is None:
            rot_err = 0.0
        else:
            rot_err = float(np.linalg.norm(_quat_error_rotation_vector(target_quat, quat)))
        return IKSolution(
            success=False, q_arm=q_arm,
            position_error=float(np.linalg.norm(target_pos - pos)),
            rotation_error=rot_err,
            iterations=MAX_ITERS,
            reason="max iterations or no progress",
        )

    def _qp_context(self):
        """Lazily build the reduced-model Pink IK context (None = unavailable).

        The full URDF has 31 configuration dimensions; solving the QP on it
        lets the torso silently absorb the motion (FrameTask error converges
        while the arm FK error stays large).  Locking every non-arm joint
        yields the exact 7-DOF chain this class already exposes.
        """
        context = getattr(self, "_pink_ctx", None)
        if context is not None:
            return context
        if getattr(self, "_qp_disabled", False):
            return None
        try:
            import pink
            from pink.limits import ConfigurationLimit
            from pink.tasks import FrameTask, PostureTask

            keep = set(self.arm_joint_ids)
            lock_ids = [jid for jid in range(1, self.model.njoints) if jid not in keep]
            locked_reference = np.zeros(self.model.nq)
            for name, value in getattr(self, "_auxiliary_q", {}).items():
                if not self.model.existJointName(name):
                    continue
                joint_id = self.model.getJointId(name)
                locked_reference[int(self.model.joints[joint_id].idx_q)] = float(value)
            reduced = pin.buildReducedModel(self.model, lock_ids, locked_reference)
            # Inset finite bounds so the soft ConfigurationLimit damping still
            # produces solutions strictly inside the physics-scene limits.
            finite_lower = np.isfinite(reduced.lowerPositionLimit)
            finite_upper = np.isfinite(reduced.upperPositionLimit)
            reduced.lowerPositionLimit[finite_lower] += QP_IK_LIMIT_GUARD_RAD
            reduced.upperPositionLimit[finite_upper] -= QP_IK_LIMIT_GUARD_RAD
            context = type(
                "_QPIKContext",
                (),
                {
                    "model": reduced,
                    "data": reduced.createData(),
                    "frame_task": FrameTask(
                        self.ee_frame,
                        position_cost=1.0,
                        orientation_cost=1.0,
                        lm_damping=QP_IK_LM_DAMPING,
                    ),
                    "posture_task": PostureTask(cost=QP_IK_POSTURE_COST),
                    "limits": [ConfigurationLimit(reduced)],
                },
            )()
            context.ee_frame_id = context.model.getFrameId(self.ee_frame)
        except Exception:
            # pink missing, quadprog unavailable, model reduction failure --
            # permanently degrade to the DLS path.
            self._qp_disabled = True
            return None
        self._pink_ctx = context
        return context

    def ik_qp(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None,
        q_init: np.ndarray,
        pos_tol: float = POS_TOL,
        rot_tol: float = ROT_TOL,
    ) -> IKSolution:
        """Single-seed bounded QP IK (Pink).  Same contract as ``_ik_once``.

        Failure reason deliberately contains "max iterations or no progress"
        so the failure-feedback classification keeps working unchanged.
        """
        import pink

        ctx = self._qp_context()
        if ctx is None:
            raise RuntimeError("qp ik unavailable")
        target_pos = np.asarray(target_pos, dtype=float)
        # Clip into the inset reduced-model bounds so ConfigurationLimit does
        # not warn about out-of-limit starting points.
        guard = QP_IK_LIMIT_GUARD_RAD
        q_arm = np.clip(np.asarray(q_init, dtype=float), self.lower + guard, self.upper - guard)
        configuration = pink.Configuration(ctx.model, ctx.data, q_arm, copy_data=False)
        ctx.frame_task.position_cost = 1.0
        ctx.frame_task.orientation_cost = 0.0 if target_quat is None else 1.0
        if target_quat is None:
            rotation = np.eye(3)
        else:
            quat = np.asarray(target_quat, dtype=float)
            quat = quat / np.linalg.norm(quat)
            # pin.Quaternion's scalar ctor follows the Eigen (w, x, y, z)
            # convention; verified against a rotation-matrix construction.
            rotation = pin.Quaternion(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])).matrix()
        ctx.frame_task.set_target(pin.SE3(rotation, target_pos))
        ctx.posture_task.set_target(q_arm)
        last_pos_err = float("inf")
        last_rot_err = float("inf")
        for iteration in range(QP_IK_MAX_ITERS):
            configuration.update()
            dv = pink.solve_ik(
                configuration,
                [ctx.frame_task, ctx.posture_task],
                QP_IK_DT,
                solver="quadprog",
                limits=ctx.limits,
                safety_break=False,
            )
            if not np.all(np.isfinite(dv)):
                raise RuntimeError("qp produced a non-finite step")
            configuration.integrate_inplace(dv, QP_IK_DT)
            # pink.Configuration does not expose frame placements: refresh the
            # reduced-model data explicitly for the truth check.
            pin.forwardKinematics(ctx.model, ctx.data, configuration.q)
            pin.updateFramePlacements(ctx.model, ctx.data)
            placement = ctx.data.oMf[ctx.ee_frame_id]
            pos_err_vec = target_pos - np.asarray(placement.translation)
            last_pos_err = float(np.linalg.norm(pos_err_vec))
            if target_quat is None:
                last_rot_err = 0.0
            else:
                coeffs = np.array(pin.Quaternion(placement.rotation).coeffs(), dtype=float)
                quat = np.roll(coeffs, 1)  # -> (w, x, y, z)
                last_rot_err = float(
                    np.linalg.norm(_quat_error_rotation_vector(np.asarray(target_quat, dtype=float), quat))
                )
            if last_pos_err < pos_tol and last_rot_err < rot_tol:
                return IKSolution(
                    success=True,
                    q_arm=np.clip(np.asarray(configuration.q, dtype=float), self.lower, self.upper),
                    position_error=last_pos_err,
                    rotation_error=last_rot_err,
                    iterations=iteration,
                )
        return IKSolution(
            success=False,
            q_arm=np.clip(np.asarray(configuration.q, dtype=float), self.lower, self.upper),
            position_error=last_pos_err,
            rotation_error=last_rot_err,
            iterations=QP_IK_MAX_ITERS,
            reason="qp max iterations or no progress",
        )

    def _solve_seed(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None,
        seed: np.ndarray,
        pos_tol: float = POS_TOL,
        rot_tol: float = ROT_TOL,
    ) -> IKSolution:
        """Solve one seed: bounded QP first, DLS fallback per module policy."""
        if not getattr(self, "_qp_disabled", False):
            try:
                solution = self.ik_qp(target_pos, target_quat, seed, pos_tol, rot_tol)
                if solution.success or not QP_FALLBACK_TO_DLS:
                    return solution
            except Exception:
                pass  # quadprog degeneracy / pink absence: fall through to DLS
        return self._ik_once(target_pos, target_quat, seed, pos_tol, rot_tol)

    def plan_segment_steps(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        vel_limit: np.ndarray,
        speed_scale: float = 0.3,
        dt: float = 1.0 / 60.0,
        min_steps: int = 30,
        ) -> int:
        """Steps needed to move q_from -> q_to within joint velocity limits.

        The segment duration is set by the slowest joint: max |dq| / (limit *
        speed_scale). speed_scale keeps the motion well under the real joint
        limits (conservative, smooth).
        """
        displacement = np.abs(q_to - q_from)
        allowed = np.maximum(np.abs(vel_limit) * speed_scale, 1e-6)
        required = float(np.max(displacement / allowed))
        return max(min_steps, int(np.ceil(required / dt)) + 1)

    def set_auxiliary_q(self, values: dict[str, float] | None) -> None:
        """Lock non-arm joints (for example the live torso) during FK/IK."""
        updated = dict(values or {})
        # The bounded QP solver uses a reduced model in which every
        # non-arm joint is locked.  Its lock values are baked into that model;
        # retaining a context created for the previous torso posture makes IK
        # silently solve the wrong robot configuration during whole-body
        # transitions.  Invalidate only the cached context when the locked
        # joints actually change.  The fallback DLS path already reads
        # ``_auxiliary_q`` through ``_full_q`` on every iteration.
        if updated != getattr(self, "_auxiliary_q", {}):
            self._pink_ctx = None
        self._auxiliary_q = updated

    def center_of_mass(self, q_arm: np.ndarray) -> np.ndarray:
        """Return the robot COM in the URDF model frame for a live torso/arm q."""
        q = self._full_q(np.asarray(q_arm, dtype=float))
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError("q_arm and auxiliary joints must form a finite model configuration")
        pin.centerOfMass(self.model, self.data, q)
        return np.asarray(self.data.com[0], dtype=float).copy()

    def torso_gravity_effort(self, q_arm: np.ndarray) -> np.ndarray:
        """Return the live Pinocchio gravity effort for the four torso joints."""
        q = self._full_q(np.asarray(q_arm, dtype=float))
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError("q_arm and auxiliary joints must form a finite model configuration")
        gravity = np.asarray(pin.computeGeneralizedGravity(self.model, self.data, q), dtype=float)
        return gravity[self.torso_nv_indices].copy()

    def _full_q(self, q_arm: np.ndarray) -> np.ndarray:
        # The URDF has a fixed base (no free-floating joint): the 3 extra nq
        # dimensions are fixed joints, so only the arm entries need setting.
        # The arm joints are written at their q indices (not joint ids).
        q = np.zeros(self.model.nq)
        q[self.arm_q_indices] = q_arm
        for name, value in getattr(self, "_auxiliary_q", {}).items():
            if not self.model.existJointName(name):
                continue
            joint_id = self.model.getJointId(name)
            q[int(self.model.joints[joint_id].idx_q)] = float(value)
        return q




def pick_min_motion_solution(solutions: list[IKSolution], q_ref: np.ndarray) -> IKSolution:
    """Among successful solutions, return the one with the smallest joint
    motion relative to ``q_ref`` (ties keep the first)."""
    return min(solutions, key=lambda s: float(np.abs(s.q_arm - q_ref).sum()))
