"""Online IK and grasp-frame contracts for the R1Pro asset."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from r1pro_data_gen.robot.kinematics import BASE_CALIBRATION_FRAMES, R1ProKinematics
from tests.support import PROJECT_ROOT


URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"
GRASP_QUAT = np.array([0.70710678, 0.0, -0.70710678, 0.0])


def test_grasp_center_target_round_trip_uses_asset_finger_geometry() -> None:
    kin = R1ProKinematics(str(URDF))
    target_center = np.array([0.55, 0.15, 1.385])
    ee_target = kin.ee_target_from_grasp_center(target_center, GRASP_QUAT)
    solutions = kin.ik_candidates(
        ee_target, GRASP_QUAT, np.zeros(7), max_candidates=3
    )

    assert solutions
    center, _ = kin.grasp_center_fk(solutions[0].q_arm)
    assert np.linalg.norm(center - target_center) < 0.015
    assert np.linalg.norm(kin.grasp_center_offset_local) > 0.03


def test_position_only_grasp_center_ik_keeps_the_physical_anchor_invariant() -> None:
    """Orientation relaxation must still solve the finger midpoint, not the wrist."""
    kin = R1ProKinematics(str(URDF))
    target_center, _ = kin.grasp_center_fk(
        np.array([0.10, 0.40, 0.20, -0.80, 0.40, -0.50, -0.80])
    )

    solutions = kin.ik_grasp_center_candidates(
        target_center,
        np.zeros(7),
        max_candidates=3,
    )

    assert solutions
    assert min(item.position_error for item in solutions) < 0.008
    for item in solutions:
        center, _ = kin.grasp_center_fk(item.q_arm)
        assert np.linalg.norm(center - target_center) < 0.012


def test_grasp_window_ik_solves_center_and_jaw_direction_without_full_pose() -> None:
    """A low-workspace pinch need not force an arbitrary 6-D wrist pose."""
    kin = R1ProKinematics(str(URDF))
    kin.set_auxiliary_q(
        {
            "torso_joint1": 1.44,
            "torso_joint2": -1.77,
            "torso_joint3": 1.56,
            "torso_joint4": 1.28,
        }
    )
    q_reference = np.array(
        [-0.052, 0.525, 1.101, -0.305, 1.779, -0.416, -0.043]
    )
    target_center = kin.grasp_center_fk(q_reference)[0] + np.array(
        [0.012, -0.009, 0.0]
    )
    _, _, current_span = kin.finger_span_fk(q_reference)
    desired_span = np.array([-current_span[1], current_span[0], 0.0])

    solutions = kin.ik_grasp_window_candidates(
        target_center,
        desired_span,
        q_reference,
        max_candidates=3,
    )

    assert solutions
    solution = solutions[0]
    center, _ = kin.grasp_center_fk(solution.q_arm)
    _, _, span = kin.finger_span_fk(solution.q_arm)
    span_xy = span[:2] / np.linalg.norm(span[:2])
    desired_xy = desired_span[:2] / np.linalg.norm(desired_span[:2])
    assert np.linalg.norm(center - target_center) < 0.008
    angle = abs(
        np.arctan2(
            np.cross(span_xy, desired_xy),
            np.dot(span_xy, desired_xy),
        )
    )
    # Finger 1/2 can exchange sides; a jaw line is sign-invariant.
    assert min(angle, abs(np.pi - angle)) < 0.20


def test_open_jaw_window_ik_uses_physical_center_and_horizontal_span() -> None:
    """The open-jaw center is physical while vertical overlap stays a box gate."""
    kin = R1ProKinematics(str(URDF))
    kin.set_auxiliary_q(
        {
            "torso_joint1": 1.40212,
            "torso_joint2": -1.56517,
            "torso_joint3": 1.41471,
            "torso_joint4": 0.92091,
        }
    )
    q_reference = np.array(
        [0.16911, 0.55344, 0.87207, -0.50948, 0.10958, 0.49587, 0.07601]
    )
    opening_offsets = (np.array([0.0, 0.05, 0.0]), np.array([0.0, -0.05, 0.0]))
    target_center = kin.finger_geometry_fk(q_reference, opening_offsets)[4]
    span = kin.finger_span_fk(q_reference, opening_offsets)[2]
    desired_span = np.array([-span[1], span[0], 0.0])

    solutions = kin.ik_grasp_window_candidates(
        target_center,
        desired_span,
        q_reference,
        max_candidates=3,
        opening_offsets=opening_offsets,
    )

    assert solutions
    solved_span = kin.finger_span_fk(solutions[0].q_arm, opening_offsets)[2]
    solved_xy = solved_span[:2] / np.linalg.norm(solved_span[:2])
    desired_xy = desired_span[:2] / np.linalg.norm(desired_span[:2])
    angle = abs(
        np.arctan2(
            np.cross(solved_xy, desired_xy),
            np.dot(solved_xy, desired_xy),
        )
    )
    assert min(angle, abs(np.pi - angle)) < 0.20


def test_grasp_window_ik_can_constrain_world_direction_through_tilted_registration() -> None:
    """World-horizontal jaw direction must survive a non-yaw model registration."""
    kin = R1ProKinematics(str(URDF))
    kin.set_auxiliary_q(
        {
            "torso_joint1": 1.40212,
            "torso_joint2": -1.56517,
            "torso_joint3": 1.41471,
            "torso_joint4": 0.92091,
        }
    )
    q_reference = np.array(
        [0.16911, 0.55344, 0.87207, -0.50948, 0.10958, 0.49587, 0.07601]
    )
    target_center = kin.finger_geometry_fk(q_reference)[4]
    span_model = kin.finger_span_fk(q_reference)[2]
    registration = Rotation.from_euler("xyz", [0.38, -0.27, 0.21]).as_matrix()
    desired_span_world = registration @ span_model

    solutions = kin.ik_grasp_window_candidates(
        target_center,
        desired_span_world,
        q_reference,
        max_candidates=3,
        span_to_constraint_rotation=registration,
    )

    assert solutions
    solved_span_world = registration @ kin.finger_span_fk(solutions[0].q_arm)[2]
    solved_xy = solved_span_world[:2] / np.linalg.norm(solved_span_world[:2])
    desired_xy = desired_span_world[:2] / np.linalg.norm(desired_span_world[:2])
    angle = abs(
        np.arctan2(
            np.cross(solved_xy, desired_xy),
            np.dot(solved_xy, desired_xy),
        )
    )
    assert min(angle, abs(np.pi - angle)) < 0.20


def test_online_ik_ranking_avoids_joint_limits_for_tabletop_reach() -> None:
    kin = R1ProKinematics(str(URDF))
    target_center = np.array([0.55, 0.15, 1.385])
    ee_target = kin.ee_target_from_grasp_center(target_center, GRASP_QUAT)
    solutions = kin.ik_candidates(
        ee_target, GRASP_QUAT, np.zeros(7), max_candidates=4
    )

    assert len(solutions) >= 2
    q = solutions[0].q_arm
    normalized_margin = np.minimum(q - kin.lower, kin.upper - q) / (kin.upper - kin.lower)
    assert normalized_margin.min() > 0.02
    assert abs(q[3]) < 1.6  # bent elbow, not the asset's extreme folded branch


def test_ik_candidates_are_stable_for_identical_input() -> None:
    kin = R1ProKinematics(str(URDF))
    target_center = np.array([0.55, 0.15, 1.385])
    target = kin.ee_target_from_grasp_center(target_center, GRASP_QUAT)

    first = kin.ik_candidates(target, GRASP_QUAT, np.zeros(7), max_candidates=4)
    second = kin.ik_candidates(target, GRASP_QUAT, np.zeros(7), max_candidates=4)

    assert len(first) == len(second)
    assert np.allclose(
        np.asarray([item.q_arm for item in first]),
        np.asarray([item.q_arm for item in second]),
    )
    keys = {
        tuple(int(round(float(q) / 0.035)) for q in item.q_arm)
        for item in first
    }
    assert len(keys) == len(first)


def test_manipulation_ready_pose_is_galaxea_untucked_rest() -> None:
    from r1pro_data_gen.robot.robot_config import R1PRO_ARM_READY_Q_BY_SIDE
    from r1pro_data_gen.robot.robot_config import R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE

    kin = R1ProKinematics(str(URDF), side="left")
    q = np.asarray(R1PRO_ARM_READY_Q_BY_SIDE["left"], dtype=float)
    pos, quat = kin.fk(q)
    # Hands in front of the chest, not flung out behind the shoulder.
    # J2 is below a full 90 deg abduction so the EE sits near shoulder height.
    assert pos[0] > 0.20
    assert 0.35 < pos[1] < 0.75
    assert 1.20 < pos[2] < 1.33
    default = np.asarray(R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE["left"], dtype=float)
    assert abs(float(np.dot(quat, default))) > 0.95


def test_calibrated_base_transform_recovers_known_rigid_motion() -> None:
    kin = R1ProKinematics(str(URDF))
    q = np.array([-0.4, 0.15, -0.2, -0.9, 0.1, 0.25, -0.1])
    model = kin.frame_positions(q, BASE_CALIBRATION_FRAMES)
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.array([1.3, -0.8, 0.12])
    measured = (rotation @ model.T).T + translation

    estimated_r, estimated_t, rms = kin.calibrated_base_transform(q, measured)

    assert np.allclose(estimated_r, rotation, atol=1e-8)
    assert np.allclose(estimated_t, translation, atol=1e-8)
    assert rms < 1e-9


def test_left_and_right_models_use_mirrored_chains_and_limits() -> None:
    left = R1ProKinematics(str(URDF), side="left")
    right = R1ProKinematics(str(URDF), side="right")

    left_home, _ = left.fk(np.zeros(7))
    right_home, _ = right.fk(np.zeros(7))
    assert left.arm_joints[0] == "left_arm_joint1"
    assert right.arm_joints[0] == "right_arm_joint1"
    assert np.allclose(left_home[[0, 2]], right_home[[0, 2]], atol=1e-6)
    assert np.isclose(left_home[1], -right_home[1], atol=1e-6)
    assert left.lower[1] > right.lower[1]
    assert right.upper[1] < left.upper[1]


def test_right_arm_online_ik_round_trip() -> None:
    kin = R1ProKinematics(str(URDF), side="right")
    q_target = np.array([-0.75, -0.55, 0.08, -0.48, 0.10, 0.0, 0.0])
    target_pos, target_quat = kin.fk(q_target)

    solutions = kin.ik_candidates(target_pos, target_quat, np.zeros(7), max_candidates=3)

    assert solutions
    reached, _ = kin.fk(solutions[0].q_arm)
    assert np.linalg.norm(reached - target_pos) < 0.015


# ---- Pink-QP bounded IK (component A of the planner upgrade) ----

def test_qp_rescues_boundary_target_where_dls_stalls() -> None:
    """A hard boundary target: DLS stalls far away, the QP converges quickly."""
    kin = R1ProKinematics(str(URDF))
    q_true = np.array([-0.8503, 2.8008, 1.2991, -1.2736, -0.9417, 0.7824, -1.5543])
    target_pos, target_quat = kin.fk(q_true)

    dls = kin._ik_once(target_pos, target_quat, kin.natural_reach_q.copy())
    qp = kin.ik_qp(target_pos, target_quat, kin.natural_reach_q.copy())

    assert not dls.success and dls.position_error > 0.3
    assert qp.success
    assert qp.position_error < 0.01 and qp.rotation_error < ROT_TOL_DEFAULT
    assert np.all(qp.q_arm >= kin.lower - 1e-9) and np.all(qp.q_arm <= kin.upper + 1e-9)
    # fk truth check: the returned configuration really reaches the pose.
    reached_pos, _ = kin.fk(qp.q_arm)
    assert np.linalg.norm(reached_pos - target_pos) < 0.01


def test_qp_solutions_respect_joint_limits() -> None:
    kin = R1ProKinematics(str(URDF))
    for q_true in (
        np.array([-0.8503, 2.8008, 1.2991, -1.2736, -0.9417, 0.7824, -1.5543]),
        np.array([-1.5144, 2.6347, 0.6584, -0.1917, -1.925, 0.0862, 0.0244]),
    ):
        pos, quat = kin.fk(q_true)
        solution = kin.ik_qp(pos, quat, kin.natural_reach_q.copy(), pos_tol=0.02)
        if solution.success:
            assert np.all(solution.q_arm >= kin.lower - 1e-9)
            assert np.all(solution.q_arm <= kin.upper + 1e-9)


def test_unreachable_target_failure_semantics_keep_feedback_marker() -> None:
    kin = R1ProKinematics(str(URDF))
    # Far outside the workspace: no solver may claim success.
    target_pos = np.array([1.20, 0.60, 1.90])
    target_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0])
    solution = kin._solve_seed(
        target_pos, target_quat, kin.natural_reach_q.copy()
    )
    assert not solution.success
    assert "max iterations" in solution.reason
    candidates = kin.ik_candidates(target_pos, target_quat, np.zeros(7))
    assert candidates == []


def test_qp_exception_falls_back_to_dls_per_seed(monkeypatch) -> None:
    import sys

    kin = R1ProKinematics(str(URDF))
    assert kin._qp_context() is not None  # pink is installed in this env

    def explode(*args, **kwargs):
        raise RuntimeError("quadprog exploded")

    monkeypatch.setattr(sys.modules["pink"], "solve_ik", explode)
    q_target = np.array([-0.45, -0.09, -0.06, -0.94, -0.06, -0.18, -0.05])
    pos, quat = kin.fk(q_target)
    # _solve_seed must swallow the exception and return the DLS verdict.
    solution = kin._solve_seed(pos, quat, q_target.copy())
    assert solution.success  # DLS solves this easy target


def test_pink_missing_degrades_to_pure_dls() -> None:
    kin = R1ProKinematics(str(URDF))
    assert kin._qp_context() is not None
    # Simulate a pink-less deployment: the context builder returns None.
    kin._qp_disabled = True
    kin._pink_ctx = None

    def fail_if_called(*args, **kwargs):
        raise AssertionError("qp path must be disabled")

    def broken_context():
        return None

    kin._qp_context = broken_context  # type: ignore[method-assign]
    q_target = np.array([-0.45, -0.09, -0.06, -0.94, -0.06, -0.18, -0.05])
    pos, quat = kin.fk(q_target)
    solution = kin._solve_seed(pos, quat, q_target.copy())
    assert solution.success


def test_right_arm_qp_round_trip() -> None:
    kin = R1ProKinematics(str(URDF), side="right")
    q_true = np.array([-0.75, 0.55, -0.08, -0.48, -0.10, 0.0, 0.0])
    pos, quat = kin.fk(q_true)
    solution = kin.ik_qp(pos, quat, kin.natural_reach_q.copy(), pos_tol=0.02)
    if solution.success:
        reached, _ = kin.fk(solution.q_arm)
        assert np.linalg.norm(reached - pos) < 0.02


ROT_TOL_DEFAULT = __import__(
    "r1pro_data_gen.robot.kinematics", fromlist=["ROT_TOL"]
).ROT_TOL
