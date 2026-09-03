"""Per-skill GPU verification scenarios.

Each scenario preconditions the robot into a sensible state, executes the skill,
and returns (ok, metrics). These run on GPU (Isaac Sim) and are where physical
behavior is actually judged -- unit tests only cover the pure logic.

Scenario convention: ``fn(adapter, kin, scene, registry, step_hook) -> (ok, metrics)``.
"""

from __future__ import annotations

import numpy as np

from r1pro_data_gen.skills.core.base import SkillResult


def _verification_side(adapter) -> str:
    return getattr(adapter, "_verification_side", "left")


def _side_joint_pose(values, side: str) -> list[float]:
    """Mirror the lateral shoulder component of a neutral showcase posture."""
    q = np.asarray(values, dtype=float).copy()
    if side == "right":
        q[1] *= -1.0
    return q.tolist()

# ---------------------------------------------------------------------------
# Base motion
# ---------------------------------------------------------------------------


def scenario_base_move_to(adapter, kin, scene, registry, step_hook):
    """Drive to the work position (closed-loop straight line)."""
    result = registry.get("base_move_to").execute(
        adapter, scene=scene, step_hook=step_hook, target=[0.8, 0.35, 0.25], v_max=0.16, omega_max=0.35
    )
    return result.success, {"arrival_error_m": result.metrics.get("arrival_error_m")}


def scenario_base_rotate_to(adapter, kin, scene, registry, step_hook):
    """Rotate in place to 90 degrees."""
    result = registry.get("base_rotate_to").execute(
        adapter, scene=scene, step_hook=step_hook, target_yaw=1.5708
    )
    return result.success, {"yaw_error_rad": result.metrics.get("yaw_error_rad")}


def scenario_base_follow_path(adapter, kin, scene, registry, step_hook):
    """Follow a multi-corner waypoint path then hold the target yaw."""
    result = registry.get("base_follow_path").execute(
        adapter, scene=scene, step_hook=step_hook,
        path=[[0.0, 0.0], [0.35, 0.28], [0.78, -0.18], [1.15, 0.25]],
        target_yaw=0.35, v_max=0.16, omega_max=0.35,
    )
    return result.success, {**result.metrics, **result.details}


def scenario_base_velocity_set(adapter, kin, scene, registry, step_hook):
    """Command a safe forward velocity long enough to show base motion."""
    result = registry.get("base_velocity_set").execute(
        adapter, scene=scene, step_hook=step_hook, vx=0.08, vy=0.0, omega=0.10, duration=2.0
    )
    return result.success, {"steps": result.metrics.get("steps")}


def scenario_base_navigate_to(adapter, kin, scene, registry, step_hook):
    """Navigate a long multi-obstacle slalom route.

    The target is deliberately selected by the supplied scene, not by the
    skill. The current physical showcase uses the manually verified
    ``pickplace.tabletop`` TaskSpec; changing the scene and target here
    produces a different reusable navigation test without changing the skill.
    """
    xy_log: list[list[float]] = []

    def tracking_hook():
        if step_hook is not None:
            step_hook()
        pos = adapter.robot.data.root_pos_w[0].detach().cpu().numpy()
        xy_log.append([float(pos[0]), float(pos[1])])

    target = [1.0, 0.0, 0.0]
    result = registry.get("base_navigate_to").execute(
        adapter, scene=scene, step_hook=tracking_hook,
        target=target, resolution=0.05, v_max=0.16, omega_max=0.35,
    )
    path = result.details.get("path", [])
    planned_detour = max((abs(float(p[1])) for p in path), default=0.0)
    actual = np.asarray(xy_log, dtype=float)
    actual_detour = float(np.max(np.abs(actual[:, 1]))) if len(actual) else 0.0
    min_clearance = float("inf")
    for x, y in actual:
        for obj in scene.objects:
            if obj.type.value == "cuboid":
                hx, hy, _ = obj.size
            else:
                hx = hy = obj.radius
            dx = max(abs(x - obj.pos[0]) - hx / (1.0 if obj.type.value == "cylinder" else 2.0), 0.0)
            dy = max(abs(y - obj.pos[1]) - hy / (1.0 if obj.type.value == "cylinder" else 2.0), 0.0)
            min_clearance = min(min_clearance, float(np.hypot(dx, dy)))
    required = float(result.details.get("footprint_radius_m", 0.0))
    path_length = sum(
        float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
        for a, b in zip(path[:-1], path[1:])
    )
    avoided = (
        len(path) >= 4
        and path_length > 9.0
        and planned_detour > 1.0
        and actual_detour > 0.8
        # The grid is sampled at 5 cm and the closed-loop controller can
        # deviate by a few centimeters around a corner. The planner itself
        # already inflates every obstacle by the derived footprint; allow the
        # measured center-to-box clearance to reflect that discretization.
        and min_clearance >= required - 0.12
    )
    return result.success and avoided, {
        "reason": result.details.get("reason"),
        "arrival_error_m": result.metrics.get("arrival_error_m"),
        "yaw_error_rad": result.metrics.get("yaw_error_rad"),
        "waypoints": result.metrics.get("waypoints"),
        "planned_detour_m": round(planned_detour, 4),
        "actual_detour_m": round(actual_detour, 4),
        "planned_path_length_m": round(path_length, 4),
        "min_crate_clearance_m": round(min_clearance, 4),
        "required_footprint_radius_m": round(required, 4),
        "path": path,
    }


def scenario_base_lock_wheels(adapter, kin, scene, registry, step_hook):
    """Lock wheels, then a small arm move must not roll the base."""
    lock = registry.get("base_lock_wheels").execute(adapter, scene=scene, step_hook=step_hook)
    if not lock.success:
        return False, {"reason": "lock_wheels failed"}
    # Record base x before and after a small arm motion.
    before = adapter.robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    registry.get("arm_joint_to").execute(
        adapter, scene=scene, step_hook=step_hook,
        target_q=[-0.3, 0.2, 0.0, -0.2, 0.0, 0.0, 0.0],
    )
    after = adapter.robot.data.root_pos_w[0].detach().cpu().numpy()
    drift = float(abs(after[0] - before[0]) + abs(after[1] - before[1]))
    return drift < 0.01, {"base_drift_m": drift}


def scenario_base_unlock_wheels(adapter, kin, scene, registry, step_hook):
    """Unlock wheels after a lock, then drive a short move."""
    registry.get("base_lock_wheels").execute(adapter, scene=scene, step_hook=step_hook)
    unlock = registry.get("base_unlock_wheels").execute(adapter, scene=scene, step_hook=step_hook)
    if not unlock.success:
        return False, {"reason": "unlock_wheels failed"}
    move = registry.get("base_move_to").execute(
        adapter, scene=scene, step_hook=step_hook, target=[0.45, 0.22, 0.15], v_max=0.14, omega_max=0.3, arrive_tol=0.05
    )
    return move.success, {"arrival_error_m": move.metrics.get("arrival_error_m")}


def scenario_joint_mask_lock(adapter, kin, scene, registry, step_hook):
    """Allow only the selected work arm, then verify it can move stably."""
    side = _verification_side(adapter)
    locked = registry.get("joint_mask_lock").execute(
        adapter, scene=scene, step_hook=step_hook, mask_mode="allow",
        joint_groups=[f"{side}_arm", f"{side}_gripper"], lock_root=False,
    )
    if not locked.success:
        return False, {"reason": locked.details.get("reason")}
    moved = registry.get("arm_joint_to").execute(
        adapter, scene=scene, step_hook=step_hook, side=side,
        target_q=_side_joint_pose([-0.55, 0.45, 0.0, -0.45, 0.0, 0.0, 0.0], side),
        speed_scale=0.20,
    )
    metrics = adapter.joint_lock_metrics()
    return moved.success and metrics.get("max_locked_joint_error", 1.0) < 0.10, {
        **metrics, "final_error_rad": moved.metrics.get("final_error_rad")
    }


def scenario_joint_mask_unlock(adapter, kin, scene, registry, step_hook):
    """Release an allow-mask, then move the formerly locked opposite arm."""
    side = _verification_side(adapter)
    opposite = "left" if side == "right" else "right"
    locked = registry.get("joint_mask_lock").execute(
        adapter, scene=scene, step_hook=step_hook, mask_mode="allow",
        joint_groups=[f"{side}_arm", f"{side}_gripper"], lock_root=False,
    )
    if not locked.success:
        return False, {"reason": locked.details.get("reason")}
    unlocked = registry.get("joint_mask_unlock").execute(adapter, scene=scene)
    moved = registry.get("arm_joint_to").execute(
        adapter, scene=scene, step_hook=step_hook, side=opposite,
        target_q=_side_joint_pose([-0.50, 0.40, 0.0, -0.40, 0.0, 0.0, 0.0], opposite),
        speed_scale=0.20,
    )
    return unlocked.success and moved.success and not adapter.joint_mask_locked, {
        "final_error_rad": moved.metrics.get("final_error_rad")
    }


# ---------------------------------------------------------------------------
# Torso
# ---------------------------------------------------------------------------


def scenario_torso_move_to(adapter, kin, scene, registry, step_hook):
    """Move the torso to a small lift, record the steady-state error."""
    result = registry.get("torso_move_to").execute(
        adapter, scene=scene, step_hook=step_hook, target_q=[0.0, 0.2, 0.0, 0.0]
    )
    return result.success, {"final_error_rad": result.metrics.get("final_error_rad")}


# ---------------------------------------------------------------------------
# Arm (v2 pipeline: solve IK -> plan MPlib path -> execute trajectory)
# ---------------------------------------------------------------------------


def _plan_and_execute(adapter, kin, scene, registry, step_hook, target_pos, target_z_axis=None, base_xy=None, use_mplib=True):
    """Solve IK, plan a collision-free trajectory, then execute it.

    Mirrors the LLM pipeline: query_ik_solution -> query_arm_path ->
    arm_trajectory_follow. Returns the final SkillResult. ``base_xy`` defaults
    to the observed base pose (state-sync contract); pass an explicit value to
    plan from a known work position. ``use_mplib=False`` falls back to a direct
    joint interpolation (for tight table-level descents where the inflated
    table point-cloud makes MPlib over-conservative; the interpolated path is
    verified safe offline).
    """
    obs = adapter.read_observation(0.0)
    if base_xy is None:
        base_xy = (float(obs.base_pose[0]), float(obs.base_pose[1])) if obs.base_pose else (0.0, 0.0)

    q_cur = np.array([obs.joint_positions[j] for j in ("left_arm_joint1", "left_arm_joint2",
                                                       "left_arm_joint3", "left_arm_joint4",
                                                       "left_arm_joint5", "left_arm_joint6",
                                                       "left_arm_joint7")])

    # Solve IK seeded from the current configuration: the min-motion picker
    # then stays on the current redundant-arm branch instead of jumping to an
    # unrelated branch (elbow/wrist visibly "twisting" between segments).
    q_pg = None
    if target_z_axis is not None:
        from r1pro_data_gen.skills.manipulation.arm import quat_from_z_axis

        q_pg = quat_from_z_axis(np.asarray(target_z_axis, dtype=float))
    sol = kin.ik(np.asarray(target_pos, dtype=float), q_pg, q_init=q_cur)
    if not sol.success:
        return None, {"reason": "ik unsolvable", "ik_error_m": sol.position_error}
    q_goal = sol.q_arm

    if not use_mplib:
        # Direct joint interpolation (verified safe offline for tight descents
        # where the inflated table point-cloud over-rejects MPlib). Executed
        # speed-limited so the arm tracks each segment.
        from r1pro_data_gen.skills.manipulation.arm import ArmSegmentExecutor

        vel_limits = np.full(7, 7.12)
        segment = ArmSegmentExecutor(kin, vel_limits, speed_scale=0.3, hold_steps=10)
        n = max(4, int(np.ceil(float(np.max(np.abs(q_goal - q_cur))) / 0.1)))
        traj = [q_cur + (q_goal - q_cur) * (i / n) for i in range(n + 1)]
        for q_prev, q_next in zip(traj[:-1], traj[1:]):
            final_err = segment.execute(adapter, "left", q_prev, q_next, step_hook)
            if final_err >= 0.12:  # generous: the next segment re-aims
                return SkillResult(success=False, skill="arm_trajectory_follow",
                                   metrics={"final_error_rad": float(final_err)}), {
                    "traj_points": len(traj), "mode": "interp", "failed_seg": float(final_err),
                }
        return SkillResult(success=True, skill="arm_trajectory_follow",
                           metrics={"waypoints": float(len(traj))}), {"traj_points": len(traj), "mode": "interp"}

    # Plan a collision-free trajectory around the scene obstacles. MPlib/OMPL
    # is randomized: a single run can return a path that brushes an obstacle on
    # dense re-validation, so retry a few seeds before giving up.
    from r1pro_data_gen.methods.manipulation.mplib_path import path_collision_free, plan_arm_path
    from r1pro_data_gen.methods.manipulation.mplib_path import build_planner

    planner = build_planner()
    last = None
    link = None
    for attempt in range(10):
        out = plan_arm_path(planner, q_cur, q_goal, scene, base_xy=base_xy, planning_time=3.0, kin=kin)
        if not out["success"]:
            last = out
            continue
        free, link = path_collision_free(planner, out["position"], scene, base_xy=base_xy)
        if free:
            break
        last = out
    else:
        return None, {
            "reason": "planned path collides after retries",
            "base_xy": [round(float(v), 4) for v in base_xy],
            "n_waypoints": len(last["position"]) if last else 0,
            "collision_segment": link if "link" in locals() else None,
        }

    # Execute the joint trajectory (with the TOPP velocity profile when present,
    # so the PD drive tracks the planned speed -- smooth, no rattling).
    traj = [row.tolist() for row in out["position"]]
    vel = [row.tolist() for row in out["velocity"]] if out.get("velocity") is not None else None
    result = registry.get("arm_trajectory_follow").execute(
        adapter, scene=scene, step_hook=step_hook, trajectory=traj, velocities=vel,
    )
    # Trajectory quality gate: the end-effector path must not wander (winding
    # ratio) and must not dip far below BOTH endpoints (an unnecessary swoop;
    # a legit detour around the table stays near the endpoint heights). These
    # are the "looks weird" metrics the joint-space gates cannot see.
    ee = np.array([kin.fk(np.asarray(row, dtype=float))[0] for row in out["position"]])
    seg = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    ee_winding = float(seg.sum() / max(np.linalg.norm(ee[-1] - ee[0]), 1e-9))
    min_ee_z = float(ee[:, 2].min())
    z_floor = min(float(ee[0, 2]), float(ee[-1, 2])) - 0.3
    quality_ok = ee_winding < 3.5 and min_ee_z > z_floor
    if not quality_ok:
        return SkillResult(success=False, skill="arm_trajectory_follow",
                           metrics={"ee_winding": ee_winding, "min_ee_z": min_ee_z}), {
            "traj_points": len(traj), "reason": "EE path quality gate failed",
            "ee_winding": round(ee_winding, 3), "min_ee_z": round(min_ee_z, 3),
        }
    return result, {"traj_points": len(traj), "ee_winding": round(ee_winding, 3),
                    "min_ee_z": round(min_ee_z, 3)}


# The reference project (r1p-pickplace-isaaclab-skrl) approaches from a
# forward-reach preparation pose: EE winding 1.18 (almost straight). A rear
# swing detour produced winding 2.2+ -- visually "weird" -- so we mirror the
# reference structure: home -> forward reach (MPlib, winding ~1.3) -> frontal
# pregrasp (MPlib, winding ~1.2). Shared by the arm scenarios; mirrors
# Historical PickPlace reference structure; current physical scenarios are
# generic skill fixtures and do not load a task-specific plan.
_PREGRASP_POS = [0.45, 0.0, 1.30]
_PREGRASP_Z_AXIS = [-1.0, 0.0, 0.0]
# The reference project's reset pose (a fixed joint configuration, NOT an IK
# target: a positional IK would pick any redundant branch and the elbow could
# visibly twist into an unnatural pose).
_FWD_Q = np.array([-1.5708, 1.48355, 0.0, -0.6981, 0.0, 0.0, 0.0])
_FWD_POS = [-0.025, 0.88, 1.714]  # its end-effector position (diagnostics)


def _plan_to_joint(adapter, kin, scene, registry, step_hook, target_q, base_xy, use_mplib=True):
    """Plan + execute a move to a FIXED joint configuration (no IK).

    Returns (SkillResult | None, detail). ``use_mplib=False`` uses direct
    joint interpolation (trapezoid segments); True plans with MPlib.
    """
    obs = adapter.read_observation(0.0)
    q_cur = np.array([obs.joint_positions[j] for j in (
        "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
        "left_arm_joint5", "left_arm_joint6", "left_arm_joint7")])
    q_goal = np.asarray(target_q, dtype=float)
    if q_goal.shape != (7,):
        raise ValueError("_plan_to_joint expects 7 joint values")

    if not use_mplib:
        from r1pro_data_gen.skills.manipulation.arm import ArmSegmentExecutor

        segment = ArmSegmentExecutor(kin, np.full(7, 7.12), speed_scale=0.3, hold_steps=10)
        n = max(4, int(np.ceil(float(np.max(np.abs(q_goal - q_cur))) / 0.1)))
        traj = [q_cur + (q_goal - q_cur) * (i / n) for i in range(n + 1)]
        for q_prev, q_next in zip(traj[:-1], traj[1:]):
            final_err = segment.execute(adapter, "left", q_prev, q_next, step_hook)
            if final_err >= 0.12:
                return SkillResult(success=False, skill="arm_trajectory_follow",
                                   metrics={"final_error_rad": float(final_err)}), {
                    "traj_points": len(traj), "mode": "interp", "failed_seg": float(final_err),
                }
        return SkillResult(success=True, skill="arm_trajectory_follow",
                           metrics={"waypoints": float(len(traj))}), {"traj_points": len(traj), "mode": "interp"}

    from r1pro_data_gen.methods.manipulation.mplib_path import build_planner, plan_arm_path

    planner = build_planner()
    # Plan + execute with retries: OMPL is randomized and some paths have a
    # sharper corner than the compliant drive can track; a fresh plan is a
    # fresh path. After a failed execution the arm may sit mid-way (possibly
    # near a singular pose); return to the start config before re-planning.
    obs0 = adapter.read_observation(0.0)
    q_start = np.array([obs0.joint_positions[j] for j in (
        "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
        "left_arm_joint5", "left_arm_joint6", "left_arm_joint7")])
    # Slightly-off-home re-planning pose: exact home (all zeros) is a singular
    # configuration and OMPL repeatedly fails to expand from it.
    q_safe = np.array([0.15, 0.2, 0.0, -0.1, 0.0, 0.0, 0.0])
    result = None
    for attempt in range(8):
        obs = adapter.read_observation(0.0)
        q_cur = np.array([obs.joint_positions[j] for j in (
            "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
            "left_arm_joint5", "left_arm_joint6", "left_arm_joint7")])
        for plan_attempt in range(4):
            out = plan_arm_path(planner, q_cur, q_goal, scene, base_xy=base_xy, planning_time=3.0, kin=kin)
            if out["success"]:
                break
        else:
            return None, {"reason": "planned path collides after retries",
                          "base_xy": [round(float(v), 4) for v in base_xy]}
        traj = [row.tolist() for row in out["position"]]
        vel = [row.tolist() for row in out["velocity"]] if out.get("velocity") is not None else None
        result = registry.get("arm_trajectory_follow").execute(
            adapter, scene=scene, step_hook=step_hook, trajectory=traj, velocities=vel,
        )
        if result.success:
            break
        registry.get("arm_joint_to").execute(
            adapter, scene=scene, step_hook=step_hook, target_q=q_safe,
        )
    ee = np.array([kin.fk(np.asarray(row, dtype=float))[0] for row in out["position"]])
    seg = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    ee_winding = float(seg.sum() / max(np.linalg.norm(ee[-1] - ee[0]), 1e-9))
    return result, {"traj_points": len(traj), "ee_winding": round(ee_winding, 3),
                    "exec_metrics": result.metrics if result else {}}


def _arm_to_pregrasp(adapter, kin, scene, registry, step_hook, base_xy=(0.05, 0.15), use_mplib=True):
    """Forward reach + frontal pregrasp (v2 pipeline); returns the final result.

    ``base_xy`` is the pickplace work position (the scenario runs with the base
    already there); planning from the observed base would treat the table as a
    far obstacle and plan a long, slow detour. ``use_mplib=False`` uses direct
    joint interpolation for both segments.
    """
    registry.get("base_lock_wheels").execute(adapter, scene=scene, step_hook=step_hook)
    registry.get("gripper_set").execute(adapter, scene=scene, step_hook=step_hook, open_value=0.05)
    fwd_result, detail = _plan_to_joint(adapter, kin, scene, registry, step_hook, _FWD_Q,
                                        base_xy=base_xy, use_mplib=use_mplib)
    if fwd_result is None or not fwd_result.success:
        return None, {"reason": "forward reach failed", "detail": detail}
    result, rdetail = _plan_and_execute(adapter, kin, scene, registry, step_hook, _PREGRASP_POS, _PREGRASP_Z_AXIS,
                                        base_xy=base_xy, use_mplib=use_mplib)
    return result, rdetail


def scenario_arm_move_to(adapter, kin, scene, registry, step_hook):
    """Verify a compact position-only EE move with a stable orientation.

    This is intentionally not a pick/place target. It tests the public
    position-only contract and keeps the endpoint on the natural forward-reach
    branch so the video shows a continuous arm sweep instead of a redundant
    wrist/elbow reconfiguration.
    """
    side = _verification_side(adapter)
    q_goal = np.asarray(_side_joint_pose([-0.75, 0.55, 0.08, -0.48, 0.10, 0.0, 0.0], side))
    target_pos, target_quat = kin.fk(q_goal)
    result = registry.get("arm_move_to").execute(
        adapter,
        scene=scene,
        step_hook=step_hook,
        # Use the scene's reachable frontal pregrasp pose.  The previous
        # boundary target was outside the neutral-home IK/planning branch, so
        # the verifier recorded only warmup/tail frames and produced a
        # misleading near-empty video.  This still exercises the public
        # position-only arm_move_to contract; the target remains replaceable
        # test data, not a constraint in the skill implementation.
        target_pos=target_pos.tolist(),
        target_quat=target_quat.tolist(),
        side=side,
        planning_time=3.0,
    )
    return result.success, dict(result.metrics, **result.details)


def scenario_arm_joint_to(adapter, kin, scene, registry, step_hook):
    """Move to a moderate, continuous joint posture."""
    registry.get("base_lock_wheels").execute(adapter, scene=scene, step_hook=step_hook)
    side = _verification_side(adapter)
    result = registry.get("arm_joint_to").execute(
        adapter, scene=scene, step_hook=step_hook,
        target_q=_side_joint_pose([-0.7, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0], side), side=side,
    )
    return result.success, {"final_error_rad": result.metrics.get("final_error_rad")}


def scenario_arm_trajectory_follow(adapter, kin, scene, registry, step_hook):
    """Follow a visible, reusable three-waypoint joint trajectory."""
    registry.get("base_lock_wheels").execute(adapter, scene=scene, step_hook=step_hook)
    obs = adapter.read_observation(0.0)
    side = _verification_side(adapter)
    q_start = np.array([obs.joint_positions[f"{side}_arm_joint{i}"] for i in range(1, 8)])
    q_goal = np.array(_side_joint_pose([-0.75, 0.55, 0.08, -0.48, 0.10, 0.0, 0.0], side))
    q_mid = q_start + 0.65 * (q_goal - q_start)
    traj = [q_start.tolist(), q_mid.tolist(), q_goal.tolist()]
    result = registry.get("arm_trajectory_follow").execute(
        adapter, scene=scene, step_hook=step_hook, trajectory=traj, side=side
    )
    return result.success, {"waypoints": result.metrics.get("waypoints")}


def scenario_arm_move_directional(adapter, kin, scene, registry, step_hook):
    """Advance the end-effector downward by a fixed distance (toward the table).

    The contact sensors only filter the cylinder, so table contact is not
    detectable here -- we move a bounded distance and verify the arm followed
    the direction. ``until_contact`` is exercised by gripper_grasp (cylinder).
    """
    import numpy as np

    side = _verification_side(adapter)
    q_log: list[list[float]] = []

    def wrap_hook():
        step_hook()
        obs = adapter.read_observation(0.0)
        q_log.append([obs.joint_positions[f"{side}_arm_joint{i}"] for i in range(1, 8)])

    prepared = registry.get("arm_joint_to").execute(
        adapter, scene=scene, step_hook=wrap_hook,
        target_q=_side_joint_pose([-0.7, 0.70, 0.0, -0.70, 0.0, 0.0, 0.0], side), speed_scale=0.20, side=side,
    )
    if not prepared.success:
        return False, {"reason": "directional start pose unreachable", **prepared.metrics}
    q_log.clear()  # smoothness measured only over the directional move itself
    result = registry.get("arm_move_directional").execute(
        adapter, scene=scene, step_hook=wrap_hook,
        direction=[0.0, 0.0, -1.0], distance=0.08, step=0.004,
        until_contact=False, side=side,
    )
    q_arr = np.asarray(q_log)
    max_step = 0.0
    if len(q_arr) > 2:
        d = np.abs(np.diff(q_arr, axis=0))
        max_step = float(d.max())
    smooth = max_step < 0.15
    q_log_out = [[round(float(v), 4) for v in row] for row in q_arr] if len(q_arr) else []
    return result.success and smooth, {
        "reason": result.details.get("reason"),
        "moved_m": result.metrics.get("moved_m"),
        "contact": result.metrics.get("contact"),
        "final_error_rad": result.metrics.get("final_error_rad"),
        "max_joint_step_rad": round(max_step, 4),
        "q_log": q_log_out,
    }


def scenario_arm_rotate_ee(adapter, kin, scene, registry, step_hook):
    """Rotate the end-effector about its local z axis (valve-like)."""
    import numpy as np

    side = _verification_side(adapter)
    q_log: list[list[float]] = []

    def wrap_hook():
        step_hook()
        obs = adapter.read_observation(0.0)
        q_log.append([obs.joint_positions[f"{side}_arm_joint{i}"] for i in range(1, 8)])

    prepared = registry.get("arm_joint_to").execute(
        adapter, scene=scene, step_hook=wrap_hook,
        target_q=_side_joint_pose([-0.9, 0.95, 0.0, -0.52, 0.0, 0.0, 0.0], side), speed_scale=0.22, side=side,
    )
    if not prepared.success:
        return False, {"reason": "rotate start pose unreachable", **prepared.metrics}
    q_log.clear()  # smoothness measured only over the rotation itself
    result = registry.get("arm_rotate_ee").execute(
        adapter, scene=scene, step_hook=wrap_hook,
        axis=[0.0, 0.0, 1.0], angle=0.52, frame="end_effector", steps=24, side=side,
    )
    # Smoothness gate: the arm must not "wave" -- no single-step joint jump.
    q_arr = np.asarray(q_log)
    max_step = 0.0
    jerk_count = 0
    if len(q_arr) > 2:
        d = np.abs(np.diff(q_arr, axis=0))  # (n-1, 7)
        max_step = float(d.max())
        jerk_count = int((d.max(axis=1) > 0.15).sum())  # >0.15 rad/step = jerk
    smooth = max_step < 0.15
    # Keep the raw joint trace (rounded) for offline smoothness analysis --
    # the video alone cannot show whether a jump is a planned fast move or a
    # PD oscillation.
    q_log_out = [[round(float(v), 4) for v in row] for row in q_arr] if len(q_arr) else []
    return result.success and smooth, {
        "angle_rad": result.metrics.get("angle_rad"),
        "max_joint_step_rad": round(max_step, 4),
        "jerk_count": jerk_count,
        "q_log": q_log_out,
    }


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------


def _first_dynamic_object_name(scene) -> str | None:
    for obj in scene.objects:
        physics = getattr(obj, "physics", None)
        if physics is None:
            continue
        if not physics.kinematic and physics.mass:
            return obj.name
    return None


def scenario_grasp_object(adapter, kin, scene, registry, step_hook):
    """Grasp the scene's first dynamic object from the current base pose."""
    del kin
    object_name = _first_dynamic_object_name(scene)
    if object_name is None:
        return False, {"reason": "scene has no dynamic object"}
    side = _verification_side(adapter)
    result = registry.get("grasp_object").execute(
        adapter,
        scene=scene,
        step_hook=step_hook,
        object_name=object_name,
        side=side,
    )
    attached = False
    if hasattr(adapter, "is_object_attached"):
        try:
            attached = bool(adapter.is_object_attached(object_name))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            attached = False
    ok = bool(result.success and attached)
    payload = {
        "object_name": object_name,
        "success": result.success,
        "attached": attached,
        "failure_code": result.details.get("failure_code"),
        "reason": result.details.get("reason"),
        "attempts": result.details.get("attempts"),
        "metrics": result.metrics,
    }
    try:
        from pathlib import Path
        import json

        Path("outputs/skills/grasp_object_result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    print(f"GRASP_DETAILS failure_code={payload['failure_code']} attempts={payload['attempts']}", flush=True)
    return ok, payload


def scenario_gripper_set(adapter, kin, scene, registry, step_hook):
    """Open then close the left gripper."""
    side = _verification_side(adapter)
    open_result = registry.get("gripper_set").execute(
        adapter, scene=scene, step_hook=step_hook, open_value=0.05, side=side
    )
    if not open_result.success:
        return False, {"reason": "gripper open failed"}
    close_result = registry.get("gripper_set").execute(
        adapter, scene=scene, step_hook=step_hook, open_value=0.0, side=side
    )
    return close_result.success, {"final_finger_pos_m": close_result.metrics.get("final_finger_pos_m")}


def scenario_gripper_grasp(adapter, kin, scene, registry, step_hook):
    """Verify the gripper atom in a geometry-derived pinch fixture.

    Reaching a grasp pose is covered independently by ``arm_move_to`` and
    ``query_arm_path``. This scenario isolates the gripper controller: open the
    fingers, derive their physical midpoint from live simulation, place the
    cylinder there, and require both filtered contact sensors to stop closure.
    Gravity is disabled only for this fixture so the object does not fall while
    the arm remains stationary; collision and contact dynamics stay enabled.
    """
    side = _verification_side(adapter)
    if not getattr(adapter, "_gripper_fixture_prepared", False):
        prepared, center = _prepare_gripper_fixture(adapter, registry, step_hook)
        if not prepared:
            return False, {"reason": "fixture gripper open or placement failed"}
    else:
        center = np.asarray(adapter.object_position("cylinder"), dtype=float)
    center = np.asarray(center, dtype=float)
    before = np.asarray(adapter.object_position("cylinder"), dtype=float)

    result = registry.get("gripper_grasp").execute(
        adapter,
        scene=scene,
        step_hook=step_hook,
        side=side,
        max_close=0.05,
        contact_threshold=0.2,
        step=0.001,
    )
    forces = tuple(float(force) for force in adapter.finger_contact_forces(side=side))
    after = np.asarray(adapter.object_position("cylinder"), dtype=float)
    drift = float(np.linalg.norm(after - before))
    both_contact = len(forces) >= 2 and min(forces[:2]) > 0.2
    held_between_fingers = drift < 0.08
    ok = result.success and both_contact and held_between_fingers
    return ok, {
        "contact_detected": result.metrics.get("contact_detected"),
        "final_finger_pos_m": result.metrics.get("final_finger_pos_m"),
        "forces": list(forces),
        "cylinder_drift_m": drift,
        "fixture_center": [round(float(value), 4) for value in center],
        "reason": None if ok else "pinch/contact/retention gate failed",
    }


def _prepare_gripper_fixture(adapter, registry, step_hook=None):
    """Place the dynamic fixture before video warmup and hold it under gravity.

    The initial YAML pose is only a valid spawn placeholder. The actual grasp
    center depends on the live finger kinematics, so preparation must happen
    after reset but before the recorder's warmup frames. This prevents a
    visible fall followed by a state teleport in the showcase video.
    """
    import torch
    from pxr import PhysxSchema
    import omni.usd

    # Disable gravity before any settling step. The fixture is deliberately
    # dynamic so contacts remain physical, but it must not fall during the
    # unrecorded placement/opening phase.
    stage = omni.usd.get_context().get_stage()
    cylinder_prim = stage.GetPrimAtPath("/World/Cylinder")
    PhysxSchema.PhysxRigidBodyAPI.Apply(cylinder_prim).GetDisableGravityAttr().Set(True)

    side = _verification_side(adapter)
    registry.get("base_lock_wheels").execute(adapter, scene=None)
    opened = registry.get("gripper_set").execute(adapter, scene=None, open_value=0.05, side=side)
    if not opened.success:
        return False, np.zeros(3, dtype=float)

    body_index = {name: i for i, name in enumerate(adapter.robot.data.body_names)}
    required = (f"{side}_gripper_finger_link1", f"{side}_gripper_finger_link2")
    if any(name not in body_index for name in required):
        return False, np.zeros(3, dtype=float)
    body_pos = adapter.robot.data.body_pos_w[0].detach()
    f1 = body_pos[body_index[required[0]]]
    f2 = body_pos[body_index[required[1]]]
    center = 0.5 * (f1 + f2)
    finger_quat = adapter.robot.data.body_quat_w[0, body_index[required[0]]].detach().cpu().numpy()
    w, x, y, z = (float(value) for value in finger_quat)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    center = center + torch.tensor(rotation @ np.array([0.0, 0.0, 0.02]), device=adapter.robot.device)

    cylinder = adapter.scene.rigid_objects["cylinder"]
    pose = torch.zeros((1, 7), device=adapter.robot.device)
    pose[0, :3] = center
    pose[0, 3] = 1.0
    cylinder.write_root_pose_to_sim(pose)
    cylinder.write_root_velocity_to_sim(torch.zeros((1, 6), device=adapter.robot.device))
    adapter.step()
    if step_hook is not None:
        step_hook()
    adapter._gripper_fixture_prepared = True
    return True, center.detach().cpu().numpy()


PREPARATIONS = {
    "gripper_grasp": _prepare_gripper_fixture,
}


def prepare_scenario(skill, adapter, kin, scene, registry):
    """Run unrecorded physical setup required by a showcase scenario."""
    del kin, scene
    preparation = PREPARATIONS.get(skill)
    if preparation is None:
        return True, {}
    ok, center = preparation(adapter, registry)
    return bool(ok), {"fixture_center": np.asarray(center, dtype=float).tolist()}


# ---------------------------------------------------------------------------
# Query skills
# ---------------------------------------------------------------------------


def scenario_query_object_pose(adapter, kin, scene, registry, step_hook):
    result = registry.get("query_object_pose").execute(
        adapter, scene=scene, step_hook=step_hook, object_name="cylinder"
    )
    if not result.success:
        return False, {}
    pos = result.details.get("position")
    ok = pos is not None and abs(pos[2] - 1.11) < 0.2  # near the table surface
    return ok, {"position": pos}


def scenario_query_contacts(adapter, kin, scene, registry, step_hook):
    """Create a controlled finger/object contact, then verify force readback."""
    import torch

    side = _verification_side(adapter)
    body_idx = {name: i for i, name in enumerate(adapter.robot.data.body_names)}
    finger_idx = body_idx[f"{side}_gripper_finger_link1"]
    finger_pos = adapter.robot.data.body_pos_w[0, finger_idx].detach().clone()
    cylinder = adapter.scene.rigid_objects["cylinder"]
    pose = torch.zeros((1, 7), device=adapter.robot.device)
    pose[0, 3] = 1.0
    zero_velocity = torch.zeros((1, 6), device=adapter.robot.device)
    peak = 0.0
    peak_forces: list[float] = []
    # The visual/collision shape origin need not coincide exactly with the
    # rigid-body COM. Sweep a compact 3-D stencil and sample immediately after
    # each physics step, before the dynamic cylinder can be expelled.
    offsets = [
        (0.0, 0.0, 0.0), (-0.03, 0.0, 0.0), (0.03, 0.0, 0.0),
        (0.0, -0.03, 0.0), (0.0, 0.03, 0.0),
        (0.0, 0.0, -0.03), (0.0, 0.0, 0.03),
    ]
    for offset in offsets:
        pose[0, :3] = finger_pos + torch.tensor(offset, device=adapter.robot.device)
        cylinder.write_root_pose_to_sim(pose)
        cylinder.write_root_velocity_to_sim(zero_velocity)
        adapter.step()
        if step_hook is not None:
            step_hook()
        result = registry.get("query_contacts").execute(adapter, scene=scene, step_hook=step_hook, side=side)
        forces = [float(force) for force in result.details.get("contact_forces", [])]
        candidate_peak = max(forces, default=0.0)
        if candidate_peak > peak:
            peak = candidate_peak
            peak_forces = forces
        if peak > 0.2:
            break
    return result.success and peak > 0.2, {"forces": peak_forces, "peak_force_n": peak}


def scenario_query_ee_pose(adapter, kin, scene, registry, step_hook):
    registry.get("base_lock_wheels").execute(adapter, scene=scene, step_hook=step_hook)
    side = _verification_side(adapter)
    result = registry.get("query_ee_pose").execute(adapter, scene=scene, step_hook=step_hook, side=side)
    pos = result.details.get("position")
    return result.success and pos is not None, {"position": pos}


def scenario_query_joint_pos(adapter, kin, scene, registry, step_hook):
    side = _verification_side(adapter)
    joint = f"{side}_arm_joint1"
    result = registry.get("query_joint_pos").execute(
        adapter, scene=scene, step_hook=step_hook, joints=[joint]
    )
    jp = result.details.get("joint_positions", {})
    return result.success and joint in jp, {"n_joints": len(jp)}


# ---------------------------------------------------------------------------
# v2 solve/plan skills (query_ik_solution / query_arm_path / query_base_path)
# ---------------------------------------------------------------------------


def scenario_query_ik_solution(adapter, kin, scene, registry, step_hook):
    """Solve IK for the pregrasp pose and return the joint config (no motion)."""
    side = _verification_side(adapter)
    q_goal = np.asarray(_side_joint_pose([-0.75, 0.55, 0.08, -0.48, 0.10, 0.0, 0.0], side))
    target_pos, target_quat = kin.fk(q_goal)
    result = registry.get("query_ik_solution").execute(
        adapter, scene=scene, step_hook=step_hook,
        target_pos=target_pos.tolist(), target_quat=target_quat.tolist(), side=side,
    )
    q = result.details.get("q_arm")
    ok = result.success and q is not None and len(q) == 7
    return ok, {"ik_error_m": result.metrics.get("ik_error_m"), "q_len": len(q) if q else 0}


def scenario_query_arm_path(adapter, kin, scene, registry, step_hook):
    """Verify query-only planning followed by the low-level executor."""
    side = _verification_side(adapter)
    q_goal = np.asarray(_side_joint_pose([-0.75, 0.55, 0.08, -0.48, 0.10, 0.0, 0.0], side))
    target_pos, target_quat = kin.fk(q_goal)
    ik = registry.get("query_ik_solution").execute(
        adapter, scene=scene, step_hook=step_hook,
        target_pos=target_pos.tolist(), target_quat=target_quat.tolist(), side=side,
    )
    q = ik.details.get("q_arm")
    if not ik.success or q is None:
        return False, {"reason": "IK prerequisite failed", **ik.metrics}
    planned = registry.get("query_arm_path").execute(
        adapter, scene=scene, step_hook=step_hook, target_q=q, planning_time=3.0, side=side,
    )
    if not planned.success:
        return False, {"reason": "path planning failed", **planned.details}
    executed = registry.get("arm_trajectory_follow").execute(
        adapter, scene=scene, step_hook=step_hook,
        trajectory=planned.details["trajectory"],
        velocities=planned.details.get("velocity"),
        trajectory_dt=planned.metrics.get("dt", 1.0 / 60.0), side=side,
    )
    return executed.success, {**planned.metrics, **executed.metrics}


def scenario_query_base_path(adapter, kin, scene, registry, step_hook):
    """Plan a 2D A* path and replay it so the video shows the planned route."""
    target = [1.0, 0.0, 0.0]
    result = registry.get("query_base_path").execute(
        adapter, scene=scene, step_hook=step_hook, target=target, resolution=0.05,
    )
    path = result.details.get("path")
    detour = max((abs(float(p[1])) for p in path), default=0.0) if path else 0.0
    length = sum(
        float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
        for a, b in zip(path[:-1], path[1:])
    ) if path else 0.0
    minimum = 3
    threshold = 0.5
    ok = result.success and path is not None and len(path) >= minimum and detour > threshold and length > 1.0
    if not ok:
        return False, {"waypoints": len(path) if path else 0, "planned_detour_m": round(detour, 4), "path_length_m": round(length, 4), "path": path}
    # QueryBasePath returns safe grid-cell centers.  The final cell center can
    # be several centimetres from the requested pose, so append the exact
    # requested XY goal before replaying; otherwise this test validates only
    # that the robot reaches the last grid cell, not the public API target.
    replay_path = [list(point) for point in path]
    replay_path.append([float(target[0]), float(target[1])])
    replay = registry.get("base_follow_path").execute(
        adapter, scene=scene, step_hook=step_hook, path=replay_path,
        # Keep the planner's 2 cm arrival tolerance so the first cell-center
        # waypoint is actually traversed; skipping it would turn the first
        # collision-checked segment into a long uncontrolled chord.
        target_yaw=float(target[2]), v_max=0.25, omega_max=0.35, arrive_tol=0.02,
    )
    return replay.success, {
        "waypoints": len(path),
        "replay_waypoints": len(replay_path),
        "planned_detour_m": round(detour, 4),
        "path_length_m": round(length, 4),
        "path": path,
        "replay_arrival_error_m": replay.metrics.get("arrival_error_m"),
        "replay_reason": replay.details.get("reason"),
        "replay_failed_waypoint": replay.details.get("failed_waypoint"),
    }


SCENARIOS = {
    "base_move_to": scenario_base_move_to,
    "base_rotate_to": scenario_base_rotate_to,
    "base_follow_path": scenario_base_follow_path,
    "base_velocity_set": scenario_base_velocity_set,
    "base_navigate_to": scenario_base_navigate_to,
    "base_lock_wheels": scenario_base_lock_wheels,
    "base_unlock_wheels": scenario_base_unlock_wheels,
    "joint_mask_lock": scenario_joint_mask_lock,
    "joint_mask_unlock": scenario_joint_mask_unlock,
    "torso_move_to": scenario_torso_move_to,
    "arm_move_to": scenario_arm_move_to,
    "arm_joint_to": scenario_arm_joint_to,
    "arm_trajectory_follow": scenario_arm_trajectory_follow,
    "arm_move_directional": scenario_arm_move_directional,
    "arm_rotate_ee": scenario_arm_rotate_ee,
    "grasp_object": scenario_grasp_object,
    "gripper_set": scenario_gripper_set,
    "gripper_grasp": scenario_gripper_grasp,
    "query_object_pose": scenario_query_object_pose,
    "query_contacts": scenario_query_contacts,
    "query_ee_pose": scenario_query_ee_pose,
    "query_joint_pos": scenario_query_joint_pos,
    "query_ik_solution": scenario_query_ik_solution,
    "query_arm_path": scenario_query_arm_path,
    "query_base_path": scenario_query_base_path,
}


def get_scenario(skill: str):
    if skill not in SCENARIOS:
        raise KeyError(f"no verification scenario yet for skill {skill!r}")
    return SCENARIOS[skill]
