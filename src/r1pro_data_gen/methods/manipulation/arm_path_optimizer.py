"""Finite-budget candidate orchestration for collision-aware arm motion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from r1pro_data_gen.methods.manipulation.contracts import (
    ArmPlanningBudget,
    ArmPlanningResult,
    ArmSequenceCandidate,
    ArmSequencePlanningResult,
    ArmWaypoint,
    ConstraintReport,
    IKCandidate,
    PathCandidate,
    WaypointIKCandidate,
)

_SCORE_SCALE = 100_000.0
# Normalized L2 joint delta beyond which an IK goal is a different redundant
# branch. Cartesian interpolants do not use this gate; it only drops those
# goals from the OMPL round-robin so a joint-space lerp cannot snap the wrist.
_MAX_NORMALIZED_JOINT_DELTA = 1.25
_MAX_SINGLE_JOINT_JUMP_RAD = 1.20


def _quantize(value: float) -> int:
    if not np.isfinite(value):
        raise ValueError("path score metrics must be finite")
    return int(round(float(value) * _SCORE_SCALE))


def _path_metrics(output: dict[str, Any], kin: Any, posture_cost: float) -> dict[str, float]:
    position = np.asarray(output["position"], dtype=np.float64)
    ee = np.asarray([kin.fk(q)[0] for q in position], dtype=np.float64)
    ee_length = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())
    span = np.maximum(np.asarray(kin.upper) - np.asarray(kin.lower), 1e-9)
    normalized = np.diff(position, axis=0) / span[None, :]
    joint_length = float(np.linalg.norm(normalized, axis=1).sum())
    acceleration = output.get("acceleration")
    if acceleration is None or len(acceleration) < 2:
        smoothness = 0.0
    else:
        acceleration = np.asarray(acceleration, dtype=np.float64)
        smoothness = float(np.square(acceleration).sum(axis=1).mean())
    return {
        "ee_path_length_m": ee_length,
        "normalized_joint_path_length": joint_length,
        "duration_s": float(output.get("duration", max(0, len(position) - 1) * float(output.get("dt", 1.0 / 60.0)))),
        "smoothness_cost": smoothness,
        "posture_cost": float(posture_cost),
        "joint_winding": float(output.get("winding", 1.0)),
        "ee_winding": float(output.get("ee_winding", 1.0)),
    }


def _score(metrics: dict[str, float], candidate_id: int, attempt_id: int, q_goal: tuple[float, ...]) -> tuple[int, ...]:
    """Stable lexicographic quality key; lower is better."""
    return (
        _quantize(metrics["ee_path_length_m"]),
        _quantize(metrics["normalized_joint_path_length"]),
        _quantize(metrics["duration_s"]),
        _quantize(metrics["smoothness_cost"]),
        _quantize(metrics["posture_cost"]),
        int(candidate_id),
        int(attempt_id),
        *(_quantize(q) for q in q_goal),
    )


def arm_request_hash(
    q_current: np.ndarray,
    full_q_current: np.ndarray,
    base_xy: tuple[float, float],
    base_yaw: float,
    scene: Any,
    ik_candidates: Iterable[IKCandidate],
) -> str:
    """Hash the frozen planning request without simulator-specific objects."""
    objects = []
    for obj in getattr(scene, "objects", ()):
        objects.append({
            "name": str(obj.name),
            "type": str(getattr(getattr(obj, "type", None), "value", getattr(obj, "type", ""))),
            "pos": [round(float(v), 8) for v in obj.pos],
            "size": None if getattr(obj, "size", None) is None else [round(float(v), 8) for v in obj.size],
            "radius": None if getattr(obj, "radius", None) is None else round(float(obj.radius), 8),
            "height": None if getattr(obj, "height", None) is None else round(float(obj.height), 8),
            "collision": bool(obj.physics.collision_enabled),
            "margin": None if obj.physics.planning_margin is None else round(float(obj.physics.planning_margin), 8),
        })
    payload = {
        "q_current": np.asarray(q_current, dtype=float).round(8).tolist(),
        "full_q_current": np.asarray(full_q_current, dtype=float).round(8).tolist(),
        "base_xy": [round(float(v), 8) for v in base_xy],
        "base_yaw": round(float(base_yaw), 8),
        "objects": objects,
        "goals": [[round(float(v), 8) for v in item.q_goal] for item in ik_candidates],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_joint_delta(kin: Any, q_from: np.ndarray, q_to: np.ndarray) -> float:
    span = np.maximum(np.asarray(kin.upper, dtype=float) - np.asarray(kin.lower, dtype=float), 1e-9)
    return float(np.linalg.norm((np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)) / span))


def _ik_continuity_ok(kin: Any, q_from: np.ndarray, q_to: np.ndarray) -> bool:
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    if float(np.max(np.abs(q_to - q_from))) > _MAX_SINGLE_JOINT_JUMP_RAD:
        return False
    return _normalized_joint_delta(kin, q_from, q_to) <= _MAX_NORMALIZED_JOINT_DELTA


def optimize_arm_path(
    planner: Any,
    kin: Any,
    q_current: np.ndarray,
    solutions: list[Any],
    scene: Any,
    *,
    base_xy: tuple[float, float],
    base_yaw: float,
    full_q_current: np.ndarray,
    planning_time: float,
    local_radius_m: float,
    speed_scale: float,
    side: str,
    attempts_per_candidate: int = 2,
    fallback_attempts_per_candidate: int = 1,
    max_joint_winding: float | None = None,
    max_ee_winding: float | None = None,
    target_pos: np.ndarray | None = None,
    target_quat: np.ndarray | None = None,
) -> ArmPlanningResult:
    """Generate, verify and select the unique best path in a bounded set."""
    from r1pro_data_gen.methods.manipulation.mplib_path import plan_arm_path
    from r1pro_data_gen.methods.manipulation.taskspace import plan_certified_task_path

    budget = ArmPlanningBudget(
        attempts_per_candidate=int(attempts_per_candidate),
        fallback_attempts_per_candidate=int(fallback_attempts_per_candidate),
        planning_time_per_attempt_s=float(planning_time),
    )
    ik_items = tuple(
        IKCandidate(
            candidate_id=index,
            q_goal=tuple(float(v) for v in solution.q_arm),
            position_error_m=float(solution.position_error),
            rotation_error_rad=float(solution.rotation_error),
            posture_cost=float(kin.posture_score(solution.q_arm, q_current)) if hasattr(kin, "posture_score") else 0.0,
        )
        for index, solution in enumerate(solutions)
    )
    request_hash = arm_request_hash(q_current, full_q_current, base_xy, base_yaw, scene, ik_items)
    reports: list[PathCandidate] = []
    valid_by_ik: set[int] = set()
    q_current = np.asarray(q_current, dtype=float)

    if target_pos is not None:
        task_output = plan_certified_task_path(
            planner,
            kin,
            q_current,
            np.asarray(target_pos, dtype=float),
            None if target_quat is None else np.asarray(target_quat, dtype=float),
            scene,
            base_xy=base_xy,
            base_yaw=base_yaw,
            full_q_current=full_q_current,
            speed_scale=float(speed_scale),
            side=side,
        )
        if task_output.get("success"):
            q_goal = tuple(float(v) for v in np.asarray(task_output["position"][-1], dtype=float))
            posture_cost = (
                float(kin.posture_score(np.asarray(q_goal), q_current))
                if hasattr(kin, "posture_score")
                else 0.0
            )
            metrics = _path_metrics(task_output, kin, posture_cost)
            winner = PathCandidate(
                candidate_id=-1,
                attempt_id=0,
                fallback=False,
                q_goal=q_goal,
                planner_status="TaskSpaceVerified",
                constraints=ConstraintReport(True, "verified"),
                metrics=metrics,
                score=_score(metrics, -1, 0, q_goal),
                output=task_output,
            )
            return ArmPlanningResult(
                success=True,
                status="success",
                reason="certified Cartesian interpolant",
                request_hash=request_hash,
                candidates=(winner,),
                winner=winner,
            )
        reports.append(PathCandidate(
            candidate_id=-1,
            attempt_id=0,
            fallback=False,
            q_goal=tuple(float(v) for v in q_current),
            planner_status=str(task_output.get("status", "TaskSpaceFailed")),
            constraints=ConstraintReport(
                False,
                str(task_output.get("failure_stage", "task_space")),
                (str(task_output.get("reason") or "Cartesian interpolant rejected"),),
            ),
        ))

    continuous_items = tuple(
        item for item in ik_items if _ik_continuity_ok(kin, q_current, np.asarray(item.q_goal))
    )
    ompl_items = continuous_items or ik_items
    rejected_discontinuous = tuple(
        item for item in ik_items if item not in ompl_items
    )
    for item in rejected_discontinuous:
        reports.append(PathCandidate(
            candidate_id=item.candidate_id,
            attempt_id=0,
            fallback=False,
            q_goal=item.q_goal,
            planner_status="DiscontinuousIK",
            constraints=ConstraintReport(
                False,
                "ik_continuity",
                ("IK goal exceeds the live-branch joint-continuity gate",),
            ),
        ))

    def run(item: IKCandidate, attempt_id: int, fallback: bool) -> None:
        output = plan_arm_path(
            planner,
            np.asarray(q_current, dtype=float),
            np.asarray(item.q_goal, dtype=float),
            scene,
            base_xy=base_xy,
            base_yaw=base_yaw,
            planning_time=budget.planning_time_per_attempt_s,
            kin=kin,
            side=side,
            local_radius_m=float(local_radius_m),
            speed_scale=float(speed_scale),
            mplib_attempts=1,
            allow_rrt_fallback=fallback,
            rrt_connect_mode="second_opinion",
            full_q_current=full_q_current,
        )
        status = str(output.get("status", "unknown"))
        if not output.get("success"):
            reports.append(PathCandidate(
                candidate_id=item.candidate_id,
                attempt_id=attempt_id,
                fallback=fallback,
                q_goal=item.q_goal,
                planner_status=status,
                constraints=ConstraintReport(False, str(output.get("failure_stage", "planning")), (str(output.get("reason") or status),)),
            ))
            return
        metrics = _path_metrics(output, kin, item.posture_cost)
        quality_reasons = []
        if max_joint_winding is not None and metrics["joint_winding"] > float(max_joint_winding):
            quality_reasons.append("task_joint_winding_limit")
        if max_ee_winding is not None and metrics["ee_winding"] > float(max_ee_winding):
            quality_reasons.append("task_ee_winding_limit")
        if quality_reasons:
            reports.append(PathCandidate(
                candidate_id=item.candidate_id,
                attempt_id=attempt_id,
                fallback=fallback,
                q_goal=item.q_goal,
                planner_status=status,
                constraints=ConstraintReport(False, "task_quality_limit", tuple(quality_reasons)),
                metrics=metrics,
            ))
            return
        score = _score(metrics, item.candidate_id, attempt_id, item.q_goal)
        reports.append(PathCandidate(
            candidate_id=item.candidate_id,
            attempt_id=attempt_id,
            fallback=fallback,
            q_goal=item.q_goal,
            planner_status=status,
            constraints=ConstraintReport(True, "verified"),
            metrics=metrics,
            score=score,
            output=output,
        ))
        valid_by_ik.add(item.candidate_id)

    # Round-robin among continuity-ok IK goals only. A distant redundant
    # branch is reachable by the Cartesian interpolant above; joint-space
    # OMPL should not snap to it.
    for round_id in range(budget.attempts_per_candidate):
        for item in ompl_items:
            run(item, round_id, False)
    for fallback_id in range(budget.fallback_attempts_per_candidate):
        for item in ompl_items:
            if item.candidate_id not in valid_by_ik:
                run(item, budget.attempts_per_candidate + fallback_id, True)

    valid = [candidate for candidate in reports if candidate.valid]
    if not valid:
        return ArmPlanningResult(
            success=False,
            status="no_collision_free_path",
            reason="no verified path was found within the finite planning budget",
            request_hash=request_hash,
            candidates=tuple(reports),
        )
    winner = min(valid, key=lambda candidate: candidate.score)
    return ArmPlanningResult(
        success=True,
        status="success",
        reason="selected the unique best verified candidate within budget",
        request_hash=request_hash,
        candidates=tuple(reports),
        winner=winner,
    )


@dataclass(slots=True)
class _SequencePrefix:
    candidates: tuple[WaypointIKCandidate, ...]
    segments: tuple[dict[str, Any], ...]
    reports: tuple[dict[str, Any], ...]
    q_last: np.ndarray
    natural_cost: float


def _sequence_request_hash(
    q_current: np.ndarray,
    full_q_current: np.ndarray,
    base_xy: tuple[float, float],
    base_yaw: float,
    scene: Any,
    waypoints: tuple[ArmWaypoint, ...],
) -> str:
    objects = [
        {
            "name": str(obj.name),
            "pos": [round(float(value), 8) for value in obj.pos],
            "collision": bool(obj.physics.collision_enabled),
            "margin": None
            if obj.physics.planning_margin is None
            else round(float(obj.physics.planning_margin), 8),
        }
        for obj in getattr(scene, "objects", ())
    ]
    payload = {
        "q_current": np.asarray(q_current, dtype=float).round(8).tolist(),
        "full_q_current": np.asarray(full_q_current, dtype=float).round(8).tolist(),
        "base_xy": [round(float(value), 8) for value in base_xy],
        "base_yaw": round(float(base_yaw), 8),
        "objects": objects,
        "waypoints": [
            {
                "name": waypoint.name,
                "poses": [
                    {
                        "position": [round(float(value), 8) for value in position],
                        "orientation": [round(float(value), 8) for value in orientation],
                    }
                    for position, orientation in waypoint.poses
                ],
                "exclude_objects": list(waypoint.exclude_objects),
                "contact": waypoint.contact,
                "speed_scale": waypoint.speed_scale,
            }
            for waypoint in waypoints
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _waypoint_candidate(
    kin: Any,
    solution: Any,
    q_from: np.ndarray,
    waypoint_id: int,
    candidate_id: int,
    orientation_id: int,
) -> WaypointIKCandidate:
    q_goal = np.asarray(solution.q_arm, dtype=float)
    lower = np.asarray(kin.lower, dtype=float)
    upper = np.asarray(kin.upper, dtype=float)
    span = np.maximum(upper - lower, 1e-9)
    normalized_delta = (q_goal - q_from) / span
    continuity = float(np.linalg.norm(normalized_delta))
    margin = np.minimum(q_goal - lower, upper - q_goal) / span
    minimum_margin = float(np.min(margin))
    wrist_motion = float(np.linalg.norm(normalized_delta[4:]))
    singular_value = (
        float(kin.minimum_singular_value(q_goal))
        if hasattr(kin, "minimum_singular_value")
        else 1.0
    )
    posture = (
        float(kin.posture_score(q_goal, q_from))
        if hasattr(kin, "posture_score")
        else continuity
    )
    singular_penalty = max(0.0, 0.04 - singular_value) / 0.04
    limit_penalty = max(0.0, 0.04 - minimum_margin) / 0.04
    natural_cost = (
        continuity
        + 0.20 * posture
        + 0.35 * wrist_motion
        + 1.50 * limit_penalty
        + 0.50 * singular_penalty
    )
    score = (
        _quantize(natural_cost),
        _quantize(continuity),
        _quantize(wrist_motion),
        _quantize(-minimum_margin),
        int(orientation_id),
        int(candidate_id),
        *(_quantize(value) for value in q_goal),
    )
    return WaypointIKCandidate(
        waypoint_id=waypoint_id,
        candidate_id=candidate_id,
        orientation_id=orientation_id,
        q_goal=tuple(float(value) for value in q_goal),
        position_error_m=float(solution.position_error),
        rotation_error_rad=float(solution.rotation_error),
        continuity_cost=continuity,
        posture_cost=posture,
        minimum_limit_margin=minimum_margin,
        wrist_motion=wrist_motion,
        minimum_singular_value=singular_value,
        score=score,
    )


def _continuous_waypoint_ik_candidates(
    kin: Any,
    position: np.ndarray,
    orientation: np.ndarray,
    q_current: np.ndarray,
    max_candidates: int,
) -> list[Any]:
    """Solve a waypoint from the live branch, retrying through Cartesian substeps.

    A long descend can be reachable with the requested pose while a single DLS
    solve from the carry-height posture falls outside its convergence basin.
    Keep the requested orientation and advance the seed along short Cartesian
    substeps; return only the final solution because the intermediate states
    are a solver aid, not additional semantic waypoints.  The normal direct
    multi-seed call remains the first choice and preserves the existing finite
    candidate budget.
    """
    position = np.asarray(position, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    q_current = np.asarray(q_current, dtype=float)
    direct = kin.ik_candidates(
        position,
        orientation,
        q_current,
        max_candidates=max(1, int(max_candidates)),
    )
    if direct:
        return direct
    if not hasattr(kin, "fk") or not hasattr(kin, "_ik_once"):
        return []

    start_position = np.asarray(kin.fk(q_current)[0], dtype=float)
    distance = float(np.linalg.norm(position - start_position))
    if not np.isfinite(distance) or distance <= 1e-6:
        return []
    steps = min(16, max(2, int(np.ceil(distance / 0.025))))
    q_seed = q_current.copy()
    final_solution = None
    for step_id in range(1, steps + 1):
        fraction = float(step_id) / float(steps)
        subtarget = start_position + fraction * (position - start_position)
        candidates = kin.ik_candidates(
            subtarget,
            orientation,
            q_seed,
            max_candidates=max(1, int(max_candidates)),
        )
        if not candidates:
            solution = kin._ik_once(
                subtarget,
                orientation,
                q_seed,
                pos_tol=0.003,
                rot_tol=0.02,
            )
            candidates = [solution] if bool(getattr(solution, "success", True)) and solution.q_arm is not None else []
        if not candidates:
            return []
        span = np.maximum(
            np.asarray(kin.upper, dtype=float) - np.asarray(kin.lower, dtype=float),
            1e-9,
        )
        final_solution = min(
            (
                candidate
                for candidate in candidates
                if bool(getattr(candidate, "success", True))
                and candidate.q_arm is not None
            ),
            key=lambda candidate: float(
                np.linalg.norm((np.asarray(candidate.q_arm, dtype=float) - q_seed) / span)
            ),
            default=None,
        )
        if final_solution is None:
            return []
        q_seed = np.asarray(final_solution.q_arm, dtype=float).copy()
    return [final_solution] if final_solution is not None else []


def _sequence_metrics(
    position: np.ndarray,
    kin: Any,
    candidates: tuple[WaypointIKCandidate, ...],
    duration: float,
) -> dict[str, float]:
    position = np.asarray(position, dtype=float)
    span = np.maximum(np.asarray(kin.upper) - np.asarray(kin.lower), 1e-9)
    normalized = np.diff(position, axis=0) / span[None, :]
    ee = np.asarray([kin.fk(q)[0] for q in position], dtype=float)
    return {
        "normalized_joint_path_length": float(np.linalg.norm(normalized, axis=1).sum()),
        "ee_path_length_m": float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum()),
        "duration_s": float(duration),
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(position, axis=0)))),
        "minimum_limit_margin": float(min(item.minimum_limit_margin for item in candidates)),
        "minimum_singular_value": float(min(item.minimum_singular_value for item in candidates)),
        "wrist_motion": float(sum(item.wrist_motion for item in candidates)),
        "naturalness_cost": float(
            sum(
                item.continuity_cost
                + 0.20 * item.posture_cost
                + 0.35 * item.wrist_motion
                for item in candidates
            )
        ),
    }


def optimize_arm_waypoint_path(
    planner: Any,
    kin: Any,
    q_current: np.ndarray,
    waypoints: Iterable[ArmWaypoint],
    scene: Any,
    *,
    scene_for_exclusions: Callable[[tuple[str, ...]], Any],
    base_xy: tuple[float, float],
    base_yaw: float,
    full_q_current: np.ndarray,
    planning_time: float,
    local_radius_m: float,
    speed_scale: float,
    side: str,
    ik_candidates_per_waypoint: int = 4,
    beam_width: int = 3,
    max_planned_edges: int = 48,
) -> ArmSequencePlanningResult:
    """Jointly select and certify one natural branch through ordered EE goals."""
    from types import SimpleNamespace

    from r1pro_data_gen.methods.manipulation.mplib_path import (
        plan_arm_path,
        retime_and_validate_path,
        validate_reference_trajectory,
    )
    from r1pro_data_gen.methods.manipulation.taskspace import plan_certified_task_path

    ordered = tuple(waypoints)
    if not ordered:
        raise ValueError("arm waypoint planning requires at least one waypoint")
    beam_width = max(1, int(beam_width))
    candidate_limit = max(1, int(ik_candidates_per_waypoint))
    edge_budget = max(1, int(max_planned_edges))
    request_hash = _sequence_request_hash(
        q_current,
        full_q_current,
        base_xy,
        base_yaw,
        scene,
        ordered,
    )
    prefixes = [
        _SequencePrefix(
            candidates=(),
            segments=(),
            reports=(),
            q_last=np.asarray(q_current, dtype=float),
            natural_cost=0.0,
        )
    ]
    rejected: list[ArmSequenceCandidate] = []
    sequence_id = 0
    planned_edges = 0

    for waypoint_id, waypoint in enumerate(ordered):
        expanded: list[_SequencePrefix] = []
        for prefix in prefixes:
            task_space_used = False
            if hasattr(kin, "ik") and hasattr(kin, "fk"):
                for orientation_id, (position, orientation) in enumerate(waypoint.poses):
                    if planned_edges >= edge_budget:
                        break
                    edge_scene = scene_for_exclusions(waypoint.exclude_objects)
                    certified = plan_certified_task_path(
                        planner,
                        kin,
                        prefix.q_last,
                        np.asarray(position, dtype=float),
                        np.asarray(orientation, dtype=float),
                        edge_scene,
                        base_xy=base_xy,
                        base_yaw=base_yaw,
                        full_q_current=full_q_current,
                        speed_scale=float(
                            waypoint.speed_scale
                            if waypoint.speed_scale is not None
                            else speed_scale
                        ),
                        side=side,
                    )
                    planned_edges += 1
                    if not certified.get("success"):
                        continue
                    q_goal = np.asarray(certified["position"][-1], dtype=float)
                    candidate = _waypoint_candidate(
                        kin,
                        SimpleNamespace(
                            q_arm=q_goal,
                            position_error=0.0,
                            rotation_error=0.0,
                        ),
                        prefix.q_last,
                        waypoint_id,
                        orientation_id,
                        orientation_id,
                    )
                    report = {
                        "waypoint": waypoint.name,
                        "candidate_id": candidate.candidate_id,
                        "orientation_id": orientation_id,
                        "exclude_objects": list(waypoint.exclude_objects),
                        "status": "TaskSpaceVerified",
                        "success": True,
                        "failure_stage": None,
                        "reason": None,
                    }
                    expanded.append(
                        _SequencePrefix(
                            candidates=prefix.candidates + (candidate,),
                            segments=prefix.segments + (dict(certified),),
                            reports=prefix.reports + (report,),
                            q_last=q_goal,
                            natural_cost=prefix.natural_cost
                            + float(candidate.score[0]) / _SCORE_SCALE,
                        )
                    )
                    task_space_used = True
            if task_space_used:
                continue
            generated: dict[tuple[int, ...], tuple[Any, int]] = {}
            for orientation_id, (position, orientation) in enumerate(waypoint.poses):
                solutions = _continuous_waypoint_ik_candidates(
                    kin,
                    np.asarray(position, dtype=float),
                    np.asarray(orientation, dtype=float),
                    prefix.q_last,
                    candidate_limit,
                )
                for solution in solutions:
                    key = tuple(
                        int(round(float(value) * 100_000.0))
                        for value in solution.q_arm
                    )
                    previous = generated.get(key)
                    if previous is None or (
                        float(solution.position_error + solution.rotation_error)
                        < float(previous[0].position_error + previous[0].rotation_error)
                    ):
                        generated[key] = (solution, orientation_id)
            candidates = [
                _waypoint_candidate(
                    kin,
                    solution,
                    prefix.q_last,
                    waypoint_id,
                    candidate_id,
                    orientation_id,
                )
                for candidate_id, (solution, orientation_id) in enumerate(
                    sorted(
                        generated.values(),
                        key=lambda item: (
                            int(item[1]),
                            *(int(round(float(value) * 100_000.0)) for value in item[0].q_arm),
                        ),
                    )
                )
            ]
            candidates.sort(key=lambda item: item.score)
            candidates = candidates[:candidate_limit]
            if not candidates:
                rejected.append(
                    ArmSequenceCandidate(
                        sequence_id=sequence_id,
                        waypoint_candidates=prefix.candidates,
                        segment_reports=prefix.reports,
                        constraints=ConstraintReport(
                            False,
                            "waypoint_ik",
                            (f"{waypoint.name}: no IK candidate",),
                        ),
                    )
                )
                sequence_id += 1
                continue
            edge_scene = scene_for_exclusions(waypoint.exclude_objects)
            for candidate in candidates:
                if planned_edges >= edge_budget:
                    rejected.append(
                        ArmSequenceCandidate(
                            sequence_id=sequence_id,
                            waypoint_candidates=prefix.candidates + (candidate,),
                            segment_reports=prefix.reports,
                            constraints=ConstraintReport(
                                False,
                                "budget_exhausted",
                                (f"edge budget {edge_budget} exhausted",),
                            ),
                        )
                    )
                    sequence_id += 1
                    continue
                q_goal = np.asarray(candidate.q_goal, dtype=float)
                output = plan_arm_path(
                    planner,
                    prefix.q_last,
                    q_goal,
                    edge_scene,
                    base_xy=base_xy,
                    base_yaw=base_yaw,
                    planning_time=float(planning_time),
                    kin=kin,
                    side=side,
                    local_radius_m=float(local_radius_m),
                    speed_scale=float(
                        waypoint.speed_scale
                        if waypoint.speed_scale is not None
                        else speed_scale
                    ),
                    mplib_attempts=1,
                    allow_rrt_fallback=True,
                    rrt_connect_mode="second_opinion",
                    full_q_current=full_q_current,
                )
                planned_edges += 1
                report = {
                    "waypoint": waypoint.name,
                    "candidate_id": candidate.candidate_id,
                    "orientation_id": candidate.orientation_id,
                    "exclude_objects": list(waypoint.exclude_objects),
                    "status": str(output.get("status", "unknown")),
                    "success": bool(output.get("success")),
                    "failure_stage": output.get("failure_stage"),
                    "reason": output.get("reason"),
                }
                if not output.get("success"):
                    rejected.append(
                        ArmSequenceCandidate(
                            sequence_id=sequence_id,
                            waypoint_candidates=prefix.candidates + (candidate,),
                            segment_reports=prefix.reports + (report,),
                            constraints=ConstraintReport(
                                False,
                                str(output.get("failure_stage", "segment_planning")),
                                (str(output.get("reason") or output.get("status")),),
                            ),
                        )
                    )
                    sequence_id += 1
                    continue
                expanded.append(
                    _SequencePrefix(
                        candidates=prefix.candidates + (candidate,),
                        segments=prefix.segments + (dict(output),),
                        reports=prefix.reports + (report,),
                        q_last=np.asarray(output["position"][-1], dtype=float),
                        natural_cost=prefix.natural_cost
                        + float(candidate.score[0]) / _SCORE_SCALE,
                    )
                )
        if not expanded:
            status = (
                "budget_exhausted"
                if planned_edges >= edge_budget
                else "no_complete_waypoint_path"
            )
            return ArmSequencePlanningResult(
                success=False,
                status=status,
                reason=f"no verified prefix reached waypoint {waypoint.name}",
                request_hash=request_hash,
                candidates=tuple(rejected),
            )
        expanded.sort(
            key=lambda prefix: (
                _quantize(prefix.natural_cost),
                *(item.score for item in prefix.candidates),
            )
        )
        prefixes = expanded[:beam_width]

    complete: list[ArmSequenceCandidate] = []
    for prefix in prefixes:
        group_outputs: list[dict[str, Any]] = []
        group_start = 0
        while group_start < len(ordered):
            exclusions = ordered[group_start].exclude_objects
            group_end = group_start + 1
            while (
                group_end < len(ordered)
                and ordered[group_end].exclude_objects == exclusions
                and ordered[group_end].contact == ordered[group_start].contact
                and ordered[group_end].speed_scale == ordered[group_start].speed_scale
            ):
                group_end += 1
            paths = [
                np.asarray(prefix.segments[index]["position"], dtype=float)
                for index in range(group_start, group_end)
            ]
            geometric = np.concatenate(
                [paths[0]] + [path[1:] for path in paths[1:]],
                axis=0,
            )
            certified = retime_and_validate_path(
                planner,
                geometric,
                scene_for_exclusions(exclusions),
                base_xy=base_xy,
                base_yaw=base_yaw,
                kin=kin,
                speed_scale=(
                    ordered[group_start].speed_scale
                    if ordered[group_start].speed_scale is not None
                    else speed_scale
                ),
                side=side,
                full_q_current=full_q_current,
            )
            if not certified.get("success"):
                rejected.append(
                    ArmSequenceCandidate(
                        sequence_id=sequence_id,
                        waypoint_candidates=prefix.candidates,
                        segment_reports=prefix.reports,
                        constraints=ConstraintReport(
                            False,
                            str(certified.get("failure_stage", "sequence_validation")),
                            (str(certified.get("reason") or certified.get("status")),),
                        ),
                    )
                )
                sequence_id += 1
                group_outputs = []
                break
            group_outputs.append(certified)
            group_start = group_end
        if not group_outputs:
            continue
        position = np.concatenate(
            [np.asarray(output["position"], dtype=float) for output in group_outputs],
            axis=0,
        )
        dt = float(group_outputs[0]["dt"])
        reference_report = validate_reference_trajectory(
            position,
            side=side,
            dt=dt,
            speed_scale=max(
                float(
                    waypoint.speed_scale
                    if waypoint.speed_scale is not None
                    else speed_scale
                )
                for waypoint in ordered
            ),
        )
        if not reference_report["valid"]:
            rejected.append(
                ArmSequenceCandidate(
                    sequence_id=sequence_id,
                    waypoint_candidates=prefix.candidates,
                    segment_reports=prefix.reports,
                    constraints=ConstraintReport(
                        False,
                        "sequence_reference_limits",
                        tuple(reference_report["reasons"]),
                    ),
                )
            )
            sequence_id += 1
            continue
        velocity = np.asarray(reference_report["velocity"], dtype=float)
        acceleration = np.asarray(reference_report["acceleration"], dtype=float)
        output = {
            "success": True,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "dt": dt,
            "times": np.arange(len(position), dtype=float) * dt,
            "duration": float(max(0, len(position) - 1) * dt),
            "status": "SequenceVerified",
            "reason": None,
        }
        metrics = _sequence_metrics(
            position,
            kin,
            prefix.candidates,
            output["duration"],
        )
        score = (
            _quantize(metrics["naturalness_cost"]),
            _quantize(metrics["normalized_joint_path_length"]),
            _quantize(metrics["ee_path_length_m"]),
            _quantize(metrics["duration_s"]),
            *(
                value
                for candidate in prefix.candidates
                for value in candidate.score
            ),
        )
        complete.append(
            ArmSequenceCandidate(
                sequence_id=sequence_id,
                waypoint_candidates=prefix.candidates,
                segment_reports=prefix.reports,
                constraints=ConstraintReport(True, "verified"),
                metrics=metrics,
                score=score,
                output=output,
            )
        )
        sequence_id += 1
    if not complete:
        return ArmSequencePlanningResult(
            success=False,
            status="all_sequences_invalid_after_smoothing",
            reason="no complete waypoint sequence passed final trajectory validation",
            request_hash=request_hash,
            candidates=tuple(rejected),
        )
    winner = min(complete, key=lambda candidate: candidate.score)
    return ArmSequencePlanningResult(
        success=True,
        status="success",
        reason="selected the unique best verified waypoint sequence within budget",
        request_hash=request_hash,
        candidates=tuple(rejected + complete),
        winner=winner,
    )
