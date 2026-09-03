"""Isaac-hosted closed-loop agent episode (imported only by product scripts)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from r1pro_data_gen.agent.loop import AgentEpisode, AgentLoop
from r1pro_data_gen.data.plan_io import plan_to_dict
from r1pro_data_gen.domain import (
    GoalSpec,
    SceneModel,
    evidence_to_dict,
    goal_spec_sha256,
)
from r1pro_data_gen.evaluation import (
    PredicateVerifier,
    VerificationPolicy,
    VerificationStatus,
)
from r1pro_data_gen.execution import Orchestrator
from r1pro_data_gen.planning.goals.compiler import GoalCompiler
from r1pro_data_gen.agent.skeleton import build_semantic_plan_skeleton
from r1pro_data_gen.planning.llm.providers import DeepSeekClient
from r1pro_data_gen.planning.context.facts import object_names, scene_to_facts
from r1pro_data_gen.robot import R1PRO_ARM_VELOCITY_LIMITS
from r1pro_data_gen.robot.kinematics import R1ProKinematics
from r1pro_data_gen.skills import build_default_registry
from r1pro_data_gen.simulation import EvidenceRecorder


@dataclass(slots=True)
class _AttemptOutcome:
    """In-memory result for one reset-to-terminal agent attempt."""

    index: int
    episode: AgentEpisode
    evidence: Any
    terminal: Any


def run_hosted_agent(
    *,
    adapter: Any,
    scene: SceneModel,
    goal_spec: GoalSpec,
    output_dir: Path,
    instruction: str,
    urdf: Path,
    max_attempts: int = 1,
    max_actions_per_attempt: int = 24,
    max_action_physics_steps: int | None = 60000,
    max_action_seconds: float | None = 600.0,
    feedback_window: int = 3,
    write_json,
    evidence_hz: float = 10.0,
    seed: int = 0,
    device: str = "cuda:0",
    physical_gpu_id: int = 6,
    recorder: Any = None,
    # Compatibility alias for callers written against the old API, where
    # ``max_actions`` was incorrectly used as the outer-attempt budget.
    max_actions: int | None = None,
) -> dict[str, Any]:
    """Run exactly one bounded closed-loop episode on a live adapter.

    ``max_attempts`` is retained as a compatibility argument, but it is
    intentionally restricted to one.  A reset-and-replay retry would turn a
    complete-task rollout into multiple physical episodes and would hide
    failures from the benchmark.  Recovery belongs inside :class:`AgentLoop`,
    which re-observes and replans while preserving the same simulator state.
    Independent benchmark samples create a new adapter/scene process.
    """
    if max_actions is not None:
        if max_actions_per_attempt != 24 and max_actions_per_attempt != max_actions:
            raise ValueError("max_actions conflicts with max_actions_per_attempt")
        max_actions_per_attempt = max_actions
    _validate_budget(max_attempts, "max_attempts")
    if max_attempts != 1:
        raise ValueError(
            "max_attempts must be 1 for a complete physical episode; "
            "use max_actions_per_attempt for bounded in-episode recovery"
        )
    _validate_budget(max_actions_per_attempt, "max_actions_per_attempt")
    if max_action_physics_steps is not None:
        _validate_budget(max_action_physics_steps, "max_action_physics_steps")
    if max_action_seconds is not None and (
        isinstance(max_action_seconds, bool)
        or not isinstance(max_action_seconds, (int, float))
        or not np.isfinite(float(max_action_seconds))
        or float(max_action_seconds) <= 0.0
    ):
        raise ValueError("max_action_seconds must be positive and finite or None")
    if isinstance(feedback_window, bool) or not isinstance(feedback_window, int) or feedback_window < 0:
        raise ValueError("feedback_window must be a non-negative integer")

    lifecycle_path = output_dir / "lifecycle_checkpoint.json"

    def lifecycle(phase: str, **details: Any) -> None:
        payload = {
            "schema_version": 1,
            "phase": phase,
            "details": _json_safe(details),
        }
        try:
            write_json(lifecycle_path, payload)
        except (OSError, TypeError, ValueError):
            # Startup telemetry is best effort and must not affect the run.
            return

    lifecycle("hosted_agent_started")

    compiled = GoalCompiler().compile(goal_spec, scene)
    write_json(output_dir / "goal_contract.json", compiled.to_dict())
    lifecycle("goal_compiled", contract_hash=compiled.contract_hash)
    kin = R1ProKinematics(str(urdf))
    lifecycle("kinematics_ready")
    registry = build_default_registry(kin, np.asarray(R1PRO_ARM_VELOCITY_LIMITS))
    lifecycle("skill_registry_ready", skill_count=len(registry.agent_descriptions()))
    facts = scene_to_facts(scene, kinematics=kin)
    # Operator debug only. Never send this mapping to the agent prompt.
    plan_skeleton = build_semantic_plan_skeleton(
        goal_spec,
        skill_catalogue=registry.agent_descriptions(),
    )
    write_json(output_dir / "plan_skeleton.json", plan_skeleton)
    lifecycle("semantic_skeleton_ready", step_count=len(plan_skeleton.get("steps", ())))
    verifier = PredicateVerifier()
    policy = VerificationPolicy()
    needs_collision = any(
        item.predicate == "collision_free"
        for item in (*goal_spec.required, *goal_spec.invariants)
    )

    provider = DeepSeekClient.from_env()
    lifecycle("provider_ready", provider=getattr(provider, "name", ""), model=getattr(provider, "model", ""))
    attempt_history: list[dict[str, Any]] = []
    outcomes: list[_AttemptOutcome] = []
    episode_index = 1
    lifecycle("episode_starting", episode=episode_index, recovery_mode="in_episode_closed_loop")
    try:
        outcome = _run_attempt(
            adapter=adapter,
            scene=scene,
            goal_spec=goal_spec,
            output_dir=output_dir,
            write_json=write_json,
            max_action_physics_steps=max_action_physics_steps,
            max_action_seconds=max_action_seconds,
            instruction=instruction,
            registry=registry,
            facts=facts,
            provider=provider,
            verifier=verifier,
            policy=policy,
            max_actions=max_actions_per_attempt,
            feedback=(),
            feedback_window=feedback_window,
            evidence_hz=evidence_hz,
            needs_collision=needs_collision,
            recorder=recorder,
            index=episode_index,
        )
    except BaseException as exc:
        # Preserve the exact boundary failure before re-raising.  This is
        # especially useful for Isaac/Omniverse shutdowns, which can surface
        # as SystemExit or another BaseException after app shutdown starts.
        lifecycle(
            "episode_exception",
            episode=episode_index,
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        raise
    lifecycle("episode_finished", episode=episode_index, status=outcome.episode.status)
    outcomes.append(outcome)
    attempt_payload = _attempt_payload(
        outcome,
        goal_spec_hash=goal_spec_sha256(goal_spec),
        contract_hash=compiled.contract_hash,
    )
    attempt_dir = output_dir / "attempt_01"
    write_json(attempt_dir / "evidence.json", evidence_to_dict(outcome.evidence))
    write_json(attempt_dir / "action_trace.json", outcome.episode.to_json())
    write_json(attempt_dir / "plan.json", plan_to_dict(outcome.episode.to_plan(scene.name)))
    write_json(attempt_dir / "result.json", attempt_payload)
    attempt_history.append(_attempt_summary(outcome))

    selected = outcome
    terminal = selected.terminal
    evidence = selected.evidence
    episode = selected.episode
    succeeded = _attempt_is_physically_acceptable(selected)
    total_action_count = len(episode.steps)
    selected_steps = episode.to_json()["steps"]
    payload = {
        "result": "passed" if succeeded else "failed",
        "status": "succeeded" if succeeded else (
            "failed" if terminal.status.value == "failed" else episode.status
        ),
        "reason": None if succeeded else (episode.reason or terminal.failure_reason),
        "evaluation_mode": "goal_spec",
        "goal_spec_hash": goal_spec_sha256(goal_spec),
        "goal_contract_hash": compiled.contract_hash,
        # Kept for compatibility with existing artifact consumers.  This is
        # the count of complete physical episodes represented by this result,
        # never an in-episode retry count.
        "attempts": 1,
        "episodes": 1,
        "success_attempt": 1 if succeeded else None,
        "recovery_mode": "in_episode_closed_loop",
        "reset_recovery_used": False,
        "action_count": len(episode.steps),
        "total_action_count": total_action_count,
        "episode_status": episode.status,
        "attempt_history": attempt_history,
        "plan_skeleton": plan_skeleton,
        "evaluation": {
            "status": terminal.status.value,
            "failure_reason": terminal.failure_reason,
            "predicates": [_json_safe(item) for item in terminal.predicates],
            "evidence_complete": terminal.evidence_complete,
            "evidence_coverage_complete": evidence.complete,
            "stage_success_complete": evidence.stage_success_complete,
            "collision_observation_complete": evidence.collision_observation_complete,
        },
        "steps": selected_steps,
        "gpu_logical_device": device,
        "physical_gpu_id": physical_gpu_id,
        "seed": seed,
        "provider": getattr(provider, "name", ""),
        "model": getattr(provider, "model", ""),
        "robot_asset": scene.robot.asset,
        "action_budget": {
            "max_physics_steps": max_action_physics_steps,
            "max_wall_time_s": max_action_seconds,
        },
    }
    write_json(output_dir / "evidence.json", evidence_to_dict(evidence))
    write_json(output_dir / "action_trace.json", episode.to_json())
    write_json(output_dir / "plan.json", plan_to_dict(episode.to_plan(scene.name)))
    return payload


def _run_attempt(
    *,
    adapter: Any,
    scene: SceneModel,
    goal_spec: GoalSpec,
    output_dir: Path,
    write_json: Any,
    max_action_physics_steps: int | None,
    max_action_seconds: float | None,
    instruction: str,
    registry: Any,
    facts: Mapping[str, Any],
    provider: Any,
    verifier: PredicateVerifier,
    policy: VerificationPolicy,
    max_actions: int,
    feedback: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    feedback_window: int,
    evidence_hz: float,
    needs_collision: bool,
    recorder: Any,
    index: int,
) -> _AttemptOutcome:
    current_stage = "__initial__"
    checkpoint_path = output_dir / f"attempt_{index:02d}" / "live_checkpoint.json"
    checkpoint_started = time.monotonic()
    checkpoint_state: dict[str, Any] = {
        "schema_version": 1,
        "attempt": index,
        "phase": "attempt_started",
        "wall_time_s": 0.0,
    }
    last_checkpoint_wall = checkpoint_started

    def _sim_time() -> float | None:
        try:
            value = float(adapter.sim.current_time)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None
        return value if np.isfinite(value) else None

    def checkpoint(event: Mapping[str, Any]) -> None:
        nonlocal checkpoint_state
        checkpoint_state.update(_json_safe(dict(event)))
        checkpoint_state["schema_version"] = 1
        checkpoint_state["attempt"] = index
        checkpoint_state["stage"] = current_stage
        sim_time = _sim_time()
        if sim_time is not None:
            checkpoint_state["sim_time_s"] = sim_time
        checkpoint_state["wall_time_s"] = round(
            time.monotonic() - checkpoint_started, 3
        )
        try:
            write_json(checkpoint_path, checkpoint_state)
        except (OSError, TypeError, ValueError):
            # The checkpoint is best effort and must not affect the physical
            # action or its acceptance result.
            return

    checkpoint({"phase": "attempt_started"})
    evidence_recorder = EvidenceRecorder(adapter, scene)
    evidence_interval_s = 1.0 / float(evidence_hz)
    evidence_recorder.capture(float(adapter.sim.current_time), stage=current_stage)

    def step_hook() -> None:
        nonlocal last_checkpoint_wall
        if recorder is not None:
            recorder.step_hook()
        sim_time = float(adapter.sim.current_time)
        evidence_recorder.capture_if_due(
            sim_time,
            stage=current_stage,
            min_interval_s=evidence_interval_s,
        )
        now = time.monotonic()
        if now - last_checkpoint_wall >= 2.0:
            last_checkpoint_wall = now
            checkpoint({"phase": "heartbeat"})

    def stage_hook(stage_name: str) -> None:
        nonlocal current_stage
        current_stage = stage_name
        checkpoint({"phase": "stage_started", "stage": stage_name})
        evidence_recorder.capture(float(adapter.sim.current_time), stage=stage_name)

    def stage_end_hook(stage_name: str, success: bool) -> None:
        checkpoint(
            {
                "phase": "stage_completed",
                "stage": stage_name,
                "stage_success": bool(success),
            }
        )
        evidence_recorder.capture(float(adapter.sim.current_time), stage=stage_name)
        evidence_recorder.finish_stage(
            float(adapter.sim.current_time), stage_name, success=success
        )

    def progress():
        bundle = evidence_recorder.finish(complete=False, expected_stages=())
        return verifier.progress(goal_spec, bundle, policy)

    orchestrator = Orchestrator(
        adapter,
        registry,
        scene=scene,
        step_hook=step_hook,
        stage_hook=stage_hook,
        stage_end_hook=stage_end_hook,
        max_action_physics_steps=max_action_physics_steps,
        max_action_seconds=max_action_seconds,
    )
    episode = AgentLoop(
        provider,
        orchestrator,
        catalog=registry.agent_descriptions(),
        scene_object_names=object_names(facts),
        scene_facts=facts,
        goal_spec=goal_spec,
        max_actions=max_actions,
        progress_fn=progress,
        registry=registry,
        prior_feedback=feedback,
        feedback_window=feedback_window,
        checkpoint_fn=checkpoint,
    ).run(task_description=instruction)
    # Invalid provider actions never enter the orchestrator and therefore do
    # not create an observable stage window. They remain in the action trace.
    stage_names = tuple(
        f"step_{step.index:02d}_{step.skill}"
        for step in episode.steps
        if step.skill != "invalid_action"
    )
    evidence = evidence_recorder.finish(
        # Coverage is an observation property, not a skill-success property.
        # This lets the physical verifier decide whether a later action reached
        # the goal despite an earlier local skill failure.
        complete=True,
        expected_stages=stage_names,
        require_collision_observation=needs_collision,
    )
    terminal = verifier.verify(goal_spec, evidence, policy)
    return _AttemptOutcome(index=index, episode=episode, evidence=evidence, terminal=terminal)


def _attempt_is_physically_acceptable(outcome: _AttemptOutcome) -> bool:
    return bool(
        outcome.terminal.status is VerificationStatus.SUCCEEDED
        and outcome.terminal.evidence_complete
        and outcome.evidence.complete
    )


def _attempt_summary(outcome: _AttemptOutcome) -> dict[str, Any]:
    return {
        "attempt": outcome.index,
        "episode_status": outcome.episode.status,
        "goal_status": outcome.terminal.status.value,
        "goal_satisfied": outcome.terminal.status is VerificationStatus.SUCCEEDED,
        "evidence_coverage_complete": outcome.evidence.complete,
        "stage_success_complete": outcome.evidence.stage_success_complete,
        "action_count": len(outcome.episode.steps),
        "reason": outcome.episode.reason or outcome.terminal.failure_reason,
    }


def _attempt_feedback(outcome: _AttemptOutcome) -> dict[str, Any]:
    """Return bounded observed facts for the next reset attempt."""
    last = outcome.episode.steps[-1] if outcome.episode.steps else None
    return {
        "attempt": outcome.index,
        "episode_status": outcome.episode.status,
        "goal_status": outcome.terminal.status.value,
        "failure_reason": outcome.episode.reason or outcome.terminal.failure_reason,
        "action_count": len(outcome.episode.steps),
        "completed_skills": [
            step.skill for step in outcome.episode.steps if step.result.success
        ],
        "failed_skills": [
            step.skill for step in outcome.episode.steps if not step.result.success
        ],
        "last_action": None
        if last is None
        else {
            "skill": last.skill,
            "parameters": dict(last.parameters),
            "success": bool(last.result.success),
            "failure_code": last.result.details.get("failure_code")
            or last.result.metrics.get("failure_code"),
        },
    }


def _attempt_payload(
    outcome: _AttemptOutcome,
    *,
    goal_spec_hash: str,
    contract_hash: str,
) -> dict[str, Any]:
    terminal = outcome.terminal
    episode = outcome.episode
    return {
        "attempt": outcome.index,
        "result": "passed" if _attempt_is_physically_acceptable(outcome) else "failed",
        "status": terminal.status.value,
        "episode_status": episode.status,
        "reason": episode.reason or terminal.failure_reason,
        "evaluation_mode": "goal_spec",
        "goal_spec_hash": goal_spec_hash,
        "goal_contract_hash": contract_hash,
        "action_count": len(episode.steps),
        "evaluation": {
            "status": terminal.status.value,
            "failure_reason": terminal.failure_reason,
            "predicates": [_json_safe(item) for item in terminal.predicates],
            "evidence_complete": terminal.evidence_complete,
            "evidence_coverage_complete": outcome.evidence.complete,
            "stage_success_complete": outcome.evidence.stage_success_complete,
            "collision_observation_complete": outcome.evidence.collision_observation_complete,
        },
        "steps": episode.to_json()["steps"],
    }


def _validate_budget(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
