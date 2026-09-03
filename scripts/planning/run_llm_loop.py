"""Bounded external-LLM failure feedback loop for R1Pro task planning.

This script is the single closed-loop entrypoint: it asks an external LLM for a
plan, replays that plan through ``scripts/tasks/run_plan.py`` (a GPU subprocess), and
on failure extracts a small redacted :class:`Feedback` record which is fed back
into the next request. The loop is bounded by ``max_attempts`` (default 10) and
a sliding feedback window of ``feedback_window`` (default 3) rounds, plus two
independent protection counters:

- ``provider_failures``: two consecutive planner provider/transport failures
  stop the loop with ``provider unavailable``.
- ``replay_runtime_failures``: two consecutive rounds where the replay produced
  no structured result stop the loop with ``replay unavailable``.

The module is import-safe (no ``isaaclab.app`` import) so unit tests load it via
``importlib.util`` and inject fake planners and replay callbacks.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from r1pro_data_gen.agent.feedback import FactFeedback, Feedback, extract_failure_feedback
from r1pro_data_gen.data.plan_io import plan_from_dict, plan_to_dict, save_plan
from r1pro_data_gen.domain import (
    goal_spec_sha256,
    goal_spec_to_dict,
    parse_goal_spec,
)
from r1pro_data_gen.planning import (
    GoalPlanner,
    GoalPlanningRequest,
    LLMTaskPlanner,
    TaskPlanner,
    TaskPlanningRequest,
    TaskPlanningResult,
)
from r1pro_data_gen.planning.goals.compiler import GoalCompiler
from r1pro_data_gen.planning.llm.providers import DeepSeekClient
from r1pro_data_gen.planning.context.facts import scene_to_facts
from r1pro_data_gen.robot import R1PRO_ARM_VELOCITY_LIMITS
from r1pro_data_gen.robot.kinematics import R1ProKinematics
from r1pro_data_gen.data.scenes import load_scene_data, write_scene_yaml
from r1pro_data_gen.skills import build_default_registry
from r1pro_data_gen.tasks import load_task_spec
from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS

_REPLAY_TIMEOUT_S = 7200
_MAX_RUNTIME_ERROR_CHARS = 2000
_SUCCESS_EXIT = 0
_FAILURE_EXIT = 1
_INTERRUPT_EXIT = 130
_USAGE_EXIT = 2

_PROVIDER_FAILURE_MARKERS = (
    "deepseek",
    "http",
    "transport",
    "connection",
    "authentication",
    "timeout",
    "provider",
    "api key",
    "api_key",
    "apikey",
    "authorization",
    "token",
    "url",
    "socket",
)

# Keys dropped recursively before persisting raw provenance / planner results so
# provider secrets never reach disk (matches the "no API key on disk" constraint).
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "api-key",
    "x-api-key",
    "token",
    "authorization",
    "auth",
    "secret",
    "password",
    "passwd",
    "credential",
    "access_key",
    "secret_key",
    "cookie",
)

# Redact embedded secret-looking values in arbitrary strings (defense in depth
# when the raw response is a plain JSON text rather than a structured mapping).
_SECRET_VALUE_PATTERN = re.compile(r"\b(?:sk|AKIA|ghp_|Bearer\s+)[-_A-Za-z0-9]{8,}\b")


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Immutable configuration for one closed-loop run.

    ``task`` is the instruction resolved from the public TaskSpec. The
    ``task_spec_path`` is retained with the configuration so every replay is
    traceable to the same data-defined task identity.
    """

    scene: Mapping[str, Any]
    task: str
    urdf: Path
    output_dir: Path
    task_spec_path: Path | None = None
    goal_spec: Path | None = None
    initial_feedback: Path | None = None
    initial_plan: Path | None = None
    max_attempts: int = 10
    feedback_window: int = 3
    fps: int = DEFAULT_VIDEO_FPS
    width: int = 960
    height: int = 544
    physical_gpu_id: int | None = 6
    device: str = "cuda:0"
    seed: int = 0
    stream_replay_logs: bool = False

    def __post_init__(self) -> None:
        _require_positive_int(self.max_attempts, "max_attempts")
        _require_positive_int(self.feedback_window, "feedback_window")
        _require_positive_int(self.fps, "fps")
        if self.fps != DEFAULT_VIDEO_FPS:
            raise ValueError(
                f"video fps is fixed at {DEFAULT_VIDEO_FPS}; got {self.fps}"
            )
        _require_positive_int(self.width, "width")
        _require_positive_int(self.height, "height")
        if self.physical_gpu_id is not None and (
            isinstance(self.physical_gpu_id, bool)
            or not isinstance(self.physical_gpu_id, int)
            or self.physical_gpu_id < 0
        ):
            raise ValueError("physical_gpu_id must be a non-negative integer or None")
        if self.physical_gpu_id is not None and self.physical_gpu_id != 6:
            raise ValueError("this project is pinned to physical GPU 6")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if self.task_spec_path is None:
            raise ValueError(
                "task_spec_path is required; construct LoopConfig from a TaskSpec"
            )
        if not Path(self.task_spec_path).is_file():
            raise FileNotFoundError(f"task spec does not exist: {self.task_spec_path}")
        task_spec = load_task_spec(self.task_spec_path)
        if not isinstance(self.scene, Mapping) or not self.scene:
            raise TypeError("LoopConfig scene must be an embedded scene mapping")
        if dict(self.scene) != task_spec.scene:
            raise ValueError("LoopConfig scene must match the TaskSpec scene data")
        if self.task != task_spec.instruction:
            raise ValueError("LoopConfig task must match the TaskSpec instruction")
        task_spec.require_human_verified()


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """Result of one GPU replay subprocess."""

    available: bool
    result: Mapping[str, Any] | None = None
    runtime_error: str | None = None
    returncode: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool")
        if self.available:
            if not isinstance(self.result, Mapping):
                raise TypeError("available replay requires a result mapping")
            if self.returncode is not None and not isinstance(self.returncode, int):
                raise TypeError("returncode must be an integer or None")
        else:
            if self.runtime_error is not None and not isinstance(self.runtime_error, str):
                raise TypeError("runtime_error must be a string or None")
            if self.returncode is not None and not isinstance(self.returncode, int):
                raise TypeError("returncode must be an integer or None")
            if self.result is not None:
                raise TypeError("unavailable replay must not carry a result")


@dataclass(frozen=True, slots=True)
class LoopOutcome:
    """Final result of a closed-loop run."""

    status: str
    attempts: int
    success_attempt: int | None
    reason: str
    last_failure: Feedback | None = None
    goal_spec_hash: str | None = None
    goal_contract_hash: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed"}:
            raise ValueError(f"unsupported loop status: {self.status!r}")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        if self.success_attempt is not None and (
            isinstance(self.success_attempt, bool)
            or not isinstance(self.success_attempt, int)
            or self.success_attempt < 1
        ):
            raise ValueError("success_attempt must be a positive integer or None")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.goal_spec_hash is not None:
            if not isinstance(self.goal_spec_hash, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", self.goal_spec_hash
            ):
                raise ValueError("goal_spec_hash must be a SHA-256 hex digest or None")
            object.__setattr__(self, "goal_spec_hash", self.goal_spec_hash.lower())
        if self.goal_contract_hash is not None:
            if not isinstance(self.goal_contract_hash, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", self.goal_contract_hash
            ):
                raise ValueError("goal_contract_hash must be a SHA-256 hex digest or None")
            object.__setattr__(self, "goal_contract_hash", self.goal_contract_hash.lower())
        if self.status == "succeeded" and self.success_attempt is None:
            raise ValueError("succeeded outcome requires success_attempt")

    def to_json(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "attempts": self.attempts,
            "success_attempt": self.success_attempt,
            "reason": self.reason,
            "last_failure": self.last_failure.to_json() if self.last_failure is not None else None,
            "goal_spec_hash": self.goal_spec_hash,
        }
        if self.goal_contract_hash is not None:
            payload["goal_contract_hash"] = self.goal_contract_hash
        return payload


def build_request(
    *,
    task_description: str,
    scene_facts: Mapping[str, Any],
    skill_catalog: Sequence[Mapping[str, Any]],
    feedbacks: Iterable[Any],
    metadata: Mapping[str, Any],
    feedback_window: int,
    active_feedback: Feedback | None = None,
    previous_plan: Mapping[str, Any] | None = None,
    goal_spec: Mapping[str, Any] | None = None,
    goal_spec_hash: str | None = None,
    goal_contract_hash: str | None = None,
) -> TaskPlanningRequest:
    """Build the planner request with only the ``feedback_window`` recent items.

    ``feedbacks`` may contain :class:`Feedback` objects or already-bounded
    strings. Typed feedback is serialized with the compact planner protocol;
    the full Markdown rendering remains an artifact for human debugging.
    """
    if isinstance(feedback_window, bool) or not isinstance(feedback_window, int) or feedback_window < 1:
        raise ValueError("feedback_window must be a positive integer")
    recent = deque(feedbacks, maxlen=feedback_window)
    serialized = [_serialize_feedback(item) for item in recent]
    constraints: dict[str, Any] = {"failure_feedback": list(serialized)}
    if previous_plan is not None:
        # The last validated semantic plan is bounded, JSON-only context for a
        # minimal repair.  Feedback describes what happened; this preserves
        # the exact stages/parameters that produced that observation without
        # turning the feedback boundary into a task-specific recipe.
        if not isinstance(previous_plan, Mapping):
            raise TypeError("previous_plan must be a mapping or None")
        constraints["previous_plan"] = previous_plan
    if active_feedback is not None:
        # Validation/provider churn must not erase the last physical failure.
        # Keep that one runtime record in a separate channel from the bounded
        # conversational window so measured repairs remain active until the
        # next physical replay supersedes them.
        constraints["active_runtime_feedback"] = [_serialize_feedback(active_feedback)]
    return TaskPlanningRequest(
        task_description=task_description,
        scene_facts=scene_facts,
        skill_catalog=tuple(skill_catalog),
        constraints=constraints,
        metadata=metadata,
        goal_spec=goal_spec,
        goal_spec_hash=goal_spec_hash,
        goal_contract_hash=goal_contract_hash,
    )


def run_loop(
    config: LoopConfig,
    *,
    planner: TaskPlanner | None = None,
    replay_runner: Callable[[Path, Path, LoopConfig], ReplayOutcome] | None = None,
    skill_catalog: Sequence[Mapping[str, Any]] | None = None,
) -> LoopOutcome:
    """Run the bounded closed loop and return its final outcome.

    ``planner`` and ``replay_runner`` are injectable for tests. ``skill_catalog``
    is an optional prebuilt catalogue that bypasses URDF/registry construction
    (which requires the R1Pro assets); when omitted it is built the same way
    ``generate_llm_plan.py`` does.
    """
    scene = load_scene_data(
        config.scene,
        source=config.task_spec_path or "<loop scene>",
    )
    goal_spec = None
    goal_spec_hash = None
    goal_contract_hash = None
    # The URDF is required for the real closed loop: it builds the skill
    # catalogue and, when present, annotates navigation candidates with IK
    # reachability.  Tests inject a placeholder path, so a missing/invalid
    # URDF degrades to no reachability annotation instead of blocking.
    kin = None
    try:
        if Path(config.urdf).is_file() and Path(config.urdf).stat().st_size > 0:
            kin = R1ProKinematics(str(config.urdf))
    except Exception:
        kin = None
    if config.goal_spec is not None:
        try:
            payload = json.loads(Path(config.goal_spec).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"goal spec is unreadable: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("goal spec JSON must contain an object")
        goal_spec = parse_goal_spec(payload, scene)
        goal_spec_hash = goal_spec_sha256(goal_spec)
        compiled_goal = GoalCompiler().compile(goal_spec, scene)
        goal_contract_hash = compiled_goal.contract_hash
        frozen_goal_path = Path(config.output_dir) / "goal_spec.json"
        frozen_goal_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_goal_path.write_text(
            json.dumps(goal_spec_to_dict(goal_spec), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (Path(config.output_dir) / "goal_contract.json").write_text(
            json.dumps(compiled_goal.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if kin is None:
            raise FileNotFoundError(f"URDF does not exist: {config.urdf}")
        registry = build_default_registry(kin, np.asarray(R1PRO_ARM_VELOCITY_LIMITS))
        catalog = registry.llm_descriptions()
    else:
        catalog = list(skill_catalog)
    scene_facts = scene_to_facts(scene, kinematics=kin)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if planner is None:
        planner = LLMTaskPlanner(DeepSeekClient.from_env())
    if replay_runner is None:
        replay_runner = run_replay

    provider_failures = 0
    replay_runtime_failures = 0
    feedback_window: deque[str] = deque(maxlen=config.feedback_window)
    last_failure: Feedback | None = None
    active_runtime_feedback = (
        _load_feedback(config.initial_feedback)
        if config.initial_feedback is not None
        else None
    )
    if active_runtime_feedback is not None:
        _validate_feedback_goal_spec_hash(
            active_runtime_feedback,
            goal_spec_hash=goal_spec_hash,
            generic=goal_spec_hash is not None,
        )
        _validate_feedback_goal_contract_hash(
            active_runtime_feedback,
            goal_contract_hash=goal_contract_hash,
            generic=goal_spec_hash is not None,
        )
        feedback_window.append(_serialize_feedback(active_runtime_feedback))

    current_attempt = 0
    previous_plan: Mapping[str, Any] | None = None
    if config.initial_plan is not None:
        try:
            raw_previous_plan = json.loads(
                Path(config.initial_plan).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"initial plan is unreadable: {exc}") from exc
        if not isinstance(raw_previous_plan, Mapping):
            raise ValueError("initial plan JSON must contain a plan object")
        try:
            parsed_previous_plan = plan_from_dict(raw_previous_plan)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"initial plan is invalid: {exc}") from exc
        _validate_plan_goal_spec_hash(
            parsed_previous_plan,
            goal_spec_hash=goal_spec_hash,
            generic=goal_spec_hash is not None,
        )
        _validate_plan_goal_contract_hash(
            parsed_previous_plan,
            goal_contract_hash=goal_contract_hash,
            generic=goal_spec_hash is not None,
        )
        previous_plan = plan_to_dict(parsed_previous_plan)
    try:
        for attempt in range(1, config.max_attempts + 1):
            current_attempt = attempt
            print(
                f"[llm-loop] attempt {attempt}/{config.max_attempts}: planning",
                flush=True,
            )
            request = build_request(
                task_description=config.task,
                scene_facts=scene_facts,
                skill_catalog=catalog,
                feedbacks=feedback_window,
                metadata={
                    "scene_name": scene.name,
                    "loop_attempt": attempt,
                    "goal_spec_hash": goal_spec_hash,
                    "goal_contract_hash": goal_contract_hash,
                },
                feedback_window=config.feedback_window,
                active_feedback=active_runtime_feedback,
                previous_plan=previous_plan,
                goal_spec=goal_spec_to_dict(goal_spec) if goal_spec is not None else None,
                goal_spec_hash=goal_spec_hash,
                goal_contract_hash=goal_contract_hash,
            )
            attempt_dir = _attempt_dir(output_dir, attempt)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            _write_planner_request(attempt_dir, request)
            result = planner.plan(request)

            if result.status == "planned":
                provider_failures = 0
                if result.plan is None:
                    raise ValueError("planned result must carry a plan")
                _validate_plan_goal_spec_hash(
                    result.plan,
                    goal_spec_hash=goal_spec_hash,
                    generic=goal_spec_hash is not None,
                )
                _validate_plan_goal_contract_hash(
                    result.plan,
                    goal_contract_hash=goal_contract_hash,
                    generic=goal_spec_hash is not None,
                )
                plan_path = attempt_dir / "plan.json"
                save_plan(result.plan, plan_path)
                _write_raw_provenance(
                    attempt_dir,
                    result,
                    goal_spec_hash=goal_spec_hash,
                    goal_contract_hash=goal_contract_hash,
                )
                previous_plan = plan_to_dict(result.plan)
                print(
                    f"[llm-loop] attempt {attempt}: replaying plan in Isaac Sim",
                    flush=True,
                )
                replay_outcome = replay_runner(plan_path, attempt_dir, config)
                if replay_outcome.available:
                    replay_runtime_failures = 0
                    _validate_replay_goal_spec_hash(
                        replay_outcome.result,
                        goal_spec_hash=goal_spec_hash,
                        generic=goal_spec_hash is not None,
                    )
                    _validate_replay_goal_contract_hash(
                        replay_outcome.result,
                        goal_contract_hash=goal_contract_hash,
                        generic=goal_spec_hash is not None,
                    )
                    if _replay_succeeded(
                        replay_outcome.result,
                        require_goal_spec=config.goal_spec is not None,
                        goal_spec_hash=goal_spec_hash,
                    ):
                        outcome = LoopOutcome(
                            status="succeeded",
                            attempts=attempt,
                            success_attempt=attempt,
                            reason="replay passed",
                            last_failure=last_failure,
                            goal_spec_hash=goal_spec_hash,
                            goal_contract_hash=goal_contract_hash,
                        )
                        write_loop_result(output_dir, outcome)
                        return outcome
                    feedback = _feedback_from_replay_result(
                        config,
                        scene_facts,
                        result.plan,
                        replay_outcome,
                        attempt,
                        goal_spec_hash=goal_spec_hash or "",
                        goal_contract_hash=goal_contract_hash or "",
                    )
                    active_runtime_feedback = feedback
                else:
                    replay_runtime_failures += 1
                    feedback = _runtime_failure_feedback(
                        attempt,
                        replay_outcome,
                        goal_spec_hash=goal_spec_hash or "",
                        goal_contract_hash=goal_contract_hash or "",
                    )
                last_failure = _archive_failure(attempt_dir, feedback_window, feedback)
                if replay_runtime_failures >= 2:
                    outcome = LoopOutcome(
                        status="failed",
                        attempts=attempt,
                        success_attempt=None,
                        reason="replay unavailable",
                        last_failure=feedback,
                        goal_spec_hash=goal_spec_hash,
                        goal_contract_hash=goal_contract_hash,
                    )
                    write_loop_result(output_dir, outcome)
                    return outcome
            elif result.status == "unsupported":
                provider_failures = 0
                feedback = extract_failure_feedback(
                    task_description=config.task,
                    scene_facts=scene_facts,
                    plan=None,
                    execution=None,
                    measurements=None,
                    evaluation=None,
                    attempt=attempt,
                    failure_type="unsupported",
                    validator_error=result.reason,
                    goal_spec_hash=goal_spec_hash or "",
                    goal_contract_hash=goal_contract_hash or "",
                )
                _write_planner_result(
                    attempt_dir,
                    result,
                    feedback,
                    goal_spec_hash=goal_spec_hash,
                    goal_contract_hash=goal_contract_hash,
                )
                last_failure = _archive_failure(attempt_dir, feedback_window, feedback)
            else:  # failed
                if _is_provider_failure(result.reason):
                    provider_failures += 1
                    failure_type = "provider"
                    provider_error, validator_error = result.reason, None
                else:
                    provider_failures = 0
                    failure_type = "validator"
                    provider_error, validator_error = None, result.reason
                feedback = extract_failure_feedback(
                    task_description=config.task,
                    scene_facts=scene_facts,
                    plan=None,
                    execution=None,
                    measurements=None,
                    evaluation=None,
                    attempt=attempt,
                    failure_type=failure_type,
                    provider_error=provider_error,
                    validator_error=validator_error,
                    goal_spec_hash=goal_spec_hash or "",
                    goal_contract_hash=goal_contract_hash or "",
                )
                _write_planner_result(
                    attempt_dir,
                    result,
                    feedback,
                    goal_spec_hash=goal_spec_hash,
                    goal_contract_hash=goal_contract_hash,
                )
                last_failure = _archive_failure(attempt_dir, feedback_window, feedback)
                if provider_failures >= 2:
                    outcome = LoopOutcome(
                        status="failed",
                        attempts=attempt,
                        success_attempt=None,
                        reason="provider unavailable",
                        last_failure=feedback,
                        goal_spec_hash=goal_spec_hash,
                        goal_contract_hash=goal_contract_hash,
                    )
                    write_loop_result(output_dir, outcome)
                    return outcome
    except KeyboardInterrupt:
        _write_interrupted_result(output_dir, current_attempt, last_failure)
        raise

    outcome = LoopOutcome(
        status="failed",
        attempts=config.max_attempts,
        success_attempt=None,
        reason="max attempts exhausted",
        last_failure=last_failure,
        goal_spec_hash=goal_spec_hash,
        goal_contract_hash=goal_contract_hash,
    )
    write_loop_result(output_dir, outcome)
    return outcome


def run_replay(plan_path: Path, attempt_dir: Path, config: LoopConfig) -> ReplayOutcome:
    """Run ``scripts/tasks/run_plan.py`` in a subprocess and parse its result.

    The caller's ``sys.executable`` is reused so the Isaac Lab interpreter (and
    its AppLauncher) is preserved. A present ``.run_plan_result_written.json``
    marker plus a parseable ``result.json`` proves structured persistence even
    when the replay exits 1 (evaluator failure); a missing marker is a runtime
    failure. No API key is synthesized into the environment.
    """
    runner = PROJECT_ROOT / "scripts" / "tasks" / "run_plan.py"
    if not runner.is_file():
        return ReplayOutcome(available=False, runtime_error=f"replay runner not found: {runner}")
    output_dir = Path(attempt_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.goal_spec is None:
        return ReplayOutcome(
            available=False,
            runtime_error="generic replay requires goal_spec",
        )
    try:
        cmd = build_replay_command(
            plan_path=plan_path,
            goal_spec_path=config.goal_spec,
            config=config,
            attempt_dir=output_dir,
        )
    except ValueError as exc:
        return ReplayOutcome(available=False, runtime_error=str(exc))
    env = dict(os.environ)
    if config.physical_gpu_id is not None and not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = str(config.physical_gpu_id)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_ROOT), env.get("PYTHONPATH", "")])
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=not config.stream_replay_logs,
            text=True,
            timeout=_REPLAY_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        return ReplayOutcome(available=False, runtime_error=f"replay interpreter not found: {exc}")
    except OSError as exc:
        return ReplayOutcome(available=False, runtime_error=f"replay subprocess failed: {exc}")
    except subprocess.TimeoutExpired:
        return ReplayOutcome(
            available=False,
            runtime_error=f"replay timed out after {_REPLAY_TIMEOUT_S}s",
        )

    marker = output_dir / ".run_plan_result_written.json"
    result_path = output_dir / "result.json"
    if marker.is_file() and result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ReplayOutcome(
                available=False,
                runtime_error=f"replay result unreadable: {exc}",
                returncode=proc.returncode,
            )
        if not isinstance(result, dict):
            return ReplayOutcome(
                available=False,
                runtime_error="replay result must be a JSON object",
                returncode=proc.returncode,
            )
        _write_stage_calls(output_dir, result)
        return ReplayOutcome(available=True, result=result, returncode=proc.returncode)

    stderr = (proc.stderr or "")[:_MAX_RUNTIME_ERROR_CHARS]
    return ReplayOutcome(
        available=False,
        runtime_error=(
            f"replay did not write a result marker (exit={proc.returncode}): {stderr}"
        ),
        returncode=proc.returncode,
    )



def build_replay_command(
    *,
    plan_path: Path,
    goal_spec_path: Path | None,
    config: LoopConfig,
    attempt_dir: Path,
) -> list[str]:
    """Build the product replay command from the canonical TaskSpec."""
    if goal_spec_path is None:
        raise ValueError("generic replay requires goal_spec")
    if config.task_spec_path is None:
        raise ValueError("generic replay requires task_spec_path")
    runner = PROJECT_ROOT / "scripts" / "tasks" / "run_plan.py"
    command = [
        sys.executable,
        str(runner),
        "--task", str(config.task_spec_path),
        "--goal-spec", str(goal_spec_path),
        "--plan", str(plan_path),
        "--external-llm-plan",
        "--output-dir", str(attempt_dir),
        "--urdf", str(config.urdf),
        "--fps", str(config.fps),
        "--width", str(config.width),
        "--height", str(config.height),
        "--device", config.device,
        "--seed", str(config.seed),
    ]
    if config.physical_gpu_id is not None:
        command.extend(["--physical-gpu-id", str(config.physical_gpu_id)])
    return command


def write_loop_result(output_dir: Path, outcome: LoopOutcome) -> None:
    """Persist ``loop_result.json`` before any return path of ``run_loop``."""
    path = Path(output_dir) / "loop_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(outcome.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_interrupted_result(
    output_dir: Path, attempt: int, last_failure: Feedback | None
) -> None:
    """Persist a partial ``loop_result.json`` for a Ctrl-C'd run.

    ``LoopOutcome`` only models ``succeeded``/``failed``, so an interrupted run
    writes a distinct ``status: "interrupted"`` record so the artifacts from the
    rounds that did complete are still auditable, then re-raises for ``main()``
    to return 130.
    """
    path = Path(output_dir) / "loop_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "interrupted",
                "attempts": attempt,
                "success_attempt": None,
                "reason": "interrupted",
                "goal_spec_hash": last_failure.goal_spec_hash if last_failure is not None else None,
                "last_failure": last_failure.to_json() if last_failure is not None else None,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded external-LLM failure feedback loop for R1Pro task planning."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="TaskSpec id or YAML path; its scene and instruction define the run",
    )
    parser.add_argument(
        "--goal-spec",
        type=Path,
        default=None,
        help="Frozen GoalSpec JSON path for the generic product replay",
    )
    parser.add_argument("--urdf", type=Path, required=True, help="R1Pro URDF for the skill catalogue")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Loop output directory (default: outputs/tasks/<scene-stem>_llm_loop)",
    )
    parser.add_argument(
        "--initial-feedback",
        type=Path,
        default=None,
        help="Prior feedback.json to seed a continuation run",
    )
    parser.add_argument(
        "--initial-plan",
        type=Path,
        default=None,
        help="Prior validated plan.json to preserve during a continuation run",
    )
    parser.add_argument("--max-attempts", type=int, default=10, help="Hard attempt ceiling (default: 10)")
    parser.add_argument(
        "--feedback-window", type=int, default=3, help="Sliding feedback window size (default: 3)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        choices=(DEFAULT_VIDEO_FPS,),
        help=f"Replay video fps (fixed: {DEFAULT_VIDEO_FPS})",
    )
    parser.add_argument("--width", type=int, default=960, help="Replay camera width (default: 960)")
    parser.add_argument("--height", type=int, default=544, help="Replay camera height (default: 544)")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic rollout seed (default: 0)")
    parser.add_argument(
        "--physical-gpu-id",
        type=int,
        default=6,
        help="Physical Vulkan/RTX GPU index passed to the replay (default: 6)",
    )
    parser.add_argument("--device", default="cuda:0", help="Replay logical CUDA device (default: cuda:0)")
    parser.add_argument(
        "--stream-replay-logs",
        action="store_true",
        help="Forward Isaac Sim replay stdout/stderr instead of buffering it",
    )
    return parser


def _generate_goal_spec(task_spec, args: argparse.Namespace, output_dir: Path) -> Path:
    """Freeze a GoalSpec from the natural-language task description.

    The public TaskSpec supplies the scene and natural-language instruction.
    The GoalPlanner turns that instruction into a closed, verifiable GoalSpec
    that the rest of the loop consumes unchanged.
    """
    scene = load_scene_data(
        task_spec.scene,
        source=task_spec.source_path or task_spec.id,
    )
    kin = R1ProKinematics(str(args.urdf))
    scene_facts = scene_to_facts(scene, kinematics=kin)
    planner = GoalPlanner(DeepSeekClient.from_env())
    result = planner.plan(
        GoalPlanningRequest(
            task_description=task_spec.instruction,
            scene_facts=scene_facts,
            scene=scene,
        )
    )
    if result.status != "planned" or result.goal_spec is None or not result.goal_spec_hash:
        raise SystemExit(f"goal planning failed: {result.reason}")
    goal_path = output_dir / "goal_spec.json"
    goal_path.parent.mkdir(parents=True, exist_ok=True)
    goal_path.write_text(
        json.dumps(goal_spec_to_dict(result.goal_spec), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    from r1pro_data_gen.planning.goals.compiler import GoalCompiler

    compiled = GoalCompiler().compile(result.goal_spec, scene)
    (output_dir / "goal_contract.json").write_text(
        json.dumps(compiled.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[llm-loop] frozen goal_spec written to {goal_path}", flush=True)
    return goal_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        task_spec = load_task_spec(args.task)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"invalid TaskSpec: {exc}", file=sys.stderr)
        return _USAGE_EXIT
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "tasks" / f"{task_spec.id.replace('.', '_')}_llm_loop"
    else:
        output_dir = args.output_dir
    try:
        config = LoopConfig(
            scene=task_spec.scene,
            task=task_spec.instruction,
            urdf=args.urdf,
            output_dir=output_dir,
            task_spec_path=task_spec.source_path,
            goal_spec=args.goal_spec,
            initial_feedback=args.initial_feedback,
            initial_plan=args.initial_plan,
            max_attempts=args.max_attempts,
            feedback_window=args.feedback_window,
            fps=args.fps,
            width=args.width,
            height=args.height,
            physical_gpu_id=args.physical_gpu_id,
            device=args.device,
            seed=args.seed,
            stream_replay_logs=args.stream_replay_logs,
        )
    except ValueError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return _USAGE_EXIT
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = write_scene_yaml(task_spec.scene, output_dir / "scene.yaml")
    (output_dir / "input.json").write_text(
        json.dumps(
            {
                "task": task_spec.to_dict(),
                "task_spec_path": str(task_spec.source_path.resolve())
                if task_spec.source_path
                else None,
                "scene_path": str(scene_path),
                "scene_human_verified": task_spec.scene_human_verified,
                "instruction": task_spec.instruction,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    # Without an authored GoalSpec file, freeze one from the TaskSpec
    # instruction before the closed loop runs.
    if config.goal_spec is None:
        config = replace(config, goal_spec=_generate_goal_spec(task_spec, args, output_dir))
    try:
        outcome = run_loop(config)
    except KeyboardInterrupt:
        return _INTERRUPT_EXIT
    print(json.dumps(outcome.to_json(), ensure_ascii=False, sort_keys=True))
    return _SUCCESS_EXIT if outcome.status == "succeeded" else _FAILURE_EXIT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_feedback(path: Path) -> Feedback:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"initial feedback is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("initial feedback must be a JSON object")
    try:
        return FactFeedback.from_json(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"initial feedback is invalid: {exc}") from exc


def _serialize_feedback(item: Any) -> str:
    if isinstance(item, Feedback):
        return json.dumps(
            item.to_planner_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    if isinstance(item, str):
        return item
    raise TypeError(
        f"feedback items must be Feedback or bounded strings, got {type(item).__name__}"
    )


def _is_provider_failure(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in _PROVIDER_FAILURE_MARKERS)


def _validate_plan_goal_spec_hash(
    plan: Any,
    *,
    goal_spec_hash: str | None,
    generic: bool,
) -> None:
    """Reject a plan not bound to the frozen generic completion contract."""
    if not generic:
        return
    if not isinstance(getattr(plan, "metadata", None), Mapping):
        raise ValueError("plan goal_spec_hash is missing")
    if plan.metadata.get("goal_spec_hash") != goal_spec_hash:
        raise ValueError("plan goal_spec_hash does not match frozen GoalSpec")


def _validate_plan_goal_contract_hash(
    plan: Any,
    *,
    goal_contract_hash: str | None,
    generic: bool,
) -> None:
    """Validate the compiled contract when the producer supplies the field.

    Older injected planners and archived plans may carry only the original
    GoalSpec hash. The production LLM planner receives a non-optional contract
    hash and is strict about preserving it; this compatibility allowance keeps
    historical replay fixtures readable.
    """
    if not generic or goal_contract_hash is None:
        return
    metadata = getattr(plan, "metadata", None)
    if not isinstance(metadata, Mapping):
        return
    actual = metadata.get("goal_contract_hash")
    if actual is not None and actual != goal_contract_hash:
        raise ValueError("plan goal_contract_hash does not match frozen GoalContract")

def _validate_feedback_goal_spec_hash(
    feedback: Feedback,
    *,
    goal_spec_hash: str | None,
    generic: bool,
) -> None:
    """Reject stale initial feedback in a generic frozen-goal run."""
    if not generic:
        return
    if feedback.goal_spec_hash != goal_spec_hash:
        raise ValueError("initial feedback goal_spec_hash does not match frozen GoalSpec")


def _validate_feedback_goal_contract_hash(
    feedback: Feedback,
    *,
    goal_contract_hash: str | None,
    generic: bool,
) -> None:
    if not generic or goal_contract_hash is None:
        return
    actual = getattr(feedback, "goal_contract_hash", "")
    if actual and actual != goal_contract_hash:
        raise ValueError("initial feedback goal_contract_hash does not match frozen GoalContract")


def _validate_replay_goal_spec_hash(
    result: Mapping[str, Any] | None,
    *,
    goal_spec_hash: str | None,
    generic: bool,
) -> None:
    """Reject replay artifacts that are absent or bound to another GoalSpec."""
    if not generic:
        return
    if not isinstance(result, Mapping) or result.get("goal_spec_hash") != goal_spec_hash:
        raise ValueError("replay goal_spec_hash does not match frozen GoalSpec")


def _validate_replay_goal_contract_hash(
    result: Mapping[str, Any] | None,
    *,
    goal_contract_hash: str | None,
    generic: bool,
) -> None:
    if not generic or goal_contract_hash is None:
        return
    if not isinstance(result, Mapping):
        return
    actual = result.get("goal_contract_hash")
    if actual is not None and actual != goal_contract_hash:
        raise ValueError("replay goal_contract_hash does not match frozen GoalContract")

def _replay_succeeded(
    result: Mapping[str, Any] | None,
    *,
    require_goal_spec: bool = True,
    goal_spec_hash: str | None = None,
) -> bool:
    if not isinstance(result, Mapping):
        return False
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    if result.get("result") != "passed" or evaluation.get("status") != "succeeded":
        return False
    if not require_goal_spec:
        return True
    if result.get("evaluation_mode") != "goal_spec":
        return False
    return goal_spec_hash is None or result.get("goal_spec_hash") == goal_spec_hash


def _feedback_from_replay_result(
    config: LoopConfig,
    scene_facts: Mapping[str, Any],
    plan: Any,
    replay_outcome: ReplayOutcome,
    attempt: int,
    *,
    goal_spec_hash: str = "",
    goal_contract_hash: str = "",
) -> Feedback:
    result = replay_outcome.result if isinstance(replay_outcome.result, Mapping) else {}
    try:
        return extract_failure_feedback(
            task_description=config.task,
            scene_facts=scene_facts,
            plan=plan,
            execution=_mapping(result, "execution"),
            measurements=_mapping(result, "measurements"),
            evaluation=_mapping(result, "evaluation"),
            attempt=attempt,
            goal_spec_hash=goal_spec_hash,
            goal_contract_hash=goal_contract_hash,
        )
    except ValueError:
        # Replay produced a failed marker but no structured failure signal.
        return _manual_failure_feedback(
            attempt=attempt,
            failure_type="gpu",
            reason="replay failed without a structured failure signal",
            goal_spec_hash=goal_spec_hash,
            goal_contract_hash=goal_contract_hash,
        )


def _runtime_failure_feedback(
    attempt: int,
    outcome: ReplayOutcome,
    *,
    goal_spec_hash: str = "",
    goal_contract_hash: str = "",
) -> Feedback:
    detail = outcome.runtime_error or f"replay unavailable (returncode={outcome.returncode})"
    return _manual_failure_feedback(
        attempt=attempt,
        failure_type="gpu",
        reason=detail,
        goal_spec_hash=goal_spec_hash,
        goal_contract_hash=goal_contract_hash,
    )


def _manual_failure_feedback(
    *,
    attempt: int,
    failure_type: str,
    reason: str,
    goal_spec_hash: str = "",
    goal_contract_hash: str = "",
) -> Feedback:
    return FactFeedback(
        attempt=attempt,
        failed_stage=None,
        skill=None,
        request={},
        observations={
            "failure_type": failure_type,
            "reason": reason,
            "raw_error": reason,
        },
        discrepancies=(),
        completed_prefix=(),
        goal_spec_hash=goal_spec_hash,
        goal_contract_hash=goal_contract_hash,
    )


def _mapping(value: Any, key: str) -> Mapping[str, Any] | None:
    item = value.get(key) if isinstance(value, Mapping) else None
    return item if isinstance(item, Mapping) else None


def _archive_failure(
    attempt_dir: Path,
    feedback_window: deque[str],
    feedback: Feedback,
) -> Feedback:
    """Persist feedback.json/feedback.md and append only then to the window."""
    _write_feedback(attempt_dir, feedback)
    feedback_window.append(_serialize_feedback(feedback))
    return feedback


def _write_feedback(attempt_dir: Path, feedback: Feedback) -> None:
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "feedback.json").write_text(
        json.dumps(feedback.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "feedback.md").write_text(feedback.to_markdown() + "\n", encoding="utf-8")


def _write_planner_request(attempt_dir: Path, request: TaskPlanningRequest) -> None:
    """Persist the exact non-secret planner input for prompt-boundary audits."""
    Path(attempt_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "task_description": request.task_description,
        "scene_facts": _strip_secrets(request.scene_facts),
        "skill_catalog": _strip_secrets(list(request.skill_catalog)),
        "constraints": _strip_secrets(request.constraints),
        "metadata": _strip_secrets(request.metadata),
    }
    if request.goal_spec is not None:
        payload["goal_spec"] = _strip_secrets(request.goal_spec)
        payload["goal_spec_hash"] = request.goal_spec_hash
        if request.goal_contract_hash is not None:
            payload["goal_contract_hash"] = request.goal_contract_hash
    (Path(attempt_dir) / "planner_request.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _strip_secrets(value: Any) -> Any:
    """Recursively drop secret-bearing keys and redact secret-looking values."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key).lower()
            if any(marker in name for marker in _SECRET_KEY_MARKERS):
                continue
            out[key] = _strip_secrets(item)
        return out
    if isinstance(value, list):
        return [_strip_secrets(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_PATTERN.sub("<redacted>", value)
    return value


def _write_raw_provenance(
    attempt_dir: Path,
    result: TaskPlanningResult,
    *,
    goal_spec_hash: str | None = None,
    goal_contract_hash: str | None = None,
) -> None:
    (Path(attempt_dir) / "plan.raw.json").write_text(
        json.dumps(
            {
                "status": result.status,
                "provider": result.provider,
                "model": result.model,
                "usage": dict(result.usage),
                "goal_spec_hash": goal_spec_hash,
                "goal_contract_hash": goal_contract_hash,
                "raw_response": _strip_secrets(result.raw_response),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_planner_result(
    attempt_dir: Path,
    result: TaskPlanningResult,
    feedback: Feedback,
    *,
    goal_spec_hash: str | None = None,
    goal_contract_hash: str | None = None,
) -> None:
    """Persist a bounded planner_result.json for failed/unsupported rounds.

    An executable plan.json is intentionally *not* written on these rounds; the
    failure record keeps only bounded status/reason/provider/model plus the
    failure type that was extracted into the feedback record.
    """
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "planner_result.json").write_text(
        json.dumps(
            {
                "status": result.status,
                "provider": result.provider,
                "model": result.model,
                "reason": _strip_secrets((result.reason or "")[:_MAX_RUNTIME_ERROR_CHARS]),
                "failure_type": feedback.failure_type,
                "goal_spec_hash": goal_spec_hash,
                "goal_contract_hash": goal_contract_hash,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_stage_calls(output_dir: Path, result: Mapping[str, Any]) -> None:
    """Copy each stage's call record from result.json into stage_calls.json."""
    execution = _mapping(result, "execution")
    if execution is None:
        return
    stage_results = _mapping(execution, "stage_results")
    if stage_results is None:
        return
    calls: dict[str, Any] = {}
    for name, stage in stage_results.items():
        if not isinstance(stage, Mapping):
            continue
        call = stage.get("call")
        if isinstance(call, Mapping):
            calls[name] = call
    if not calls:
        return
    (Path(output_dir) / "stage_calls.json").write_text(
        json.dumps(calls, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _attempt_dir(output_dir: Path, attempt: int) -> Path:
    return Path(output_dir) / f"attempt_{attempt:02d}"


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


if __name__ == "__main__":
    raise SystemExit(main())
