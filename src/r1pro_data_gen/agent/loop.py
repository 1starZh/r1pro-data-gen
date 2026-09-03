"""Bounded observe-act loop over one persistent simulation episode."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from r1pro_data_gen.data.plan_io import plan_from_dict
from r1pro_data_gen.domain import GoalSpec, goal_spec_to_dict
from r1pro_data_gen.evaluation.predicates import VerificationReport, VerificationStatus
from r1pro_data_gen.agent.contracts import (
    AGENT_SCHEMA_VERSION,
    AgentActionValidationError,
    parse_agent_response,
    validate_action_envelope,
)
from r1pro_data_gen.agent.skeleton import build_semantic_plan_skeleton
from r1pro_data_gen.planning.llm.providers.protocol import ProviderError, TaskPlanningProvider
from r1pro_data_gen.skills import SkillResult

from .observation import build_agent_observation
from .prompt import system_prompt, user_prompt


ProgressFn = Callable[[], VerificationReport | None]
CheckpointFn = Callable[[Mapping[str, Any]], None]


@dataclass(slots=True)
class AgentStep:
    """One executed agent action."""

    index: int
    skill: str
    parameters: dict[str, Any]
    result: SkillResult
    progress_status: str | None = None


@dataclass(slots=True)
class AgentEpisode:
    """Outcome of one bounded closed-loop episode."""

    status: str
    reason: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    success_action: int | None = None
    verification: VerificationReport | None = None
    provider: str = ""
    model: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            # This object represents one bounded episode. Keep episode and
            # action counts separate so callers cannot mistake an action for
            # an outer retry attempt.
            "attempts": 1,
            "success_attempt": 1 if self.status == "succeeded" else None,
            "action_count": len(self.steps),
            "success_action": self.success_action,
            "provider": self.provider,
            "model": self.model,
            "steps": [
                {
                    "index": step.index,
                    "skill": step.skill,
                    "parameters": step.parameters,
                    "success": step.result.success,
                    "failure_code": step.result.details.get("failure_code")
                    or step.result.metrics.get("failure_code"),
                    "progress_status": step.progress_status,
                    # Preserve bounded, action-local diagnostics so a failed
                    # semantic action can be repaired from evidence rather
                    # than being reduced to a bare boolean in the episode
                    # trace.  These are diagnostics only; acceptance still
                    # uses the physical GoalSpec verifier.
                    "diagnostics": {
                        "metrics": _json_safe(step.result.metrics),
                        "details": _json_safe(step.result.details),
                    },
                }
                for step in self.steps
            ],
        }

    def to_plan_dict(self, task_name: str = "agent_episode") -> dict[str, Any]:
        """Materialize the action trace as a replayable Plan mapping."""
        stages = []
        previous = None
        for step in self.steps:
            name = f"step_{step.index:02d}_{step.skill}"
            stages.append(
                {
                    "name": name,
                    "goal": step.skill,
                    "depends_on": [] if previous is None else [previous],
                    "parameters": {"skill": step.skill, **step.parameters},
                    "outputs": [],
                    "preconditions": [],
                    "postconditions": [],
                }
            )
            previous = name
        return {
            "task_name": task_name,
            "stages": stages,
            "metadata": {
                "source": "closed_loop_agent",
                "schema_version": AGENT_SCHEMA_VERSION,
                "status": self.status,
            },
        }

    def to_plan(self, task_name: str = "agent_episode"):
        return plan_from_dict(self.to_plan_dict(task_name))


@contextmanager
def _planning_clock_guard(adapter: Any) -> Iterator[None]:
    """Pause physics while the LLM and observation snapshot run.

    Recorded video samples one frame per physics ``step()``.  Freezing the
    clock here is what keeps the trajectory continuous across planner waits.
    """
    freeze = getattr(adapter, "freeze_simulation_clock", None)
    if callable(freeze):
        with freeze():
            yield
        return
    yield


def _json_safe(value: Any) -> Any:
    """Convert common runtime values into JSON-safe action diagnostics."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    # Numpy arrays/scalars expose one of these without requiring the agent
    # loop to import a numerical backend at module import time.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return value


class AgentLoop:
    """Ask an LLM for one skill at a time and execute it in the live scene."""

    def __init__(
        self,
        provider: TaskPlanningProvider,
        orchestrator: Any,
        *,
        catalog: Sequence[Mapping[str, Any]],
        scene_object_names: Sequence[str] = (),
        scene_facts: Mapping[str, Any] | None = None,
        goal_spec: GoalSpec | None = None,
        max_actions: int = 12,
        max_consecutive_failures: int = 3,
        progress_fn: ProgressFn | None = None,
        registry: Any = None,
        prior_feedback: Sequence[Mapping[str, Any]] = (),
        feedback_window: int = 3,
        checkpoint_fn: CheckpointFn | None = None,
    ) -> None:
        if max_actions < 1:
            raise ValueError("max_actions must be at least 1")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")
        if feedback_window < 0:
            raise ValueError("feedback_window must be non-negative")
        self.provider = provider
        self.orchestrator = orchestrator
        self.catalog = tuple(catalog)
        self.scene_object_names = tuple(scene_object_names)
        self.scene_facts = dict(scene_facts or {})
        self.goal_spec = goal_spec
        self.max_actions = max_actions
        self.max_consecutive_failures = max_consecutive_failures
        self.progress_fn = progress_fn
        self.registry = registry
        self.prior_feedback = tuple(dict(item) for item in prior_feedback)
        self.feedback_window = feedback_window
        # Checkpoints are operational diagnostics only.  A broken checkpoint
        # sink must never change the task result or interrupt a physical
        # episode.
        self.checkpoint_fn = checkpoint_fn
        self.plan_skeleton = build_semantic_plan_skeleton(
            self.goal_spec,
            skill_catalogue=self.catalog,
        )

    def run(self, *, task_description: str) -> AgentEpisode:
        if not task_description.strip():
            raise ValueError("task_description must not be empty")
        system = system_prompt(self.catalog)
        steps: list[AgentStep] = []
        consecutive_failures = 0
        last_result: SkillResult | None = None
        last_skill: str | None = None
        last_parameters: dict[str, Any] | None = None
        goal_payload = goal_spec_to_dict(self.goal_spec) if self.goal_spec is not None else None

        self._checkpoint(
            {
                "phase": "episode_started",
                "max_actions": self.max_actions,
                "task_description": task_description,
            }
        )

        opening = self._progress()
        if opening is not None and opening.status is VerificationStatus.SUCCEEDED:
            return AgentEpisode(
                status="succeeded",
                reason="goal already satisfied",
                verification=opening,
                provider=getattr(self.provider, "name", ""),
                model=getattr(self.provider, "model", ""),
            )

        for index in range(1, self.max_actions + 1):
            self._checkpoint(
                {
                    "phase": "planning",
                    "action_index": index,
                    "remaining_actions": self.max_actions - index + 1,
                    "last_skill": last_skill,
                }
            )
            adapter = getattr(self.orchestrator, "adapter", None)
            try:
                with _planning_clock_guard(adapter):
                    observation = build_agent_observation(
                        adapter=self.orchestrator.adapter,
                        scene=self.orchestrator.scene,
                        scene_facts=self.scene_facts,
                        last_result=last_result,
                        last_skill=last_skill,
                        last_parameters=last_parameters,
                        progress=self._progress(),
                        remaining_actions=self.max_actions - index + 1,
                        skill_catalogue=self.catalog,
                        prior_feedback=(
                            self.prior_feedback[-self.feedback_window:]
                            if self.feedback_window
                            else ()
                        ),
                        plan_skeleton=self.plan_skeleton,
                    )
                    user = user_prompt(
                        task_description=task_description,
                        observation=observation,
                        goal_spec=goal_payload,
                    )
                    response = self.provider.complete(system=system, user=user)
            except ProviderError as exc:
                self._checkpoint(
                    {
                        "phase": "episode_finished",
                        "status": "failed",
                        "reason": f"provider error: {exc}",
                    }
                )
                return AgentEpisode(
                    status="failed",
                    reason=f"provider error: {exc}",
                    steps=steps,
                    provider=getattr(self.provider, "name", ""),
                    model=getattr(self.provider, "model", ""),
                )
            try:
                live = observation.get("live") if isinstance(observation, Mapping) else {}
                live = live if isinstance(live, Mapping) else {}
                action = validate_action_envelope(
                    parse_agent_response(response.text),
                    skill_catalog=self.catalog,
                    registry=self.registry,
                    scene_object_names=self.scene_object_names,
                    scene=getattr(self.orchestrator, "scene", None),
                    attachments=live.get("attachments"),
                    object_positions=live.get("objects"),
                    base_pose=live.get("base_pose"),
                )
            except AgentActionValidationError as exc:
                result = SkillResult(
                    False,
                    "invalid_action",
                    details={
                        "reason": str(exc),
                        "failure_code": "invalid_action",
                    },
                )
                step = AgentStep(
                    index=index,
                    skill="invalid_action",
                    parameters={},
                    result=result,
                    progress_status=None,
                )
                steps.append(step)
                self._checkpoint(
                    {
                        "phase": "action_completed",
                        "action_index": index,
                        "skill": "invalid_action",
                        "parameters": {},
                        "success": False,
                        "failure_code": "invalid_action",
                        "reason": str(exc),
                    }
                )
                last_result = result
                last_skill = "invalid_action"
                last_parameters = {}
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    return AgentEpisode(
                        status="failed",
                        reason=f"invalid agent action: {exc}",
                        steps=steps,
                        provider=response.provider,
                        model=response.model,
                    )
                continue
            if action is None:
                self._checkpoint(
                    {
                        "phase": "episode_finished",
                        "status": "unsupported",
                        "reason": "agent returned unsupported",
                        "action_index": index,
                    }
                )
                return AgentEpisode(
                    status="unsupported",
                    reason="agent returned unsupported",
                    steps=steps,
                    provider=response.provider,
                    model=response.model,
                )
            self._checkpoint(
                {
                    "phase": "action_started",
                    "action_index": index,
                    "skill": action.skill,
                    "parameters": dict(action.parameters),
                }
            )
            result, _call = self.orchestrator.execute_skill(
                action.skill,
                action.parameters,
                stage_name=f"step_{index:02d}_{action.skill}",
            )
            progress = self._progress()
            step = AgentStep(
                index=index,
                skill=action.skill,
                parameters=dict(action.parameters),
                result=result,
                progress_status=None if progress is None else progress.status.value,
            )
            steps.append(step)
            self._checkpoint(
                {
                    "phase": "action_completed",
                    "action_index": index,
                    "skill": action.skill,
                    "parameters": dict(action.parameters),
                    "success": bool(result.success),
                    "failure_code": result.details.get("failure_code")
                    or result.metrics.get("failure_code"),
                    "progress_status": None if progress is None else progress.status.value,
                    "diagnostics": {
                        "metrics": _json_safe(result.metrics),
                        "details": _json_safe(result.details),
                    },
                }
            )
            last_result = result
            last_skill = action.skill
            last_parameters = dict(action.parameters)
            if progress is not None and progress.status is VerificationStatus.SUCCEEDED:
                self._checkpoint(
                    {
                        "phase": "episode_finished",
                        "status": "succeeded",
                        "reason": "goal predicates satisfied",
                        "action_index": index,
                    }
                )
                return AgentEpisode(
                    status="succeeded",
                    reason="goal predicates satisfied",
                    steps=steps,
                    success_action=index,
                    verification=progress,
                    provider=response.provider,
                    model=response.model,
                )
            if progress is not None and progress.status is VerificationStatus.FAILED:
                self._checkpoint(
                    {
                        "phase": "episode_finished",
                        "status": "failed",
                        "reason": progress.failure_reason or "invariant violated",
                        "action_index": index,
                    }
                )
                return AgentEpisode(
                    status="failed",
                    reason=progress.failure_reason or "invariant violated",
                    steps=steps,
                    verification=progress,
                    provider=response.provider,
                    model=response.model,
                )
            if result.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    self._checkpoint(
                        {
                            "phase": "episode_finished",
                            "status": "failed",
                            "reason": "consecutive skill failures exhausted",
                            "action_index": index,
                        }
                    )
                    return AgentEpisode(
                        status="failed",
                        reason="consecutive skill failures exhausted",
                        steps=steps,
                        verification=progress,
                        provider=response.provider,
                        model=response.model,
                    )
        self._checkpoint(
            {
                "phase": "episode_finished",
                "status": "failed",
                "reason": "max actions exhausted",
            }
        )
        return AgentEpisode(
            status="failed",
            reason="max actions exhausted",
            steps=steps,
            verification=self._progress(),
            provider=getattr(self.provider, "name", ""),
            model=getattr(self.provider, "model", ""),
        )

    def _progress(self) -> VerificationReport | None:
        if self.progress_fn is None:
            return None
        return self.progress_fn()

    def _checkpoint(self, event: Mapping[str, Any]) -> None:
        """Send a best-effort operational checkpoint to the host."""
        if self.checkpoint_fn is None:
            return
        try:
            self.checkpoint_fn(_json_safe(dict(event)))
        except Exception:
            # Checkpointing is deliberately non-authoritative.  In
            # particular, a full action diagnostic can contain a backend
            # object that a custom sink cannot serialize.
            return


__all__ = ["AgentEpisode", "AgentLoop", "AgentStep"]
