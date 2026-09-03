"""Plan executor: runs a skill-sequence plan against the simulation.

The orchestrator is the generic loop of the data-generation pipeline: for
every stage of a :class:`Plan` it looks the stage's skill up in a registry,
calls it with the stage parameters (plus the shared step hook), and records
the result. It knows nothing about specific tasks or skills -- new tasks plug
in their own plan templates and skill registries.

Failure recovery / replanning hooks are intentionally left as future work:
for now a failed stage stops the run with the failure recorded.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import signal
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from r1pro_data_gen.domain import Plan, SceneModel
from r1pro_data_gen.planning.context.runtime_refs import RuntimeReferenceError, resolve_parameters

from .contracts import PhysicalSafetyViolation
from ..skills import SkillResult

StepHook = Callable[[], None]
StageHook = Callable[[str], None]
StageEndHook = Callable[[str, bool], None]


class ActionBudgetExceeded(RuntimeError):
    """Raised internally when one physical skill call exceeds its budget."""

    def __init__(
        self,
        *,
        steps: int,
        elapsed_s: float,
        max_steps: int | None,
        max_seconds: float | None,
        phase: str,
    ) -> None:
        self.steps = int(steps)
        self.elapsed_s = float(elapsed_s)
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.phase = phase
        limits = []
        if max_steps is not None:
            limits.append(f"{max_steps} physics steps")
        if max_seconds is not None:
            limits.append(f"{max_seconds:g}s wall time")
        super().__init__(
            f"physical action budget exceeded during {phase}: "
            f"{steps} steps, {elapsed_s:.3f}s elapsed "
            f"(limits: {', '.join(limits)})"
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "physics_steps": self.steps,
            "elapsed_s": self.elapsed_s,
            "max_physics_steps": self.max_steps,
            "max_wall_time_s": self.max_seconds,
            "budget_phase": self.phase,
            "failure_code": "action_budget_exceeded",
        }


class _ActionBudgetAdapter:
    """Proxy an adapter and bound every simulator step of one skill call."""

    def __init__(
        self,
        adapter: Any,
        *,
        max_steps: int | None,
        max_seconds: float | None,
    ) -> None:
        self._adapter = adapter
        self._max_steps = max_steps
        self._max_seconds = max_seconds
        self._started = time.monotonic()
        self.steps = 0

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self._started)

    def _check(self, phase: str) -> None:
        elapsed = self.elapsed_s
        if self._max_steps is not None and self.steps >= self._max_steps:
            raise ActionBudgetExceeded(
                steps=self.steps,
                elapsed_s=elapsed,
                max_steps=self._max_steps,
                max_seconds=self._max_seconds,
                phase=phase,
            )
        if self._max_seconds is not None and elapsed >= self._max_seconds:
            raise ActionBudgetExceeded(
                steps=self.steps,
                elapsed_s=elapsed,
                max_steps=self._max_steps,
                max_seconds=self._max_seconds,
                phase=phase,
            )

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self._check("before_step")
        result = self._adapter.step(*args, **kwargs)
        self.steps += 1
        violation_reader = getattr(self._adapter, "physical_safety_violation", None)
        if callable(violation_reader):
            violation = violation_reader()
            if violation is not None:
                metrics_reader = getattr(self._adapter, "physical_metrics", None)
                metrics = metrics_reader() if callable(metrics_reader) else {}
                raise PhysicalSafetyViolation(str(violation), metrics=metrics)
        self._check("after_step")
        return result

    def check(self, phase: str = "after_action") -> None:
        self._check(phase)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


@contextmanager
def _action_budget_guard(adapter: _ActionBudgetAdapter):
    """Bound Python-side planning as well as simulator stepping.

    A skill can spend its entire budget in IK/path planning before it calls
    ``adapter.step()``.  The step proxy cannot observe that case, so hosted
    runs also install a short-lived main-thread timer around the complete
    skill call.  The guard is deliberately best-effort outside the main
    thread (where Python cannot install a signal handler); the step-level
    checks remain active there.
    """
    timeout = adapter._max_seconds
    can_interrupt = (
        timeout is not None
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_interrupt:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _timeout_handler(_signum: int, _frame: Any) -> None:
        adapter._check("planning_or_execution")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(float(timeout), 1e-3))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


@dataclass(slots=True)
class StageCall:
    """Audit record for one explicit Plan stage invocation."""

    skill: str
    raw_parameters: Mapping[str, Any]
    resolved_parameters: Mapping[str, Any] | None = None
    success: bool = False
    error: str | None = None


@dataclass(slots=True)
class PlanExecution:
    """Result of running a plan stage by stage."""

    stage_results: dict[str, SkillResult] = field(default_factory=dict)
    stage_calls: dict[str, StageCall] = field(default_factory=dict)
    completed: tuple[str, ...] = ()
    failed: str | None = None
    failure_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.failed is None


class Orchestrator:
    """Execute a skill-sequence plan against a simulation adapter."""

    def __init__(
        self,
        adapter: Any,
        skills: Mapping[str, Any],
        scene: SceneModel | None = None,
        step_hook: StepHook | None = None,
        stage_hook: StageHook | None = None,
        stage_end_hook: StageEndHook | None = None,
        frame_converter: Callable[[Any, str, str], Any] | None = None,
        max_action_physics_steps: int | None = 60000,
        max_action_seconds: float | None = 600.0,
    ) -> None:
        if max_action_physics_steps is not None:
            if (
                isinstance(max_action_physics_steps, bool)
                or not isinstance(max_action_physics_steps, int)
                or max_action_physics_steps < 1
            ):
                raise ValueError("max_action_physics_steps must be a positive integer or None")
        if max_action_seconds is not None:
            if (
                isinstance(max_action_seconds, bool)
                or not isinstance(max_action_seconds, (int, float))
                or not math.isfinite(float(max_action_seconds))
                or float(max_action_seconds) <= 0.0
            ):
                raise ValueError("max_action_seconds must be positive and finite or None")
        self.adapter = adapter
        self.skills = skills
        self.scene = scene
        self.step_hook = step_hook
        self.stage_hook = stage_hook
        self.stage_end_hook = stage_end_hook
        self.frame_converter = frame_converter
        self.max_action_physics_steps = max_action_physics_steps
        self.max_action_seconds = None if max_action_seconds is None else float(max_action_seconds)

    def execute_skill(
        self,
        skill_name: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        stage_name: str | None = None,
        stage_results: Mapping[str, SkillResult] | None = None,
        stage_outputs: Mapping[str, set[str]] | None = None,
        stage_dependencies: Sequence[str] = (),
    ) -> tuple[SkillResult, StageCall]:
        """Execute one skill call and return its result plus an audit record."""
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("skill_name must be a non-empty string")
        stage = stage_name or skill_name
        raw_params = dict(parameters or {})
        raw_params.pop("skill", None)
        call = StageCall(skill=skill_name, raw_parameters=raw_params)
        skill = self.skills.get(skill_name)
        if skill is None:
            result = SkillResult(
                False,
                skill_name,
                details={"reason": f"unknown skill {skill_name!r}", "failure_code": "unknown_skill"},
            )
            call.error = result.details["reason"]
            if self.stage_hook is not None:
                self.stage_hook(stage)
            if self.stage_end_hook is not None:
                self.stage_end_hook(stage, False)
            return result, call
        if self.stage_hook is not None:
            self.stage_hook(stage)
        action_adapter = _ActionBudgetAdapter(
            self.adapter,
            max_steps=self.max_action_physics_steps,
            max_seconds=self.max_action_seconds,
        )
        try:
            with _action_budget_guard(action_adapter):
                observation = self.adapter.read_observation(0.0)
                params = resolve_parameters(
                    raw_params,
                    stage_results=dict(stage_results or {}),
                    observation=observation,
                    scene=self.scene,
                    current_stage=stage,
                    stage_outputs=dict(stage_outputs or {}),
                    stage_dependencies=tuple(stage_dependencies),
                    frame_converter=self.frame_converter,
                )
                call.resolved_parameters = params
                if hasattr(self.skills, "execute"):
                    result = self.skills.execute(
                        skill_name,
                        action_adapter,
                        scene=self.scene,
                        step_hook=self.step_hook,
                        **params,
                    )
                else:
                    result = skill.execute(
                        action_adapter, scene=self.scene, step_hook=self.step_hook, **params
                    )
                action_adapter.check()
        except ActionBudgetExceeded as exc:
            call.error = str(exc)
            result = SkillResult(
                success=False,
                skill=skill_name,
                metrics=exc.metrics(),
                details={
                    "reason": str(exc),
                    "failure_code": "action_budget_exceeded",
                    "exception_type": type(exc).__name__,
                },
            )
        except PhysicalSafetyViolation as exc:
            call.error = str(exc)
            physical_metrics = exc.metrics()
            result = SkillResult(
                success=False,
                skill=skill_name,
                metrics=physical_metrics,
                details={
                    "reason": str(exc),
                    "failure_code": exc.code,
                    "exception_type": type(exc).__name__,
                    "physical_safety_violation": True,
                    "physical_metrics": physical_metrics,
                },
            )
        except (KeyError, TypeError, ValueError, RuntimeReferenceError) as exc:
            call.error = str(exc)
            result = SkillResult(
                success=False,
                skill=skill_name,
                details={
                    "reason": "invalid skill call",
                    "failure_code": "invalid_skill_call",
                    "exception": str(exc),
                    "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        except BaseException:
            if self.stage_end_hook is not None:
                self.stage_end_hook(stage, False)
            raise
        call.success = bool(result.success)
        if self.stage_end_hook is not None:
            self.stage_end_hook(stage, bool(result.success))
        return result, call

    def run_plan(self, plan: Plan) -> PlanExecution:
        """Run explicit stages in dependency order and stop on the first failure."""
        execution = PlanExecution()
        outputs = {stage.name: set(stage.outputs) for stage in plan.stages}
        known = set()

        def fail_stage(stage_name: str, reason: str) -> None:
            execution.failed = stage_name
            execution.failure_reason = reason
            if self.stage_end_hook is not None:
                self.stage_end_hook(stage_name, False)

        for stage in plan.stages:
            missing = set(stage.depends_on) - known
            if missing:
                fail_stage(
                    stage.name,
                    f"dependencies are not completed: {sorted(missing)}",
                )
                break
            if any(dep not in execution.completed for dep in stage.depends_on):
                fail_stage(stage.name, "dependency did not complete successfully")
                break
            skill_name = stage.parameters.get("skill")
            if not isinstance(skill_name, str):
                fail_stage(stage.name, "stage has no valid skill")
                break
            skill = self.skills.get(skill_name)
            if skill is None:
                fail_stage(stage.name, f"unknown skill {skill_name!r}")
                break
            raw_params = {k: v for k, v in stage.parameters.items() if k != "skill"}
            call = StageCall(skill=skill_name, raw_parameters=raw_params)
            execution.stage_calls[stage.name] = call
            if self.stage_hook is not None:
                self.stage_hook(stage.name)
            try:
                action_adapter = _ActionBudgetAdapter(
                    self.adapter,
                    max_steps=self.max_action_physics_steps,
                    max_seconds=self.max_action_seconds,
                )
                with _action_budget_guard(action_adapter):
                    observation = self.adapter.read_observation(0.0)
                    params = resolve_parameters(
                        raw_params,
                        stage_results=execution.stage_results,
                        observation=observation,
                        scene=self.scene,
                        current_stage=stage.name,
                        stage_outputs=outputs,
                        stage_dependencies=stage.depends_on,
                        frame_converter=self.frame_converter,
                    )
                    call.resolved_parameters = params
                    if hasattr(self.skills, "execute"):
                        result = self.skills.execute(
                            skill_name,
                            action_adapter,
                            scene=self.scene,
                            step_hook=self.step_hook,
                            **params,
                        )
                    else:
                        result = skill.execute(
                            action_adapter, scene=self.scene, step_hook=self.step_hook, **params
                        )
                    action_adapter.check()
            except ActionBudgetExceeded as exc:
                call.error = str(exc)
                result = SkillResult(
                    success=False,
                    skill=skill_name,
                    metrics=exc.metrics(),
                    details={
                        "reason": str(exc),
                        "failure_code": "action_budget_exceeded",
                        "exception_type": type(exc).__name__,
                    },
                )
            except PhysicalSafetyViolation as exc:
                # A physical gate crossing is a failed stage, not a Python
                # crash.  Preserve the measured boundary values in the
                # ordinary plan/evidence contract so generic task runners can
                # reject the episode and diagnose the responsible actuator.
                call.error = str(exc)
                physical_metrics = exc.metrics()
                result = SkillResult(
                    success=False,
                    skill=skill_name,
                    metrics=physical_metrics,
                    details={
                        "reason": str(exc),
                        "failure_code": exc.code,
                        "exception_type": type(exc).__name__,
                        "physical_safety_violation": True,
                        "physical_metrics": physical_metrics,
                    },
                )
            except (KeyError, TypeError, ValueError, RuntimeReferenceError) as exc:
                call.error = str(exc)
                result = SkillResult(
                    success=False,
                    skill=skill_name,
                    details={
                        "reason": "invalid skill call",
                        "exception": str(exc),
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
            except BaseException:
                if self.stage_end_hook is not None:
                    self.stage_end_hook(stage.name, False)
                raise
            execution.stage_results[stage.name] = result
            call.success = bool(result.success)
            if self.stage_end_hook is not None:
                self.stage_end_hook(stage.name, bool(result.success))
            if not result.success:
                execution.failed = stage.name
                execution.failure_reason = str(result.details.get("reason", "skill returned failure"))
                break
            execution.completed = (*execution.completed, stage.name)
            known.add(stage.name)
        return execution
