"""Task and motion-intent planning.

The package answers "what should happen next" and turns task intent into a
validated :class:`~r1pro_data_gen.domain.Plan`. It deliberately does not own
closed-loop agent state (see :mod:`r1pro_data_gen.agent`) or motion algorithms
(see :mod:`r1pro_data_gen.methods`).

Subpackages are organized by planning concern:

``goals``
    GoalSpec compilation and completeness checks.
``navigation``
    Semantic navigation target resolution and reachability.
``context``
    Scene facts, interaction targets, and runtime references.
``task``
    Task-level planner contracts and implementations.
``llm``
    External-provider contracts and provider adapters.
"""

from .context import (
    InteractionTargetError,
    InteractionTargetResolution,
    RuntimeReferenceError,
    object_names,
    resolve_interaction_target,
    resolve_parameters,
    resolve_reference,
    scene_facts_from_mapping,
    scene_to_facts,
)
from .contracts import Planner, PlannerRequest, PlannerResult
from .goals import (
    CompiledGoalContract,
    GoalCompileError,
    GoalCompiler,
    GoalPlanner,
    GoalPlanningRequest,
    GoalPlanningResult,
    goal_spec_completeness_errors,
)
from .navigation import (
    InteractionReachabilityReport,
    NavigationTargetError,
    NavigationTargetResolution,
    assess_interaction_target,
    resolve_navigation_target,
)
from .task import (
    LLMProvider,
    LLMTaskPlanner,
    StageSpec,
    TaskPlanner,
    TaskPlanningRequest,
    TaskPlanningResult,
    TemplatePlanner,
)

__all__ = [
    "CompiledGoalContract",
    "GoalCompileError",
    "GoalCompiler",
    "GoalPlanner",
    "GoalPlanningRequest",
    "GoalPlanningResult",
    "InteractionReachabilityReport",
    "InteractionTargetError",
    "InteractionTargetResolution",
    "LLMProvider",
    "LLMTaskPlanner",
    "NavigationTargetError",
    "NavigationTargetResolution",
    "Planner",
    "PlannerRequest",
    "PlannerResult",
    "RuntimeReferenceError",
    "StageSpec",
    "TaskPlanner",
    "TaskPlanningRequest",
    "TaskPlanningResult",
    "TemplatePlanner",
    "assess_interaction_target",
    "goal_spec_completeness_errors",
    "object_names",
    "resolve_interaction_target",
    "resolve_navigation_target",
    "resolve_parameters",
    "resolve_reference",
    "scene_facts_from_mapping",
    "scene_to_facts",
]
