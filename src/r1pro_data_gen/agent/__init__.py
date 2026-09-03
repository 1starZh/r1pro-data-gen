"""Closed-loop agent: observe, choose one semantic skill, execute, repeat.

Agent action contracts and factual replanning feedback live beside the loop;
they are not task-planning modules.
"""

from .contracts import (
    AGENT_PUBLIC_SKILLS,
    AGENT_SCHEMA_VERSION,
    AgentAction,
    AgentActionValidationError,
    validate_action_envelope,
)
from .feedback import (
    Discrepancy,
    FactFeedback,
    Feedback,
    extract_failure_feedback,
)
from .loop import AgentEpisode, AgentLoop, AgentStep
from .observation import build_agent_observation
from .skeleton import build_semantic_plan_skeleton

__all__ = [
    "AGENT_PUBLIC_SKILLS",
    "AGENT_SCHEMA_VERSION",
    "AgentAction",
    "AgentActionValidationError",
    "AgentEpisode",
    "AgentLoop",
    "AgentStep",
    "Discrepancy",
    "FactFeedback",
    "Feedback",
    "build_agent_observation",
    "build_semantic_plan_skeleton",
    "extract_failure_feedback",
    "validate_action_envelope",
]
