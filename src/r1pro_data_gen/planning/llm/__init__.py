"""LLM task-planning package.

The provider is isolated from task contracts so offline fake providers can be
used in tests and production code never exposes simulator handles to a model.
"""

from .contracts import (
    DEFAULT_PLAN_LIMITS,
    LLMPlanValidationError,
    LLM_PUBLIC_SKILLS,
    LLM_SCHEMA_VERSION,
    PlanLimits,
    parse_json_object,
    validate_envelope,
    validate_plan,
    validate_plan_dict,
)
from .providers import DeepSeekClient, DeepSeekConfig, ProviderError, ProviderResponse

__all__ = [
    "DEFAULT_PLAN_LIMITS",
    "DeepSeekClient",
    "DeepSeekConfig",
    "LLMPlanValidationError",
    "LLM_PUBLIC_SKILLS",
    "LLM_SCHEMA_VERSION",
    "PlanLimits",
    "ProviderError",
    "ProviderResponse",
    "parse_json_object",
    "validate_envelope",
    "validate_plan",
    "validate_plan_dict",
]
