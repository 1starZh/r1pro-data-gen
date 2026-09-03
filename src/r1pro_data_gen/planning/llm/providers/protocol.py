"""Provider-neutral response contracts for task-level LLMs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Normalized provider response; the body remains untrusted JSON text."""

    text: str
    provider: str
    model: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None


class ProviderError(RuntimeError):
    """Raised for transport, HTTP, authentication or provider payload errors."""


class TaskPlanningProvider(Protocol):
    """Protocol implemented by DeepSeek and offline fake providers."""

    name: str
    model: str

    def complete(self, *, system: str, user: str) -> ProviderResponse:
        """Return one non-streaming completion."""
        ...


__all__ = ["ProviderError", "ProviderResponse", "TaskPlanningProvider"]
