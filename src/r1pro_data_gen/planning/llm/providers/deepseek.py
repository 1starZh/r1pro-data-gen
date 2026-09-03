"""Small standard-library DeepSeek Chat Completions client.

The client deliberately has no retry loop hidden inside it. Callers control the
bounded retry policy and can distinguish provider failures from invalid plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import socket
from urllib import error, request

from .protocol import ProviderError, ProviderResponse

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-v4-flash"
_DEFAULT_API_KEY_FILE = Path(__file__).resolve().parents[5] / "llm_api.txt"
_DEFAULT_TIMEOUT_S = 220.0
_DEFAULT_MAX_TOKENS = 16384
_DEFAULT_THINKING = "enabled"
_DEFAULT_REASONING_EFFORT = "low"


@dataclass(frozen=True, slots=True)
class DeepSeekConfig:
    """Explicit, environment-backed DeepSeek connection settings."""

    api_key: str
    base_url: str = _DEFAULT_BASE_URL
    model: str = _DEFAULT_MODEL
    timeout_s: float = _DEFAULT_TIMEOUT_S
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    thinking: str = _DEFAULT_THINKING
    reasoning_effort: str | None = _DEFAULT_REASONING_EFFORT

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or _read_api_key_file(
            _DEFAULT_API_KEY_FILE
        )
        if not api_key:
            raise ProviderError(
                "DEEPSEEK_API_KEY is not set and llm_api.txt is unavailable"
            )
        timeout_s = float(
            os.environ.get("DEEPSEEK_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S))
        )
        max_tokens = int(
            os.environ.get("DEEPSEEK_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS))
        )
        thinking = os.environ.get(
            "DEEPSEEK_THINKING", _DEFAULT_THINKING
        ).strip().lower()
        reasoning_effort = os.environ.get(
            "DEEPSEEK_REASONING_EFFORT", _DEFAULT_REASONING_EFFORT
        ) or None
        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL).rstrip("/"),
            model=os.environ.get("DEEPSEEK_MODEL", _DEFAULT_MODEL),
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("DeepSeek api_key must not be empty")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("DeepSeek base_url must be an HTTP(S) URL")
        if not self.model.strip():
            raise ValueError("DeepSeek model must not be empty")
        if self.timeout_s <= 0 or self.max_tokens <= 0:
            raise ValueError("DeepSeek timeout and max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("DeepSeek temperature must be in [0, 2]")
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("DeepSeek thinking must be 'enabled' or 'disabled'")
        if self.reasoning_effort is not None and self.reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("DeepSeek reasoning_effort must be 'low', 'high', 'max', or None")


class DeepSeekClient:
    """One-shot JSON-mode client for DeepSeek task planning."""

    name = "deepseek"

    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config
        self.model = config.model

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        return cls(DeepSeekConfig.from_env())

    def complete(self, *, system: str, user: str) -> ProviderResponse:
        if not system.strip() or not user.strip():
            raise ValueError("DeepSeek messages must not be empty")
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            # V4 models default to thinking enabled.  A bounded JSON planner
            # needs the answer channel, not an unbounded hidden reasoning
            # budget; callers can opt in with DEEPSEEK_THINKING=enabled.
            "thinking": {"type": self.config.thinking},
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_s) as response:
                raw = response.read()
                status = response.status
        except error.HTTPError as exc:
            detail = _read_error_body(exc)
            raise ProviderError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        except (error.URLError, IncompleteRead, TimeoutError, socket.timeout, OSError) as exc:
            raise ProviderError(f"DeepSeek transport failure: {exc}") from exc
        if status < 200 or status >= 300:
            raise ProviderError(f"DeepSeek unexpected HTTP status: {status}")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError("DeepSeek returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderError("DeepSeek response must be a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("DeepSeek response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
            usage = data.get("usage", {})
            completion_details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
            reasoning_tokens = (
                completion_details.get("reasoning_tokens")
                if isinstance(completion_details, dict)
                else None
            )
            detail = f"finish_reason={finish_reason!r}"
            if reasoning_tokens is not None:
                detail += f", reasoning_tokens={reasoning_tokens}"
            raise ProviderError(f"DeepSeek response choice has no text content ({detail})")
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        request_id = data.get("id")
        return ProviderResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=usage,
            request_id=request_id if isinstance(request_id, str) else None,
        )


def _read_api_key_file(path: Path) -> str:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return ""
    if len(lines) != 1:
        raise ProviderError("llm_api.txt must contain exactly one non-empty line")
    line = lines[0]
    prefix = "DEEPSEEK_API_KEY="
    if not line.startswith(prefix):
        raise ProviderError("llm_api.txt must use DEEPSEEK_API_KEY=<value>")
    value = line[len(prefix):].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    if not value:
        raise ProviderError("llm_api.txt contains an empty DEEPSEEK_API_KEY")
    return value


def _read_error_body(exc: error.HTTPError) -> str:
    try:
        raw = exc.read(4096)
        text = raw.decode("utf-8", errors="replace").strip()
        return text or "empty error body"
    except OSError:
        return "unreadable error body"


__all__ = ["DeepSeekClient", "DeepSeekConfig"]
