from __future__ import annotations

import json
from http.client import IncompleteRead
from urllib.error import HTTPError

import pytest

from r1pro_data_gen.planning.llm.providers.deepseek import DeepSeekClient, DeepSeekConfig
from r1pro_data_gen.planning.llm.providers.protocol import ProviderError


def test_deepseek_config_prefers_environment(monkeypatch, tmp_path):
    key_file = tmp_path / "llm_api.txt"
    key_file.write_text("DEEPSEEK_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setattr(
        "r1pro_data_gen.planning.llm.providers.deepseek._DEFAULT_API_KEY_FILE",
        key_file,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    client = DeepSeekClient.from_env()
    assert client.model == "deepseek-v4-flash"
    assert client.config.api_key == "test-secret"


def test_deepseek_config_falls_back_to_ignored_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "llm_api.txt"
    key_file.write_text("DEEPSEEK_API_KEY='file-secret'\n", encoding="utf-8")
    monkeypatch.setattr(
        "r1pro_data_gen.planning.llm.providers.deepseek._DEFAULT_API_KEY_FILE",
        key_file,
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    client = DeepSeekClient.from_env()

    assert client.config.api_key == "file-secret"


def test_deepseek_config_uses_validated_planner_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    for name in (
        "DEEPSEEK_MODEL",
        "DEEPSEEK_THINKING",
        "DEEPSEEK_REASONING_EFFORT",
        "DEEPSEEK_MAX_TOKENS",
        "DEEPSEEK_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)

    client = DeepSeekClient.from_env()

    assert client.model == "deepseek-v4-flash"
    assert client.config.thinking == "enabled"
    assert client.config.reasoning_effort == "low"
    assert client.config.max_tokens == 16384
    assert client.config.timeout_s == 220.0


def test_deepseek_config_can_opt_into_thinking_mode(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "high")
    client = DeepSeekClient.from_env()
    assert client.config.thinking == "enabled"
    assert client.config.reasoning_effort == "high"


def test_deepseek_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        "r1pro_data_gen.planning.llm.providers.deepseek._DEFAULT_API_KEY_FILE",
        tmp_path / "missing.txt",
    )
    with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient.from_env()


def test_deepseek_request_and_response(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "id": "req-1",
                    "choices": [{"message": {"content": '{"status":"unsupported"}'}}],
                    "usage": {"total_tokens": 3},
                }
            ).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data)
        captured["auth"] = req.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr("r1pro_data_gen.planning.llm.providers.deepseek.request.urlopen", fake_urlopen)
    client = DeepSeekClient(DeepSeekConfig(api_key="secret", timeout_s=7.0))
    response = client.complete(system="system", user="user")
    assert response.text == '{"status":"unsupported"}'
    assert captured["url"].endswith("/chat/completions")
    assert captured["timeout"] == 7.0
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "low"
    assert captured["auth"] == "Bearer secret"


def test_deepseek_http_error_is_normalized(monkeypatch):
    def fake_urlopen(req, timeout):
        raise HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("r1pro_data_gen.planning.llm.providers.deepseek.request.urlopen", fake_urlopen)
    client = DeepSeekClient(DeepSeekConfig(api_key="secret"))
    with pytest.raises(ProviderError, match="HTTP 401"):
        client.complete(system="system", user="user")


def test_deepseek_incomplete_response_is_normalized(monkeypatch):
    class TruncatedResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            raise IncompleteRead(b"")

    monkeypatch.setattr(
        "r1pro_data_gen.planning.llm.providers.deepseek.request.urlopen",
        lambda req, timeout: TruncatedResponse(),
    )
    client = DeepSeekClient(DeepSeekConfig(api_key="secret"))
    with pytest.raises(ProviderError, match="DeepSeek transport failure"):
        client.complete(system="system", user="user")
