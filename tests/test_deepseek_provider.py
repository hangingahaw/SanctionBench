from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import sanctionbench.providers.deepseek_provider as deepseek_module
from sanctionbench.providers.deepseek_provider import DeepSeekProvider


class _Completions:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"answer":"ok"}'),
                )
            ]
        )


def test_deepseek_constructor_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        base_url = "https://api.deepseek.com"

    def fake_openai(**kwargs: object) -> FakeClient:
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-not-a-secret")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setattr(deepseek_module, "OpenAI", fake_openai)
    provider = DeepSeekProvider("fixture-model")

    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["max_retries"] == 0
    assert provider.runtime_identity()["sdk_max_retries"] == "0"


def test_deepseek_json_completion_uses_json_mode_and_disables_thinking() -> None:
    completions = _Completions()
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.model = "deepseek-v4-flash"
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = provider._json_completion(
        system="system", user="user", schema={"type": "object", "required": ["answer"]}
    )

    assert result == {"answer": "ok"}
    assert completions.request is not None
    assert completions.request["model"] == "deepseek-v4-flash"
    assert completions.request["response_format"] == {"type": "json_object"}
    assert completions.request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert completions.request["temperature"] == 0
    assert completions.request["max_tokens"] == 32_768
    assert '"required":["answer"]' in completions.request["messages"][0]["content"]


def test_deepseek_runtime_identity_hashes_effective_endpoint() -> None:
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.model = "deepseek-v4-flash"
    provider.client = SimpleNamespace(base_url="https://example.test/v1")

    identity = provider.runtime_identity()

    assert identity["protocol_version"] == provider.protocol_version
    assert identity["max_output_tokens"] == "32768"
    assert len(identity["endpoint_sha256"]) == 64
    assert "example.test" not in str(identity)


def test_provider_identity_rejects_credentials_in_base_url() -> None:
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.model = "deepseek-v4-flash"
    provider.client = SimpleNamespace(base_url="https://token@example.test/v1")

    try:
        provider.runtime_identity()
    except ValueError as error:
        assert "must not contain credentials" in str(error)
    else:
        raise AssertionError("credential-bearing provider base URL was accepted")


def test_deepseek_rejects_length_truncated_json_before_parsing() -> None:
    class TruncatedCompletions:
        def create(self, **_: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content='{"answer":"unterminated'),
                    )
                ]
            )

    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.model = "deepseek-v4-flash"
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=TruncatedCompletions()))

    try:
        provider._json_completion(
            system="system",
            user="user",
            schema={"type": "object"},
        )
    except ValueError as error:
        assert "32768-token limit" in str(error)
    else:
        raise AssertionError("length-truncated JSON was accepted")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.deepseek.com",
        "https://attacker.example/v1",
        "https://api.deepseek.com:8443/v1",
        "https://api.deepseek.com/v1?forward=1",
    ],
)
def test_deepseek_constructor_rejects_nonofficial_authenticated_origins(
    base_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", base_url)
    with pytest.raises(ValueError, match="official HTTPS"):
        DeepSeekProvider("deepseek-v4-flash")
