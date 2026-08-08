from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import sanctionbench.providers.anthropic_provider as anthropic_module
from sanctionbench.providers.anthropic_provider import AnthropicProvider


class _Messages:
    def __init__(self, *, tool_input: object, stop_reason: str | None = "tool_use") -> None:
        self.tool_input = tool_input
        self.stop_reason = stop_reason
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.request = kwargs
        tool_name = str(kwargs["tool_choice"]["name"])
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name=tool_name, input=self.tool_input)],
            stop_reason=self.stop_reason,
        )


def test_anthropic_constructor_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        base_url = "https://api.anthropic.com"

    def fake_anthropic(**kwargs: object) -> FakeClient:
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(anthropic_module, "Anthropic", fake_anthropic)
    provider = AnthropicProvider("fixture-model")

    assert captured == {"max_retries": 0}
    assert provider.runtime_identity()["sdk_max_retries"] == "0"


def test_anthropic_structured_completion_forces_named_schema_tool() -> None:
    messages = _Messages(tool_input={"answer": "ok"})
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "fixture-model"
    provider.client = SimpleNamespace(messages=messages)

    result = provider._structured_completion(
        system="system",
        user="user",
        schema={"type": "object", "required": ["answer"]},
        tool_name="submit_fixture",
        max_tokens=100,
    )

    assert result == {"answer": "ok"}
    assert messages.request is not None
    assert messages.request["temperature"] == 0
    assert messages.request["tools"][0]["input_schema"] == {
        "type": "object",
        "required": ["answer"],
    }
    assert messages.request["tool_choice"] == {"type": "tool", "name": "submit_fixture"}


def test_anthropic_structured_completion_rejects_nonobject_input() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "fixture-model"
    provider.client = SimpleNamespace(messages=_Messages(tool_input="not-an-object"))

    with pytest.raises(ValueError, match="did not contain an object"):
        provider._structured_completion(
            system="system",
            user="user",
            schema={"type": "object"},
            tool_name="submit_fixture",
            max_tokens=100,
        )


def test_anthropic_structured_completion_rejects_token_truncation() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "fixture-model"
    provider.client = SimpleNamespace(
        messages=_Messages(tool_input={"answer": "partial"}, stop_reason="max_tokens")
    )

    with pytest.raises(ValueError, match="100-token limit"):
        provider._structured_completion(
            system="system",
            user="user",
            schema={"type": "object"},
            tool_name="submit_fixture",
            max_tokens=100,
        )


def test_anthropic_structured_completion_requires_positive_stop_reason() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "fixture-model"
    provider.client = SimpleNamespace(
        messages=_Messages(tool_input={"answer": "looks-complete"}, stop_reason=None)
    )

    with pytest.raises(ValueError, match="stop_reason=None"):
        provider._structured_completion(
            system="system",
            user="user",
            schema={"type": "object"},
            tool_name="submit_fixture",
            max_tokens=100,
        )
