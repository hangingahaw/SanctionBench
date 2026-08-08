from __future__ import annotations

from types import SimpleNamespace

import pytest

import sanctionbench.providers.openai_provider as openai_module
from sanctionbench.providers.openai_provider import OpenAIProvider


def _response(*, finish_reason: str, content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_openai_content_requires_complete_nonempty_json_text() -> None:
    assert (
        OpenAIProvider._content(
            _response(finish_reason="stop", content='{"answer":"ok"}'),
            task="fixture",
            max_output_tokens=100,
        )
        == '{"answer":"ok"}'
    )

    with pytest.raises(ValueError, match="100-token limit"):
        OpenAIProvider._content(
            _response(finish_reason="length", content='{"answer":'),
            task="fixture",
            max_output_tokens=100,
        )
    with pytest.raises(ValueError, match="no fixture content"):
        OpenAIProvider._content(
            _response(finish_reason="stop", content=""),
            task="fixture",
            max_output_tokens=100,
        )


def test_openai_constructor_disables_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        base_url = "https://api.openai.com/v1"

    def fake_openai(**kwargs: object) -> FakeClient:
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(openai_module, "OpenAI", fake_openai)
    provider = OpenAIProvider("fixture-model")

    assert captured == {"max_retries": 0}
    assert provider.runtime_identity()["sdk_max_retries"] == "0"
