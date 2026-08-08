"""Provider construction with environment-only credential checks."""

from __future__ import annotations

import os

from .anthropic_provider import AnthropicProvider
from .base import Provider
from .deepseek_provider import DeepSeekProvider
from .google_provider import GoogleProvider
from .mock import MockProvider
from .openai_provider import OpenAIProvider


def create_provider(name: str, model: str) -> Provider:
    normalized = name.lower()
    if normalized == "mock":
        return MockProvider(model)
    if normalized == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        return OpenAIProvider(model)
    if normalized == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY is required for provider=deepseek")
        return DeepSeekProvider(model)
    if normalized == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")
        return AnthropicProvider(model)
    if normalized in {"google", "gemini"}:
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for provider=google")
        return GoogleProvider(model)
    raise ValueError(f"Unknown provider: {name}")
