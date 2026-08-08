"""Anthropic standard-SDK adapter."""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from sanctionbench.models import (
    CitationItem,
    Condition,
    DocumentPrediction,
    DocumentScenario,
    OrganicDocumentInput,
    OrganicDocumentPrediction,
    Prediction,
)

from .base import (
    DOCUMENT_PREDICTION_SCHEMA,
    DOCUMENT_SYSTEM_PROMPT,
    ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
    ORGANIC_DOCUMENT_SYSTEM_PROMPT,
    PREDICTION_SCHEMA,
    SYSTEM_PROMPT,
    Provider,
    build_document_user_prompt,
    build_organic_document_user_prompt,
    build_user_prompt,
    document_prediction_from_payload,
    organic_document_prediction_from_payload,
    prediction_from_payload,
    validate_authenticated_endpoint,
)


class AnthropicProvider(Provider):
    provider_name = "anthropic"
    protocol_version = "anthropic-messages-forced-tool-json-schema-v5"
    sdk_distribution = "anthropic"
    sdk_max_retries = 0

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = Anthropic(max_retries=self.sdk_max_retries)
        validate_authenticated_endpoint(
            str(self.client.base_url),
            allowed_hosts={"api.anthropic.com"},
            provider_name="Anthropic",
        )

    def _identity_endpoint(self) -> str:
        return str(self.client.base_url)

    def _structured_completion(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        tool_name: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Submit the complete SanctionBench structured prediction.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason != "tool_use":
            if stop_reason == "max_tokens":
                raise ValueError(
                    f"Anthropic structured output reached the {max_tokens}-token limit"
                )
            raise ValueError(f"Anthropic structured output stopped with stop_reason={stop_reason}")
        matches = [
            block
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == tool_name
        ]
        if len(matches) != 1:
            raise ValueError(f"Anthropic returned {len(matches)} matching structured tool calls")
        payload = getattr(matches[0], "input", None)
        if not isinstance(payload, dict):
            raise ValueError("Anthropic structured tool call did not contain an object")
        return payload

    def predict(
        self,
        item: CitationItem,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> Prediction:
        payload = self._structured_completion(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(item, condition, tool_evidence),
            schema=PREDICTION_SCHEMA,
            tool_name="submit_sanctionbench_prediction",
            max_tokens=700,
        )
        calls = []
        if tool_evidence is not None:
            calls.append({"name": "citation_lookup", "result": tool_evidence})
        return prediction_from_payload(item.item_id, payload, tool_calls=calls)

    def predict_document(
        self,
        scenario: DocumentScenario,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> DocumentPrediction:
        payload = self._structured_completion(
            system=DOCUMENT_SYSTEM_PROMPT,
            user=build_document_user_prompt(scenario, condition, tool_evidence),
            schema=DOCUMENT_PREDICTION_SCHEMA,
            tool_name="submit_sanctionbench_document_prediction",
            max_tokens=3_500,
        )
        calls = (
            [{"name": "citation_lookup_batch", "result": tool_evidence}] if tool_evidence else []
        )
        return document_prediction_from_payload(scenario.item_id, payload, tool_calls=calls)

    def predict_organic_document(
        self,
        document: OrganicDocumentInput,
        condition: Condition,
    ) -> OrganicDocumentPrediction:
        payload = self._structured_completion(
            system=ORGANIC_DOCUMENT_SYSTEM_PROMPT,
            user=build_organic_document_user_prompt(document, condition),
            schema=ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
            tool_name="submit_sanctionbench_organic_document_prediction",
            max_tokens=6_000,
        )
        return organic_document_prediction_from_payload(
            document.item_id,
            payload,
            tool_calls=[],
        )
