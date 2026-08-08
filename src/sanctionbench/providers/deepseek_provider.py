"""DeepSeek OpenAI-compatible adapter with explicit JSON-output mode."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlsplit

from openai import OpenAI

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
    parse_json_object,
    prediction_from_payload,
)


class DeepSeekProvider(Provider):
    """Call a named immutable DeepSeek model through its hosted API."""

    provider_name = "deepseek"
    protocol_version = "deepseek-chat-completions-json-mode-schema-prompt-v5"
    sdk_distribution = "openai"
    sdk_max_retries = 0
    max_output_tokens = 32_768

    def __init__(self, model: str) -> None:
        super().__init__(model)
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme.casefold() != "https"
            or parsed.hostname != "api.deepseek.com"
            or parsed.port not in {None, 443}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "DEEPSEEK_BASE_URL must be the official HTTPS api.deepseek.com origin; "
                "custom OpenAI-compatible endpoints require a separate adapter and credential"
            )
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=base_url,
            max_retries=self.sdk_max_retries,
        )

    def _identity_endpoint(self) -> str:
        return str(self.client.base_url)

    def runtime_identity(self) -> dict[str, str]:
        identity = super().runtime_identity()
        identity["max_output_tokens"] = str(self.max_output_tokens)
        return identity

    def _json_completion(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        schema_instruction = (
            " Return a JSON object only, using exactly this JSON Schema: "
            + json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system + schema_instruction},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_output_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise ValueError(
                f"DeepSeek JSON output reached the {self.max_output_tokens}-token limit"
            )
        if finish_reason != "stop":
            raise ValueError(f"DeepSeek JSON output stopped with finish_reason={finish_reason}")
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise ValueError("DeepSeek returned no JSON content")
        return parse_json_object(content)

    def predict(
        self,
        item: CitationItem,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> Prediction:
        payload = self._json_completion(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(item, condition, tool_evidence),
            schema=PREDICTION_SCHEMA,
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
        payload = self._json_completion(
            system=DOCUMENT_SYSTEM_PROMPT,
            user=build_document_user_prompt(scenario, condition, tool_evidence),
            schema=DOCUMENT_PREDICTION_SCHEMA,
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
        payload = self._json_completion(
            system=ORGANIC_DOCUMENT_SYSTEM_PROMPT,
            user=build_organic_document_user_prompt(document, condition),
            schema=ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
        )
        return organic_document_prediction_from_payload(
            document.item_id,
            payload,
            tool_calls=[],
        )
