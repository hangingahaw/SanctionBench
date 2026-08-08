"""OpenAI standard-SDK adapter."""

from __future__ import annotations

from typing import Any

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
    validate_authenticated_endpoint,
)


class OpenAIProvider(Provider):
    provider_name = "openai"
    protocol_version = "openai-chat-completions-json-schema-v4"
    sdk_distribution = "openai"
    sdk_max_retries = 0
    citation_max_output_tokens = 4_096
    document_max_output_tokens = 16_384
    organic_max_output_tokens = 32_768

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = OpenAI(max_retries=self.sdk_max_retries)
        validate_authenticated_endpoint(
            str(self.client.base_url),
            allowed_hosts={"api.openai.com"},
            provider_name="OpenAI",
        )

    def _identity_endpoint(self) -> str:
        return str(self.client.base_url)

    @staticmethod
    def _content(response: Any, *, task: str, max_output_tokens: int) -> str:
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason == "length":
            raise ValueError(f"OpenAI {task} output reached the {max_output_tokens}-token limit")
        if finish_reason != "stop":
            raise ValueError(f"OpenAI {task} output stopped with finish_reason={finish_reason}")
        content = choice.message.content
        if content is None or not content.strip():
            raise ValueError(f"OpenAI returned no {task} content")
        return str(content)

    def predict(
        self,
        item: CitationItem,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> Prediction:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item, condition, tool_evidence)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sanctionbench_prediction",
                    "strict": True,
                    "schema": PREDICTION_SCHEMA,
                },
            },
            max_completion_tokens=self.citation_max_output_tokens,
        )
        content = self._content(
            response,
            task="citation-audit",
            max_output_tokens=self.citation_max_output_tokens,
        )
        calls = []
        if tool_evidence is not None:
            calls.append({"name": "citation_lookup", "result": tool_evidence})
        return prediction_from_payload(item.item_id, parse_json_object(content), tool_calls=calls)

    def predict_document(
        self,
        scenario: DocumentScenario,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> DocumentPrediction:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": DOCUMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_document_user_prompt(scenario, condition, tool_evidence),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sanctionbench_document_prediction",
                    "strict": True,
                    "schema": DOCUMENT_PREDICTION_SCHEMA,
                },
            },
            max_completion_tokens=self.document_max_output_tokens,
        )
        content = self._content(
            response,
            task="document-audit",
            max_output_tokens=self.document_max_output_tokens,
        )
        calls = (
            [{"name": "citation_lookup_batch", "result": tool_evidence}] if tool_evidence else []
        )
        return document_prediction_from_payload(
            scenario.item_id, parse_json_object(content), tool_calls=calls
        )

    def predict_organic_document(
        self,
        document: OrganicDocumentInput,
        condition: Condition,
    ) -> OrganicDocumentPrediction:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": ORGANIC_DOCUMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_organic_document_user_prompt(document, condition),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sanctionbench_organic_document_prediction",
                    "strict": True,
                    "schema": ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
                },
            },
            max_completion_tokens=self.organic_max_output_tokens,
        )
        content = self._content(
            response,
            task="organic-document-audit",
            max_output_tokens=self.organic_max_output_tokens,
        )
        return organic_document_prediction_from_payload(
            document.item_id,
            parse_json_object(content),
            tool_calls=[],
        )
