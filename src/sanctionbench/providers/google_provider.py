"""Google Gen AI standard-SDK adapter."""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

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


class GoogleProvider(Provider):
    provider_name = "google"
    protocol_version = "google-generate-content-json-schema-v4"
    sdk_distribution = "google-genai"
    citation_max_output_tokens = 4_096
    document_max_output_tokens = 16_384
    organic_max_output_tokens = 32_768

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        validate_authenticated_endpoint(
            self._identity_endpoint(),
            allowed_hosts={"generativelanguage.googleapis.com"},
            provider_name="Google",
        )

    def _identity_endpoint(self) -> str:
        api_client = getattr(self.client, "_api_client", None)
        options_reader = getattr(api_client, "get_read_only_http_options", None)
        if not callable(options_reader):
            raise RuntimeError("Google SDK did not expose effective HTTP options")
        endpoint = getattr(options_reader(), "base_url", None)
        if not isinstance(endpoint, str) or not endpoint:
            raise RuntimeError("Google SDK did not expose an effective base URL")
        return endpoint

    @staticmethod
    def _text(response: Any, *, task: str, max_output_tokens: int) -> str:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise ValueError(f"Google returned no {task} completion candidate")
        finish_reason = getattr(candidates[0], "finish_reason", None)
        finish_name = getattr(finish_reason, "name", str(finish_reason)).upper()
        if finish_name == "MAX_TOKENS":
            raise ValueError(f"Google {task} output reached the {max_output_tokens}-token limit")
        if finish_name != "STOP":
            raise ValueError(f"Google {task} output stopped with finish_reason={finish_name}")
        if not response.text or not response.text.strip():
            raise ValueError(f"Google returned no {task} content")
        return str(response.text)

    def predict(
        self,
        item: CitationItem,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> Prediction:
        response = self.client.models.generate_content(
            model=self.model,
            contents=build_user_prompt(item, condition, tool_evidence),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=self.citation_max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=PREDICTION_SCHEMA,
            ),
        )
        text = self._text(
            response,
            task="citation-audit",
            max_output_tokens=self.citation_max_output_tokens,
        )
        calls = []
        if tool_evidence is not None:
            calls.append({"name": "citation_lookup", "result": tool_evidence})
        return prediction_from_payload(item.item_id, parse_json_object(text), tool_calls=calls)

    def predict_document(
        self,
        scenario: DocumentScenario,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> DocumentPrediction:
        response = self.client.models.generate_content(
            model=self.model,
            contents=build_document_user_prompt(scenario, condition, tool_evidence),
            config=types.GenerateContentConfig(
                system_instruction=DOCUMENT_SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=self.document_max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=DOCUMENT_PREDICTION_SCHEMA,
            ),
        )
        text = self._text(
            response,
            task="document-audit",
            max_output_tokens=self.document_max_output_tokens,
        )
        calls = (
            [{"name": "citation_lookup_batch", "result": tool_evidence}] if tool_evidence else []
        )
        return document_prediction_from_payload(
            scenario.item_id, parse_json_object(text), tool_calls=calls
        )

    def predict_organic_document(
        self,
        document: OrganicDocumentInput,
        condition: Condition,
    ) -> OrganicDocumentPrediction:
        response = self.client.models.generate_content(
            model=self.model,
            contents=build_organic_document_user_prompt(document, condition),
            config=types.GenerateContentConfig(
                system_instruction=ORGANIC_DOCUMENT_SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=self.organic_max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
            ),
        )
        text = self._text(
            response,
            task="organic-document-audit",
            max_output_tokens=self.organic_max_output_tokens,
        )
        return organic_document_prediction_from_payload(
            document.item_id,
            parse_json_object(text),
            tool_calls=[],
        )
