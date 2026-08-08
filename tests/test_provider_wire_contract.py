from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from sanctionbench.providers.base import (
    DOCUMENT_PREDICTION_SCHEMA,
    ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
    document_prediction_from_payload,
    organic_document_prediction_from_payload,
    prediction_from_payload,
    validate_authenticated_endpoint,
)


def _citation_payload() -> dict[str, Any]:
    return {
        "predicted_label": "real",
        "fake_probability": 0.1,
        "rationale": "The authority appears valid.",
        "cited_evidence": [],
    }


def _document_payload() -> dict[str, Any]:
    return {
        "assessments": [
            {
                "authority_id": "A01",
                "predicted_label": "uncertain_needs_review",
                "fake_probability": 0.5,
                "rationale": "Verification is incomplete.",
            }
        ]
    }


def _organic_payload() -> dict[str, Any]:
    return {
        "findings": [
            {
                "citation_text": "Imaginary v. Fiction, 123 F.9th 1",
                "case_name": "Imaginary v. Fiction",
                "page_number": 1,
                "quoted_text": "Imaginary v. Fiction, 123 F.9th 1, controls.",
                "predicted_label": "nonexistent_case",
                "fake_probability": 0.9,
                "rationale": "The citation warrants verification.",
            }
        ]
    }


@pytest.mark.parametrize(
    ("parser", "payload_factory"),
    [
        (prediction_from_payload, _citation_payload),
        (document_prediction_from_payload, _document_payload),
        (organic_document_prediction_from_payload, _organic_payload),
    ],
)
def test_provider_wire_parsers_reject_unknown_top_level_keys(
    parser: Callable[..., object], payload_factory: Callable[[], dict[str, Any]]
) -> None:
    payload = payload_factory()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parser("item-1", payload, tool_calls=[])


def test_citation_wire_parser_requires_cited_evidence() -> None:
    payload = _citation_payload()
    del payload["cited_evidence"]

    with pytest.raises(ValidationError, match="Field required"):
        prediction_from_payload("item-1", payload, tool_calls=[])


@pytest.mark.parametrize(
    ("field", "value"),
    [("page_number", "1"), ("fake_probability", "0.9")],
)
def test_organic_wire_parser_rejects_numeric_strings(field: str, value: str) -> None:
    payload = _organic_payload()
    payload["findings"][0][field] = value

    with pytest.raises(ValidationError):
        organic_document_prediction_from_payload("item-1", payload, tool_calls=[])


def test_organic_wire_parser_rejects_real_findings_and_nested_extras() -> None:
    real_payload = _organic_payload()
    real_payload["findings"][0]["predicted_label"] = "real"
    with pytest.raises(ValidationError):
        organic_document_prediction_from_payload("item-1", real_payload, tool_calls=[])

    extra_payload = _organic_payload()
    extra_payload["findings"][0]["unexpected"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        organic_document_prediction_from_payload("item-1", extra_payload, tool_calls=[])


@pytest.mark.parametrize(
    "schema",
    [PREDICTION_SCHEMA, DOCUMENT_PREDICTION_SCHEMA, ORGANIC_DOCUMENT_PREDICTION_SCHEMA],
)
def test_provider_json_schemas_forbid_unknown_top_level_keys(schema: dict[str, Any]) -> None:
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False


def test_organic_schema_is_exact_and_forbids_nested_extras() -> None:
    assert ORGANIC_DOCUMENT_PREDICTION_SCHEMA["required"] == ["findings"]
    finding_ref = ORGANIC_DOCUMENT_PREDICTION_SCHEMA["properties"]["findings"]["items"]["$ref"]
    finding_schema = ORGANIC_DOCUMENT_PREDICTION_SCHEMA["$defs"][finding_ref.rsplit("/", 1)[1]]

    assert finding_schema["additionalProperties"] is False
    assert set(finding_schema["required"]) == {
        "citation_text",
        "case_name",
        "page_number",
        "quoted_text",
        "predicted_label",
        "fake_probability",
        "rationale",
    }
    assert finding_schema["properties"]["predicted_label"]["enum"] == [
        "nonexistent_case",
        "fabricated_quote",
        "misattributed_holding",
        "uncertain_needs_review",
    ]
    assert ORGANIC_DOCUMENT_PREDICTION_SCHEMA["properties"]["findings"]["maxItems"] == 512


def test_provider_wire_contract_enforces_collection_and_string_budgets() -> None:
    citation = _citation_payload()
    citation["cited_evidence"] = ["evidence"] * 21
    with pytest.raises(ValidationError):
        prediction_from_payload("item-1", citation, tool_calls=[])

    citation = _citation_payload()
    citation["cited_evidence"] = ["x" * 2_001]
    with pytest.raises(ValidationError):
        prediction_from_payload("item-1", citation, tool_calls=[])

    document = _document_payload()
    document["assessments"] = document["assessments"] * 513
    with pytest.raises(ValidationError):
        document_prediction_from_payload("item-1", document, tool_calls=[])

    organic = _organic_payload()
    organic["findings"] = organic["findings"] * 513
    with pytest.raises(ValidationError):
        organic_document_prediction_from_payload("item-1", organic, tool_calls=[])


def test_authenticated_provider_endpoint_is_scoped_to_approved_https_origin() -> None:
    validate_authenticated_endpoint(
        "https://api.openai.com/v1",
        allowed_hosts={"api.openai.com"},
        provider_name="OpenAI",
    )
    with pytest.raises(ValueError, match="approved HTTPS origin"):
        validate_authenticated_endpoint(
            "https://proxy.example/v1",
            allowed_hosts={"api.openai.com"},
            provider_name="OpenAI",
        )
    with pytest.raises(ValueError, match="approved HTTPS origin"):
        validate_authenticated_endpoint(
            "http://api.openai.com/v1",
            allowed_hosts={"api.openai.com"},
            provider_name="OpenAI",
        )

    organic = _organic_payload()
    organic["findings"][0]["page_number"] = 100_001
    with pytest.raises(ValidationError):
        organic_document_prediction_from_payload("item-1", organic, tool_calls=[])
