from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sanctionbench.models import Condition, OrganicDocumentInput
from sanctionbench.providers.base import ORGANIC_DOCUMENT_PREDICTION_SCHEMA
from sanctionbench.providers.google_provider import GoogleProvider


def test_google_runtime_identity_hashes_effective_sdk_endpoint() -> None:
    provider = GoogleProvider.__new__(GoogleProvider)
    provider.model = "fixture-model"
    provider.client = SimpleNamespace(
        _api_client=SimpleNamespace(
            get_read_only_http_options=lambda: SimpleNamespace(
                base_url="https://generativelanguage.example.test/"
            )
        )
    )

    identity = provider.runtime_identity()

    assert identity["protocol_version"] == provider.protocol_version
    assert len(identity["endpoint_sha256"]) == 64
    assert "example.test" not in str(identity)


def test_google_organic_request_uses_json_schema_field_for_generated_contract() -> None:
    class Models:
        request: dict[str, Any] | None = None

        def generate_content(self, **kwargs: Any) -> SimpleNamespace:
            self.request = kwargs
            return SimpleNamespace(
                text='{"findings":[]}',
                candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
            )

    models = Models()
    provider = GoogleProvider.__new__(GoogleProvider)
    provider.model = "fixture-model"
    provider.client = SimpleNamespace(models=models)
    document = OrganicDocumentInput(
        item_id="organic-1",
        title="Fixture",
        document_markdown="<!-- SANCTIONBENCH_PAGE:1 -->\n\n## Page 1\n\nText\n",
        document_markdown_sha256="a" * 64,
        page_count=1,
    )

    prediction = provider.predict_organic_document(document, Condition.CLOSED_BOOK)

    assert prediction.findings == []
    assert models.request is not None
    config = models.request["config"]
    assert config.response_schema is None
    assert config.response_json_schema == ORGANIC_DOCUMENT_PREDICTION_SCHEMA
    assert config.max_output_tokens == 32_768


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [SimpleNamespace(finish_reason=None)],
        [SimpleNamespace(finish_reason=SimpleNamespace(name="FINISH_REASON_UNSPECIFIED"))],
    ],
)
def test_google_text_requires_positive_stop_reason(candidates: list[SimpleNamespace]) -> None:
    response = SimpleNamespace(text='{"findings":[]}', candidates=candidates)

    with pytest.raises(ValueError, match="completion candidate|finish_reason"):
        GoogleProvider._text(response, task="organic-document", max_output_tokens=100)
