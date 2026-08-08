"""Deterministic no-spend client used to prove the full benchmark loop."""

from __future__ import annotations

import hashlib
from typing import Any

from sanctionbench.models import (
    CitationItem,
    Condition,
    DocumentAssessment,
    DocumentPrediction,
    DocumentScenario,
    OrganicDocumentInput,
    OrganicDocumentPrediction,
    PredictedLabel,
    Prediction,
)

from .base import Provider


class MockProvider(Provider):
    provider_name = "mock"
    protocol_version = "deterministic-mock-v1"
    is_mock = True

    def predict(
        self,
        item: CitationItem,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> Prediction:
        tool_calls: list[dict[str, Any]] = []
        if condition == Condition.TOOL_ASSISTED and tool_evidence is not None:
            tool_calls.append(
                {
                    "name": "citation_lookup",
                    "arguments": {
                        "citation": item.citation,
                        "case_name": item.case_name,
                        "proposition": item.proposition,
                    },
                    "result": tool_evidence,
                }
            )
            matches = tool_evidence.get("matches", [])
            if int(tool_evidence.get("exact_identity_match_count", 0)) == 0:
                return Prediction(
                    item_id=item.item_id,
                    predicted_label=PredictedLabel.NONEXISTENT_CASE,
                    fake_probability=0.94,
                    rationale=(
                        "The standardized CourtListener lookup returned no matching "
                        "case-name/citation identity."
                    ),
                    cited_evidence=[str(tool_evidence.get("source", ""))],
                    tool_calls=tool_calls,
                )
            if tool_evidence.get("claim_search", {}).get("reported_count") == 0:
                return Prediction(
                    item_id=item.item_id,
                    predicted_label=PredictedLabel.MISATTRIBUTED_HOLDING,
                    fake_probability=0.82,
                    rationale=(
                        "The authority resolved, but exact proposition search returned no matching case; "
                        "the mock flags this for substantive review."
                    ),
                    cited_evidence=[str(tool_evidence.get("source", ""))],
                    tool_calls=tool_calls,
                )
            return Prediction(
                item_id=item.item_id,
                predicted_label=PredictedLabel.REAL,
                fake_probability=0.18,
                rationale="The lookup returned at least one candidate; this mock does not compare holdings.",
                cited_evidence=[str(match.get("url", "")) for match in matches[:2]],
                tool_calls=tool_calls,
            )

        digest = hashlib.sha256(f"{item.case_name}|{item.citation}".encode()).digest()
        suspicious = digest[0] < 96
        if suspicious:
            label = PredictedLabel.NONEXISTENT_CASE
            probability = 0.68
            rationale = (
                "Deterministic closed-book mock heuristic marked the unfamiliar string for review."
            )
        else:
            label = PredictedLabel.REAL
            probability = 0.27
            rationale = "Deterministic closed-book mock heuristic did not flag the authority."
        return Prediction(
            item_id=item.item_id,
            predicted_label=label,
            fake_probability=probability,
            rationale=rationale,
            cited_evidence=[],
            tool_calls=tool_calls,
        )

    def predict_document(
        self,
        scenario: DocumentScenario,
        condition: Condition,
        tool_evidence: dict[str, Any] | None,
    ) -> DocumentPrediction:
        assessments: list[DocumentAssessment] = []
        calls: list[dict[str, Any]] = []
        for authority in scenario.authorities:
            evidence = tool_evidence.get(authority.authority_id) if tool_evidence else None
            claim_missing = False
            if evidence is not None:
                calls.append(
                    {
                        "name": "citation_lookup",
                        "authority_id": authority.authority_id,
                        "result": evidence,
                    }
                )
                suspicious = int(evidence.get("exact_identity_match_count", 0)) == 0
                claim_missing = evidence.get("claim_search", {}).get("reported_count") == 0
                if suspicious:
                    probability = 0.94
                    rationale = "Standardized lookup returned no matching authority."
                elif claim_missing:
                    probability = 0.82
                    rationale = "Authority resolved but exact proposition search returned no match."
                else:
                    probability = 0.18
                    rationale = "Lookup returned a candidate and no contrary proposition signal."
            else:
                digest = hashlib.sha256(
                    f"{authority.authority_id}|{authority.citation}".encode()
                ).digest()
                suspicious = digest[0] < 96
                probability = 0.68 if suspicious else 0.27
                rationale = "Deterministic no-spend document heuristic."
            assessments.append(
                DocumentAssessment(
                    authority_id=authority.authority_id,
                    predicted_label=(
                        PredictedLabel.NONEXISTENT_CASE
                        if suspicious
                        else (
                            PredictedLabel.MISATTRIBUTED_HOLDING
                            if claim_missing
                            else PredictedLabel.REAL
                        )
                    ),
                    fake_probability=probability,
                    rationale=rationale,
                )
            )
        return DocumentPrediction(
            item_id=scenario.item_id,
            assessments=assessments,
            tool_calls=calls,
        )

    def predict_organic_document(
        self,
        document: OrganicDocumentInput,
        condition: Condition,
    ) -> OrganicDocumentPrediction:
        # The mock receives only model-visible input and deliberately has no oracle access to gold.
        # Returning no findings proves checkpointing and grading without creating fake performance.
        return OrganicDocumentPrediction(item_id=document.item_id, findings=[], tool_calls=[])
