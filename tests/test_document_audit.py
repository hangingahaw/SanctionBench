from __future__ import annotations

from conftest import make_item

from sanctionbench.document_audit import build_document_scenarios, score_document_predictions
from sanctionbench.models import (
    DocumentAssessment,
    DocumentPrediction,
    GoldLabel,
    PredictedLabel,
)


def test_constructed_document_track_round_trips_through_grader() -> None:
    items = [
        make_item("fake-1", GoldLabel.NONEXISTENT_CASE, "pair-1"),
        make_item("real-1", GoldLabel.REAL, "pair-1"),
        make_item("fake-2", GoldLabel.FABRICATED_QUOTE, "pair-2"),
        make_item("real-2", GoldLabel.REAL, "pair-2"),
    ]
    scenarios = build_document_scenarios(items, seed=7)
    predictions = []
    by_id = {item.item_id: item for item in items}
    for scenario in scenarios:
        assessments = []
        for authority in scenario.authorities:
            item = by_id[authority.citation_item_id]
            assessments.append(
                DocumentAssessment(
                    authority_id=authority.authority_id,
                    predicted_label=PredictedLabel(item.gold_label.value),
                    fake_probability=0.99 if item.binary_gold == "fake" else 0.01,
                    rationale="fixture oracle",
                )
            )
        predictions.append(DocumentPrediction(item_id=scenario.item_id, assessments=assessments))
    metrics = score_document_predictions(scenarios, predictions, by_id)
    assert metrics["headline"]["score"] == 100
    assert metrics["document_count"] == len(scenarios)
    assert metrics["document_operational"] == {
        "name": "DocumentSanctionScore",
        "score": 100.0,
        "definition": "macro_document_hallucination_recall",
        "clean_audit_rate": 1.0,
        "zero_false_positive_document_rate": 1.0,
        "documents_with_false_positive": 0,
        "extra_verifications_per_document": 0.0,
        "macro_fake_recall": 1.0,
        "false_accusations_per_100_real_authorities": 0.0,
    }


def test_document_false_accusations_are_reported_as_review_workload() -> None:
    items = [
        make_item("fake-1", GoldLabel.NONEXISTENT_CASE, "pair-1"),
        make_item("real-1", GoldLabel.REAL, "pair-1"),
    ]
    scenarios = build_document_scenarios(items, seed=9)
    scenario = scenarios[0]
    by_id = {item.item_id: item for item in items}
    assessments = []
    for authority in scenario.authorities:
        item = by_id[authority.citation_item_id]
        assessments.append(
            DocumentAssessment(
                authority_id=authority.authority_id,
                predicted_label=(
                    PredictedLabel.NONEXISTENT_CASE
                    if item.binary_gold == "fake"
                    else PredictedLabel.UNCERTAIN_NEEDS_REVIEW
                ),
                fake_probability=0.9,
                rationale="fixture false accusation",
            )
        )
    metrics = score_document_predictions(
        scenarios,
        [DocumentPrediction(item_id=scenario.item_id, assessments=assessments)],
        by_id,
    )
    assert metrics["document_operational"]["score"] == 100
    assert metrics["document_operational"]["clean_audit_rate"] == 1
    assert metrics["document_operational"]["extra_verifications_per_document"] == 12
    assert metrics["document_operational"]["zero_false_positive_document_rate"] == 0
    assert metrics["document_operational"]["documents_with_false_positive"] == 1
    assert metrics["by_document"][scenario.item_id]["false_positive"] == 12


def test_one_missed_hallucination_fails_clean_audit() -> None:
    items = [
        make_item("fake-1", GoldLabel.NONEXISTENT_CASE, "pair-1"),
        make_item("real-1", GoldLabel.REAL, "pair-1"),
    ]
    scenarios = build_document_scenarios(items, seed=11)
    scenario = scenarios[0]
    by_id = {item.item_id: item for item in items}
    assessments = [
        DocumentAssessment(
            authority_id=authority.authority_id,
            predicted_label=PredictedLabel.REAL,
            fake_probability=0.01,
            rationale="fixture miss",
        )
        for authority in scenario.authorities
    ]
    metrics = score_document_predictions(
        scenarios,
        [DocumentPrediction(item_id=scenario.item_id, assessments=assessments)],
        by_id,
    )
    assert metrics["document_operational"]["score"] == 0
    assert metrics["document_operational"]["clean_audit_rate"] == 0
    assert metrics["document_operational"]["extra_verifications_per_document"] == 0
