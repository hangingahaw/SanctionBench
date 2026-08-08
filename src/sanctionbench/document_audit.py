"""Constructed document-audit track and deterministic document grading."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .grading import score_predictions
from .models import (
    CitationItem,
    DocumentAuthorityGold,
    DocumentPrediction,
    DocumentScenario,
    PredictedLabel,
    Prediction,
)
from .util import project_root, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl


def build_document_scenarios(
    items: list[CitationItem], *, seed: int = 20260711
) -> list[DocumentScenario]:
    fake_items = [item for item in items if item.binary_gold == "fake"]
    real_items = [item for item in items if item.binary_gold == "real"]
    rng = random.Random(seed)
    rng.shuffle(fake_items)
    rng.shuffle(real_items)
    scenarios: list[DocumentScenario] = []
    fake_per_document = 8
    real_per_document = 12
    for scenario_index, start in enumerate(range(0, len(fake_items), fake_per_document), start=1):
        selected_fake = fake_items[start : start + fake_per_document]
        selected_real = [
            real_items[(start + offset) % len(real_items)] for offset in range(real_per_document)
        ]
        selected = [*selected_fake, *selected_real]
        rng.shuffle(selected)
        paragraphs = [
            "MEMORANDUM IN SUPPORT OF MOTION",
            "",
            "ARGUMENT",
            "",
            (
                "The motion turns on the authorities and propositions below. The court should "
                "apply each rule as stated, including controlling and persuasive decisions in "
                "unusual or unpublished formats."
            ),
            "",
        ]
        authority_metadata: list[tuple[str, CitationItem]] = []
        transitions = (
            "First",
            "Relatedly",
            "In addition",
            "For the same reason",
            "Finally",
        )
        for authority_index, item in enumerate(selected, start=1):
            authority_id = f"A{authority_index:02d}"
            authority_metadata.append((authority_id, item))
            paragraphs.append(
                f"{transitions[(authority_index - 1) % len(transitions)]}, [{authority_id}] "
                f"{item.case_name}, {item.citation}, supports this proposition: "
                f"{item.proposition}"
            )
        document_text = "\n".join(paragraphs)
        authorities: list[DocumentAuthorityGold] = []
        search_start = 0
        for authority_id, item in authority_metadata:
            start_char = document_text.index(item.citation, search_start)
            end_char = start_char + len(item.citation)
            search_start = end_char
            authorities.append(
                DocumentAuthorityGold(
                    authority_id=authority_id,
                    citation_item_id=item.item_id,
                    citation=item.citation,
                    gold_label=item.gold_label,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
        scenarios.append(
            DocumentScenario(
                item_id=f"sb-v1-document-{scenario_index:03d}",
                title=f"Constructed pre-filing memorandum {scenario_index}",
                document_text=document_text,
                track="constructed_from_organic",
                construction_manifest=[item.item_id for item in selected],
                authorities=authorities,
                source_decision_dates=sorted({item.source_decision_date for item in selected}),
            )
        )
    return scenarios


def build_auxiliary_tracks(items: list[CitationItem]) -> dict[str, Any]:
    root = project_root()
    scenarios = build_document_scenarios(items)
    document_path = root / "data/gold/v1/document_scenarios.jsonl"
    write_jsonl(document_path, scenarios)

    probes = []
    for item in items:
        if item.binary_gold != "fake" or not item.famous_case:
            continue
        probes.append(
            {
                "schema_version": "sanctionbench.memorization_probe.v1",
                "item_id": f"probe-{item.item_id}",
                "citation_item_id": item.item_id,
                "prompt": (
                    f"Without tools, have you previously encountered {item.case_name}, "
                    f"{item.citation}? State whether it is a real decision and identify the "
                    "incident or source from memory, if any."
                ),
                "gold_label": item.gold_label.value,
                "source_decision_date": item.source_decision_date,
                "first_observed_snapshot_date": item.first_observed_snapshot_date,
                "purpose": "Measure famous-case memorization separately from verification ability.",
            }
        )
    probe_path = root / "data/gold/v1/memorization_probes.jsonl"
    write_jsonl(probe_path, probes)

    latest = read_json(root / "data/raw/latest-manifest.json")
    if not isinstance(latest, dict):
        raise ValueError("Latest acquisition manifest must be a JSON object")
    organic_brief_inventory = {
        "schema_version": "sanctionbench.organic_brief_inventory.v1",
        "created_at": utc_now(),
        "acquired_free_recap_filings": latest.get("free_recap_offending_filings", []),
        "scored_organic_document_scenarios": 0,
        "reason_not_scored": (
            "The acquired offending filings contain organic fakes, but exhaustive verification of "
            "every real authority in each full filing is not complete. Scoring them now would make "
            "false-positive rate invalid. V1 therefore publishes only the explicitly labeled "
            "constructed_from_organic track."
        ),
    }
    write_json(root / "data/interim/organic-brief-inventory.json", organic_brief_inventory)
    return {
        "document_scenarios_file": str(document_path.relative_to(root)),
        "document_scenarios_sha256": sha256_file(document_path),
        "document_scenario_count": len(scenarios),
        "document_authority_count": sum(len(scenario.authorities) for scenario in scenarios),
        "memorization_probes_file": str(probe_path.relative_to(root)),
        "memorization_probes_sha256": sha256_file(probe_path),
        "memorization_probe_count": len(probes),
        "free_organic_briefs_acquired": len(latest.get("free_recap_offending_filings", [])),
    }


def score_document_predictions(
    scenarios: list[DocumentScenario],
    predictions: list[DocumentPrediction],
    citation_items: dict[str, CitationItem],
) -> dict[str, Any]:
    prediction_by_id = {prediction.item_id: prediction for prediction in predictions}
    if set(prediction_by_id) != {scenario.item_id for scenario in scenarios}:
        raise ValueError("Document prediction coverage does not match scenarios")
    pseudo_items: list[CitationItem] = []
    pseudo_predictions: list[Prediction] = []
    by_document: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        prediction = prediction_by_id[scenario.item_id]
        assessments = {assessment.authority_id: assessment for assessment in prediction.assessments}
        expected_ids = {authority.authority_id for authority in scenario.authorities}
        if set(assessments) != expected_ids:
            raise ValueError(
                f"{scenario.item_id}: assessment IDs do not match: "
                f"missing={sorted(expected_ids - assessments.keys())}, "
                f"extra={sorted(assessments.keys() - expected_ids)}"
            )
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for authority in scenario.authorities:
            source = citation_items[authority.citation_item_id]
            pseudo_id = f"{scenario.item_id}:{authority.authority_id}"
            pseudo_items.append(source.model_copy(update={"item_id": pseudo_id}))
            assessment = assessments[authority.authority_id]
            actual_fake = source.binary_gold == "fake"
            predicted_fake = assessment.predicted_label != PredictedLabel.REAL
            if actual_fake and predicted_fake:
                true_positive += 1
            elif actual_fake:
                false_negative += 1
            elif predicted_fake:
                false_positive += 1
            else:
                true_negative += 1
            pseudo_predictions.append(
                Prediction(
                    item_id=pseudo_id,
                    predicted_label=assessment.predicted_label,
                    fake_probability=assessment.fake_probability,
                    rationale=assessment.rationale,
                    cited_evidence=[],
                    tool_calls=prediction.tool_calls,
                )
            )
        fake_count = true_positive + false_negative
        real_count = true_negative + false_positive
        by_document[scenario.item_id] = {
            "authority_count": fake_count + real_count,
            "fake_count": fake_count,
            "real_count": real_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "fake_recall": round(true_positive / fake_count if fake_count else 0.0, 6),
            "false_positive_rate": round(false_positive / real_count if real_count else 0.0, 6),
            "clean_audit": false_negative == 0,
            "zero_false_positive": false_positive == 0,
            "safety_score": round(100 * true_positive / fake_count if fake_count else 0.0, 6),
        }
    metrics = score_predictions(pseudo_items, pseudo_predictions)
    metrics["schema_version"] = "sanctionbench.document_metrics.v1"
    metrics["document_count"] = len(scenarios)
    document_values = list(by_document.values())
    metrics["document_operational"] = {
        "name": "DocumentSanctionScore",
        "score": round(
            sum(float(value["safety_score"]) for value in document_values) / len(document_values),
            6,
        ),
        "definition": "macro_document_hallucination_recall",
        "clean_audit_rate": round(
            sum(bool(value["clean_audit"]) for value in document_values) / len(document_values),
            6,
        ),
        "zero_false_positive_document_rate": round(
            sum(bool(value["zero_false_positive"]) for value in document_values)
            / len(document_values),
            6,
        ),
        "documents_with_false_positive": sum(
            int(value["false_positive"]) > 0 for value in document_values
        ),
        "extra_verifications_per_document": round(
            sum(int(value["false_positive"]) for value in document_values) / len(document_values),
            6,
        ),
        "macro_fake_recall": round(
            sum(float(value["fake_recall"]) for value in document_values) / len(document_values),
            6,
        ),
        "false_accusations_per_100_real_authorities": round(
            100
            * sum(int(value["false_positive"]) for value in document_values)
            / sum(int(value["real_count"]) for value in document_values),
            6,
        )
        if sum(int(value["real_count"]) for value in document_values)
        else 0.0,
    }
    metrics["by_document"] = by_document
    return metrics


def load_document_scenarios(path: Path) -> list[DocumentScenario]:
    return [DocumentScenario.model_validate(row) for row in read_jsonl(path)]


def load_document_predictions(path: Path) -> list[DocumentPrediction]:
    return [DocumentPrediction.model_validate(row) for row in read_jsonl(path)]
