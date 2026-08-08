from __future__ import annotations

from conftest import make_item

from sanctionbench.grading import score_predictions
from sanctionbench.models import GoldLabel, PredictedLabel, Prediction


def _prediction(item_id: str, label: PredictedLabel, probability: float) -> Prediction:
    return Prediction(
        item_id=item_id,
        predicted_label=label,
        fake_probability=probability,
        rationale="fixture",
    )


def test_perfect_predictions_score_100() -> None:
    items = [
        make_item("fake", GoldLabel.NONEXISTENT_CASE, "pair"),
        make_item("real", GoldLabel.REAL, "pair"),
    ]
    predictions = [
        _prediction("fake", PredictedLabel.NONEXISTENT_CASE, 0.99),
        _prediction("real", PredictedLabel.REAL, 0.01),
    ]
    metrics = score_predictions(items, predictions)
    assert metrics["headline"]["score"] == 100
    assert metrics["binary"]["false_positive_rate"] == 0


def test_false_positive_is_workload_not_sanction_harm() -> None:
    items = [
        make_item("fake", GoldLabel.NONEXISTENT_CASE, "pair"),
        make_item("real", GoldLabel.REAL, "pair"),
    ]
    predictions = [
        _prediction("fake", PredictedLabel.NONEXISTENT_CASE, 0.99),
        _prediction("real", PredictedLabel.UNCERTAIN_NEEDS_REVIEW, 0.99),
    ]
    metrics = score_predictions(items, predictions)
    assert metrics["headline"]["score"] == 100
    assert metrics["binary"]["false_positive"] == 1
    assert metrics["review_workload"]["extra_verification_count"] == 1


def test_missing_a_hallucination_reduces_sanction_score() -> None:
    items = [
        make_item("fake-1", GoldLabel.NONEXISTENT_CASE, "pair-1"),
        make_item("fake-2", GoldLabel.FABRICATED_QUOTE, "pair-2"),
    ]
    predictions = [
        _prediction("fake-1", PredictedLabel.NONEXISTENT_CASE, 0.99),
        _prediction("fake-2", PredictedLabel.REAL, 0.01),
    ]
    metrics = score_predictions(items, predictions)
    assert metrics["headline"] == {
        "name": "SanctionScore",
        "score": 50.0,
        "definition": "hallucination_recall",
        "false_negative_count": 1,
        "normalization_fake_count": 2,
    }


def test_duplicate_distractors_and_source_clustering_are_reported() -> None:
    items = []
    predictions = []
    for index in range(4):
        source = "Large Source" if index < 3 else "Small Source"
        fake = make_item(
            f"fake-{index}",
            GoldLabel.NONEXISTENT_CASE,
            f"pair-{index}",
            source_case_name=source,
        )
        real = make_item(
            f"real-{index}",
            GoldLabel.REAL,
            f"pair-{index}",
            citation="123 F.3d 456" if index < 3 else "789 F.3d 101",
            source_case_name=source,
        )
        if index < 3:
            real = real.model_copy(update={"case_name": "Obscure v. Authority"})
        else:
            real = real.model_copy(update={"case_name": "Different v. Authority"})
        items.extend([fake, real])
        fake_label = PredictedLabel.NONEXISTENT_CASE if index < 3 else PredictedLabel.REAL
        real_label = PredictedLabel.REAL if index < 3 else PredictedLabel.NONEXISTENT_CASE
        predictions.extend(
            [
                _prediction(fake.item_id, fake_label, 0.99 if index < 3 else 0.01),
                _prediction(real.item_id, real_label, 0.01 if index < 3 else 0.99),
            ]
        )

    metrics = score_predictions(items, predictions)
    robustness = metrics["distractor_robustness"]
    assert robustness["real_item_count"] == 4
    assert robustness["unique_real_authority_count"] == 2
    assert robustness["duplicate_real_item_count"] == 2
    assert robustness["maximum_authority_multiplicity"] == 3
    assert robustness["item_false_positive_rate"] == 0.25
    assert robustness["unique_authority_macro_false_positive_rate"] == 0.5
    assert metrics["by_source"]["Large Source"]["exact_accuracy"] == 1.0
    assert metrics["by_source"]["Small Source"]["exact_accuracy"] == 0.0
    assert metrics["macro_by_source"]["exact_accuracy"] == 0.5
