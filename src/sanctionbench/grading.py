"""Deterministic citation-level grading and false-positive-sensitive metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import CitationItem, GoldLabel, PredictedLabel, Prediction
from .util import read_jsonl, utc_now, write_json


def _flagged(label: PredictedLabel) -> bool:
    return label != PredictedLabel.REAL


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _round(value: float) -> float:
    return round(value, 6)


def _authority_key(item: CitationItem) -> tuple[str, str]:
    def normalize(value: str) -> str:
        return "".join(character.lower() for character in value if character.isalnum())

    return normalize(item.case_name), normalize(item.citation)


def _calibration(
    probabilities: list[float], outcomes: list[int], bins: int = 10
) -> tuple[float, float, list[dict[str, Any]]]:
    if not probabilities:
        return 0.0, 0.0, []
    brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    ) / len(probabilities)
    records: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probabilities[position] for position in members) / len(members)
        accuracy = sum(outcomes[position] for position in members) / len(members)
        ece += len(members) / len(probabilities) * abs(confidence - accuracy)
        records.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_fake_probability": _round(confidence),
                "empirical_fake_rate": _round(accuracy),
            }
        )
    return _round(brier), _round(ece), records


def _recall_at_fpr(
    probabilities: list[float], outcomes: list[int], maximum_fpr: float
) -> dict[str, float]:
    best_recall = 0.0
    best_threshold = 1.0
    best_fpr = 0.0
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    tp = 0
    fp = 0

    def consider(threshold: float) -> None:
        nonlocal best_recall, best_threshold, best_fpr
        recall = _safe_div(tp, positives)
        fpr = _safe_div(fp, negatives)
        if fpr <= maximum_fpr and recall >= best_recall:
            best_recall = recall
            best_threshold = threshold
            best_fpr = fpr

    consider(1.000001)
    ordered = sorted(zip(probabilities, outcomes, strict=True), reverse=True)
    position = 0
    while position < len(ordered):
        threshold = ordered[position][0]
        while position < len(ordered) and ordered[position][0] == threshold:
            if ordered[position][1] == 1:
                tp += 1
            else:
                fp += 1
            position += 1
        consider(threshold)
    consider(-0.000001)
    return {
        "maximum_fpr": maximum_fpr,
        "recall": _round(best_recall),
        "observed_fpr": _round(best_fpr),
        "threshold": _round(best_threshold),
    }


def _class_metrics(
    gold: list[GoldLabel], predictions: list[PredictedLabel]
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for label in GoldLabel:
        tp = sum(
            gold_value == label and prediction.value == label.value
            for gold_value, prediction in zip(gold, predictions, strict=True)
        )
        fp = sum(
            gold_value != label and prediction.value == label.value
            for gold_value, prediction in zip(gold, predictions, strict=True)
        )
        fn = sum(
            gold_value == label and prediction.value != label.value
            for gold_value, prediction in zip(gold, predictions, strict=True)
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[label.value] = {
            "support": sum(value == label for value in gold),
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }
    return metrics


def score_predictions(items: list[CitationItem], predictions: list[Prediction]) -> dict[str, Any]:
    by_id = {prediction.item_id: prediction for prediction in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction item IDs")
    expected = {item.item_id for item in items}
    missing = expected - by_id.keys()
    extra = by_id.keys() - expected
    if missing or extra:
        raise ValueError(
            f"Prediction coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    ordered_predictions = [by_id[item.item_id] for item in items]
    gold_fake = [item.binary_gold == "fake" for item in items]
    predicted_flag = [_flagged(prediction.predicted_label) for prediction in ordered_predictions]
    tp = sum(
        actual and predicted for actual, predicted in zip(gold_fake, predicted_flag, strict=True)
    )
    fp = sum(
        not actual and predicted
        for actual, predicted in zip(gold_fake, predicted_flag, strict=True)
    )
    tn = sum(
        not actual and not predicted
        for actual, predicted in zip(gold_fake, predicted_flag, strict=True)
    )
    fn = sum(
        actual and not predicted
        for actual, predicted in zip(gold_fake, predicted_flag, strict=True)
    )
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    beta = 0.5
    f_beta = _safe_div((1 + beta**2) * precision * recall, beta**2 * precision + recall)
    fake_count = sum(gold_fake)
    sanction_score = 100 * recall
    exact = sum(
        item.gold_label.value == prediction.predicted_label.value
        for item, prediction in zip(items, ordered_predictions, strict=True)
    )
    probabilities = [prediction.fake_probability for prediction in ordered_predictions]
    outcomes = [int(value) for value in gold_fake]
    brier, ece, calibration_bins = _calibration(probabilities, outcomes)

    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unique_real_flags: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for item, prediction in zip(items, ordered_predictions, strict=True):
        result = (
            "correct" if item.gold_label.value == prediction.predicted_label.value else "incorrect"
        )
        source_counts[item.source_case_name][result] += 1
        actual_fake = item.binary_gold == "fake"
        predicted_fake = _flagged(prediction.predicted_label)
        source_counts[item.source_case_name]["support"] += 1
        source_counts[item.source_case_name]["fake_support" if actual_fake else "real_support"] += 1
        if actual_fake and predicted_fake:
            source_counts[item.source_case_name]["true_positive"] += 1
        elif actual_fake:
            source_counts[item.source_case_name]["false_negative"] += 1
        elif predicted_fake:
            source_counts[item.source_case_name]["false_positive"] += 1
        else:
            source_counts[item.source_case_name]["true_negative"] += 1
        if not actual_fake:
            unique_real_flags[_authority_key(item)].append(predicted_fake)

    by_source: dict[str, dict[str, float | int]] = {}
    for source, counts in sorted(source_counts.items()):
        source_tp = counts["true_positive"]
        source_fn = counts["false_negative"]
        source_fp = counts["false_positive"]
        source_tn = counts["true_negative"]
        by_source[source] = {
            **dict(counts),
            "exact_accuracy": _round(_safe_div(counts["correct"], counts["support"])),
            "binary_recall": _round(_safe_div(source_tp, source_tp + source_fn)),
            "binary_false_positive_rate": _round(_safe_div(source_fp, source_fp + source_tn)),
        }
    source_values = list(by_source.values())
    macro_by_source = {
        "source_count": len(source_values),
        "exact_accuracy": _round(
            _safe_div(
                sum(float(value["exact_accuracy"]) for value in source_values),
                len(source_values),
            )
        ),
        "binary_recall": _round(
            _safe_div(
                sum(float(value["binary_recall"]) for value in source_values),
                len(source_values),
            )
        ),
        "binary_false_positive_rate": _round(
            _safe_div(
                sum(float(value["binary_false_positive_rate"]) for value in source_values),
                len(source_values),
            )
        ),
    }
    unique_real_rates = [_safe_div(sum(flags), len(flags)) for flags in unique_real_flags.values()]
    real_item_count = sum(len(flags) for flags in unique_real_flags.values())
    distractor_robustness = {
        "real_item_count": real_item_count,
        "unique_real_authority_count": len(unique_real_flags),
        "duplicate_real_item_count": real_item_count - len(unique_real_flags),
        "maximum_authority_multiplicity": max(
            (len(flags) for flags in unique_real_flags.values()), default=0
        ),
        "item_false_positive_rate": _round(fpr),
        "unique_authority_macro_false_positive_rate": _round(
            _safe_div(sum(unique_real_rates), len(unique_real_rates))
        ),
    }

    return {
        "schema_version": "sanctionbench.metrics.v1",
        "scored_at": utc_now(),
        "item_count": len(items),
        "headline": {
            "name": "SanctionScore",
            "score": _round(sanction_score),
            "definition": "hallucination_recall",
            "false_negative_count": fn,
            "normalization_fake_count": fake_count,
        },
        "review_workload": {
            "extra_verification_count": fp,
            "verification_overhead_rate": _round(fpr),
            "false_accusations_per_100_real_authorities": _round(100 * fpr),
        },
        "binary": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": _round(precision),
            "recall": _round(recall),
            "specificity": _round(specificity),
            "false_positive_rate": _round(fpr),
            "f0_5": _round(f_beta),
        },
        "four_way": {
            "exact_accuracy": _round(_safe_div(exact, len(items))),
            "per_class": _class_metrics(
                [item.gold_label for item in items],
                [prediction.predicted_label for prediction in ordered_predictions],
            ),
            "uncertain_prediction_count": sum(
                prediction.predicted_label == PredictedLabel.UNCERTAIN_NEEDS_REVIEW
                for prediction in ordered_predictions
            ),
        },
        "calibration": {
            "brier_score": brier,
            "expected_calibration_error_10_bin": ece,
            "bins": calibration_bins,
        },
        "operating_points": {
            "recall_at_fpr_1_percent": _recall_at_fpr(probabilities, outcomes, 0.01),
            "recall_at_fpr_5_percent": _recall_at_fpr(probabilities, outcomes, 0.05),
        },
        "macro_by_source": macro_by_source,
        "distractor_robustness": distractor_robustness,
        "by_source": by_source,
    }


def score_files(
    gold_path: Path, predictions_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    items = [CitationItem.model_validate(row) for row in read_jsonl(gold_path)]
    predictions = [Prediction.model_validate(row) for row in read_jsonl(predictions_path)]
    metrics = score_predictions(items, predictions)
    if output_path:
        write_json(output_path, metrics)
    return metrics
