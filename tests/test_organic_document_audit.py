from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sanctionbench.models import (
    Condition,
    GoldLabel,
    OrganicAuthorityGold,
    OrganicDocumentFinding,
    OrganicDocumentGold,
    OrganicDocumentNormalization,
    OrganicDocumentPrediction,
    OrganicDocumentSourceDecision,
    PredictedLabel,
)
from sanctionbench.organic_document_audit import (
    regrade_organic_run,
    render_document_pages,
    score_organic_document_predictions,
    validate_organic_document_gold,
)
from sanctionbench.providers.base import build_organic_document_user_prompt
from sanctionbench.runner import _select_organic_documents, run_organic_manifest
from sanctionbench.submissions import build_leaderboard, package_submission
from sanctionbench.util import canonical_json, sha256_bytes, sha256_file, write_json, write_jsonl


def _source() -> OrganicDocumentSourceDecision:
    return OrganicDocumentSourceDecision(
        decision_id="decision-1",
        case_name="Fixture Order",
        decision_date="2026-01-02",
        order_url="https://example.test/order.pdf",
        order_sha256="a" * 64,
        source_format="pdf",
    )


def _authority(
    occurrence_id: str,
    citation: str,
    label: GoldLabel,
    page: int,
    excerpt: str,
) -> OrganicAuthorityGold:
    source_fields = (
        {
            "source_decision_id": "decision-1",
            "source_order_page": 1,
            "source_order_excerpt": f"The court identified {citation} as defective.",
        }
        if label != GoldLabel.REAL
        else {}
    )
    return OrganicAuthorityGold(
        occurrence_id=occurrence_id,
        citation_text=citation,
        case_name=citation.split(",", maxsplit=1)[0],
        proposition="Fixture proposition",
        gold_label=label,
        document_page=page,
        document_excerpt=excerpt,
        **source_fields,
    )


def _document(
    item_id: str,
    *,
    kind: str,
    pages: list[str],
    authorities: list[OrganicAuthorityGold],
    cleared_public: bool = False,
) -> OrganicDocumentGold:
    text = render_document_pages(pages)
    return OrganicDocumentGold(
        item_id=item_id,
        title="Neutral filing",
        document_sha256=("b" if kind == "offending" else "c") * 64,
        document_markdown=text,
        document_markdown_sha256=sha256_bytes(text.encode()),
        page_count=len(pages),
        normalization=OrganicDocumentNormalization(
            parser_version="2.0.0",
            ocr_used=False,
            parser_output_sha256="e" * 64,
        ),
        document_kind=kind,
        authority_inventory_complete=True,
        authorities=authorities,
        source_decisions=(
            [_source()]
            if any(authority.gold_label != GoldLabel.REAL for authority in authorities)
            else []
        ),
        review_receipt_sha256="d" * 64,
        reviewer_count=2,
        reviewer_types=["human", "human"],
        adjudicator_type="human",
        human_reviewed=True,
        curation_method="independent_human_double_review",
        release_tier="human_adjudicated",
        redistribution_status=("cleared_public" if cleared_public else "private_evaluation_only"),
    )


def _finding(
    citation: str,
    page: int,
    label: PredictedLabel,
    quoted_text: str,
) -> OrganicDocumentFinding:
    return OrganicDocumentFinding(
        citation_text=citation,
        case_name=None,
        page_number=page,
        quoted_text=quoted_text,
        predicted_label=label,
        fake_probability=0.9,
        rationale="Fixture finding",
    )


def test_open_ended_grader_scores_misses_false_flags_duplicates_and_clean_controls() -> None:
    fake_one_text = "Fake v. Fiction, 123 F.9th 1, supports the first proposition."
    real_text = "Real v. Valid, 10 U.S. 20, supports the ordinary rule."
    fake_two_text = "Quote v. Wrong, 11 F.4th 30, contains a fabricated quotation."
    offending = _document(
        "offending",
        kind="offending",
        pages=[f"{fake_one_text} {real_text}", fake_two_text],
        authorities=[
            _authority(
                "F1", "Fake v. Fiction, 123 F.9th 1", GoldLabel.NONEXISTENT_CASE, 1, fake_one_text
            ),
            _authority("R1", "Real v. Valid, 10 U.S. 20", GoldLabel.REAL, 1, real_text),
            _authority(
                "F2", "Quote v. Wrong, 11 F.4th 30", GoldLabel.FABRICATED_QUOTE, 2, fake_two_text
            ),
        ],
        cleared_public=True,
    )
    clean_text = "Control v. Sound, 99 U.S. 1, is a legitimate authority."
    clean = _document(
        "clean",
        kind="clean_control",
        pages=[clean_text],
        authorities=[
            _authority("R2", "Control v. Sound, 99 U.S. 1", GoldLabel.REAL, 1, clean_text)
        ],
        cleared_public=True,
    )
    predictions = [
        OrganicDocumentPrediction(
            item_id="offending",
            findings=[
                _finding(
                    "Fake v. Fiction, 123 F.9th 1",
                    1,
                    PredictedLabel.MISATTRIBUTED_HOLDING,
                    fake_one_text,
                ),
                _finding(
                    "Fake v. Fiction, 123 F.9th 1",
                    1,
                    PredictedLabel.NONEXISTENT_CASE,
                    fake_one_text,
                ),
                _finding(
                    "Real v. Valid, 10 U.S. 20",
                    1,
                    PredictedLabel.UNCERTAIN_NEEDS_REVIEW,
                    real_text,
                ),
                _finding(
                    "Imaginary v. Uncatalogued, 1 F.99 1",
                    1,
                    PredictedLabel.NONEXISTENT_CASE,
                    "Imaginary authority",
                ),
            ],
        ),
        OrganicDocumentPrediction(item_id="clean", findings=[]),
    ]

    metrics = score_organic_document_predictions([offending, clean], predictions)

    assert metrics["matching_contract_version"] == "sanctionbench.organic_matching.v2"
    assert metrics["headline"] == {
        "name": "OrganicDocumentSanctionScore",
        "score": 50.0,
        "definition": "macro_offending_document_hallucination_recall",
        "clean_audit_rate": 0.0,
    }
    assert metrics["authority_detection"]["true_positive"] == 1
    assert metrics["authority_detection"]["false_negative"] == 1
    assert metrics["authority_detection"]["diagnosis_accuracy_on_caught"] == 0
    assert metrics["review_workload"]["extra_verification_count"] == 2
    assert metrics["clean_controls"] == {
        "document_count": 1,
        "pass_rate": 1.0,
        "false_alarm_rate": 0.0,
    }
    assert metrics["by_document"]["offending"]["duplicate_finding_count"] == 1


def test_open_ended_grader_matches_fuller_citation_by_verbatim_tokens_and_page() -> None:
    fake_text = "Curtis v. Trevino, 631 F.3d 880 (9th Cir. 2011), is cited here."
    offending = _document(
        "contained-alias",
        kind="offending",
        pages=[fake_text],
        authorities=[_authority("F1", "631 F.3d 880", GoldLabel.NONEXISTENT_CASE, 1, fake_text)],
        cleared_public=True,
    )
    clean_text = "Control v. Sound, 99 U.S. 1, is legitimate."
    clean = _document(
        "contained-alias-clean",
        kind="clean_control",
        pages=[clean_text],
        authorities=[
            _authority("R1", "Control v. Sound, 99 U.S. 1", GoldLabel.REAL, 1, clean_text)
        ],
        cleared_public=True,
    )
    predictions = [
        OrganicDocumentPrediction(
            item_id=offending.item_id,
            findings=[
                _finding(
                    "Curtis v. Trevino, 631 F.3d 880 (9th Cir. 2011)",
                    1,
                    PredictedLabel.NONEXISTENT_CASE,
                    fake_text,
                )
            ],
        ),
        OrganicDocumentPrediction(item_id=clean.item_id, findings=[]),
    ]

    metrics = score_organic_document_predictions([offending, clean], predictions)

    assert metrics["headline"]["score"] == 100.0
    assert metrics["authority_detection"]["true_positive"] == 1
    assert metrics["review_workload"]["extra_verification_count"] == 0
    assert (
        metrics["by_document"][offending.item_id]["matched_occurrences"]["F1"]["match_status"]
        == "contained_alias_and_page"
    )


def test_regrade_organic_run_preserves_history_and_binds_receipts(tmp_path: Path) -> None:
    fake_text = "Curtis v. Trevino, 631 F.3d 880 (9th Cir. 2011), is cited here."
    offending = _document(
        "offending-regrade",
        kind="offending",
        pages=[fake_text],
        authorities=[_authority("F1", "631 F.3d 880", GoldLabel.NONEXISTENT_CASE, 1, fake_text)],
        cleared_public=True,
    )
    clean_text = "Control v. Sound, 99 U.S. 1, is legitimate."
    clean = _document(
        "clean-regrade",
        kind="clean_control",
        pages=[clean_text],
        authorities=[
            _authority("R1", "Control v. Sound, 99 U.S. 1", GoldLabel.REAL, 1, clean_text)
        ],
        cleared_public=True,
    )
    predictions = [
        OrganicDocumentPrediction(
            item_id=offending.item_id,
            findings=[
                _finding(
                    "Curtis v. Trevino, 631 F.3d 880 (9th Cir. 2011)",
                    1,
                    PredictedLabel.NONEXISTENT_CASE,
                    fake_text,
                )
            ],
        ),
        OrganicDocumentPrediction(item_id=clean.item_id, findings=[]),
    ]
    dataset_path = tmp_path / "data/private/organic.jsonl"
    prediction_path = tmp_path / "results/run/predictions.jsonl"
    metrics_path = tmp_path / "results/run/metrics.json"
    identity_path = tmp_path / "results/run/identity.json"
    attempts_path = tmp_path / "results/run/request_attempts.json"
    run_path = tmp_path / "results/run/run.json"
    write_jsonl(dataset_path, [offending, clean])
    write_jsonl(prediction_path, predictions)
    write_json(metrics_path, {"headline": {"score": 0.0}})
    run_identity = {
        "schema_version": "fixture.identity.v1",
        "task_type": "organic_document_audit",
        "selected_ids": [offending.item_id, clean.item_id],
    }
    run_identity_sha256 = sha256_bytes(canonical_json(run_identity).encode("utf-8"))
    write_json(
        identity_path,
        {"identity": run_identity, "run_identity_sha256": run_identity_sha256},
    )
    write_json(
        attempts_path,
        {
            "schema_version": "sanctionbench.provider_request_attempts.v1",
            "provider_request_count": 2,
            "run_identity_sha256": run_identity_sha256,
            "selected_ids": [offending.item_id, clean.item_id],
            "attempts_started_by_item": {offending.item_id: 1, clean.item_id: 1},
        },
    )
    run: dict[str, Any] = {
        "run_id": "fixture-organic-run",
        "dataset_path": str(dataset_path.relative_to(tmp_path)),
        "dataset_sha256": sha256_file(dataset_path),
        "predictions_path": str(prediction_path.relative_to(tmp_path)),
        "metrics_path": str(metrics_path.relative_to(tmp_path)),
        "metadata": {
            "task_type": "organic_document_audit",
            "run_identity": run_identity,
            "run_identity_sha256": run_identity_sha256,
            "provider_request_count": 2,
            "successful_response_count": 2,
            "provider_request_attempts_path": str(attempts_path.relative_to(tmp_path)),
        },
    }
    finalized = {
        "schema_version": "sanctionbench.finalized_run_identity.v2",
        "benchmark_release_status": "development_public_gold",
        "run_identity_sha256": run_identity_sha256,
        "provider_request_count": 2,
        "successful_response_count": 2,
        "provider_request_attempts_sha256": sha256_file(attempts_path),
        "courtlistener_request_attempts_sha256": None,
        "tool_evidence_sha256": None,
        "predictions_sha256": sha256_file(prediction_path),
        "metrics_sha256": sha256_file(metrics_path),
        "run_record_core_sha256": sha256_bytes(canonical_json(run).encode("utf-8")),
    }
    finalized_sha256 = sha256_bytes(canonical_json(finalized).encode("utf-8"))
    run["metadata"]["finalized_run_identity"] = finalized
    run["metadata"]["finalized_run_identity_sha256"] = finalized_sha256
    write_json(run_path, run)

    result = regrade_organic_run(run_path, root=tmp_path)

    assert result["metrics"]["headline"]["score"] == 100.0
    assert result["receipt"]["provider_requests_added"] == 0
    assert result["receipt"]["schema_version"] == "sanctionbench.organic_regrade_receipt.v2"
    assert result["receipt"]["source_finalized_run_identity_sha256"] == finalized_sha256
    assert result["receipt"]["matching_contract_version"] == ("sanctionbench.organic_matching.v2")
    assert result["receipt"]["artifacts"]["historical_metrics"]["sha256"] == sha256_file(
        metrics_path
    )
    assert result["receipt"]["artifacts"]["request_attempts"]["sha256"] == sha256_file(
        attempts_path
    )
    assert (
        metrics_path.read_text(encoding="utf-8") == '{\n  "headline": {\n    "score": 0.0\n  }\n}\n'
    )
    with pytest.raises(FileExistsError):
        regrade_organic_run(run_path, root=tmp_path)

    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8").replace("Curtis v. Trevino", "Altered v. Data"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finalized identity"):
        regrade_organic_run(
            run_path,
            root=tmp_path,
            output_path=run_path.parent / "metrics.tampered.json",
            receipt_path=run_path.parent / "receipt.tampered.json",
        )


def test_open_ended_grader_does_not_guess_ambiguous_contained_alias() -> None:
    first_text = "Alpha v. One, 123 F.3d 10, is discussed."
    second_text = "Beta v. Two, 123 F.3d 10, is also discussed."
    offending = _document(
        "ambiguous-contained-alias",
        kind="offending",
        pages=[f"{first_text} {second_text}"],
        authorities=[
            _authority(
                "F1", "Alpha v. One, 123 F.3d 10", GoldLabel.NONEXISTENT_CASE, 1, first_text
            ),
            _authority(
                "F2", "Beta v. Two, 123 F.3d 10", GoldLabel.NONEXISTENT_CASE, 1, second_text
            ),
        ],
        cleared_public=True,
    )
    clean_text = "Control v. Sound, 99 U.S. 1, is legitimate."
    clean = _document(
        "ambiguous-contained-alias-clean",
        kind="clean_control",
        pages=[clean_text],
        authorities=[
            _authority("R1", "Control v. Sound, 99 U.S. 1", GoldLabel.REAL, 1, clean_text)
        ],
        cleared_public=True,
    )
    predictions = [
        OrganicDocumentPrediction(
            item_id=offending.item_id,
            findings=[
                _finding(
                    "123 F.3d 10",
                    1,
                    PredictedLabel.NONEXISTENT_CASE,
                    "The filing cites 123 F.3d 10.",
                )
            ],
        ),
        OrganicDocumentPrediction(item_id=clean.item_id, findings=[]),
    ]

    metrics = score_organic_document_predictions([offending, clean], predictions)

    assert metrics["authority_detection"]["true_positive"] == 0
    assert metrics["review_workload"]["adjudication_required_count"] == 1
    assert (
        metrics["by_document"][offending.item_id]["adjudication_required_findings"][0][
            "match_status"
        ]
        == "ambiguous_contained_alias"
    )


def test_organic_prompt_is_neutral_and_contains_no_gold_inventory() -> None:
    fake_text = "Fake v. Fiction, 123 F.9th 1, is cited here."
    document = _document(
        "neutral",
        kind="offending",
        pages=[fake_text],
        authorities=[
            _authority(
                "F1", "Fake v. Fiction, 123 F.9th 1", GoldLabel.NONEXISTENT_CASE, 1, fake_text
            )
        ],
    )

    prompt = build_organic_document_user_prompt(document.model_input(), Condition.CLOSED_BOOK)

    payload = json.loads(prompt)
    assert payload["task"] == "organic_document_audit"
    assert "authorities" not in payload
    assert "gold" not in prompt.casefold()
    assert "offending" not in prompt.casefold()
    assert "empty findings" in payload["instructions"]


def test_clean_control_model_rejects_fake_gold() -> None:
    fake_text = "Fake v. Fiction, 123 F.9th 1, is cited here."
    with pytest.raises(ValueError, match="clean-control"):
        _document(
            "bad-clean",
            kind="clean_control",
            pages=[fake_text],
            authorities=[
                _authority(
                    "F1",
                    "Fake v. Fiction, 123 F.9th 1",
                    GoldLabel.NONEXISTENT_CASE,
                    1,
                    fake_text,
                )
            ],
        )


def test_gold_validation_rejects_trivial_document_excerpt() -> None:
    citation = "Control v. Sound, 99 U.S. 1"
    document = _document(
        "trivial-excerpt",
        kind="clean_control",
        pages=[f"The filing cites {citation} in its authorities section."],
        authorities=[_authority("R1", citation, GoldLabel.REAL, 1, ".")],
    )

    with pytest.raises(ValueError, match="excerpt not found"):
        validate_organic_document_gold([document])


def test_gold_validation_requires_an_offending_document() -> None:
    citation = "Control v. Sound, 99 U.S. 1"
    text = f"The filing cites {citation} in its authorities section."
    control = _document(
        "control-only",
        kind="clean_control",
        pages=[text],
        authorities=[_authority("R1", citation, GoldLabel.REAL, 1, text)],
    )

    with pytest.raises(ValueError, match="offending document"):
        validate_organic_document_gold([control])


def test_sampled_organic_run_preserves_positive_and_clean_control() -> None:
    fake_text = "Fake v. Fiction, 123 F.9th 1, is cited here."
    clean_text = "Control v. Sound, 99 U.S. 1, is legitimate."
    offending = _document(
        "sample-offending",
        kind="offending",
        pages=[fake_text],
        authorities=[
            _authority(
                "F1", "Fake v. Fiction, 123 F.9th 1", GoldLabel.NONEXISTENT_CASE, 1, fake_text
            )
        ],
    )
    clean_one = _document(
        "sample-clean-1",
        kind="clean_control",
        pages=[clean_text],
        authorities=[
            _authority("R1", "Control v. Sound, 99 U.S. 1", GoldLabel.REAL, 1, clean_text)
        ],
    )
    clean_two = clean_one.model_copy(update={"item_id": "sample-clean-2"})

    selected = _select_organic_documents([offending, clean_one, clean_two], maximum=2, seed=17)

    assert {document.document_kind for document in selected} == {"offending", "clean_control"}
    with pytest.raises(ValueError, match="at least two"):
        _select_organic_documents([offending, clean_one], maximum=1, seed=17)


def test_organic_runner_makes_one_checkpointed_call_per_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_text = "Fake v. Fiction, 123 F.9th 1, is cited here."
    offending = _document(
        "run-offending",
        kind="offending",
        pages=[fake_text],
        authorities=[
            _authority(
                "F1", "Fake v. Fiction, 123 F.9th 1", GoldLabel.NONEXISTENT_CASE, 1, fake_text
            )
        ],
        cleared_public=True,
    )
    clean_text = "Control v. Sound, 99 U.S. 1, is legitimate."
    clean = _document(
        "run-clean",
        kind="clean_control",
        pages=[clean_text],
        authorities=[
            _authority("R1", "Control v. Sound, 99 U.S. 1", GoldLabel.REAL, 1, clean_text)
        ],
        cleared_public=True,
    )
    dataset = tmp_path / "organic.jsonl"
    write_jsonl(dataset, [offending, clean])
    config = tmp_path / "organic.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: organic.jsonl",
                "output_dir: results",
                "providers:",
                "  - provider: mock",
                "    model: deterministic-organic-v1",
                "conditions: [closed_book]",
                "repetitions: 2",
                "seed: 7",
                "resume: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sanctionbench.runner.project_root", lambda: tmp_path)
    monkeypatch.setattr("sanctionbench.submissions.project_root", lambda: tmp_path)

    summaries = run_organic_manifest(config)

    assert len(summaries) == 2
    assert sum(int(summary["model_call_count"]) for summary in summaries) == 4
    assert sum(int(summary["provider_request_count"]) for summary in summaries) == 4
    index = json.loads((tmp_path / "results/index.json").read_text(encoding="utf-8"))
    assert index["planned_model_call_count"] == 4
    assert index["provider_request_count"] == 4
    assert index["one_isolated_successful_response_per_document"] is True
    assert all(summary["mock"] for summary in summaries)

    submission_path, bundle = package_submission(
        result_index_path=tmp_path / "results/index.json",
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization=None,
        model_revision="fixture-v1",
        model_endpoint_type="mock",
        benchmark_version="v1-organic",
    )
    assert submission_path.exists()
    assert bundle.run_count == 2
    assert bundle.model_query_count == 4
    assert [run.repetition for run in bundle.runs] == [1, 2]
    assert all(run.task_type == "organic_document_audit" for run in bundle.runs)

    leaderboard = build_leaderboard(
        submissions_dir=tmp_path / "submissions",
        json_output=tmp_path / "leaderboard.json",
        markdown_output=tmp_path / "LEADERBOARD.md",
        html_output=tmp_path / "leaderboard.html",
    )
    entry = leaderboard["entries"][0]
    assert entry["organic_closed_sanction_score"] == 0.0
    assert entry["organic_closed_clean_audit_rate"] == 0.0
    assert entry["organic_clean_control_false_alarm_rate"] == 0.0
    assert entry["organic_extra_verifications_per_document"] == 0.0


def test_private_organic_result_bundle_is_nonpublishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_text = "Fake v. Fiction, 123 F.9th 1, is cited here."
    clean_text = "Control v. Sound, 99 U.S. 1, is legitimate."
    dataset = tmp_path / "organic-private.jsonl"
    write_jsonl(
        dataset,
        [
            _document(
                "private-offending",
                kind="offending",
                pages=[fake_text],
                authorities=[
                    _authority(
                        "F1",
                        "Fake v. Fiction, 123 F.9th 1",
                        GoldLabel.NONEXISTENT_CASE,
                        1,
                        fake_text,
                    )
                ],
            ),
            _document(
                "private-clean",
                kind="clean_control",
                pages=[clean_text],
                authorities=[
                    _authority("R1", "Control v. Sound, 99 U.S. 1", GoldLabel.REAL, 1, clean_text)
                ],
            ),
        ],
    )
    config = tmp_path / "organic-private.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: organic-private.jsonl",
                "output_dir: private-results",
                "providers:",
                "  - provider: mock",
                "    model: deterministic-organic-v1",
                "conditions: [closed_book]",
                "repetitions: 1",
                "seed: 7",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sanctionbench.runner.project_root", lambda: tmp_path)
    monkeypatch.setattr("sanctionbench.submissions.project_root", lambda: tmp_path)

    run_organic_manifest(config)
    submission_path, bundle = package_submission(
        result_index_path=tmp_path / "private-results/index.json",
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization=None,
        model_revision="fixture-private-v1",
        model_endpoint_type="mock",
    )

    assert bundle.benchmark_release_status == "provisional_private_evaluation"
    assert bundle.publication.publishable is False
    with pytest.raises(ValueError, match="cannot enter the public leaderboard"):
        build_leaderboard(
            submissions_dir=submission_path.parent,
            json_output=tmp_path / "leaderboard.json",
            markdown_output=tmp_path / "LEADERBOARD.md",
            html_output=tmp_path / "index.html",
        )
