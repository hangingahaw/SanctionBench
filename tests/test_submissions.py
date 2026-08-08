from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanctionbench.submissions import (
    SubmissionBundle,
    _filename_slug,
    _markdown_cell,
    _reconciled_tool_evidence_sha256,
    _resolve_from_root,
    build_leaderboard,
    package_submission,
    validate_submission,
)


def test_submission_filename_slug_is_cross_platform_safe() -> None:
    assert _filename_slug("../Provider\\Model:Alias") == "provider-model-alias"
    assert _filename_slug("***") == "unnamed"


def test_relative_artifact_path_cannot_escape_project_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes project root"):
        _resolve_from_root(tmp_path / "root", "../outside.json")


def test_markdown_leaderboard_cell_escapes_untrusted_model_metadata() -> None:
    rendered = _markdown_cell("<script>|`[*_payload_]")

    assert "<script>" not in rendered
    assert "|" not in rendered
    assert "`" not in rendered
    assert "[" not in rendered
    assert "*" not in rendered
    assert "_" not in rendered
    assert "&lt;script&gt;" in rendered


def test_smoke_result_packages_and_builds_static_leaderboard(tmp_path: Path) -> None:
    submission_path, bundle = package_submission(
        result_index_path=Path("results/smoke/index.json"),
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization="SanctionBench",
        model_revision="deterministic-rule-based-v1",
        model_endpoint_type="mock",
    )
    assert bundle.submission_tier == "development_mock"
    assert bundle.official is False
    assert bundle.mock is True
    assert bundle.run_count == 4
    assert bundle.model_query_count == 50
    assert bundle.publication.contains_raw_predictions is False
    assert all(run.prompt_and_output_schema_sha256 for run in bundle.runs)
    assert all(
        (run.tool_evidence_sha256 is not None) == (run.condition == "tool_assisted")
        for run in bundle.runs
    )
    assert all(
        (run.courtlistener_request_attempts_sha256 is not None)
        == (run.condition == "tool_assisted")
        for run in bundle.runs
    )
    assert validate_submission(submission_path)["bundle_sha256"]

    output = tmp_path / "leaderboard"
    leaderboard = build_leaderboard(
        submissions_dir=tmp_path / "submissions",
        json_output=output / "leaderboard.json",
        markdown_output=output / "LEADERBOARD.md",
        html_output=output / "index.html",
    )
    assert leaderboard["benchmark_version"] == "1.0.0"
    assert leaderboard["generated_from_submission_count"] == 1
    assert leaderboard["official_submission_count"] == 0
    entry = leaderboard["entries"][0]
    assert entry["tier"] == "development_mock"
    assert entry["model_query_count"] == 50
    assert entry["document_tool_clean_audit_rate"] == 1.0
    assert entry["document_tool_extra_verifications_per_document"] == 1.0
    markdown_output = (output / "LEADERBOARD.md").read_text()
    assert "not official" in markdown_output
    assert "N/A" in markdown_output
    assert "—" not in markdown_output
    assert "https://github.com/hangingahaw/SanctionBench" in markdown_output
    html_output = (output / "index.html").read_text()
    assert "SanctionBench Leaderboard" in html_output
    assert "Benchmark version <strong>1.0.0</strong>" in html_output
    assert "Metric definitions" in html_output
    assert "Built from 1 validated aggregate submission." in html_output
    assert 'class="table-scroll"' in html_output
    assert 'aria-label="Leaderboard metrics"' in html_output
    assert "Scroll horizontally to view all metrics." in html_output
    assert "N/A" in html_output
    assert "—" not in html_output
    assert 'href="https://github.com/hangingahaw/SanctionBench"' in html_output
    assert (
        'href="https://github.com/hangingahaw/SanctionBench/blob/main/docs/SCORING.md"'
        in html_output
    )
    assert json.loads((output / "leaderboard.json").read_text()) == leaderboard


def test_leaderboard_rejects_mixed_benchmark_versions(
    tmp_path: Path,
) -> None:
    first_path, _ = package_submission(
        result_index_path=Path("results/smoke/index.json"),
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization="SanctionBench",
        model_revision="deterministic-rule-based-v1",
        model_endpoint_type="mock",
        benchmark_version="1.0.0",
    )
    submission_path, _ = package_submission(
        result_index_path=Path("results/smoke/index.json"),
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization="SanctionBench",
        model_revision="deterministic-rule-based-v1",
        model_endpoint_type="mock",
        benchmark_version="2.0.0",
    )
    assert first_path.is_file()
    assert submission_path.is_file()

    with pytest.raises(ValueError, match="cannot mix benchmark versions: 1.0.0, 2.0.0"):
        build_leaderboard(
            submissions_dir=tmp_path / "submissions",
            json_output=tmp_path / "leaderboard.json",
            markdown_output=tmp_path / "LEADERBOARD.md",
            html_output=tmp_path / "index.html",
        )


def test_submission_validation_rejects_oversized_bundle_before_json_parse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (10 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="JSON file exceeds"):
        validate_submission(path)


def test_submission_rejects_cross_model_result_index(tmp_path: Path) -> None:
    source = json.loads(Path("results/smoke/index.json").read_text())
    source["runs"][0]["run_file"] = source["runs"][1]["run_file"]
    tampered = tmp_path / "index.json"
    tampered.write_text(json.dumps(source))
    try:
        package_submission(
            result_index_path=tampered,
            output_dir=tmp_path / "submissions",
            submitter_name="test",
            organization=None,
            model_revision="test",
            model_endpoint_type="mock",
        )
    except ValueError as error:
        assert "identity" in str(error) or "duplicate" in str(error)
    else:
        raise AssertionError("tampered result index was accepted")


def test_submission_recomputes_run_identity_digest(tmp_path: Path) -> None:
    source = json.loads(Path("results/smoke/index.json").read_text())
    original_run = Path(str(source["runs"][0]["run_file"]))
    tampered_run = json.loads(original_run.read_text())
    tampered_run["metadata"]["run_identity"]["seed"] += 1
    tampered_run_path = tmp_path / "run.json"
    tampered_run_path.write_text(json.dumps(tampered_run))
    source["runs"][0]["run_file"] = str(tampered_run_path)
    result_index = tmp_path / "index.json"
    result_index.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="identity digest does not reconcile"):
        package_submission(
            result_index_path=result_index,
            output_dir=tmp_path / "submissions",
            submitter_name="SanctionBench",
            organization=None,
            model_revision="tampered",
            model_endpoint_type="mock",
        )


def test_submission_rejects_finalized_run_identity_tampering(tmp_path: Path) -> None:
    source = json.loads(Path("results/smoke/index.json").read_text())
    source["runs"][0]["finalized_run_identity_sha256"] = "0" * 64
    result_index = tmp_path / "index.json"
    result_index.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="finalized run identity does not reconcile"):
        package_submission(
            result_index_path=result_index,
            output_dir=tmp_path / "submissions",
            submitter_name="SanctionBench",
            organization=None,
            model_revision="tampered",
            model_endpoint_type="mock",
        )


def test_tool_evidence_receipt_is_reopened_and_rehashed(tmp_path: Path) -> None:
    source = json.loads(Path("results/smoke/index.json").read_text())
    summary = next(run for run in source["runs"] if run["condition"] == "tool_assisted")
    original_run = json.loads(Path(str(summary["run_file"])).read_text())
    original_evidence = Path(str(original_run["metadata"]["tool_evidence_path"]))
    evidence_path = tmp_path / "tool_evidence.json"
    evidence_path.write_bytes(original_evidence.read_bytes())
    run_path = tmp_path / "run.json"
    original_run["metadata"]["tool_evidence_path"] = str(evidence_path)
    run_path.write_text(json.dumps(original_run))

    observed = _reconciled_tool_evidence_sha256(
        Path.cwd(),
        run_path=run_path,
        run=original_run,
        identity=original_run["metadata"]["run_identity"],
        identity_sha256=original_run["metadata"]["run_identity_sha256"],
        condition="tool_assisted",
    )
    assert observed == original_run["metadata"]["tool_evidence_sha256"]

    evidence_path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        _reconciled_tool_evidence_sha256(
            Path.cwd(),
            run_path=run_path,
            run=original_run,
            identity=original_run["metadata"]["run_identity"],
            identity_sha256=original_run["metadata"]["run_identity_sha256"],
            condition="tool_assisted",
        )


def test_submission_rejects_release_status_tampering(tmp_path: Path) -> None:
    source = json.loads(Path("results/smoke/index.json").read_text())
    source["benchmark_release_status"] = "provisional_private_evaluation"
    result_index = tmp_path / "index.json"
    result_index.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="differs from dataset release metadata"):
        package_submission(
            result_index_path=result_index,
            output_dir=tmp_path / "submissions",
            submitter_name="SanctionBench",
            organization=None,
            model_revision="private-fixture",
            model_endpoint_type="mock",
        )


def test_submission_rejects_missing_release_metadata(tmp_path: Path) -> None:
    source = json.loads(Path("results/smoke/index.json").read_text())
    source.pop("benchmark_release_status", None)
    result_index = tmp_path / "index.json"
    result_index.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="missing benchmark_release_status"):
        package_submission(
            result_index_path=result_index,
            output_dir=tmp_path / "submissions",
            submitter_name="SanctionBench",
            organization=None,
            model_revision="missing-metadata-fixture",
            model_endpoint_type="mock",
        )


def test_public_bundle_schema_rejects_forged_official_holdout_status(tmp_path: Path) -> None:
    _, bundle = package_submission(
        result_index_path=Path("results/smoke/index.json"),
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization=None,
        model_revision="fixture",
        model_endpoint_type="mock",
    )
    payload = bundle.model_dump(mode="json")
    payload["benchmark_release_status"] = "official_private_holdout"

    with pytest.raises(ValueError, match="benchmark_release_status"):
        SubmissionBundle.model_validate(payload)


def test_submission_schema_rejects_nested_raw_payload_keys(tmp_path: Path) -> None:
    _, bundle = package_submission(
        result_index_path=Path("results/smoke/index.json"),
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization=None,
        model_revision="fixture",
        model_endpoint_type="mock",
    )
    payload = bundle.model_dump(mode="json")
    payload["publication"]["raw_predictions"] = [{"private": "payload"}]

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SubmissionBundle.model_validate(payload)


def test_submission_schema_rejects_unsafe_id_and_mock_tier_forgery(tmp_path: Path) -> None:
    _, bundle = package_submission(
        result_index_path=Path("results/smoke/index.json"),
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization=None,
        model_revision="fixture",
        model_endpoint_type="mock",
    )
    payload = bundle.model_dump(mode="json")
    payload["submission_id"] = "../unsafe"
    with pytest.raises(ValueError, match="submission_id"):
        SubmissionBundle.model_validate(payload)

    payload = bundle.model_dump(mode="json")
    payload["submission_tier"] = "self_reported"
    with pytest.raises(ValueError, match="tier does not reconcile"):
        SubmissionBundle.model_validate(payload)


def test_submission_id_binds_finalized_receipts(tmp_path: Path) -> None:
    _, bundle = package_submission(
        result_index_path=Path("results/smoke/index.json"),
        output_dir=tmp_path / "submissions",
        submitter_name="SanctionBench",
        organization=None,
        model_revision="fixture",
        model_endpoint_type="mock",
    )
    payload = bundle.model_dump(mode="json")
    payload["runs"][0]["metrics_file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="submission_id does not bind"):
        SubmissionBundle.model_validate(payload)
