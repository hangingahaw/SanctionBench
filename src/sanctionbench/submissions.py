"""Deterministic submission bundles and static leaderboard generation."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .util import (
    DEFAULT_MAX_MANIFEST_BYTES,
    canonical_json,
    project_root,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
)

FINALIZED_RUN_IDENTITY_VERSION = "sanctionbench.finalized_run_identity.v2"
MAX_SUBMISSION_BUNDLE_BYTES = 10 * 1024 * 1024
MAX_SUBMISSION_RUNS = 256
MAX_SUBMISSION_FILES = 1_000
MAX_REFERENCED_ARTIFACT_BYTES = 64 * 1024 * 1024
REPOSITORY_URL = "https://github.com/hangingahaw/SanctionBench"
SCORING_URL = f"{REPOSITORY_URL}/blob/main/docs/SCORING.md"


def _filename_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return (slug or "unnamed")[:80]


def _submission_identity_material(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Select every publication-critical bundle field except the ID itself."""

    keys = (
        "created_at",
        "benchmark_version",
        "benchmark_release_status",
        "submission_tier",
        "official",
        "submitter_name",
        "organization",
        "model_provider",
        "model_name",
        "model_revision",
        "model_endpoint_type",
        "source_result_index_sha256",
        "dataset_sha256",
        "dataset_release_tiers",
        "dataset_redistribution_statuses",
        "runner_commit",
        "runner_dirty",
        "model_query_count",
        "run_count",
        "conditions",
        "mock",
        "runs",
        "attestation",
        "publication",
    )
    return {key: payload[key] for key in keys}


def _submission_id(payload: Mapping[str, Any]) -> str:
    digest = sha256_bytes(canonical_json(_submission_identity_material(payload)).encode())
    return (
        f"{_filename_slug(str(payload['model_provider']))}-"
        f"{_filename_slug(str(payload['model_name']))}-{digest[:16]}"
    )


class SubmissionRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=200)
    task_type: Literal["citation_verification", "document_audit", "organic_document_audit"]
    condition: Literal["closed_book", "tool_assisted"]
    repetition: int | None = Field(default=None, ge=1)
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=300)
    model_query_count: int = Field(ge=1)
    scored_authority_count: int = Field(ge=1)
    sanction_score: float = Field(ge=0, le=100)
    false_positive_rate: float | None = Field(default=None, ge=0, le=1)
    fake_recall: float = Field(ge=0, le=1)
    document_sanction_score: float | None = Field(default=None, ge=0, le=100)
    clean_audit_rate: float | None = Field(default=None, ge=0, le=1)
    extra_verifications_per_document: float | None = Field(default=None, ge=0)
    zero_false_positive_document_rate: float | None = Field(default=None, ge=0, le=1)
    clean_control_false_alarm_rate: float | None = Field(default=None, ge=0, le=1)
    diagnosis_accuracy_on_caught: float | None = Field(default=None, ge=0, le=1)
    page_accuracy_on_caught: float | None = Field(default=None, ge=0, le=1)
    prompt_and_output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finalized_run_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_request_attempts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    courtlistener_request_attempts_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    predictions_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    run_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_tool_evidence_receipt(self) -> SubmissionRun:
        if self.condition == "tool_assisted" and self.tool_evidence_sha256 is None:
            raise ValueError("tool-assisted runs require a tool-evidence receipt hash")
        if self.condition == "tool_assisted" and self.courtlistener_request_attempts_sha256 is None:
            raise ValueError("tool-assisted runs require a CourtListener request receipt hash")
        if self.condition == "closed_book" and self.tool_evidence_sha256 is not None:
            raise ValueError("closed-book runs cannot claim tool evidence")
        if (
            self.condition == "closed_book"
            and self.courtlistener_request_attempts_sha256 is not None
        ):
            raise ValueError("closed-book runs cannot claim CourtListener requests")
        return self


class SubmissionAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["self_reported"] = "self_reported"
    organizer_verified: Literal[False] = False
    warning: str = Field(min_length=1, max_length=500)


class SubmissionPublication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contains_raw_predictions: Literal[False] = False
    contains_private_prompts: Literal[False] = False
    contains_reasoning_traces: Literal[False] = False
    result_scope: Literal["aggregate metrics and reproducibility hashes only"] = (
        "aggregate metrics and reproducibility hashes only"
    )
    publishable: bool


class SubmissionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sanctionbench.submission.v1"] = "sanctionbench.submission.v1"
    submission_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,159}$")
    created_at: str = Field(min_length=1, max_length=100)
    benchmark_version: str = Field(min_length=1, max_length=100)
    benchmark_release_status: Literal[
        "development_public_gold",
        "provisional_private_evaluation",
    ]
    submission_tier: Literal["development_mock", "self_reported"]
    official: Literal[False] = False
    submitter_name: str = Field(min_length=1, max_length=200)
    organization: str | None = Field(default=None, max_length=200)
    model_provider: str = Field(min_length=1, max_length=200)
    model_name: str = Field(min_length=1, max_length=300)
    model_revision: str = Field(min_length=1, max_length=300)
    model_endpoint_type: Literal["mock", "hosted_api", "open_weights", "other"]
    source_result_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_release_tiers: list[
        Literal[
            "public_development_gold",
            "human_adjudicated",
            "constructed_control",
            "provisional_model_assisted",
            "unclassified_external",
        ]
    ] = Field(min_length=1)
    dataset_redistribution_statuses: list[Literal["cleared_public", "private_evaluation_only"]] = (
        Field(min_length=1)
    )
    runner_commit: str = Field(min_length=1, max_length=200)
    runner_dirty: bool | None
    model_query_count: int = Field(ge=1)
    run_count: int = Field(ge=1)
    conditions: list[Literal["closed_book", "tool_assisted"]] = Field(min_length=1)
    mock: bool
    runs: list[SubmissionRun] = Field(min_length=1, max_length=MAX_SUBMISSION_RUNS)
    attestation: SubmissionAttestation
    publication: SubmissionPublication

    @model_validator(mode="after")
    def validate_aggregate_fields(self) -> SubmissionBundle:
        if self.run_count != len(self.runs):
            raise ValueError("run_count does not match runs")
        if self.model_query_count != sum(run.model_query_count for run in self.runs):
            raise ValueError("model_query_count does not reconcile")
        if self.conditions != sorted({run.condition for run in self.runs}):
            raise ValueError("conditions do not reconcile")
        if len({(run.task_type, run.condition, run.repetition) for run in self.runs}) != len(
            self.runs
        ):
            raise ValueError("duplicate task/condition/repetition run")
        if any(
            run.task_type == "organic_document_audit"
            and (run.condition != "closed_book" or run.repetition is None)
            for run in self.runs
        ):
            raise ValueError("organic document runs require closed_book and a repetition number")
        if any(
            run.provider != self.model_provider or run.model != self.model_name for run in self.runs
        ):
            raise ValueError("run provider/model differs from submission model")
        if self.mock != all(run.provider == "mock" for run in self.runs):
            raise ValueError("mock flag does not reconcile")
        if self.mock != (self.submission_tier == "development_mock"):
            raise ValueError("submission tier does not reconcile with mock status")
        if self.mock != (self.model_endpoint_type == "mock"):
            raise ValueError("endpoint type does not reconcile with mock status")
        if self.dataset_release_tiers != sorted(set(self.dataset_release_tiers)):
            raise ValueError("dataset release tiers must be sorted and duplicate-free")
        if self.dataset_redistribution_statuses != sorted(
            set(self.dataset_redistribution_statuses)
        ):
            raise ValueError("redistribution statuses must be sorted and duplicate-free")
        private_dataset = (
            "private_evaluation_only" in self.dataset_redistribution_statuses
            or "provisional_model_assisted" in self.dataset_release_tiers
            or "unclassified_external" in self.dataset_release_tiers
        )
        expected_status = (
            "provisional_private_evaluation" if private_dataset else "development_public_gold"
        )
        if self.benchmark_release_status != expected_status:
            raise ValueError("benchmark status conflicts with dataset release metadata")
        if not private_dataset and set(self.dataset_redistribution_statuses) != {"cleared_public"}:
            raise ValueError("public datasets must be cleared for redistribution")
        if (
            self.benchmark_release_status == "provisional_private_evaluation"
            and self.publication.publishable is not False
        ):
            raise ValueError("private provisional submissions must be marked non-publishable")
        if self.submission_id != _submission_id(self.model_dump(mode="json")):
            raise ValueError("submission_id does not bind the finalized bundle")
        return self


def _resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Relative artifact path escapes project root: {value}") from error
    return resolved


def _reconciled_release_metadata(
    root: Path, index: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    """Fail closed unless index metadata exactly matches the referenced dataset."""

    raw_status = index.get("benchmark_release_status")
    raw_tiers = index.get("dataset_release_tiers")
    raw_redistribution = index.get("dataset_redistribution_statuses")
    if not isinstance(raw_status, str):
        raise ValueError("Result index is missing benchmark_release_status")
    if (
        not isinstance(raw_tiers, list)
        or not raw_tiers
        or not all(isinstance(value, str) for value in raw_tiers)
    ):
        raise ValueError("Result index is missing dataset_release_tiers")
    if (
        not isinstance(raw_redistribution, list)
        or not raw_redistribution
        or not all(isinstance(value, str) for value in raw_redistribution)
    ):
        raise ValueError("Result index is missing dataset_redistribution_statuses")
    tiers = sorted(set(raw_tiers))
    redistribution = sorted(set(raw_redistribution))
    if raw_tiers != tiers or raw_redistribution != redistribution:
        raise ValueError("Result index release metadata must be sorted and duplicate-free")

    schema_version = index.get("schema_version")
    if schema_version == "sanctionbench.organic_result_index.v1":
        dataset_value = index.get("dataset")
        if not isinstance(dataset_value, str):
            raise ValueError("Organic result index is missing its dataset path")
        dataset_path = _resolve_from_root(root, dataset_value)
        if not dataset_path.is_file() or sha256_file(dataset_path) != index.get("dataset_sha256"):
            raise ValueError("Organic result index dataset is missing or its hash differs")
        rows = read_jsonl(dataset_path)
        observed_tiers = sorted({str(row.get("release_tier")) for row in rows})
        observed_redistribution = sorted({str(row.get("redistribution_status")) for row in rows})
        if tiers != observed_tiers or redistribution != observed_redistribution:
            raise ValueError("Organic result index release metadata differs from its dataset")
        derived_status = (
            "provisional_private_evaluation"
            if "private_evaluation_only" in observed_redistribution
            or "provisional_model_assisted" in observed_tiers
            else "development_public_gold"
        )
    elif schema_version == "sanctionbench.result_index.v1":
        dataset_value = index.get("dataset")
        if not isinstance(dataset_value, str):
            raise ValueError("v1 result index is missing its citation dataset path")
        citation_path = _resolve_from_root(root, dataset_value)
        citation_digest = str(index.get("dataset_sha256") or "")
        if not citation_path.is_file() or sha256_file(citation_path) != citation_digest:
            raise ValueError("v1 citation dataset is missing or its hash differs")
        document_value = index.get("document_dataset")
        document_digest = index.get("document_dataset_sha256")
        has_document_runs = any(
            summary.get("task_type") == "document_audit" for summary in index.get("runs") or []
        )
        if has_document_runs and document_value is None:
            raise ValueError("v1 document-audit runs require document dataset metadata")
        if document_value is not None:
            if not isinstance(document_value, str) or not isinstance(document_digest, str):
                raise ValueError("v1 document dataset metadata is incomplete")
            document_path = _resolve_from_root(root, document_value)
            if not document_path.is_file() or sha256_file(document_path) != document_digest:
                raise ValueError("v1 document dataset is missing or its hash differs")
        elif document_digest is not None:
            raise ValueError("v1 document dataset hash has no dataset path")
        manifest_path = root / "data/gold/v1/manifest.json"
        manifest = (
            read_json(manifest_path, max_bytes=DEFAULT_MAX_MANIFEST_BYTES)
            if manifest_path.is_file()
            else {}
        )
        if not isinstance(manifest, dict):
            raise ValueError("Public gold manifest must be a JSON object")
        public_dataset = (
            manifest.get("release_status") == "development_public_gold"
            and manifest.get("citation_items_sha256") == citation_digest
            and (
                document_digest is None
                or manifest.get("document_scenarios_sha256") == document_digest
            )
        )
        if public_dataset:
            derived_status = "development_public_gold"
            observed_tiers = ["public_development_gold"]
            observed_redistribution = ["cleared_public"]
        else:
            derived_status = "provisional_private_evaluation"
            observed_tiers = ["unclassified_external"]
            observed_redistribution = ["private_evaluation_only"]
        if tiers != observed_tiers or redistribution != observed_redistribution:
            raise ValueError("v1 result index release metadata differs from manifest-bound data")
    else:
        raise ValueError(f"Unknown result index schema: {schema_version}")

    if raw_status != derived_status:
        raise ValueError("benchmark_release_status differs from dataset release metadata")
    return raw_status, tiers, redistribution


def _provider_request_count(
    root: Path,
    *,
    run_path: Path,
    run: dict[str, Any],
    summary: dict[str, Any],
    identity: dict[str, Any],
    identity_sha256: str,
) -> int:
    metadata = run.get("metadata") or {}
    attempts_value = metadata.get("provider_request_attempts_path")
    if not isinstance(attempts_value, str):
        raise ValueError(f"{run_path}: provider-request ledger path is missing")
    attempts_path = _resolve_from_root(root, attempts_value)
    if not attempts_path.is_file():
        raise ValueError(f"{run_path}: provider-request ledger is missing")
    ledger = read_json(attempts_path, max_bytes=MAX_SUBMISSION_BUNDLE_BYTES)
    if not isinstance(ledger, dict):
        raise ValueError(f"{run_path}: provider-request ledger must be a JSON object")
    counts = ledger.get("attempts_started_by_item")
    if (
        ledger.get("schema_version") != "sanctionbench.provider_request_attempts.v1"
        or ledger.get("run_identity_sha256") != identity_sha256
        or ledger.get("selected_ids") != identity.get("selected_ids")
        or not isinstance(counts, dict)
        or any(not isinstance(value, int) or value < 0 for value in counts.values())
    ):
        raise ValueError(f"{run_path}: provider-request ledger does not reconcile")
    count = sum(counts.values())
    if (
        ledger.get("provider_request_count") != count
        or metadata.get("provider_request_count") != count
        or summary.get("provider_request_count") != count
    ):
        raise ValueError(f"{run_path}: provider-request count does not reconcile")
    return count


def _reconciled_tool_evidence_sha256(
    root: Path,
    *,
    run_path: Path,
    run: dict[str, Any],
    identity: dict[str, Any],
    identity_sha256: str,
    condition: str,
) -> str | None:
    metadata = run.get("metadata") or {}
    claimed_path = metadata.get("tool_evidence_path")
    claimed_sha256 = metadata.get("tool_evidence_sha256")
    claimed_count = metadata.get("tool_evidence_item_count", 0)
    if condition == "closed_book":
        if (
            claimed_path is not None
            or claimed_sha256 is not None
            or claimed_count not in {None, 0}
            or identity.get("tool_spec_version") is not None
        ):
            raise ValueError(f"{run_path}: closed-book run claims tool evidence")
        return None

    if (
        not isinstance(claimed_path, str)
        or not isinstance(claimed_sha256, str)
        or not isinstance(claimed_count, int)
        or claimed_count < 1
        or not isinstance(identity.get("tool_spec_version"), str)
    ):
        raise ValueError(f"{run_path}: tool-assisted run lacks an evidence receipt")
    evidence_path = _resolve_from_root(root, claimed_path)
    if evidence_path.parent != run_path.parent or evidence_path.name != "tool_evidence.json":
        raise ValueError(f"{run_path}: tool-evidence receipt is outside the run directory")
    if not evidence_path.is_file() or sha256_file(evidence_path) != claimed_sha256:
        raise ValueError(f"{run_path}: tool-evidence receipt hash mismatch")
    evidence = read_json(evidence_path, max_bytes=MAX_REFERENCED_ARTIFACT_BYTES)
    if not isinstance(evidence, dict):
        raise ValueError(f"{run_path}: tool-evidence receipt must be a JSON object")
    selected_ids = identity.get("selected_ids")
    evidence_by_item = evidence.get("evidence_by_item")
    if (
        evidence.get("schema_version") != "sanctionbench.tool_evidence.v1"
        or evidence.get("run_identity_sha256") != identity_sha256
        or evidence.get("tool_spec_version") != identity.get("tool_spec_version")
        or evidence.get("selected_ids") != selected_ids
        or not isinstance(selected_ids, list)
        or not isinstance(evidence_by_item, dict)
        or set(evidence_by_item) != set(selected_ids)
        or not all(isinstance(value, dict) for value in evidence_by_item.values())
        or evidence.get("evidence_item_count") != len(selected_ids)
        or claimed_count != len(selected_ids)
    ):
        raise ValueError(f"{run_path}: tool-evidence receipt does not reconcile")
    return claimed_sha256


def _reconciled_courtlistener_attempts_sha256(
    root: Path,
    *,
    run_path: Path,
    run: dict[str, Any],
    summary: dict[str, Any],
    identity_sha256: str,
    condition: str,
) -> str | None:
    metadata = run.get("metadata") or {}
    claimed_path = metadata.get("courtlistener_request_attempts_path")
    claimed_sha256 = metadata.get("courtlistener_request_attempts_sha256")
    claimed_count = metadata.get("courtlistener_request_count", 0)
    if condition == "closed_book":
        if claimed_path is not None or claimed_sha256 is not None or claimed_count not in {None, 0}:
            raise ValueError(f"{run_path}: closed-book run claims CourtListener requests")
        return None
    if (
        not isinstance(claimed_path, str)
        or not isinstance(claimed_sha256, str)
        or not isinstance(claimed_count, int)
        or isinstance(claimed_count, bool)
        or claimed_count < 0
    ):
        raise ValueError(f"{run_path}: tool-assisted run lacks a CourtListener request receipt")
    attempts_path = _resolve_from_root(root, claimed_path)
    if (
        attempts_path.parent != run_path.parent
        or attempts_path.name != "courtlistener_request_attempts.json"
        or not attempts_path.is_file()
        or sha256_file(attempts_path) != claimed_sha256
    ):
        raise ValueError(f"{run_path}: CourtListener request receipt hash mismatch")
    ledger = read_json(attempts_path, max_bytes=MAX_SUBMISSION_BUNDLE_BYTES)
    if not isinstance(ledger, dict):
        raise ValueError(f"{run_path}: invalid CourtListener request receipt")
    selected_ids = ledger.get("selected_ids")
    counts = ledger.get("attempts_started_by_lookup")
    if (
        ledger.get("schema_version") != "sanctionbench.courtlistener_request_attempts.v1"
        or ledger.get("run_identity_sha256") != identity_sha256
        or not isinstance(selected_ids, list)
        or len(selected_ids) > 10_000
        or len(selected_ids) != len(set(selected_ids))
        or not all(isinstance(value, str) and len(value) <= 300 for value in selected_ids)
        or not isinstance(counts, dict)
        or not set(counts).issubset(set(selected_ids))
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 12
            for value in counts.values()
        )
    ):
        raise ValueError(f"{run_path}: CourtListener request receipt does not reconcile")
    observed_count = sum(counts.values())
    if (
        ledger.get("courtlistener_request_count") != observed_count
        or claimed_count != observed_count
        or summary.get("courtlistener_request_count") != observed_count
    ):
        raise ValueError(f"{run_path}: CourtListener request count does not reconcile")
    return claimed_sha256


def _reconciled_finalized_run_identity(
    root: Path,
    *,
    run_path: Path,
    run: dict[str, Any],
    summary: dict[str, Any],
    benchmark_release_status: str,
    identity_sha256: str,
    model_query_count: int,
    courtlistener_request_attempts_sha256: str | None,
    tool_evidence_sha256: str | None,
) -> tuple[str, str, str]:
    """Recompute the completed run digest from its immutable artifact receipts."""

    metadata = run.get("metadata") or {}
    finalized = metadata.get("finalized_run_identity")
    finalized_sha256 = metadata.get("finalized_run_identity_sha256")
    if not isinstance(finalized, dict) or not isinstance(finalized_sha256, str):
        raise ValueError(f"{run_path}: finalized run identity is missing")

    attempts_path = _resolve_from_root(root, str(metadata.get("provider_request_attempts_path")))
    predictions_path = _resolve_from_root(root, str(run.get("predictions_path")))
    metrics_path = _resolve_from_root(root, str(run.get("metrics_path")))
    for path, expected_name in (
        (attempts_path, "request_attempts.json"),
        (predictions_path, "predictions.jsonl"),
        (metrics_path, "metrics.json"),
    ):
        if path.parent != run_path.parent or path.name != expected_name or not path.is_file():
            raise ValueError(f"{run_path}: finalized artifact is outside the run directory")

    attempts_sha256 = sha256_file(attempts_path)
    predictions_sha256 = sha256_file(predictions_path)
    metrics_sha256 = sha256_file(metrics_path)
    core = json.loads(json.dumps(run))
    core_metadata = core.get("metadata") or {}
    core_metadata.pop("finalized_run_identity", None)
    core_metadata.pop("finalized_run_identity_sha256", None)
    expected = {
        "schema_version": FINALIZED_RUN_IDENTITY_VERSION,
        "benchmark_release_status": benchmark_release_status,
        "run_identity_sha256": identity_sha256,
        "provider_request_count": model_query_count,
        "successful_response_count": int(metadata.get("successful_response_count", -1)),
        "provider_request_attempts_sha256": attempts_sha256,
        "courtlistener_request_attempts_sha256": (courtlistener_request_attempts_sha256),
        "tool_evidence_sha256": tool_evidence_sha256,
        "predictions_sha256": predictions_sha256,
        "metrics_sha256": metrics_sha256,
        "run_record_core_sha256": sha256_bytes(canonical_json(core).encode()),
    }
    expected_digest = sha256_bytes(canonical_json(expected).encode())
    if (
        finalized != expected
        or finalized_sha256 != expected_digest
        or summary.get("finalized_run_identity_sha256") != expected_digest
    ):
        raise ValueError(f"{run_path}: finalized run identity does not reconcile")
    return expected_digest, attempts_sha256, predictions_sha256


def package_submission(
    *,
    result_index_path: Path,
    output_dir: Path,
    submitter_name: str,
    organization: str | None,
    model_revision: str,
    model_endpoint_type: Literal["mock", "hosted_api", "open_weights", "other"],
    benchmark_version: str | None = None,
) -> tuple[Path, SubmissionBundle]:
    """Package a complete one-model result index without publishing raw predictions."""

    root = project_root()
    index = read_json(result_index_path, max_bytes=MAX_SUBMISSION_BUNDLE_BYTES)
    if not isinstance(index, dict):
        raise ValueError("Result index must be a JSON object")
    benchmark_release_status, dataset_release_tiers, dataset_redistribution_statuses = (
        _reconciled_release_metadata(root, index)
    )
    resolved_benchmark_version = benchmark_version or (
        "organic-1.0.0"
        if index.get("schema_version") == "sanctionbench.organic_result_index.v1"
        else "1.0.0"
    )
    run_summaries = index.get("runs") or []
    if not run_summaries:
        raise ValueError("Result index contains no runs")
    if not isinstance(run_summaries, list) or len(run_summaries) > MAX_SUBMISSION_RUNS:
        raise ValueError(f"Result index cannot contain more than {MAX_SUBMISSION_RUNS} runs")

    runs: list[SubmissionRun] = []
    providers: set[str] = set()
    models: set[str] = set()
    for summary in run_summaries:
        run_path = _resolve_from_root(root, str(summary["run_file"]))
        run = read_json(run_path, max_bytes=MAX_REFERENCED_ARTIFACT_BYTES)
        if not isinstance(run, dict):
            raise ValueError(f"{run_path}: run record must be a JSON object")
        metrics_path = _resolve_from_root(root, str(run["metrics_path"]))
        metrics = read_json(metrics_path, max_bytes=MAX_REFERENCED_ARTIFACT_BYTES)
        if not isinstance(metrics, dict):
            raise ValueError(f"{metrics_path}: metrics must be a JSON object")
        identity_sha256 = str((run.get("metadata") or {}).get("run_identity_sha256") or "")
        if identity_sha256 != str(summary["run_identity_sha256"]):
            raise ValueError(f"{run_path}: run identity differs from result index")
        identity = (run.get("metadata") or {}).get("run_identity")
        if (
            not isinstance(identity, dict)
            or sha256_bytes(canonical_json(identity).encode()) != identity_sha256
        ):
            raise ValueError(f"{run_path}: run identity digest does not reconcile")
        provider = str(run["provider"])
        model = str(run["model"])
        providers.add(provider)
        models.add(model)
        task_type = str(summary["task_type"])
        condition = str(run["condition"])
        if task_type not in {
            "citation_verification",
            "document_audit",
            "organic_document_audit",
        }:
            raise ValueError(f"Unknown task type in result index: {task_type}")
        if condition not in {"closed_book", "tool_assisted"}:
            raise ValueError(f"Unknown condition in run: {condition}")
        if identity.get("task_type") != task_type:
            raise ValueError(f"{run_path}: run identity task type differs")
        expected_dataset_sha256 = (
            index.get("document_dataset_sha256")
            if task_type == "document_audit"
            else index.get("dataset_sha256")
        )
        if (
            run.get("dataset_sha256") != expected_dataset_sha256
            or identity.get("dataset_sha256") != expected_dataset_sha256
        ):
            raise ValueError(f"{run_path}: run dataset differs from result index")
        if task_type == "document_audit" and identity.get("reference_dataset_sha256") != index.get(
            "dataset_sha256"
        ):
            raise ValueError(f"{run_path}: document run citation reference differs")
        model_query_count = _provider_request_count(
            root,
            run_path=run_path,
            run=run,
            summary=summary,
            identity=identity,
            identity_sha256=identity_sha256,
        )
        tool_evidence_sha256 = _reconciled_tool_evidence_sha256(
            root,
            run_path=run_path,
            run=run,
            identity=identity,
            identity_sha256=identity_sha256,
            condition=condition,
        )
        courtlistener_request_attempts_sha256 = _reconciled_courtlistener_attempts_sha256(
            root,
            run_path=run_path,
            run=run,
            summary=summary,
            identity_sha256=identity_sha256,
            condition=condition,
        )
        (
            finalized_run_identity_sha256,
            provider_request_attempts_sha256,
            predictions_file_sha256,
        ) = _reconciled_finalized_run_identity(
            root,
            run_path=run_path,
            run=run,
            summary=summary,
            benchmark_release_status=benchmark_release_status,
            identity_sha256=identity_sha256,
            model_query_count=model_query_count,
            courtlistener_request_attempts_sha256=(courtlistener_request_attempts_sha256),
            tool_evidence_sha256=tool_evidence_sha256,
        )
        clean_audit_rate: float | None
        extra_verifications_per_document: float | None
        zero_false_positive_document_rate: float | None
        if task_type == "organic_document_audit":
            detection = metrics["authority_detection"]
            workload = metrics["review_workload"]
            clean_controls = metrics["clean_controls"]
            scored_authority_count = int(detection["fake_authority_count"]) + int(
                workload["real_authority_count"]
            )
            false_positive_rate = None
            fake_recall = float(detection["micro_recall"])
            document_sanction_score = None
            clean_audit_rate = float(metrics["headline"]["clean_audit_rate"])
            extra_verifications_per_document = float(workload["extra_verifications_per_document"])
            zero_false_positive_document_rate = float(workload["zero_false_positive_document_rate"])
            clean_control_false_alarm_rate = (
                float(clean_controls["false_alarm_rate"])
                if clean_controls["false_alarm_rate"] is not None
                else None
            )
            diagnosis_accuracy_on_caught = (
                float(detection["diagnosis_accuracy_on_caught"])
                if detection["diagnosis_accuracy_on_caught"] is not None
                else None
            )
            page_accuracy_on_caught = (
                float(detection["page_accuracy_on_caught"])
                if detection["page_accuracy_on_caught"] is not None
                else None
            )
        else:
            scored_authority_count = int(metrics["item_count"])
            false_positive_rate = float(metrics["binary"]["false_positive_rate"])
            fake_recall = float(metrics["binary"]["recall"])
            operational = metrics.get("document_operational") or {}
            document_sanction_score = float(operational["score"]) if operational else None
            clean_audit_rate = float(operational["clean_audit_rate"]) if operational else None
            extra_verifications_per_document = (
                float(operational["extra_verifications_per_document"]) if operational else None
            )
            zero_false_positive_document_rate = (
                float(operational["zero_false_positive_document_rate"]) if operational else None
            )
            clean_control_false_alarm_rate = None
            diagnosis_accuracy_on_caught = None
            page_accuracy_on_caught = None
        runs.append(
            SubmissionRun(
                run_id=str(run["run_id"]),
                task_type=cast(
                    Literal["citation_verification", "document_audit", "organic_document_audit"],
                    task_type,
                ),
                condition=cast(Literal["closed_book", "tool_assisted"], condition),
                repetition=(
                    int(summary["repetition"]) if task_type == "organic_document_audit" else None
                ),
                provider=provider,
                model=model,
                model_query_count=model_query_count,
                scored_authority_count=scored_authority_count,
                sanction_score=float(metrics["headline"]["score"]),
                false_positive_rate=false_positive_rate,
                fake_recall=fake_recall,
                document_sanction_score=document_sanction_score,
                clean_audit_rate=clean_audit_rate,
                extra_verifications_per_document=extra_verifications_per_document,
                zero_false_positive_document_rate=zero_false_positive_document_rate,
                clean_control_false_alarm_rate=clean_control_false_alarm_rate,
                diagnosis_accuracy_on_caught=diagnosis_accuracy_on_caught,
                page_accuracy_on_caught=page_accuracy_on_caught,
                prompt_and_output_schema_sha256=str(
                    run["metadata"]["prompt_and_output_schema_sha256"]
                ),
                run_identity_sha256=identity_sha256,
                finalized_run_identity_sha256=finalized_run_identity_sha256,
                provider_request_attempts_sha256=provider_request_attempts_sha256,
                courtlistener_request_attempts_sha256=(courtlistener_request_attempts_sha256),
                predictions_file_sha256=predictions_file_sha256,
                tool_evidence_sha256=tool_evidence_sha256,
                run_file_sha256=sha256_file(run_path),
                metrics_file_sha256=sha256_file(metrics_path),
            )
        )
    if len(providers) != 1 or len(models) != 1:
        raise ValueError("A submission bundle must contain exactly one provider/model")
    runs.sort(key=lambda value: (value.task_type, value.condition, value.repetition or 0))
    provider = next(iter(providers))
    model = next(iter(models))
    mock = provider == "mock"
    if mock and model_endpoint_type != "mock":
        raise ValueError("Mock results require model_endpoint_type=mock")
    if not mock and model_endpoint_type == "mock":
        raise ValueError("Live results cannot use model_endpoint_type=mock")
    result_index_sha256 = sha256_file(result_index_path)
    git_state = index.get("git_state_at_manifest_start") or {}
    provisional_private = benchmark_release_status == "provisional_private_evaluation"
    attestation = SubmissionAttestation(
        warning=(
            "This aggregate bundle is for a provisional private evaluation and must not be "
            "published or added to the public leaderboard."
            if provisional_private
            else "This bundle is not an official private-holdout result. Official status "
            "requires an organizer-run evaluation and a separate organizer attestation."
        )
    )
    publication = SubmissionPublication(publishable=not provisional_private)
    bundle_values: dict[str, Any] = {
        "created_at": utc_now(),
        "benchmark_version": resolved_benchmark_version,
        "benchmark_release_status": benchmark_release_status,
        "submission_tier": "development_mock" if mock else "self_reported",
        "official": False,
        "submitter_name": submitter_name,
        "organization": organization,
        "model_provider": provider,
        "model_name": model,
        "model_revision": model_revision,
        "model_endpoint_type": model_endpoint_type,
        "source_result_index_sha256": result_index_sha256,
        "dataset_sha256": str(index["dataset_sha256"]),
        "dataset_release_tiers": dataset_release_tiers,
        "dataset_redistribution_statuses": dataset_redistribution_statuses,
        "runner_commit": str(git_state.get("commit") or "UNAVAILABLE"),
        "runner_dirty": git_state.get("dirty"),
        "model_query_count": sum(run.model_query_count for run in runs),
        "run_count": len(runs),
        "conditions": sorted({run.condition for run in runs}),
        "mock": mock,
        "runs": [run.model_dump(mode="json") for run in runs],
        "attestation": attestation.model_dump(mode="json"),
        "publication": publication.model_dump(mode="json"),
    }
    submission_id = _submission_id(bundle_values)
    bundle = SubmissionBundle.model_validate({"submission_id": submission_id, **bundle_values})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{submission_id}.json"
    write_json(
        output_path,
        bundle.model_dump(mode="json"),
        max_bytes=MAX_SUBMISSION_BUNDLE_BYTES,
    )
    return output_path, bundle


def validate_submission(path: Path) -> dict[str, Any]:
    payload = read_json(path, max_bytes=MAX_SUBMISSION_BUNDLE_BYTES)
    bundle = SubmissionBundle.model_validate(payload)
    return {
        "submission_id": bundle.submission_id,
        "submission_tier": bundle.submission_tier,
        "official": bundle.official,
        "model": f"{bundle.model_provider}/{bundle.model_name}",
        "model_query_count": bundle.model_query_count,
        "run_count": bundle.run_count,
        "bundle_sha256": sha256_file(path),
    }


def _score_for(runs: list[SubmissionRun], task: str, condition: str, field: str) -> float | None:
    values = [
        float(value)
        for run in runs
        if run.task_type == task and run.condition == condition
        if (value := getattr(run, field)) is not None
    ]
    return sum(values) / len(values) if values else None


def build_leaderboard(
    *,
    submissions_dir: Path,
    json_output: Path,
    markdown_output: Path,
    html_output: Path,
) -> dict[str, Any]:
    paths = sorted(submissions_dir.glob("*.json"))
    if len(paths) > MAX_SUBMISSION_FILES:
        raise ValueError(
            f"Leaderboard cannot ingest more than {MAX_SUBMISSION_FILES} submission files"
        )
    bundles = [
        SubmissionBundle.model_validate(read_json(path, max_bytes=MAX_SUBMISSION_BUNDLE_BYTES))
        for path in paths
    ]
    private_ids = [
        bundle.submission_id
        for bundle in bundles
        if bundle.benchmark_release_status == "provisional_private_evaluation"
    ]
    if private_ids:
        raise ValueError(
            "Private provisional submissions cannot enter the public leaderboard: "
            + ", ".join(private_ids)
        )
    benchmark_versions = {bundle.benchmark_version for bundle in bundles}
    if len(benchmark_versions) > 1:
        raise ValueError(
            "Leaderboard cannot mix benchmark versions: " + ", ".join(sorted(benchmark_versions))
        )
    benchmark_version = next(iter(benchmark_versions), __version__)
    ids = [bundle.submission_id for bundle in bundles]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate submission IDs")
    entries: list[dict[str, Any]] = []
    for bundle in bundles:
        entries.append(
            {
                "submission_id": bundle.submission_id,
                "tier": bundle.submission_tier,
                "official": bundle.official,
                "model_provider": bundle.model_provider,
                "model_name": bundle.model_name,
                "model_revision": bundle.model_revision,
                "citation_closed_sanction_score": _score_for(
                    bundle.runs, "citation_verification", "closed_book", "sanction_score"
                ),
                "citation_closed_false_positive_rate": _score_for(
                    bundle.runs, "citation_verification", "closed_book", "false_positive_rate"
                ),
                "citation_tool_sanction_score": _score_for(
                    bundle.runs, "citation_verification", "tool_assisted", "sanction_score"
                ),
                "citation_tool_false_positive_rate": _score_for(
                    bundle.runs, "citation_verification", "tool_assisted", "false_positive_rate"
                ),
                "document_closed_sanction_score": _score_for(
                    bundle.runs, "document_audit", "closed_book", "document_sanction_score"
                ),
                "document_tool_sanction_score": _score_for(
                    bundle.runs, "document_audit", "tool_assisted", "document_sanction_score"
                ),
                "document_tool_clean_audit_rate": _score_for(
                    bundle.runs, "document_audit", "tool_assisted", "clean_audit_rate"
                ),
                "document_tool_extra_verifications_per_document": _score_for(
                    bundle.runs,
                    "document_audit",
                    "tool_assisted",
                    "extra_verifications_per_document",
                ),
                "document_tool_zero_fp_rate": _score_for(
                    bundle.runs,
                    "document_audit",
                    "tool_assisted",
                    "zero_false_positive_document_rate",
                ),
                "organic_closed_sanction_score": _score_for(
                    bundle.runs, "organic_document_audit", "closed_book", "sanction_score"
                ),
                "organic_closed_clean_audit_rate": _score_for(
                    bundle.runs, "organic_document_audit", "closed_book", "clean_audit_rate"
                ),
                "organic_clean_control_false_alarm_rate": _score_for(
                    bundle.runs,
                    "organic_document_audit",
                    "closed_book",
                    "clean_control_false_alarm_rate",
                ),
                "organic_extra_verifications_per_document": _score_for(
                    bundle.runs,
                    "organic_document_audit",
                    "closed_book",
                    "extra_verifications_per_document",
                ),
                "organic_diagnosis_accuracy_on_caught": _score_for(
                    bundle.runs,
                    "organic_document_audit",
                    "closed_book",
                    "diagnosis_accuracy_on_caught",
                ),
                "organic_page_accuracy_on_caught": _score_for(
                    bundle.runs,
                    "organic_document_audit",
                    "closed_book",
                    "page_accuracy_on_caught",
                ),
                "model_query_count": bundle.model_query_count,
                "dataset_sha256": bundle.dataset_sha256,
                "runner_commit": bundle.runner_commit,
            }
        )
    entries.sort(
        key=lambda value: (
            not bool(value["official"]),
            value["organic_closed_sanction_score"] is None,
            -(float(value["organic_closed_sanction_score"] or 0)),
            -(float(value["organic_closed_clean_audit_rate"] or 0)),
            float(value["organic_clean_control_false_alarm_rate"] or 0),
            float(value["organic_extra_verifications_per_document"] or 0),
            -(float(value["document_tool_clean_audit_rate"] or 0)),
            -(float(value["citation_tool_sanction_score"] or 0)),
            -(float(value["document_tool_sanction_score"] or 0)),
            float(value["document_tool_extra_verifications_per_document"] or 0),
            float(value["citation_tool_false_positive_rate"] or 0),
            str(value["model_provider"]),
            str(value["model_name"]),
        )
    )
    leaderboard = {
        "schema_version": "sanctionbench.leaderboard.v1",
        "benchmark_version": benchmark_version,
        "generated_from_submission_count": len(entries),
        "official_submission_count": sum(bool(entry["official"]) for entry in entries),
        "warning": (
            "Development mock and self-reported results are not official. Official results require "
            "an organizer-run private-holdout evaluation."
        ),
        "entries": entries,
    }
    write_json(json_output, leaderboard)

    header = (
        "| Tier | Model | Revision | Organic closed | Organic clean audits | Control alarms | "
        "Organic extra reviews/doc | Citation closed | Closed FPR | Citation tool | Tool FPR | "
        "Document closed | Document tool | Tool clean audits | Tool extra reviews/doc | Queries |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |"
    )
    rows = [
        "# SanctionBench Leaderboard",
        "",
        f"Benchmark version `{benchmark_version}`. See the [scoring contract]({SCORING_URL}) for metric definitions and ordering, or return to the [SanctionBench repository]({REPOSITORY_URL}).",
        "",
        f"> {leaderboard['warning']}",
        "",
        header,
    ]
    for entry in entries:
        rows.append(
            "| {tier} | {provider}/{model} | `{revision}` | {organic} | {organic_clean} | "
            "{control_alarms} | {organic_extra} | {cc} | {cfpr} | {ct} | {tfpr} | {dc} | "
            "{dt} | {clean} | {extra} | {queries} |".format(
                tier=_markdown_cell(entry["tier"]),
                provider=_markdown_cell(entry["model_provider"]),
                model=_markdown_cell(entry["model_name"]),
                revision=_markdown_cell(entry["model_revision"]),
                organic=_format_metric(entry["organic_closed_sanction_score"]),
                organic_clean=_format_rate(entry["organic_closed_clean_audit_rate"]),
                control_alarms=_format_rate(entry["organic_clean_control_false_alarm_rate"]),
                organic_extra=_format_metric(entry["organic_extra_verifications_per_document"]),
                cc=_format_metric(entry["citation_closed_sanction_score"]),
                cfpr=_format_rate(entry["citation_closed_false_positive_rate"]),
                ct=_format_metric(entry["citation_tool_sanction_score"]),
                tfpr=_format_rate(entry["citation_tool_false_positive_rate"]),
                dc=_format_metric(entry["document_closed_sanction_score"]),
                dt=_format_metric(entry["document_tool_sanction_score"]),
                clean=_format_rate(entry["document_tool_clean_audit_rate"]),
                extra=_format_metric(entry["document_tool_extra_verifications_per_document"]),
                queries=entry["model_query_count"],
            )
        )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(
        _leaderboard_html(
            entries,
            str(leaderboard["warning"]),
            benchmark_version=benchmark_version,
            submission_count=len(entries),
        ),
        encoding="utf-8",
    )
    return leaderboard


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def _markdown_cell(value: Any) -> str:
    """Render untrusted metadata as inert text inside a Markdown table cell."""

    escaped = html.escape(str(value), quote=True)
    for character, entity in (
        ("|", "&#124;"),
        ("`", "&#96;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("*", "&#42;"),
        ("_", "&#95;"),
    ):
        escaped = escaped.replace(character, entity)
    return " ".join(escaped.splitlines())


def _leaderboard_html(
    entries: list[dict[str, Any]],
    warning: str,
    *,
    benchmark_version: str,
    submission_count: int,
) -> str:
    body_rows = []
    for entry in entries:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry['tier']))}</td>"
            f"<td>{html.escape(str(entry['model_provider']))}/{html.escape(str(entry['model_name']))}</td>"
            f"<td><code>{html.escape(str(entry['model_revision']))}</code></td>"
            f"<td>{_format_metric(entry['organic_closed_sanction_score'])}</td>"
            f"<td>{_format_rate(entry['organic_closed_clean_audit_rate'])}</td>"
            f"<td>{_format_rate(entry['organic_clean_control_false_alarm_rate'])}</td>"
            f"<td>{_format_metric(entry['organic_extra_verifications_per_document'])}</td>"
            f"<td>{_format_metric(entry['citation_closed_sanction_score'])}</td>"
            f"<td>{_format_rate(entry['citation_closed_false_positive_rate'])}</td>"
            f"<td>{_format_metric(entry['citation_tool_sanction_score'])}</td>"
            f"<td>{_format_rate(entry['citation_tool_false_positive_rate'])}</td>"
            f"<td>{_format_metric(entry['document_closed_sanction_score'])}</td>"
            f"<td>{_format_metric(entry['document_tool_sanction_score'])}</td>"
            f"<td>{_format_rate(entry['document_tool_clean_audit_rate'])}</td>"
            f"<td>{_format_metric(entry['document_tool_extra_verifications_per_document'])}</td>"
            f"<td>{entry['model_query_count']}</td>"
            "</tr>"
        )
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SanctionBench Leaderboard</title><style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#17202a}
.table-scroll{max-width:100%;overflow-x:auto}table{border-collapse:collapse;min-width:1100px;width:100%}
th,td{border:1px solid #d5d8dc;padding:8px;text-align:right}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}th{background:#f4f6f7}code{font-size:.9em}
.warning{background:#fff4d6;border-left:4px solid #d4a017;padding:12px}
.metric-notes{background:#f4f6f7;padding:12px;margin:16px 0}.meta{color:#566573}
.scroll-hint{display:none;font-size:.9rem;margin:.5rem 0;color:#566573}
@media(max-width:800px){.scroll-hint{display:block}}
</style></head><body><h1>SanctionBench Leaderboard</h1>
<p class="meta">Benchmark version <strong>"""
        + html.escape(benchmark_version)
        + "</strong>. Built from "
        + str(submission_count)
        + " validated aggregate submission"
        + ("" if submission_count == 1 else "s")
        + '.</p><nav aria-label="Project links"><a href="'
        + REPOSITORY_URL
        + '">Repository</a> &middot; <a href="'
        + SCORING_URL
        + '">Scoring contract</a></nav>'
        + """<p class="warning">"""
        + html.escape(warning)
        + """</p><section class="metric-notes" aria-labelledby="metric-definitions"><h2 id="metric-definitions">Metric definitions</h2><p><strong>SanctionScore</strong> is offending-authority recall on a 0 to 100 scale. <strong>Clean audits</strong> are documents with no missed offending authorities. <strong>FPR</strong> and <strong>control alarms</strong> measure incorrect flags against real authorities or clean control documents. <strong>Extra reviews per document</strong> counts flags that require additional verification.</p></section><p class="scroll-hint">Scroll horizontally to view all metrics.</p><div class="table-scroll" role="region" aria-label="Leaderboard metrics" tabindex="0"><table><thead><tr>
<th>Tier</th><th>Model</th><th>Revision</th><th>Organic closed</th><th>Organic clean audits</th>
<th>Control alarms</th><th>Organic extra reviews/doc</th><th>Citation closed</th><th>Closed FPR</th>
<th>Citation tool</th><th>Tool FPR</th><th>Document closed</th><th>Document tool</th>
<th>Tool clean audits</th><th>Tool extra reviews/doc</th><th>Queries</th></tr></thead><tbody>"""
        + "".join(body_rows)
        + """</tbody></table></div></body></html>
"""
    )
