"""Open-ended whole-document audit grading for adjudicated organic filings."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from .models import (
    GoldLabel,
    OrganicAuthorityGold,
    OrganicDocumentGold,
    OrganicDocumentPrediction,
)
from .util import (
    canonical_json,
    project_root,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
)

PAGE_MARKER_PREFIX = "<!-- SANCTIONBENCH_PAGE:"
PAGE_MARKER_SUFFIX = " -->"
ORGANIC_MATCHING_CONTRACT_VERSION = "sanctionbench.organic_matching.v2"
ORGANIC_REGRADE_RECEIPT_VERSION = "sanctionbench.organic_regrade_receipt.v2"
FINALIZED_RUN_IDENTITY_VERSION = "sanctionbench.finalized_run_identity.v2"
ORGANIC_GRADER_SOURCE_FILES = ("models.py", "organic_document_audit.py", "util.py")


def normalize_authority_text(value: str) -> str:
    """Normalize reviewer-approved authority aliases without fuzzy inference."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return " ".join(tokens)


def render_document_pages(pages: list[str]) -> str:
    rendered_pages: list[str] = []
    for page_number, page_text in enumerate(pages, start=1):
        canonical_lines: list[str] = []
        previous_blank = False
        for source_line in page_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
            line = " ".join(source_line.expandtabs(4).split())
            if not line:
                if previous_blank:
                    continue
                previous_blank = True
                canonical_lines.append("")
                continue
            previous_blank = False
            canonical_lines.append(line)
        canonical_page = "\n".join(canonical_lines).strip()
        rendered_pages.append(
            f"{PAGE_MARKER_PREFIX}{page_number}{PAGE_MARKER_SUFFIX}\n\n"
            f"## Page {page_number}\n\n{canonical_page}"
        )
    return "\n\n".join(rendered_pages) + "\n"


def parse_document_pages(document_text: str) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    current_page: int | None = None
    for line in document_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(PAGE_MARKER_PREFIX) and stripped.endswith(PAGE_MARKER_SUFFIX):
            value = stripped[len(PAGE_MARKER_PREFIX) : -len(PAGE_MARKER_SUFFIX)]
            try:
                page_number = int(value)
            except ValueError:
                page_number = 0
            if page_number < 1 or page_number in pages:
                raise ValueError(f"Invalid or duplicate document page marker: {stripped}")
            current_page = page_number
            pages[current_page] = []
            continue
        if current_page is None:
            if stripped:
                raise ValueError("Document text must begin with a one-based page marker")
            continue
        pages[current_page].append(line)
    return {page: "\n".join(lines).strip() for page, lines in pages.items()}


def load_organic_document_gold(path: Path) -> list[OrganicDocumentGold]:
    records = [OrganicDocumentGold.model_validate(row) for row in read_jsonl(path)]
    validate_organic_document_gold(records)
    return records


def load_organic_document_predictions(path: Path) -> list[OrganicDocumentPrediction]:
    return [OrganicDocumentPrediction.model_validate(row) for row in read_jsonl(path)]


def validate_organic_document_gold(
    documents: list[OrganicDocumentGold], *, require_clean_controls: bool = True
) -> dict[str, Any]:
    if not documents:
        raise ValueError("Organic document gold is empty")
    item_ids = [document.item_id for document in documents]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Organic document item IDs must be unique")
    document_hashes = [document.document_sha256 for document in documents]
    if len(document_hashes) != len(set(document_hashes)):
        raise ValueError("Organic document gold must deduplicate identical document bytes")

    fake_count = 0
    real_count = 0
    offending_count = 0
    clean_count = 0
    filed_count = 0
    constructed_control_count = 0
    for document in documents:
        observed_markdown_hash = sha256_bytes(document.document_markdown.encode("utf-8"))
        if observed_markdown_hash != document.document_markdown_sha256:
            raise ValueError(f"{document.item_id}: document Markdown SHA-256 mismatch")
        pages = parse_document_pages(document.document_markdown)
        if set(pages) != set(range(1, document.page_count + 1)):
            raise ValueError(f"{document.item_id}: page markers do not match page_count")
        offending_count += int(document.document_kind == "offending")
        clean_count += int(document.document_kind == "clean_control")
        filed_count += int(document.document_origin == "filed_party_document")
        constructed_control_count += int(
            document.document_origin == "constructed_verified_real_control"
        )
        for authority in document.authorities:
            aliases = [authority.citation_text, *authority.citation_aliases]
            normalized_aliases = [normalize_authority_text(alias) for alias in aliases]
            if any(not alias for alias in normalized_aliases):
                raise ValueError(f"{document.item_id}:{authority.occurrence_id}: empty alias")
            if len(normalized_aliases[0]) < 4:
                raise ValueError(
                    f"{document.item_id}:{authority.occurrence_id}: citation text is too short"
                )
            if len(normalized_aliases) != len(set(normalized_aliases)):
                raise ValueError(
                    f"{document.item_id}:{authority.occurrence_id}: duplicate normalized alias"
                )
            page_text = normalize_authority_text(pages[authority.document_page])
            normalized_excerpt = normalize_authority_text(authority.document_excerpt)
            if len(normalized_excerpt) < 12 or normalized_excerpt not in page_text:
                raise ValueError(
                    f"{document.item_id}:{authority.occurrence_id}: excerpt not found on page"
                )
            if not any(alias in page_text for alias in normalized_aliases):
                raise ValueError(
                    f"{document.item_id}:{authority.occurrence_id}: citation alias not found on page"
                )
            if authority.gold_label == GoldLabel.REAL:
                real_count += 1
            else:
                fake_count += 1
    if require_clean_controls and clean_count == 0:
        raise ValueError("Organic document gold requires at least one adjudicated clean control")
    if offending_count == 0:
        raise ValueError("Organic document gold requires at least one offending document")
    return {
        "schema_version": "sanctionbench.organic_document_validation.v1",
        "document_count": len(documents),
        "offending_document_count": offending_count,
        "clean_control_document_count": clean_count,
        "filed_party_document_count": filed_count,
        "constructed_clean_control_document_count": constructed_control_count,
        "fake_authority_occurrence_count": fake_count,
        "real_authority_occurrence_count": real_count,
        "clean_controls_required": require_clean_controls,
    }


def _authority_aliases(authority: OrganicAuthorityGold) -> set[str]:
    return {
        normalize_authority_text(value)
        for value in [authority.citation_text, *authority.citation_aliases]
    }


def _is_contiguous_token_sequence(*, shorter: str, longer: str) -> bool:
    """Return whether one normalized multi-token citation occurs verbatim in another."""

    shorter_tokens = shorter.split()
    longer_tokens = longer.split()
    if len(shorter_tokens) < 2 or len(shorter_tokens) > len(longer_tokens):
        return False
    width = len(shorter_tokens)
    return any(
        longer_tokens[index : index + width] == shorter_tokens
        for index in range(len(longer_tokens) - width + 1)
    )


def _citation_contains_alias(citation_key: str, alias: str) -> bool:
    """Match only verbatim normalized token sequences, never edit-distance or fuzzy text."""

    return _is_contiguous_token_sequence(shorter=alias, longer=citation_key) or (
        _is_contiguous_token_sequence(shorter=citation_key, longer=alias)
    )


def _match_finding(
    *,
    citation_text: str,
    quoted_text: str,
    page_number: int,
    authorities: list[OrganicAuthorityGold],
) -> tuple[OrganicAuthorityGold | None, str]:
    citation_key = normalize_authority_text(citation_text)
    candidates = [
        authority for authority in authorities if citation_key in _authority_aliases(authority)
    ]
    page_candidates = [
        authority for authority in candidates if authority.document_page == page_number
    ]
    if len(page_candidates) == 1:
        return page_candidates[0], "exact_alias_and_page"
    if len(candidates) == 1:
        return candidates[0], "exact_alias_unique_wrong_page"
    quote_key = normalize_authority_text(quoted_text)
    quote_candidates = [
        authority
        for authority in page_candidates or candidates
        if normalize_authority_text(authority.document_excerpt) in quote_key
        or quote_key in normalize_authority_text(authority.document_excerpt)
    ]
    if len(quote_candidates) == 1:
        return quote_candidates[0], "exact_alias_and_excerpt"
    if len(candidates) > 1:
        return None, "ambiguous_exact_alias"

    contained_candidates = [
        authority
        for authority in authorities
        if any(
            _citation_contains_alias(citation_key, alias) for alias in _authority_aliases(authority)
        )
    ]
    contained_page_candidates = [
        authority for authority in contained_candidates if authority.document_page == page_number
    ]
    if len(contained_page_candidates) == 1:
        return contained_page_candidates[0], "contained_alias_and_page"
    if len(contained_candidates) == 1:
        return contained_candidates[0], "contained_alias_unique_wrong_page"
    contained_quote_candidates = [
        authority
        for authority in contained_page_candidates or contained_candidates
        if normalize_authority_text(authority.document_excerpt) in quote_key
        or quote_key in normalize_authority_text(authority.document_excerpt)
    ]
    if len(contained_quote_candidates) == 1:
        return contained_quote_candidates[0], "contained_alias_and_excerpt"
    if contained_candidates:
        return None, "ambiguous_contained_alias"
    return None, "unmatched_citation"


def score_organic_document_predictions(
    documents: list[OrganicDocumentGold],
    predictions: list[OrganicDocumentPrediction],
) -> dict[str, Any]:
    validation = validate_organic_document_gold(documents)
    prediction_by_id = {prediction.item_id: prediction for prediction in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Duplicate organic document prediction IDs")
    expected_ids = {document.item_id for document in documents}
    if set(prediction_by_id) != expected_ids:
        raise ValueError(
            "Organic document prediction coverage mismatch: "
            f"missing={sorted(expected_ids - prediction_by_id.keys())}, "
            f"extra={sorted(prediction_by_id.keys() - expected_ids)}"
        )

    by_document: dict[str, dict[str, Any]] = {}
    total_fake = 0
    total_real = 0
    total_true_positive = 0
    total_false_negative = 0
    total_false_positive = 0
    total_matched_fake = 0
    total_correct_diagnosis = 0
    total_correct_page = 0
    total_adjudication_required = 0

    for document in documents:
        prediction = prediction_by_id[document.item_id]
        matched_occurrences: dict[str, dict[str, Any]] = {}
        duplicate_finding_count = 0
        unmatched_findings: list[dict[str, Any]] = []
        adjudication_findings: list[dict[str, Any]] = []
        for finding_index, finding in enumerate(prediction.findings):
            authority, match_status = _match_finding(
                citation_text=finding.citation_text,
                quoted_text=finding.quoted_text,
                page_number=finding.page_number,
                authorities=document.authorities,
            )
            finding_record = {
                "finding_index": finding_index,
                "citation_text": finding.citation_text,
                "page_number": finding.page_number,
                "predicted_label": finding.predicted_label.value,
                "match_status": match_status,
            }
            if authority is None:
                if match_status in {"ambiguous_exact_alias", "ambiguous_contained_alias"}:
                    adjudication_findings.append(finding_record)
                else:
                    unmatched_findings.append(finding_record)
                continue
            if authority.occurrence_id in matched_occurrences:
                duplicate_finding_count += 1
                continue
            matched_occurrences[authority.occurrence_id] = {
                **finding_record,
                "gold_label": authority.gold_label.value,
                "page_correct": finding.page_number == authority.document_page,
                "diagnosis_correct": finding.predicted_label.value == authority.gold_label.value,
            }

        fake_authorities = [
            authority
            for authority in document.authorities
            if authority.gold_label != GoldLabel.REAL
        ]
        real_authorities = [
            authority
            for authority in document.authorities
            if authority.gold_label == GoldLabel.REAL
        ]
        matched_fake = [
            authority
            for authority in fake_authorities
            if authority.occurrence_id in matched_occurrences
        ]
        matched_real = [
            authority
            for authority in real_authorities
            if authority.occurrence_id in matched_occurrences
        ]
        missed_fake_ids = [
            authority.occurrence_id
            for authority in fake_authorities
            if authority.occurrence_id not in matched_occurrences
        ]
        false_positive_count = (
            len(matched_real) + len(unmatched_findings) + len(adjudication_findings)
        )
        correct_diagnosis_count = sum(
            bool(matched_occurrences[authority.occurrence_id]["diagnosis_correct"])
            for authority in matched_fake
        )
        correct_page_count = sum(
            bool(matched_occurrences[authority.occurrence_id]["page_correct"])
            for authority in matched_fake
        )
        fake_count = len(fake_authorities)
        true_positive_count = len(matched_fake)
        false_negative_count = len(missed_fake_ids)
        by_document[document.item_id] = {
            "document_kind": document.document_kind,
            "fake_authority_count": fake_count,
            "real_authority_count": len(real_authorities),
            "true_positive": true_positive_count,
            "false_negative": false_negative_count,
            "false_positive": false_positive_count,
            "fake_recall": round(true_positive_count / fake_count, 6) if fake_count else None,
            "clean_audit": false_negative_count == 0 if fake_count else None,
            "clean_control_pass": false_positive_count == 0 if not fake_count else None,
            "matched_occurrences": matched_occurrences,
            "missed_fake_occurrence_ids": missed_fake_ids,
            "unmatched_findings": unmatched_findings,
            "adjudication_required_findings": adjudication_findings,
            "duplicate_finding_count": duplicate_finding_count,
        }
        total_fake += fake_count
        total_real += len(real_authorities)
        total_true_positive += true_positive_count
        total_false_negative += false_negative_count
        total_false_positive += false_positive_count
        total_matched_fake += len(matched_fake)
        total_correct_diagnosis += correct_diagnosis_count
        total_correct_page += correct_page_count
        total_adjudication_required += len(adjudication_findings)

    offending_values = [
        value for value in by_document.values() if value["document_kind"] == "offending"
    ]
    clean_values = [
        value for value in by_document.values() if value["document_kind"] == "clean_control"
    ]
    macro_recall = (
        sum(float(value["fake_recall"]) for value in offending_values) / len(offending_values)
        if offending_values
        else 0.0
    )
    clean_audit_rate = (
        sum(bool(value["clean_audit"]) for value in offending_values) / len(offending_values)
        if offending_values
        else 0.0
    )
    clean_document_pass_rate = (
        sum(bool(value["clean_control_pass"]) for value in clean_values) / len(clean_values)
        if clean_values
        else None
    )
    item_recall = total_true_positive / total_fake if total_fake else 0.0
    return {
        "schema_version": "sanctionbench.organic_document_metrics.v1",
        "matching_contract_version": ORGANIC_MATCHING_CONTRACT_VERSION,
        "validation": validation,
        "headline": {
            "name": "OrganicDocumentSanctionScore",
            "score": round(100 * macro_recall, 6),
            "definition": "macro_offending_document_hallucination_recall",
            "clean_audit_rate": round(clean_audit_rate, 6),
        },
        "authority_detection": {
            "fake_authority_count": total_fake,
            "true_positive": total_true_positive,
            "false_negative": total_false_negative,
            "micro_recall": round(item_recall, 6),
            "diagnosis_accuracy_on_caught": (
                round(total_correct_diagnosis / total_matched_fake, 6)
                if total_matched_fake
                else None
            ),
            "page_accuracy_on_caught": (
                round(total_correct_page / total_matched_fake, 6) if total_matched_fake else None
            ),
        },
        "review_workload": {
            "real_authority_count": total_real,
            "extra_verification_count": total_false_positive,
            "extra_verifications_per_document": round(total_false_positive / len(documents), 6),
            "zero_false_positive_document_rate": round(
                1
                - sum(int(value["false_positive"]) > 0 for value in by_document.values())
                / len(documents),
                6,
            ),
            "false_accusations_per_100_real_authorities": (
                round(100 * total_false_positive / total_real, 6) if total_real else None
            ),
            "documents_with_false_positive": sum(
                int(value["false_positive"]) > 0 for value in by_document.values()
            ),
            "adjudication_required_count": total_adjudication_required,
        },
        "clean_controls": {
            "document_count": len(clean_values),
            "pass_rate": (
                round(clean_document_pass_rate, 6) if clean_document_pass_rate is not None else None
            ),
            "false_alarm_rate": (
                round(1 - clean_document_pass_rate, 6)
                if clean_document_pass_rate is not None
                else None
            ),
        },
        "by_document": by_document,
    }


def score_organic_document_files(
    gold_path: Path,
    prediction_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    metrics = score_organic_document_predictions(
        load_organic_document_gold(gold_path),
        load_organic_document_predictions(prediction_path),
    )
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics


def _load_json_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _artifact_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Run record is missing {field}")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Run artifact does not exist for {field}: {resolved}")
    return resolved


def _path_reference(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def organic_grader_source_receipt() -> dict[str, Any]:
    """Hash the complete local source closure used by the organic grader."""

    package_root = Path(__file__).resolve().parent
    files = [
        {"path": name, "sha256": sha256_file(package_root / name)}
        for name in ORGANIC_GRADER_SOURCE_FILES
    ]
    return {
        "files": files,
        "aggregate_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }


def regrade_organic_run(
    run_path: Path,
    *,
    output_path: Path | None = None,
    receipt_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Regrade preserved predictions without mutating the historical run bundle."""

    effective_root = (root or project_root()).resolve()
    resolved_run_path = run_path.resolve()
    run = _load_json_object(resolved_run_path)
    metadata = run.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("task_type") != "organic_document_audit":
        raise ValueError("Run record is not an organic document audit")

    dataset_path = _artifact_path(effective_root, run.get("dataset_path"), field="dataset_path")
    prediction_path = _artifact_path(
        effective_root, run.get("predictions_path"), field="predictions_path"
    )
    source_metrics_path = _artifact_path(
        effective_root, run.get("metrics_path"), field="metrics_path"
    )
    expected_dataset_sha256 = run.get("dataset_sha256")
    if not isinstance(expected_dataset_sha256, str) or (
        sha256_file(dataset_path) != expected_dataset_sha256
    ):
        raise ValueError("Run dataset SHA-256 does not match the preserved dataset")

    identity_path = resolved_run_path.parent / "identity.json"
    if not identity_path.is_file():
        raise ValueError(f"Run identity artifact does not exist: {identity_path}")
    identity_receipt = _load_json_object(identity_path)
    source_identity = identity_receipt.get("identity")
    source_identity_sha256 = identity_receipt.get("run_identity_sha256")
    if not isinstance(source_identity, dict) or not isinstance(source_identity_sha256, str):
        raise ValueError("Run identity artifact is malformed")
    if sha256_bytes(canonical_json(source_identity).encode("utf-8")) != source_identity_sha256:
        raise ValueError("Run identity artifact digest is invalid")
    if (
        metadata.get("run_identity") != source_identity
        or metadata.get("run_identity_sha256") != source_identity_sha256
    ):
        raise ValueError("Run record and identity artifact do not match")
    request_attempts_value = metadata.get("provider_request_attempts_path")
    if request_attempts_value is None:
        raise ValueError("Run record lacks a provider request-attempt ledger")
    request_attempts_path = _artifact_path(
        effective_root,
        request_attempts_value,
        field="metadata.provider_request_attempts_path",
    )
    run_dir = resolved_run_path.parent
    for artifact, expected_name in (
        (prediction_path, "predictions.jsonl"),
        (source_metrics_path, "metrics.json"),
        (request_attempts_path, "request_attempts.json"),
    ):
        if artifact.parent != run_dir or artifact.name != expected_name:
            raise ValueError(f"Finalized run artifact is outside its run directory: {artifact}")

    request_attempts = _load_json_object(request_attempts_path)
    counts = request_attempts.get("attempts_started_by_item")
    selected_ids = source_identity.get("selected_ids")
    if (
        request_attempts.get("schema_version") != "sanctionbench.provider_request_attempts.v1"
        or request_attempts.get("run_identity_sha256") != source_identity_sha256
        or request_attempts.get("selected_ids") != selected_ids
        or not isinstance(selected_ids, list)
        or not isinstance(counts, dict)
        or set(counts) != set(selected_ids)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in counts.values()
        )
    ):
        raise ValueError("Run provider request-attempt ledger does not reconcile")
    provider_request_count = sum(counts.values())
    if (
        request_attempts.get("provider_request_count") != provider_request_count
        or metadata.get("provider_request_count") != provider_request_count
        or metadata.get("successful_response_count") != len(selected_ids)
    ):
        raise ValueError("Run record and provider request-attempt ledger do not match")

    finalized = metadata.get("finalized_run_identity")
    finalized_sha256 = metadata.get("finalized_run_identity_sha256")
    if not isinstance(finalized, dict) or not isinstance(finalized_sha256, str):
        raise ValueError("Run record lacks a finalized run identity")
    core = json.loads(json.dumps(run))
    core_metadata = core.get("metadata") or {}
    core_metadata.pop("finalized_run_identity", None)
    core_metadata.pop("finalized_run_identity_sha256", None)
    expected_finalized = {
        "schema_version": FINALIZED_RUN_IDENTITY_VERSION,
        "benchmark_release_status": finalized.get("benchmark_release_status"),
        "run_identity_sha256": source_identity_sha256,
        "provider_request_count": provider_request_count,
        "successful_response_count": len(selected_ids),
        "provider_request_attempts_sha256": sha256_file(request_attempts_path),
        "courtlistener_request_attempts_sha256": None,
        "tool_evidence_sha256": None,
        "predictions_sha256": sha256_file(prediction_path),
        "metrics_sha256": sha256_file(source_metrics_path),
        "run_record_core_sha256": sha256_bytes(canonical_json(core).encode("utf-8")),
    }
    expected_finalized_sha256 = sha256_bytes(canonical_json(expected_finalized).encode("utf-8"))
    if finalized != expected_finalized or finalized_sha256 != expected_finalized_sha256:
        raise ValueError("Run finalized identity does not reconcile with preserved artifacts")

    resolved_output_path = output_path or resolved_run_path.parent / "metrics.regraded-v2.json"
    resolved_receipt_path = receipt_path or resolved_run_path.parent / "regrade-receipt-v2.json"
    protected_paths = {
        resolved_run_path,
        dataset_path,
        prediction_path,
        source_metrics_path,
        identity_path,
        request_attempts_path,
    }
    if resolved_output_path.resolve() in protected_paths:
        raise ValueError("Regraded metrics must not overwrite a preserved run artifact")
    if resolved_receipt_path.resolve() in protected_paths:
        raise ValueError("Regrade receipt must not overwrite a preserved run artifact")
    if resolved_output_path.resolve() == resolved_receipt_path.resolve():
        raise ValueError("Regraded metrics and receipt paths must be distinct")
    if resolved_output_path.exists() or resolved_receipt_path.exists():
        raise FileExistsError("Regrade outputs already exist; choose fresh paths")

    metrics = score_organic_document_files(dataset_path, prediction_path)
    write_json(resolved_output_path, metrics)
    receipt: dict[str, Any] = {
        "schema_version": ORGANIC_REGRADE_RECEIPT_VERSION,
        "created_at": utc_now(),
        "matching_contract_version": ORGANIC_MATCHING_CONTRACT_VERSION,
        "source_run_id": run.get("run_id"),
        "source_run_identity_sha256": source_identity_sha256,
        "source_finalized_run_identity_sha256": expected_finalized_sha256,
        "provider_request_count_before_regrade": metadata.get("provider_request_count"),
        "provider_requests_added": 0,
        "historical_metrics_superseded": True,
        "grader_source": organic_grader_source_receipt(),
        "artifacts": {
            "run": {
                "path": _path_reference(effective_root, resolved_run_path),
                "sha256": sha256_file(resolved_run_path),
            },
            "identity": {
                "path": _path_reference(effective_root, identity_path),
                "sha256": sha256_file(identity_path),
            },
            "dataset": {
                "path": _path_reference(effective_root, dataset_path),
                "sha256": sha256_file(dataset_path),
            },
            "predictions": {
                "path": _path_reference(effective_root, prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "historical_metrics": {
                "path": _path_reference(effective_root, source_metrics_path),
                "sha256": sha256_file(source_metrics_path),
            },
            "regraded_metrics": {
                "path": _path_reference(effective_root, resolved_output_path),
                "sha256": sha256_file(resolved_output_path),
            },
        },
    }
    receipt["artifacts"]["request_attempts"] = {
        "path": _path_reference(effective_root, request_attempts_path),
        "sha256": sha256_file(request_attempts_path),
    }
    write_json(resolved_receipt_path, receipt)
    return {"metrics": metrics, "receipt": receipt}
