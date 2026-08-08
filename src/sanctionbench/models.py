"""Versioned schemas for gold items, predictions, and provenance."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_ID_LENGTH = 200
MAX_TITLE_LENGTH = 1_000
MAX_CITATION_LENGTH = 1_000
MAX_CASE_NAME_LENGTH = 500
MAX_PROPOSITION_LENGTH = 20_000
MAX_DOCUMENT_TEXT_LENGTH = 4_000_000
MAX_DOCUMENT_AUTHORITIES = 512
MAX_SOURCE_DECISIONS = 512


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldLabel(StrEnum):
    REAL = "real"
    NONEXISTENT_CASE = "nonexistent_case"
    FABRICATED_QUOTE = "fabricated_quote"
    MISATTRIBUTED_HOLDING = "misattributed_holding"


class PredictedLabel(StrEnum):
    REAL = "real"
    NONEXISTENT_CASE = "nonexistent_case"
    FABRICATED_QUOTE = "fabricated_quote"
    MISATTRIBUTED_HOLDING = "misattributed_holding"
    UNCERTAIN_NEEDS_REVIEW = "uncertain_needs_review"


class TaskType(StrEnum):
    CITATION_VERIFICATION = "citation_verification"
    DOCUMENT_AUDIT = "document_audit"
    ORGANIC_DOCUMENT_AUDIT = "organic_document_audit"
    MEMORIZATION_PROBE = "memorization_probe"


class Condition(StrEnum):
    CLOSED_BOOK = "closed_book"
    TOOL_ASSISTED = "tool_assisted"


class TemporalCutoff(StrictModel):
    """Contamination filter applied before matched-pair sampling.

    The cutoff is an exclusive lower bound: only items whose selected date is
    strictly later than ``cutoff_date`` are eligible for a run.
    """

    field: Literal["first_observed_snapshot_date", "database_entry_date"]
    cutoff_date: str
    missing: Literal["exclude", "error"] = "exclude"

    @field_validator("cutoff_date", mode="before")
    @classmethod
    def cutoff_is_iso_date(cls, value: Any) -> str:
        normalized = str(value)
        try:
            parsed = date.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError("cutoff_date must be an ISO 8601 date (YYYY-MM-DD)") from error
        if parsed.isoformat() != normalized:
            raise ValueError("cutoff_date must use canonical YYYY-MM-DD form")
        return normalized


class SourceProvenance(StrictModel):
    database_name: str
    database_row_key: str
    database_url: str
    order_url: str
    order_sha256: str
    order_page: int | None = None
    order_excerpt: str = Field(max_length=600)
    offending_document_url: str | None = None
    offending_document_sha256: str | None = None
    extraction_method: str
    reviewed_by: str = "SanctionBench maintainers"


class VerificationEvidence(StrictModel):
    provider: Literal["courtlistener_v4_search"] = "courtlistener_v4_search"
    checked_at: str
    queries: list[str]
    result_counts: list[int]
    exact_match_found: bool
    matched_urls: list[str] = Field(default_factory=list)
    status: Literal[
        "confirmed_exists",
        "confirmed_not_found",
        "confirmed_quote_not_found",
        "confirmed_mismatch",
    ]
    response_sha256: list[str]
    limitations: str
    response_retrieved_at: list[str]
    result_summaries: list[dict[str, Any]] = Field(default_factory=list)


class CitationItem(StrictModel):
    schema_version: Literal["sanctionbench.citation.v1"] = "sanctionbench.citation.v1"
    item_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    task_type: Literal[TaskType.CITATION_VERIFICATION] = TaskType.CITATION_VERIFICATION
    citation: str = Field(min_length=1, max_length=MAX_CITATION_LENGTH)
    case_name: str = Field(min_length=1, max_length=MAX_CASE_NAME_LENGTH)
    proposition: str = Field(max_length=MAX_PROPOSITION_LENGTH)
    gold_label: GoldLabel
    binary_gold: Literal["real", "fake"]
    track: Literal["organic", "matched_real"]
    matched_pair_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    jurisdiction: str = Field(max_length=200)
    source_case_name: str = Field(max_length=MAX_CASE_NAME_LENGTH)
    source_decision_date: str
    database_entry_date: str | None
    database_entry_date_status: Literal["provided", "not_provided_by_source"]
    first_observed_snapshot_date: str
    temporal_bucket: str = Field(max_length=200)
    famous_case: bool = False
    provenance: SourceProvenance
    verification: VerificationEvidence
    tags: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("binary_gold")
    @classmethod
    def binary_matches_label(cls, value: str, info: Any) -> str:
        label = info.data.get("gold_label")
        expected = "real" if label == GoldLabel.REAL else "fake"
        if label is not None and value != expected:
            raise ValueError(f"binary_gold must be {expected!r} for {label!s}")
        return value


class DocumentAuthorityGold(StrictModel):
    authority_id: str = Field(min_length=1, max_length=64)
    citation_item_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    citation: str = Field(min_length=1, max_length=MAX_CITATION_LENGTH)
    gold_label: GoldLabel
    start_char: int
    end_char: int


class DocumentScenario(StrictModel):
    schema_version: Literal["sanctionbench.document.v1"] = "sanctionbench.document.v1"
    item_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    task_type: Literal[TaskType.DOCUMENT_AUDIT] = TaskType.DOCUMENT_AUDIT
    title: str = Field(max_length=MAX_TITLE_LENGTH)
    document_text: str = Field(max_length=MAX_DOCUMENT_TEXT_LENGTH)
    track: Literal["organic_brief", "constructed_from_organic"]
    construction_manifest: list[str] = Field(
        default_factory=list, max_length=MAX_DOCUMENT_AUTHORITIES
    )
    authorities: list[DocumentAuthorityGold] = Field(max_length=MAX_DOCUMENT_AUTHORITIES)
    source_decision_dates: list[str] = Field(max_length=MAX_SOURCE_DECISIONS)


class DocumentAssessment(StrictModel):
    authority_id: str
    predicted_label: PredictedLabel
    fake_probability: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=2_000)


class DocumentPrediction(StrictModel):
    item_id: str
    assessments: list[DocumentAssessment]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class OrganicDocumentSourceDecision(StrictModel):
    decision_id: str
    case_name: str
    decision_date: str
    order_url: str
    order_sha256: str
    source_format: Literal["pdf", "html", "text"]


class OrganicReviewActor(StrictModel):
    """Secret-free provenance for one human or model curation actor."""

    actor_id: str
    actor_type: Literal["human", "model"]
    provider: str | None
    model: str | None
    role: Literal["reviewer_a", "reviewer_b", "adjudicator", "human_reviewer"]
    call_id: str | None
    prompt_sha256: str | None
    response_sha256: str | None
    backend_version: str | None
    receipt_file: str | None
    receipt_sha256: str | None
    isolated_from_other_reviews: bool

    @model_validator(mode="after")
    def actors_require_verifiable_provenance(self) -> OrganicReviewActor:
        if self.actor_type == "model":
            missing = [
                name
                for name, value in (
                    ("provider", self.provider),
                    ("model", self.model),
                    ("call_id", self.call_id),
                    ("prompt_sha256", self.prompt_sha256),
                    ("response_sha256", self.response_sha256),
                    ("backend_version", self.backend_version),
                    ("receipt_file", self.receipt_file),
                    ("receipt_sha256", self.receipt_sha256),
                )
                if not value
            ]
            if missing:
                raise ValueError("model review actor is missing: " + ", ".join(missing))
        else:
            if not self.receipt_file or not self.receipt_sha256:
                raise ValueError("human review actors require a verifiable receipt file and hash")
            model_only = [
                name
                for name, value in (
                    ("provider", self.provider),
                    ("model", self.model),
                    ("call_id", self.call_id),
                    ("prompt_sha256", self.prompt_sha256),
                    ("response_sha256", self.response_sha256),
                    ("backend_version", self.backend_version),
                )
                if value is not None
            ]
            if model_only:
                raise ValueError(
                    "human review actor has model-only fields: " + ", ".join(model_only)
                )
        return self


class OrganicAuthorityGold(StrictModel):
    occurrence_id: str
    citation_text: str = Field(min_length=1, max_length=1_000)
    citation_aliases: list[str] = Field(default_factory=list)
    case_name: str | None = Field(default=None, max_length=500)
    proposition: str | None = Field(default=None, max_length=2_000)
    gold_label: GoldLabel
    document_page: int = Field(ge=1)
    document_excerpt: str = Field(min_length=1, max_length=1_500)
    source_decision_id: str | None = None
    source_order_page: int | None = Field(default=None, ge=1)
    source_order_excerpt: str | None = Field(default=None, max_length=1_500)

    @model_validator(mode="after")
    def fake_authorities_require_court_evidence(self) -> OrganicAuthorityGold:
        if self.gold_label != GoldLabel.REAL:
            missing = [
                name
                for name, value in (
                    ("source_decision_id", self.source_decision_id),
                    ("source_order_page", self.source_order_page),
                    ("source_order_excerpt", self.source_order_excerpt),
                )
                if value is None or value == ""
            ]
            if missing:
                raise ValueError(
                    "fake authority occurrences require court evidence: " + ", ".join(missing)
                )
        return self


class OrganicDocumentInput(StrictModel):
    item_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    title: str = Field(max_length=MAX_TITLE_LENGTH)
    document_markdown: str = Field(max_length=MAX_DOCUMENT_TEXT_LENGTH)
    document_markdown_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)


class OrganicDocumentNormalization(StrictModel):
    input_format: Literal["canonical_markdown"] = "canonical_markdown"
    parser: Literal["liteparse", "deterministic_constructed_markdown"] = "liteparse"
    parser_version: str
    parser_output_format: Literal["json", "canonical_markdown"] = "json"
    ocr_used: bool
    page_marker_contract: Literal["sanctionbench_page_comment_v1"] = "sanctionbench_page_comment_v1"
    parser_output_sha256: str


class OrganicDocumentGold(StrictModel):
    schema_version: Literal["sanctionbench.organic_document.v1"] = (
        "sanctionbench.organic_document.v1"
    )
    item_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    task_type: Literal[TaskType.ORGANIC_DOCUMENT_AUDIT] = TaskType.ORGANIC_DOCUMENT_AUDIT
    title: str = Field(max_length=MAX_TITLE_LENGTH)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_markdown: str = Field(max_length=MAX_DOCUMENT_TEXT_LENGTH)
    document_markdown_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    normalization: OrganicDocumentNormalization
    document_kind: Literal["offending", "clean_control"]
    document_origin: Literal["filed_party_document", "constructed_verified_real_control"] = (
        "filed_party_document"
    )
    authority_inventory_complete: Literal[True]
    authorities: list[OrganicAuthorityGold] = Field(max_length=MAX_DOCUMENT_AUTHORITIES)
    source_decisions: list[OrganicDocumentSourceDecision] = Field(
        default_factory=list, max_length=MAX_SOURCE_DECISIONS
    )
    review_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_count: int = Field(ge=0)
    reviewer_types: list[Literal["human", "model"]] = Field(max_length=16)
    adjudicator_type: Literal["human", "model", "deterministic"]
    human_reviewed: bool
    curation_method: Literal[
        "independent_human_double_review",
        "frontier_model_double_review_with_court_source_adjudication",
        "deterministic_constructed_control_from_verified_real_gold",
    ]
    release_tier: Literal["human_adjudicated", "provisional_model_assisted", "constructed_control"]
    redistribution_status: Literal["cleared_public", "private_evaluation_only"]

    @model_validator(mode="after")
    def validate_document_gold_contract(self) -> OrganicDocumentGold:
        if len(self.reviewer_types) != self.reviewer_count:
            raise ValueError("reviewer_types must reconcile with reviewer_count")
        if self.document_origin == "constructed_verified_real_control":
            if self.document_kind != "clean_control":
                raise ValueError("constructed controls must be clean-control documents")
            if self.human_reviewed:
                raise ValueError("deterministically constructed controls are not human reviewed")
            if self.reviewer_count != 0 or self.reviewer_types:
                raise ValueError("constructed controls must not invent document reviewers")
            if self.adjudicator_type != "deterministic":
                raise ValueError("constructed controls require a deterministic adjudicator type")
            if self.curation_method != "deterministic_constructed_control_from_verified_real_gold":
                raise ValueError("constructed controls require the explicit construction method")
            if self.release_tier != "constructed_control":
                raise ValueError("constructed controls require the constructed-control tier")
            if self.redistribution_status != "cleared_public":
                raise ValueError("constructed controls must contain only redistributable text")
            if (
                self.normalization.parser != "deterministic_constructed_markdown"
                or self.normalization.parser_output_format != "canonical_markdown"
            ):
                raise ValueError("constructed controls require constructed-Markdown provenance")
        elif self.human_reviewed:
            if self.normalization.parser != "liteparse":
                raise ValueError("filed documents require LiteParse normalization provenance")
            if self.reviewer_count < 2:
                raise ValueError("filed documents require at least two reviewers")
            if self.curation_method != "independent_human_double_review":
                raise ValueError("human-reviewed gold requires the human review method")
            if self.release_tier != "human_adjudicated":
                raise ValueError("human-reviewed gold requires the human-adjudicated tier")
            if set(self.reviewer_types) != {"human"} or self.adjudicator_type != "human":
                raise ValueError("human-reviewed gold may name only human review actors")
        else:
            if self.normalization.parser != "liteparse":
                raise ValueError("filed documents require LiteParse normalization provenance")
            if self.reviewer_count < 2:
                raise ValueError("filed documents require at least two reviewers")
            if (
                self.curation_method
                != "frontier_model_double_review_with_court_source_adjudication"
            ):
                raise ValueError("model-assisted gold requires the explicit model review method")
            if self.release_tier != "provisional_model_assisted":
                raise ValueError("model-assisted gold must remain provisional")
            if set(self.reviewer_types) != {"model"} or self.adjudicator_type != "model":
                raise ValueError("model-assisted gold must identify model review actors")
        occurrence_ids = [authority.occurrence_id for authority in self.authorities]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("organic document authority occurrence IDs must be unique")
        source_ids = {source.decision_id for source in self.source_decisions}
        unknown_sources = {
            authority.source_decision_id
            for authority in self.authorities
            if authority.source_decision_id is not None
            and authority.source_decision_id not in source_ids
        }
        if unknown_sources:
            raise ValueError(
                f"authority occurrences reference unknown source decisions: {sorted(unknown_sources)}"
            )
        fake_count = sum(authority.gold_label != GoldLabel.REAL for authority in self.authorities)
        if self.document_kind == "offending" and fake_count == 0:
            raise ValueError("offending organic documents require at least one fake authority")
        if self.document_kind == "clean_control" and fake_count != 0:
            raise ValueError("clean-control organic documents may contain only real authorities")
        for authority in self.authorities:
            if authority.document_page > self.page_count:
                raise ValueError(
                    f"{authority.occurrence_id}: document_page exceeds document page_count"
                )
        return self

    def model_input(self) -> OrganicDocumentInput:
        return OrganicDocumentInput(
            item_id=self.item_id,
            title=self.title,
            document_markdown=self.document_markdown,
            document_markdown_sha256=self.document_markdown_sha256,
            page_count=self.page_count,
        )


class OrganicDocumentFinding(StrictModel):
    citation_text: str = Field(min_length=1, max_length=1_000)
    case_name: str | None = Field(default=None, max_length=500)
    page_number: int = Field(ge=1)
    quoted_text: str = Field(min_length=1, max_length=1_500)
    predicted_label: PredictedLabel
    fake_probability: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=2_000)

    @field_validator("predicted_label")
    @classmethod
    def findings_are_flags(cls, value: PredictedLabel) -> PredictedLabel:
        if value == PredictedLabel.REAL:
            raise ValueError("organic findings must contain only flagged authorities")
        return value


class OrganicDocumentPrediction(StrictModel):
    item_id: str
    findings: list[OrganicDocumentFinding]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class Prediction(StrictModel):
    item_id: str
    predicted_label: PredictedLabel
    fake_probability: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=2_000)
    cited_evidence: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class RunRecord(StrictModel):
    schema_version: Literal["sanctionbench.run.v1"] = "sanctionbench.run.v1"
    run_id: str
    created_at: str
    provider: str
    model: str
    condition: Condition
    dataset_path: str
    dataset_sha256: str
    seed: int
    mock: bool
    predictions_path: str
    metrics_path: str
    item_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
