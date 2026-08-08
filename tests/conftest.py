from __future__ import annotations

from sanctionbench.models import CitationItem, GoldLabel, SourceProvenance, VerificationEvidence


def make_item(
    item_id: str,
    label: GoldLabel,
    pair_id: str,
    *,
    citation: str | None = None,
    source_case_name: str = "Fixture v. Fixture",
) -> CitationItem:
    real = label == GoldLabel.REAL
    return CitationItem(
        item_id=item_id,
        citation=citation or f"{100 + len(item_id)} F.3d {200 + len(item_id)}",
        case_name=f"Example {item_id} v. Example",
        proposition="The authority exists under this case name and citation.",
        gold_label=label,
        binary_gold="real" if real else "fake",
        track="matched_real" if real else "organic",
        matched_pair_id=pair_id,
        jurisdiction="USA",
        source_case_name=source_case_name,
        source_decision_date="2025-01-01",
        database_entry_date=None,
        database_entry_date_status="not_provided_by_source",
        first_observed_snapshot_date="2026-07-11",
        temporal_bucket="2025",
        provenance=SourceProvenance(
            database_name="fixture",
            database_row_key="fixture|2025-01-01",
            database_url="https://example.test/database",
            order_url="https://example.test/order.pdf",
            order_sha256="0" * 64,
            order_page=1,
            order_excerpt="Fixture court finding.",
            extraction_method="fixture",
        ),
        verification=VerificationEvidence(
            checked_at="2026-07-11T00:00:00Z",
            queries=["fixture"],
            result_counts=[1 if real else 0],
            exact_match_found=real,
            matched_urls=["https://example.test/opinion"] if real else [],
            status="confirmed_exists" if real else "confirmed_not_found",
            response_sha256=["1" * 64],
            response_retrieved_at=["2026-07-11T00:00:00Z"],
            result_summaries=(
                [
                    {
                        "query_index": 0,
                        "case_name": f"Example {item_id} v. Example",
                        "citations": [
                            citation or f"{100 + len(item_id)} F.3d {200 + len(item_id)}"
                        ],
                        "url": "https://example.test/opinion",
                    }
                ]
                if real
                else []
            ),
            limitations="fixture",
        ),
    )
