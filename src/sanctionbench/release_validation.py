"""Publication-safe dataset validation with no acquisition or raw-file dependency."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import CitationItem, DocumentScenario
from .util import DEFAULT_MAX_MANIFEST_BYTES, read_json, read_jsonl, sha256_file


def validate_public_datasets(
    citation_path: Path,
    document_path: Path | None = None,
) -> dict[str, Any]:
    """Validate public schemas and cross-item invariants without private provenance files."""

    items = [CitationItem.model_validate(row) for row in read_jsonl(citation_path)]
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate citation item IDs")
    pairs: dict[str, list[CitationItem]] = defaultdict(list)
    for item in items:
        pairs[item.matched_pair_id].append(item)
        if not item.verification.response_sha256:
            raise ValueError(f"{item.item_id}: verification response hashes are missing")
        if not item.verification.limitations:
            raise ValueError(f"{item.item_id}: verification limitations are missing")
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"{pair_id}: expected exactly two matched items")
        if Counter(item.binary_gold for item in pair) != Counter({"fake": 1, "real": 1}):
            raise ValueError(f"{pair_id}: expected one fake and one real item")

    result: dict[str, Any] = {
        "citation_item_count": len(items),
        "matched_pair_count": len(pairs),
        "fake_count": sum(item.binary_gold == "fake" for item in items),
        "real_count": sum(item.binary_gold == "real" for item in items),
        "citation_dataset_sha256": sha256_file(citation_path),
        "document_scenario_count": 0,
    }
    if document_path is None:
        return result

    scenarios = [DocumentScenario.model_validate(row) for row in read_jsonl(document_path)]
    scenario_ids = [scenario.item_id for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Duplicate document scenario IDs")
    item_by_id = {item.item_id: item for item in items}
    authority_count = 0
    for scenario in scenarios:
        authority_ids = [authority.authority_id for authority in scenario.authorities]
        if len(authority_ids) != len(set(authority_ids)):
            raise ValueError(f"{scenario.item_id}: duplicate authority IDs")
        for authority in scenario.authorities:
            citation_item = item_by_id.get(authority.citation_item_id)
            if citation_item is None:
                raise ValueError(
                    f"{scenario.item_id}: unknown citation item {authority.citation_item_id}"
                )
            if (
                authority.gold_label != citation_item.gold_label
                or authority.citation != citation_item.citation
            ):
                raise ValueError(
                    f"{scenario.item_id}:{authority.authority_id}: citation gold mismatch"
                )
            observed = scenario.document_text[authority.start_char : authority.end_char]
            if observed != authority.citation:
                raise ValueError(
                    f"{scenario.item_id}:{authority.authority_id}: character span mismatch"
                )
            authority_count += 1
    result.update(
        {
            "document_scenario_count": len(scenarios),
            "document_authority_count": authority_count,
            "document_dataset_sha256": sha256_file(document_path),
        }
    )
    manifest_path = citation_path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path, max_bytes=DEFAULT_MAX_MANIFEST_BYTES)
        if not isinstance(manifest, dict):
            raise ValueError("Public gold manifest must be a JSON object")
        expected = {
            "citation_items_sha256": result["citation_dataset_sha256"],
            "document_scenarios_sha256": result["document_dataset_sha256"],
            "item_count": result["citation_item_count"],
            "document_scenario_count": result["document_scenario_count"],
            "document_authority_count": result["document_authority_count"],
            "release_status": "development_public_gold",
        }
        mismatches = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Public gold manifest differs from datasets: {mismatches}")
        result["manifest_sha256"] = sha256_file(manifest_path)
        result["manifest_reconciled"] = True
    return result
