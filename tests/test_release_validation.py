import shutil
from pathlib import Path

import pytest

from sanctionbench.release_validation import validate_public_datasets
from sanctionbench.util import DEFAULT_MAX_MANIFEST_BYTES


def test_public_release_datasets_validate_without_private_files() -> None:
    result = validate_public_datasets(
        Path("data/gold/v1/citation_items.jsonl"),
        Path("data/gold/v1/document_scenarios.jsonl"),
    )
    assert result["citation_item_count"] == 152
    assert result["matched_pair_count"] == 76
    assert result["fake_count"] == result["real_count"] == 76
    assert result["document_scenario_count"] == 10
    assert result["manifest_reconciled"] is True
    assert len(result["manifest_sha256"]) == 64
    assert result["document_authority_count"] == 196


def test_public_release_rejects_oversized_adjacent_manifest(tmp_path: Path) -> None:
    source = Path("data/gold/v1")
    destination = tmp_path / "gold"
    destination.mkdir()
    for name in ("citation_items.jsonl", "document_scenarios.jsonl"):
        shutil.copy2(source / name, destination / name)
    (destination / "manifest.json").write_bytes(b" " * (DEFAULT_MAX_MANIFEST_BYTES + 1))

    with pytest.raises(ValueError, match="JSON file exceeds"):
        validate_public_datasets(
            destination / "citation_items.jsonl",
            destination / "document_scenarios.jsonl",
        )
