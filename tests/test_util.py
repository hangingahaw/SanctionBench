from __future__ import annotations

from pathlib import Path

import pytest

from sanctionbench.util import (
    append_jsonl_record,
    read_jsonl,
    read_text_bounded,
    write_json,
    write_jsonl,
)


def test_jsonl_gzip_roundtrip_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl.gz"
    rows = [{"b": 2, "a": "é"}, {"value": [1, 2, 3]}]

    write_jsonl(path, rows)
    first = path.read_bytes()
    write_jsonl(path, rows)

    assert path.read_bytes() == first
    assert int.from_bytes(first[4:8], byteorder="little") == 0
    assert read_jsonl(path) == rows


def test_write_json_enforces_serialized_byte_limit_before_destination_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bounded.json"

    with pytest.raises(ValueError, match="Serialized JSON exceeds"):
        write_json(path, {"value": "x" * 100}, max_bytes=32)

    assert not path.exists()
    assert not list(tmp_path.glob(".bounded.json.*"))


def test_jsonl_plain_roundtrip_remains_supported(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [{"value": "plain"}]

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_jsonl_reader_enforces_line_row_and_expansion_limits(tmp_path: Path) -> None:
    plain = tmp_path / "records.jsonl"
    write_jsonl(plain, [{"value": "a"}, {"value": "b"}])
    with pytest.raises(ValueError, match="row limit"):
        read_jsonl(plain, max_rows=1)
    with pytest.raises(ValueError, match="line exceeds"):
        read_jsonl(plain, max_line_bytes=8)

    compressed = tmp_path / "records.jsonl.gz"
    write_jsonl(compressed, [{"value": "x" * 10_000}])
    with pytest.raises(ValueError, match="line exceeds"):
        read_jsonl(
            compressed,
            max_line_bytes=64,
            max_expanded_bytes=100_000,
        )
    with pytest.raises(ValueError, match="expanded JSONL exceeds"):
        read_jsonl(compressed, max_expanded_bytes=100)


def test_bounded_text_reader_rejects_size_and_invalid_utf8(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.yaml"
    oversized.write_text("value: too-large\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text file exceeds"):
        read_text_bounded(oversized, max_bytes=4)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_text_bounded(invalid, max_bytes=4)


def test_durable_jsonl_append_enforces_file_and_line_limits(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    first_size = append_jsonl_record(path, {"sequence": 1}, max_file_bytes=100)

    assert first_size == path.stat().st_size
    assert read_jsonl(path) == [{"sequence": 1}]
    with pytest.raises(ValueError, match="line exceeds"):
        append_jsonl_record(
            path,
            {"value": "x" * 100},
            max_file_bytes=1_000,
            max_line_bytes=16,
        )
    with pytest.raises(ValueError, match="file exceeds"):
        append_jsonl_record(path, {"sequence": 2}, max_file_bytes=first_size)
