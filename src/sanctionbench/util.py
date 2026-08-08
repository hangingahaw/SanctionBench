"""Small deterministic I/O helpers shared by pipeline stages."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_JSONL_COMPRESSED_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_JSONL_EXPANDED_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_JSONL_ROWS = 100_000
DEFAULT_MAX_TEXT_BYTES = 1024 * 1024
DEFAULT_MAX_MANIFEST_BYTES = 1024 * 1024


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with second precision."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(
    path: Path,
    value: Any,
    *,
    indent: int | None = 2,
    max_bytes: int | None = None,
) -> None:
    """Atomically write deterministic UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded: bytes | None = None
    if max_bytes is not None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        encoded = (
            json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError(f"Serialized JSON exceeds the {max_bytes}-byte limit")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        if encoded is not None:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=indent, sort_keys=True)
                handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        if path.suffix == ".gz":
            with (
                os.fdopen(fd, "wb") as raw_handle,
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_handle,
                    mtime=0,
                ) as compressed_handle,
                io.TextIOWrapper(compressed_handle, encoding="utf-8") as handle,
            ):
                _write_jsonl_rows(handle, rows)
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                _write_jsonl_rows(handle, rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def append_jsonl_record(
    path: Path,
    row: Any,
    *,
    max_file_bytes: int,
    max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
) -> int:
    """Durably append one bounded JSON object and return its encoded byte count."""

    if min(max_file_bytes, max_line_bytes) < 1:
        raise ValueError("JSONL append limits must be positive")
    if path.suffix == ".gz":
        raise ValueError("Durable JSONL append does not support gzip files")
    if hasattr(row, "model_dump"):
        row = row.model_dump(mode="json")
    encoded = (canonical_json(row) + "\n").encode("utf-8")
    if len(encoded) > max_line_bytes:
        raise ValueError(f"{path}: JSONL line exceeds the {max_line_bytes}-byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_size = path.stat().st_size if path.exists() else 0
    if existing_size + len(encoded) > max_file_bytes:
        raise ValueError(f"{path}: JSONL file exceeds the {max_file_bytes}-byte limit")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("durable JSONL append made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return len(encoded)


def _write_jsonl_rows(handle: Any, rows: Iterable[Any]) -> None:
    for row in rows:
        if hasattr(row, "model_dump"):
            row = row.model_dump(mode="json")
        handle.write(canonical_json(row))
        handle.write("\n")


def read_json(path: Path, *, max_bytes: int = DEFAULT_MAX_JSON_BYTES) -> Any:
    """Read one UTF-8 JSON file without allowing an unbounded allocation."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path}: JSON file exceeds the {max_bytes}-byte limit")
    with path.open("rb") as handle:
        encoded = handle.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise ValueError(f"{path}: JSON file exceeds the {max_bytes}-byte limit")
    try:
        return json.loads(encoded.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: JSON file is not valid UTF-8") from error


def read_text_bounded(path: Path, *, max_bytes: int = DEFAULT_MAX_TEXT_BYTES) -> str:
    """Read one UTF-8 text file without allowing an unbounded allocation."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path}: text file exceeds the {max_bytes}-byte limit")
    with path.open("rb") as handle:
        encoded = handle.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise ValueError(f"{path}: text file exceeds the {max_bytes}-byte limit")
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: text file is not valid UTF-8") from error


def read_jsonl(
    path: Path,
    *,
    max_compressed_bytes: int = DEFAULT_MAX_JSONL_COMPRESSED_BYTES,
    max_expanded_bytes: int = DEFAULT_MAX_JSONL_EXPANDED_BYTES,
    max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
    max_rows: int = DEFAULT_MAX_JSONL_ROWS,
) -> list[dict[str, Any]]:
    """Read bounded JSONL, including an explicit gzip expansion budget."""

    if min(max_compressed_bytes, max_expanded_bytes, max_line_bytes, max_rows) < 1:
        raise ValueError("JSONL read limits must be positive")
    if path.stat().st_size > max_compressed_bytes:
        raise ValueError(
            f"{path}: JSONL file exceeds the {max_compressed_bytes}-byte compressed limit"
        )
    rows: list[dict[str, Any]] = []
    expanded_bytes = 0
    with gzip.open(path, mode="rb") if path.suffix == ".gz" else path.open("rb") as handle:
        line_number = 0
        while True:
            # Enforce the line ceiling during the read. A size-less iterator can
            # otherwise materialize one huge expanded gzip line before rejection.
            encoded_line = handle.readline(max_line_bytes + 1)
            if not encoded_line:
                break
            line_number += 1
            expanded_bytes += len(encoded_line)
            if len(encoded_line) > max_line_bytes:
                raise ValueError(
                    f"{path}:{line_number}: JSONL line exceeds the {max_line_bytes}-byte limit"
                )
            if expanded_bytes > max_expanded_bytes:
                raise ValueError(
                    f"{path}: expanded JSONL exceeds the {max_expanded_bytes}-byte limit"
                )
            if not encoded_line.strip():
                continue
            if len(rows) >= max_rows:
                raise ValueError(f"{path}: JSONL exceeds the {max_rows}-row limit")
            try:
                line = encoded_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSONL line is not valid UTF-8") from error
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def project_root() -> Path:
    configured = os.environ.get("SANCTIONBENCH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("Could not locate SanctionBench project root")
