#!/usr/bin/env python3
"""Verify or refresh the content-hashed public export manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("PUBLIC_EXPORT_MANIFEST.json")
IGNORED_FILES = {".coverage"}
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
FORBIDDEN_DIRECTORIES = {
    ".claude",
    ".codex",
    ".gstack",
}
ALLOWED_EXTENSIONLESS_FILES = {
    ".gitignore",
    ".gitkeep",
    "LICENSE",
    "Makefile",
}
FORBIDDEN_PREFIXES = (
    "data/raw/",
    "data/cache/",
    "data/interim/",
    "data/curation/",
    "data/private/",
)
FORBIDDEN_BASENAMES = {
    "REPORT.md",
    "data-inventory.json",
    "data-source-licenses.json",
    "pacer-purchase-queue.json",
    "acquisition.py",
    "extraction.py",
    "gold.py",
    "population_filings.py",
    "targetless_discovery.py",
    "cli.py",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:OPENAI|ANTHROPIC|GOOGLE|DEEPSEEK|COURTLISTENER)_API_(?:KEY|TOKEN)=\S+"),
    re.compile(r"\bAWS_SECRET_ACCESS_KEY=\S+"),
)
EXPECTED_MANIFEST_SCHEMA = "sanctionbench.public_export.v1"
ALLOWED_PUBLIC_RELEASE_STATUSES = {"development_public_gold"}
MAX_PUBLIC_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_PUBLIC_FILE_BYTES = 64 * 1024 * 1024
MAX_PUBLIC_PAYLOAD_BYTES = 512 * 1024 * 1024
EXPECTED_MANIFEST_KEYS = {
    "created_at",
    "excluded_categories",
    "file_count_excluding_manifest",
    "files",
    "fresh_history_required",
    "official_private_holdout_included",
    "public_release_status",
    "schema_version",
    "source_internal_commit",
    "source_internal_dirty",
    "warning",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_tracked_paths(root: Path) -> set[str]:
    """Return tracked paths only when root itself is a Git worktree."""

    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root.resolve():
        return set()
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return {path.decode("utf-8") for path in result.stdout.split(b"\0") if path}


def _scan_public_file(
    path: Path, *, relative_text: str, max_bytes: int = MAX_PUBLIC_FILE_BYTES
) -> None:
    """Scan every textual file and reject unclassified binary payloads."""

    if path.stat().st_size > max_bytes:
        raise SystemExit(f"public release file exceeds the byte limit: {relative_text}")
    try:
        with path.open("rb") as handle:
            encoded = handle.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            raise SystemExit(f"public release file exceeds the byte limit: {relative_text}")
        contents = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"unclassified binary file in public release: {relative_text}") from error
    if any(pattern.search(contents) for pattern in SECRET_PATTERNS):
        raise SystemExit(f"possible secret in public release: {relative_text}")


def _validate_manifest_policy(
    manifest: dict[str, Any], *, allow_dirty_source_candidate: bool
) -> None:
    """Enforce the publication metadata, not just the payload hashes."""

    commit = manifest.get("source_internal_commit")
    if (
        set(manifest) != EXPECTED_MANIFEST_KEYS
        or manifest.get("schema_version") != EXPECTED_MANIFEST_SCHEMA
        or manifest.get("fresh_history_required") is not True
        or manifest.get("official_private_holdout_included") is not False
        or manifest.get("public_release_status") not in ALLOWED_PUBLIC_RELEASE_STATUSES
        or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None
    ):
        raise SystemExit("public manifest release-policy metadata is invalid")
    if manifest.get("source_internal_dirty") is not False and not allow_dirty_source_candidate:
        raise SystemExit("publishable public manifest requires source_internal_dirty=false")


def release_payload() -> list[dict[str, Any]]:
    root = Path.cwd()
    tracked_paths = _git_tracked_paths(root)
    for tracked_text in sorted(tracked_paths):
        tracked = Path(tracked_text)
        if (
            tracked.name in IGNORED_FILES
            or IGNORED_DIRECTORIES.intersection(tracked.parts)
            or any(part.endswith(".egg-info") for part in tracked.parts)
        ):
            raise SystemExit(f"tracked file is hidden by a runtime/build exclusion: {tracked_text}")
    paths: list[Path] = []
    payload_bytes = 0
    for absolute_path in sorted(root.rglob("*")):
        relative = absolute_path.relative_to(root)
        if FORBIDDEN_DIRECTORIES.intersection(relative.parts):
            raise SystemExit(f"runtime-control directory in public release: {relative.as_posix()}")
        ignored_by_runtime_rule = (
            absolute_path.name in IGNORED_FILES
            or IGNORED_DIRECTORIES.intersection(relative.parts)
            or any(part.endswith(".egg-info") for part in relative.parts)
        )
        if relative.parts[:1] == ("leaderboard",) and ignored_by_runtime_rule:
            raise SystemExit(
                "deployable leaderboard contains a path hidden from the release audit: "
                f"{relative.as_posix()}"
            )
        if ignored_by_runtime_rule:
            continue
        if absolute_path.is_symlink():
            raise SystemExit(f"public release contains a symlink: {relative.as_posix()}")
        if not absolute_path.is_file() or relative == MANIFEST_PATH:
            continue
        relative_text = relative.as_posix()
        payload_bytes += absolute_path.stat().st_size
        if payload_bytes > MAX_PUBLIC_PAYLOAD_BYTES:
            raise SystemExit("public release payload exceeds the aggregate byte limit")
        if not absolute_path.suffix and absolute_path.name not in ALLOWED_EXTENSIONLESS_FILES:
            raise SystemExit(f"unrecognized extensionless file in public release: {relative_text}")
        if (
            relative_text.startswith(FORBIDDEN_PREFIXES)
            or absolute_path.name in FORBIDDEN_BASENAMES
        ):
            raise SystemExit(f"forbidden internal artifact in public release: {relative_text}")
        if absolute_path.name == ".env":
            raise SystemExit("populated .env must never enter the public release")
        _scan_public_file(absolute_path, relative_text=relative_text)
        paths.append(relative)
    records = [
        {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    observed_paths = {str(record["path"]) for record in records}
    hidden_tracked = tracked_paths - observed_paths - {MANIFEST_PATH.as_posix()}
    if hidden_tracked:
        raise SystemExit(
            "tracked public files are absent from the manifest payload: "
            + ", ".join(sorted(hidden_tracked))
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--allow-dirty-source-candidate",
        action="store_true",
        help="Validate or refresh a non-publishable development candidate without weakening audit-release.",
    )
    args = parser.parse_args()

    _scan_public_file(
        MANIFEST_PATH,
        relative_text=MANIFEST_PATH.as_posix(),
        max_bytes=MAX_PUBLIC_MANIFEST_BYTES,
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = release_payload()
    if args.write:
        expected_paths = {str(record["path"]) for record in manifest.get("files", [])}
        observed_paths = {str(record["path"]) for record in records}
        if expected_paths != observed_paths:
            raise SystemExit(
                "refusing to refresh a manifest whose release payload file set changed"
            )
        _validate_manifest_policy(
            manifest,
            allow_dirty_source_candidate=args.allow_dirty_source_candidate,
        )
        manifest["file_count_excluding_manifest"] = len(records)
        manifest["files"] = records
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"refreshed {MANIFEST_PATH} with {len(records)} release payload files")
        return

    if manifest.get("file_count_excluding_manifest") != len(records):
        raise SystemExit("public manifest file count does not match release payload")
    if manifest.get("files") != records:
        raise SystemExit("public manifest paths, sizes, or hashes do not match release payload")
    _validate_manifest_policy(
        manifest,
        allow_dirty_source_candidate=args.allow_dirty_source_candidate,
    )
    print(f"verified {len(records)} release payload files")


if __name__ == "__main__":
    main()
