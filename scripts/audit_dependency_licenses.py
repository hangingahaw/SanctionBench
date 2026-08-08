#!/usr/bin/env python3
"""Verify the pinned public environment and retain its dependency-license metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

LOCK_PATH = Path("requirements.lock")
REPORT_PATH = Path("THIRD_PARTY_LICENSES.json")
FORBIDDEN_LICENSE_MARKERS = ("AGPL", "Affero", "GPL-", "LGPL-")
LICENSE_PATTERNS = {
    "Apache-2.0": re.compile(r"\bApache(?: License)?(?:,? Version)?[ -]?2\.0\b", re.I),
    "BSD-2-Clause": re.compile(r"\bBSD-2-Clause\b", re.I),
    "BSD-3-Clause": re.compile(r"\bBSD-3-Clause\b", re.I),
    "BSD": re.compile(r"\bBSD(?: License)?\b", re.I),
    "CNRI-Python": re.compile(r"\bCNRI-Python\b", re.I),
    "ISC": re.compile(r"\bISC(?: License)?\b", re.I),
    "MIT-0": re.compile(r"\bMIT-0\b", re.I),
    "MIT": re.compile(r"\bMIT(?: License)?\b", re.I),
    "MPL-2.0": re.compile(r"\b(?:MPL-2\.0|Mozilla Public License 2\.0)\b", re.I),
    "PSF-2.0": re.compile(r"\bPSF-2\.0\b", re.I),
    "Public-Domain": re.compile(r"\bPublic[- ]Domain\b", re.I),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_packages() -> list[tuple[str, str]]:
    packages: list[tuple[str, str]] = []
    for raw_line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(f"Unsupported lockfile requirement: {line}")
        name, version = line.split("==", maxsplit=1)
        packages.append((name, version))
    return packages


def license_metadata(distribution: importlib.metadata.Distribution) -> str:
    metadata = distribution.metadata
    values = [metadata.get("License-Expression") or "", metadata.get("License") or ""]
    values.extend(
        value.removeprefix("License :: ")
        for value in metadata.get_all("Classifier", [])
        if value.startswith("License :: ")
    )
    return " | ".join(value.strip() for value in values if value.strip())


def build_report() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name, expected_version in locked_packages():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected_version:
            raise ValueError(
                f"{name}: installed {distribution.version}, lock requires {expected_version}"
            )
        raw_license = license_metadata(distribution)
        if not raw_license:
            raise ValueError(f"{name}: package metadata does not declare a license")
        forbidden = [marker for marker in FORBIDDEN_LICENSE_MARKERS if marker in raw_license]
        if forbidden:
            raise ValueError(f"{name}: public dependency has a forbidden license: {raw_license}")
        recognized = sorted(
            license_id
            for license_id, pattern in LICENSE_PATTERNS.items()
            if pattern.search(raw_license)
        )
        if not recognized:
            raise ValueError(f"{name}: unreviewed license metadata: {raw_license}")
        records.append(
            {
                "name": distribution.metadata.get("Name") or name,
                "version": distribution.version,
                "license_metadata": raw_license,
                "recognized_license_ids": recognized,
                "project_url": (
                    distribution.metadata.get("Home-page")
                    or next(
                        iter(distribution.metadata.get_all("Project-URL", [])),
                        "UNAVAILABLE",
                    )
                ),
            }
        )
    records.sort(key=lambda value: str(value["name"]).casefold())
    return {
        "schema_version": "sanctionbench.third_party_licenses.v1",
        "requirements_lock_sha256": sha256_file(LOCK_PATH),
        "package_count": len(records),
        "all_dependencies_reviewed": True,
        "forbidden_copyleft_dependency_present": False,
        "packages": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write:
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {REPORT_PATH} for {report['package_count']} packages")
        return
    observed = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if observed != report:
        raise SystemExit("dependency-license report differs from installed locked environment")
    print(f"verified {report['package_count']} dependency licenses")


if __name__ == "__main__":
    main()
