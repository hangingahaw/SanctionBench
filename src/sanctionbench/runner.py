"""Checkpointed benchmark runner for closed-book and tool-assisted conditions."""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from collections.abc import Callable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from tenacity import Retrying, stop_after_attempt, wait_exponential

from .courtlistener import MAX_HTTP_ATTEMPTS_PER_SEARCH, CourtListenerClient
from .document_audit import load_document_scenarios, score_document_predictions
from .grading import score_predictions
from .models import (
    CitationItem,
    Condition,
    DocumentPrediction,
    DocumentScenario,
    OrganicDocumentGold,
    OrganicDocumentPrediction,
    Prediction,
    RunRecord,
    TemporalCutoff,
)
from .organic_document_audit import (
    load_organic_document_gold,
    score_organic_document_predictions,
)
from .providers.base import (
    DOCUMENT_PREDICTION_SCHEMA,
    DOCUMENT_SYSTEM_PROMPT,
    MAX_PROVIDER_REQUEST_INPUT_BYTES,
    ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
    ORGANIC_DOCUMENT_SYSTEM_PROMPT,
    PREDICTION_SCHEMA,
    SYSTEM_PROMPT,
    Provider,
    build_document_user_prompt,
    build_organic_document_user_prompt,
    build_user_prompt,
    provider_request_input_bytes,
    validate_provider_request_input,
)
from .providers.factory import create_provider
from .util import (
    DEFAULT_MAX_MANIFEST_BYTES,
    append_jsonl_record,
    canonical_json,
    project_root,
    read_json,
    read_jsonl,
    read_text_bounded,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)

RUN_IDENTITY_VERSION = "sanctionbench.run_identity.v6"
FINALIZED_RUN_IDENTITY_VERSION = "sanctionbench.finalized_run_identity.v2"
TOOL_SPEC_VERSION = "courtlistener_v4_citation_lookup.v6"
MAX_PROVIDER_ATTEMPTS_PER_ITEM = 3
MAX_PROVIDER_CONFIGS = 16
MAX_CAMPAIGN_REPETITIONS = 64
MAX_PLANNED_SUBRUNS = 256
MAX_SELECTED_RECORDS_PER_RUN = 10_000
MAX_TOOL_EVIDENCE_ITEM_BYTES = 1024 * 1024
MAX_TOOL_EVIDENCE_PROMPT_RESERVATION_BYTES = MAX_TOOL_EVIDENCE_ITEM_BYTES + 256 * 1024
MAX_TOOL_EVIDENCE_LEDGER_BYTES = 256 * 1024 * 1024
MAX_REQUEST_LEDGER_BYTES = 16 * 1024 * 1024
MAX_CAMPAIGN_PROVIDER_INPUT_BYTES = 8 * 1024 * 1024 * 1024
MAX_CAMPAIGN_CONFIG_BYTES = 256 * 1024
MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES = 256 * 1024 * 1024
MAX_CHECKPOINT_WRITE_JOURNAL_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_WRITE_EVENTS = 200_000
MAX_CHECKPOINT_WRITE_EVENT_BYTES = 2 * 1024
RUNNER_MAX_COURTLISTENER_RESPONSE_BYTES = 256 * 1024
MAX_COURTLISTENER_ATTEMPTS_PER_LOOKUP = 2 * MAX_HTTP_ATTEMPTS_PER_SEARCH
MAX_COURTLISTENER_LOOKUPS_PER_RUN = 512
RUN_IDENTITY_SOURCE_FILES = (
    "courtlistener.py",
    "document_audit.py",
    "grading.py",
    "models.py",
    "organic_document_audit.py",
    "runner.py",
    "util.py",
    "providers/__init__.py",
    "providers/anthropic_provider.py",
    "providers/base.py",
    "providers/deepseek_provider.py",
    "providers/factory.py",
    "providers/google_provider.py",
    "providers/mock.py",
    "providers/openai_provider.py",
)
MODEL_ENV_BY_PROVIDER = {
    "anthropic": "SANCTIONBENCH_ANTHROPIC_MODEL",
    "deepseek": "SANCTIONBENCH_DEEPSEEK_MODEL",
    "google": "SANCTIONBENCH_GOOGLE_MODEL",
    "openai": "SANCTIONBENCH_OPENAI_MODEL",
}
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,199}$")
SECRET_SHAPED_MODEL_PREFIXES = ("sk-", "AIza", "AKIA", "ghp_", "gho_", "ghu_", "ghs_", "ghr_")
PATH_ENV_BY_FIELD = {
    "dataset": {
        "SANCTIONBENCH_ORGANIC_DOCUMENT_DATASET",
        "SANCTIONBENCH_PRIVATE_CITATION_DATASET",
    },
    "document_dataset": {"SANCTIONBENCH_PRIVATE_DOCUMENT_DATASET"},
    "output_dir": {"SANCTIONBENCH_PRIVATE_RESULTS_DIR"},
}
PATH_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


def _path_reference(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _configured_path(root: Path, value: str, *, field: str) -> Path:
    """Resolve one manifest path without exposing arbitrary process environment values."""

    match = PATH_ENV_REFERENCE.fullmatch(value)
    if "$" in value:
        variable = match.group(1) if match else None
        if variable is None or variable not in PATH_ENV_BY_FIELD.get(field, set()):
            raise ValueError(f"{field} uses an unsupported path environment reference")
        expanded = os.environ.get(variable)
        if not expanded:
            raise RuntimeError(f"{field} requires the configured {variable} path")
        return Path(expanded).expanduser().resolve()

    path = Path(value)
    if path.is_absolute() or value.startswith("~"):
        raise ValueError(
            f"{field} literal paths must be relative to the project root; use a documented "
            "SANCTIONBENCH path variable for an external location"
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{field} literal path escapes the project root") from error
    return resolved


def _resolved_provider_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve only the documented provider-specific model environment variables."""

    resolved: list[dict[str, Any]] = []
    providers = config.get("providers")
    if not isinstance(providers, list):
        raise ValueError("providers must be a list")
    if not providers:
        raise ValueError("providers must contain at least one provider")
    if len(providers) > MAX_PROVIDER_CONFIGS:
        raise ValueError(f"providers cannot contain more than {MAX_PROVIDER_CONFIGS} entries")
    seen: set[tuple[str, str]] = set()
    for provider_config in providers:
        if not isinstance(provider_config, dict):
            raise ValueError("each provider configuration must be a mapping")
        if not set(provider_config).issubset({"provider", "model", "model_env"}):
            raise ValueError("provider configurations contain unsupported fields")
        provider_name = str(provider_config.get("provider", ""))
        if provider_name not in {*MODEL_ENV_BY_PROVIDER, "mock"}:
            raise ValueError(f"unsupported provider: {provider_name!r}")
        model = provider_config.get("model")
        model_env = provider_config.get("model_env")
        if model is None and model_env is not None:
            expected_model_env = MODEL_ENV_BY_PROVIDER.get(provider_name)
            if model_env != expected_model_env:
                raise ValueError(
                    f"model_env for provider {provider_name!r} must be "
                    f"{expected_model_env or 'omitted'}"
                )
            model = os.environ.get(str(model_env))
        if not isinstance(model, str) or not model:
            expected_model_env = MODEL_ENV_BY_PROVIDER.get(provider_name)
            raise RuntimeError(
                f"Model is unset for provider {provider_name}; "
                f"set {expected_model_env or 'the model field'}"
            )
        if MODEL_ID_PATTERN.fullmatch(model) is None or model.startswith(
            SECRET_SHAPED_MODEL_PREFIXES
        ):
            raise ValueError(
                f"Model identifier for provider {provider_name!r} contains unsupported characters "
                "or exceeds 200 characters"
            )
        identity = (provider_name, model)
        if identity in seen:
            raise ValueError(f"duplicate provider/model configuration: {provider_name}/{model}")
        seen.add(identity)
        resolved.append({**provider_config, "model": model})
    return resolved


def _resolved_conditions(config: dict[str, Any], *, organic: bool) -> list[Condition]:
    raw_conditions = config.get("conditions", ["closed_book"] if organic else None)
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("conditions must contain at least one condition")
    conditions = [Condition(value) for value in raw_conditions]
    if len(conditions) != len(set(conditions)):
        raise ValueError("conditions must be duplicate-free")
    if organic and any(condition != Condition.CLOSED_BOOK for condition in conditions):
        raise ValueError("Organic document v1 permits only the closed_book condition")
    return conditions


def _planned_v1_input_bytes_per_provider(
    *,
    items: list[CitationItem],
    scenarios: list[DocumentScenario],
    conditions: list[Condition],
) -> int:
    """Return a conservative per-provider input-byte plan before client creation."""

    total = 0
    for condition in conditions:
        tool_reservation = (
            MAX_TOOL_EVIDENCE_PROMPT_RESERVATION_BYTES
            if condition == Condition.TOOL_ASSISTED
            else 0
        )
        for item in items:
            size = (
                provider_request_input_bytes(
                    "citation_verification", build_user_prompt(item, condition, None)
                )
                + tool_reservation
            )
            if size > MAX_PROVIDER_REQUEST_INPUT_BYTES:
                raise ValueError(
                    f"citation_verification planned input exceeds the per-request "
                    f"{MAX_PROVIDER_REQUEST_INPUT_BYTES}-byte limit"
                )
            total += size
        for scenario in scenarios:
            size = (
                provider_request_input_bytes(
                    "document_audit", build_document_user_prompt(scenario, condition, None)
                )
                + tool_reservation
            )
            if size > MAX_PROVIDER_REQUEST_INPUT_BYTES:
                raise ValueError(
                    f"document_audit planned input exceeds the per-request "
                    f"{MAX_PROVIDER_REQUEST_INPUT_BYTES}-byte limit"
                )
            total += size
    return total


def _planned_organic_input_bytes_per_provider(
    *, documents: list[OrganicDocumentGold], conditions: list[Condition], repetitions: int
) -> int:
    total = 0
    for condition in conditions:
        for document in documents:
            size = provider_request_input_bytes(
                "organic_document_audit",
                build_organic_document_user_prompt(document.model_input(), condition),
            )
            if size > MAX_PROVIDER_REQUEST_INPUT_BYTES:
                raise ValueError(
                    f"organic_document_audit planned input exceeds the per-request "
                    f"{MAX_PROVIDER_REQUEST_INPUT_BYTES}-byte limit"
                )
            total += size
    return total * repetitions


def _require_provider_input_budget(
    *,
    resolved_providers: list[dict[str, Any]],
    planned_input_bytes_per_provider: int,
    max_provider_input_bytes: int | None,
) -> int:
    """Require a retry-inclusive paid-input approval and a fixed local safety ceiling."""

    live_provider_count = sum(
        str(provider["provider"]) != "mock" for provider in resolved_providers
    )
    retry_inclusive_input_ceiling = (
        planned_input_bytes_per_provider * live_provider_count * MAX_PROVIDER_ATTEMPTS_PER_ITEM
    )
    if retry_inclusive_input_ceiling > MAX_CAMPAIGN_PROVIDER_INPUT_BYTES:
        raise ValueError(
            "retry-inclusive provider input plan exceeds the fixed "
            f"{MAX_CAMPAIGN_PROVIDER_INPUT_BYTES}-byte campaign safety ceiling"
        )
    if max_provider_input_bytes is not None and max_provider_input_bytes < 1:
        raise ValueError("max_provider_input_bytes must be positive when provided")
    if retry_inclusive_input_ceiling and max_provider_input_bytes is None:
        raise ValueError(
            "Live campaigns require an independent max_provider_input_bytes approval before "
            "any provider client is created"
        )
    if (
        max_provider_input_bytes is not None
        and max_provider_input_bytes < retry_inclusive_input_ceiling
    ):
        raise ValueError(
            "max_provider_input_bytes is below the retry-inclusive campaign plan: "
            f"requires at least {retry_inclusive_input_ceiling}"
        )
    return retry_inclusive_input_ceiling


def _require_courtlistener_request_budget(
    *,
    resolved_providers: list[dict[str, Any]],
    conditions: list[Condition],
    lookup_count_per_tool_run: int,
    max_courtlistener_requests: int | None,
) -> int:
    tool_run_count = len(resolved_providers) * sum(
        condition == Condition.TOOL_ASSISTED for condition in conditions
    )
    retry_inclusive_ceiling = (
        lookup_count_per_tool_run * tool_run_count * MAX_COURTLISTENER_ATTEMPTS_PER_LOOKUP
    )
    if max_courtlistener_requests is not None and max_courtlistener_requests < 1:
        raise ValueError("max_courtlistener_requests must be positive when provided")
    if retry_inclusive_ceiling and max_courtlistener_requests is None:
        raise ValueError(
            "Tool-assisted campaigns require an independent max_courtlistener_requests "
            "approval before any CourtListener or provider client is created"
        )
    if (
        max_courtlistener_requests is not None
        and max_courtlistener_requests < retry_inclusive_ceiling
    ):
        raise ValueError(
            "max_courtlistener_requests is below the retry-inclusive tool plan: "
            f"requires at least {retry_inclusive_ceiling}"
        )
    return retry_inclusive_ceiling


def _require_planned_subruns(count: int) -> None:
    if count < 1 or count > MAX_PLANNED_SUBRUNS:
        raise ValueError(
            f"campaign plans {count} sub-runs; the allowed range is 1-{MAX_PLANNED_SUBRUNS}"
        )


def _require_live_request_budget(
    *,
    resolved_providers: list[dict[str, Any]],
    logical_calls_per_provider: int,
    max_provider_requests: int | None,
) -> tuple[int, int]:
    """Require an independent retry-inclusive ceiling before any live provider work."""

    if logical_calls_per_provider < 0:
        raise ValueError("logical call count cannot be negative")
    live_provider_count = sum(
        str(provider["provider"]) != "mock" for provider in resolved_providers
    )
    planned_live_calls = logical_calls_per_provider * live_provider_count
    retry_inclusive_maximum = planned_live_calls * MAX_PROVIDER_ATTEMPTS_PER_ITEM
    if max_provider_requests is not None and max_provider_requests < 1:
        raise ValueError("max_provider_requests must be positive when provided")
    if planned_live_calls and max_provider_requests is None:
        raise ValueError(
            "Live campaigns require an independent max_provider_requests approval before any "
            "provider client is created"
        )
    if max_provider_requests is not None and max_provider_requests < retry_inclusive_maximum:
        raise ValueError(
            "max_provider_requests is below the retry-inclusive campaign plan: "
            f"requires at least {retry_inclusive_maximum}"
        )
    return planned_live_calls, retry_inclusive_maximum


def _validate_result_index_destination(
    output_root: Path,
    *,
    protected_inputs: list[Path],
) -> Path:
    """Reject config-selected output aliases before any provider work begins."""

    index_path = (output_root / "index.json").resolve()
    protected = {path.resolve() for path in protected_inputs}
    if index_path in protected:
        raise ValueError("output_dir/index.json must not overwrite a benchmark input")
    if index_path.exists():
        raise FileExistsError(
            f"Result index already exists and is immutable: {index_path}; choose a fresh output_dir"
        )
    return index_path


def _v1_release_metadata(
    root: Path, *, citation_dataset_sha256: str, document_dataset_sha256: str | None
) -> tuple[str, list[str], list[str]]:
    """Classify only exact manifest-bound v1 data as public development gold."""

    manifest_path = root / "data/gold/v1/manifest.json"
    is_public = False
    if manifest_path.is_file():
        manifest = read_json(manifest_path, max_bytes=DEFAULT_MAX_MANIFEST_BYTES)
        if not isinstance(manifest, dict):
            raise ValueError("Public gold manifest must be a JSON object")
        is_public = (
            manifest.get("release_status") == "development_public_gold"
            and manifest.get("citation_items_sha256") == citation_dataset_sha256
            and (
                document_dataset_sha256 is None
                or manifest.get("document_scenarios_sha256") == document_dataset_sha256
            )
        )
    if is_public:
        return "development_public_gold", ["public_development_gold"], ["cleared_public"]
    return (
        "provisional_private_evaluation",
        ["unclassified_external"],
        ["private_evaluation_only"],
    )


def _select_organic_documents(
    documents: list[OrganicDocumentGold], *, maximum: int, seed: int
) -> list[OrganicDocumentGold]:
    if maximum < 1:
        raise ValueError("max_documents must be at least one")
    if maximum >= len(documents):
        return documents
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    clean = [document for document in shuffled if document.document_kind == "clean_control"]
    offending = [document for document in shuffled if document.document_kind == "offending"]
    if clean and offending:
        if maximum < 2:
            raise ValueError(
                "max_documents must be at least two to preserve one offending document and one "
                "clean control"
            )
        required_ids = {clean[0].item_id, offending[0].item_id}
        selected = [clean[0], offending[0]]
        selected.extend(document for document in shuffled if document.item_id not in required_ids)
        return sorted(selected[:maximum], key=lambda item: item.item_id)
    return sorted(shuffled[:maximum], key=lambda item: item.item_id)


def _slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(filter(None, normalized.split("-")))


def _git_state(root: Path) -> dict[str, Any]:
    """Capture the commit and worktree state at run-manifest invocation."""

    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        return {
            "commit": "UNAVAILABLE",
            "dirty": None,
            "status_entry_count": None,
            "status_sha256": None,
        }
    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status_result.returncode != 0:
        return {
            "commit": commit_result.stdout.strip(),
            "dirty": None,
            "status_entry_count": None,
            "status_sha256": None,
        }
    status_lines = [line for line in status_result.stdout.splitlines() if line]
    return {
        "commit": commit_result.stdout.strip(),
        "dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "status_sha256": sha256_bytes(status_result.stdout.encode()),
    }


def _records_sha256(records: list[Any]) -> str:
    payload = [
        record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        for record in records
    ]
    return sha256_bytes(canonical_json(payload).encode())


@lru_cache(maxsize=1)
def _runtime_source_sha256() -> str:
    """Hash the executable package source so dirty-tree changes invalidate resume state."""

    package_root = Path(__file__).resolve().parent
    manifest = []
    for relative in RUN_IDENTITY_SOURCE_FILES:
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(f"Run-identity source file is missing: {path}")
        manifest.append({"path": relative, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json(manifest).encode())


def _prompt_sha256(task_type: str) -> str:
    if task_type == "citation_verification":
        payload = {
            "system": SYSTEM_PROMPT,
            "schema": PREDICTION_SCHEMA,
            "user_prompt_builder_source": inspect.getsource(build_user_prompt),
        }
    elif task_type == "document_audit":
        payload = {
            "system": DOCUMENT_SYSTEM_PROMPT,
            "schema": DOCUMENT_PREDICTION_SCHEMA,
            "user_prompt_builder_source": inspect.getsource(build_document_user_prompt),
        }
    elif task_type == "organic_document_audit":
        payload = {
            "system": ORGANIC_DOCUMENT_SYSTEM_PROMPT,
            "schema": ORGANIC_DOCUMENT_PREDICTION_SCHEMA,
            "user_prompt_builder_source": inspect.getsource(build_organic_document_user_prompt),
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return sha256_bytes(canonical_json(payload).encode())


def _run_identity(
    *,
    task_type: str,
    dataset_sha256: str,
    subset_sha256: str,
    selected_ids: list[str],
    config_sha256: str,
    provider_name: str,
    provider_protocol_version: str,
    provider_runtime_identity: dict[str, str],
    runtime_source_sha256: str,
    model: str,
    condition: Condition,
    seed: int,
    prompt_sha256: str,
    temporal_cutoff: TemporalCutoff | None,
    resource_budget: dict[str, int | None],
    reference_dataset_sha256: str | None = None,
    repetition: int | None = None,
) -> tuple[dict[str, Any], str]:
    identity: dict[str, Any] = {
        "schema_version": RUN_IDENTITY_VERSION,
        "task_type": task_type,
        "dataset_sha256": dataset_sha256,
        "subset_sha256": subset_sha256,
        "selected_ids": selected_ids,
        "config_sha256": config_sha256,
        "provider": provider_name,
        "provider_protocol_version": provider_protocol_version,
        "provider_runtime_identity": provider_runtime_identity,
        "runtime_source_sha256": runtime_source_sha256,
        "model": model,
        "condition": condition.value,
        "seed": seed,
        "prompt_and_output_schema_sha256": prompt_sha256,
        "temporal_cutoff": (temporal_cutoff.model_dump(mode="json") if temporal_cutoff else None),
        "resource_budget": resource_budget,
        "tool_spec_version": (TOOL_SPEC_VERSION if condition == Condition.TOOL_ASSISTED else None),
        "reference_dataset_sha256": reference_dataset_sha256,
    }
    if repetition is not None:
        identity["repetition"] = repetition
    return identity, sha256_bytes(canonical_json(identity).encode())


def _finalize_run_record(
    *,
    record: RunRecord,
    benchmark_release_status: str,
    attempts_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    tool_evidence_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    """Bind the completed run and all publication-critical receipts into one digest."""

    record_payload = record.model_dump(mode="json")
    material = {
        "schema_version": FINALIZED_RUN_IDENTITY_VERSION,
        "benchmark_release_status": benchmark_release_status,
        "run_identity_sha256": str(record.metadata["run_identity_sha256"]),
        "provider_request_count": int(record.metadata["provider_request_count"]),
        "successful_response_count": int(record.metadata["successful_response_count"]),
        "provider_request_attempts_sha256": sha256_file(attempts_path),
        "courtlistener_request_attempts_sha256": record.metadata.get(
            "courtlistener_request_attempts_sha256"
        ),
        "tool_evidence_sha256": tool_evidence_sha256,
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "run_record_core_sha256": sha256_bytes(canonical_json(record_payload).encode()),
    }
    finalized_sha256 = sha256_bytes(canonical_json(material).encode())
    record_payload["metadata"]["finalized_run_identity"] = material
    record_payload["metadata"]["finalized_run_identity_sha256"] = finalized_sha256
    return record_payload, finalized_sha256


def _verify_existing_finalized_run(run_path: Path, *, identity_sha256: str) -> None:
    """Verify a completed run in place, then keep it immutable."""

    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Existing finalized run is unreadable: {run_path}") from error
    metadata = run.get("metadata") if isinstance(run, dict) else None
    if not isinstance(metadata, dict) or metadata.get("run_identity_sha256") != identity_sha256:
        raise RuntimeError(f"Existing finalized run has the wrong run identity: {run_path}")
    finalized = metadata.get("finalized_run_identity")
    finalized_sha256 = metadata.get("finalized_run_identity_sha256")
    if not isinstance(finalized, dict) or not isinstance(finalized_sha256, str):
        raise RuntimeError(f"Existing run lacks a finalized identity: {run_path}")

    run_dir = run_path.parent
    attempts_path = run_dir / "request_attempts.json"
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    for artifact in (attempts_path, predictions_path, metrics_path):
        if not artifact.is_file():
            raise RuntimeError(f"Existing finalized run artifact is missing: {artifact}")
    tool_evidence_sha256 = finalized.get("tool_evidence_sha256")
    if tool_evidence_sha256 is not None:
        tool_evidence_path = run_dir / "tool_evidence.json"
        if (
            not isinstance(tool_evidence_sha256, str)
            or not tool_evidence_path.is_file()
            or sha256_file(tool_evidence_path) != tool_evidence_sha256
        ):
            raise RuntimeError(f"Existing finalized tool evidence does not reconcile: {run_path}")
    courtlistener_attempts_sha256 = finalized.get("courtlistener_request_attempts_sha256")
    if courtlistener_attempts_sha256 is not None:
        courtlistener_attempts_path = run_dir / "courtlistener_request_attempts.json"
        if (
            not isinstance(courtlistener_attempts_sha256, str)
            or not courtlistener_attempts_path.is_file()
            or sha256_file(courtlistener_attempts_path) != courtlistener_attempts_sha256
        ):
            raise RuntimeError(
                f"Existing finalized CourtListener request ledger does not reconcile: {run_path}"
            )

    core = json.loads(json.dumps(run))
    core_metadata = core.get("metadata") or {}
    core_metadata.pop("finalized_run_identity", None)
    core_metadata.pop("finalized_run_identity_sha256", None)
    expected = {
        "schema_version": FINALIZED_RUN_IDENTITY_VERSION,
        "benchmark_release_status": finalized.get("benchmark_release_status"),
        "run_identity_sha256": identity_sha256,
        "provider_request_count": int(metadata.get("provider_request_count", -1)),
        "successful_response_count": int(metadata.get("successful_response_count", -1)),
        "provider_request_attempts_sha256": sha256_file(attempts_path),
        "courtlistener_request_attempts_sha256": courtlistener_attempts_sha256,
        "tool_evidence_sha256": tool_evidence_sha256,
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "run_record_core_sha256": sha256_bytes(canonical_json(core).encode()),
    }
    expected_sha256 = sha256_bytes(canonical_json(expected).encode())
    if finalized != expected or finalized_sha256 != expected_sha256:
        raise RuntimeError(f"Existing finalized run identity does not reconcile: {run_path}")


def _prepare_run_directory(
    *,
    output_root: Path,
    run_name: str,
    identity: dict[str, Any],
    identity_sha256: str,
) -> tuple[str, Path, bool]:
    run_id = f"{run_name}--{identity_sha256[:16]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "identity.json"
    expected = {"run_identity_sha256": identity_sha256, "identity": identity}
    if identity_path.exists():
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                f"Refusing incompatible resume in {run_dir}: identity.json does not match"
            )
        run_path = run_dir / "run.json"
        if run_path.exists():
            _verify_existing_finalized_run(run_path, identity_sha256=identity_sha256)
            return run_id, run_dir, True
    else:
        checkpoint_names = {
            "checkpoint_writes.jsonl",
            "predictions.jsonl",
            "request_attempts.json",
            "courtlistener_request_attempts.json",
            "tool_evidence.json",
            "metrics.json",
            "run.json",
        }
        existing_checkpoints = checkpoint_names.intersection(
            path.name for path in run_dir.iterdir()
        )
        if existing_checkpoints:
            raise RuntimeError(f"Refusing unverified resume in {run_dir}: missing identity.json")
        write_json(identity_path, expected)
    return run_id, run_dir, False


def _summary_from_finalized_run(*, root: Path, run_path: Path, task_type: str) -> dict[str, Any]:
    """Return a hash-reconciled finalized run without rewriting it or calling its provider."""

    record = RunRecord.model_validate(json.loads(run_path.read_text(encoding="utf-8")))
    recorded_task_type = record.metadata.get("task_type", "citation_verification")
    if recorded_task_type != task_type:
        raise RuntimeError(
            f"Existing finalized run has task_type {recorded_task_type!r}, expected {task_type!r}"
        )
    metrics_path = run_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    finalized_identity_sha256 = record.metadata.get("finalized_run_identity_sha256")
    run_identity_sha256 = record.metadata.get("run_identity_sha256")
    provider_request_count = record.metadata.get("provider_request_count")
    courtlistener_request_count = record.metadata.get("courtlistener_request_count", 0)
    if not isinstance(finalized_identity_sha256, str):
        raise RuntimeError(f"Existing finalized run lacks its finalized identity: {run_path}")
    if not isinstance(run_identity_sha256, str):
        raise RuntimeError(f"Existing finalized run lacks its run identity: {run_path}")
    if not isinstance(provider_request_count, int) or isinstance(provider_request_count, bool):
        raise RuntimeError(f"Existing finalized run has an invalid request count: {run_path}")
    if not isinstance(courtlistener_request_count, int) or isinstance(
        courtlistener_request_count, bool
    ):
        raise RuntimeError(
            f"Existing finalized run has an invalid CourtListener request count: {run_path}"
        )

    summary: dict[str, Any] = {
        "run_id": record.run_id,
        "task_type": task_type,
        "mock": record.mock,
        "condition": record.condition.value,
        "provider_request_count": provider_request_count,
        "courtlistener_request_count": courtlistener_request_count,
        "run_identity_sha256": run_identity_sha256,
        "finalized_run_identity_sha256": finalized_identity_sha256,
        "run_file": _path_reference(root, run_path),
    }
    if task_type == "organic_document_audit":
        document_count = record.metadata.get("document_count")
        repetition = record.metadata.get("repetition")
        if not isinstance(document_count, int) or not isinstance(repetition, int):
            raise RuntimeError(f"Existing organic run has invalid summary metadata: {run_path}")
        summary.update(
            {
                "document_count": document_count,
                "model_call_count": provider_request_count,
                "repetition": repetition,
                "organic_document_sanction_score": metrics["headline"]["score"],
                "clean_audit_rate": metrics["headline"]["clean_audit_rate"],
                "clean_control_false_alarm_rate": metrics["clean_controls"]["false_alarm_rate"],
            }
        )
    else:
        summary.update(
            {
                "item_count": record.item_count,
                "sanction_score": metrics["headline"]["score"],
                "false_positive_rate": metrics["binary"]["false_positive_rate"],
            }
        )
    return summary


class _CheckpointWriteBudget:
    """Persist a cumulative ceiling for repeated crash-safe checkpoint writes."""

    def __init__(self, path: Path, *, identity_sha256: str) -> None:
        self.path = path
        self.identity_sha256 = identity_sha256
        self.total_bytes = 0
        self.event_count = 0
        if not path.is_file():
            return
        rows = read_jsonl(
            path,
            max_compressed_bytes=MAX_CHECKPOINT_WRITE_JOURNAL_BYTES,
            max_expanded_bytes=MAX_CHECKPOINT_WRITE_JOURNAL_BYTES,
            max_line_bytes=MAX_CHECKPOINT_WRITE_EVENT_BYTES,
            max_rows=MAX_CHECKPOINT_WRITE_EVENTS,
        )
        for sequence, event in enumerate(rows, start=1):
            payload_bytes = event.get("payload_bytes")
            journal_bytes = event.get("journal_bytes")
            cumulative_bytes = event.get("cumulative_bytes")
            if (
                event.get("schema_version") != "sanctionbench.checkpoint_write.v1"
                or event.get("run_identity_sha256") != identity_sha256
                or event.get("sequence") != sequence
                or not isinstance(event.get("category"), str)
                or len(event["category"]) > 80
                or not isinstance(payload_bytes, int)
                or isinstance(payload_bytes, bool)
                or payload_bytes < 1
                or not isinstance(journal_bytes, int)
                or isinstance(journal_bytes, bool)
                or journal_bytes < 1
                or not isinstance(cumulative_bytes, int)
                or isinstance(cumulative_bytes, bool)
                or cumulative_bytes != self.total_bytes + payload_bytes + journal_bytes
                or cumulative_bytes > MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES
            ):
                raise ValueError(f"{path}: invalid checkpoint-write budget journal")
            self.total_bytes = cumulative_bytes
            self.event_count = sequence

    def reserve(self, *, category: str, payload_bytes: int) -> None:
        """Durably reserve the exact next checkpoint payload before writing it."""

        if not category or len(category) > 80 or payload_bytes < 1:
            raise ValueError("Checkpoint write reservation is invalid")
        if self.event_count >= MAX_CHECKPOINT_WRITE_EVENTS:
            raise ValueError("Checkpoint write event ceiling is exhausted")
        sequence = self.event_count + 1
        journal_bytes = 0
        while True:
            cumulative_bytes = self.total_bytes + payload_bytes + journal_bytes
            event = {
                "schema_version": "sanctionbench.checkpoint_write.v1",
                "run_identity_sha256": self.identity_sha256,
                "sequence": sequence,
                "category": category,
                "payload_bytes": payload_bytes,
                "journal_bytes": journal_bytes,
                "cumulative_bytes": cumulative_bytes,
            }
            observed_journal_bytes = len((canonical_json(event) + "\n").encode("utf-8"))
            if observed_journal_bytes == journal_bytes:
                break
            journal_bytes = observed_journal_bytes
        if cumulative_bytes > MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES:
            raise ValueError(
                "Cumulative checkpoint write budget is exhausted: "
                f"at most {MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES} bytes are permitted per run"
            )
        appended = append_jsonl_record(
            self.path,
            event,
            max_file_bytes=MAX_CHECKPOINT_WRITE_JOURNAL_BYTES,
            max_line_bytes=MAX_CHECKPOINT_WRITE_EVENT_BYTES,
        )
        if appended != journal_bytes:
            raise RuntimeError("Checkpoint journal byte accounting did not reconcile")
        self.total_bytes = cumulative_bytes
        self.event_count = sequence

    def reserve_json(self, *, category: str, payload: Any) -> None:
        encoded_bytes = len(
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
        self.reserve(category=category, payload_bytes=encoded_bytes)

    def reserve_jsonl(self, *, category: str, rows: list[Any]) -> None:
        encoded_bytes = 0
        for row in rows:
            value = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            encoded_bytes += len((canonical_json(value) + "\n").encode("utf-8"))
        if encoded_bytes < 1:
            raise ValueError("Checkpoint JSONL payload must not be empty")
        self.reserve(category=category, payload_bytes=encoded_bytes)


def _checkpoint_write_metadata(budget: _CheckpointWriteBudget) -> dict[str, int]:
    return {
        "checkpoint_write_event_count": budget.event_count,
        "cumulative_checkpoint_bytes_written": budget.total_bytes,
        "maximum_cumulative_checkpoint_write_bytes": MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES,
    }


def _write_prediction_checkpoint(
    path: Path,
    completed: dict[str, Any],
    *,
    budget: _CheckpointWriteBudget,
    category: str,
) -> None:
    ordered = [completed[key] for key in sorted(completed)]
    budget.reserve_jsonl(category=category, rows=ordered)
    write_jsonl(path, ordered)


class _RequestAttemptLedger:
    """Durably count provider requests before each network/client invocation."""

    def __init__(
        self,
        path: Path,
        *,
        identity_sha256: str,
        selected_ids: list[str],
        write_budget: _CheckpointWriteBudget | None = None,
    ) -> None:
        self.path = path
        self.identity_sha256 = identity_sha256
        self.selected_ids = selected_ids
        self._selected = set(selected_ids)
        self.write_budget = write_budget or _CheckpointWriteBudget(
            path.with_name("checkpoint_writes.jsonl"), identity_sha256=identity_sha256
        )
        self.counts: dict[str, int] = {}
        if path.is_file():
            payload = read_json(path, max_bytes=MAX_REQUEST_LEDGER_BYTES)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: invalid provider-request ledger")
            if (
                payload.get("schema_version") != "sanctionbench.provider_request_attempts.v1"
                or payload.get("run_identity_sha256") != identity_sha256
                or payload.get("selected_ids") != selected_ids
            ):
                raise ValueError(f"{path}: provider-request ledger identity differs")
            raw_counts = payload.get("attempts_started_by_item")
            if not isinstance(raw_counts, dict):
                raise ValueError(f"{path}: invalid provider-request counts")
            for item_id, count in raw_counts.items():
                if item_id not in self._selected or not isinstance(count, int) or count < 0:
                    raise ValueError(f"{path}: invalid provider-request count for {item_id}")
                self.counts[item_id] = count

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def record_started(self, item_id: str) -> None:
        if item_id not in self._selected:
            raise ValueError(f"Provider request item is outside the run identity: {item_id}")
        if self.counts.get(item_id, 0) >= MAX_PROVIDER_ATTEMPTS_PER_ITEM:
            raise RuntimeError(
                f"Provider request ceiling exhausted for {item_id}: "
                f"at most {MAX_PROVIDER_ATTEMPTS_PER_ITEM} attempts are permitted across resumes"
            )
        self.counts[item_id] = self.counts.get(item_id, 0) + 1
        payload = {
            "schema_version": "sanctionbench.provider_request_attempts.v1",
            "run_identity_sha256": self.identity_sha256,
            "selected_ids": self.selected_ids,
            "attempts_started_by_item": self.counts,
            "provider_request_count": self.total,
            "counting_semantics": (
                "durable pre-request count; a crash after checkpointing but before transport "
                "may conservatively overcount one attempt"
            ),
        }
        self.write_budget.reserve_json(category="provider_request_ledger", payload=payload)
        write_json(
            self.path,
            payload,
            max_bytes=MAX_REQUEST_LEDGER_BYTES,
        )


class _CourtListenerAttemptLedger:
    """Durably account for every CourtListener wire attempt before transport."""

    def __init__(
        self,
        path: Path,
        *,
        identity_sha256: str,
        selected_ids: list[str],
        write_budget: _CheckpointWriteBudget | None = None,
    ) -> None:
        if len(selected_ids) > MAX_COURTLISTENER_LOOKUPS_PER_RUN:
            raise ValueError(
                "CourtListener lookup count exceeds the per-run "
                f"{MAX_COURTLISTENER_LOOKUPS_PER_RUN}-lookup limit"
            )
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("CourtListener lookup identities must be duplicate-free")
        self.path = path
        self.identity_sha256 = identity_sha256
        self.selected_ids = selected_ids
        self._selected = set(selected_ids)
        self.write_budget = write_budget or _CheckpointWriteBudget(
            path.with_name("checkpoint_writes.jsonl"), identity_sha256=identity_sha256
        )
        self.counts: dict[str, int] = {}
        if path.is_file():
            payload = read_json(path, max_bytes=MAX_REQUEST_LEDGER_BYTES)
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version")
                != "sanctionbench.courtlistener_request_attempts.v1"
                or payload.get("run_identity_sha256") != identity_sha256
                or payload.get("selected_ids") != selected_ids
                or payload.get("maximum_attempts_per_lookup")
                != MAX_COURTLISTENER_ATTEMPTS_PER_LOOKUP
            ):
                raise ValueError(f"{path}: CourtListener-request ledger identity differs")
            raw_counts = payload.get("attempts_started_by_lookup")
            if not isinstance(raw_counts, dict):
                raise ValueError(f"{path}: invalid CourtListener-request counts")
            for lookup_id, count in raw_counts.items():
                if (
                    lookup_id not in self._selected
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 0
                    or count > MAX_COURTLISTENER_ATTEMPTS_PER_LOOKUP
                ):
                    raise ValueError(f"{path}: invalid CourtListener count for {lookup_id}")
                self.counts[lookup_id] = count
        else:
            self._write()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def _write(self) -> None:
        payload = {
            "schema_version": "sanctionbench.courtlistener_request_attempts.v1",
            "run_identity_sha256": self.identity_sha256,
            "selected_ids": self.selected_ids,
            "maximum_attempts_per_lookup": MAX_COURTLISTENER_ATTEMPTS_PER_LOOKUP,
            "attempts_started_by_lookup": self.counts,
            "courtlistener_request_count": self.total,
            "counting_semantics": (
                "durable pre-transport wire-attempt count; cache hits do not increment and a "
                "crash after checkpointing may conservatively overcount one attempt"
            ),
        }
        self.write_budget.reserve_json(category="courtlistener_request_ledger", payload=payload)
        write_json(
            self.path,
            payload,
            max_bytes=MAX_REQUEST_LEDGER_BYTES,
        )

    def record_started(self, lookup_id: str) -> None:
        if lookup_id not in self._selected:
            raise ValueError(f"CourtListener lookup is outside the run identity: {lookup_id}")
        if self.counts.get(lookup_id, 0) >= MAX_COURTLISTENER_ATTEMPTS_PER_LOOKUP:
            raise RuntimeError(
                f"CourtListener request ceiling exhausted for {lookup_id}: at most "
                f"{MAX_COURTLISTENER_ATTEMPTS_PER_LOOKUP} attempts are permitted"
            )
        self.counts[lookup_id] = self.counts.get(lookup_id, 0) + 1
        self._write()


def _require_checkpoint_request_receipts(
    completed_ids: set[str], attempts: _RequestAttemptLedger
) -> None:
    """Do not accept a resumed prediction with no recorded provider invocation."""

    missing = {item_id for item_id in completed_ids if attempts.counts.get(item_id, 0) < 1}
    if missing:
        raise ValueError(
            "Completed checkpoint predictions lack provider-request receipts for "
            f"{len(missing)} item(s)"
        )


class _ToolEvidenceLedger:
    """Checkpoint the exact secret-free evidence supplied to tool-assisted providers."""

    def __init__(
        self,
        path: Path,
        *,
        identity_sha256: str,
        selected_ids: list[str],
        write_budget: _CheckpointWriteBudget | None = None,
    ) -> None:
        self.path = path
        self.identity_sha256 = identity_sha256
        self.selected_ids = selected_ids
        self._selected = set(selected_ids)
        self.write_budget = write_budget or _CheckpointWriteBudget(
            path.with_name("checkpoint_writes.jsonl"), identity_sha256=identity_sha256
        )
        self.evidence_by_item: dict[str, dict[str, Any]] = {}
        if path.is_file():
            payload = read_json(path, max_bytes=MAX_TOOL_EVIDENCE_LEDGER_BYTES)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: invalid tool-evidence payload")
            if (
                payload.get("schema_version") != "sanctionbench.tool_evidence.v1"
                or payload.get("run_identity_sha256") != identity_sha256
                or payload.get("selected_ids") != selected_ids
                or payload.get("tool_spec_version") != TOOL_SPEC_VERSION
            ):
                raise ValueError(f"{path}: tool-evidence ledger identity differs")
            raw_evidence = payload.get("evidence_by_item")
            if not isinstance(raw_evidence, dict):
                raise ValueError(f"{path}: invalid tool-evidence payload")
            for item_id, evidence in raw_evidence.items():
                if item_id not in self._selected or not isinstance(evidence, dict):
                    raise ValueError(f"{path}: invalid tool evidence for {item_id}")
                self.evidence_by_item[item_id] = evidence

    @property
    def count(self) -> int:
        return len(self.evidence_by_item)

    def get(self, item_id: str) -> dict[str, Any] | None:
        if item_id not in self._selected:
            raise ValueError(f"Tool-evidence item is outside the run identity: {item_id}")
        return self.evidence_by_item.get(item_id)

    def record(self, item_id: str, evidence: dict[str, Any]) -> None:
        if item_id not in self._selected:
            raise ValueError(f"Tool-evidence item is outside the run identity: {item_id}")
        if len(canonical_json(evidence).encode("utf-8")) > MAX_TOOL_EVIDENCE_ITEM_BYTES:
            raise ValueError(
                f"Tool evidence for {item_id} exceeds the "
                f"{MAX_TOOL_EVIDENCE_ITEM_BYTES}-byte item limit"
            )
        existing = self.evidence_by_item.get(item_id)
        if existing is not None and canonical_json(existing) != canonical_json(evidence):
            raise ValueError(f"Tool evidence changed for already-checkpointed item: {item_id}")
        self.evidence_by_item[item_id] = evidence
        payload = {
            "schema_version": "sanctionbench.tool_evidence.v1",
            "run_identity_sha256": self.identity_sha256,
            "tool_spec_version": TOOL_SPEC_VERSION,
            "selected_ids": self.selected_ids,
            "evidence_by_item": self.evidence_by_item,
            "evidence_item_count": self.count,
            "receipt_semantics": (
                "exact secret-free model-visible CourtListener evidence, checkpointed before "
                "the corresponding provider request and reused on resume"
            ),
        }
        self.write_budget.reserve_json(category="tool_evidence_ledger", payload=payload)
        write_json(
            self.path,
            payload,
            max_bytes=MAX_TOOL_EVIDENCE_LEDGER_BYTES,
        )

    def require_complete(self) -> None:
        missing = self._selected - self.evidence_by_item.keys()
        if missing:
            raise ValueError(
                f"{self.path}: missing tool evidence for {len(missing)} selected item(s)"
            )


def _tool_evidence_metadata(*, root: Path, ledger: _ToolEvidenceLedger | None) -> dict[str, Any]:
    if ledger is None:
        return {
            "tool_evidence_path": None,
            "tool_evidence_sha256": None,
            "tool_evidence_item_count": 0,
        }
    ledger.require_complete()
    return {
        "tool_evidence_path": _path_reference(root, ledger.path),
        "tool_evidence_sha256": sha256_file(ledger.path),
        "tool_evidence_item_count": ledger.count,
    }


def _courtlistener_request_metadata(
    *, root: Path, ledger: _CourtListenerAttemptLedger | None
) -> dict[str, Any]:
    if ledger is None:
        return {
            "courtlistener_request_attempts_path": None,
            "courtlistener_request_attempts_sha256": None,
            "courtlistener_request_count": 0,
        }
    return {
        "courtlistener_request_attempts_path": _path_reference(root, ledger.path),
        "courtlistener_request_attempts_sha256": sha256_file(ledger.path),
        "courtlistener_request_count": ledger.total,
    }


def _select_items(
    items: list[CitationItem], max_items: int | None, seed: int
) -> list[CitationItem]:
    if max_items is None or max_items >= len(items):
        return items
    if max_items < 2:
        raise ValueError("max_items must be at least 2 for the matched-pair design")
    by_pair: dict[str, list[CitationItem]] = defaultdict(list)
    for item in items:
        by_pair[item.matched_pair_id].append(item)
    complete_pairs = [pair for pair in by_pair.values() if len(pair) == 2]
    random.Random(seed).shuffle(complete_pairs)
    selected_pairs = complete_pairs[: max_items // 2]
    selected = [item for pair in selected_pairs for item in pair]
    return sorted(selected, key=lambda item: item.item_id)


def _apply_temporal_cutoff(
    items: list[CitationItem], cutoff: TemporalCutoff | None
) -> tuple[list[CitationItem], dict[str, Any]]:
    """Filter complete matched pairs using an exclusive temporal lower bound."""

    if cutoff is None:
        return items, {
            "applied": False,
            "input_item_count": len(items),
            "eligible_item_count": len(items),
        }

    cutoff_date = date.fromisoformat(cutoff.cutoff_date)
    by_pair: dict[str, list[CitationItem]] = defaultdict(list)
    for item in items:
        by_pair[item.matched_pair_id].append(item)

    selected: list[CitationItem] = []
    excluded_missing = 0
    excluded_on_or_before = 0
    excluded_incomplete_pairs = 0
    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            raise ValueError(
                f"Temporal filtering requires complete two-item pairs; {pair_id!r} has {len(pair)}"
            )
        pair_dates: list[date] = []
        pair_has_missing = False
        for item in pair:
            raw_value = getattr(item, cutoff.field)
            if raw_value is None:
                pair_has_missing = True
                if cutoff.missing == "error":
                    raise ValueError(
                        f"{item.item_id}: {cutoff.field} is missing under missing=error"
                    )
                continue
            try:
                pair_dates.append(date.fromisoformat(raw_value))
            except ValueError as error:
                raise ValueError(
                    f"{item.item_id}: invalid {cutoff.field} date {raw_value!r}"
                ) from error
        if pair_has_missing:
            excluded_missing += len(pair)
            excluded_incomplete_pairs += 1
            continue
        if all(value > cutoff_date for value in pair_dates):
            selected.extend(pair)
        else:
            excluded_on_or_before += len(pair)

    selected.sort(key=lambda item: item.item_id)
    report = {
        "applied": True,
        **cutoff.model_dump(mode="json"),
        "semantics": "include only complete matched pairs whose dates are strictly after cutoff_date",
        "input_item_count": len(items),
        "eligible_item_count": len(selected),
        "excluded_missing_date_item_count": excluded_missing,
        "excluded_on_or_before_cutoff_item_count": excluded_on_or_before,
        "excluded_pair_count_due_to_missing_date": excluded_incomplete_pairs,
    }
    if not selected:
        raise ValueError(
            "Temporal cutoff removed every complete matched pair; adjust cutoff_date, field, or "
            "missing policy"
        )
    return selected, report


def _filter_document_scenarios(
    scenarios: list[DocumentScenario], eligible_citation_ids: set[str]
) -> tuple[list[DocumentScenario], dict[str, int]]:
    eligible = [
        scenario
        for scenario in scenarios
        if all(
            authority.citation_item_id in eligible_citation_ids
            for authority in scenario.authorities
        )
    ]
    return eligible, {
        "input_document_count": len(scenarios),
        "eligible_document_count": len(eligible),
        "excluded_document_count": len(scenarios) - len(eligible),
    }


def _load_citation_checkpoint(path: Path, expected_ids: set[str]) -> dict[str, Prediction]:
    completed: dict[str, Prediction] = {}
    for row in read_jsonl(path):
        prediction = Prediction.model_validate(row)
        if prediction.item_id in completed:
            raise ValueError(f"Duplicate checkpoint prediction for {prediction.item_id}")
        if prediction.item_id not in expected_ids:
            raise ValueError(
                f"Checkpoint contains item outside this run identity: {prediction.item_id}"
            )
        completed[prediction.item_id] = prediction
    return completed


def _load_document_checkpoint(path: Path, expected_ids: set[str]) -> dict[str, DocumentPrediction]:
    completed: dict[str, DocumentPrediction] = {}
    for row in read_jsonl(path):
        prediction = DocumentPrediction.model_validate(row)
        if prediction.item_id in completed:
            raise ValueError(f"Duplicate checkpoint prediction for {prediction.item_id}")
        if prediction.item_id not in expected_ids:
            raise ValueError(
                f"Checkpoint contains scenario outside this run identity: {prediction.item_id}"
            )
        completed[prediction.item_id] = prediction
    return completed


def _load_organic_document_checkpoint(
    path: Path, expected_ids: set[str]
) -> dict[str, OrganicDocumentPrediction]:
    completed: dict[str, OrganicDocumentPrediction] = {}
    for row in read_jsonl(path):
        prediction = OrganicDocumentPrediction.model_validate(row)
        if prediction.item_id in completed:
            raise ValueError(f"Duplicate checkpoint prediction for {prediction.item_id}")
        if prediction.item_id not in expected_ids:
            raise ValueError(
                f"Checkpoint contains organic document outside this run identity: "
                f"{prediction.item_id}"
            )
        completed[prediction.item_id] = prediction
    return completed


def _predict_with_retry(
    provider: Provider,
    item: CitationItem,
    condition: Condition,
    evidence: dict[str, Any] | None,
    record_attempt: Callable[[str], None],
) -> Prediction:
    actual_input_bytes = validate_provider_request_input(
        "citation_verification", build_user_prompt(item, condition, evidence)
    )
    if evidence is not None:
        reserved_input_bytes = (
            provider_request_input_bytes(
                "citation_verification", build_user_prompt(item, condition, None)
            )
            + MAX_TOOL_EVIDENCE_PROMPT_RESERVATION_BYTES
        )
        if actual_input_bytes > reserved_input_bytes:
            raise ValueError("citation tool evidence exceeds its campaign input-byte reservation")
    retrying = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            record_attempt(item.item_id)
            return provider.predict(item, condition, evidence)
    raise RuntimeError("Retry loop exited without a prediction")


def _predict_document_with_retry(
    provider: Provider,
    scenario: DocumentScenario,
    condition: Condition,
    evidence: dict[str, Any] | None,
    record_attempt: Callable[[str], None],
) -> DocumentPrediction:
    actual_input_bytes = validate_provider_request_input(
        "document_audit", build_document_user_prompt(scenario, condition, evidence)
    )
    if evidence is not None:
        reserved_input_bytes = (
            provider_request_input_bytes(
                "document_audit", build_document_user_prompt(scenario, condition, None)
            )
            + MAX_TOOL_EVIDENCE_PROMPT_RESERVATION_BYTES
        )
        if actual_input_bytes > reserved_input_bytes:
            raise ValueError("document tool evidence exceeds its campaign input-byte reservation")
    retrying = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            record_attempt(scenario.item_id)
            return provider.predict_document(scenario, condition, evidence)
    raise RuntimeError("Retry loop exited without a document prediction")


def _predict_organic_document_with_retry(
    provider: Provider,
    document: OrganicDocumentGold,
    condition: Condition,
    record_attempt: Callable[[str], None],
) -> OrganicDocumentPrediction:
    validate_provider_request_input(
        "organic_document_audit",
        build_organic_document_user_prompt(document.model_input(), condition),
    )
    retrying = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            record_attempt(document.item_id)
            return provider.predict_organic_document(document.model_input(), condition)
    raise RuntimeError("Retry loop exited without an organic document prediction")


def _run_one(
    *,
    items: list[CitationItem],
    dataset_path: Path,
    output_root: Path,
    provider_name: str,
    model: str,
    condition: Condition,
    seed: int,
    resume: bool,
    config_sha256: str,
    temporal_cutoff: TemporalCutoff | None,
    temporal_filter_report: dict[str, Any],
    git_state: dict[str, Any],
    benchmark_release_status: str,
    resource_budget: dict[str, int | None],
) -> dict[str, Any]:
    root = project_root()
    provider = create_provider(provider_name, model)
    dataset_digest = sha256_file(dataset_path)
    prompt_hash = _prompt_sha256("citation_verification")
    identity, identity_digest = _run_identity(
        task_type="citation_verification",
        dataset_sha256=dataset_digest,
        subset_sha256=_records_sha256(items),
        selected_ids=[item.item_id for item in items],
        config_sha256=config_sha256,
        provider_name=provider_name,
        provider_protocol_version=provider.protocol_version,
        provider_runtime_identity=provider.runtime_identity(),
        runtime_source_sha256=_runtime_source_sha256(),
        model=model,
        condition=condition,
        seed=seed,
        prompt_sha256=prompt_hash,
        temporal_cutoff=temporal_cutoff,
        resource_budget=resource_budget,
    )
    run_name = f"{_slug(provider_name)}--{_slug(model)}--{condition.value.replace('_', '-')}"
    run_id, run_dir, already_finalized = _prepare_run_directory(
        output_root=output_root,
        run_name=run_name,
        identity=identity,
        identity_sha256=identity_digest,
    )
    if already_finalized:
        return _summary_from_finalized_run(
            root=root,
            run_path=run_dir / "run.json",
            task_type="citation_verification",
        )
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_path = run_dir / "run.json"
    attempts_path = run_dir / "request_attempts.json"
    courtlistener_attempts_path = run_dir / "courtlistener_request_attempts.json"
    tool_evidence_path = run_dir / "tool_evidence.json"
    checkpoint_write_budget = _CheckpointWriteBudget(
        run_dir / "checkpoint_writes.jsonl", identity_sha256=identity_digest
    )
    attempts = _RequestAttemptLedger(
        attempts_path,
        identity_sha256=identity_digest,
        selected_ids=[item.item_id for item in items],
        write_budget=checkpoint_write_budget,
    )
    tool_evidence_ledger = (
        _ToolEvidenceLedger(
            tool_evidence_path,
            identity_sha256=identity_digest,
            selected_ids=[item.item_id for item in items],
            write_budget=checkpoint_write_budget,
        )
        if condition == Condition.TOOL_ASSISTED
        else None
    )
    courtlistener_attempts = (
        _CourtListenerAttemptLedger(
            courtlistener_attempts_path,
            identity_sha256=identity_digest,
            selected_ids=[item.item_id for item in items],
            write_budget=checkpoint_write_budget,
        )
        if condition == Condition.TOOL_ASSISTED
        else None
    )

    completed: dict[str, Prediction] = {}
    if resume and predictions_path.exists():
        completed = _load_citation_checkpoint(predictions_path, {item.item_id for item in items})
        _require_checkpoint_request_receipts(set(completed), attempts)
    if tool_evidence_ledger is not None:
        missing_for_completed = {
            item_id for item_id in completed if tool_evidence_ledger.get(item_id) is None
        }
        if missing_for_completed:
            raise ValueError(
                "Completed predictions are missing their tool-evidence receipt for "
                f"{len(missing_for_completed)} item(s)"
            )

    lookup_client: CourtListenerClient | None = None
    current_lookup_id: str | None = None

    def record_courtlistener_attempt() -> None:
        if courtlistener_attempts is None or current_lookup_id is None:
            raise RuntimeError("CourtListener attempted a request outside an identified lookup")
        courtlistener_attempts.record_started(current_lookup_id)

    try:
        for item in items:
            if item.item_id in completed:
                continue
            evidence: dict[str, Any] | None = None
            if tool_evidence_ledger is not None:
                evidence = tool_evidence_ledger.get(item.item_id)
                if evidence is None:
                    if lookup_client is None:
                        lookup_client = CourtListenerClient(
                            delay_seconds=2.0,
                            max_response_bytes=RUNNER_MAX_COURTLISTENER_RESPONSE_BYTES,
                            before_http_request=record_courtlistener_attempt,
                        )
                    current_lookup_id = item.item_id
                    try:
                        evidence = lookup_client.citation_lookup(
                            item.citation, item.case_name, item.proposition
                        )
                    finally:
                        current_lookup_id = None
                    tool_evidence_ledger.record(item.item_id, evidence)
            prediction = _predict_with_retry(
                provider, item, condition, evidence, attempts.record_started
            )
            if prediction.item_id != item.item_id:
                raise ValueError(
                    f"Provider returned item_id {prediction.item_id!r} for {item.item_id!r}"
                )
            completed[item.item_id] = prediction
            _write_prediction_checkpoint(
                predictions_path,
                completed,
                budget=checkpoint_write_budget,
                category="citation_prediction_checkpoint",
            )
    finally:
        if lookup_client:
            lookup_client.close()

    ordered_predictions = [completed[item.item_id] for item in items]
    metrics = score_predictions(items, ordered_predictions)
    write_json(metrics_path, metrics)
    record = RunRecord(
        run_id=run_id,
        created_at=utc_now(),
        provider=provider_name,
        model=model,
        condition=condition,
        dataset_path=_path_reference(root, dataset_path),
        dataset_sha256=dataset_digest,
        seed=seed,
        mock=bool(provider.is_mock),
        predictions_path=_path_reference(root, predictions_path),
        metrics_path=_path_reference(root, metrics_path),
        item_count=len(items),
        metadata={
            "run_identity_sha256": identity_digest,
            "run_identity": identity,
            "git_commit": git_state["commit"],
            "git_dirty": git_state["dirty"],
            "git_state_at_manifest_start": git_state,
            "prompt_and_output_schema_sha256": prompt_hash,
            "temperature": 0,
            "matched_pair_sampling": True,
            "resume_enabled": resume,
            "provider_request_attempts_path": _path_reference(root, attempts_path),
            "provider_request_count": attempts.total,
            "successful_response_count": len(items),
            "retry_or_interrupted_request_count": attempts.total - len(items),
            **_checkpoint_write_metadata(checkpoint_write_budget),
            "temporal_filter": temporal_filter_report,
            "tool": (
                "CourtListener v4 opinion search via citation_lookup"
                if condition == Condition.TOOL_ASSISTED
                else None
            ),
            **_tool_evidence_metadata(root=root, ledger=tool_evidence_ledger),
            **_courtlistener_request_metadata(root=root, ledger=courtlistener_attempts),
            "mock_warning": (
                "NO LIVE MODEL CALLS: deterministic rule-based client" if provider.is_mock else None
            ),
            "usage_accounting": "request_count_recorded_tokens_unavailable",
        },
    )
    finalized_record, finalized_identity_sha256 = _finalize_run_record(
        record=record,
        benchmark_release_status=benchmark_release_status,
        attempts_path=attempts_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        tool_evidence_sha256=record.metadata.get("tool_evidence_sha256"),
    )
    write_json(run_path, finalized_record)
    return {
        "run_id": run_id,
        "task_type": "citation_verification",
        "mock": record.mock,
        "condition": condition.value,
        "item_count": len(items),
        "provider_request_count": attempts.total,
        "courtlistener_request_count": (
            courtlistener_attempts.total if courtlistener_attempts is not None else 0
        ),
        "run_identity_sha256": identity_digest,
        "finalized_run_identity_sha256": finalized_identity_sha256,
        "sanction_score": metrics["headline"]["score"],
        "false_positive_rate": metrics["binary"]["false_positive_rate"],
        "run_file": _path_reference(root, run_path),
    }


def _run_document_one(
    *,
    scenarios: list[DocumentScenario],
    citation_items: dict[str, CitationItem],
    dataset_path: Path,
    output_root: Path,
    provider_name: str,
    model: str,
    condition: Condition,
    seed: int,
    resume: bool,
    config_sha256: str,
    citation_dataset_sha256: str,
    temporal_cutoff: TemporalCutoff | None,
    temporal_filter_report: dict[str, Any],
    document_filter_report: dict[str, int],
    git_state: dict[str, Any],
    benchmark_release_status: str,
    resource_budget: dict[str, int | None],
) -> dict[str, Any]:
    root = project_root()
    provider = create_provider(provider_name, model)
    dataset_digest = sha256_file(dataset_path)
    prompt_hash = _prompt_sha256("document_audit")
    identity, identity_digest = _run_identity(
        task_type="document_audit",
        dataset_sha256=dataset_digest,
        subset_sha256=_records_sha256(scenarios),
        selected_ids=[scenario.item_id for scenario in scenarios],
        config_sha256=config_sha256,
        provider_name=provider_name,
        provider_protocol_version=provider.protocol_version,
        provider_runtime_identity=provider.runtime_identity(),
        runtime_source_sha256=_runtime_source_sha256(),
        model=model,
        condition=condition,
        seed=seed,
        prompt_sha256=prompt_hash,
        temporal_cutoff=temporal_cutoff,
        resource_budget=resource_budget,
        reference_dataset_sha256=citation_dataset_sha256,
    )
    run_name = (
        f"{_slug(provider_name)}--{_slug(model)}--document-audit--"
        f"{condition.value.replace('_', '-')}"
    )
    run_id, run_dir, already_finalized = _prepare_run_directory(
        output_root=output_root,
        run_name=run_name,
        identity=identity,
        identity_sha256=identity_digest,
    )
    if already_finalized:
        return _summary_from_finalized_run(
            root=root,
            run_path=run_dir / "run.json",
            task_type="document_audit",
        )
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_path = run_dir / "run.json"
    attempts_path = run_dir / "request_attempts.json"
    courtlistener_attempts_path = run_dir / "courtlistener_request_attempts.json"
    tool_evidence_path = run_dir / "tool_evidence.json"
    checkpoint_write_budget = _CheckpointWriteBudget(
        run_dir / "checkpoint_writes.jsonl", identity_sha256=identity_digest
    )
    attempts = _RequestAttemptLedger(
        attempts_path,
        identity_sha256=identity_digest,
        selected_ids=[scenario.item_id for scenario in scenarios],
        write_budget=checkpoint_write_budget,
    )
    tool_evidence_ledger = (
        _ToolEvidenceLedger(
            tool_evidence_path,
            identity_sha256=identity_digest,
            selected_ids=[scenario.item_id for scenario in scenarios],
            write_budget=checkpoint_write_budget,
        )
        if condition == Condition.TOOL_ASSISTED
        else None
    )
    document_lookup_ids = [
        f"{scenario.item_id}:{authority.authority_id}"
        for scenario in scenarios
        for authority in scenario.authorities
    ]
    courtlistener_attempts = (
        _CourtListenerAttemptLedger(
            courtlistener_attempts_path,
            identity_sha256=identity_digest,
            selected_ids=document_lookup_ids,
            write_budget=checkpoint_write_budget,
        )
        if condition == Condition.TOOL_ASSISTED
        else None
    )
    completed: dict[str, DocumentPrediction] = {}
    if resume and predictions_path.exists():
        completed = _load_document_checkpoint(
            predictions_path, {scenario.item_id for scenario in scenarios}
        )
        _require_checkpoint_request_receipts(set(completed), attempts)
    if tool_evidence_ledger is not None:
        missing_for_completed = {
            item_id for item_id in completed if tool_evidence_ledger.get(item_id) is None
        }
        if missing_for_completed:
            raise ValueError(
                "Completed document predictions are missing their tool-evidence receipt for "
                f"{len(missing_for_completed)} item(s)"
            )
    lookup_client: CourtListenerClient | None = None
    current_lookup_id: str | None = None

    def record_courtlistener_attempt() -> None:
        if courtlistener_attempts is None or current_lookup_id is None:
            raise RuntimeError("CourtListener attempted a request outside an identified lookup")
        courtlistener_attempts.record_started(current_lookup_id)

    try:
        for scenario in scenarios:
            if scenario.item_id in completed:
                continue
            tool_evidence: dict[str, Any] | None = None
            if tool_evidence_ledger is not None:
                tool_evidence = tool_evidence_ledger.get(scenario.item_id)
                if tool_evidence is None:
                    if lookup_client is None:
                        lookup_client = CourtListenerClient(
                            delay_seconds=2.0,
                            max_response_bytes=RUNNER_MAX_COURTLISTENER_RESPONSE_BYTES,
                            before_http_request=record_courtlistener_attempt,
                        )
                    tool_evidence = {}
                    for authority in scenario.authorities:
                        item = citation_items[authority.citation_item_id]
                        current_lookup_id = f"{scenario.item_id}:{authority.authority_id}"
                        try:
                            tool_evidence[authority.authority_id] = lookup_client.citation_lookup(
                                item.citation, item.case_name, item.proposition
                            )
                        finally:
                            current_lookup_id = None
                    tool_evidence_ledger.record(scenario.item_id, tool_evidence)
            prediction = _predict_document_with_retry(
                provider,
                scenario,
                condition,
                tool_evidence,
                attempts.record_started,
            )
            if prediction.item_id != scenario.item_id:
                raise ValueError(
                    f"Provider returned item_id {prediction.item_id!r} for {scenario.item_id!r}"
                )
            completed[scenario.item_id] = prediction
            _write_prediction_checkpoint(
                predictions_path,
                completed,
                budget=checkpoint_write_budget,
                category="document_prediction_checkpoint",
            )
    finally:
        if lookup_client:
            lookup_client.close()
    ordered = [completed[scenario.item_id] for scenario in scenarios]
    metrics = score_document_predictions(scenarios, ordered, citation_items)
    write_json(metrics_path, metrics)
    record = RunRecord(
        run_id=run_id,
        created_at=utc_now(),
        provider=provider_name,
        model=model,
        condition=condition,
        dataset_path=_path_reference(root, dataset_path),
        dataset_sha256=dataset_digest,
        seed=seed,
        mock=bool(provider.is_mock),
        predictions_path=_path_reference(root, predictions_path),
        metrics_path=_path_reference(root, metrics_path),
        item_count=sum(len(scenario.authorities) for scenario in scenarios),
        metadata={
            "task_type": "document_audit",
            "document_count": len(scenarios),
            "run_identity_sha256": identity_digest,
            "run_identity": identity,
            "git_commit": git_state["commit"],
            "git_dirty": git_state["dirty"],
            "git_state_at_manifest_start": git_state,
            "prompt_and_output_schema_sha256": prompt_hash,
            "temperature": 0,
            "track": "constructed_from_organic",
            "resume_enabled": resume,
            "provider_request_attempts_path": _path_reference(root, attempts_path),
            "provider_request_count": attempts.total,
            "successful_response_count": len(scenarios),
            "retry_or_interrupted_request_count": attempts.total - len(scenarios),
            **_checkpoint_write_metadata(checkpoint_write_budget),
            "temporal_filter": temporal_filter_report,
            "document_temporal_filter": document_filter_report,
            **_tool_evidence_metadata(root=root, ledger=tool_evidence_ledger),
            **_courtlistener_request_metadata(root=root, ledger=courtlistener_attempts),
            "mock_warning": (
                "NO LIVE MODEL CALLS: deterministic rule-based client" if provider.is_mock else None
            ),
            "usage_accounting": "request_count_recorded_tokens_unavailable",
        },
    )
    finalized_record, finalized_identity_sha256 = _finalize_run_record(
        record=record,
        benchmark_release_status=benchmark_release_status,
        attempts_path=attempts_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        tool_evidence_sha256=record.metadata.get("tool_evidence_sha256"),
    )
    write_json(run_path, finalized_record)
    return {
        "run_id": run_id,
        "task_type": "document_audit",
        "mock": record.mock,
        "condition": condition.value,
        "item_count": record.item_count,
        "provider_request_count": attempts.total,
        "courtlistener_request_count": (
            courtlistener_attempts.total if courtlistener_attempts is not None else 0
        ),
        "run_identity_sha256": identity_digest,
        "finalized_run_identity_sha256": finalized_identity_sha256,
        "sanction_score": metrics["headline"]["score"],
        "false_positive_rate": metrics["binary"]["false_positive_rate"],
        "run_file": _path_reference(root, run_path),
    }


def _run_organic_document_one(
    *,
    documents: list[OrganicDocumentGold],
    dataset_path: Path,
    output_root: Path,
    provider_name: str,
    model: str,
    condition: Condition,
    seed: int,
    repetition: int,
    resume: bool,
    config_sha256: str,
    git_state: dict[str, Any],
    benchmark_release_status: str,
    resource_budget: dict[str, int | None],
) -> dict[str, Any]:
    if condition != Condition.CLOSED_BOOK:
        raise ValueError(
            "Organic document v1 supports closed_book only; self-directed tool use is a separate "
            "future track and authority-prefetched evidence would leak the gold inventory"
        )
    root = project_root()
    provider = create_provider(provider_name, model)
    dataset_digest = sha256_file(dataset_path)
    prompt_hash = _prompt_sha256("organic_document_audit")
    identity, identity_digest = _run_identity(
        task_type="organic_document_audit",
        dataset_sha256=dataset_digest,
        subset_sha256=_records_sha256(documents),
        selected_ids=[document.item_id for document in documents],
        config_sha256=config_sha256,
        provider_name=provider_name,
        provider_protocol_version=provider.protocol_version,
        provider_runtime_identity=provider.runtime_identity(),
        runtime_source_sha256=_runtime_source_sha256(),
        model=model,
        condition=condition,
        seed=seed,
        prompt_sha256=prompt_hash,
        temporal_cutoff=None,
        resource_budget=resource_budget,
        repetition=repetition,
    )
    run_name = (
        f"{_slug(provider_name)}--{_slug(model)}--organic-document-audit--"
        f"{condition.value.replace('_', '-')}--repeat-{repetition:02d}"
    )
    run_id, run_dir, already_finalized = _prepare_run_directory(
        output_root=output_root,
        run_name=run_name,
        identity=identity,
        identity_sha256=identity_digest,
    )
    if already_finalized:
        return _summary_from_finalized_run(
            root=root,
            run_path=run_dir / "run.json",
            task_type="organic_document_audit",
        )
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_path = run_dir / "run.json"
    attempts_path = run_dir / "request_attempts.json"
    checkpoint_write_budget = _CheckpointWriteBudget(
        run_dir / "checkpoint_writes.jsonl", identity_sha256=identity_digest
    )
    attempts = _RequestAttemptLedger(
        attempts_path,
        identity_sha256=identity_digest,
        selected_ids=[document.item_id for document in documents],
        write_budget=checkpoint_write_budget,
    )
    completed: dict[str, OrganicDocumentPrediction] = {}
    if resume and predictions_path.exists():
        completed = _load_organic_document_checkpoint(
            predictions_path, {document.item_id for document in documents}
        )
        _require_checkpoint_request_receipts(set(completed), attempts)
    for document in documents:
        if document.item_id in completed:
            continue
        prediction = _predict_organic_document_with_retry(
            provider, document, condition, attempts.record_started
        )
        if prediction.item_id != document.item_id:
            raise ValueError(
                f"Provider returned item_id {prediction.item_id!r} for {document.item_id!r}"
            )
        completed[document.item_id] = prediction
        _write_prediction_checkpoint(
            predictions_path,
            completed,
            budget=checkpoint_write_budget,
            category="organic_prediction_checkpoint",
        )
    ordered = [completed[document.item_id] for document in documents]
    metrics = score_organic_document_predictions(documents, ordered)
    write_json(metrics_path, metrics)
    authority_count = sum(len(document.authorities) for document in documents)
    record = RunRecord(
        run_id=run_id,
        created_at=utc_now(),
        provider=provider_name,
        model=model,
        condition=condition,
        dataset_path=_path_reference(root, dataset_path),
        dataset_sha256=dataset_digest,
        seed=seed,
        mock=bool(provider.is_mock),
        predictions_path=_path_reference(root, predictions_path),
        metrics_path=_path_reference(root, metrics_path),
        item_count=len(documents),
        metadata={
            "task_type": "organic_document_audit",
            "document_count": len(documents),
            "authority_occurrence_count": authority_count,
            "model_call_count": attempts.total,
            "successful_response_count": len(documents),
            "retry_or_interrupted_request_count": attempts.total - len(documents),
            **_checkpoint_write_metadata(checkpoint_write_budget),
            "provider_request_attempts_path": _path_reference(root, attempts_path),
            "provider_request_count": attempts.total,
            "one_isolated_successful_response_per_document": True,
            "repetition": repetition,
            "run_identity_sha256": identity_digest,
            "run_identity": identity,
            "git_commit": git_state["commit"],
            "git_dirty": git_state["dirty"],
            "git_state_at_manifest_start": git_state,
            "prompt_and_output_schema_sha256": prompt_hash,
            "temperature": 0,
            "gold_fields_visible_to_provider": False,
            "tool_assistance": None,
            "resume_enabled": resume,
            "mock_warning": (
                "NO LIVE MODEL CALLS: deterministic client returns no organic findings"
                if provider.is_mock
                else None
            ),
            "usage_accounting": "request_count_recorded_tokens_unavailable",
        },
    )
    finalized_record, finalized_identity_sha256 = _finalize_run_record(
        record=record,
        benchmark_release_status=benchmark_release_status,
        attempts_path=attempts_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        tool_evidence_sha256=None,
    )
    write_json(run_path, finalized_record)
    return {
        "run_id": run_id,
        "task_type": "organic_document_audit",
        "mock": record.mock,
        "condition": condition.value,
        "document_count": len(documents),
        "model_call_count": attempts.total,
        "provider_request_count": attempts.total,
        "repetition": repetition,
        "run_identity_sha256": identity_digest,
        "finalized_run_identity_sha256": finalized_identity_sha256,
        "organic_document_sanction_score": metrics["headline"]["score"],
        "clean_audit_rate": metrics["headline"]["clean_audit_rate"],
        "clean_control_false_alarm_rate": metrics["clean_controls"]["false_alarm_rate"],
        "run_file": _path_reference(root, run_path),
    }


def run_manifest(
    config_path: Path,
    *,
    temporal_cutoff_override: dict[str, str] | None = None,
    max_provider_requests: int | None = None,
    max_provider_input_bytes: int | None = None,
    max_courtlistener_requests: int | None = None,
) -> list[dict[str, Any]]:
    root = project_root().resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(read_text_bounded(config_path, max_bytes=MAX_CAMPAIGN_CONFIG_BYTES))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: benchmark config must be a YAML mapping")
    config_digest = sha256_file(config_path)
    git_state = _git_state(root)
    cutoff_value = (
        temporal_cutoff_override
        if temporal_cutoff_override is not None
        else config.get("temporal_cutoff")
    )
    temporal_cutoff = (
        TemporalCutoff.model_validate(cutoff_value) if cutoff_value is not None else None
    )
    dataset_path = _configured_path(root, str(config["dataset"]), field="dataset")
    items = [CitationItem.model_validate(row) for row in read_jsonl(dataset_path)]
    citation_dataset_digest = sha256_file(dataset_path)
    document_dataset_value = config.get("document_dataset")
    document_dataset_path = (
        _configured_path(root, str(document_dataset_value), field="document_dataset")
        if document_dataset_value
        else None
    )
    document_dataset_digest = (
        sha256_file(document_dataset_path) if document_dataset_path is not None else None
    )
    benchmark_release_status, dataset_release_tiers, dataset_redistribution_statuses = (
        _v1_release_metadata(
            root,
            citation_dataset_sha256=citation_dataset_digest,
            document_dataset_sha256=document_dataset_digest,
        )
    )
    eligible_items, temporal_filter_report = _apply_temporal_cutoff(items, temporal_cutoff)
    seed = int(config.get("seed", 0))
    selected = _select_items(eligible_items, config.get("max_items"), seed)
    if len(selected) > MAX_SELECTED_RECORDS_PER_RUN:
        raise ValueError(
            f"selected citation count exceeds the {MAX_SELECTED_RECORDS_PER_RUN}-record limit"
        )
    citation_items_by_id = {item.item_id: item for item in items}
    output_root = _configured_path(root, str(config["output_dir"]), field="output_dir")
    index_path = _validate_result_index_destination(
        output_root,
        protected_inputs=[
            config_path,
            dataset_path,
            *([document_dataset_path] if document_dataset_path is not None else []),
        ],
    )
    resume = bool(config.get("resume", True))
    resolved_providers = _resolved_provider_configs(config)
    conditions = _resolved_conditions(config, organic=False)
    scenarios: list[DocumentScenario] = []
    document_filter_report: dict[str, Any] | None = None
    if document_dataset_path is not None:
        scenarios = load_document_scenarios(document_dataset_path)
        unknown_citation_ids = {
            authority.citation_item_id
            for scenario in scenarios
            for authority in scenario.authorities
            if authority.citation_item_id not in citation_items_by_id
        }
        if unknown_citation_ids:
            raise ValueError(
                "Document scenarios reference unknown citation items: "
                f"{sorted(unknown_citation_ids)}"
            )
        scenarios, document_filter_report = _filter_document_scenarios(
            scenarios, {item.item_id for item in eligible_items}
        )
        if not scenarios:
            raise ValueError(
                "Temporal cutoff removed every document scenario because each selected "
                "scenario must contain only eligible citation items"
            )
        maximum_documents = config.get("max_document_scenarios")
        if maximum_documents is not None:
            if int(maximum_documents) < 1:
                raise ValueError("max_document_scenarios must be at least one")
            scenarios = scenarios[: int(maximum_documents)]
        if not scenarios:
            raise ValueError("max_document_scenarios selected zero scenarios")
        document_filter_report["selected_document_count"] = len(scenarios)
        if len(scenarios) > MAX_SELECTED_RECORDS_PER_RUN:
            raise ValueError(
                "selected document scenario count exceeds the "
                f"{MAX_SELECTED_RECORDS_PER_RUN}-record limit"
            )

    planned_live_calls, retry_inclusive_maximum = _require_live_request_budget(
        resolved_providers=resolved_providers,
        logical_calls_per_provider=len(conditions) * (len(selected) + len(scenarios)),
        max_provider_requests=max_provider_requests,
    )
    _require_planned_subruns(
        len(resolved_providers) * len(conditions) * (1 + int(document_dataset_path is not None))
    )
    planned_input_bytes_per_provider = _planned_v1_input_bytes_per_provider(
        items=selected,
        scenarios=scenarios,
        conditions=conditions,
    )
    retry_inclusive_input_ceiling = _require_provider_input_budget(
        resolved_providers=resolved_providers,
        planned_input_bytes_per_provider=planned_input_bytes_per_provider,
        max_provider_input_bytes=max_provider_input_bytes,
    )
    lookup_count_per_tool_run = len(selected) + sum(
        len(scenario.authorities) for scenario in scenarios
    )
    if lookup_count_per_tool_run > MAX_COURTLISTENER_LOOKUPS_PER_RUN:
        raise ValueError(
            "CourtListener lookup count exceeds the per-run "
            f"{MAX_COURTLISTENER_LOOKUPS_PER_RUN}-lookup limit"
        )
    retry_inclusive_courtlistener_ceiling = _require_courtlistener_request_budget(
        resolved_providers=resolved_providers,
        conditions=conditions,
        lookup_count_per_tool_run=lookup_count_per_tool_run,
        max_courtlistener_requests=max_courtlistener_requests,
    )
    resource_budget: dict[str, int | None] = {
        "approved_max_provider_requests": max_provider_requests,
        "retry_inclusive_provider_request_ceiling": retry_inclusive_maximum,
        "approved_max_provider_input_bytes": max_provider_input_bytes,
        "retry_inclusive_provider_input_byte_ceiling": retry_inclusive_input_ceiling,
        "approved_max_courtlistener_requests": max_courtlistener_requests,
        "maximum_courtlistener_lookups_per_run": MAX_COURTLISTENER_LOOKUPS_PER_RUN,
        "runner_courtlistener_response_byte_limit": RUNNER_MAX_COURTLISTENER_RESPONSE_BYTES,
        "maximum_cumulative_checkpoint_write_bytes": MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES,
        "retry_inclusive_courtlistener_request_ceiling": (retry_inclusive_courtlistener_ceiling),
        "per_provider_request_input_byte_limit": MAX_PROVIDER_REQUEST_INPUT_BYTES,
    }

    summaries: list[dict[str, Any]] = []
    for provider_config in resolved_providers:
        for condition in conditions:
            summaries.append(
                _run_one(
                    items=selected,
                    dataset_path=dataset_path,
                    output_root=output_root,
                    provider_name=provider_config["provider"],
                    model=provider_config["model"],
                    condition=condition,
                    seed=seed,
                    resume=resume,
                    config_sha256=config_digest,
                    temporal_cutoff=temporal_cutoff,
                    temporal_filter_report=temporal_filter_report,
                    git_state=git_state,
                    benchmark_release_status=benchmark_release_status,
                    resource_budget=resource_budget,
                )
            )
    if document_dataset_path is not None:
        assert document_filter_report is not None
        for provider_config in resolved_providers:
            for condition in conditions:
                summaries.append(
                    _run_document_one(
                        scenarios=scenarios,
                        citation_items=citation_items_by_id,
                        dataset_path=document_dataset_path,
                        output_root=output_root,
                        provider_name=provider_config["provider"],
                        model=provider_config["model"],
                        condition=condition,
                        seed=seed,
                        resume=resume,
                        config_sha256=config_digest,
                        citation_dataset_sha256=citation_dataset_digest,
                        temporal_cutoff=temporal_cutoff,
                        temporal_filter_report=temporal_filter_report,
                        document_filter_report=document_filter_report,
                        git_state=git_state,
                        benchmark_release_status=benchmark_release_status,
                        resource_budget=resource_budget,
                    )
                )
    index = {
        "schema_version": "sanctionbench.result_index.v1",
        "created_at": utc_now(),
        "config": _path_reference(root, config_path),
        "config_sha256": config_digest,
        "dataset": _path_reference(root, dataset_path),
        "dataset_sha256": citation_dataset_digest,
        "document_dataset": (
            _path_reference(root, document_dataset_path)
            if document_dataset_path is not None
            else None
        ),
        "document_dataset_sha256": document_dataset_digest,
        "benchmark_release_status": benchmark_release_status,
        "dataset_release_tiers": dataset_release_tiers,
        "dataset_redistribution_statuses": dataset_redistribution_statuses,
        "git_state_at_manifest_start": git_state,
        "temporal_filter": temporal_filter_report,
        "selected_citation_item_count": len(selected),
        "selected_document_count": len(scenarios),
        "planned_live_model_call_count": planned_live_calls,
        "retry_inclusive_live_provider_request_ceiling": retry_inclusive_maximum,
        "approved_max_provider_requests": max_provider_requests,
        "planned_provider_input_bytes_per_provider": planned_input_bytes_per_provider,
        "retry_inclusive_live_provider_input_byte_ceiling": retry_inclusive_input_ceiling,
        "approved_max_provider_input_bytes": max_provider_input_bytes,
        "maximum_cumulative_checkpoint_write_bytes": MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES,
        "retry_inclusive_courtlistener_request_ceiling": (retry_inclusive_courtlistener_ceiling),
        "approved_max_courtlistener_requests": max_courtlistener_requests,
        "maximum_courtlistener_lookups_per_run": MAX_COURTLISTENER_LOOKUPS_PER_RUN,
        "runner_courtlistener_response_byte_limit": RUNNER_MAX_COURTLISTENER_RESPONSE_BYTES,
        "provider_request_count": sum(
            int(summary["provider_request_count"]) for summary in summaries
        ),
        "courtlistener_request_count": sum(
            int(summary["courtlistener_request_count"]) for summary in summaries
        ),
        "runs": summaries,
        "live_model_smoke_completed": any(not summary["mock"] for summary in summaries),
    }
    write_json(index_path, index)
    return summaries


def run_organic_manifest(
    config_path: Path,
    *,
    max_provider_requests: int | None = None,
    max_provider_input_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Run the neutral one-call-per-filing organic document track."""

    root = project_root().resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(read_text_bounded(config_path, max_bytes=MAX_CAMPAIGN_CONFIG_BYTES))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: organic benchmark config must be a YAML mapping")
    config_digest = sha256_file(config_path)
    git_state = _git_state(root)
    dataset_path = _configured_path(root, str(config["dataset"]), field="dataset")
    documents = sorted(load_organic_document_gold(dataset_path), key=lambda item: item.item_id)
    dataset_release_tiers = sorted({document.release_tier for document in documents})
    dataset_redistribution_statuses = sorted(
        {document.redistribution_status for document in documents}
    )
    benchmark_release_status = (
        "provisional_private_evaluation"
        if "private_evaluation_only" in dataset_redistribution_statuses
        or "provisional_model_assisted" in dataset_release_tiers
        else "development_public_gold"
    )
    seed = int(config.get("seed", 0))
    maximum_documents = config.get("max_documents")
    if maximum_documents is not None:
        documents = _select_organic_documents(
            documents,
            maximum=int(maximum_documents),
            seed=seed,
        )
    output_root = _configured_path(root, str(config["output_dir"]), field="output_dir")
    index_path = _validate_result_index_destination(
        output_root,
        protected_inputs=[config_path, dataset_path],
    )
    resume = bool(config.get("resume", True))
    repetitions = int(config.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if repetitions > MAX_CAMPAIGN_REPETITIONS:
        raise ValueError(
            f"repetitions cannot exceed the {MAX_CAMPAIGN_REPETITIONS}-repetition limit"
        )
    if len(documents) > MAX_SELECTED_RECORDS_PER_RUN:
        raise ValueError(
            f"selected document count exceeds the {MAX_SELECTED_RECORDS_PER_RUN}-record limit"
        )
    conditions = _resolved_conditions(config, organic=True)
    resolved_providers = _resolved_provider_configs(config)
    planned_live_calls, retry_inclusive_maximum = _require_live_request_budget(
        resolved_providers=resolved_providers,
        logical_calls_per_provider=len(documents) * len(conditions) * repetitions,
        max_provider_requests=max_provider_requests,
    )
    planned_call_count = len(documents) * len(resolved_providers) * len(conditions) * repetitions
    _require_planned_subruns(len(resolved_providers) * len(conditions) * repetitions)
    planned_input_bytes_per_provider = _planned_organic_input_bytes_per_provider(
        documents=documents,
        conditions=conditions,
        repetitions=repetitions,
    )
    retry_inclusive_input_ceiling = _require_provider_input_budget(
        resolved_providers=resolved_providers,
        planned_input_bytes_per_provider=planned_input_bytes_per_provider,
        max_provider_input_bytes=max_provider_input_bytes,
    )
    resource_budget = {
        "approved_max_provider_requests": max_provider_requests,
        "retry_inclusive_provider_request_ceiling": retry_inclusive_maximum,
        "approved_max_provider_input_bytes": max_provider_input_bytes,
        "retry_inclusive_provider_input_byte_ceiling": retry_inclusive_input_ceiling,
        "approved_max_courtlistener_requests": None,
        "retry_inclusive_courtlistener_request_ceiling": 0,
        "per_provider_request_input_byte_limit": MAX_PROVIDER_REQUEST_INPUT_BYTES,
        "maximum_cumulative_checkpoint_write_bytes": MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES,
        "runner_courtlistener_response_byte_limit": None,
    }

    summaries: list[dict[str, Any]] = []
    for provider_config in resolved_providers:
        for condition in conditions:
            for repetition in range(1, repetitions + 1):
                summaries.append(
                    _run_organic_document_one(
                        documents=documents,
                        dataset_path=dataset_path,
                        output_root=output_root,
                        provider_name=str(provider_config["provider"]),
                        model=str(provider_config["model"]),
                        condition=condition,
                        seed=seed,
                        repetition=repetition,
                        resume=resume,
                        config_sha256=config_digest,
                        git_state=git_state,
                        benchmark_release_status=benchmark_release_status,
                        resource_budget=resource_budget,
                    )
                )
    index = {
        "schema_version": "sanctionbench.organic_result_index.v1",
        "created_at": utc_now(),
        "config": _path_reference(root, config_path),
        "config_sha256": config_digest,
        "dataset": _path_reference(root, dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "benchmark_release_status": benchmark_release_status,
        "dataset_release_tiers": dataset_release_tiers,
        "dataset_redistribution_statuses": dataset_redistribution_statuses,
        "git_state_at_manifest_start": git_state,
        "document_count": len(documents),
        "provider_count": len(resolved_providers),
        "condition_count": len(conditions),
        "repetitions": repetitions,
        "planned_model_call_count": planned_call_count,
        "planned_live_model_call_count": planned_live_calls,
        "retry_inclusive_live_provider_request_ceiling": retry_inclusive_maximum,
        "approved_max_provider_requests": max_provider_requests,
        "planned_provider_input_bytes_per_provider": planned_input_bytes_per_provider,
        "retry_inclusive_live_provider_input_byte_ceiling": retry_inclusive_input_ceiling,
        "approved_max_provider_input_bytes": max_provider_input_bytes,
        "maximum_cumulative_checkpoint_write_bytes": MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES,
        "provider_request_count": sum(
            int(summary["provider_request_count"]) for summary in summaries
        ),
        "one_isolated_successful_response_per_document": True,
        "runs": summaries,
        "live_model_smoke_completed": any(not summary["mock"] for summary in summaries),
    }
    write_json(index_path, index)
    return summaries
