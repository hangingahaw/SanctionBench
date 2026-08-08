from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest
from conftest import make_item

import sanctionbench.runner as runner_module
from sanctionbench.models import Condition, GoldLabel, TemporalCutoff
from sanctionbench.runner import (
    _apply_temporal_cutoff,
    _CheckpointWriteBudget,
    _configured_path,
    _CourtListenerAttemptLedger,
    _git_state,
    _path_reference,
    _prepare_run_directory,
    _RequestAttemptLedger,
    _require_courtlistener_request_budget,
    _require_live_request_budget,
    _require_planned_subruns,
    _require_provider_input_budget,
    _resolved_conditions,
    _resolved_provider_configs,
    _run_identity,
    _ToolEvidenceLedger,
    _v1_release_metadata,
    _validate_result_index_destination,
    run_manifest,
)
from sanctionbench.util import (
    canonical_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)


def test_provider_request_ledger_is_durable_and_identity_bound(tmp_path: Path) -> None:
    path = tmp_path / "request_attempts.json"
    ledger = _RequestAttemptLedger(
        path, identity_sha256="identity-a", selected_ids=["item-a", "item-b"]
    )
    ledger.record_started("item-a")
    ledger.record_started("item-a")

    resumed = _RequestAttemptLedger(
        path, identity_sha256="identity-a", selected_ids=["item-a", "item-b"]
    )
    assert resumed.total == 2
    assert resumed.counts == {"item-a": 2}
    resumed.record_started("item-a")
    with pytest.raises(RuntimeError, match="ceiling exhausted"):
        resumed.record_started("item-a")
    with pytest.raises(ValueError, match="identity differs|budget journal"):
        _RequestAttemptLedger(path, identity_sha256="identity-b", selected_ids=["item-a", "item-b"])


def test_courtlistener_request_ledger_counts_every_wire_attempt_and_caps_lookup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "courtlistener_request_attempts.json"
    ledger = _CourtListenerAttemptLedger(
        path, identity_sha256="identity-a", selected_ids=["item-a"]
    )
    for _ in range(12):
        ledger.record_started("item-a")
    assert ledger.total == 12
    with pytest.raises(RuntimeError, match="ceiling exhausted"):
        ledger.record_started("item-a")
    assert (
        _CourtListenerAttemptLedger(
            path, identity_sha256="identity-a", selected_ids=["item-a"]
        ).total
        == 12
    )
    with pytest.raises(ValueError, match="identity differs|budget journal"):
        _CourtListenerAttemptLedger(path, identity_sha256="identity-b", selected_ids=["item-a"])


def test_tool_evidence_ledger_is_exact_durable_and_identity_bound(tmp_path: Path) -> None:
    path = tmp_path / "tool_evidence.json"
    ledger = _ToolEvidenceLedger(
        path, identity_sha256="identity-a", selected_ids=["item-a", "item-b"]
    )
    evidence = {"query": '"123 F.3d 456"', "matches": []}
    ledger.record("item-a", evidence)
    assert ledger.get("item-a") == evidence
    with pytest.raises(ValueError, match="missing tool evidence"):
        ledger.require_complete()
    with pytest.raises(ValueError, match="changed"):
        ledger.record("item-a", {"query": "different"})

    resumed = _ToolEvidenceLedger(
        path, identity_sha256="identity-a", selected_ids=["item-a", "item-b"]
    )
    assert resumed.get("item-a") == evidence
    resumed.record("item-b", {"authorities": {"A01": evidence}})
    resumed.require_complete()
    assert resumed.count == 2

    with pytest.raises(ValueError, match="identity differs|budget journal"):
        _ToolEvidenceLedger(path, identity_sha256="identity-b", selected_ids=["item-a", "item-b"])


def test_checkpoint_write_budget_is_durable_identity_bound_and_cumulative(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint_writes.jsonl"
    budget = _CheckpointWriteBudget(path, identity_sha256="identity-a")
    budget.reserve(category="fixture", payload_bytes=100)
    first_total = budget.total_bytes
    assert first_total > 100
    assert budget.event_count == 1

    resumed = _CheckpointWriteBudget(path, identity_sha256="identity-a")
    assert resumed.total_bytes == first_total
    resumed.reserve(category="fixture", payload_bytes=50)
    assert resumed.total_bytes > first_total + 50
    assert resumed.event_count == 2
    rows = read_jsonl(path)
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[-1]["cumulative_bytes"] == resumed.total_bytes

    with pytest.raises(ValueError, match="budget journal"):
        _CheckpointWriteBudget(path, identity_sha256="identity-b")


def test_checkpoint_write_budget_fails_before_crossing_cumulative_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "MAX_CUMULATIVE_CHECKPOINT_WRITE_BYTES", 400)
    budget = _CheckpointWriteBudget(
        tmp_path / "checkpoint_writes.jsonl", identity_sha256="identity-a"
    )
    budget.reserve(category="fixture", payload_bytes=50)

    with pytest.raises(ValueError, match="budget is exhausted"):
        budget.reserve(category="fixture", payload_bytes=200)


def test_manifest_model_environment_is_provider_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANCTIONBENCH_OPENAI_MODEL", "gpt-5.6")
    assert (
        _resolved_provider_configs(
            {"providers": [{"provider": "openai", "model_env": "SANCTIONBENCH_OPENAI_MODEL"}]}
        )[0]["model"]
        == "gpt-5.6"
    )

    with pytest.raises(ValueError, match="must be SANCTIONBENCH_OPENAI_MODEL"):
        _resolved_provider_configs(
            {"providers": [{"provider": "openai", "model_env": "OPENAI_API_KEY"}]}
        )
    with pytest.raises(ValueError, match="unsupported characters"):
        _resolved_provider_configs(
            {"providers": [{"provider": "openai", "model": "sk-" + "example-secret-value"}]}
        )


def test_manifest_providers_and_conditions_are_nonempty_known_and_unique() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        _resolved_provider_configs({"providers": []})
    with pytest.raises(ValueError, match="unsupported provider"):
        _resolved_provider_configs(
            {"providers": [{"provider": "custom", "model": "fixture-model"}]}
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        _resolved_provider_configs(
            {"providers": [{"provider": "mock", "model": "fixture-model", "api_key": "prohibited"}]}
        )
    with pytest.raises(ValueError, match="duplicate provider/model"):
        _resolved_provider_configs(
            {
                "providers": [
                    {"provider": "mock", "model": "fixture-model"},
                    {"provider": "mock", "model": "fixture-model"},
                ]
            }
        )
    with pytest.raises(ValueError, match="duplicate-free"):
        _resolved_conditions({"conditions": ["closed_book", "closed_book"]}, organic=False)
    with pytest.raises(ValueError, match="only the closed_book"):
        _resolved_conditions({"conditions": ["tool_assisted"]}, organic=True)
    with pytest.raises(ValueError, match="more than 16"):
        _resolved_provider_configs(
            {
                "providers": [
                    {"provider": "mock", "model": f"fixture-{index}"} for index in range(17)
                ]
            }
        )
    with pytest.raises(ValueError, match="plans 257 sub-runs"):
        _require_planned_subruns(257)


def test_live_request_budget_is_independent_and_retry_inclusive() -> None:
    live = [{"provider": "deepseek", "model": "deepseek-chat"}]
    with pytest.raises(ValueError, match="independent max_provider_requests"):
        _require_live_request_budget(
            resolved_providers=live,
            logical_calls_per_provider=125,
            max_provider_requests=None,
        )
    with pytest.raises(ValueError, match="requires at least 375"):
        _require_live_request_budget(
            resolved_providers=live,
            logical_calls_per_provider=125,
            max_provider_requests=374,
        )
    assert _require_live_request_budget(
        resolved_providers=live,
        logical_calls_per_provider=125,
        max_provider_requests=375,
    ) == (125, 375)
    assert _require_live_request_budget(
        resolved_providers=[{"provider": "mock", "model": "fixture"}],
        logical_calls_per_provider=125,
        max_provider_requests=None,
    ) == (0, 0)


def test_input_and_courtlistener_budgets_are_independent_and_retry_inclusive() -> None:
    live = [{"provider": "deepseek", "model": "deepseek-chat"}]
    with pytest.raises(ValueError, match="independent max_provider_input_bytes"):
        _require_provider_input_budget(
            resolved_providers=live,
            planned_input_bytes_per_provider=1_000,
            max_provider_input_bytes=None,
        )
    assert (
        _require_provider_input_budget(
            resolved_providers=live,
            planned_input_bytes_per_provider=1_000,
            max_provider_input_bytes=3_000,
        )
        == 3_000
    )
    with pytest.raises(ValueError, match="max_courtlistener_requests"):
        _require_courtlistener_request_budget(
            resolved_providers=[{"provider": "mock", "model": "fixture"}],
            conditions=[Condition.TOOL_ASSISTED],
            lookup_count_per_tool_run=2,
            max_courtlistener_requests=None,
        )
    assert (
        _require_courtlistener_request_budget(
            resolved_providers=[{"provider": "mock", "model": "fixture"}],
            conditions=[Condition.TOOL_ASSISTED],
            lookup_count_per_tool_run=2,
            max_courtlistener_requests=24,
        )
        == 24
    )


def test_manifest_rejects_missing_live_budget_before_provider_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "data/gold.jsonl"
    write_jsonl(
        dataset,
        [
            make_item("fake", GoldLabel.NONEXISTENT_CASE, "pair"),
            make_item("real", GoldLabel.REAL, "pair"),
        ],
    )
    config = tmp_path / "campaign.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: data/gold.jsonl",
                "output_dir: results/run",
                "providers:",
                "  - provider: openai",
                "    model: fixture-model",
                "conditions: [closed_book]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_module, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runner_module,
        "_git_state",
        lambda root: {
            "commit": "fixture",
            "dirty": False,
            "status_entry_count": 0,
            "status_sha256": sha256_bytes(b""),
        },
    )

    def prohibited_provider_construction(provider_name: str, model: str) -> None:
        raise AssertionError(f"provider constructed unexpectedly: {provider_name}/{model}")

    monkeypatch.setattr(runner_module, "create_provider", prohibited_provider_construction)

    with pytest.raises(ValueError, match="before any provider client is created"):
        run_manifest(config)

    with pytest.raises(ValueError, match="max_provider_input_bytes"):
        run_manifest(config, max_provider_requests=6)


def test_manifest_rejects_oversized_yaml_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text("dataset: data/gold.jsonl\n", encoding="utf-8")
    monkeypatch.setattr(runner_module, "project_root", lambda: tmp_path)
    monkeypatch.setattr(runner_module, "MAX_CAMPAIGN_CONFIG_BYTES", 8)

    with pytest.raises(ValueError, match="text file exceeds"):
        run_manifest(config)


def test_tool_manifest_rejects_missing_courtlistener_budget_before_provider_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "data/gold.jsonl"
    write_jsonl(
        dataset,
        [
            make_item("fake", GoldLabel.NONEXISTENT_CASE, "pair"),
            make_item("real", GoldLabel.REAL, "pair"),
        ],
    )
    config = tmp_path / "campaign.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: data/gold.jsonl",
                "output_dir: results/run",
                "providers:",
                "  - provider: mock",
                "    model: fixture-model",
                "conditions: [tool_assisted]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_module, "project_root", lambda: tmp_path)

    def prohibited_provider_construction(provider_name: str, model: str) -> None:
        raise AssertionError(f"provider constructed unexpectedly: {provider_name}/{model}")

    monkeypatch.setattr(runner_module, "create_provider", prohibited_provider_construction)
    with pytest.raises(ValueError, match="max_courtlistener_requests"):
        run_manifest(config)


def test_finalized_run_directory_is_verified_and_immutable(tmp_path: Path) -> None:
    identity = {"schema_version": "fixture", "selected_ids": ["item-a"]}
    identity_sha256 = sha256_bytes(canonical_json(identity).encode("utf-8"))
    run_id, run_dir, already_finalized = _prepare_run_directory(
        output_root=tmp_path,
        run_name="fixture",
        identity=identity,
        identity_sha256=identity_sha256,
    )
    assert already_finalized is False
    attempts_path = run_dir / "request_attempts.json"
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    attempts_path.write_text("{}\n", encoding="utf-8")
    predictions_path.write_text('{"item_id":"item-a"}\n', encoding="utf-8")
    metrics_path.write_text("{}\n", encoding="utf-8")
    run: dict[str, object] = {
        "run_id": run_id,
        "metadata": {
            "run_identity_sha256": identity_sha256,
            "provider_request_count": 1,
            "successful_response_count": 1,
        },
    }
    finalized = {
        "schema_version": "sanctionbench.finalized_run_identity.v2",
        "benchmark_release_status": "development_public_gold",
        "run_identity_sha256": identity_sha256,
        "provider_request_count": 1,
        "successful_response_count": 1,
        "provider_request_attempts_sha256": sha256_file(attempts_path),
        "courtlistener_request_attempts_sha256": None,
        "tool_evidence_sha256": None,
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "run_record_core_sha256": sha256_bytes(canonical_json(run).encode("utf-8")),
    }
    metadata = run["metadata"]
    assert isinstance(metadata, dict)
    metadata["finalized_run_identity"] = finalized
    metadata["finalized_run_identity_sha256"] = sha256_bytes(
        canonical_json(finalized).encode("utf-8")
    )
    write_json(run_dir / "run.json", run)

    resumed_run_id, resumed_run_dir, already_finalized = _prepare_run_directory(
        output_root=tmp_path,
        run_name="fixture",
        identity=identity,
        identity_sha256=identity_sha256,
    )
    assert (resumed_run_id, resumed_run_dir, already_finalized) == (run_id, run_dir, True)

    predictions_path.write_text('{"item_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not reconcile"):
        _prepare_run_directory(
            output_root=tmp_path,
            run_name="fixture",
            identity=identity,
            identity_sha256=identity_sha256,
        )


def test_private_dataset_paths_expand_environment_without_entering_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private" / "citation.jsonl"
    monkeypatch.setenv("SANCTIONBENCH_ORGANIC_DOCUMENT_DATASET", str(private))
    assert (
        _configured_path(
            tmp_path,
            "${SANCTIONBENCH_ORGANIC_DOCUMENT_DATASET}",
            field="dataset",
        )
        == private
    )
    monkeypatch.delenv("SANCTIONBENCH_ORGANIC_DOCUMENT_DATASET")
    with pytest.raises(RuntimeError, match="requires the configured"):
        _configured_path(
            tmp_path,
            "${SANCTIONBENCH_ORGANIC_DOCUMENT_DATASET}",
            field="dataset",
        )


def test_manifest_paths_reject_secret_expansion_and_literal_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "".join(("s", "k-", "example-must-never-appear"))
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    with pytest.raises(ValueError, match="unsupported path environment reference") as caught:
        _configured_path(tmp_path, "${OPENAI_API_KEY}", field="output_dir")
    assert secret not in str(caught.value)
    assert not (tmp_path / secret).exists()

    with pytest.raises(ValueError, match="must be relative"):
        _configured_path(tmp_path, str(tmp_path / "absolute.jsonl"), field="dataset")
    with pytest.raises(ValueError, match="escapes the project root"):
        _configured_path(tmp_path, "../outside.jsonl", field="dataset")

    external_output = tmp_path.parent / "private-results"
    monkeypatch.setenv("SANCTIONBENCH_PRIVATE_RESULTS_DIR", str(external_output))
    assert (
        _configured_path(
            tmp_path,
            "${SANCTIONBENCH_PRIVATE_RESULTS_DIR}",
            field="output_dir",
        )
        == external_output.resolve()
    )


def test_external_dataset_path_is_representable_without_relative_path_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    external = tmp_path / "mounted-private" / "organic.jsonl"

    assert _path_reference(root, root / "data/gold.jsonl") == "data/gold.jsonl"
    assert _path_reference(root, external) == str(external.resolve())


def test_result_index_must_not_alias_an_input_or_existing_file(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    protected = output_root / "index.json"
    with pytest.raises(ValueError, match="must not overwrite"):
        _validate_result_index_destination(output_root, protected_inputs=[protected])

    output_root.mkdir()
    protected.write_text("preserve me\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists and is immutable"):
        _validate_result_index_destination(output_root, protected_inputs=[])


def _pair(prefix: str, observed: str, database_entry: str | None = None):
    fake = make_item(f"{prefix}-a", GoldLabel.NONEXISTENT_CASE, prefix).model_copy(
        update={
            "first_observed_snapshot_date": observed,
            "database_entry_date": database_entry,
            "database_entry_date_status": (
                "provided" if database_entry else "not_provided_by_source"
            ),
        }
    )
    real = make_item(f"{prefix}-b", GoldLabel.REAL, prefix).model_copy(
        update={
            "first_observed_snapshot_date": observed,
            "database_entry_date": database_entry,
            "database_entry_date_status": (
                "provided" if database_entry else "not_provided_by_source"
            ),
        }
    )
    return [fake, real]


def test_temporal_cutoff_is_exclusive_and_preserves_pairs() -> None:
    items = [
        *_pair("before", "2025-01-01"),
        *_pair("exact", "2025-06-01"),
        *_pair("after", "2025-06-02"),
    ]
    cutoff = TemporalCutoff(
        field="first_observed_snapshot_date",
        cutoff_date="2025-06-01",
    )

    selected, report = _apply_temporal_cutoff(items, cutoff)

    assert {item.matched_pair_id for item in selected} == {"after"}
    assert len(selected) == 2
    assert report["eligible_item_count"] == 2
    assert report["excluded_on_or_before_cutoff_item_count"] == 4
    assert "strictly after" in report["semantics"]


def test_database_entry_cutoff_missing_policy_is_explicit() -> None:
    items = [*_pair("missing", "2026-01-01"), *_pair("dated", "2026-01-01", "2026-01-02")]
    cutoff = TemporalCutoff(
        field="database_entry_date",
        cutoff_date="2026-01-01",
        missing="exclude",
    )

    selected, report = _apply_temporal_cutoff(items, cutoff)

    assert {item.matched_pair_id for item in selected} == {"dated"}
    assert report["excluded_missing_date_item_count"] == 2

    with pytest.raises(ValueError, match="missing under missing=error"):
        _apply_temporal_cutoff(items, cutoff.model_copy(update={"missing": "error"}))


def test_temporal_cutoff_normalizes_yaml_date_objects() -> None:
    cutoff = TemporalCutoff.model_validate(
        {
            "field": "first_observed_snapshot_date",
            "cutoff_date": date(2025, 12, 31),
        }
    )
    assert cutoff.cutoff_date == "2025-12-31"


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("dataset_sha256", "dataset-b"),
        ("subset_sha256", "subset-b"),
        ("selected_ids", ["item-b"]),
        ("config_sha256", "config-b"),
        ("provider_name", "anthropic"),
        ("provider_protocol_version", "anthropic-messages-tool-schema-v1"),
        ("provider_runtime_identity", {"protocol_version": "changed"}),
        ("runtime_source_sha256", "source-b"),
        ("model", "model-b"),
        ("condition", Condition.TOOL_ASSISTED),
        ("prompt_sha256", "prompt-b"),
        ("resource_budget", {"approved_max_provider_requests": 6}),
    ],
)
def test_run_identity_changes_for_every_compatibility_dimension(
    override: str, value: object
) -> None:
    base = {
        "task_type": "citation_verification",
        "dataset_sha256": "dataset-a",
        "subset_sha256": "subset-a",
        "selected_ids": ["item-a"],
        "config_sha256": "config-a",
        "provider_name": "openai",
        "provider_protocol_version": "openai-chat-completions-json-schema-v1",
        "provider_runtime_identity": {
            "protocol_version": "openai-chat-completions-json-schema-v1",
            "sdk_version": "fixture",
            "endpoint_sha256": "endpoint-a",
        },
        "runtime_source_sha256": "source-a",
        "model": "model-a",
        "condition": Condition.CLOSED_BOOK,
        "seed": 7,
        "prompt_sha256": "prompt-a",
        "temporal_cutoff": None,
        "resource_budget": {"approved_max_provider_requests": 3},
    }
    _, baseline_digest = _run_identity(**base)  # type: ignore[arg-type]
    changed = {**base, override: value}
    _, changed_digest = _run_identity(**changed)  # type: ignore[arg-type]
    assert changed_digest != baseline_digest


def test_v1_release_metadata_fails_closed_without_matching_manifest(tmp_path: Path) -> None:
    private = _v1_release_metadata(
        tmp_path,
        citation_dataset_sha256="citation",
        document_dataset_sha256="documents",
    )
    assert private == (
        "provisional_private_evaluation",
        ["unclassified_external"],
        ["private_evaluation_only"],
    )

    manifest = tmp_path / "data/gold/v1/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "release_status": "development_public_gold",
                "citation_items_sha256": "citation",
                "document_scenarios_sha256": "documents",
            }
        ),
        encoding="utf-8",
    )
    public = _v1_release_metadata(
        tmp_path,
        citation_dataset_sha256="citation",
        document_dataset_sha256="documents",
    )
    assert public == (
        "development_public_gold",
        ["public_development_gold"],
        ["cleared_public"],
    )


def test_git_state_records_commit_and_dirty_boolean(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    clean = _git_state(tmp_path)
    assert clean["commit"] != "UNAVAILABLE"
    assert clean["dirty"] is False
    assert clean["status_entry_count"] == 0

    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = _git_state(tmp_path)
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True
    assert dirty["status_entry_count"] == 1
    assert dirty["status_sha256"] != clean["status_sha256"]
