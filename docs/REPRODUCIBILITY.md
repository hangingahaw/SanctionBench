# Reproducibility

Clone the published repository, enter its root directory, and run:

```bash
git clone https://github.com/hangingahaw/SanctionBench.git
cd SanctionBench
make install
make validate
make test
make lint
make leaderboard
make audit-dependencies
make wheel
make sdist
make audit-release
```

These commands verify the pristine, content-hashed release payload. Afterward, the optional `make submission-smoke` target makes no paid model calls. It runs a deterministic mock through both task types and both conditions, writes new content-addressed predictions and metrics under the ignored `results/smoke-local/` directory, packages an aggregate submission, and rebuilds the JSON, Markdown, and HTML leaderboards. The immutable bundled evidence in `results/smoke/` is never overwritten. Run the target in a disposable clone if tracked leaderboard changes would be inconvenient; new run identities bind the current Git commit and therefore need not match the bundled construction-time smoke IDs.

`make wheel` fixes `SOURCE_DATE_EPOCH` to a documented normalization value, so repeated builds of the same release payload produce byte-identical wheels. `make sdist` also produces the standard source distribution shipped with a Python release. Override the normalization variable only when intentionally establishing a different release-build epoch.

Live runs durably record a secret-free request-attempt ledger immediately before each provider invocation. Submission `model_query_count` therefore includes retries and resume attempts rather than assuming one provider request per successful logical item. Every non-mock command must receive both `--max-provider-requests` and `--max-provider-input-bytes` at or above the retry-inclusive plans. Tool-assisted commands, including mock campaigns, must also receive `--max-courtlistener-requests`. The runner computes the complete citation-plus-document or organic plan before constructing either client, rejects an absent or insufficient approval, and permits no more than three model attempts per item across resumes. The full three-provider example requires `--max-provider-requests 2916`, `--max-provider-input-bytes 2108017008`, and `--max-courtlistener-requests 12528`. Campaign YAML is capped at 256 KiB before UTF-8 decoding and safe parsing.

Provider manifests may read only the documented provider-specific `SANCTIONBENCH_<PROVIDER>_MODEL` variables; they cannot select arbitrary process environment variables. Authenticated endpoints require HTTPS, and provider credentials are scoped to intended official origins.

Dataset and result paths do not expand arbitrary process variables. Literal paths must be relative to and contained by the project root. External private paths are accepted only through the exact field-specific variables `SANCTIONBENCH_ORGANIC_DOCUMENT_DATASET`, `SANCTIONBENCH_PRIVATE_CITATION_DATASET`, `SANCTIONBENCH_PRIVATE_DOCUMENT_DATASET`, and `SANCTIONBENCH_PRIVATE_RESULTS_DIR`; secret variables such as API keys are rejected as paths before expansion.

Tool-assisted runs also checkpoint the exact secret-free CourtListener evidence supplied to each item before its provider request. Resume reuses that snapshot, `run.json` records its path and SHA-256, and aggregate submissions retain the hash without copying private prompts or predictions. `courtlistener_request_attempts.json` is written before every wire attempt, survives resume, and is also bound into finalized runs and submissions; cache hits do not increment it. Tool runs accept at most 512 lookup identities per sub-run and cap each runner-owned CourtListener response at 256 KiB. Repeated checkpoint replacements are charged in advance to the durable append-only `checkpoint_writes.jsonl` journal and stop before cumulative checkpoint plus journal writes exceed 256 MiB per run. Final metadata records that event count, byte total, and ceiling.

The resumable run identity selects a stable checkpoint directory before inference. A distinct finalized-run identity is produced after scoring and binds the release status, request-attempt ledger, tool-evidence receipt when present, predictions, metrics, and run-record core. Submission IDs bind the complete aggregate bundle and are recomputed during import, so changing a receipt, result-index hash, endpoint type, Git state, or publication field invalidates the bundle.

These SHA-256 identities prove internal consistency, not authorship. A party able to replace an entire local result directory can recompute unkeyed hashes. Treat result directories and partial checkpoints as organizer-controlled state, and use the separate organizer verification process for official results; self-reported bundles never become official merely because their hashes reconcile.

Completed runs are immutable. A repeat invocation verifies an existing finalized identity and then reuses its hash-reconciled summary without rewriting `run.json`, predictions, or metrics and without calling the provider. This permits an interrupted multi-run manifest to continue past completed sub-runs. Partial checkpoint rows require a matching provider-request receipt. Organic offline regrades require the finalized identity and reconcile the preserved request ledger, predictions, historical metrics, and run-record core before writing fresh derived artifacts; legacy unfinalized runs are rejected.

CourtListener cache v3 records bind the exact canonical request, hashed endpoint, and authentication mode. Cache files must be regular no-follow files within a local byte ceiling and are written as bounded compact JSON. Remote response bytes, per-page and aggregate result/page bytes, result counts, and retry delays are bounded; compact model-visible evidence normalizes and caps remote metadata and excludes opinion and snippet text.

SDK-managed retries are disabled for OpenAI, DeepSeek, and Anthropic. The durable runner loop is the only model retry controller. JSONL and gzip expansion, row, line, prompt, evidence-ledger, campaign, submission-file, run-count, YAML-manifest, CourtListener-response, and identifier dimensions have explicit local ceilings.

Live campaigns are never part of `make ci`. Provider keys come only from the process environment. Every run records dataset, subset, config, prompt/schema, model, condition, seed, tool, Git commit, and run-identity hashes. Submission bundles omit raw predictions and private prompts.

This repository cannot reproduce source acquisition or private gold construction by design. Its reproducibility contract begins at the immutable, hashed public release artifacts.
