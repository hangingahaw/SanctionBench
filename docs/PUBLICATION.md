# Publication architecture

SanctionBench is maintained as two repositories with different trust boundaries.

## Internal construction repository

This repository remains private. It contains source acquisition, extraction, docket research, human-review ledgers, unresolved-target work, and the quarantined document store. It is the source of truth for producing versioned benchmark releases, but it is not itself the public benchmark.

PACER access and purchasing are prohibited. If an offending filing is not already freely public, the pipeline records only its exact case, docket, and filing-entry locator. Locator records authorize zero access and zero spend and are excluded from the public repository.

## Public benchmark repository

The private construction pipeline creates a fresh-history repository from an explicit allowlist. It contains only:

- the provider-neutral runner, prompts, schemas, and deterministic graders;
- the labeled public development set and constructed document scenarios;
- public-safe dataset validation;
- aggregate-only submission bundles and static leaderboard generation;
- tests, pinned dependencies, CI, GitHub Pages configuration, and publication documentation;
- contributing, conduct, security-reporting, citation, and changelog files; and
- committed mock evidence proving the query-to-score-to-leaderboard loop.

It excludes acquisition and scraping code, population-resolution scripts, raw/interim/curation ledgers, API receipts, docket locator manifests, private annotations, and downloaded filings. The export is a new Git repository rather than a branch or filtered copy, so excluded files do not remain in public Git history.

## Development versus official results

The public labels are development gold. Anybody can run them, inspect the answers, package a self-reported result, and submit that result for a development leaderboard. Those results are never marked official.

An official result requires an organizer-run evaluation against a future private holdout. The organizer freezes the model revision, provider settings, prompt hashes, dataset manifest, query budget, and submitted operating threshold before inference. A vendor supplies a temporary hosted API credential through a secure channel, or an immutable open-weight revision. Secrets never enter the repository or a public submission form.

## Private document storage

Benchmark-ready holdout documents belong in encrypted, access-controlled object storage, keyed by content SHA-256. The private manifest records source URL, retrieval date, case/native docket, filing entry, content and extraction hashes, page count, redistribution status, and release assignment. The evaluator reads the private manifest and sends the document text to the pinned model endpoint; the model provider does not receive a document-download API.

Party-filed documents are not published merely because they are public court filings. Every public development document requires a source-specific redistribution review. Retired private releases may be published only after the same review.

## Deployment shape

The generated public repository uses GitHub as the submission and audit surface and can use GitHub Pages as the read-only portal. Pages deployment is manual-dispatch only; pushing the repository does not authorize or trigger deployment. A submission issue contains metadata only and explicitly forbids credentials. The static leaderboard consumes validated aggregate bundles. This keeps the first release simple and auditable while leaving room for a private scheduled evaluator or Hugging Face Space later.

The Pages job refuses non-`main` refs and runs the complete public CI/release gate before uploading the leaderboard. The release verifier scans every UTF-8 payload regardless of suffix, rejects unclassified binaries, detects tracked files hidden under local build/runtime exclusions, and enforces clean internal source provenance plus holdout exclusion. The development-only `--allow-dirty-source-candidate` flag can check a prepared candidate, but `make audit-release` never uses it and therefore cannot certify a dirty export.

## Authorization-gated release ceremony

Preparing and validating a candidate does not authorize Git or publication actions. After explicit authorization, the internal release ceremony is:

1. Selectively stage and commit only the benchmark-release changes, leaving unrelated construction work unstaged.
2. Create a clean detached worktree or clean clone at that exact internal commit. The shared construction checkout is not an acceptable export source while preserved unrelated work keeps it dirty.
3. Run `scripts/build_public_release.py` from the clean source without `--allow-dirty-source`. Verify that `PUBLIC_EXPORT_MANIFEST.json` records the exact commit and `source_internal_dirty=false`.
4. In the generated candidate, install from `requirements.lock`; validate the dataset and schemas; run tests, Ruff, strict mypy, leaderboard regeneration, dependency-license and vulnerability audits, wheel construction, installed-wheel smoke, and the post-build manifest check. `make audit-release` must reject any dirty-source manifest; do not use the development-candidate override during the release ceremony.
5. Confirm that two normalized wheel builds are byte-identical and archive their SHA-256 plus the public manifest hash in the release evidence.

Creating public Git history, tagging, pushing, publishing a release or DOI, deploying Pages, and performing outreach each remain separate external actions requiring explicit authorization.
