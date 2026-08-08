# Contributing

SanctionBench welcomes fixes to the runner, deterministic grading, schemas, documentation, and public development fixtures. Contributions must preserve the benchmark's evidence and privacy boundaries.

Before opening a pull request:

1. Create a focused branch and avoid committing credentials, provider responses, court-document downloads, private predictions, or private curation records.
2. Run `make test`, `make lint`, and `make validate`. Run `make submission-smoke` for changes to the runner, submission packager, or leaderboard; it writes ignored local run data and tracked generated leaderboard output.
3. Regenerate schemas and other derived artifacts when their source models change.
4. Explain behavior changes and add a regression test. Do not relabel an item without primary-source evidence and a documented review receipt.
5. Keep constructed and organic material explicitly separated. Never represent model-generated annotation as human review.

By contributing, you agree that original code and documentation are licensed under Apache-2.0. Third-party data keeps its own legal status and must not be added merely because it is publicly accessible.
