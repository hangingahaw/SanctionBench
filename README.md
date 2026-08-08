# SanctionBench

**Can an AI catch a hallucinated legal authority before it reaches a court, without calling a real case fake?**

Courts in multiple jurisdictions have sanctioned lawyers and litigants, or imposed other public consequences, after filings cited cases that did not exist, quoted language no court wrote, or attributed a holding to the wrong authority. SanctionBench is named for the consequence, but it is designed around the preventable failure: a bad citation survived research, drafting, and final review before becoming part of the judicial record.

SanctionBench turns documented failures into an open, evidence-traceable benchmark for legal citation verification and pre-filing document audit.

> **Release status:** SanctionBench 1.0.0 is the first public release. Its development-set labels are visible, so it supports engineering and reproducible comparison rather than blind evaluation. Official evaluation will require an organizer-run private, time-forward holdout. Mock, self-reported, and organizer-run results are kept visibly separate.

## Why this benchmark is necessary

Legal writing does not become reliable because it sounds plausible. A citation must resolve to a real authority, a quotation must appear in the source, and the authority must support the proposition for which it is offered. The failures documented by courts show that fluent systems can break at each of those steps.

The source research snapshot behind SanctionBench comes from [Damien Charlotin's AI Hallucination Cases Database](https://www.damiencharlotin.com/hallucinations/) and contains 1,745 rows dated from April 2023 through July 2026. That figure shows that the problem is recurrent and international. It is not a count of sanctioned lawyers, a prevalence estimate, or a model failure rate. The database records incidents that were detected, addressed in a decision, found by the curator, and publicly accessible.

Most legal AI benchmarks ask a model to answer a legal question, summarize a supplied authority, or reason from a fixed record. Those are important capabilities, but they do not answer the operational question that matters at filing time: can a system find an authority problem in work that otherwise looks polished?

That question has two sides. A missed hallucination can expose a client, lawyer, or court to serious cost. An overaggressive checker can bury reviewers in false alarms and wrongly cast doubt on an obscure but genuine case. A checker that catches fabrications by distrusting every unfamiliar authority is not safe. It has merely exchanged one verification failure for another.

## What makes SanctionBench different

- **Court-found failures:** Public positive items begin with a court's own finding, not a synthetic error or a model's unsupported guess.
- **Matched real authorities:** Every court-found problem is paired with a verified real authority, so blanket suspicion is measurable rather than rewarded.
- **Citation and document tasks:** Systems are tested on individual authority claims and on brief-formatted authority inventories containing many authorities.
- **Evidence-traceable labels:** Public items include source URLs, short court-finding excerpts, hashes, verification metadata, and explicit limitations.
- **Reproducible evaluation:** Versioned schemas, provider-neutral prompts, deterministic grading, resumable runs, and aggregate submission bundles make results inspectable.
- **Open implementation:** The public release includes the runnable evaluator, data contracts, tests, release manifest, and deterministic leaderboard builder rather than only selected examples or reported scores.

## What is included

| Evaluation surface | First public release |
| --- | ---: |
| Citation verification | 152 items in 76 matched fake and real pairs |
| Court-found positive items | 76 total: 57 nonexistent cases, 2 fabricated quotations, and 17 misattributed holdings or authorities |
| Matched real items | 76 items representing 55 unique real authorities |
| Constructed document audit | 10 scenarios with 196 authority occurrences |
| Memorization analysis | 18 separately scored probes |

The public development data is intentionally inspectable. It supports prompt development, pipeline testing, error analysis, and reproducible research. It should not be presented as a secret test set or used to make claims about unobserved generalization.

See [the dataset card](docs/DATASET_CARD.md), [methodology](docs/METHODOLOGY.md), and [scoring contract](docs/SCORING.md) for the complete data and evaluation definitions.

## Tasks and scoring

- **Citation verification:** Classify a supplied authority and proposition as real, nonexistent, falsely quoted, misattributed, or uncertain and requiring review.
- **Document audit:** Assess every enumerated authority in a constructed brief-formatted authority inventory.
- **Organic document audit:** Find suspect authorities in a complete neutral filing without being given the authority inventory.
- **Evaluation conditions:** Run closed-book or with organizer-standardized CourtListener evidence.

SanctionScore measures hallucination recall, so only missed offending authorities reduce it. DocumentSanctionScore macro-averages recall across documents. Clean-Audit Rate requires zero misses in a document. False positives are reported separately as added verification workload and break ties between equally safe systems. This keeps the benchmark focused on catching dangerous errors while making indiscriminate flagging costly and visible.

## Quick start

SanctionBench requires Python 3.11, 3.12, or 3.13. The Makefile defaults to `python3.12`; override it with `make install PYTHON=python3.11` or `make install PYTHON=python3.13` when needed.

```bash
git clone https://github.com/hangingahaw/SanctionBench.git
cd SanctionBench
make install
make validate
make schemas
make test
make lint
make leaderboard
make audit-dependencies
make audit-release
```

These commands validate the release payload without making model API calls. The optional end-to-end smoke test writes fresh run artifacts under the ignored `results/smoke-local/` directory and creates a content-addressed aggregate submission. It never overwrites the immutable bundled evidence in `results/smoke/`. Because it also rebuilds the tracked leaderboard, run it afterward or in a disposable clone if a clean working tree matters:

```bash
make submission-smoke
```

## Running models

Live runs require environment-only provider credentials and explicit request, input-byte, and tool budgets. Review provider pricing before approving a run.

```bash
.venv/bin/sanctionbench run \
  --config configs/campaign.example.yaml \
  --max-provider-requests 2916 \
  --max-provider-input-bytes 2108017008 \
  --max-courtlistener-requests 12528
```

The complete public campaign plans 324 successful responses per model: 152 citation items and 10 document scenarios, each under closed-book and standardized tool-assisted conditions. Across the three-provider example, that is 972 planned successful responses. The approval ceilings permit no more than 2,916 provider attempts and about 1.96 GiB of retry-inclusive serialized input. The 18 published memorization probes are auxiliary and excluded from that count. Retries are durably counted, process resumes cannot reset the ceilings, and an absent or insufficient approval fails before inference.

An organizer-cleared organic document dataset can be evaluated with one neutral, isolated successful response per canonical Markdown filing:

```bash
.venv/bin/sanctionbench validate-organic --dataset <organic-document-gold.jsonl>
.venv/bin/sanctionbench run-organic \
  --config configs/organic-campaign.example.yaml \
  --max-provider-requests <three-times-planned-live-calls> \
  --max-provider-input-bytes <retry-inclusive-planned-input-bytes>
```

The organizer supplies that dataset separately only after its filings are cleared for the stated release. The public runner never receives sanction-order evidence or hidden authority inventories. See [the organic benchmark contract](docs/ORGANIC_DOCUMENT_BENCHMARK.md) for the release and provenance requirements.

## Interpreting results responsibly

- The source population measures detected and published incidents, not the prevalence of legal hallucinations or misconduct.
- The public dataset is development gold. Model developers can inspect it, so strong public scores may reflect tuning or memorization.
- Constructed document scenarios are labeled as constructed and are not filed legal documents.
- CourtListener coverage is broad but incomplete. Non-retrieval alone is not universal proof that an authority does not exist. Court findings provide the primary labels for positive items.
- False positives matter. They represent extra lawyer review and can turn a verification tool into a source of misplaced confidence or accusation.
- The future organic holdout must retain explicit human-review provenance. Model-assisted annotations must never be described as independent human review.
- Results should evaluate systems, not rank or sensationalize identifiable lawyers, parties, or courts.

## Evidence, privacy, and release boundaries

The source material concerns public judicial records and can name lawyers, parties, and sanctions. The public dataset minimizes republication. It stores authority strings, short court-finding excerpts, hashes, and source URLs rather than full raw rulings or briefs.

This publication-safe repository includes the runner, prompts, schemas, deterministic grader, cleared structured development data, and aggregate leaderboard artifacts. It excludes source scraping, acquisition records, docket research, private curation, provider receipts, downloaded party filings, and credentials. The public export is built from an explicit allowlist in a fresh Git history.

See [licensing](docs/LICENSING.md), [reproducibility](docs/REPRODUCIBILITY.md), [publication policy](docs/PUBLICATION.md), and the [release claims policy](docs/CLAIMS.md) before redistributing data or reporting results.

## Third-party submissions

See [the submission guide](docs/SUBMISSIONS.md). Public users can package aggregate development results. Official results require an organizer-run private evaluation. Never put model-provider credentials, raw provider responses, private filings, or private annotations in an issue, pull request, submission bundle, or repository file.

## Repository map

```text
configs/                 frozen run manifests
data/gold/v1/            public development gold
docs/                    task, scoring, licensing, and submission contracts
leaderboard/             aggregate bundles plus JSON, Markdown, and HTML views
results/smoke/           immutable bundled zero-cost evidence
schemas/                 versioned public artifact schemas
src/sanctionbench/       runner, providers, graders, and submission tooling
tests/                   offline regression suite
```

The private construction repository is maintained separately and has no shared Git history with the public export.

## License and citation

Code is released under the Apache License 2.0. Dataset fields retain the source-specific terms and limitations described in [docs/LICENSING.md](docs/LICENSING.md). Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The public development leaderboard is available at [hangingahaw.github.io/SanctionBench](https://hangingahaw.github.io/SanctionBench/).
