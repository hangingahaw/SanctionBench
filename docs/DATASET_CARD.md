# Dataset card: SanctionBench 1.0.0

## Summary

SanctionBench 1.0.0 is a development benchmark for detecting hallucinated legal authorities that entered real judicial proceedings. Organic labels originate in court findings indexed by Damien Charlotin's AI Hallucination Cases Database. The public dataset is appropriate for development, regression testing, and methods research, not blind leaderboard claims.

The authoritative counts and hashes are generated in `data/gold/v1/manifest.json`.

## Source population

The 2026-07-11 snapshot contains 1,745 database rows dated 2023-04-14 through 2026-07-10. The United States accounts for 1,206 rows; the next largest are Canada (190), Australia (96), the United Kingdom (60), and Israel (54). The source is therefore not a prevalence sample. It reflects where hallucinations were detected, addressed in a decision, found by the curator, and publicly accessible.

The source's stated inclusion rule is court-found or court-implied use of hallucinated material, with a small alleged-only exception. SanctionBench excludes alleged-only rows.

## First-release composition

- 152 citation items in 76 matched pairs.
- 76 organic court-found positives: 57 nonexistent cases, 2 fabricated quotations in real cases, and 17 misattributed holdings/authorities.
- 76 matched real distractor items, representing 55 unique real authorities.
- 10 separately labeled constructed document scenarios containing 196 authority occurrences.
- 18 separately scored memorization probes from the famous-source subset.
- Organic source split: 43 Powhatan County School Board v. Skinger, 15 Kessler v. City of Atwater, and 18 Mata v. Avianca positives.

The generated manifest remains authoritative for hashes, verification statuses, and future versioned counts.

## Whole-document development lane

The separate whole-document lane begins from 125 acquired party-filed PDFs. Its review queue is not gold and contains no automatic labels. Private model-assisted annotations are explicitly provisional, carry model and receipt provenance, and require a literal court-source excerpt for every offending occurrence. They must not be described as independent human review. Constructed clean controls use the 76 verified-real citation items and are tagged separately from filed-party documents. Full filings remain excluded from the public repository unless separately cleared for redistribution. Because these controls are visibly constructed authority inventories, their false-alarm results do not estimate performance on naturally clean filed briefs.

## Intended use

- Compare calibrated closed-book skepticism with evidence-backed verification.
- Test pre-filing citation audit systems.
- Measure false accusations against obscure real authorities.
- Study contamination and memorization of publicized sanctions incidents.
- Regression-test citation verification tools.

## Out-of-scope use

- Estimating how often lawyers or models hallucinate in the underlying population.
- Treating CourtListener non-retrieval as universal proof that a case does not exist.
- Evaluating non-U.S. authority coverage from the first-release gold subset.
- Claiming a blind held-out score from the public labels.
- Using benchmark output as legal advice.

## Known biases and limitations

- Strong U.S. and recent-year skew in the source population.
- Court-found-only selection captures detected failures and undercounts silent ones.
- The first-release organic gold draws from three high-yield U.S. rulings rather than a random sample.
- Charlotin's export does not provide database entry creation dates. SanctionBench stores `null`, records `not_provided_by_source`, and preserves the first-observed snapshot date instead of substituting the decision date.
- CourtListener coverage is broad but incomplete, especially for state, unpublished, proprietary-citation, and foreign materials.
- Real propositions are mainly identity/existence checks; richer matched true holdings remain future work.
- Full party-filed briefs can carry copyright interests and are quarantined pending publication-specific review.

## Sensitive information

The source concerns public judicial records and can name lawyers, parties, and sanctions. SanctionBench minimizes republication: it stores authority strings, short court-finding excerpts, hashes, and URLs, not full raw rulings or briefs. Maintainers should consider fairness, correction requests, and source updates when publishing results about identifiable people.

## Licensing

The code is Apache-2.0. The source database page's Dataset JSON-LD declares CC0 1.0 for the structured database. That declaration does not relicense linked PDFs or party filings. See [LICENSING.md](LICENSING.md) for the public, dated source evidence and reuse boundary.
