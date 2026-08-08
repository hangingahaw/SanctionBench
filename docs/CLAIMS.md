# Release claims policy

This document defines the claims that may be made about SanctionBench 1.0.0 and the claims that must never be made.

## Substantiated release claims

- 152 public citation-verification items in 76 matched fake and real pairs.
- 10 constructed document-audit scenarios with 196 authority occurrences.
- 18 separately identified memorization probes.
- Court-source provenance, content hashes, repeated CourtListener search receipts, deterministic grading, resumable provider runs, aggregate-only submissions, and a static development leaderboard.
- A fresh-history allowlisted export that excludes source-acquisition records, downloaded filings, private curation, provider receipts, and credentials.

The generated `data/gold/v1/manifest.json` is authoritative for public dataset counts and hashes.

## Claims not to make

- Do not call the public development dataset a blind or official leaderboard.
- Do not claim that the source population measures hallucination prevalence.
- Do not describe model-assisted whole-document annotations as human-reviewed gold.
- Do not merge constructed controls or scenarios into organic filed-document counts.
- Do not imply that CourtListener non-retrieval proves universal nonexistence.
- Do not announce a public whole-document filing set until redistribution clearance and release validators pass for that exact set.
