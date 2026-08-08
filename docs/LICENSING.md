# Licensing and data boundary

Original SanctionBench software and documentation are licensed under Apache License 2.0. That license does not relicense third-party material represented in benchmark provenance.

The public development release contains structured authority facts, propositions, short judicial finding excerpts, source links, hashes, and constructed document scenarios. It contains no bundled party-filed PDFs or full extracted filings.

Federal court-authored orders are generally U.S. government works. Party-filed briefs present a more complicated copyright question; public availability is not treated as blanket republication permission. Private or retired organic-document releases require document-specific review before publication. State and non-U.S. source reuse remains source- and jurisdiction-specific.

## Public source-license evidence

These source statements were rechecked on 2026-08-07 and are links, not a transfer of rights:

- The [AI Hallucination Cases Database](https://www.damiencharlotin.com/hallucinations/) exposes a Dataset JSON-LD license value of CC0 1.0 for its structured database. SanctionBench does not treat that metadata as permission to republish linked rulings or party filings.
- Free Law Project states that its [CourtListener bulk data](https://wiki.free.law/c/courtlistener/help/api/bulk-data/bulk-legal-data) are free of known copyright restrictions and marks them with the Public Domain Mark. That statement concerns the bulk data and does not erase source-specific rights in contributed files.
- Free Law Project separately explains that [third-party court filings may be copyrighted](https://wiki.free.law/c/courtlistener/help/recap/is-sharing-court-documents-a-violation-of-copyright-law) and describes the supporting fair-use case law as sparse. SanctionBench therefore keeps full party-filed documents private unless a document-specific release review clears them.

The public export manifest identifies the exact internal commit and SHA-256 of every exported file. It does not expose the internal acquisition repository or imply that users may reconstruct private holdout material from an evaluator API.

The public package has its own transitive `requirements.lock`; it does not reuse the internal construction environment. `THIRD_PARTY_LICENSES.json` records the installed license metadata for every pinned public dependency, and `make audit-dependencies` fails on a missing/version-mismatched, unreviewed, AGPL, GPL, or LGPL dependency. PyMuPDF is used only by the private construction pipeline and is intentionally absent from the public package and lockfile.
