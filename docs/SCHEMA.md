# Schemas

Machine-readable JSON Schemas live under `schemas/` and are generated from strict Pydantic models with `extra="forbid"`.

## Citation item

Each line of `data/gold/v1/citation_items.jsonl` uses `sanctionbench.citation.v1` and contains:

- Stable neutral `item_id` and `matched_pair_id`.
- `citation`, `case_name`, and `proposition` model input.
- Four-way `gold_label` plus binary `real`/`fake` operational label.
- `organic` or `matched_real` track.
- Jurisdiction, decision date, temporal bucket, source case, famous-case tag.
- Honest database temporal fields: nullable entry date, status, and first-observed snapshot.
- Full provenance: database row key, order URL/hash/page/short excerpt, optional offending-filing URL/hash, extraction method.
- Independent verification: exact queries, result counts, matched URLs, response hashes, status, time, and limitations.

Gold-only fields are never included in provider prompts.

## Document scenario

`sanctionbench.document.v1` contains the input `document_text`, a labeled track, a construction manifest, source dates, and gold authority spans. Neutral authority IDs such as `A01` are visible to the model solely for complete deterministic grading.

## Organic document

`sanctionbench.organic_document.v1` is the open-ended whole-filing track. It stores canonical [LiteParse](https://github.com/run-llama/liteparse)-derived Markdown, source PDF and Markdown hashes, parser/OCR receipts, one-based page markers, document class and origin, complete occurrence-level authority gold, court-order evidence for every offender, typed reviewer/adjudicator provenance, curation/release tier, and redistribution status. Filed documents may be human-adjudicated or explicitly provisional model-assisted. Every claimed human or model actor references a local receipt artifact whose SHA-256 is reverified during gold construction; model actors additionally bind provider, model, call, prompt, response, reasoning backend version, and isolation role. Deterministically constructed clean controls carry no invented reviewers and use a separate origin, curation method, adjudicator type, and release tier. Providers receive only the `OrganicDocumentInput` projection: neutral ID/title, Markdown, Markdown hash, and page count.

## Predictions

Citation predictions contain a label, fake probability, rationale, cited evidence, and tool-call receipt. Document predictions contain exactly one assessment per authority ID. The grader rejects missing or extra document IDs.

Organic document predictions are open-ended. Each finding contains the copied citation, optional case name, page, quoted text, flagged class, confidence, and rationale. An empty findings array is a valid clean audit. `real` is not a valid finding label because findings are flags, not a full gold inventory.

The provider wire schemas are also resource boundaries: at most 512 organic findings or document assessments, at most 20 citation-level evidence strings, bounded text fields, and page numbers from 1 through 100,000. Provider adapters reject truncation and nonterminal completion states before local schema validation. Input JSONL and gzip expansion, row, and line sizes are bounded before retention. Model-visible citation, document, and organic text fields and authority collections are bounded, and every final serialized request must fit a 16 MiB provider-independent input ceiling.

## Run record

Every run records provider, immutable model, provider wire-protocol and SDK versions, a one-way hash of the effective endpoint, an executable runner-source-closure hash, condition, task dataset path/hash, seed, result paths, mock/live status, item count, Git revision, prompt/schema hash where applicable, decoding settings, and limitations in usage accounting. Tool-assisted runs additionally record the path, item count, and SHA-256 of the exact secret-free model-visible CourtListener evidence snapshot, plus the path, actual wire-attempt count, and SHA-256 of the durable CourtListener request ledger.

## Validation

```bash
.venv/bin/sanctionbench validate --citations data/gold/v1/citation_items.jsonl
.venv/bin/python scripts/export_schemas.py
```
