# Methodology

## Scope

SanctionBench measures whether a model catches legal authorities that a court found nonexistent, falsely quoted, or used for a proposition the authority does not support. It is not a general legal-reasoning or citation-generation benchmark.

The public benchmark evidence chain is:

```text
Charlotin database row
  -> linked ruling downloaded and hashed
  -> court's exact faulty authority/quote finding transcribed with page
  -> CourtListener v4 opinion search repeated independently
  -> one organic fake paired with one obscure real authority
  -> versioned JSONL item and deterministic score
```

No source narrative becomes gold automatically. Eyecite and PyMuPDF generate review candidates; a reviewed curation row plus an independent search are both required. A failed real-authority resolution or a fake authority that unexpectedly resolves stops the build.

## Tasks

### Citation verification

Input: case name, citation, and proposition. Output: `real`, `nonexistent_case`, `fabricated_quote`, `misattributed_holding`, or `uncertain_needs_review`, plus a calibrated probability that the item is not real.

The binary operational layer treats every label except `real` as a flag. `uncertain_needs_review` therefore creates extra verification workload when applied to a real authority. It does not reduce the safety score, because a review request is not treated as comparable to a missed hallucination.

### Document audit

The first release includes a separate `constructed_from_organic` track. Each brief-formatted authority inventory mixes organic court-found fake items with matched, CourtListener-confirmed real authorities under neutral `[Axx]` identifiers. The text contains no answer labels, and real/fake lines use the same formatting.

The private construction pipeline has acquired organic offending filings from free public sources. They remain quarantined. A filed document enters the provisional whole-document dataset only after two isolated frontier-model reviews, separate model adjudication, exhaustive candidate accounting, literal court-source support for every positive, and fail-closed deterministic validation. Those records are `provisional_model_assisted` with `human_reviewed=false`; unresolved documents are withheld. They are suitable for private pipeline and contestant validation, not human-gold or public holdout claims. The publication-safe repository contains constructed development scenarios and the organic runner/grader but excludes filed documents, annotations, receipts, and private results. An official organic-document holdout still requires human adjudication and source-specific redistribution clearance.

### Memorization probes

Famous-source items, initially Mata v. Avianca, receive separate closed-book prompts asking whether the model recognizes the purported decision and incident. Probe performance is reported separately and never mixed into SanctionScore.

## Conditions

- `closed_book`: the model receives only the task input.
- `tool_assisted`: the runner invokes the same `citation_lookup` tool for each provider and supplies the result. The tool uses CourtListener REST v4 opinion search. Mandatory standardized invocation avoids confounding provider-specific tool-use policies with verification quality. The exact secret-free, model-visible evidence is checkpointed before the provider request, reused on resume, and content-hashed in the run record. Every actual CourtListener wire attempt is independently approved and durably counted before transport, including mock-provider tool runs; cache hits are recorded as evidence reuse and do not increment the wire count.

CourtListener is incomplete. Tool output explicitly says that zero results are evidence rather than universal proof. Public citation gold rests on the source court's finding, human curation, and independent resolution checks. Private whole-document annotations instead use explicitly model-assisted double review and adjudication, remain provisional, and must never be described as human review.

## Gold and distractors

Organic fakes preserve the authority string as presented by the ruling. Where a court identifies the real opinion occupying the reporter location, that authority is a preferred matched distractor. Other distractors are obscure real opinions cited in the same source ruling. Every real item must positively resolve in CourtListener.

Of the 152 first-release citation propositions, 137 test authority existence or identity. The remaining 15 test court-identified quotation or holding misuse. Within the 17 `misattributed_holding` positives, four are authority-identity mismatches and 13 concern proposition-level support. This composition makes false-positive rate meaningful, but it does not yet provide a large matched set of obscure real holdings. Expanding independently verified real proposition pairs is a stated next-wave priority.

## Deterministic grading

All public gold grading is deterministic. Citation predictions map directly to labels. Document assessments map by neutral authority ID. An LLM judge is neither used nor needed. A future semantic-edge track may add a judge only after a human validation set establishes its error rate.

See [SCORING.md](SCORING.md) for the exact headline and supporting metrics.
