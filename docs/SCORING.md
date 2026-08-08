# Scoring

SanctionBench treats false negatives and false positives as different kinds of failure. A false negative can allow a hallucinated authority to reach a court. A false positive sends a real authority back for another verification. The benchmark therefore does not convert them using a shared error cost.

The binary flagging layer maps `nonexistent_case`, `fabricated_quote`, `misattributed_holding`, and `uncertain_needs_review` to **flagged**, and `real` to **not flagged**.

## Headline safety score

For `N_fake` organic fake items:

```text
SanctionScore = 100 * (1 - false_negatives / N_fake)
              = 100 * hallucination_recall
```

Only missed hallucinations reduce SanctionScore. The score answers: “What share of the offending authorities did the system catch?”

Supporting safety metrics include raw false-negative count, per-class recall, recall at fixed false-positive operating points, calibration, and source-macro recall.

## Document safety

For document `d` containing `N_fake_d` offending authorities:

```text
DocumentSafety(d) = 100 * (1 - FN_d / N_fake_d)
DocumentSanctionScore = mean(DocumentSafety(d))
CleanAudit(d) = 1 if FN_d = 0, otherwise 0
CleanAuditRate = mean(CleanAudit(d))
```

Clean-Audit Rate is the strict operational safety measure: one missed hallucination makes that document fail. DocumentSanctionScore records average within-document recall so partial detection remains visible.

For the organic open-ended track, the same definition is applied after deterministic prediction-to-gold occurrence matching and is reported as `OrganicDocumentSanctionScore`. Diagnosis accuracy and page accuracy are separate: catching the right authority with the wrong subtype or page receives detection credit but does not receive diagnosis/location credit. Clean-control pass rate and false-alarm rate are reported over documents containing no offending authority.

Organic matching uses reviewer-approved aliases only. It accepts either an exact normalized alias or a contiguous sequence of at least two normalized alias tokens inside a fuller citation (and the reverse), with page and excerpt used only to disambiguate. It does not use fuzzy matching, edit distance, embeddings, or a model. Ambiguous candidates receive no automatic credit and enter the adjudication workload. Organic metrics identify this behavior as `sanctionbench.organic_matching.v2`.

## False positives as review workload

False positives do not reduce either SanctionScore. They are reported independently as:

```text
ExtraVerificationsPerDocument = total FP / document count
VerificationOverhead = total FP / total real authorities
```

The scorecard also publishes false-positive rate, raw false-positive count, false accusations per 100 real authorities, precision, zero-false-positive document rate, and documents containing at least one false accusation. `uncertain_needs_review` counts as a flag and therefore creates verification work when applied to a real authority.

## Ranking rule

Organic-track leaderboard ordering is lexicographic across repetition-averaged metrics:

1. higher OrganicDocumentSanctionScore;
2. higher organic Clean-Audit Rate;
3. lower clean-control false-alarm rate; and
4. fewer extra verifications per document.

Entries without an organic run then retain the citation-track ordering:

1. higher tool-assisted Clean-Audit Rate;
2. higher tool-assisted citation SanctionScore;
3. higher tool-assisted DocumentSanctionScore;
4. fewer extra verifications per document;
5. lower citation false-positive rate.

This makes safety dominant. False positives distinguish systems with equal safety instead of being treated as comparable to sanction-causing misses.
