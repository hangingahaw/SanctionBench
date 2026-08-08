# Contamination policy

The underlying incidents are public and some, especially Mata v. Avianca, are heavily discussed. A frontier model may recognize a fake authority from training data instead of verifying it.

SanctionBench addresses, but cannot eliminate, this threat:

1. Every item retains the source decision date and first-observed database snapshot date.
2. The absent database-entry creation date is `null`, never backfilled with a more convenient date.
3. Campaigns can filter on decision date or first-observed snapshot relative to a documented model cutoff.
4. Famous incidents have a separate memorization-probe track.
5. Each organic fake has a matched obscure real distractor so familiarity is not a safe classifier.
6. Organic and constructed tracks are separate. Constructed whole-document clean controls carry `document_origin=constructed_verified_real_control` and are reported independently; they are never counted as filed-party documents or organic positives.
7. Release hashes make silent item replacement detectable.

Public gold is a development set, and the verified-real authorities used in constructed controls are also public. A credible leaderboard requires a future private, time-forward evaluation set with immutable release timestamps and third-party audit. Model-assisted curation also contaminates every curator model revision recorded in those records' provenance metadata, including GPT-5.6 Sol, GPT-5.6 Terra, Claude Opus 5, and Claude Sonnet 4.6. Results from a curator model revision must not be reported against records that revision reviewed. Hiding labels is helpful but not sufficient; temporal evaluation and post-cutoff collection remain essential.
