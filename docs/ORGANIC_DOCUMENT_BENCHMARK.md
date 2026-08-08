# Organic whole-document benchmark

## Objective

The organic track measures whether a model can find hallucinated or materially misused legal authorities in a complete party filing. Each model receives one neutral, isolated request per document. The sanction order, gold authority inventory, document class, and reviewer evidence are never included in model input.

The planned successful-response budget is explicit:

```text
successful responses = documents * providers * conditions * repetitions
```

The first release supports `closed_book` only. A future tool-assisted track must let the model choose its own lookups. Prefetching evidence for every gold authority would reveal the hidden authority inventory.

## Canonical PDF-to-Markdown input

Provider-native PDF parsers are not used because their extraction behavior differs by vendor and model revision. Every filing is normalized once with [LiteParse](https://github.com/run-llama/liteparse) `2.0.0` in JSON mode. Born-digital PDFs use `--no-ocr`; a cheap local text-layer check selects OCR only when the PDF lacks usable text.

The page array is converted to canonical Markdown with stable markers:

```markdown
<!-- SANCTIONBENCH_PAGE:1 -->

## Page 1

Document text...
```

The gold record preserves the source PDF SHA-256, LiteParse version, OCR decision, raw parser-output SHA-256, canonical Markdown SHA-256, and page-marker contract. The same Markdown bytes are sent to every provider. Parsing is done once per document and the ignored `.md` file is reused on later review-queue builds.

In the private construction repository, the pinned parser is installed and the review queue is built with:

```bash
make install-liteparse
make organic-review-queue
```

The current queue starts from 128 verified component mappings represented by 125 unique PDFs. It contains source-order suggestions and filing citation candidates, but **zero automatic gold labels**.

## Adjudication and gold construction

Automation narrows the review surface; it does not create human review. A release-quality filed document must receive:

1. a complete occurrence-level inventory of every legal authority;
2. an exact document page, citation alias, and excerpt for every occurrence;
3. a court-order page and excerpt for every offending occurrence;
4. an independent substantive classification as `nonexistent_case`, `fabricated_quote`, `misattributed_holding`, or `real`;
5. two distinct reviewers plus separate adjudication, with actor type and locally reverified immutable receipt files and hashes;
6. a redistribution decision; and
7. `authority_inventory_complete=true` only after the entire filing has been checked.

The private construction repository includes a model-assisted pre-curation path:

```bash
sanctionbench run-frontier-model-review --concurrency 2
```

It makes two isolated reviews with different frontier provider/model identities, then gives both reviews to a separate adjudicator. The preferred pinned roles are GPT-5.6 Sol, Claude Opus 5, and a separate GPT-5.6 Sol adjudication call. Reviewer B is restricted to an explicit allowlist: Claude Opus 5, Claude Sonnet 4.6, or GPT-5.6 Terra. A fallback is used only when the preferred Claude tier is unavailable, and the selected provider/model is bound into every receipt and the run summary. One full review pass uses one reviewer-B model consistently; changing it creates new content-addressed receipts instead of mixing identities.

Models receive only the canonical filing, content-addressed candidates, and literal court-source snippets. They cannot use tools. Raw responses are retained in ignored, secret-free receipts; deterministic code copies filing citation, page, and excerpt fields rather than trusting rewritten model text.

Unique one-edit candidate-ID repairs are recorded. A response item ID may be normalized only for a bounded typo or an appended hexadecimal identifier in the single content-addressed call; the unchanged raw response remains in its receipt. A reviewer candidate-coverage failure gets at most one content-addressed full-response correction, with the selected receipt bound to the prior receipt hash. A failed correction, uncertain classification, nonliteral excerpt, unsupported source evidence, or document without an individually supported offending occurrence remains withheld or fails closed rather than receiving a fabricated label.

When repeated source-hint text would exceed a reviewer request limit, the pipeline keeps the complete filing and every literal court excerpt but replaces duplicate copies with content-addressed hint references. Prompt hashes make this compact encoding a distinct resumable call contract. Receipt identity also binds provider, immutable model name, reasoning effort, and the exact reviewer CLI version. Changing any of those fields creates a new receipt instead of silently reusing an older response. The run summary separately hashes the complete local validation and promotion source closure, so deterministic pipeline-code changes stay visible without replaying an unchanged paid model call.

For Terra reviewer-B calls with at least 250 canonical candidates, the pipeline uses a high-volume output schema that still requires one decision for every candidate but defers optional citation aliases, case-name enrichment, and proposition text to adjudication. The complete filing, candidate inventory, and court evidence are unchanged. The compact schema and exact required count are included in the prompt and receipt hashes; exhaustive coverage is validated identically, so a short response fails closed. The tracked resolution queue and summary retain only typed issue/error classes and hashes; raw model-controlled descriptions, backend messages, filing text, and responses remain in ignored private receipts.

Model-assisted records have `human_reviewed=false`, curation method `frontier_model_double_review_with_court_source_adjudication`, and release tier `provisional_model_assisted`. They are useful for a private initial evaluation set, but they are not represented as independent human review or public-release gold.

The gold builder reopens and hashes every filed document and sanction order, verifies excerpts against their one-based pages using the same spatially sorted court-PDF extraction contract used to create source evidence, regenerates canonical Markdown, and rejects incomplete inventories. A release build also requires clean controls. The current controls are transparent constructed authority inventories containing only the 76 public authorities with positive CourtListener existence matches. They carry zero invented document reviewers, `document_origin=constructed_verified_real_control`, and a distinct `constructed_control` release tier. They are not organic filed documents and must be reported separately.

In the private construction repository, the adjudicated dataset is built with:

```bash
sanctionbench build-organic-gold \
  --reviews data/private/frontier-model-review/organic-document-reviews.jsonl \
  --constructed-clean-controls 8 \
  --output data/private/organic_documents.jsonl
```

Full filing text remains private by default. A document may enter a public development set only when its review record says `redistribution_status=cleared_public` and the build uses `--require-public-clearance`.

## Model output

The neutral prompt tells the model that the document may contain no problem. It returns an empty `findings` array or one structured record per suspected authority containing:

- citation text exactly as shown;
- optional case name;
- one-based page number;
- quoted filing text;
- predicted error class;
- calibrated probability; and
- concise rationale.

The exact autonomous wire contract is:

```json
{
  "findings": [
    {
      "citation_text": "string, 1 to 1000 characters",
      "case_name": "string up to 500 characters, or null",
      "page_number": 1,
      "quoted_text": "string, 1 to 1500 characters",
      "predicted_label": "nonexistent_case | fabricated_quote | misattributed_holding | uncertain_needs_review",
      "fake_probability": 0.0,
      "rationale": "string up to 2000 characters"
    }
  ]
}
```

All seven finding fields and the top-level `findings` field are required. A response may contain at most 512 findings. `page_number` is a JSON integer from 1 through 100,000; `fake_probability` is a JSON number between zero and one. An empty `findings` array is valid. Unknown top-level or finding fields, missing fields, numeric strings such as `"1"` or `"0.9"`, `real` findings, and values outside these bounds are invalid. The provider-facing JSON Schema and the local parser are generated from the same strict typed wire models, so prompt documentation cannot silently diverge from enforcement. A malformed response is a failed attempt under the campaign retry policy; it is never coerced into a prediction.

The prompt never says a document is offending. Clean controls are mixed into the same input stream and use the same prompt. The current controls are visibly constructed one-page authority inventories, not naturally clean filed briefs; their false-alarm rate must be reported as a constructed-control diagnostic and must not be generalized to real clean filings.

## Deterministic grading

Reviewer-approved citation aliases are normalized deterministically; no model or fuzzy semantic matcher grades another model. Exact alias plus page is preferred. The grader also recognizes a reviewer-approved alias as a contiguous sequence of at least two normalized tokens inside a fuller model citation, or the reverse. This handles outputs such as a full case caption around the gold reporter citation without edit distance, semantic similarity, or guessed abbreviation expansion. A unique exact or contained alias can still receive detection credit with a wrong-page receipt, while diagnosis and page accuracy are reported separately. Repeated findings for the same gold occurrence are collapsed. Flags against real authorities, unmatched flags, and ambiguous alias flags become extra verifications; ambiguous matches are also surfaced in an adjudication queue and receive no automatic credit. Metrics bind `sanctionbench.organic_matching.v2` so an older grader cannot be mistaken for this matching contract.

The scorecard reports:

- macro OrganicDocumentSanctionScore;
- micro offending-authority recall and false-negative count;
- Clean-Audit Rate across offending documents;
- diagnosis and page accuracy on caught authorities;
- extra verifications and false accusations per 100 real authorities; and
- clean-control pass and false-alarm rates.

## Running models

```bash
SANCTIONBENCH_ORGANIC_DOCUMENT_DATASET=data/private/organic_documents.jsonl \
  sanctionbench run-organic \
    --config configs/organic-campaign.example.yaml \
    --max-provider-requests <three-times-planned-live-calls> \
    --max-provider-input-bytes <retry-inclusive-planned-input-bytes>
```

Runs checkpoint after every document. Their identity includes dataset, selected documents, provider, immutable model name, prompt/output-schema hash, condition, repetition, seed, configuration, an exact executable runner-source-closure hash, provider SDK version, hashed effective endpoint, wire-protocol version, and Markdown input hashes. Git state is recorded separately for inspection. An adapter, source, SDK, or endpoint change therefore cannot silently resume an older checkpoint. Paid live calls remain an explicit human-approved campaign step. DeepSeek is supported as a benchmark contestant through `configs/organic-campaign.deepseek.example.yaml`; it is not a gold reviewer. Its adapter binds an explicit output-token ceiling into provider runtime identity and rejects `finish_reason=length` before attempting to parse truncated JSON.

`--max-provider-requests` is an independent retry-inclusive approval, not a manifest field. Compute `3 * documents * live providers * conditions * repetitions`; the runner validates the entire plan before constructing a live provider client and caps each item at three attempts across resumes. For example, the one-provider DeepSeek manifest over 125 documents requires `375`. Mock-only campaigns need no model-request or provider-input approval flag. Live campaigns also require `--max-provider-input-bytes` at or above the runner's conservative retry-inclusive serialized-input plan; the validation error reports the required value before client construction.

Once an organic run is complete, its finalized identity also binds the release status, request ledger, predictions, metrics, and run-record core. A directory containing `run.json` is immutable: the runner verifies the completion identity and reuses its summary without re-finalizing, overwriting, or calling the provider. This allows later sub-runs in an interrupted manifest to continue. Only an incomplete checkpoint can resume, and every accepted checkpoint row must have a recorded provider invocation. This completion identity, rather than only the resumable pre-run identity, enters the aggregate submission identity.

Offline regrading is derived-data-only. It requires a complete finalized identity and reconciles the preserved request ledger, predictions, historical metrics, and run-record core before scoring. Runs created before finalized identities existed are explicitly legacy evidence and are not upgraded into hash-reconciled regrade receipts.

Completed one-model result indexes can use the ordinary aggregate-only submission packager. The bundle preserves repetition numbers, and leaderboard values are arithmetic means across repetitions. Raw filings, gold inventories, predictions, and prompts are not copied into the bundle.

Every provider request attempt is durably counted immediately before invocation in a secret-free run ledger. Transport, rate-limit, or schema retries therefore increase `provider_request_count`; they are never hidden behind the planned successful-response count. A crash between ledger write and transport can conservatively overcount one request, but cannot undercount a started request. SDK-managed retries are disabled for OpenAI, DeepSeek, and Anthropic, so only the durable outer loop can retry. Each serialized organic prompt is also checked against the 16 MiB request-input limit.
