# Security policy

## Supported version

Once a release is published, only the latest published SanctionBench release is supported.

## Reporting a vulnerability

Use GitHub private vulnerability reporting on the published repository for credential exposure, arbitrary code execution, path traversal, unsafe archive handling, or leakage of private holdout material. If that channel is unavailable, open a public issue requesting a secure contact channel without including sensitive details or proof-of-concept secrets.

Do not place provider credentials, private filings, raw model outputs, or exploit details in a public issue or pull request. Ordinary correctness bugs that do not expose sensitive material can use the normal issue tracker.

Run manifests are data, not trusted code, but should still come from a reviewed source. SanctionBench allowlists provider-specific model environment variables, scopes authenticated clients to HTTPS official origins, treats CourtListener evidence as untrusted remote data, and refuses mutation or regrading of completed runs whose finalized identity does not reconcile. It bounds YAML before parsing and persists a cumulative checkpoint-write budget across resumes so nominally bounded campaigns cannot amplify local serialization without limit.

Run and submission hashes are unkeyed consistency receipts, not signatures. Result directories and partial checkpoints are organizer-controlled state; an actor able to replace the whole directory can recompute those hashes. Official status therefore requires a separate organizer-run verification and cannot be self-asserted by a hash-valid submission.
