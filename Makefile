PYTHON ?= python3.12
SOURCE_DATE_EPOCH ?= 946684800

.PHONY: install validate schemas smoke submission-smoke leaderboard audit-release audit-dependencies test lint wheel sdist ci

install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -r requirements.lock
	.venv/bin/python -m pip install --no-deps --no-build-isolation -e .

validate:
	.venv/bin/sanctionbench validate

schemas:
	.venv/bin/python scripts/export_schemas.py

smoke:
	.venv/bin/sanctionbench run --config configs/smoke.yaml --max-courtlistener-requests 528

submission-smoke: smoke
	.venv/bin/sanctionbench package-submission --results results/smoke-local/index.json --submitter SanctionBench --organization SanctionBench --model-revision deterministic-rule-based-v1 --endpoint-type mock
	$(MAKE) leaderboard

leaderboard:
	.venv/bin/sanctionbench build-leaderboard

audit-release:
	.venv/bin/python scripts/verify_public_manifest.py

audit-dependencies:
	.venv/bin/python scripts/audit_dependency_licenses.py

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests
	.venv/bin/mypy src

wheel:
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) .venv/bin/python -m pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir dist .

sdist:
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) .venv/bin/python -m build --sdist --no-isolation --outdir dist .

ci: validate schemas test lint leaderboard audit-dependencies wheel sdist audit-release
	git diff --exit-code -- leaderboard schemas
