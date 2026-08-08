from __future__ import annotations

from conftest import make_item

from sanctionbench.document_audit import build_document_scenarios
from sanctionbench.models import Condition, GoldLabel
from sanctionbench.providers.mock import MockProvider


def test_mock_document_provider_runs_both_conditions() -> None:
    items = [
        make_item("fake-1", GoldLabel.NONEXISTENT_CASE, "pair-1"),
        make_item("real-1", GoldLabel.REAL, "pair-1"),
    ]
    scenario = build_document_scenarios(items, seed=3)[0]
    provider = MockProvider("fixture")

    closed = provider.predict_document(scenario, Condition.CLOSED_BOOK, None)
    assert len(closed.assessments) == len(scenario.authorities)

    evidence = {
        authority.authority_id: {"reported_count": 0, "matches": []}
        for authority in scenario.authorities
    }
    assisted = provider.predict_document(scenario, Condition.TOOL_ASSISTED, evidence)
    assert len(assisted.assessments) == len(scenario.authorities)
