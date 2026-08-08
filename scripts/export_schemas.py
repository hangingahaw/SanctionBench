#!/usr/bin/env python3
"""Export the Pydantic schemas used by publication-safe benchmark artifacts."""

from __future__ import annotations

from sanctionbench.models import (
    CitationItem,
    DocumentPrediction,
    DocumentScenario,
    OrganicDocumentGold,
    OrganicDocumentPrediction,
    Prediction,
    RunRecord,
)
from sanctionbench.submissions import SubmissionBundle
from sanctionbench.util import project_root, write_json


def main() -> None:
    destination = project_root() / "schemas"
    schemas = {
        "citation-item-v1.schema.json": CitationItem.model_json_schema(),
        "document-scenario-v1.schema.json": DocumentScenario.model_json_schema(),
        "citation-prediction-v1.schema.json": Prediction.model_json_schema(),
        "document-prediction-v1.schema.json": DocumentPrediction.model_json_schema(),
        "organic-document-v1.schema.json": OrganicDocumentGold.model_json_schema(),
        "organic-document-prediction-v1.schema.json": (
            OrganicDocumentPrediction.model_json_schema()
        ),
        "run-record-v1.schema.json": RunRecord.model_json_schema(),
        "submission-v1.schema.json": SubmissionBundle.model_json_schema(),
    }
    for name, schema in schemas.items():
        write_json(destination / name, schema)
    print(f"Exported {len(schemas)} public schemas to {destination}")


if __name__ == "__main__":
    main()
