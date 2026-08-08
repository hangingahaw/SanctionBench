"""Publication-safe SanctionBench CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.pretty import Pretty

from .util import project_root

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run and submit SanctionBench.")
console = Console()


def _path(value: Path) -> Path:
    return value if value.is_absolute() else project_root() / value


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Option(help="Benchmark run manifest.")] = Path(
        "configs/smoke.yaml"
    ),
    max_provider_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Explicit retry-inclusive ceiling for live provider requests; required for "
                "any non-mock campaign."
            ),
        ),
    ] = None,
    max_provider_input_bytes: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Explicit retry-inclusive provider input-byte ceiling for live campaigns.",
        ),
    ] = None,
    max_courtlistener_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Explicit retry-inclusive CourtListener wire-request ceiling for tool runs.",
        ),
    ] = None,
) -> None:
    """Run provider/condition combinations and score outputs."""

    from .runner import run_manifest

    result = run_manifest(
        _path(config),
        max_provider_requests=max_provider_requests,
        max_provider_input_bytes=max_provider_input_bytes,
        max_courtlistener_requests=max_courtlistener_requests,
    )
    console.print(f"[green]Completed {len(result)} run(s)[/green]")
    console.print(Pretty(result))


@app.command("run-organic")
def run_organic_command(
    config: Annotated[Path, typer.Option(help="Organic document run manifest.")],
    max_provider_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Explicit retry-inclusive ceiling for live provider requests; required for "
                "any non-mock campaign."
            ),
        ),
    ] = None,
    max_provider_input_bytes: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Explicit retry-inclusive provider input-byte ceiling for live campaigns.",
        ),
    ] = None,
) -> None:
    """Run one neutral, isolated successful response per organic document."""

    from .runner import run_organic_manifest

    result = run_organic_manifest(
        _path(config),
        max_provider_requests=max_provider_requests,
        max_provider_input_bytes=max_provider_input_bytes,
    )
    console.print(f"[green]Completed {len(result)} organic document run(s)[/green]")
    console.print(Pretty(result))


@app.command("score")
def score_command(
    gold: Annotated[Path, typer.Option(help="Gold citation JSONL.")],
    predictions: Annotated[Path, typer.Option(help="Prediction JSONL.")],
    output: Annotated[Path | None, typer.Option(help="Optional metrics output.")] = None,
) -> None:
    """Deterministically score citation predictions."""

    from .grading import score_files

    console.print(
        Pretty(score_files(_path(gold), _path(predictions), _path(output) if output else None))
    )


@app.command("score-organic")
def score_organic_command(
    gold: Annotated[Path, typer.Option(help="Organic document gold JSONL.")],
    predictions: Annotated[Path, typer.Option(help="Organic document prediction JSONL.")],
    output: Annotated[Path | None, typer.Option(help="Optional metrics output.")] = None,
) -> None:
    """Deterministically score open-ended whole-document findings."""

    from .organic_document_audit import score_organic_document_files

    console.print(
        Pretty(
            score_organic_document_files(
                _path(gold), _path(predictions), _path(output) if output else None
            )
        )
    )


@app.command("validate")
def validate_command(
    citations: Annotated[Path, typer.Option(help="Public citation dataset.")] = Path(
        "data/gold/v1/citation_items.jsonl"
    ),
    documents: Annotated[Path, typer.Option(help="Public document scenarios.")] = Path(
        "data/gold/v1/document_scenarios.jsonl"
    ),
) -> None:
    """Validate public datasets without private source files."""

    from .release_validation import validate_public_datasets

    console.print(Pretty(validate_public_datasets(_path(citations), _path(documents))))


@app.command("validate-organic")
def validate_organic_command(
    dataset: Annotated[Path, typer.Option(help="Organic document gold JSONL.")],
) -> None:
    """Validate canonical Markdown and occurrence-level organic gold."""

    from .organic_document_audit import load_organic_document_gold

    documents = load_organic_document_gold(_path(dataset))
    console.print(f"[green]Organic document dataset valid[/green] ({len(documents)} document(s))")


@app.command("package-submission")
def package_submission_command(
    results: Annotated[Path, typer.Option(help="Result index.json from a completed run.")],
    output_dir: Annotated[Path, typer.Option(help="Submission bundle directory.")] = Path(
        "leaderboard/submissions"
    ),
    submitter: Annotated[str, typer.Option(help="Human or team submitting the run.")] = "unknown",
    organization: Annotated[str | None, typer.Option(help="Optional organization.")] = None,
    model_revision: Annotated[
        str, typer.Option(help="Immutable model/API revision.")
    ] = "UNVERIFIED",
    endpoint_type: Annotated[
        str, typer.Option(help="mock, hosted_api, open_weights, or other.")
    ] = "other",
    benchmark_version: Annotated[
        str | None,
        typer.Option(help="Benchmark version; inferred as 1.0.0 or organic-1.0.0 when omitted."),
    ] = None,
) -> None:
    """Create a public aggregate-only submission bundle."""

    from .submissions import package_submission

    path, bundle = package_submission(
        result_index_path=_path(results),
        output_dir=_path(output_dir),
        submitter_name=submitter,
        organization=organization,
        model_revision=model_revision,
        model_endpoint_type=endpoint_type,  # type: ignore[arg-type]
        benchmark_version=benchmark_version,
    )
    console.print(f"[green]Wrote {path}[/green]")
    console.print(Pretty(bundle.model_dump(mode="json")))


@app.command("validate-submission")
def validate_submission_command(
    submission: Annotated[Path, typer.Option(help="Submission bundle JSON.")],
) -> None:
    """Validate a submission bundle and its aggregate invariants."""

    from .submissions import validate_submission

    console.print(Pretty(validate_submission(_path(submission))))


@app.command("build-leaderboard")
def build_leaderboard_command(
    submissions: Annotated[Path, typer.Option(help="Submission bundle directory.")] = Path(
        "leaderboard/submissions"
    ),
    output_dir: Annotated[Path, typer.Option(help="Generated leaderboard directory.")] = Path(
        "leaderboard"
    ),
) -> None:
    """Build deterministic JSON, Markdown, and static HTML leaderboards."""

    from .submissions import build_leaderboard

    destination = _path(output_dir)
    result = build_leaderboard(
        submissions_dir=_path(submissions),
        json_output=destination / "leaderboard.json",
        markdown_output=destination / "LEADERBOARD.md",
        html_output=destination / "index.html",
    )
    console.print(Pretty(result))


if __name__ == "__main__":
    app()
