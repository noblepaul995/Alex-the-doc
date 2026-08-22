"""
Eval harness for tracking documentation-quality metrics across repeated
real runs.

Why this exists: across this project's iteration, "did that prompt change
help" was repeatedly judged by eyeballing a single generated README —
which is not a reliable signal. LLM output varies run to run; a single
sample can look better or worse than the underlying change actually
warrants, in either direction. This script turns that judgment call into
a small set of deterministic, reproducible numbers, computed the exact
same way every time from a finished pipeline run.

**Deliberately reuses the pipeline's own deterministic checks rather than
inventing new ones** — `compute_metrics()` reads `state.review` (already
populated by `agents.review_agent`'s banned-phrase and file-existence
scans) rather than re-implementing that logic here. One source of truth.

Two halves, split so the metric logic can be tested without a live model:

  - `compute_metrics(state)` — pure, deterministic, unit-tested in
    tests/test_eval_harness.py with synthetic states. No provider, no
    network, no randomness.
  - `run_once()` / `main()` — the actual driver. Requires a real, working
    LLM provider (whatever `ALEX_DEFAULT_PROVIDER` points at) — this is
    not a mock test, it's meant to be run against your actual Ollama/
    hosted setup, e.g. before and after a prompt change, to compare real
    numbers instead of a single-sample impression.

Usage:
    python tools/eval_harness.py /path/to/repo1 [/path/to/repo2 ...] --runs 3

Prints a per-run table and an aggregate summary (mean for numeric
metrics, count for boolean ones) across all runs of all given repos.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.graph import build_graph  # noqa: E402
from graph.state import RepositoryState  # noqa: E402


@dataclass
class RunMetrics:
    repo: str
    run_index: int

    readme_word_count: int = 0
    readme_has_overview: bool = False
    readme_has_core_components: bool = False
    readme_has_key_design_decisions: bool = False
    readme_has_toc: bool = False

    flagged_document_count: int = 0
    banned_phrase_issue_count: int = 0
    nonexistent_file_issue_count: int = 0
    other_issue_count: int = 0

    readme_generated: bool = False
    architecture_generated: bool = False
    structure_generated: bool = False
    changelog_generated: bool = False


def compute_metrics(state: RepositoryState, *, repo: str = "", run_index: int = 0) -> RunMetrics:
    """
    Pure function: given a finished `RepositoryState`, compute deterministic
    quality metrics. No provider calls, no I/O beyond reading the state
    already in memory — safe to call repeatedly with synthetic data in
    tests.
    """
    from exporters.assemble import build_export_documents

    metrics = RunMetrics(repo=repo, run_index=run_index)

    documents = build_export_documents(state)
    readme = documents.get("README", "")

    metrics.readme_generated = "readme" in state.generated_docs
    metrics.architecture_generated = "architecture" in state.generated_docs
    metrics.structure_generated = "structure" in state.generated_docs
    metrics.changelog_generated = "changelog" in state.generated_docs

    metrics.readme_word_count = len(readme.split())
    metrics.readme_has_overview = "## Overview" in readme
    metrics.readme_has_core_components = "## Core Components" in readme
    metrics.readme_has_key_design_decisions = "## Key Design Decisions" in readme
    metrics.readme_has_toc = "## Table of Contents" in readme

    metrics.flagged_document_count = len(state.review.flagged_documents())
    for result in state.review.results:
        for issue in result.issues:
            if "banned generic phrase" in issue:
                metrics.banned_phrase_issue_count += 1
            elif "doesn't exist in this repository" in issue:
                metrics.nonexistent_file_issue_count += 1
            else:
                metrics.other_issue_count += 1

    return metrics


async def run_once(repo_path: Path, *, run_index: int, save_dir: Path | None = None) -> tuple[RunMetrics, RepositoryState]:
    """Run the real pipeline once against `repo_path`. Returns (metrics, final_state)."""
    graph = build_graph()
    result: dict[str, object] = {}
    async for mode, payload in graph.astream(  # type: ignore[attr-defined]
        RepositoryState(repository_path=repo_path), stream_mode=["updates", "values"]
    ):
        if mode == "values":
            result = dict(payload)

    state = RepositoryState(repository_path=repo_path)
    for key, value in result.items():
        if hasattr(state, key):
            setattr(state, key, value)

    if save_dir is not None:
        from exporters.assemble import build_export_documents

        run_dir = save_dir / f"{repo_path.name}-run{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, content in build_export_documents(state).items():
            (run_dir / f"{name}.md").write_text(content, encoding="utf-8")
        print(f"  Saved documents to {run_dir}", file=sys.stderr)

    return compute_metrics(state, repo=str(repo_path), run_index=run_index), state


def _print_issue_detail(all_metrics: list[RunMetrics], all_states: list[RepositoryState]) -> None:
    """Print the actual flagged issue text per run — the counts in the table alone aren't enough to diagnose a spike."""
    for metrics, state in zip(all_metrics, all_states):
        flagged = state.review.flagged_documents()
        if not flagged:
            continue
        print(f"\n--- {Path(metrics.repo).name} run {metrics.run_index}: flagged issues ---")
        for result in flagged:
            print(f"  [{result.doc_name}]")
            for issue in result.issues:
                print(f"    - {issue}")


def _print_table(all_metrics: list[RunMetrics]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Eval Harness — Per-Run Metrics")
    table.add_column("Repo")
    table.add_column("Run")
    table.add_column("Words")
    table.add_column("Sections OK")
    table.add_column("ToC")
    table.add_column("Flagged Docs")
    table.add_column("Banned Phrases")
    table.add_column("Bad File Refs")
    table.add_column("Other Issues")

    for m in all_metrics:
        sections_ok = m.readme_has_overview and m.readme_has_core_components and m.readme_has_key_design_decisions
        table.add_row(
            Path(m.repo).name,
            str(m.run_index),
            str(m.readme_word_count),
            "yes" if sections_ok else "NO",
            "yes" if m.readme_has_toc else "no",
            str(m.flagged_document_count),
            str(m.banned_phrase_issue_count),
            str(m.nonexistent_file_issue_count),
            str(m.other_issue_count),
        )
    console.print(table)

    if len(all_metrics) > 1:
        summary = Table(title="Aggregate Summary (across all runs)")
        summary.add_column("Metric")
        summary.add_column("Mean")
        summary.add_column("Min")
        summary.add_column("Max")
        numeric_fields = [
            "readme_word_count", "flagged_document_count", "banned_phrase_issue_count",
            "nonexistent_file_issue_count", "other_issue_count",
        ]
        for field_name in numeric_fields:
            values = [getattr(m, field_name) for m in all_metrics]
            summary.add_row(field_name, f"{statistics.mean(values):.2f}", str(min(values)), str(max(values)))
        console.print(summary)


async def _main_async(repo_paths: list[Path], runs: int, save_dir: Path | None) -> None:
    all_metrics: list[RunMetrics] = []
    all_states: list[RepositoryState] = []
    for repo_path in repo_paths:
        for run_index in range(1, runs + 1):
            print(f"Running {repo_path} (run {run_index}/{runs})...", file=sys.stderr)
            metrics, state = await run_once(repo_path, run_index=run_index, save_dir=save_dir)
            all_metrics.append(metrics)
            all_states.append(state)
    _print_table(all_metrics)
    _print_issue_detail(all_metrics, all_states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repos", nargs="+", type=Path, help="One or more repository paths to run the pipeline against")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run the pipeline against each repo (default: 1)")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="DIR",
        help="Also write each run's real generated documents (README.md, REVIEW.md, etc.) to DIR/{repo}-run{N}/, "
        "so you can open them directly instead of relying on the summary table alone.",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args.repos, args.runs, args.save))


if __name__ == "__main__":
    main()
