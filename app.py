"""
alex — AI Documentation Engine.

CLI entry point. In the foundation stage, most commands are stubs that
explain what they'll do and exit cleanly — real behaviour is layered in
as each pipeline stage (Scanner, Parser, Chunker, ...) is implemented.

`doctor` is fully functional today: it exercises the real provider
abstraction end-to-end (config resolution -> factory -> health check),
which is the best available proof that the foundation actually works.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from agents.export_agent import SUPPORTED_FORMATS, export_documentation
from config.providers import PROVIDER_REGISTRY, ProviderType, resolve_provider_config
from config.settings import get_settings
from graph.graph import build_graph, build_scan_graph
from graph.state import RepositoryState
from llm.base import ProviderError
from llm.factory import create_provider
from utils.logger import configure_logging, get_logger
from utils.progress import progress_scope

console = Console()

# Friendly, ordered labels for `build_graph()`'s nodes, used to drive the
# pipeline progress bar. Order only matters for readability here — actual
# execution order (and the fan-out of architecture/api/readme) is defined
# in graph/graph.py.
_STAGE_LABELS: dict[str, str] = {
    "initialize": "Initializing",
    "scan": "Scanning repository",
    "parse": "Parsing (Tree-sitter)",
    "resolve_dependencies": "Resolving dependencies",
    "chunk": "Chunking files",
    "chunk_doc": "Documenting chunks",
    "file_doc": "Documenting files",
    "knowledge_graph": "Building knowledge graph",
    "embed": "Embedding summaries",
    "vector_db": "Writing vector database",
    "repository_understanding": "Summarizing repository",
    "architecture": "Drafting architecture doc",
    "api": "Drafting API reference",
    "readme": "Drafting README",
    "structure": "Tabulating folder structure",
    "changelog": "Reading recent commit history",
    "review": "Reviewing generated docs",
}


async def _run_with_progress(graph: object, initial_state: RepositoryState) -> dict[str, object]:
    """
    Run a compiled graph via `astream()` instead of `ainvoke()`, driving a
    live progress display so a medium/large repo doesn't look stuck:

      - one bar advances once per pipeline stage as it completes
      - `chunk_doc` and `file_doc` (the two stages that dominate runtime,
        one LLM call per chunk/file) get their own live item-count bars,
        fed by `utils.progress.report_progress` via `progress_scope`

    Returns the same shape `ainvoke()` would: a dict of the final state's
    fields. Safe because every node in this codebase returns the *full*
    current value of whatever it touches (e.g. `state.errors` after any
    appends), never a partial diff — so later updates can just overwrite
    earlier ones for the same key.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        pipeline_task = progress.add_task("Pipeline", total=len(_STAGE_LABELS))
        item_tasks: dict[str, int] = {}

        def _on_item_progress(stage: str, completed: int, total: int) -> None:
            if stage not in item_tasks:
                label = "chunks" if stage == "chunk_doc" else "files"
                item_tasks[stage] = progress.add_task(f"  documenting {label}", total=total)
            progress.update(item_tasks[stage], completed=completed, total=total)

        result: dict[str, object] = {}
        with progress_scope(_on_item_progress):
            # Two stream modes at once: "updates" gives per-node partial
            # output (just for driving the pipeline bar/labels), "values"
            # gives the full, already-reducer-merged state after each
            # superstep. We must use "values" for `result` — accumulating
            # from "updates" alone would re-introduce the exact bug the
            # generated_docs/errors/statistics reducers exist to fix: each
            # concurrent branch's "updates" payload only contains its own
            # partial write, so naively last-writer-wins merging those here
            # would silently drop whichever of architecture/api/readme's
            # output isn't processed last.
            async for mode, payload in graph.astream(  # type: ignore[attr-defined]
                initial_state, stream_mode=["updates", "values"]
            ):
                if mode == "values":
                    result = dict(payload)
                    continue
                for node_name, _node_output in payload.items():
                    progress.update(
                        pipeline_task,
                        advance=1,
                        description=_STAGE_LABELS.get(node_name, node_name),
                    )
        progress.update(pipeline_task, description="Done")
        return result
cli = typer.Typer(
    name="alex",
    help="AI Documentation Engine — understands and documents codebases of any size.",
    no_args_is_help=True,
)
log = get_logger(__name__)


def _setup(verbose: bool = False) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level.value)


@cli.command()
def scan(
    path: Annotated[Path, typer.Argument(help="Repository path to scan.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Walk the repository and report what would be scanned."""
    _setup(verbose)
    graph = build_scan_graph()
    result = graph.invoke(RepositoryState(repository_path=path.resolve()))

    if result.get("errors"):
        console.print("[red]Errors during scan:[/red]")
        seen_messages: set[str] = set()
        for error in result["errors"]:
            message = error.message if hasattr(error, "message") else str(error)
            if message in seen_messages:
                continue
            seen_messages.add(message)
            console.print(f"  - {message}")
        if not result.get("file_metadata"):
            raise typer.Exit(code=1)

    file_metadata = result.get("file_metadata", {})
    stats = result.get("statistics")

    console.print(f"[bold]Scanned[/bold] [cyan]{path.resolve()}[/cyan]")
    console.print(
        f"  {len(file_metadata)} files tracked, {stats.files_skipped if stats else 0} skipped "
        f"(binary/ignored)"
    )
    if stats:
        console.print(f"  {stats.files_changed} file(s) changed since last run (Memory System)")
    if stats and stats.files_skipped_secret:
        console.print(
            f"  [yellow]{stats.files_skipped_secret} file(s) skipped as likely secrets[/yellow] "
            f"(.env, keys, credentials, ...) — never read or documented, regardless of .gitignore"
        )

    if not file_metadata:
        console.print("[yellow]No trackable files found.[/yellow]")
        return

    by_language: dict[str, int] = {}
    for project_file in file_metadata.values():
        key = project_file.language or "unknown"
        by_language[key] = by_language.get(key, 0) + 1

    table = Table(title="Files by language")
    table.add_column("Language")
    table.add_column("Count", justify="right")
    for language, count in sorted(by_language.items(), key=lambda kv: -kv[1]):
        table.add_row(language, str(count))
    console.print(table)

    if verbose:
        file_table = Table(title="Files")
        file_table.add_column("Path")
        file_table.add_column("Language")
        file_table.add_column("Size", justify="right")
        file_table.add_column("SHA256 (first 12)")
        for relative_path in sorted(file_metadata):
            f = file_metadata[relative_path]
            table_hash = (f.content_hash or "")[:12]
            file_table.add_row(relative_path, f.language or "-", str(f.size_bytes), table_hash)
        console.print(file_table)


@cli.command()
def document(
    path: Annotated[Path, typer.Argument(help="Repository path to document.")] = Path("."),
    provider: Annotated[str | None, typer.Option(help="Provider name, e.g. ollama, openai, anthropic.")] = None,
    model: Annotated[str | None, typer.Option(help="Model name.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """
    Run the documentation pipeline through the last implemented stage
    (currently: Chunk Documentation). Later stages (File Documentation,
    Architecture, README, ...) will extend what this command produces
    as they're implemented — it always runs the full graph end to end.
    """
    _setup(verbose)
    if provider or model:
        console.print(
            "[yellow]--provider/--model overrides aren't wired to the graph yet[/yellow] "
            "(resolve_provider_config() currently reads only from settings/env); "
            f"running with the configured default instead."
        )

    graph = build_graph()
    result = asyncio.run(_run_with_progress(graph, RepositoryState(repository_path=path.resolve())))

    if result.get("errors"):
        console.print("[red]Errors during run:[/red]")
        seen_messages: set[str] = set()
        for error in result["errors"]:
            message = error.message if hasattr(error, "message") else str(error)
            if message in seen_messages:
                continue
            seen_messages.add(message)
            console.print(f"  - {message}")

    chunk_docs = result.get("chunk_docs")
    docs = chunk_docs.docs if chunk_docs else []
    if not docs:
        console.print("[yellow]No chunks were documented.[/yellow]")
        return

    stats = result.get("statistics")
    console.print(f"[bold]Documented[/bold] [cyan]{path.resolve()}[/cyan]")
    console.print(
        f"  {len([d for d in docs if not d.error])}/{len(docs)} chunk(s) documented"
        + (f", {stats.tokens_used} token(s) used" if stats else "")
    )

    table = Table(title="Chunk summaries")
    table.add_column("File")
    table.add_column("Lines")
    table.add_column("Summary")
    for doc in docs:
        if doc.error:
            continue
        chunk = next((c for c in result.get("chunks").chunks if c.id == doc.chunk_id), None)
        lines = f"{chunk.start_line}-{chunk.end_line}" if chunk else "-"
        table.add_row(doc.file_path, lines, doc.summary)
    console.print(table)

    file_docs = result.get("file_docs")
    file_doc_list = file_docs.docs if file_docs else []
    if file_doc_list:
        file_table = Table(title="File summaries")
        file_table.add_column("File")
        file_table.add_column("Summary")
        for doc in file_doc_list:
            if doc.error:
                continue
            file_table.add_row(doc.file_path, doc.summary)
        console.print(file_table)

    repo_summary = result.get("repo_summary")
    if repo_summary:
        console.print(f"\n[bold]Repository summary:[/bold]\n{repo_summary}\n")

    generated_docs = result.get("generated_docs") or {}
    for name, doc_text in generated_docs.items():
        console.print(f"[bold]Generated {name}[/bold] ({len(doc_text)} chars) — full text written to state.generated_docs[{name!r}]")

    review = result.get("review")
    if review and review.results:
        review_table = Table(title="Review findings")
        review_table.add_column("Document")
        review_table.add_column("Status")
        review_table.add_column("Issues")
        for r in review.results:
            status = "[green]PASS[/green]" if r.passed else "[red]ISSUES[/red]"
            issues = "; ".join(r.issues) if r.issues else "-"
            review_table.add_row(r.doc_name, status, issues)
        console.print(review_table)


@cli.command()
def update(
    path: Annotated[Path, typer.Argument(help="Repository path to incrementally update.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Regenerate documentation only for changed files (Memory System — not yet implemented)."""
    _setup(verbose)
    console.print("[yellow]Incremental update not yet implemented.[/yellow] Requires the Memory System stage.")


@cli.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language search query.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Semantic search over documented code (Retrieval stage — not yet implemented)."""
    _setup(verbose)
    console.print(f"[yellow]Search not yet implemented.[/yellow] Query received: [cyan]{query}[/cyan]")


@cli.command()
def serve(
    port: Annotated[int, typer.Option(help="Port to serve documentation on.")] = 8420,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Serve generated documentation locally (Export stage — not yet implemented)."""
    _setup(verbose)
    console.print(f"[yellow]Serve not yet implemented.[/yellow] Would serve docs on port {port}.")


@cli.command()
def export(
    path: Annotated[Path, typer.Argument(help="Repository path to document and export.")] = Path("."),
    fmt: Annotated[str, typer.Option("--format", help="Comma-separated: markdown, html, pdf, json")] = "markdown",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output directory. Defaults to ALEX_DOCS_DIR.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full documentation pipeline and export the results to disk in the given format(s)."""
    _setup(verbose)
    formats = [f.strip().lower() for f in fmt.split(",") if f.strip()]
    unknown = set(formats) - SUPPORTED_FORMATS
    if unknown:
        console.print(f"[red]Unknown format(s): {sorted(unknown)}. Supported: {sorted(SUPPORTED_FORMATS)}[/red]")
        raise typer.Exit(code=1)

    settings = get_settings()
    output_dir = output or settings.docs_dir

    graph = build_graph()
    result = asyncio.run(_run_with_progress(graph, RepositoryState(repository_path=path.resolve())))

    if result.get("errors"):
        console.print("[red]Errors during run:[/red]")
        seen_messages: set[str] = set()
        for error in result["errors"]:
            message = error.message if hasattr(error, "message") else str(error)
            if message in seen_messages:
                continue
            seen_messages.add(message)
            console.print(f"  - {message}")

    state = RepositoryState(**result)
    written = export_documentation(state, formats, output_dir)

    console.print(f"\n[bold]Exported to[/bold] [cyan]{output_dir.resolve()}[/cyan]")
    for format_name, paths in written.items():
        if not paths:
            console.print(f"  {format_name}: nothing to write")
            continue
        for p in paths:
            console.print(f"  {format_name}: {p}")


@cli.command(name="clear-cache")
def clear_cache(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Clear the Memory System's cache (file hashes, parsed results, chunk/file docs)."""
    _setup(verbose)
    settings = get_settings()
    if settings.db_path.exists():
        settings.db_path.unlink()
        console.print(f"[green]Cleared[/green] {settings.db_path}")
    else:
        console.print(f"[yellow]Nothing to clear[/yellow] — {settings.db_path} doesn't exist.")
    console.print(f"[dim]Note: {settings.cache_dir} is reserved for future use and isn't written to by any implemented stage yet.[/dim]")


@cli.command(name="rebuild-index")
def rebuild_index(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Rebuild the vector index from stored summaries (Vector Database stage — not yet implemented)."""
    _setup(verbose)
    console.print("[yellow]Index rebuild not yet implemented.[/yellow] Requires the Vector Database stage.")


@cli.command()
def stats(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Show statistics about the last run (Memory System — not yet implemented)."""
    _setup(verbose)
    console.print("[yellow]Stats not yet implemented.[/yellow] Requires the Memory System stage.")


@cli.command()
def doctor(
    provider: Annotated[str | None, typer.Option(help="Check a single provider instead of all configured ones.")] = None,
    model: Annotated[str | None, typer.Option(help="Model to check the provider with.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Check provider connectivity/auth. Fully functional today — validates the foundation."""
    _setup(verbose)
    settings = get_settings()

    targets = [ProviderType(provider)] if provider else list(PROVIDER_REGISTRY.keys())

    table = Table(title="alex doctor")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Latency")
    table.add_column("Detail")

    async def check_all() -> None:
        for provider_type in targets:
            entry = PROVIDER_REGISTRY[provider_type]
            if entry.requires_api_key:
                # Skip providers we have no credentials for rather than reporting
                # a misleading failure.
                config_probe = resolve_provider_config(provider=provider_type.value, model=model or "probe", settings=settings)
                if not config_probe.api_key:
                    table.add_row(entry.name, "-", "[dim]skipped (no API key)[/dim]", "-", "")
                    continue

            config = resolve_provider_config(provider=provider_type.value, model=model, settings=settings)
            try:
                async with create_provider(config) as client:
                    status = await client.health()
            except ProviderError as exc:
                table.add_row(entry.name, config.model, "[red]error[/red]", "-", str(exc))
                continue

            status_label = "[green]healthy[/green]" if status.healthy else "[red]unhealthy[/red]"
            latency = f"{status.latency_ms:.0f} ms" if status.latency_ms is not None else "-"
            table.add_row(entry.name, status.model, status_label, latency, status.detail)

    asyncio.run(check_all())
    console.print(table)


if __name__ == "__main__":
    cli()
