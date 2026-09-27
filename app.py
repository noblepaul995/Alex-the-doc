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
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from agents.export_agent import SUPPORTED_FORMATS, export_documentation
from agents.qa_agent import investigate
from config.model_recommendations import recommend_tier
from config.providers import PROVIDER_REGISTRY, ProviderType, resolve_provider_config, resolve_synthesis_provider_config
from config.settings import Settings, get_settings
from embeddings.lancedb import LanceVectorStore
from graph.graph import QA_GRAPH_NODE_COUNT, build_graph, build_qa_graph, build_scan_graph
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from utils.github_repo import RepoAccessResult, check_repo_access, clone_repo, parse_github_url
from utils.hardware_detect import HardwareInfo, detect_hardware
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
    "package_dependencies": "Reading package manifest(s)",
    "githubReadme": "Building GitHub README",
    "review": "Reviewing generated docs",
}


async def _run_with_progress(
    graph: object, initial_state: RepositoryState, *, total_stages: int | None = None
) -> dict[str, object]:
    """
    Run a compiled graph via `astream()` instead of `ainvoke()`, driving a
    live progress display so a medium/large repo doesn't look stuck:

      - one bar advances once per pipeline stage as it completes
      - `chunk_doc` and `file_doc` (the two stages that dominate runtime,
        one LLM call per chunk/file) get their own live item-count bars,
        fed by `utils.progress.report_progress` via `progress_scope`

    `total_stages` overrides the bar's total node count — needed for any
    graph other than the full `build_graph()` (e.g. `build_qa_graph()`,
    which only runs 6 of the pipeline's ~19 nodes), since the bar would
    otherwise stall partway and look broken rather than reaching 100%.
    Defaults to `len(_STAGE_LABELS)`, correct for `build_graph()` itself.

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
        pipeline_task = progress.add_task("Pipeline", total=total_stages if total_stages is not None else len(_STAGE_LABELS))
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


def _force_rmtree(path: Path) -> None:
    """
    `shutil.rmtree`, but tolerant of read-only files — specifically git's
    own pack/object files, which git (and Windows' filesystem permission
    model in particular) leaves read-only after a clone. Plain
    `shutil.rmtree` on Windows raises `PermissionError: Access is denied`
    the moment it hits one of these, since Windows (unlike POSIX, where
    the containing directory's permissions are what matter) checks the
    file's own read-only bit before allowing a delete. The `onerror`
    hook here clears that bit and retries the specific failed operation
    once, rather than aborting the whole tree removal.
    """

    def _on_error(func, failed_path, exc_info) -> None:  # noqa: ANN001 — matches shutil's onerror signature
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    shutil.rmtree(path, onerror=_on_error)


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


_API_KEY_ENV_VAR: dict[ProviderType, str] = {
    ProviderType.OPENAI: "OPENAI_API_KEY",
    ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderType.GEMINI: "GEMINI_API_KEY",
    ProviderType.GROQ: "GROQ_API_KEY",
    ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
}
"""Providers not in this dict (OLLAMA, LMSTUDIO, CUSTOM) don't take an API key at all."""


async def _check_provider_availability(settings: Settings) -> dict[ProviderType, tuple[bool, str]]:
    """Probe every registered provider (except CUSTOM, which has no sensible default to probe) and report whether each is usable right now."""
    results: dict[ProviderType, tuple[bool, str]] = {}
    for provider_type, entry in PROVIDER_REGISTRY.items():
        if provider_type == ProviderType.CUSTOM:
            continue
        config = resolve_provider_config(provider=provider_type.value, model="probe", settings=settings)
        if entry.requires_api_key and not config.api_key:
            results[provider_type] = (False, "no API key set")
            continue
        try:
            async with create_provider(config) as client:
                status = await client.health()
            detail = status.detail or ("reachable" if status.healthy else "unreachable")
            results[provider_type] = (status.healthy, detail)
        except ProviderError as exc:
            results[provider_type] = (False, str(exc))
    return results


async def _list_ollama_models(settings: Settings) -> list[str]:
    """Real models already pulled locally, straight from Ollama's own `/api/tags` — not the `OllamaProvider` class, since listing models isn't part of the `BaseProvider` interface and reaching into that class's private HTTP client just for this would be worse than a small, self-contained request here."""
    endpoint = settings.ollama_endpoint.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{endpoint}/api/tags")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError:
        return []
    return [model["name"] for model in data.get("models", [])]


def _print_hardware(hardware: HardwareInfo) -> None:
    parts = []
    if hardware.total_ram_gb is not None:
        parts.append(f"{hardware.total_ram_gb} GB RAM")
    if hardware.gpu_name is not None:
        vram = f" ({hardware.gpu_vram_gb} GB VRAM)" if hardware.gpu_vram_gb is not None else ""
        parts.append(f"{hardware.gpu_name}{vram}")
    console.print("  " + (", ".join(parts) if parts else "[dim]nothing detected[/dim]"))


def _write_env(*, provider: str, model: str, api_key: str | None = None, api_key_var: str | None = None) -> None:
    """
    Persist the chosen default provider/model (and, if given, an API key)
    to `.env` in the current directory. Preserves every other line
    already there — only the specific keys being set are touched, same
    file `config/settings.py` already reads from via `pydantic-settings`'
    `env_file=".env"`, so this takes effect on every future `alex` run
    without the user editing anything by hand.
    """
    env_path = Path(".env")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    def _set(key: str, value: str) -> None:
        prefix = f"{key}="
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = f"{key}={value}"
                return
        lines.append(f"{key}={value}")

    _set("ALEX_DEFAULT_PROVIDER", provider)
    _set("ALEX_DEFAULT_MODEL", model)
    if api_key and api_key_var:
        _set(api_key_var, api_key)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[dim]Wrote ALEX_DEFAULT_PROVIDER={provider}, ALEX_DEFAULT_MODEL={model} to .env[/dim]")
    get_settings.cache_clear()  # so the rest of *this* process picks up the change immediately, not just future runs


def _test_provider(provider_type: ProviderType, model: str) -> None:
    """Send a minimal real request ("Hi") through the exact same provider abstraction the pipeline uses, and print what comes back — the most direct proof setup actually worked."""
    console.print(f'\n[bold]Testing[/bold] {provider_type.value}/{model} — sending "Hi"...')
    settings = get_settings()
    config = resolve_provider_config(provider=provider_type.value, model=model, settings=settings)

    async def _run() -> str:
        async with create_provider(config) as client:
            result = await client.generate([ChatMessage(role=Role.USER, content="Hi")], temperature=0.2, max_tokens=50)
        return result.text.strip()

    try:
        response_text = asyncio.run(_run())
    except ProviderError as exc:
        console.print(f"[red]Test call failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Response:[/green] {response_text or '[dim](empty response)[/dim]'}")
    console.print(f"\n[green]Setup complete.[/green] alex will use [cyan]{provider_type.value}/{model}[/cyan] by default.")


@cli.command()
def setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """
    Interactive first-run setup: see which providers are already usable,
    pick one (or configure a new one), and — if you want to run models
    locally instead — get hardware-based model recommendations, pull
    one via Ollama, and verify it works with a real test call.
    """
    _setup(verbose)
    settings = get_settings()

    console.print("[bold]alex setup[/bold]\n")
    console.print("Checking which providers are already usable...\n")
    availability = asyncio.run(_check_provider_availability(settings))

    table = Table(title="Providers")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Detail")
    for provider_type, (ok, detail) in availability.items():
        entry = PROVIDER_REGISTRY[provider_type]
        status = "[green]available[/green]" if ok else "[dim]not configured[/dim]"
        table.add_row(entry.name, status, detail)
    console.print(table)

    available_types = [pt for pt, (ok, _detail) in availability.items() if ok]
    console.print()
    choices = [pt.value for pt in availability] + ["local"]
    default_choice = available_types[0].value if available_types else "local"
    picked = Prompt.ask(
        "Which provider would you like to use? (pick one already available, name any other to configure it, "
        "or 'local' for a guided local-model setup)",
        choices=choices,
        default=default_choice,
        show_choices=False,
    )

    if picked != "local":
        provider_type = ProviderType(picked)
        entry = PROVIDER_REGISTRY[provider_type]
        already_ok, _detail = availability[provider_type]

        api_key = None
        if entry.requires_api_key and not already_ok:
            api_key = Prompt.ask(f"Enter your {entry.name} API key", password=True)

        default_model = settings.default_model if provider_type.value == settings.default_provider else ""
        model = Prompt.ask("Model name", default=default_model or None)

        _write_env(
            provider=provider_type.value,
            model=model,
            api_key=api_key,
            api_key_var=_API_KEY_ENV_VAR.get(provider_type),
        )
        _test_provider(provider_type, model)
        return

    # --- Local-model path (Ollama) ------------------------------------
    console.print("\n[bold]Setting up a local model with Ollama[/bold]")
    ollama_ok, _detail = availability.get(ProviderType.OLLAMA, (False, "not checked"))
    if not ollama_ok:
        console.print(
            "[red]Ollama doesn't appear to be running.[/red] Install it from "
            "[cyan]https://ollama.com[/cyan] and make sure it's running, then re-run "
            "[cyan]alex setup[/cyan]."
        )
        raise typer.Exit(code=1)

    existing_models = asyncio.run(_list_ollama_models(settings))
    if existing_models:
        console.print(f"\nFound {len(existing_models)} model(s) already pulled: {', '.join(existing_models)}")
        if Confirm.ask("Use one of these instead of pulling a new one?", default=True):
            model = Prompt.ask("Which model", choices=existing_models, default=existing_models[0])
            _write_env(provider="ollama", model=model)
            _test_provider(ProviderType.OLLAMA, model)
            return

    console.print("\nDetecting your hardware to recommend models that will run well locally...")
    hardware = detect_hardware()
    _print_hardware(hardware)

    if hardware.total_ram_gb is None and hardware.gpu_vram_gb is None:
        console.print("[yellow]Couldn't auto-detect your hardware.[/yellow]")
        manual_ram = IntPrompt.ask("Roughly how much system RAM do you have, in GB", default=8)
        hardware = HardwareInfo(total_ram_gb=float(manual_ram), gpu_name=None, gpu_vram_gb=None)

    tier = recommend_tier(hardware)
    console.print(f"\n[bold]Recommended tier:[/bold] {tier.name}\n")

    rec_table = Table(title="Recommended models")
    rec_table.add_column("#")
    rec_table.add_column("Model")
    rec_table.add_column("Size")
    rec_table.add_column("Notes")
    for i, rec in enumerate(tier.models, start=1):
        rec_table.add_row(str(i), rec.tag, rec.params, rec.blurb)
    console.print(rec_table)
    console.print(
        "[dim]This is a curated starting point, not a live or benchmarked ranking — check "
        "https://ollama.com/library for anything newer.[/dim]\n"
    )

    pick = Prompt.ask(
        "Choose a model (1-5), type a model tag directly, or say 'you choose for me'",
        default="you choose for me",
    )
    normalized = pick.strip().lower()
    if normalized in ("you choose for me", "auto", "choose for me", ""):
        chosen = tier.models[0].tag
    elif normalized.isdigit() and 1 <= int(normalized) <= len(tier.models):
        chosen = tier.models[int(normalized) - 1].tag
    else:
        chosen = pick.strip()

    console.print(f"\n[bold]Pulling[/bold] [cyan]{chosen}[/cyan] — this can take a while for larger models...")
    if shutil.which("ollama") is None:
        console.print(
            "[red]The `ollama` command isn't on PATH.[/red] Install it from "
            "[cyan]https://ollama.com[/cyan], then re-run [cyan]alex setup[/cyan]."
        )
        raise typer.Exit(code=1)

    pull_result = subprocess.run(["ollama", "pull", chosen])
    if pull_result.returncode != 0:
        console.print(f"[red]`ollama pull {chosen}` failed[/red] — see the output above.")
        raise typer.Exit(code=1)

    _write_env(provider="ollama", model=chosen)
    _test_provider(ProviderType.OLLAMA, chosen)


async def _qa_loop(state: RepositoryState, settings: Settings) -> None:
    """
    Interactive question-answering loop, backed by `agents/qa_agent.py`'s
    agentic investigation loop rather than a single retrieve-then-answer
    pass — a real test run showed the single-shot version falling back
    to "it can be inferred"/"it seems that" on multi-hop questions
    (e.g. "how are the server, scheduler, and engine connected?") when
    the first retrieval pass didn't happen to cover the whole answer.
    Giving the model tools to keep investigating instead of answering
    with whatever the first search returned is the fix.
    """
    store = await LanceVectorStore.connect(settings.vector_dir)
    if await store.count() == 0:
        console.print("[yellow]No indexed content found[/yellow] — nothing to answer questions from.")
        return

    # The Q&A investigation loop is multi-step, reasoning-heavy, and runs once per question rather
    # than once per chunk/file — the same cost/quality tradeoff `resolve_synthesis_provider_config()`
    # already exists for (see its docstring / `settings.synthesis_provider`). A small per-chunk model
    # that's fine for summarizing individual files is frequently unreliable at multi-step tool-use
    # (malformed ACTION calls, ignoring the protocol, asserting untested conclusions after the step
    # cap) — using the stronger synthesis-tier model here directly attacks that, the same way it
    # already does for the README/Architecture/Repository/Review agents.
    config = resolve_synthesis_provider_config(settings)
    console.print(
        "\n[bold]Ask anything about this repository.[/bold] Type [cyan]exit[/cyan] or [cyan]quit[/cyan] to stop.\n"
    )
    async with create_provider(config) as provider:
        while True:
            question = Prompt.ask("[bold cyan]?[/bold cyan]").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            try:
                answer = await investigate(question, state, state.repo_summary, provider, store)
            except ProviderError as exc:
                console.print(f"[red]Couldn't get an answer:[/red] {exc}")
                continue
            console.print(f"\n{answer}\n")


@cli.command()
def github(
    repo: Annotated[str, typer.Argument(help="A GitHub repo: a URL, git@ SSH form, or plain 'owner/repo'.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """
    Point alex at a GitHub repository: checks whether it's public or
    private (and whether you have access if it's private), clones it,
    then lets you choose between building full documentation or asking
    ad-hoc questions about it (Q&A, via an agentic investigation loop —
    the model can search the repository, read real file contents, look
    up symbols, and trace import relationships across several steps
    before answering, rather than a single retrieve-then-answer pass).
    """
    _setup(verbose)

    parsed = parse_github_url(repo)
    if parsed is None:
        console.print(
            f"[red]Couldn't parse a GitHub repo from[/red] {repo!r}. "
            "Try a full URL (https://github.com/owner/repo), the SSH form, or just 'owner/repo'."
        )
        raise typer.Exit(code=1)
    owner, repo_name = parsed
    console.print(f"[bold]alex github[/bold] — {owner}/{repo_name}\n")

    settings = get_settings()
    token = settings.github_token

    console.print("Checking repository access...")
    access: RepoAccessResult = asyncio.run(check_repo_access(owner, repo_name, token))

    if not access.exists and not token:
        console.print(f"[yellow]{owner}/{repo_name} wasn't found anonymously[/yellow] ({access.detail}).")
        if Confirm.ask("Do you have a GitHub token to check private access with?", default=False):
            token = Prompt.ask("GitHub token", password=True)
            access = asyncio.run(check_repo_access(owner, repo_name, token))

    if not access.exists:
        console.print(f"[red]Can't access {owner}/{repo_name}.[/red] {access.detail}.")
        raise typer.Exit(code=1)

    visibility = "private" if access.private else "public"
    console.print(f"[green]Found it[/green] — {owner}/{repo_name} is {visibility}, and you have access.\n")

    console.print("[bold]What would you like to do?[/bold]")
    console.print("  [cyan]1[/cyan]. Build full documentation (README, architecture, API reference, ...)")
    console.print("  [cyan]2[/cyan]. Ask questions about the repo (Q&A)")
    choice = Prompt.ask("Choice", choices=["1", "2"], default="1")

    cache_root = Path.home() / ".alex" / "github-repos" / f"{owner}__{repo_name}"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    console.print(f"\nCloning {owner}/{repo_name}...")
    try:
        clone_repo(owner, repo_name, cache_root, token=token, branch=access.default_branch)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Cloned[/green] to {cache_root}\n")

    # Every relative .alex/* path (cache, vector store, docs output) is
    # resolved against the process's current working directory, same as
    # a normal `alex document <path>` run assumes — chdir'ing into the
    # clone for the duration of this command is what keeps its cache and
    # vector store scoped to *this* clone rather than colliding with
    # whatever `.alex/` directory happens to sit in the user's own cwd.
    original_cwd = Path.cwd()
    repo_path = cache_root.resolve()
    try:
        os.chdir(repo_path)
        get_settings.cache_clear()
        settings = get_settings()

        if choice == "1":
            graph = build_graph()
            result = asyncio.run(_run_with_progress(graph, RepositoryState(repository_path=repo_path)))
            if result.get("errors"):
                console.print("[red]Errors during run:[/red]")
                for error in result["errors"]:
                    console.print(f"  - {error.message if hasattr(error, 'message') else error}")

            state = RepositoryState(**result)
            output_dir = settings.docs_dir
            written = export_documentation(state, ["markdown"], output_dir)
            console.print(f"\n[bold]Exported to[/bold] [cyan]{(repo_path / output_dir).resolve()}[/cyan]")
            for format_name, paths in written.items():
                for path in paths:
                    console.print(f"  {format_name}: {path}")
        else:
            graph = build_qa_graph()
            result = asyncio.run(
                _run_with_progress(
                    graph, RepositoryState(repository_path=repo_path), total_stages=QA_GRAPH_NODE_COUNT
                )
            )
            if result.get("errors"):
                console.print("[red]Errors during run:[/red]")
                for error in result["errors"]:
                    console.print(f"  - {error.message if hasattr(error, 'message') else error}")
            asyncio.run(_qa_loop(RepositoryState(**result), settings))
    finally:
        os.chdir(original_cwd)
        get_settings.cache_clear()


if __name__ == "__main__":
    cli()