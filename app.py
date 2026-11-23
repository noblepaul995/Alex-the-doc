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
import html as html_lib
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated

import httpx
import markdown as markdown_lib
import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table

from agents.api_agent import detect_endpoints
from agents.export_agent import SUPPORTED_FORMATS, export_documentation
from agents.qa_agent import investigate
from exporters.html import build_search_index, ordered_doc_names, render_docs_page
from tools.api_tester import build_test_plan, parse_kv_flags, run_test_plan
from config.model_recommendations import recommend_tier
from config.providers import PROVIDER_REGISTRY, ProviderType, resolve_provider_config, resolve_synthesis_provider_config
from config.settings import Settings, get_settings
from embeddings.embed import EmbeddingItem, embed_items
from embeddings.lancedb import LanceVectorStore
from embeddings.vector import RecordType
from graph.graph import QA_GRAPH_NODE_COUNT, build_graph, build_qa_graph, build_scan_graph
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from memory.repository import RepositoryMemory
from retrieval.search import semantic_search
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
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=32, complete_style="cyan", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
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
    rich_markup_mode="rich",
)
log = get_logger(__name__)


def _time_greeting(hour: int | None = None) -> str:
    """Return a friendly time-of-day greeting based on local hour."""
    h = datetime.now().hour if hour is None else hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 21:
        return "Good evening"
    return "Good night"


def _read_key() -> str:
    """Read a single keypress (`up`, `down`, `enter`, `esc`, or character) cross-platform."""
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(code, "")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x1b":
            return "esc"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _select_menu_option(
    options: list[tuple[str, str, str]],
    *,
    title: str = "What would you like to do?",
    col_header: str = "Command",
    default_index: int = 0,
    cancel_value: str | None = "exit",
) -> str:
    """
    Interactive arrow-key selector rendered with Rich `Live`.
    Falls back to `Prompt.ask` when stdin is not a TTY.
    """
    if not sys.stdin.isatty():
        menu = _styled_table(title)
        menu.add_column("#", justify="right", style="cyan bold")
        menu.add_column(col_header, style="bold")
        menu.add_column("Description")
        for num, name, desc in options:
            menu.add_row(num, name, desc)
        console.print(menu)
        default_num = options[default_index][0] if 0 <= default_index < len(options) else options[0][0]
        raw = Prompt.ask("Choose an option (number or name)", default=default_num).strip().lower()
        aliases = {num: name for num, name, _ in options}
        if cancel_value is not None:
            aliases.update({"q": cancel_value, "quit": cancel_value})
        return aliases.get(raw, raw)

    idx = max(0, min(default_index, len(options) - 1))

    def _render(selected: int) -> Panel:
        table = Table(
            box=None,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("", width=2, no_wrap=True)
        table.add_column(col_header, width=18, no_wrap=True)
        table.add_column("Description")

        for i, (_num, name, desc) in enumerate(options):
            if i == selected:
                table.add_row(
                    "[bold cyan]❯[/bold cyan]",
                    f"[bold black on cyan] {name} [/bold black on cyan]",
                    f"[bold white]{desc}[/bold white]",
                )
            else:
                table.add_row(" ", f"[cyan]{name}[/cyan]", f"[dim]{desc}[/dim]")

        quick_hint = f"1-{len(options)}" if len(options) > 1 else "1"
        return Panel(
            table,
            title=f"[bold white]{title}[/bold white]",
            subtitle=f"[dim]↑/↓ move  •  Enter select  •  {quick_hint} quick pick  •  Esc/q cancel[/dim]",
            border_style="bright_black",
            box=box.ROUNDED,
        )

    number_map = {num: name for num, name, _ in options}

    with Live(_render(idx), console=console, refresh_per_second=30, transient=True) as live:
        while True:
            try:
                key = _read_key()
            except KeyboardInterrupt:
                if cancel_value is not None:
                    return cancel_value
                raise typer.Exit(code=0)
            if key in ("up", "k"):
                idx = (idx - 1) % len(options)
                live.update(_render(idx))
            elif key in ("down", "j"):
                idx = (idx + 1) % len(options)
                live.update(_render(idx))
            elif key == "enter":
                return options[idx][1]
            elif key in ("esc", "q"):
                if cancel_value is not None:
                    return cancel_value
                raise typer.Exit(code=0)
            elif key in number_map:
                return number_map[key]


@cli.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Interactive welcome launcher when `alex` is invoked with no subcommand."""
    if ctx.invoked_subcommand is not None:
        return

    greeting = _time_greeting()
    console.print()
    console.print(
        Panel(
            f"[bold cyan]{greeting}![/bold cyan] Welcome to [bold white]alex[/bold white] — AI Documentation Engine.\n"
            f"[dim]Workspace:[/dim] [cyan]{Path.cwd()}[/cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    options = [
        ("1", "scan", "Walk the repository and preview tracked files (offline)"),
        ("2", "document", "Run the full documentation pipeline"),
        ("3", "export", "Document and export to Markdown, HTML, PDF, or JSON"),
        ("4", "update", "Incrementally re-document only changed files"),
        ("5", "search", "Semantic search over indexed code summaries"),
        ("6", "serve", "Serve generated documentation in a local web UI"),
        ("7", "api-test", "Smoke-test discovered HTTP endpoints against a live server"),
        ("8", "github", "Clone a GitHub repo to document or run Q&A"),
        ("9", "stats", "Show cached file, doc, and vector index statistics"),
        ("10", "doctor", "Check LLM provider connectivity & auth"),
        ("11", "setup", "Interactive first-run provider & model setup"),
        ("12", "rebuild-index", "Rebuild vector index from cached summaries"),
        ("13", "clear-cache", "Clear cached file hashes, parses, and summaries"),
        ("0", "exit", "Exit alex"),
    ]
    selected = _select_menu_option(options)

    if selected == "exit":
        return
    if selected == "scan":
        target = Path(Prompt.ask("Repository path", default="."))
        scan(path=target)
    elif selected == "document":
        target = Path(Prompt.ask("Repository path", default="."))
        document(path=target)
    elif selected == "export":
        target = Path(Prompt.ask("Repository path", default="."))
        fmt = Prompt.ask("Format(s) (markdown,html,pdf,json)", default="markdown")
        out_raw = Prompt.ask("Output directory (leave blank for default)", default="").strip()
        export(path=target, fmt=fmt, output=Path(out_raw) if out_raw else None)
    elif selected == "update":
        target = Path(Prompt.ask("Repository path", default="."))
        update(path=target)
    elif selected == "search":
        q = Prompt.ask("Search query").strip()
        if q:
            search(query=q)
    elif selected == "serve":
        port_raw = Prompt.ask("Port", default="8420").strip()
        serve(port=int(port_raw) if port_raw.isdigit() else 8420)
    elif selected == "api-test":
        target = Path(Prompt.ask("Repository path", default="."))
        base_url = Prompt.ask("Server base URL", default="http://localhost:8000").strip()
        api_test(path=target, url=base_url)
    elif selected == "github":
        repo_arg = Prompt.ask("GitHub repo (owner/repo or URL)").strip()
        if repo_arg:
            github(repo=repo_arg)
    elif selected == "stats":
        stats()
    elif selected == "doctor":
        doctor()
    elif selected == "setup":
        setup()
    elif selected == "rebuild-index":
        rebuild_index()
    elif selected == "clear-cache":
        clear_cache()
    else:
        console.print(f"[red]Unknown option:[/red] {selected!r}. Run [cyan]alex --help[/cyan] to see all commands.")
        raise typer.Exit(code=1)


def _print_banner(command: str, subtitle: str | None = None) -> None:
    """Render a clean, consistent header rule for CLI commands."""
    title = f"[bold cyan]alex[/bold cyan] [bold white]{command}[/bold white]"
    if subtitle:
        title += f" [dim]— {subtitle}[/dim]"
    console.print()
    console.print(Rule(title, style="bright_black", align="left"))


def _styled_table(title: str | None = None, *, show_lines: bool = False) -> Table:
    """Build a Rich Table with consistent rounded borders and header styling."""
    return Table(
        title=title,
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold cyan",
        title_style="bold white",
        show_lines=show_lines,
    )


def _print_errors(errors: list[object], title: str = "Errors during run") -> None:
    """Render deduplicated pipeline errors inside a red-bordered panel."""
    seen: set[str] = set()
    lines: list[str] = []
    for error in errors:
        message = error.message if hasattr(error, "message") else str(error)
        if message not in seen:
            seen.add(message)
            lines.append(f"• {message}")
    if lines:
        console.print(Panel("\n".join(lines), title=f"[bold red]{title}[/bold red]", border_style="red", box=box.ROUNDED))


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
    _print_banner("scan", str(path.resolve()))
    graph = build_scan_graph()
    result = graph.invoke(RepositoryState(repository_path=path.resolve()))

    if result.get("errors"):
        _print_errors(result["errors"], title="Errors during scan")
        if not result.get("file_metadata"):
            raise typer.Exit(code=1)

    file_metadata = result.get("file_metadata", {})
    stats = result.get("statistics")

    summary_lines = [
        f"[bold]Repository:[/bold] [cyan]{path.resolve()}[/cyan]",
        f"[bold]Tracked:[/bold] {len(file_metadata)} file(s)   [bold]Skipped:[/bold] {stats.files_skipped if stats else 0} (binary/ignored)",
    ]
    if stats:
        summary_lines.append(f"[bold]Changed since last run:[/bold] {stats.files_changed} file(s)")
    if stats and stats.files_skipped_secret:
        summary_lines.append(
            f"[yellow]Secrets skipped:[/yellow] {stats.files_skipped_secret} file(s) (.env, keys, credentials — never read)"
        )
    console.print(Panel("\n".join(summary_lines), title="Scan Summary", border_style="cyan", box=box.ROUNDED))

    if not file_metadata:
        console.print("[yellow]No trackable files found.[/yellow]")
        return

    by_language: dict[str, int] = {}
    for project_file in file_metadata.values():
        key = project_file.language or "unknown"
        by_language[key] = by_language.get(key, 0) + 1

    table = _styled_table("Files by language")
    table.add_column("Language", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    for language, count in sorted(by_language.items(), key=lambda kv: -kv[1]):
        table.add_row(language, str(count))
    console.print(table)

    if verbose:
        file_table = _styled_table("Files")
        file_table.add_column("Path")
        file_table.add_column("Language", style="cyan")
        file_table.add_column("Size", justify="right")
        file_table.add_column("SHA256 (first 12)", style="dim")
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
    _print_banner("document", str(path.resolve()))
    if provider or model:
        console.print(
            "[yellow]--provider/--model overrides aren't wired to the graph yet[/yellow] "
            "(resolve_provider_config() currently reads only from settings/env); "
            "running with the configured default instead."
        )

    graph = build_graph()
    result = asyncio.run(_run_with_progress(graph, RepositoryState(repository_path=path.resolve())))

    if result.get("errors"):
        _print_errors(result["errors"])

    chunk_docs = result.get("chunk_docs")
    docs = chunk_docs.docs if chunk_docs else []
    if not docs:
        console.print("[yellow]No chunks were documented.[/yellow]")
        return

    stats = result.get("statistics")
    ok_chunks = len([d for d in docs if not d.error])
    doc_summary = [
        f"[bold]Repository:[/bold] [cyan]{path.resolve()}[/cyan]",
        f"[bold]Chunks documented:[/bold] {ok_chunks}/{len(docs)}"
        + (f"   [bold]Tokens used:[/bold] {stats.tokens_used:,}" if stats else ""),
    ]
    console.print(Panel("\n".join(doc_summary), title="Run Summary", border_style="green", box=box.ROUNDED))

    table = _styled_table("Chunk summaries")
    table.add_column("File", style="cyan")
    table.add_column("Lines", justify="right", style="dim")
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
        file_table = _styled_table("File summaries")
        file_table.add_column("File", style="cyan")
        file_table.add_column("Summary")
        for doc in file_doc_list:
            if doc.error:
                continue
            file_table.add_row(doc.file_path, doc.summary)
        console.print(file_table)

    repo_summary = result.get("repo_summary")
    if repo_summary:
        console.print(Panel(repo_summary, title="Repository Summary", border_style="cyan", box=box.ROUNDED))

    generated_docs = result.get("generated_docs") or {}
    if generated_docs:
        gen_table = _styled_table("Generated documents")
        gen_table.add_column("Document", style="cyan")
        gen_table.add_column("Size", justify="right")
        for name, doc_text in generated_docs.items():
            gen_table.add_row(name, f"{len(doc_text):,} chars")
        console.print(gen_table)

    review = result.get("review")
    if review and review.results:
        review_table = _styled_table("Review findings")
        review_table.add_column("Document", style="cyan")
        review_table.add_column("Status")
        review_table.add_column("Issues")
        for r in review.results:
            status = "[green]● PASS[/green]" if r.passed else "[red]✖ ISSUES[/red]"
            issues = "; ".join(r.issues) if r.issues else "[dim]-[/dim]"
            review_table.add_row(r.doc_name, status, issues)
        console.print(review_table)


@cli.command()
def update(
    path: Annotated[Path, typer.Argument(help="Repository path to incrementally update.")] = Path("."),
    fmt: Annotated[str, typer.Option("--format", help="Comma-separated: markdown, html, pdf, json")] = "markdown",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output directory. Defaults to ALEX_DOCS_DIR.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run even if zero files changed.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Incrementally re-document changed files using the Memory System and export updated docs."""
    _setup(verbose)
    repo_path = path.resolve()
    _print_banner("update", str(repo_path))

    formats = [f.strip().lower() for f in fmt.split(",") if f.strip()]
    unknown = set(formats) - SUPPORTED_FORMATS
    if unknown:
        console.print(f"[red]Unknown format(s): {sorted(unknown)}. Supported: {sorted(SUPPORTED_FORMATS)}[/red]")
        raise typer.Exit(code=1)

    settings = get_settings()
    output_dir = output or settings.docs_dir

    if not force:
        scan_result = build_scan_graph().invoke(RepositoryState(repository_path=repo_path))
        scan_stats = scan_result.get("statistics")
        if scan_stats and scan_stats.files_changed == 0 and output_dir.exists() and any(output_dir.glob("*.md")):
            console.print(
                Panel(
                    f"[green]Everything is up to date[/green] — 0 of {scan_stats.files_total} file(s) changed since last run.\n"
                    "[dim]Pass --force to regenerate docs anyway.[/dim]",
                    title="Incremental Update",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
            return

    graph = build_graph()
    result = asyncio.run(_run_with_progress(graph, RepositoryState(repository_path=repo_path)))

    if result.get("errors"):
        _print_errors(result["errors"])

    state = RepositoryState(**result)
    written = export_documentation(state, formats, output_dir)

    export_table = _styled_table(f"Updated docs in {output_dir.resolve()}")
    export_table.add_column("Format", style="cyan")
    export_table.add_column("Output Path")
    for format_name, paths in written.items():
        for p in paths:
            export_table.add_row(format_name, str(p))
    console.print()
    console.print(export_table)


@cli.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language search query.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum results to display.")] = 8,
    kind: Annotated[str | None, typer.Option("--type", "-t", help="Filter by record type: file or chunk.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Semantic search over documented code summaries in the local vector store."""
    _setup(verbose)
    _print_banner("search", query)
    settings = get_settings()

    record_type: RecordType | None = None
    if kind is not None:
        normalized = kind.strip().lower()
        if normalized not in ("file", "chunk"):
            console.print("[red]Invalid --type.[/red] Use 'file' or 'chunk'.")
            raise typer.Exit(code=1)
        record_type = RecordType.FILE if normalized == "file" else RecordType.CHUNK

    async def _run_search():
        config = resolve_provider_config(settings=settings)
        async with create_provider(config) as provider:
            return await semantic_search(query, settings.vector_dir, provider, limit=limit, record_type=record_type)

    try:
        results = asyncio.run(_run_search())
    except ProviderError as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print(
            "[yellow]No matching results found[/yellow] (or the vector index is empty — "
            "run [cyan]alex document .[/cyan] or [cyan]alex export .[/cyan] first)."
        )
        return

    table = _styled_table(f"Top {len(results)} result(s)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Type", style="magenta")
    table.add_column("File", style="cyan")
    table.add_column("Distance", justify="right", style="dim")
    table.add_column("Summary")
    for i, item in enumerate(results, start=1):
        summary_preview = item.text if len(item.text) <= 180 else item.text[:177] + "..."
        table.add_row(str(i), item.record_type.value, item.file_path, f"{item.distance:.3f}", summary_preview)
    console.print(table)


def _render_serve_page(doc_name: str, docs: dict[str, Path]) -> str:
    """Render one Markdown file from `docs` using the shared documentation HTML shell."""
    all_texts = {name: path.read_text(encoding="utf-8", errors="replace") for name, path in docs.items()}
    nav_items = [(name, f"/{name}") for name in docs]
    search_index = build_search_index(all_texts, url_pattern="/{name}")
    return render_docs_page(
        doc_name,
        all_texts[doc_name],
        nav_items=nav_items,
        active_name=doc_name,
        search_index=search_index,
    )


@cli.command()
def serve(
    port: Annotated[int, typer.Option(help="Port to serve documentation on.")] = 8420,
    docs_dir: Annotated[Path | None, typer.Option("--dir", "-d", help="Directory containing generated Markdown docs.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Serve generated documentation locally with a navigation sidebar."""
    _setup(verbose)
    settings = get_settings()
    target_dir = (docs_dir or settings.docs_dir).resolve()
    _print_banner("serve", f"{target_dir} on port {port}")

    if not target_dir.is_dir():
        console.print(f"[red]Directory not found:[/red] {target_dir}. Run [cyan]alex export .[/cyan] first.")
        raise typer.Exit(code=1)

    found = {p.stem: p for p in sorted(target_dir.glob("*.md"))}
    if not found:
        console.print(f"[yellow]No Markdown files found in {target_dir}.[/yellow]")
        raise typer.Exit(code=1)

    ordered = {name: found[name] for name in ordered_doc_names(list(found.keys()))}
    default_doc = next(iter(ordered))

    class _DocsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler naming
            slug = self.path.split("?", 1)[0].strip("/")
            if not slug:
                slug = default_doc
            if slug.endswith(".md"):
                slug = slug[:-3]
            if slug not in ordered:
                self.send_error(404, f"Document not found: {slug}")
                return
            html_bytes = _render_serve_page(slug, ordered).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            log.debug("serve: " + format, *args)

    server = ThreadingHTTPServer(("127.0.0.1", port), _DocsHandler)
    console.print(
        Panel(
            f"[bold]Serving {len(ordered)} document(s)[/bold] from [cyan]{target_dir}[/cyan]\n"
            f"[bold]URL:[/bold] [link=http://127.0.0.1:{port}]http://127.0.0.1:{port}[/link]\n"
            "[dim]Press Ctrl+C to stop.[/dim]",
            title="alex serve",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    finally:
        server.server_close()


@cli.command()
def export(
    path: Annotated[Path, typer.Argument(help="Repository path to document and export.")] = Path("."),
    fmt: Annotated[str, typer.Option("--format", help="Comma-separated: markdown, html, pdf, json")] = "markdown",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output directory. Defaults to ALEX_DOCS_DIR.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full documentation pipeline and export the results to disk in the given format(s)."""
    _setup(verbose)
    _print_banner("export", f"{path.resolve()} ({fmt})")
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
        _print_errors(result["errors"])

    state = RepositoryState(**result)
    written = export_documentation(state, formats, output_dir)

    export_table = _styled_table(f"Exported to {output_dir.resolve()}")
    export_table.add_column("Format", style="cyan")
    export_table.add_column("Output Path")
    for format_name, paths in written.items():
        if not paths:
            export_table.add_row(format_name, "[dim]nothing to write[/dim]")
            continue
        for p in paths:
            export_table.add_row(format_name, str(p))
    console.print()
    console.print(export_table)


@cli.command(name="api-test")
def api_test(
    path: Annotated[Path, typer.Argument(help="Repository path to discover HTTP endpoints from.")] = Path("."),
    url: Annotated[str, typer.Option("--url", "-u", help="Base URL of the running server, e.g. http://localhost:8000")] = "http://localhost:8000",
    include_writes: Annotated[bool, typer.Option("--include-writes", help="Also execute non-GET/HEAD methods (POST, PUT, PATCH, DELETE).")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview planned requests without sending HTTP traffic.")] = False,
    param: Annotated[list[str], typer.Option("--param", "-p", help="Path parameter substitution as key=value (repeatable).")] = [],
    header: Annotated[list[str], typer.Option("--header", "-H", help="Extra HTTP header as 'Key: Value' (repeatable).")] = [],
    timeout: Annotated[float, typer.Option("--timeout", help="Per-request timeout in seconds.")] = 10.0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Discover HTTP endpoints in a repository and smoke-test them against a live server."""
    _setup(verbose)
    repo_path = path.resolve()
    _print_banner("api-test", f"{repo_path} -> {url}")

    try:
        param_map = parse_kv_flags(list(param), sep="=", label="--param")
        header_map = parse_kv_flags(list(header), sep=":", label="--header")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    scan_result = build_scan_graph().invoke(RepositoryState(repository_path=repo_path))
    parse_results = scan_result.get("parse_results") or {}
    candidates = detect_endpoints(parse_results, repository_root=repo_path)

    if not candidates:
        console.print("[yellow]No HTTP endpoints detected in this repository.[/yellow]")
        return

    plan = build_test_plan(candidates, include_writes=include_writes, params=param_map)
    if not plan:
        console.print("[yellow]No testable HTTP routes could be extracted from detected endpoints.[/yellow]")
        return

    if dry_run:
        plan_table = _styled_table(f"Dry-run plan ({len(plan)} route(s))")
        plan_table.add_column("Method", style="bold cyan")
        plan_table.add_column("Route")
        plan_table.add_column("Target Path", style="cyan")
        plan_table.add_column("Status")
        plan_table.add_column("Note", style="dim")
        for case in plan:
            status_badge = "[green]READY[/green]" if case.status == "PENDING" else "[yellow]SKIPPED[/yellow]"
            plan_table.add_row(
                case.method,
                case.route_template,
                case.resolved_path or "-",
                status_badge,
                case.skip_reason or case.source,
            )
        console.print(plan_table)
        return

    results = asyncio.run(run_test_plan(plan, url, headers=header_map, timeout=timeout))

    res_table = _styled_table(f"API Smoke Test Results ({url})")
    res_table.add_column("Method", style="bold cyan")
    res_table.add_column("Route")
    res_table.add_column("Status", justify="right")
    res_table.add_column("Latency", justify="right")
    res_table.add_column("Result")
    res_table.add_column("Detail", style="dim")

    passed = failed = skipped = 0
    for r in results:
        if r.outcome == "PASS":
            passed += 1
            badge = "[green]● PASS[/green]"
        elif r.outcome == "SKIPPED":
            skipped += 1
            badge = "[yellow]○ SKIPPED[/yellow]"
        else:
            failed += 1
            badge = f"[red]✖ {r.outcome}[/red]"

        status_str = str(r.status_code) if r.status_code is not None else "-"
        latency_str = f"{r.latency_ms:.0f} ms" if r.latency_ms is not None else "-"
        route_display = r.case.resolved_path or r.case.route_template
        res_table.add_row(r.case.method, route_display, status_str, latency_str, badge, r.detail)

    console.print(res_table)
    border = "green" if failed == 0 else "red"
    console.print(
        Panel(
            f"[green]Passed:[/green] {passed}   [red]Failed/Errors:[/red] {failed}   [yellow]Skipped:[/yellow] {skipped}",
            title="Summary",
            border_style=border,
            box=box.ROUNDED,
        )
    )
    if failed > 0:
        raise typer.Exit(code=1)


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
    """Rebuild the LanceDB vector index from cached SQLite summaries without re-running LLM calls."""
    _setup(verbose)
    _print_banner("rebuild-index", "Re-embedding cached summaries into LanceDB")
    settings = get_settings()

    if not settings.db_path.exists():
        console.print(f"[yellow]No cache database found at {settings.db_path}.[/yellow] Run [cyan]alex document .[/cyan] first.")
        raise typer.Exit(code=1)

    memory = RepositoryMemory(settings.db_path)
    chunk_docs, file_docs = memory.load_all_cached_docs()
    if not chunk_docs and not file_docs:
        console.print("[yellow]No cached chunk or file summaries found to index.[/yellow]")
        return

    items: list[EmbeddingItem] = [
        EmbeddingItem(
            id=f"chunk:{d.chunk_id}",
            record_type=RecordType.CHUNK,
            file_path=d.file_path,
            text=d.summary,
        )
        for d in chunk_docs
    ]
    items.extend(
        EmbeddingItem(
            id=f"file:{d.file_path}",
            record_type=RecordType.FILE,
            file_path=d.file_path,
            text=d.summary,
        )
        for d in file_docs
    )

    async def _rebuild() -> tuple[int, int]:
        config = resolve_provider_config(settings=settings)
        async with create_provider(config) as provider:
            collection = await embed_items(
                items, provider, batch_size=settings.embedding_batch_size, max_concurrency=settings.max_concurrent_requests
            )
        valid = [r for r in collection.records if r.error is None and r.vector]
        store = await LanceVectorStore.connect(settings.vector_dir)
        written = await store.write_records(valid)
        return written, len(collection.failed())

    try:
        written, failed = asyncio.run(_rebuild())
    except ProviderError as exc:
        console.print(f"[red]Rebuild failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[green]Rebuilt vector index[/green] at [cyan]{settings.vector_dir}[/cyan]\n"
            f"[bold]Indexed:[/bold] {written} vector(s) ({len(chunk_docs)} chunk, {len(file_docs)} file)"
            + (f"   [yellow]Failed:[/yellow] {failed}" if failed else ""),
            title="Index Rebuilt",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


@cli.command()
def stats(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Show cached file, summary, and vector store statistics."""
    _setup(verbose)
    _print_banner("stats", "Memory & vector store metrics")
    settings = get_settings()

    counts: dict[str, int] = {}
    if settings.db_path.exists():
        counts = RepositoryMemory(settings.db_path).get_stats()

    async def _vector_count() -> int:
        if not settings.vector_dir.exists():
            return 0
        store = await LanceVectorStore.connect(settings.vector_dir)
        return await store.count()

    vec_count = asyncio.run(_vector_count())

    table = _styled_table("Storage & Cache Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    table.add_column("Location", style="dim")
    table.add_row("Tracked file hashes", str(counts.get("file_hash", 0)), str(settings.db_path))
    table.add_row("Cached parse results", str(counts.get("parse_result", 0)), str(settings.db_path))
    table.add_row("Cached chunk doc sets", str(counts.get("chunk_docs", 0)), str(settings.db_path))
    table.add_row("Cached file docs", str(counts.get("file_doc", 0)), str(settings.db_path))
    table.add_row("SQLite database size", _format_bytes(_dir_size_bytes(settings.db_path)), str(settings.db_path))
    table.add_row("Indexed vectors", str(vec_count), str(settings.vector_dir))
    table.add_row("Vector store size", _format_bytes(_dir_size_bytes(settings.vector_dir)), str(settings.vector_dir))
    console.print(table)


@cli.command()
def doctor(
    provider: Annotated[str | None, typer.Option(help="Check a single provider instead of all configured ones.")] = None,
    model: Annotated[str | None, typer.Option(help="Model to check the provider with.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Check provider connectivity/auth. Fully functional today — validates the foundation."""
    _setup(verbose)
    _print_banner("doctor", "Provider connectivity & auth")
    settings = get_settings()

    targets = [ProviderType(provider)] if provider else list(PROVIDER_REGISTRY.keys())

    table = _styled_table()
    table.add_column("Provider", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Detail", style="dim")

    async def check_all() -> None:
        for provider_type in targets:
            entry = PROVIDER_REGISTRY[provider_type]
            if entry.requires_api_key:
                # Skip providers we have no credentials for rather than reporting
                # a misleading failure.
                config_probe = resolve_provider_config(provider=provider_type.value, model=model or "probe", settings=settings)
                if not config_probe.api_key:
                    table.add_row(entry.name, "-", "[dim]○ skipped (no API key)[/dim]", "-", "")
                    continue

            config = resolve_provider_config(provider=provider_type.value, model=model, settings=settings)
            try:
                async with create_provider(config) as client:
                    status = await client.health()
            except ProviderError as exc:
                table.add_row(entry.name, config.model, "[red]✖ error[/red]", "-", str(exc))
                continue

            status_label = "[green]● healthy[/green]" if status.healthy else "[red]● unhealthy[/red]"
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
    _print_banner("setup", "Interactive provider configuration")
    settings = get_settings()

    console.print("Checking which providers are already usable...\n")
    availability = asyncio.run(_check_provider_availability(settings))

    table = _styled_table("Providers")
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Detail", style="dim")
    for provider_type, (ok, detail) in availability.items():
        entry = PROVIDER_REGISTRY[provider_type]
        status = "[green]● available[/green]" if ok else "[dim]○ not configured[/dim]"
        table.add_row(entry.name, status, detail)
    console.print(table)

    available_types = [pt for pt, (ok, _detail) in availability.items() if ok]
    console.print()
    provider_options: list[tuple[str, str, str]] = []
    for i, (pt, (ok, detail)) in enumerate(availability.items(), start=1):
        badge = "available" if ok else detail
        provider_options.append((str(i), pt.value, f"{PROVIDER_REGISTRY[pt].name} ({badge})"))
    provider_options.append((str(len(provider_options) + 1), "local", "Guided local-model setup with Ollama"))

    default_provider_idx = (
        next((i for i, pt in enumerate(availability) if pt in available_types), len(provider_options) - 1)
    )
    picked = _select_menu_option(
        provider_options,
        title="Choose a provider",
        col_header="Provider",
        default_index=default_provider_idx,
        cancel_value=None,
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
        console.print(f"\nFound {len(existing_models)} model(s) already pulled locally.")
        existing_options = [(str(i), m, "Already pulled locally") for i, m in enumerate(existing_models, start=1)]
        existing_options.append((str(len(existing_options) + 1), "__pull__", "Pull a new recommended model instead"))
        chosen_existing = _select_menu_option(
            existing_options,
            title="Select a local Ollama model",
            col_header="Model",
            default_index=0,
            cancel_value=None,
        )
        if chosen_existing != "__pull__":
            _write_env(provider="ollama", model=chosen_existing)
            _test_provider(ProviderType.OLLAMA, chosen_existing)
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

    rec_table = _styled_table("Recommended models")
    rec_table.add_column("#", justify="right", style="dim")
    rec_table.add_column("Model", style="cyan bold")
    rec_table.add_column("Size", justify="right")
    rec_table.add_column("Notes")
    for i, rec in enumerate(tier.models, start=1):
        rec_table.add_row(str(i), rec.tag, rec.params, rec.blurb)
    console.print(rec_table)
    console.print(
        "[dim]This is a curated starting point, not a live or benchmarked ranking — check "
        "https://ollama.com/library for anything newer.[/dim]\n"
    )

    rec_options = [
        (str(i), rec.tag, f"{rec.params} — {rec.blurb}") for i, rec in enumerate(tier.models, start=1)
    ]
    rec_options.append((str(len(rec_options) + 1), "__custom__", "Enter a custom Ollama model tag"))
    picked_model = _select_menu_option(
        rec_options,
        title=f"Choose an Ollama model ({tier.name})",
        col_header="Model",
        default_index=0,
        cancel_value=None,
    )
    if picked_model == "__custom__":
        chosen = Prompt.ask("Enter Ollama model tag", default=tier.models[0].tag).strip() or tier.models[0].tag
    else:
        chosen = picked_model

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
            console.print(Panel(Markdown(answer), title="Answer", border_style="cyan", box=box.ROUNDED))


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
    _print_banner("github", f"{owner}/{repo_name}")

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

    choice = _select_menu_option(
        [
            ("1", "1", "Build full documentation (README, architecture, API reference, ...)"),
            ("2", "2", "Ask questions about the repo (Q&A)"),
        ],
        title=f"What would you like to do with {owner}/{repo_name}?",
        col_header="Action",
        default_index=0,
        cancel_value=None,
    )

    cache_root = Path.home() / ".alex" / "github-repos" / f"{owner}__{repo_name}"
    if cache_root.exists():
        _force_rmtree(cache_root)
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