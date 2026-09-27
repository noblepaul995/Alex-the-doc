"""
Investigation tools for `alex github`'s agentic Q&A loop (see
`agents/qa_agent.py`). Each function here answers one narrow, concrete
question by querying data the pipeline already verified — the knowledge
graph, the dependency graph, file documentation, raw source, and the
vector store — never by asking an LLM to guess or infer. The one
exception is `web_search_tool`, which reaches outside the repository on
purpose — see its own docstring for why that's still in scope here.

This is the direct fix for the failure mode a real test run surfaced:
single-shot retrieve-then-answer, when the retrieved chunks don't
happen to cover a multi-hop question ("how are the server, scheduler,
and engine connected?"), falls back to phrases like "it can be
inferred" or "it seems that" — confident-sounding language covering an
actual gap in evidence. Giving the model a way to keep investigating
instead of just answering with what the first retrieval pass returned
is the fix; each tool here is designed to return either real, specific
evidence or an honest "not found," never a plausible-sounding guess.
"""

from __future__ import annotations

from pathlib import Path

from embeddings.lancedb import LanceVectorStore
from graph.knowledge import EdgeType, NodeType
from graph.state import RepositoryState
from llm.base import BaseProvider
from utils.web_search import WebSearchError, is_web_search_available, is_web_search_configured, render_results
from utils.web_search import web_search as _web_search

_MAX_FILE_CHARS = 6000
"""Cap on how much of a file's raw content `get_file` returns — enough for real tracing, not enough to blow the context window on one call."""


async def search_repository(query: str, state: RepositoryState, provider: BaseProvider, store: LanceVectorStore, *, limit: int = 8) -> str:
    """Semantic search over the repository's embedded chunk/file summaries — one investigation step among several available to the agentic loop."""
    query_vector = (await provider.embed([query])).vectors[0]
    results = await store.search(query_vector, limit=limit)
    if not results:
        return "No matching content found in the repository's indexed summaries."
    lines = []
    for result in results:
        kind = "file summary" if result.record_type.value == "file" else "code chunk"
        lines.append(f"- [{kind} — {result.file_path}]: {result.text}")
    return "\n".join(lines)


def get_file(path: str, state: RepositoryState) -> str:
    """
    Real file content (capped at `_MAX_FILE_CHARS`), read straight off
    disk relative to `state.repository_path` — for actually tracing a
    call graph, which a summary alone often can't support ("the GenSpec
    object is then used to generate responses" is a summary; the actual
    next function call is only in the source). Falls back to the file's
    `FileDocumentation` summary if the raw file can't be read (e.g. it
    was filtered out during scanning), and reports plainly if neither
    exists rather than guessing at the file's contents.
    """
    normalized = path.strip().lstrip("/")
    full_path = state.repository_path / normalized
    if full_path.is_file():
        try:
            text = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return f"Found `{normalized}` but couldn't read it: {exc}"
        if len(text) > _MAX_FILE_CHARS:
            text = text[:_MAX_FILE_CHARS] + f"\n... (truncated, {len(text)} characters total)"
        return f"Contents of `{normalized}`:\n\n{text}"

    doc = next((d for d in state.file_docs.docs if d.file_path == normalized), None)
    if doc is not None and doc.summary:
        return f"Raw file not found on disk, but here's its recorded summary:\n{doc.summary}"

    return f"No file found at `{normalized}` (checked both the working copy and recorded file summaries)."


def get_symbol(name: str, state: RepositoryState) -> str:
    """
    Look up a symbol (function, class, method, ...) by name in the
    knowledge graph — exact match first, falling back to a
    case-insensitive substring match if nothing matches exactly, since a
    question is more likely to say "the scheduler class" than the
    symbol's precise declared name. Reports what actually defines it,
    what it belongs to (for a method/member), and what it's connected to
    via real graph edges — never guesses at a symbol's behavior beyond
    what the graph and its recorded summary state.

    Real limitation worth knowing when deciding whether to use this tool
    versus `get_file`: the knowledge graph has no call-graph analysis
    (see `graph/knowledge.py`'s module docstring) — it only knows
    file-to-file imports and symbol-to-parent membership, not which
    functions call which. "What does function X call?" needs `get_file`
    on X's defining file and reading the actual source, not this tool.
    """
    symbols = state.knowledge_graph.nodes_by_type(NodeType.SYMBOL)
    exact = [n for n in symbols if n.label == name]
    matches = exact or [n for n in symbols if name.lower() in n.label.lower()]

    if not matches:
        return f'No symbol named or matching "{name}" found in the knowledge graph.'

    if len(matches) > 1 and not exact:
        listing = "\n".join(f"- {n.label} ({n.symbol_kind or 'symbol'}) in `{n.file_path}`" for n in matches[:10])
        return f'Multiple symbols match "{name}":\n{listing}\n\nAsk again naming one of these precisely to see its details.'

    node = matches[0]
    lines = [f"`{node.label}` ({node.symbol_kind or 'symbol'}) — defined in `{node.file_path}`."]
    if node.summary:
        lines.append(f"Summary: {node.summary}")

    member_edges = state.knowledge_graph.edges_by_type(EdgeType.MEMBER_OF)
    parent = next((e.target for e in member_edges if e.source == node.id), None)
    if parent:
        parent_node = state.knowledge_graph.node(parent)
        if parent_node:
            lines.append(f"Belongs to: {parent_node.label} (in `{parent_node.file_path}`).")

    return "\n".join(lines)


def list_dependencies(path: str, state: RepositoryState) -> str:
    """
    Real, resolved import relationships for a file, in both directions —
    `state.dependency_graph.dependencies_of`/`dependents_of`, exact
    static-analysis results, not an inference about what a file
    "probably" imports based on its name or summary.
    """
    normalized = path.strip().lstrip("/")
    imports = state.dependency_graph.dependencies_of(normalized)
    imported_by = state.dependency_graph.dependents_of(normalized)

    if not imports and not imported_by:
        return f"No resolved internal import relationships found for `{normalized}` — it may not exist, or only imports external packages."

    lines = [f"Import relationships for `{normalized}`:"]
    lines.append(f"  Imports (within this repo): {', '.join(imports) if imports else '(none resolved internally)'}")
    lines.append(f"  Imported by: {', '.join(imported_by) if imported_by else '(nothing in this repo imports it)'}")
    return "\n".join(lines)


async def web_search_tool(query: str) -> str:
    """
    The one tool here that reaches outside the repository on purpose —
    everything else in this module answers from data the pipeline
    already verified about *this* codebase; this is for when a question
    genuinely needs current or external information the repository
    itself can't contain (whether a dependency version is still
    supported, what an external API's current behavior is, ...).

    `prompts/qa_agent.md` is the place that actually enforces
    "repository evidence first, web search only when repository
    evidence is insufficient or the question is explicitly about
    current/external information" — this function itself has no way to
    enforce that priority, it just executes the search when the model
    decides to call it. Delegates entirely to `utils/web_search.py`
    (see that module's docstring for why it's a standalone utility, not
    something defined here) and never raises: no backend available or a
    failed request becomes a plain-text result the model can react to
    (e.g. by answering from repository evidence alone instead), the same
    "always return a string, never crash the loop" contract every other
    tool in this module follows.

    This is the point `interactive=True` matters: if open-websearch
    isn't already running, this is where the person actually gets asked
    whether to install and start it (see `utils/web_search.py`) — not
    when the tool is merely listed as available (that check, via
    `is_web_search_configured`, never prompts).
    """
    if not await is_web_search_available(interactive=True):
        return "Web search isn't available for this run (no open-websearch daemon running/startable, and no TAVILY_API_KEY set) — answer from repository evidence only."
    try:
        results = await _web_search(query)
    except WebSearchError as exc:
        return f"Web search failed: {exc}"
    return render_results(results)