"""
LangGraph assembly.

Nineteen nodes are wired up so far, but at two different positions depending
on which compiled graph you're looking at:

    build_scan_graph():
        initialize -> scan -> parse -> resolve_dependencies -> chunk
            -> knowledge_graph -> END

    build_graph():
        initialize -> scan -> parse -> resolve_dependencies -> chunk
            -> chunk_doc -> file_doc -> knowledge_graph -> embed -> vector_db
            -> repository_understanding -> {architecture, api, readme,
            structure, changelog, package_dependencies}
            -> readme -> githubReadme -> review -> END

    `architecture`, `api`, `readme`, `structure`, `changelog`, and
    `package_dependencies` fan out from `repository_understanding` and
    converge on `review` rather than chaining sequentially — see the
    "Why architecture/api/readme fan out" note below. `githubReadme` is the
    one exception: it runs after `readme` specifically (not alongside it),
    since it transforms `readme`'s own output rather than drafting
    independently — see agents/github_readme_agent.py.

`initialize` validates the repository path. `scan` (the Scanner Agent)
walks the repository, obeys `.gitignore`, skips binaries, detects
languages, and hashes file contents, populating `RepositoryState`.
`parse` (the Parser Agent) runs Tree-sitter over every changed file in
a supported language, populating `state.parse_results` with symbols
and imports. `resolve_dependencies` (the Dependency Agent) resolves
those imports into a `state.dependency_graph` of file-to-file edges.
`chunk` (the Chunk Agent) breaks each parsed file into AST-aware
`Chunk`s sized for documentation agents, populating `state.chunks`.
`chunk_doc` (the Chunk Documentation Agent) asks the configured LLM
provider for a summary of each chunk, populating `state.chunk_docs`.
`file_doc` (the File Documentation Agent) synthesizes each file's chunk
summaries into one whole-file summary, populating `state.file_docs`.
`knowledge_graph` (the Knowledge Graph Agent) builds a file/symbol
relationship graph from `state.parse_results` and
`state.dependency_graph`, populating `state.knowledge_graph`. `embed`
(the Embedding Agent) embeds every successful chunk/file summary (never
raw source) into vector space, populating `state.vectors`. `vector_db`
(the Vector Database Agent) persists those vectors to a local LanceDB
database on disk. `repository_understanding` (the Repository Agent)
synthesizes a whole-repository summary from `state.file_docs` and
`state.dependency_graph`, prioritizing the most depended-upon files
when not everything fits in one prompt, populating `state.repo_summary`.
`architecture` (the Architecture Agent) builds a Mermaid dependency
diagram deterministically from `state.dependency_graph` (no LLM
involvement in the diagram itself) plus an LLM narrative describing it,
populating `state.generated_docs["architecture"]`. `api` (the API
Agent) detects route-decorated Python functions and documents them as
an API reference, populating `state.generated_docs["api"]` — skipped
entirely (no LLM call) when no endpoints are found, which is the
expected case for most repositories. `readme` (the README Agent) drafts
a README grounded in the repository summary and central files,
populating `state.generated_docs["readme"]`. `githubReadme` (the GitHub
README Agent) runs immediately after `readme` and wraps its
already-verified text with a centered title and real, verified badges
(license, primary language, files documented) — no independent drafting,
no new LLM call, populating `state.generated_docs["githubReadme"]`.
`review` (the Review
Agent) checks each generated document against the grounding context it
was produced from, populating `state.review` with findings — it never
rewrites `state.generated_docs`, only reports.

**Why architecture/api/readme fan out:** each of these three only reads
state that's already settled by the time `repository_understanding`
finishes (repo_summary, parse_results, dependency_graph) and each writes
to its own key in the `generated_docs` dict, so none of them depends on
another's output. Chaining them sequentially serialized up to three LLM
calls for no reason; fanning them out from `repository_understanding` and
converging on `review` lets LangGraph run all three concurrently instead.

**Why `knowledge_graph` sits in two different places:** it's a plain
synchronous function with no network call — every node/edge it builds
comes from data the Parser and Dependency Agents already produced, both
pure and local. The only thing it can optionally add is a file's
`state.file_docs` summary, when that stage has already populated it.
That optionality is exactly why the position differs: in
`build_scan_graph()` (which never runs `file_doc` at all) it sits right
after `chunk`, giving `scan` users a structural graph with no
summaries. In `build_graph()` it deliberately sits *after* `file_doc`
instead, so the same node call gets real summary enrichment where the
data already exists — putting it any earlier in the full pipeline would
silently produce summary-less nodes there too, defeating the point of
running it later.

**Important:** `chunk_doc` is the first node that makes a network call,
which makes it the first `async def` node in the graph. LangGraph
handles a mix of sync and async nodes fine, but only via `ainvoke()` —
a compiled graph containing any async node raises `TypeError` if you
call the sync `invoke()` on it (verified directly, not assumed). Every
node from here on is expected to grow further LLM-calling stages
(Architecture, API, README, Review), so `build_graph()` is invoked via
`await graph.ainvoke(...)` — see `app.py`'s `document` command.

`scan`, the CLI's free/offline preview command, deliberately does *not*
run `build_graph()` — it calls `build_scan_graph()` instead, a separate
compilation containing only network-free nodes, still invoked
synchronously. Without that split, every use of `scan` would silently
start requiring a reachable LLM provider (Ollama by default), which
would be a real, unannounced behavior regression for anyone using it
purely to preview what would be scanned.

Each later stage (Repository Understanding, Documentation Generator,
Review Agent, ...) will be added as its own node via
`graph.add_node(...)` / `graph.add_edge(...)`, per the pipeline diagram
in the project spec:

    Scanner -> Language Detection -> Dependency Graph -> AST Parser ->
    Chunk Generator -> Chunk Documentation -> File Documentation ->
    Embeddings -> Vector Database -> Knowledge Graph ->
    Repository Understanding -> Documentation Generator ->
    Review Agent -> Output

`review` is the last stage of the pipeline as currently implemented;
Exporters and the Memory System remain unimplemented, intentionally.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.api_agent import api_node
from agents.architecture_agent import architecture_node
from agents.changelog_agent import changelog_node
from agents.chunk_doc import chunk_doc_node
from agents.chunker import chunk_repository_node
from agents.dependencies_agent import dependencies_node
from agents.dependency_agent import dependency_graph_node
from agents.embedder import embedder_node
from agents.file_doc import file_doc_node
from agents.github_readme_agent import github_readme_node
from agents.knowledge_graph import knowledge_graph_node
from agents.parser import parse_repository_node
from agents.readme_agent import readme_node
from agents.repo_agent import repository_understanding_node
from agents.review_agent import review_node
from agents.scanner import scan_repository_node
from agents.structure_agent import structure_node
from agents.vector_db import vector_db_node
from graph.state import RepositoryState
from graph.stages import STAGE_ORDER, render_stage_order  # noqa: F401 - re-exported for convenience
from utils.logger import get_logger

log = get_logger(__name__)


def initialize(state: RepositoryState) -> dict[str, object]:
    """
    Bootstrap node: validates the repository path exists and marks the
    run as started. Real work begins in the `scan` node.
    """
    if not state.repository_path.exists():
        state.record_error("initialize", f"Repository path does not exist: {state.repository_path}")
        log.error("Repository path does not exist: %s", state.repository_path)
    else:
        log.info("Initialized run for repository: %s", state.repository_path)

    return {"statistics": state.statistics, "errors": state.errors}


def _add_scan_pipeline_nodes(graph: StateGraph) -> None:
    """Nodes shared by both `build_scan_graph()` and `build_graph()`: everything before the first LLM call."""
    graph.add_node("initialize", initialize)
    graph.add_node("scan", scan_repository_node)
    graph.add_node("parse", parse_repository_node)
    graph.add_node("resolve_dependencies", dependency_graph_node)
    graph.add_node("chunk", chunk_repository_node)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "scan")
    graph.add_edge("scan", "parse")
    graph.add_edge("parse", "resolve_dependencies")
    graph.add_edge("resolve_dependencies", "chunk")


def build_scan_graph() -> StateGraph:
    """
    Construct and compile the network-free preview pipeline:
    `initialize -> scan -> parse -> resolve_dependencies -> chunk ->
    knowledge_graph -> END`.

    Every node in this graph is synchronous and makes no network calls,
    so it's safe to `.invoke()` directly — this is what the `scan` CLI
    command uses, deliberately kept separate from `build_graph()` so
    that command never starts requiring a reachable LLM provider.
    """
    graph = StateGraph(RepositoryState)
    _add_scan_pipeline_nodes(graph)
    graph.add_node("knowledge_graph", knowledge_graph_node)
    graph.add_edge("chunk", "knowledge_graph")
    graph.add_edge("knowledge_graph", END)
    return graph.compile()


# See graph/stages.py for STAGE_ORDER / render_stage_order() — the real,
# verified execution order of the nodes below, kept in a separate module so
# agents can import it as ground truth without a circular import (this
# module already imports every agent's node function). If you change the
# wiring below, update graph/stages.py's STAGE_ORDER to match.


def build_graph() -> StateGraph:
    """
    Construct and compile the full documentation pipeline (everything
    `build_scan_graph()` has, plus `chunk_doc` and every stage after
    it as they're implemented).

    Contains at least one `async def` node (`chunk_doc`), so this graph
    must be invoked via `await graph.ainvoke(...)`, not the sync
    `.invoke()` — see this module's docstring for why. `knowledge_graph`
    is added here too, but at a different position than in
    `build_scan_graph()` — see this module's docstring for why.
    """
    graph = StateGraph(RepositoryState)
    _add_scan_pipeline_nodes(graph)
    graph.add_node("chunk_doc", chunk_doc_node)
    graph.add_node("file_doc", file_doc_node)
    graph.add_node("knowledge_graph", knowledge_graph_node)
    graph.add_node("embed", embedder_node)
    graph.add_node("vector_db", vector_db_node)
    graph.add_node("repository_understanding", repository_understanding_node)
    graph.add_node("architecture", architecture_node)
    graph.add_node("api", api_node)
    graph.add_node("readme", readme_node)
    graph.add_node("githubReadme", github_readme_node)
    graph.add_node("structure", structure_node)
    graph.add_node("changelog", changelog_node)
    graph.add_node("package_dependencies", dependencies_node)
    graph.add_node("review", review_node)
    graph.add_edge("chunk", "chunk_doc")
    graph.add_edge("chunk_doc", "file_doc")
    graph.add_edge("file_doc", "knowledge_graph")
    graph.add_edge("knowledge_graph", "embed")
    graph.add_edge("embed", "vector_db")
    graph.add_edge("vector_db", "repository_understanding")
    # `architecture`, `api`, `readme`, `structure`, `changelog`, and
    # `package_dependencies` each read only state that's already populated
    # by `repository_understanding` (repo_summary, parse_results,
    # dependency_graph, file_docs) and write to their own key in
    # `generated_docs` — none of them reads another's output. `structure`,
    # `changelog`, and `package_dependencies` don't strictly need
    # `repo_summary` (only `file_docs`/the raw repository path, both
    # available earlier), but running them here alongside their siblings
    # keeps the graph simple and costs nothing since none of the three
    # makes an LLM call.
    graph.add_edge("repository_understanding", "architecture")
    graph.add_edge("repository_understanding", "api")
    graph.add_edge("repository_understanding", "readme")
    graph.add_edge("repository_understanding", "structure")
    graph.add_edge("repository_understanding", "changelog")
    graph.add_edge("repository_understanding", "package_dependencies")
    graph.add_edge("architecture", "review")
    graph.add_edge("api", "review")
    # `githubReadme` is the one exception to the "fan out from
    # repository_understanding, converge on review" shape above: it
    # transforms `readme`'s own output (see agents/github_readme_agent.py's
    # docstring for why it doesn't independently redraft), so it can only
    # start once `readme` finishes, not alongside it.
    graph.add_edge("readme", "githubReadme")
    graph.add_edge("githubReadme", "review")
    graph.add_edge("structure", "review")
    graph.add_edge("changelog", "review")
    graph.add_edge("package_dependencies", "review")
    graph.add_edge("review", END)
    return graph.compile()