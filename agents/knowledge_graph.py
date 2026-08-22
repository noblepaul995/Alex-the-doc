"""
Knowledge Graph Agent — builds a file/symbol relationship graph.

Runs after the Vector Database Agent in the full pipeline, but needs
none of the LLM-derived data those earlier stages produce to do its
core job: every node and edge here comes straight from `state.parse_results`
(the Parser Agent) and `state.dependency_graph` (the Dependency Agent),
both of which are pure, local, already-verified data. The only thing
this stage borrows from later-in-pipeline data is optional: a file's
`state.file_docs` summary, attached to its node when that stage has
already run. Because of that, `knowledge_graph_node` is a plain
synchronous function — no network call, no `async def` — and is wired
into *both* `build_scan_graph()` and `build_graph()`, giving `scan`
users a structural graph preview even without touching an LLM, while
the full pipeline gets the same graph with summaries attached.

See `graph/knowledge.py`'s module docstring for what relationships this
graph does and doesn't claim to represent.
"""

from __future__ import annotations

from graph.knowledge import EdgeType, KnowledgeEdge, KnowledgeGraph, KnowledgeNode, NodeType
from graph.state import RepositoryState
from parsers.metadata import ParseResult, Symbol
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)


def knowledge_graph_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: build `state.knowledge_graph` from parsed symbols and resolved dependencies."""
    with Stopwatch("knowledge_graph") as sw:
        nodes: list[KnowledgeNode] = []
        edges: list[KnowledgeEdge] = []

        for file_path, parse_result in state.parse_results.items():
            nodes.append(_file_node(file_path, parse_result, state))
            for symbol in parse_result.symbols:
                node, edge = _symbol_node_and_edge(file_path, parse_result, symbol)
                nodes.append(node)
                edges.append(edge)

        known_files = {n.id for n in nodes if n.node_type == NodeType.FILE}
        for dep_edge in state.dependency_graph.internal_edges():
            for target in dep_edge.targets:
                if dep_edge.source in known_files and target in known_files:
                    edges.append(KnowledgeEdge(source=dep_edge.source, target=target, edge_type=EdgeType.IMPORTS))

    graph = KnowledgeGraph(nodes=nodes, edges=edges)
    log.info(
        "Knowledge Graph Agent: %d node(s) (%d file, %d symbol), %d edge(s) in %.2fs",
        len(graph.nodes),
        len(graph.nodes_by_type(NodeType.FILE)),
        len(graph.nodes_by_type(NodeType.SYMBOL)),
        len(graph.edges),
        sw.elapsed_seconds,
    )

    return {"knowledge_graph": graph}


def _file_node(file_path: str, parse_result: ParseResult, state: RepositoryState) -> KnowledgeNode:
    file_doc = state.file_docs.by_path(file_path) if state.file_docs.docs else None
    summary = file_doc.summary if file_doc is not None and file_doc.error is None else None
    return KnowledgeNode(id=file_path, node_type=NodeType.FILE, label=file_path, file_path=file_path, language=parse_result.language, summary=summary)


def _symbol_node_and_edge(file_path: str, parse_result: ParseResult, symbol: Symbol) -> tuple[KnowledgeNode, KnowledgeEdge]:
    node_id = _symbol_id(file_path, symbol)
    node = KnowledgeNode(
        id=node_id,
        node_type=NodeType.SYMBOL,
        label=symbol.name,
        file_path=file_path,
        language=parse_result.language,
        symbol_kind=symbol.kind,
    )

    if symbol.parent is None:
        edge = KnowledgeEdge(source=file_path, target=node_id, edge_type=EdgeType.DEFINES)
    else:
        parent_id = f"{file_path}::{symbol.parent}"
        edge = KnowledgeEdge(source=node_id, target=parent_id, edge_type=EdgeType.MEMBER_OF)

    return node, edge


def _symbol_id(file_path: str, symbol: Symbol) -> str:
    """`{file_path}::{name}` for a top-level symbol, `{file_path}::{parent}.{name}` for a member — disambiguates same-named methods across different classes/structs in the same file."""
    if symbol.parent:
        return f"{file_path}::{symbol.parent}.{symbol.name}"
    return f"{file_path}::{symbol.name}"
