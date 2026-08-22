"""
Knowledge graph data model — the output shape of the Knowledge Graph Agent.

Deliberately scoped to what the pipeline actually knows by this stage:
files and symbols (from the Parser Agent), file-to-file import edges
(from the Dependency Agent), and symbol-to-symbol/file "member of"
relationships (from `Symbol.parent`). There is no call-graph analysis
here — knowing that function A calls function B would need much deeper
analysis than any implemented stage currently does, so this graph
doesn't claim to have it. Later stages (Repository Understanding,
Architecture Agent) can add richer edge types once they exist; this
model isn't meant to be the final word, just an honest graph over the
relationships already established by real data.

Note on `MEMBER_OF` edges: this is one of exactly two places in the
codebase that uses `Symbol.parent` directly for a structural
relationship rather than geometric byte-span containment (the Chunk
Agent deliberately does *not* do this — see `agents/chunker.py`'s
module docstring for why chunk-splitting needs geometry, not the
logical `parent` field). Here, the opposite is true: `parent` is
exactly the right field, because a knowledge graph cares about the
logical relationship a language's semantics assign ("this method
belongs to this struct/class"), not where its bytes happen to sit in
the file.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from parsers.metadata import SymbolKind


class NodeType(StrEnum):
    FILE = "file"
    SYMBOL = "symbol"


class EdgeType(StrEnum):
    IMPORTS = "imports"
    """file -> file, from the Dependency Agent's resolved internal edges."""
    DEFINES = "defines"
    """file -> symbol, for that symbol's top-level (parent-less) definitions."""
    MEMBER_OF = "member_of"
    """symbol -> symbol, from `Symbol.parent` (a method belongs to its class/struct)."""


class KnowledgeNode(BaseModel):
    """One file or symbol in the knowledge graph."""

    id: str
    """`file_path` for a FILE node; `{file_path}::{symbol_name}` for a SYMBOL node (disambiguates same-named symbols across files)."""
    node_type: NodeType
    label: str
    file_path: str
    language: str | None = None
    symbol_kind: SymbolKind | None = None
    """Set only for SYMBOL nodes."""
    summary: str | None = None
    """Optional enrichment from `state.file_docs`/`state.chunk_docs`, when those stages have already run."""


class KnowledgeEdge(BaseModel):
    """One directed relationship between two node ids."""

    source: str
    target: str
    edge_type: EdgeType


class KnowledgeGraph(BaseModel):
    """The complete knowledge graph for a repository: files, symbols, and the relationships between them."""

    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)

    def nodes_by_type(self, node_type: NodeType) -> list[KnowledgeNode]:
        return [n for n in self.nodes if n.node_type == node_type]

    def edges_by_type(self, edge_type: EdgeType) -> list[KnowledgeEdge]:
        return [e for e in self.edges if e.edge_type == edge_type]

    def neighbors(self, node_id: str) -> list[str]:
        """Every node this node has an outgoing edge to, any edge type."""
        return [e.target for e in self.edges if e.source == node_id]

    def node(self, node_id: str) -> KnowledgeNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)
