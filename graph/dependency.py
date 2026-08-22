"""
Dependency graph data model — the output shape of the Dependency Agent.

A `DependencyEdge` represents one resolved (or attempted) import: which
file it came from, what was written in the source, and which file(s)
in the repository it was resolved to, if any. An edge with an empty
`targets` list is not a failure — it usually means the import points
outside the repository (stdlib, a third-party package, an external
crate) — so `is_external` distinguishes "resolved to nothing because
this isn't a local import" from "local import; here's the file".
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field


class DependencyEdge(BaseModel):
    """One import statement, resolved (or not) to file(s) in the repository."""

    source: str
    """relative_path of the file containing the import."""
    module: str
    """The raw module string as written, e.g. `os.path`, `./utils`, `crate::foo`."""
    targets: list[str] = Field(default_factory=list)
    """relative_path(s) this import resolves to. Empty if unresolved/external."""
    imported_names: list[str] = Field(default_factory=list)
    is_relative: bool = False

    @property
    def is_external(self) -> bool:
        """True if this import could not be resolved to a file in the repository."""
        return not self.targets


class DependencyGraph(BaseModel):
    """The complete set of resolved import edges for a repository."""

    edges: list[DependencyEdge] = Field(default_factory=list)

    def internal_edges(self) -> list[DependencyEdge]:
        return [e for e in self.edges if not e.is_external]

    def external_edges(self) -> list[DependencyEdge]:
        return [e for e in self.edges if e.is_external]

    def dependencies_of(self, relative_path: str) -> list[str]:
        """Every file `relative_path` imports from, within the repository."""
        targets: list[str] = []
        for edge in self.edges:
            if edge.source == relative_path:
                targets.extend(edge.targets)
        return targets

    def dependents_of(self, relative_path: str) -> list[str]:
        """Every file that imports `relative_path`, within the repository."""
        sources: list[str] = []
        for edge in self.edges:
            if relative_path in edge.targets:
                sources.append(edge.source)
        return sources

    def to_adjacency(self) -> dict[str, list[str]]:
        """source -> [target, ...], internal edges only, for graph algorithms/exporters."""
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in self.internal_edges():
            adjacency[edge.source].extend(edge.targets)
        return dict(adjacency)
