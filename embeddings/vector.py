"""
Vector record models shared across collections.

A `VectorRecord` pairs an embedded vector with enough metadata to
reconstruct where it came from and re-associate it with the
chunk/file it summarizes once the Vector Database stage stores it.
Deliberately generic across "what got embedded" (`record_type`) rather
than having separate `ChunkVector`/`FileVector` classes — the Vector
Database stage needs to store both kinds in the same collection shape,
and every consumer of `VectorRecord` cares about the same fields
(id, text, vector, dimensions) regardless of which stage produced it.

Per `agents/embedder.py`'s module docstring: only LLM-generated
*summaries* are ever embedded here, never raw source code — `text` is
always a `ChunkDocumentation.summary` or `FileDocumentation.summary`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RecordType(StrEnum):
    CHUNK = "chunk"
    FILE = "file"


class VectorRecord(BaseModel):
    """One embedded summary and the vector it produced."""

    id: str
    """The chunk id or file_path this record was embedded from."""
    record_type: RecordType
    file_path: str
    text: str
    """The summary text that was embedded — never raw source code."""
    vector: list[float] = Field(default_factory=list)
    model: str
    dimensions: int = 0
    error: str | None = None
    """Set (with `vector` left empty) if embedding failed after retries."""


class VectorCollection(BaseModel):
    """The complete set of vector records produced for a repository."""

    records: list[VectorRecord] = Field(default_factory=list)

    def by_type(self, record_type: RecordType) -> list[VectorRecord]:
        return [r for r in self.records if r.record_type == record_type]

    def by_file(self, file_path: str) -> list[VectorRecord]:
        return [r for r in self.records if r.file_path == file_path]

    def failed(self) -> list[VectorRecord]:
        return [r for r in self.records if r.error is not None]


class VectorSearchResult(BaseModel):
    """One result from a similarity search against the vector store."""

    id: str
    record_type: RecordType
    file_path: str
    text: str
    distance: float
