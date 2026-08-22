"""
Chunk data model — the output shape of the Chunker (Chunk Generator)
stage.

A `Chunk` is a contiguous, documentation-sized slice of one source
file. Chunk boundaries are AST-aware: wherever possible, a chunk holds
one or more *complete* top-level symbols (functions, classes, ...)
rather than an arbitrary line range, so the Documentation Agents that
consume chunks later never have to reason about a function cut off
mid-body. When a single symbol is itself too large for one chunk (a
large class, a very long function), it's split as gracefully as the
AST allows — by its own children (methods) if it has any, or by line
range as a last resort — and `is_partial` + `parent_symbol` record that
this chunk is one piece of a larger whole.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One contiguous, documentation-sized slice of a single source file."""

    id: str
    """Stable identifier: `{file_path}:{start_line}-{end_line}`."""
    file_path: str
    language: str
    start_line: int
    """1-indexed, inclusive."""
    end_line: int
    """1-indexed, inclusive."""
    start_byte: int
    end_byte: int
    content: str
    """The chunk's source text, with `context_header` (if any) prepended."""
    symbol_names: list[str] = Field(default_factory=list)
    """Top-level (or, for a partial chunk, child) symbol names this chunk covers."""
    parent_symbol: str | None = None
    """Set when this chunk is one piece of a larger symbol that had to be split."""
    is_partial: bool = False
    """True if this chunk is one of several pieces covering a single oversized symbol."""
    context_header: str | None = None
    """Synthetic line(s) prepended to `content` for a partial chunk, e.g. `# class Foo (continued)`."""
    token_estimate: int = 0


class ChunkCollection(BaseModel):
    """The complete set of chunks produced for a repository."""

    chunks: list[Chunk] = Field(default_factory=list)

    def by_file(self, file_path: str) -> list[Chunk]:
        return [c for c in self.chunks if c.file_path == file_path]

    def partial_chunks(self) -> list[Chunk]:
        return [c for c in self.chunks if c.is_partial]


class ChunkDocumentation(BaseModel):
    """One chunk's LLM-generated documentation, produced by the Chunk Documentation Agent."""

    chunk_id: str
    file_path: str
    summary: str
    """A short (1-3 sentence) plain-text description of what this chunk does."""
    symbol_names: list[str] = Field(default_factory=list)
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    """Set (with `summary` left empty) if generation failed for this chunk after retries."""


class ChunkDocumentationCollection(BaseModel):
    """The complete set of chunk documentation produced for a repository."""

    docs: list[ChunkDocumentation] = Field(default_factory=list)

    def by_chunk_id(self, chunk_id: str) -> ChunkDocumentation | None:
        return next((d for d in self.docs if d.chunk_id == chunk_id), None)

    def by_file(self, file_path: str) -> list[ChunkDocumentation]:
        return [d for d in self.docs if d.file_path == file_path]

    def failed(self) -> list[ChunkDocumentation]:
        return [d for d in self.docs if d.error is not None]
