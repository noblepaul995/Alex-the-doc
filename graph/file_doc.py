"""
File-level documentation data model — the output shape of the File
Documentation Agent.

A `FileDocumentation` rolls up a file's `ChunkDocumentation`s (from the
Chunk Documentation Agent) into one whole-file summary. It's a
synthesis, not a concatenation: the prompt asks the LLM to describe the
file's overall purpose and role given its pieces, rather than just
stitching chunk summaries together.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileDocumentation(BaseModel):
    """One file's LLM-generated whole-file summary, produced by the File Documentation Agent."""

    file_path: str
    language: str
    summary: str
    """A short (2-5 sentence) plain-text description of the file's overall purpose."""
    symbol_names: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    """Set (with `summary` left empty) if generation failed after retries, or every chunk in this file already failed."""


class FileDocumentationCollection(BaseModel):
    """The complete set of file documentation produced for a repository."""

    docs: list[FileDocumentation] = Field(default_factory=list)

    def by_path(self, file_path: str) -> FileDocumentation | None:
        return next((d for d in self.docs if d.file_path == file_path), None)

    def failed(self) -> list[FileDocumentation]:
        return [d for d in self.docs if d.error is not None]
