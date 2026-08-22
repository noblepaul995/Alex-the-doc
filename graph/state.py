"""
The single source of truth for what flows through the documentation
pipeline. Every LangGraph node reads from and writes to a
`RepositoryState` instance.

Only the foundation-level shape is filled in here. Fields that belong to
stages not yet implemented (dependency graph, chunk docs, embeddings,
etc.) are typed as generic, order-preserving containers (`dict`, `list`)
rather than the rich domain models those stages will eventually own —
this keeps the contract stable so later stages can slot in without
reshaping the state, while not pretending those shapes are decided yet.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from embeddings.vector import VectorCollection
from graph.chunk import ChunkCollection, ChunkDocumentationCollection
from graph.dependency import DependencyGraph
from graph.file_doc import FileDocumentationCollection
from graph.knowledge import KnowledgeGraph
from graph.review import ReviewCollection
from parsers.metadata import ParseResult


def _merge_dict_update(current: dict[str, str], update: dict[str, str]) -> dict[str, str]:
    """
    Reducer for `RepositoryState.generated_docs`.

    `architecture`, `api`, and `readme` run as concurrent graph branches and
    each return the full `generated_docs` dict with only their own key
    freshly set (see agents/architecture_agent.py and siblings). Since each
    branch reads its starting `generated_docs` independently at the top of
    the same superstep, a plain last-write-wins merge would silently drop
    whichever branch's key isn't in the final write. Merging key-by-key
    instead keeps every branch's contribution regardless of write order.
    """
    merged = dict(current)
    merged.update(update)
    return merged


def _merge_error_list(current: list[ErrorRecord], update: list[ErrorRecord]) -> list[ErrorRecord]:
    """
    Reducer for `RepositoryState.errors`.

    Every node returns the full current `errors` list (not a diff), so when
    `architecture`/`api`/`readme` run concurrently each branch's write needs
    to merge instead of overwrite, or a sibling branch's recorded errors
    would silently disappear. De-duplicates rather than concatenating,
    since branches that start from the same checkpoint value will each
    return the pre-existing entries too.
    """
    merged = list(current)
    for item in update:
        if item not in merged:
            merged.append(item)
    return merged


def _merge_statistics(current: RunStatistics, update: RunStatistics) -> RunStatistics:
    """
    Reducer for `RepositoryState.statistics`.

    `architecture`/`api`/`readme` each independently increment
    `tokens_used` from the same starting value when they run concurrently,
    so a naive last-write-wins merge would drop two of the three branches'
    token counts. Field-wise max is a safe, monotonic merge for every
    counter here (they only ever increase within a run) — the one known
    imprecision is `tokens_used` specifically: if two branches both add
    tokens from the same baseline, the smaller addition is undercounted
    rather than summed. That only affects the token stat shown to the
    user, never document content or pipeline correctness.
    """
    return RunStatistics(
        files_total=max(current.files_total, update.files_total),
        files_changed=max(current.files_changed, update.files_changed),
        files_skipped=max(current.files_skipped, update.files_skipped),
        files_skipped_secret=max(current.files_skipped_secret, update.files_skipped_secret),
        chunks_documented=max(current.chunks_documented, update.chunks_documented),
        embeddings_generated=max(current.embeddings_generated, update.embeddings_generated),
        vectors_persisted=max(current.vectors_persisted, update.vectors_persisted),
        tokens_used=max(current.tokens_used, update.tokens_used),
        started_at=min(current.started_at, update.started_at),
        finished_at=(
            max(current.finished_at, update.finished_at)
            if current.finished_at is not None and update.finished_at is not None
            else (current.finished_at or update.finished_at)
        ),
    )



class ProjectFile(BaseModel):
    """
    Minimal, preliminary description of a single file in the repository.

    This is a foundation-stage contract only: the Scanner Agent (a later
    stage) is the real owner of this model and will very likely extend
    it with language-specific metadata, ignore-rule provenance, etc.
    """

    path: Path
    relative_path: str
    size_bytes: int
    content_hash: str | None = None
    language: str | None = None
    mtime: float | None = None


class ErrorRecord(BaseModel):
    """A single recoverable error captured during a pipeline run."""

    stage: str
    message: str
    exception_type: str | None = None
    file_path: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RunStatistics(BaseModel):
    """Aggregate counters/timings for a single pipeline run."""

    files_total: int = 0
    files_changed: int = 0
    files_skipped: int = 0
    files_skipped_secret: int = 0
    chunks_documented: int = 0
    embeddings_generated: int = 0
    vectors_persisted: int = 0
    tokens_used: int = 0
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at


class RepositoryState(BaseModel):
    """
    The strongly-typed state object threaded through every LangGraph
    node in the pipeline, per the field list in the project spec.
    """

    model_config = {"arbitrary_types_allowed": True}

    # --- Identity / cursor ---------------------------------------------------
    repository_path: Path
    current_file: ProjectFile | None = None
    current_chunk: dict[str, Any] | None = None

    # --- Per-file metadata -----------------------------------------------------
    file_metadata: dict[str, ProjectFile] = Field(default_factory=dict)

    # --- Parser Agent output -----------------------------------------------------
    parse_results: dict[str, ParseResult] = Field(default_factory=dict)
    """relative_path -> ParseResult, one entry per successfully parsed file."""

    # --- Graphs (populated by later stages) -------------------------------------
    dependency_graph: DependencyGraph = Field(default_factory=DependencyGraph)
    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)

    # --- Chunker output -----------------------------------------------------------
    chunks: ChunkCollection = Field(default_factory=ChunkCollection)

    # --- Documentation artifacts (populated by later stages) --------------------
    chunk_docs: ChunkDocumentationCollection = Field(default_factory=ChunkDocumentationCollection)
    file_docs: FileDocumentationCollection = Field(default_factory=FileDocumentationCollection)

    # --- Embedding Agent output ---------------------------------------------------
    vectors: VectorCollection = Field(default_factory=VectorCollection)

    # --- Retrieval layer (populated by later stages) -----------------------------
    embeddings: dict[str, Any] = Field(default_factory=dict)
    vector_store: dict[str, Any] = Field(default_factory=dict)

    # --- Incremental run bookkeeping ----------------------------------------------
    changed_files: list[str] = Field(default_factory=list)

    # --- Outputs -------------------------------------------------------------------
    repo_summary: str | None = None
    # `architecture`, `api`, and `readme` run as concurrent branches (see
    # graph/graph.py) and each write only their own key here in the same
    # LangGraph superstep. A plain `dict` field uses LastValue channel
    # semantics, which reject more than one write per step — hence the
    # merge reducer, which lets LangGraph combine the three partial
    # updates instead of raising InvalidUpdateError.
    generated_docs: Annotated[dict[str, str], _merge_dict_update] = Field(default_factory=dict)
    review: ReviewCollection = Field(default_factory=ReviewCollection)

    # --- Observability ---------------------------------------------------------------
    # `architecture`/`api`/`readme` run concurrently and each return the
    # full current value of these two fields alongside their own output —
    # see the reducer docstrings above for why both need merge logic
    # rather than LastValue's default one-write-per-step semantics.
    statistics: Annotated[RunStatistics, _merge_statistics] = Field(default_factory=RunStatistics)
    errors: Annotated[list[ErrorRecord], _merge_error_list] = Field(default_factory=list)

    def record_error(self, stage: str, message: str, *, exception: BaseException | None = None, file_path: str | None = None) -> None:
        """Append an `ErrorRecord` without interrupting the pipeline run."""
        self.errors.append(
            ErrorRecord(
                stage=stage,
                message=message,
                exception_type=type(exception).__name__ if exception else None,
                file_path=file_path,
            )
        )
