"""
SQLAlchemy model for the Memory System.

One generic table, not a table per record type. Every cached thing this
system stores — a file's content hash, a parsed `ParseResult`, a file's
`ChunkDocumentation`s, a `FileDocumentation` — is the same shape at
rest: "for this (repository, file), keyed by what its content hash was
last time, here's a JSON blob." A relational schema mirroring each
Pydantic model's fields would need a migration every time one of those
models gained a field; storing the already-validated `model_dump_json()`
output instead means this table's shape never needs to change when
`Symbol`, `Chunk`, or `ChunkDocumentation` do.

`content_hash` is a first-class column (not buried in the JSON payload)
specifically so the Memory System can answer "is this file's cached
`record_type` still valid?" with a plain column comparison, no JSON
parsing needed, before ever deserializing the payload.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class Base(DeclarativeBase):
    pass


class CacheRecord(Base):
    """One cached artifact for one file in one repository, at one content hash."""

    __tablename__ = "cache_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repository_path: Mapped[str] = mapped_column(String, nullable=False)
    relative_path: Mapped[str] = mapped_column(String, nullable=False)
    record_type: Mapped[str] = mapped_column(String, nullable=False)
    """One of: `file_hash`, `parse_result`, `chunk_docs`, `file_doc`."""
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str] = mapped_column(String, nullable=False)
    """JSON — either a `model_dump_json()` of the relevant Pydantic model, or (for `file_hash`) a small `{"language": ...}` object."""
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (UniqueConstraint("repository_path", "relative_path", "record_type", name="uq_cache_record_key"),)
