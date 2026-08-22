"""
Repository Memory — the SQLite-backed cache that makes incremental runs
possible: "which files changed since last time" (`load_previous_hashes`,
feeding `scanner.hashes.diff_hashes`) and "reuse what we already computed
for files that didn't change" (`load_parse_result`, `load_chunk_docs`,
`load_file_doc`).

A cache lookup is only ever trusted when the stored `content_hash`
matches the file's *current* hash — every `load_*` method takes the
current hash as a required argument and returns `None` on any mismatch,
never a stale result silently reused. This is the single invariant the
rest of the pipeline's incremental behavior depends on, so it's enforced
here in one place rather than left to every caller to remember.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from database.models import CacheRecord
from database.sqlite import create_memory_engine, create_session_factory
from graph.chunk import ChunkDocumentation
from graph.file_doc import FileDocumentation
from parsers.metadata import ParseResult

_FILE_HASH = "file_hash"
_PARSE_RESULT = "parse_result"
_CHUNK_DOCS = "chunk_docs"
_FILE_DOC = "file_doc"


class RepositoryMemory:
    """One connection to the Memory System's SQLite database."""

    def __init__(self, db_path: Path) -> None:
        engine = create_memory_engine(db_path)
        self._session_factory = create_session_factory(engine)

    # --- Hashes (drives scanner.hashes.diff_hashes) --------------------------------

    def load_previous_hashes(self, repository_path: str) -> dict[str, str] | None:
        """
        `{relative_path: content_hash}` as of the last run for this
        repository, or `None` if no prior run is recorded — the
        distinction `diff_hashes` needs to tell "first run, everything
        is new" apart from "previous run recorded, but happened to
        touch zero files" (an empty dict).
        """
        with self._session_factory() as session:
            rows = session.execute(
                select(CacheRecord.relative_path, CacheRecord.content_hash).where(
                    CacheRecord.repository_path == repository_path, CacheRecord.record_type == _FILE_HASH
                )
            ).all()
        if not rows:
            return None
        return {relative_path: content_hash for relative_path, content_hash in rows}

    def save_file_hash(self, repository_path: str, relative_path: str, content_hash: str, language: str | None) -> None:
        self._upsert(repository_path, relative_path, _FILE_HASH, content_hash, json.dumps({"language": language}))

    # --- Parse results ---------------------------------------------------------

    def load_parse_result(self, repository_path: str, relative_path: str, content_hash: str) -> ParseResult | None:
        payload = self._get_matching(repository_path, relative_path, _PARSE_RESULT, content_hash)
        return ParseResult.model_validate_json(payload) if payload is not None else None

    def save_parse_result(self, repository_path: str, relative_path: str, content_hash: str, result: ParseResult) -> None:
        self._upsert(repository_path, relative_path, _PARSE_RESULT, content_hash, result.model_dump_json())

    # --- Chunk documentation -----------------------------------------------------

    def load_chunk_docs(self, repository_path: str, relative_path: str, content_hash: str) -> list[ChunkDocumentation] | None:
        payload = self._get_matching(repository_path, relative_path, _CHUNK_DOCS, content_hash)
        if payload is None:
            return None
        return [ChunkDocumentation.model_validate(item) for item in json.loads(payload)]

    def save_chunk_docs(self, repository_path: str, relative_path: str, content_hash: str, docs: list[ChunkDocumentation]) -> None:
        payload = json.dumps([doc.model_dump(mode="json") for doc in docs])
        self._upsert(repository_path, relative_path, _CHUNK_DOCS, content_hash, payload)

    # --- File documentation -------------------------------------------------------

    def load_file_doc(self, repository_path: str, relative_path: str, content_hash: str) -> FileDocumentation | None:
        payload = self._get_matching(repository_path, relative_path, _FILE_DOC, content_hash)
        return FileDocumentation.model_validate_json(payload) if payload is not None else None

    def save_file_doc(self, repository_path: str, relative_path: str, content_hash: str, doc: FileDocumentation) -> None:
        self._upsert(repository_path, relative_path, _FILE_DOC, content_hash, doc.model_dump_json())

    # --- Internal helpers --------------------------------------------------------

    def _get_matching(self, repository_path: str, relative_path: str, record_type: str, content_hash: str) -> str | None:
        """Return the stored payload only if `content_hash` matches what's cached — a stale cache entry is treated exactly like no entry."""
        with self._session_factory() as session:
            record = session.execute(
                select(CacheRecord).where(
                    CacheRecord.repository_path == repository_path,
                    CacheRecord.relative_path == relative_path,
                    CacheRecord.record_type == record_type,
                )
            ).scalar_one_or_none()
        if record is None or record.content_hash != content_hash:
            return None
        return record.payload

    def _upsert(self, repository_path: str, relative_path: str, record_type: str, content_hash: str, payload: str) -> None:
        with self._session_factory() as session:
            existing = session.execute(
                select(CacheRecord).where(
                    CacheRecord.repository_path == repository_path,
                    CacheRecord.relative_path == relative_path,
                    CacheRecord.record_type == record_type,
                )
            ).scalar_one_or_none()

            if existing is not None:
                existing.content_hash = content_hash
                existing.payload = payload
            else:
                session.add(
                    CacheRecord(
                        repository_path=repository_path,
                        relative_path=relative_path,
                        record_type=record_type,
                        content_hash=content_hash,
                        payload=payload,
                    )
                )
            session.commit()
