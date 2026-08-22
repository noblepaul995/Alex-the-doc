"""Tests for the Vector Database layer (embeddings/lancedb.py + agents/vector_db.py).

Uses a real LanceDB instance in a temp directory rather than mocking —
LanceDB is a local, embedded, offline database, so there's no network
dependency to avoid and mocking it would just re-describe the API
instead of verifying it actually works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.vector_db import vector_db_node
from embeddings.lancedb import LanceVectorStore
from embeddings.vector import RecordType, VectorCollection, VectorRecord
from graph.state import RepositoryState


def _record(id_: str, record_type: RecordType = RecordType.CHUNK, file_path: str = "a.py", vector: list[float] | None = None, error: str | None = None) -> VectorRecord:
    return VectorRecord(
        id=id_,
        record_type=record_type,
        file_path=file_path,
        text="a summary",
        vector=vector if vector is not None else ([] if error else [0.1, 0.2, 0.3]),
        model="fake",
        dimensions=3 if vector is None and error is None else len(vector or []),
        error=error,
    )


class TestLanceVectorStoreWrite:
    @pytest.mark.asyncio
    async def test_write_and_count(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        written = await store.write_records([_record("a"), _record("b")])
        assert written == 2
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_empty_write_is_noop(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        await store.write_records([_record("a")])
        written = await store.write_records([])
        assert written == 0
        assert await store.count() == 1  # untouched, not reset to 0

    @pytest.mark.asyncio
    async def test_write_overwrites_not_accumulates(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        await store.write_records([_record("a"), _record("b"), _record("c")])
        await store.write_records([_record("a")])
        assert await store.count() == 1


class TestLanceVectorStoreSearch:
    @pytest.mark.asyncio
    async def test_search_returns_nearest_first(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        await store.write_records(
            [
                _record("close", vector=[0.1, 0.1, 0.1]),
                _record("far", vector=[9.0, 9.0, 9.0]),
            ]
        )
        results = await store.search([0.1, 0.1, 0.1], limit=2)
        assert results[0].id == "close"
        assert results[0].distance <= results[1].distance

    @pytest.mark.asyncio
    async def test_search_filters_by_record_type(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        await store.write_records(
            [
                _record("c1", record_type=RecordType.CHUNK, vector=[0.1, 0.1, 0.1]),
                _record("f1", record_type=RecordType.FILE, vector=[0.1, 0.1, 0.1]),
            ]
        )
        results = await store.search([0.1, 0.1, 0.1], limit=10, record_type=RecordType.FILE)
        assert len(results) == 1
        assert results[0].id == "f1"

    @pytest.mark.asyncio
    async def test_search_on_missing_table_returns_empty(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        results = await store.search([0.1, 0.2, 0.3])
        assert results == []


class TestLanceVectorStoreUpsert:
    @pytest.mark.asyncio
    async def test_upsert_creates_table_when_missing(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        written = await store.upsert_records([_record("a")])
        assert written == 1
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_id_in_place(self, tmp_path: Path) -> None:
        store = await LanceVectorStore.connect(tmp_path)
        await store.write_records([_record("a"), _record("b")])
        await store.upsert_records([_record("a", vector=[0.9, 0.9, 0.9])])
        assert await store.count() == 2  # still 2, not 3 — "a" was updated, not appended


class TestVectorDbNode:
    @pytest.mark.asyncio
    async def test_writes_only_successful_records(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import config.settings as settings_module

        settings_module.get_settings.cache_clear()
        monkeypatch.setenv("ALEX_VECTOR_DIR", str(tmp_path / "vectors"))
        settings_module.get_settings.cache_clear()

        state = RepositoryState(repository_path=tmp_path)
        state.vectors = VectorCollection(records=[_record("a"), _record("b", error="failed")])

        result = await vector_db_node(state)
        assert result["statistics"].vectors_persisted == 1

        store = await LanceVectorStore.connect(tmp_path / "vectors")
        assert await store.count() == 1
        settings_module.get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_empty_vectors_is_noop(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.vectors = VectorCollection(records=[])
        result = await vector_db_node(state)
        assert result["statistics"].vectors_persisted == 0
