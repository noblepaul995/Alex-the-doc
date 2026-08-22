"""Tests for the Embedding Agent (embeddings/embed.py + agents/embedder.py)."""

from __future__ import annotations

import pytest

from agents.embedder import _collect_items
from embeddings.embed import EmbeddingItem, embed_items
from embeddings.vector import RecordType
from graph.chunk import ChunkDocumentation, ChunkDocumentationCollection
from graph.file_doc import FileDocumentation, FileDocumentationCollection
from graph.state import RepositoryState
from llm.base import BaseProvider, EmbeddingResult, HealthStatus, ModelInfo, ProviderError


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import embeddings.embed as embed_module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(0),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )

    monkeypatch.setattr(embed_module, "with_retry", _fast_with_retry)


def _item(id_: str, text: str = "A summary.", file_path: str = "a.py", record_type: RecordType = RecordType.CHUNK) -> EmbeddingItem:
    return EmbeddingItem(id=id_, record_type=record_type, file_path=file_path, text=text)


class _FakeEmbeddingProvider(BaseProvider):
    def __init__(self, *, dims: int = 4, fail_times: int = 0, mismatch: bool = False) -> None:
        self.batches: list[list[str]] = []
        self._dims = dims
        self._fail_times = fail_times
        self._mismatch = mismatch
        self._call_count = 0

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        self.batches.append(texts)
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ProviderError("simulated embedding failure", provider="fake", retryable=True)
        vectors = [[float(i)] * self._dims for i in range(len(texts))]
        if self._mismatch:
            vectors = vectors[:-1] if vectors else vectors
        return EmbeddingResult(vectors=vectors, model="fake-embed", dimensions=self._dims)

    async def generate(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-embed")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-embed", supports_streaming=False, supports_embeddings=True)


class TestEmbedItems:
    @pytest.mark.asyncio
    async def test_one_record_per_item(self) -> None:
        provider = _FakeEmbeddingProvider()
        items = [_item("a.py:1-3"), _item("a.py:4-6")]
        result = await embed_items(items, provider, batch_size=10, max_concurrency=4)
        assert len(result.records) == 2
        assert {r.id for r in result.records} == {"a.py:1-3", "a.py:4-6"}

    @pytest.mark.asyncio
    async def test_batches_multiple_items_into_one_call(self) -> None:
        provider = _FakeEmbeddingProvider()
        items = [_item(f"c{i}") for i in range(5)]
        await embed_items(items, provider, batch_size=5, max_concurrency=1)
        assert len(provider.batches) == 1
        assert len(provider.batches[0]) == 5

    @pytest.mark.asyncio
    async def test_batch_size_splits_into_multiple_calls(self) -> None:
        provider = _FakeEmbeddingProvider()
        items = [_item(f"c{i}") for i in range(7)]
        await embed_items(items, provider, batch_size=3, max_concurrency=4)
        assert len(provider.batches) == 3  # 3 + 3 + 1
        assert sum(len(b) for b in provider.batches) == 7

    @pytest.mark.asyncio
    async def test_vectors_and_dimensions_carried_through(self) -> None:
        provider = _FakeEmbeddingProvider(dims=8)
        items = [_item("a.py:1-3")]
        result = await embed_items(items, provider, batch_size=10, max_concurrency=1)
        record = result.records[0]
        assert record.dimensions == 8
        assert len(record.vector) == 8
        assert record.error is None

    @pytest.mark.asyncio
    async def test_persistent_batch_failure_records_error_for_all_items_in_batch(self) -> None:
        provider = _FakeEmbeddingProvider(fail_times=99)
        items = [_item("a.py:1-3"), _item("a.py:4-6")]
        result = await embed_items(items, provider, batch_size=10, max_concurrency=1)
        assert all(r.error is not None for r in result.records)
        assert all(r.vector == [] for r in result.records)

    @pytest.mark.asyncio
    async def test_transient_failure_recovers_via_retry(self) -> None:
        provider = _FakeEmbeddingProvider(fail_times=2)
        items = [_item("a.py:1-3")]
        result = await embed_items(items, provider, batch_size=10, max_concurrency=1)
        assert result.records[0].error is None

    @pytest.mark.asyncio
    async def test_vector_count_mismatch_fails_whole_batch_defensively(self) -> None:
        provider = _FakeEmbeddingProvider(mismatch=True)
        items = [_item("a.py:1-3"), _item("a.py:4-6")]
        result = await embed_items(items, provider, batch_size=10, max_concurrency=1)
        assert all(r.error is not None for r in result.records)

    @pytest.mark.asyncio
    async def test_one_bad_batch_does_not_block_others(self) -> None:
        class _SelectiveFailProvider(_FakeEmbeddingProvider):
            async def embed(self, texts):
                if "bad" in texts[0]:
                    raise ProviderError("nope", provider="fake", retryable=False)
                return await super().embed(texts)

        provider = _SelectiveFailProvider()
        items = [_item("bad1", text="bad text"), _item("good1", text="good text")]
        result = await embed_items(items, provider, batch_size=1, max_concurrency=4)
        by_id = {r.id: r for r in result.records}
        assert by_id["bad1"].error is not None
        assert by_id["good1"].error is None

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_collection(self) -> None:
        provider = _FakeEmbeddingProvider()
        result = await embed_items([], provider, batch_size=10, max_concurrency=4)
        assert result.records == []


class TestCollectItems:
    def test_gathers_successful_chunk_and_file_docs(self) -> None:
        state = RepositoryState(repository_path=__import__("pathlib").Path("."))
        state.chunk_docs = ChunkDocumentationCollection(
            docs=[
                ChunkDocumentation(chunk_id="a.py:1-3", file_path="a.py", summary="chunk summary", model="fake"),
                ChunkDocumentation(chunk_id="a.py:4-6", file_path="a.py", summary="", model="fake", error="failed"),
            ]
        )
        state.file_docs = FileDocumentationCollection(
            docs=[FileDocumentation(file_path="a.py", language="python", summary="file summary", model="fake")]
        )
        items = _collect_items(state)
        assert len(items) == 2
        types = {i.record_type for i in items}
        assert types == {RecordType.CHUNK, RecordType.FILE}

    def test_excludes_failed_and_empty_summaries(self) -> None:
        state = RepositoryState(repository_path=__import__("pathlib").Path("."))
        state.chunk_docs = ChunkDocumentationCollection(
            docs=[
                ChunkDocumentation(chunk_id="a.py:1-3", file_path="a.py", summary="", model="fake", error="failed"),
                ChunkDocumentation(chunk_id="a.py:4-6", file_path="a.py", summary="", model="fake"),
            ]
        )
        items = _collect_items(state)
        assert items == []
