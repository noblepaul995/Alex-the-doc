"""Tests for the File Documentation Agent (agents/file_doc.py)."""

from __future__ import annotations

import pytest

from agents.file_doc import _document_files
from graph.chunk import Chunk, ChunkDocumentation, ChunkDocumentationCollection
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError
from parsers.metadata import ParseResult, Span, Symbol, SymbolKind


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.file_doc as file_doc_module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(0),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )

    monkeypatch.setattr(file_doc_module, "with_retry", _fast_with_retry)


def _chunk(chunk_id: str, file_path: str, start_line: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        file_path=file_path,
        language="python",
        start_line=start_line,
        end_line=start_line + 2,
        start_byte=0,
        end_byte=10,
        content="def foo(): pass",
        token_estimate=5,
    )


def _chunk_doc(chunk_id: str, file_path: str, summary: str = "Does a thing.", *, error: str | None = None) -> ChunkDocumentation:
    return ChunkDocumentation(chunk_id=chunk_id, file_path=file_path, summary=summary, model="fake", error=error)


def _parse_result(file_path: str, symbol_names: list[str]) -> ParseResult:
    symbols = [
        Symbol(name=n, kind=SymbolKind.FUNCTION, language="python", span=Span(start_byte=0, end_byte=1, start_line=0, end_line=0))
        for n in symbol_names
    ]
    return ParseResult(file_path=file_path, language="python", symbols=symbols)


class _FakeProvider(BaseProvider):
    def __init__(self, *, responses: list[str] | None = None, fail_times: int = 0) -> None:
        self.calls: list[list[ChatMessage]] = []
        self._responses = responses or []
        self._fail_times = fail_times
        self._call_count = 0
        self._success_count = 0

    async def generate(self, messages, *, temperature=0.2, max_tokens=None) -> GenerationResult:
        self.calls.append(messages)
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ProviderError("simulated failure", provider="fake", retryable=True)
        text = self._responses[self._success_count] if self._success_count < len(self._responses) else "A default file summary."
        self._success_count += 1
        return GenerationResult(text=text, model="fake-model", prompt_tokens=20, completion_tokens=10)

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def embed(self, texts):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-model")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-model", supports_streaming=False, supports_embeddings=False)


class TestDocumentFiles:
    @pytest.mark.asyncio
    async def test_groups_chunk_docs_by_file(self) -> None:
        provider = _FakeProvider(responses=["File one summary.", "File two summary."])
        chunk_docs = ChunkDocumentationCollection(
            docs=[
                _chunk_doc("a.py:1-3", "a.py"),
                _chunk_doc("a.py:4-6", "a.py"),
                _chunk_doc("b.py:1-3", "b.py"),
            ]
        )
        chunks = [_chunk("a.py:1-3", "a.py", 1), _chunk("a.py:4-6", "a.py", 4), _chunk("b.py:1-3", "b.py", 1)]
        result = await _document_files(chunk_docs, chunks, {}, provider, max_concurrency=4)
        assert {d.file_path for d in result.docs} == {"a.py", "b.py"}
        a_doc = result.by_path("a.py")
        assert a_doc.chunk_count == 2

    @pytest.mark.asyncio
    async def test_prompt_includes_chunk_summaries_in_line_order(self) -> None:
        provider = _FakeProvider()
        chunk_docs = ChunkDocumentationCollection(
            docs=[
                _chunk_doc("a.py:10-12", "a.py", "Second chunk summary."),
                _chunk_doc("a.py:1-3", "a.py", "First chunk summary."),
            ]
        )
        chunks = [_chunk("a.py:10-12", "a.py", 10), _chunk("a.py:1-3", "a.py", 1)]
        await _document_files(chunk_docs, chunks, {}, provider, max_concurrency=1)
        prompt_text = provider.calls[0][0].content
        assert prompt_text.index("First chunk summary.") < prompt_text.index("Second chunk summary.")

    @pytest.mark.asyncio
    async def test_symbol_names_come_from_parse_result(self) -> None:
        provider = _FakeProvider()
        chunk_docs = ChunkDocumentationCollection(docs=[_chunk_doc("a.py:1-3", "a.py")])
        chunks = [_chunk("a.py:1-3", "a.py")]
        parse_results = {"a.py": _parse_result("a.py", ["foo", "Bar"])}
        result = await _document_files(chunk_docs, chunks, parse_results, provider, max_concurrency=1)
        assert set(result.docs[0].symbol_names) == {"foo", "Bar"}
        assert "foo" in provider.calls[0][0].content

    @pytest.mark.asyncio
    async def test_file_with_all_chunks_failed_skips_llm_call(self) -> None:
        provider = _FakeProvider()
        chunk_docs = ChunkDocumentationCollection(docs=[_chunk_doc("a.py:1-3", "a.py", error="boom")])
        chunks = [_chunk("a.py:1-3", "a.py")]
        result = await _document_files(chunk_docs, chunks, {}, provider, max_concurrency=1)
        assert result.docs[0].error is not None
        assert provider.calls == []  # never called — nothing to synthesize

    @pytest.mark.asyncio
    async def test_file_with_some_chunks_failed_uses_only_successful_ones(self) -> None:
        provider = _FakeProvider(responses=["Synthesized."])
        chunk_docs = ChunkDocumentationCollection(
            docs=[
                _chunk_doc("a.py:1-3", "a.py", "Good summary.", error=None),
                _chunk_doc("a.py:4-6", "a.py", error="failed"),
            ]
        )
        chunks = [_chunk("a.py:1-3", "a.py", 1), _chunk("a.py:4-6", "a.py", 4)]
        result = await _document_files(chunk_docs, chunks, {}, provider, max_concurrency=1)
        assert result.docs[0].error is None
        assert "Good summary." in provider.calls[0][0].content

    @pytest.mark.asyncio
    async def test_persistent_failure_recorded_not_raised(self) -> None:
        provider = _FakeProvider(fail_times=99)
        chunk_docs = ChunkDocumentationCollection(docs=[_chunk_doc("a.py:1-3", "a.py")])
        chunks = [_chunk("a.py:1-3", "a.py")]
        result = await _document_files(chunk_docs, chunks, {}, provider, max_concurrency=1)
        assert result.docs[0].error is not None

    @pytest.mark.asyncio
    async def test_empty_chunk_docs_returns_empty_collection(self) -> None:
        provider = _FakeProvider()
        result = await _document_files(ChunkDocumentationCollection(), [], {}, provider, max_concurrency=4)
        assert result.docs == []
