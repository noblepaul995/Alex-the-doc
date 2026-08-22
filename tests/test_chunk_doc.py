"""Tests for the Chunk Documentation Agent (agents/chunk_doc.py)."""

from __future__ import annotations

import pytest

from agents.chunk_doc import _document_chunks
from graph.chunk import Chunk
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry logic itself is exercised for real; only the exponential-backoff sleep is skipped, to keep tests fast."""
    import agents.chunk_doc as chunk_doc_module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(0),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )

    monkeypatch.setattr(chunk_doc_module, "with_retry", _fast_with_retry)


def _chunk(chunk_id: str = "a.py:1-3", *, file_path: str = "a.py", symbol_names: list[str] | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        file_path=file_path,
        language="python",
        start_line=1,
        end_line=3,
        start_byte=0,
        end_byte=20,
        content="def foo():\n    return 1\n",
        symbol_names=symbol_names or ["foo"],
        token_estimate=10,
    )


class _FakeProvider(BaseProvider):
    """In-memory stand-in for a real provider — no network access."""

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
        text = self._responses[self._success_count] if self._success_count < len(self._responses) else "A default summary."
        self._success_count += 1
        return GenerationResult(text=text, model="fake-model", prompt_tokens=10, completion_tokens=5)

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def embed(self, texts):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-model")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-model", supports_streaming=False, supports_embeddings=False)


class TestDocumentChunks:
    @pytest.mark.asyncio
    async def test_generates_one_doc_per_chunk(self) -> None:
        provider = _FakeProvider(responses=["Summary one.", "Summary two."])
        chunks = [_chunk("a.py:1-3"), _chunk("a.py:4-6")]
        result = await _document_chunks(chunks, provider, max_concurrency=4)
        assert len(result.docs) == 2
        assert {d.chunk_id for d in result.docs} == {"a.py:1-3", "a.py:4-6"}

    @pytest.mark.asyncio
    async def test_summary_and_metadata_carried_through(self) -> None:
        provider = _FakeProvider(responses=["It returns 1."])
        chunks = [_chunk(symbol_names=["foo", "bar"])]
        result = await _document_chunks(chunks, provider, max_concurrency=1)
        doc = result.docs[0]
        assert doc.summary == "It returns 1."
        assert doc.symbol_names == ["foo", "bar"]
        assert doc.model == "fake-model"
        assert doc.prompt_tokens == 10
        assert doc.completion_tokens == 5
        assert doc.error is None

    @pytest.mark.asyncio
    async def test_prompt_includes_chunk_content_and_symbols(self) -> None:
        provider = _FakeProvider()
        chunks = [_chunk(symbol_names=["foo"])]
        await _document_chunks(chunks, provider, max_concurrency=1)
        prompt_text = provider.calls[0][0].content
        assert "def foo():" in prompt_text
        assert "foo" in prompt_text
        assert "a.py" in prompt_text

    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_and_succeeds(self) -> None:
        provider = _FakeProvider(responses=["Recovered."], fail_times=2)
        chunks = [_chunk()]
        result = await _document_chunks(chunks, provider, max_concurrency=1)
        assert result.docs[0].error is None
        assert result.docs[0].summary == "Recovered."
        assert provider._call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_persistent_failure_recorded_not_raised(self) -> None:
        provider = _FakeProvider(fail_times=99)
        chunks = [_chunk()]
        result = await _document_chunks(chunks, provider, max_concurrency=1)
        assert result.docs[0].error is not None
        assert result.docs[0].summary == ""

    @pytest.mark.asyncio
    async def test_one_failing_chunk_does_not_block_others(self) -> None:
        class _PartialFailProvider(_FakeProvider):
            async def generate(self, messages, *, temperature=0.2, max_tokens=None):
                if "bad.py" in messages[0].content:
                    raise ProviderError("nope", provider="fake", retryable=False)
                return await super().generate(messages, temperature=temperature, max_tokens=max_tokens)

        provider = _PartialFailProvider(responses=["fine"])
        chunks = [_chunk("bad.py:1-3", file_path="bad.py"), _chunk("good.py:1-3", file_path="good.py")]
        result = await _document_chunks(chunks, provider, max_concurrency=4)
        by_id = {d.chunk_id: d for d in result.docs}
        assert by_id["bad.py:1-3"].error is not None
        assert by_id["good.py:1-3"].error is None

    @pytest.mark.asyncio
    async def test_empty_chunk_list_returns_empty_collection(self) -> None:
        provider = _FakeProvider()
        result = await _document_chunks([], provider, max_concurrency=4)
        assert result.docs == []
