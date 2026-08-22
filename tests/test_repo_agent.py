"""Tests for the Repository Agent (agents/repo_agent.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.repo_agent import build_digest, summarize_repository
from graph.dependency import DependencyEdge, DependencyGraph
from graph.file_doc import FileDocumentation, FileDocumentationCollection
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError
from parsers.metadata import ParseResult, Span, Symbol, SymbolKind


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.repo_agent as repo_agent_module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(0),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )

    monkeypatch.setattr(repo_agent_module, "with_retry", _fast_with_retry)


def _state_with_files(tmp_path: Path, file_paths: list[str], *, dependents: dict[str, list[str]] | None = None) -> RepositoryState:
    state = RepositoryState(repository_path=tmp_path)
    state.file_docs = FileDocumentationCollection(
        docs=[FileDocumentation(file_path=p, language="python", summary=f"Summary of {p}.", model="fake") for p in file_paths]
    )
    for p in file_paths:
        symbol = Symbol(name="foo", kind=SymbolKind.FUNCTION, language="python", span=Span(start_byte=0, end_byte=1, start_line=0, end_line=0))
        state.parse_results[p] = ParseResult(file_path=p, language="python", symbols=[symbol])

    edges = []
    if dependents:
        for target, sources in dependents.items():
            for source in sources:
                edges.append(DependencyEdge(source=source, module=target, targets=[target]))
    state.dependency_graph = DependencyGraph(edges=edges)
    return state


class _FakeProvider(BaseProvider):
    def __init__(self, *, response: str = "A repository summary.", fail_times: int = 0) -> None:
        self.calls: list[list[ChatMessage]] = []
        self._response = response
        self._fail_times = fail_times
        self._call_count = 0

    async def generate(self, messages, *, temperature=0.2, max_tokens=None) -> GenerationResult:
        self.calls.append(messages)
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ProviderError("simulated failure", provider="fake", retryable=True)
        return GenerationResult(text=self._response, model="fake-model", prompt_tokens=15, completion_tokens=10)

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def embed(self, texts):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-model")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-model", supports_streaming=False, supports_embeddings=False)


class TestBuildDigest:
    def test_returns_none_when_no_successful_docs(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        assert build_digest(state, max_files=40) is None

    def test_includes_all_files_under_the_cap(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, ["a.py", "b.py"])
        digest = build_digest(state, max_files=40)
        assert digest.file_count == 2
        assert digest.truncated is False
        assert "a.py" in digest.file_digest
        assert "b.py" in digest.file_digest

    def test_truncates_and_prioritizes_by_dependents(self, tmp_path: Path) -> None:
        # c.py has the most dependents, should survive truncation to 1 file.
        state = _state_with_files(
            tmp_path,
            ["a.py", "b.py", "c.py"],
            dependents={"c.py": ["a.py", "b.py"]},  # both a and b import c
        )
        digest = build_digest(state, max_files=1)
        assert digest.truncated is True
        assert "c.py" in digest.file_digest
        assert "a.py" not in digest.file_digest
        assert "b.py" not in digest.file_digest

    def test_excludes_failed_file_docs(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.file_docs = FileDocumentationCollection(
            docs=[FileDocumentation(file_path="bad.py", language="python", summary="", model="fake", error="failed")]
        )
        assert build_digest(state, max_files=40) is None

    def test_language_breakdown_reflects_parse_results(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, ["a.py"])
        digest = build_digest(state, max_files=40)
        assert "python: 1" in digest.language_breakdown

    def test_symbol_count_sums_across_files(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, ["a.py", "b.py"])
        digest = build_digest(state, max_files=40)
        assert digest.symbol_count == 2  # one "foo" symbol per file


class TestSummarizeRepository:
    @pytest.mark.asyncio
    async def test_returns_summary_and_token_counts(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, ["a.py"])
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(response="This repo does X.")
        summary, prompt_tokens, completion_tokens = await summarize_repository(digest, provider)
        assert summary == "This repo does X."
        assert prompt_tokens == 15
        assert completion_tokens == 10

    @pytest.mark.asyncio
    async def test_prompt_includes_digest_content(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, ["a.py"])
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider()
        await summarize_repository(digest, provider)
        prompt_text = provider.calls[0][0].content
        assert "a.py" in prompt_text
        assert "python: 1" in prompt_text

    @pytest.mark.asyncio
    async def test_transient_failure_recovers_via_retry(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, ["a.py"])
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(fail_times=2)
        summary, _, _ = await summarize_repository(digest, provider)
        assert summary == "A repository summary."

    @pytest.mark.asyncio
    async def test_persistent_failure_raises_provider_error(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, ["a.py"])
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(fail_times=99)
        with pytest.raises(ProviderError):
            await summarize_repository(digest, provider)
