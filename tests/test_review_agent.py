"""Tests for the Review Agent (agents/review_agent.py + graph/review.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.review_agent import ReviewItem, _apply_deterministic_checks, build_review_items, parse_review_response, review_documents
from graph.review import ReviewCollection, ReviewResult
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError
from parsers.metadata import ParseResult, Span, Symbol, SymbolKind


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.review_agent as module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(stop=stop_after_attempt(max_attempts), wait=wait_fixed(0), retry=retry_if_exception_type(exceptions), reraise=True)

    monkeypatch.setattr(module, "with_retry", _fast_with_retry)


class _FakeProvider(BaseProvider):
    def __init__(self, *, response: str = "STATUS: PASS", fail_times: int = 0) -> None:
        self.calls: list[list[ChatMessage]] = []
        self._response = response
        self._fail_times = fail_times
        self._call_count = 0

    async def generate(self, messages, *, temperature=0.2, max_tokens=None) -> GenerationResult:
        self.calls.append(messages)
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ProviderError("simulated failure", provider="fake", retryable=True)
        return GenerationResult(text=self._response, model="fake-model", prompt_tokens=10, completion_tokens=5)

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def embed(self, texts):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-model")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-model", supports_streaming=False, supports_embeddings=False)


class TestParseReviewResponse:
    def test_pass_with_no_issues(self) -> None:
        passed, issues = parse_review_response("STATUS: PASS\n\nNo issues found.")
        assert passed is True
        assert issues == []

    def test_issues_found_with_bullets(self) -> None:
        text = "STATUS: ISSUES_FOUND\n\n- Claims GraphQL support, not in grounding.\n- Minor grammar issue."
        passed, issues = parse_review_response(text)
        assert passed is False
        assert len(issues) == 2

    def test_case_insensitive_status(self) -> None:
        passed, _ = parse_review_response("Status: pass")
        assert passed is True

    def test_issues_found_but_no_bullets_gets_fallback_message(self) -> None:
        passed, issues = parse_review_response("STATUS: ISSUES_FOUND\nSomething's wrong but no list.")
        assert passed is False
        assert issues == ["Review flagged issues but did not list specifics."]

    def test_no_status_line_defaults_to_passed(self) -> None:
        passed, issues = parse_review_response("Just some prose with no status line.")
        assert passed is True
        assert issues == []


class TestBuildReviewItems:
    def test_architecture_and_readme_grounded_in_repo_summary(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.repo_summary = "A summary."
        state.generated_docs = {"architecture": "Architecture doc.", "readme": "# README"}
        items = build_review_items(state)
        names = {i.doc_name for i in items}
        assert names == {"architecture", "readme"}
        assert all(i.grounding == "A summary." for i in items)

    def test_skips_docs_without_repo_summary(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"architecture": "Architecture doc."}
        assert build_review_items(state) == []

    def test_api_doc_grounded_in_freshly_rebuilt_endpoint_digest(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        symbol = Symbol(
            name="get_items",
            kind=SymbolKind.FUNCTION,
            language="python",
            span=Span(start_byte=0, end_byte=1, start_line=0, end_line=0),
            decorators=['@app.get("/items")'],
        )
        state.parse_results["api.py"] = ParseResult(file_path="api.py", language="python", symbols=[symbol])
        state.generated_docs = {"api": "# API Reference"}
        items = build_review_items(state)
        assert len(items) == 1
        assert items[0].doc_name == "api"
        assert "get_items" in items[0].grounding

    def test_api_doc_skipped_when_no_endpoints_found(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"api": "# API Reference"}  # doc exists but no candidates re-derivable
        assert build_review_items(state) == []

    def test_no_generated_docs_yields_no_items(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.repo_summary = "A summary."
        assert build_review_items(state) == []


class TestReviewDocuments:
    @pytest.mark.asyncio
    async def test_one_result_per_item(self) -> None:
        provider = _FakeProvider()
        items = [ReviewItem("architecture", "doc text", "grounding"), ReviewItem("readme", "doc text 2", "grounding")]
        collection = await review_documents(items, provider, max_concurrency=4)
        assert len(collection.results) == 2

    @pytest.mark.asyncio
    async def test_passed_and_flagged_split_correctly(self) -> None:
        class _MixedProvider(_FakeProvider):
            async def generate(self, messages, *, temperature=0.2, max_tokens=None):
                if "bad_doc" in messages[0].content:
                    return GenerationResult(text="STATUS: ISSUES_FOUND\n- unsupported claim", model="fake", prompt_tokens=1, completion_tokens=1)
                return await super().generate(messages, temperature=temperature, max_tokens=max_tokens)

        provider = _MixedProvider()
        items = [ReviewItem("good", "fine text", "g"), ReviewItem("bad_doc", "bad text", "g")]
        collection = await review_documents(items, provider, max_concurrency=4)
        assert len(collection.flagged_documents()) == 1
        assert collection.by_doc("bad_doc").passed is False
        assert collection.by_doc("good").passed is True

    @pytest.mark.asyncio
    async def test_persistent_failure_recorded_not_raised(self) -> None:
        provider = _FakeProvider(fail_times=99)
        items = [ReviewItem("architecture", "doc text", "grounding")]
        collection = await review_documents(items, provider, max_concurrency=1)
        assert collection.results[0].error is not None
        assert len(collection.failed_reviews()) == 1

    @pytest.mark.asyncio
    async def test_transient_failure_recovers(self) -> None:
        provider = _FakeProvider(fail_times=2, response="STATUS: PASS")
        items = [ReviewItem("architecture", "doc text", "grounding")]
        collection = await review_documents(items, provider, max_concurrency=1)
        assert collection.results[0].error is None
        assert collection.results[0].passed is True


class TestApplyDeterministicChecks:
    def _state(self, tmp_path: Path, generated_docs: dict[str, str]) -> RepositoryState:
        from graph.state import ProjectFile

        state = RepositoryState(repository_path=tmp_path)
        state.file_metadata["agents/repo_agent.py"] = ProjectFile(
            path=tmp_path / "agents" / "repo_agent.py", relative_path="agents/repo_agent.py", size_bytes=10
        )
        state.generated_docs = generated_docs
        return state

    def test_clean_document_produces_no_new_result(self, tmp_path: Path) -> None:
        state = self._state(tmp_path, {"readme": "See `agents/repo_agent.py` for the digest builder."})
        collection = _apply_deterministic_checks(ReviewCollection(), state)
        assert collection.results == []

    def test_banned_phrase_creates_a_failing_result(self, tmp_path: Path) -> None:
        state = self._state(tmp_path, {"readme": "This promotes flexibility and maintainability."})
        collection = _apply_deterministic_checks(ReviewCollection(), state)
        result = collection.by_doc("readme")
        assert result.passed is False
        assert any("banned generic phrase" in issue for issue in result.issues)

    def test_nonexistent_file_reference_creates_a_failing_result(self, tmp_path: Path) -> None:
        state = self._state(tmp_path, {"readme": "See `agents/fake_agent.py` for details."})
        collection = _apply_deterministic_checks(ReviewCollection(), state)
        result = collection.by_doc("readme")
        assert result.passed is False
        assert any("doesn't exist" in issue for issue in result.issues)

    def test_merges_into_existing_llm_review_result_rather_than_overwriting(self, tmp_path: Path) -> None:
        existing = ReviewResult(doc_name="readme", passed=False, issues=["LLM-found issue"], model="fake-model")
        state = self._state(tmp_path, {"readme": "This promotes flexibility and maintainability."})
        collection = _apply_deterministic_checks(ReviewCollection(results=[existing]), state)
        result = collection.by_doc("readme")
        assert "LLM-found issue" in result.issues
        assert any("banned generic phrase" in issue for issue in result.issues)

    def test_covers_documents_with_no_llm_review_at_all(self, tmp_path: Path) -> None:
        # "structure" and "changelog" never go through build_review_items (deterministic
        # themselves), but a banned phrase or bad reference in them should still be caught.
        state = self._state(tmp_path, {"structure": "This promotes flexibility and maintainability."})
        collection = _apply_deterministic_checks(ReviewCollection(), state)
        assert collection.by_doc("structure").passed is False
