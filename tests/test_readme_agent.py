"""Tests for the README Agent (agents/readme_agent.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.readme_agent import _fix_banned_phrases, _locate_sentence, generate_readme
from agents.repo_agent import build_digest
from graph.file_doc import FileDocumentation, FileDocumentationCollection
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError
from parsers.metadata import ParseResult, Span, Symbol, SymbolKind


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.readme_agent as module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(stop=stop_after_attempt(max_attempts), wait=wait_fixed(0), retry=retry_if_exception_type(exceptions), reraise=True)

    monkeypatch.setattr(module, "with_retry", _fast_with_retry)


def _state_with_file(tmp_path: Path, file_path: str = "a.py") -> RepositoryState:
    state = RepositoryState(repository_path=tmp_path)
    state.file_docs = FileDocumentationCollection(
        docs=[FileDocumentation(file_path=file_path, language="python", summary="Does a thing.", model="fake")]
    )
    symbol = Symbol(name="foo", kind=SymbolKind.FUNCTION, language="python", span=Span(start_byte=0, end_byte=1, start_line=0, end_line=0))
    state.parse_results[file_path] = ParseResult(file_path=file_path, language="python", symbols=[symbol])
    return state


class _FakeProvider(BaseProvider):
    def __init__(self, *, response: str = "# Project\n\n## Overview\n", responses: list[str] | None = None, fail_times: int = 0) -> None:
        self.calls: list[list[ChatMessage]] = []
        self._response = response
        self._responses = responses
        self._fail_times = fail_times
        self._call_count = 0

    async def generate(self, messages, *, temperature=0.2, max_tokens=None) -> GenerationResult:
        self.calls.append(messages)
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ProviderError("simulated failure", provider="fake", retryable=True)
        if self._responses is not None:
            index = min(self._call_count - 1, len(self._responses) - 1)
            text = self._responses[index]
        else:
            text = self._response
        return GenerationResult(text=text, model="fake-model", prompt_tokens=30, completion_tokens=50)

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def embed(self, texts):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-model")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-model", supports_streaming=False, supports_embeddings=False)


class TestGenerateReadme:
    @pytest.mark.asyncio
    async def test_returns_readme_and_tokens_when_no_issues_found(self, tmp_path: Path) -> None:
        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(responses=["# MyProject\n\n## Overview\nIt does things.", "No issues found."])
        readme_text, total_tokens, revision_count = await generate_readme(digest, "A repo that does things.", provider)
        assert readme_text.startswith("# MyProject")
        assert total_tokens == 160  # 2 calls * (30 prompt + 50 completion)
        assert revision_count == 0
        assert len(provider.calls) == 2  # draft + critique only, no revise call needed

    @pytest.mark.asyncio
    async def test_revision_runs_when_critique_flags_an_issue(self, tmp_path: Path) -> None:
        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(
            responses=[
                "# Draft\n\nX and Y run in parallel.",
                "- X and Y run in parallel.",
                "# Draft\n\nX and Y are both involved in processing.",
                "No issues found.",
            ]
        )
        readme_text, total_tokens, revision_count = await generate_readme(digest, "A repo that does things.", provider)
        assert readme_text == "# Draft\n\nX and Y are both involved in processing."
        assert revision_count == 1
        assert total_tokens == 320  # 4 calls * (30 + 50)
        assert len(provider.calls) == 4  # draft, critique(dirty), revise, critique(clean)

    @pytest.mark.asyncio
    async def test_loop_continues_across_multiple_dirty_rounds(self, tmp_path: Path) -> None:
        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(
            responses=[
                "# Draft v1",  # draft
                "- issue one",  # critique round 1: dirty
                "# Draft v2",  # revise round 1
                "- issue two",  # critique round 2: dirty
                "# Draft v3",  # revise round 2
                "No issues found.",  # critique round 3: clean
            ]
        )
        readme_text, total_tokens, revision_count = await generate_readme(digest, "A repo.", provider)
        assert readme_text == "# Draft v3"
        assert revision_count == 2
        assert len(provider.calls) == 6  # draft + 2*(critique+revise) + final clean critique

    @pytest.mark.asyncio
    async def test_loop_stops_at_cap_even_if_still_dirty(self, tmp_path: Path) -> None:
        from agents.readme_agent import MAX_CRITIQUE_ROUNDS

        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        # Every critique call flags an issue, every revise call bumps a counter into the text —
        # this never produces "No issues found.", so the loop must be stopped by the cap alone.
        provider = _FakeProvider(response="# Draft")

        class _AlwaysDirtyProvider(_FakeProvider):
            async def generate(self, messages, *, temperature=0.2, max_tokens=None):
                self.calls.append(messages)
                self._call_count += 1
                content = messages[0].content
                if "fact-checking" in content.lower():
                    text = "- always an issue"
                else:
                    text = f"# Draft (revision {self._call_count})"
                return GenerationResult(text=text, model="fake-model", prompt_tokens=1, completion_tokens=1)

        provider = _AlwaysDirtyProvider()
        readme_text, total_tokens, revision_count = await generate_readme(digest, "A repo.", provider)
        assert revision_count == MAX_CRITIQUE_ROUNDS
        # draft (1) + MAX_CRITIQUE_ROUNDS * (critique + revise), never hitting a clean critique
        assert len(provider.calls) == 1 + 2 * MAX_CRITIQUE_ROUNDS

    @pytest.mark.asyncio
    async def test_prompt_includes_repo_summary_and_digest(self, tmp_path: Path) -> None:
        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(responses=["# Draft", "No issues found."])
        await generate_readme(digest, "A repo that does things.", provider)
        prompt_text = provider.calls[0][0].content
        assert "A repo that does things." in prompt_text
        assert "a.py" in prompt_text

    @pytest.mark.asyncio
    async def test_prompt_forbids_inventing_install_commands(self, tmp_path: Path) -> None:
        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(responses=["# Draft", "No issues found."])
        await generate_readme(digest, None, provider)
        prompt_text = provider.calls[0][0].content
        assert "Do NOT invent installation commands" in prompt_text

    @pytest.mark.asyncio
    async def test_persistent_failure_raises(self, tmp_path: Path) -> None:
        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(fail_times=99)
        with pytest.raises(ProviderError):
            await generate_readme(digest, None, provider)


class TestLocateSentence:
    def test_finds_containing_sentence(self) -> None:
        text = "First sentence here. This promotes flexibility and maintainability. Third sentence."
        start, end, sentence = _locate_sentence(text, "promotes flexibility")
        assert sentence.strip() == "This promotes flexibility and maintainability."

    def test_returns_none_when_phrase_absent(self) -> None:
        assert _locate_sentence("Nothing generic here.", "promotes flexibility") is None

    def test_handles_phrase_at_start_of_text(self) -> None:
        text = "This promotes flexibility. Second sentence."
        start, end, sentence = _locate_sentence(text, "promotes flexibility")
        assert start == 0
        assert sentence.strip() == "This promotes flexibility."


class TestFixBannedPhrases:
    @pytest.mark.asyncio
    async def test_isolated_fix_succeeds_where_bundled_revise_previously_failed(self) -> None:
        # Reproduces the exact real-run failure mode: a banned phrase that survived
        # a bundled multi-issue revise call unchanged. The isolated micro-fix, given
        # only the one short sentence, should actually remove it.
        provider = _FakeProvider(
            responses=["The system caches parsed results by content hash to avoid redundant work."]
        )
        text = "Intro sentence. This promotes flexibility and maintainability. Closing sentence."
        fixed, tokens = await _fix_banned_phrases(text, provider)
        assert "promotes flexibility" not in fixed.lower()
        assert "Intro sentence." in fixed
        assert "Closing sentence." in fixed
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_no_op_when_no_banned_phrases_present(self) -> None:
        provider = _FakeProvider(response="should never be called")
        text = "Nothing generic in this sentence at all."
        fixed, tokens = await _fix_banned_phrases(text, provider)
        assert fixed == text
        assert tokens == 0
        assert len(provider.calls) == 0

    @pytest.mark.asyncio
    async def test_keeps_original_if_fix_still_contains_banned_phrasing(self) -> None:
        # If the model's rewrite still has a banned phrase, keep the original sentence
        # rather than splice in a "fix" that didn't actually fix anything.
        provider = _FakeProvider(response="This still promotes flexibility somehow.")
        text = "This promotes flexibility and maintainability."
        fixed, tokens = await _fix_banned_phrases(text, provider)
        assert fixed == text  # unchanged — the bad "fix" was rejected

    @pytest.mark.asyncio
    async def test_deduplicates_repeated_phrase_across_document(self) -> None:
        provider = _FakeProvider(response="A clean sentence.")
        text = "This promotes flexibility. Later, this also promotes flexibility again."
        fixed, tokens = await _fix_banned_phrases(text, provider)
        # Only one micro-fix call for the (deduplicated) phrase, not one per occurrence.
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_provider_failure_leaves_original_sentence_intact(self) -> None:
        provider = _FakeProvider(fail_times=99)
        text = "This promotes flexibility and maintainability."
        fixed, tokens = await _fix_banned_phrases(text, provider)
        assert fixed == text
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_preserves_paragraph_break_after_the_fixed_sentence(self) -> None:
        # Regression test: an earlier version of the splice logic stripped the tail's
        # leading whitespace, collapsing a blank-line paragraph break (e.g. before a
        # "## Heading") onto the same line as the fixed sentence.
        provider = _FakeProvider(response="The pipeline caches parsed results by content hash.")
        text = "Intro.\n\nThis promotes flexibility and maintainability.\n\n## Core Components\n\nStuff."
        fixed, tokens = await _fix_banned_phrases(text, provider)
        assert "content hash.\n\n## Core Components" in fixed

    @pytest.mark.asyncio
    async def test_end_to_end_readme_generation_actually_removes_known_banned_phrase(self, tmp_path: Path) -> None:
        # The real regression this whole mechanism exists for: a banned phrase that
        # a bundled critique/revise call didn't remove should be gone from the final
        # document once the isolated micro-fix pass runs.
        state = _state_with_file(tmp_path)
        digest = build_digest(state, max_files=40)
        provider = _FakeProvider(
            responses=[
                "# Demo\n\nThis promotes flexibility and maintainability across the pipeline.",  # draft
                "The pipeline stays flexible and easy to maintain across components.",  # micro-fix
                "No issues found.",  # critique (clean after micro-fix)
            ]
        )
        readme_text, total_tokens, revision_count = await generate_readme(digest, "A repo.", provider)
        assert "promotes flexibility" not in readme_text.lower()
        assert "The pipeline stays flexible" in readme_text
