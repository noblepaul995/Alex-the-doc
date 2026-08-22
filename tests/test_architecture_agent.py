"""Tests for the Architecture Agent (agents/architecture_agent.py)."""

from __future__ import annotations

import pytest

from agents.architecture_agent import build_dependency_diagram, generate_architecture_narrative, select_central_files
from graph.dependency import DependencyEdge, DependencyGraph
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.architecture_agent as module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(stop=stop_after_attempt(max_attempts), wait=wait_fixed(0), retry=retry_if_exception_type(exceptions), reraise=True)

    monkeypatch.setattr(module, "with_retry", _fast_with_retry)


class _FakeProvider(BaseProvider):
    def __init__(self, *, response: str = "An architecture narrative.", fail_times: int = 0) -> None:
        self.calls: list[list[ChatMessage]] = []
        self._response = response
        self._fail_times = fail_times
        self._call_count = 0

    async def generate(self, messages, *, temperature=0.2, max_tokens=None) -> GenerationResult:
        self.calls.append(messages)
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ProviderError("simulated failure", provider="fake", retryable=True)
        return GenerationResult(text=self._response, model="fake-model", prompt_tokens=12, completion_tokens=6)

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def embed(self, texts):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-model")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-model", supports_streaming=False, supports_embeddings=False)


class TestSelectCentralFiles:
    def test_ranks_by_combined_degree(self) -> None:
        dg = DependencyGraph(
            edges=[
                DependencyEdge(source="a.py", module="c", targets=["c.py"]),
                DependencyEdge(source="b.py", module="c", targets=["c.py"]),
            ]
        )
        selected = select_central_files(dg, ["a.py", "b.py", "c.py"], max_nodes=1)
        assert selected == ["c.py"]  # 2 dependents, most central

    def test_returns_all_when_under_cap(self) -> None:
        dg = DependencyGraph()
        selected = select_central_files(dg, ["b.py", "a.py"], max_nodes=10)
        assert selected == ["a.py", "b.py"]  # stable alphabetical order


class TestBuildDependencyDiagram:
    def test_produces_valid_flowchart_header(self) -> None:
        diagram = build_dependency_diagram(["a.py"], DependencyGraph())
        assert diagram.startswith("flowchart TD")

    def test_includes_a_node_per_file(self) -> None:
        diagram = build_dependency_diagram(["a.py", "b.py"], DependencyGraph())
        assert 'n0["a.py"]' in diagram
        assert 'n1["b.py"]' in diagram

    def test_includes_edge_for_internal_dependency(self) -> None:
        dg = DependencyGraph(edges=[DependencyEdge(source="a.py", module="b", targets=["b.py"])])
        diagram = build_dependency_diagram(["a.py", "b.py"], dg)
        assert "n0 --> n1" in diagram

    def test_excludes_edge_to_file_outside_selection(self) -> None:
        dg = DependencyGraph(edges=[DependencyEdge(source="a.py", module="b", targets=["b.py"])])
        diagram = build_dependency_diagram(["a.py"], dg)  # b.py not selected
        assert "-->" not in diagram

    def test_sanitizes_quotes_in_file_paths(self) -> None:
        diagram = build_dependency_diagram(['weird"file.py'], DependencyGraph())
        assert '"' not in diagram.split("\n")[1].strip('n0[]').replace('"weird\'file.py"', "")  # no unescaped quote breaks the label
        assert "weird'file.py" in diagram

    def test_no_duplicate_edges(self) -> None:
        dg = DependencyGraph(
            edges=[
                DependencyEdge(source="a.py", module="b", targets=["b.py"]),
                DependencyEdge(source="a.py", module="b2", targets=["b.py"]),  # two imports, same target
            ]
        )
        diagram = build_dependency_diagram(["a.py", "b.py"], dg)
        assert diagram.count("n0 --> n1") == 1


class TestGenerateArchitectureNarrative:
    @pytest.mark.asyncio
    async def test_returns_narrative_and_tokens(self) -> None:
        from agents.architecture_agent import ArchitectureDigest

        digest = ArchitectureDigest(file_paths=["a.py"], diagram="flowchart TD", file_digest="- a.py: does a thing")
        provider = _FakeProvider(response="This is the architecture.")
        narrative, prompt_tokens, completion_tokens = await generate_architecture_narrative(digest, "repo summary", provider)
        assert narrative == "This is the architecture."
        assert prompt_tokens == 12
        assert completion_tokens == 6

    @pytest.mark.asyncio
    async def test_persistent_failure_raises(self) -> None:
        from agents.architecture_agent import ArchitectureDigest

        digest = ArchitectureDigest(file_paths=["a.py"], diagram="flowchart TD", file_digest="- a.py")
        provider = _FakeProvider(fail_times=99)
        with pytest.raises(ProviderError):
            await generate_architecture_narrative(digest, None, provider)
