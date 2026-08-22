"""Tests for the API Agent (agents/api_agent.py)."""

from __future__ import annotations

import pytest

from agents.api_agent import build_endpoint_digest, detect_endpoints, generate_api_reference
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError
from parsers.go_parser import parse_go_file
from parsers.python_parser import parse_python_file


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.api_agent as module
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

    def _fast_with_retry(*, max_attempts=3, exceptions=(Exception,), min_wait=1.0, max_wait=20.0):
        return AsyncRetrying(stop=stop_after_attempt(max_attempts), wait=wait_fixed(0), retry=retry_if_exception_type(exceptions), reraise=True)

    monkeypatch.setattr(module, "with_retry", _fast_with_retry)


class _FakeProvider(BaseProvider):
    def __init__(self, *, response: str = "# API Reference", fail_times: int = 0) -> None:
        self.calls: list[list[ChatMessage]] = []
        self._response = response
        self._fail_times = fail_times
        self._call_count = 0

    async def generate(self, messages, *, temperature=0.2, max_tokens=None) -> GenerationResult:
        self.calls.append(messages)
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ProviderError("simulated failure", provider="fake", retryable=True)
        return GenerationResult(text=self._response, model="fake-model", prompt_tokens=20, completion_tokens=40)

    def stream(self, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError

    async def embed(self, texts):
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="fake", model="fake-model")

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="fake", model="fake-model", supports_streaming=False, supports_embeddings=False)


class TestDetectEndpoints:
    def test_detects_fastapi_style_decorators(self) -> None:
        src = b'@app.get("/items/{item_id}")\ndef read_item(item_id: int):\n    """Get an item."""\n    return {}\n'
        parse_result = parse_python_file("api.py", src)
        candidates = detect_endpoints({"api.py": parse_result})
        assert len(candidates) == 1
        assert candidates[0].symbol.name == "read_item"

    def test_detects_flask_route_decorator(self) -> None:
        src = b'@app.route("/items", methods=["GET"])\ndef list_items():\n    pass\n'
        parse_result = parse_python_file("api.py", src)
        candidates = detect_endpoints({"api.py": parse_result})
        assert len(candidates) == 1

    def test_detects_drf_api_view_decorator(self) -> None:
        src = b'@api_view(["GET", "POST"])\ndef items(request):\n    pass\n'
        parse_result = parse_python_file("api.py", src)
        candidates = detect_endpoints({"api.py": parse_result})
        assert len(candidates) == 1

    def test_ignores_non_route_decorators(self) -> None:
        src = b"@staticmethod\ndef helper():\n    pass\n\n@property\ndef value(self):\n    pass\n"
        parse_result = parse_python_file("api.py", src)
        candidates = detect_endpoints({"api.py": parse_result})
        assert candidates == []

    def test_no_endpoints_when_no_decorators_at_all(self) -> None:
        src = b"def plain_function():\n    pass\n"
        parse_result = parse_python_file("api.py", src)
        assert detect_endpoints({"api.py": parse_result}) == []

    def test_go_files_never_produce_candidates_decorators_not_supported(self) -> None:
        # Real, documented limitation: only Python captures decorators at all.
        src = b'package main\n\nfunc handler() {}\n'
        parse_result = parse_go_file("main.go", src)
        assert detect_endpoints({"main.go": parse_result}) == []


class TestBuildEndpointDigest:
    def test_includes_signature_decorator_and_docstring(self) -> None:
        src = b'@app.get("/x")\ndef foo(y: int) -> bool:\n    """Does a thing."""\n    return True\n'
        parse_result = parse_python_file("api.py", src)
        candidates = detect_endpoints({"api.py": parse_result})
        digest = build_endpoint_digest(candidates)
        assert "@app.get" in digest
        assert "def foo(y: int) -> bool" in digest
        assert "Does a thing." in digest

    def test_missing_docstring_noted_not_invented(self) -> None:
        src = b'@app.get("/x")\ndef foo():\n    pass\n'
        parse_result = parse_python_file("api.py", src)
        candidates = detect_endpoints({"api.py": parse_result})
        digest = build_endpoint_digest(candidates)
        assert "(no docstring)" in digest


class TestGenerateApiReference:
    @pytest.mark.asyncio
    async def test_returns_reference_and_tokens(self) -> None:
        provider = _FakeProvider(response="# Endpoints\n\n## GET /items")
        reference, prompt_tokens, completion_tokens = await generate_api_reference("- endpoint digest", provider)
        assert "GET /items" in reference
        assert prompt_tokens == 20
        assert completion_tokens == 40

    @pytest.mark.asyncio
    async def test_persistent_failure_raises(self) -> None:
        provider = _FakeProvider(fail_times=99)
        with pytest.raises(ProviderError):
            await generate_api_reference("- endpoint digest", provider)
