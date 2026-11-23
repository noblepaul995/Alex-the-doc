"""Tests for the API Agent (agents/api_agent.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.api_agent import build_endpoint_digest, detect_endpoints, generate_api_reference
from llm.base import BaseProvider, ChatMessage, GenerationResult, HealthStatus, ModelInfo, ProviderError
from parsers.go_parser import parse_go_file
from parsers.python_parser import parse_python_file
from parsers.rust_parser import parse_rust_file
from parsers.ts_parser import parse_typescript_file


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
        # Plain Go functions without route registrations produce no candidates.
        src = b'package main\n\nfunc handler() {}\n'
        parse_result = parse_go_file("main.go", src)
        assert detect_endpoints({"main.go": parse_result}) == []

    def test_detects_nextjs_app_router_endpoints(self) -> None:
        src = b"export async function GET(req: Request): Promise<Response> { return new Response('ok'); }\n"
        parse_result = parse_typescript_file("src/app/users/route.ts", src)
        candidates = detect_endpoints({"src/app/users/route.ts": parse_result})
        assert len(candidates) == 1
        assert candidates[0].route_hint == "GET /users"

    def test_detects_express_routes_from_source(self, tmp_path: Path) -> None:
        src = 'function listUsers(req, res) { res.json([]); }\napp.get("/users", listUsers);\n'
        (tmp_path / "routes.ts").write_text(src, encoding="utf-8")
        parse_result = parse_typescript_file("routes.ts", src.encode("utf-8"))
        candidates = detect_endpoints({"routes.ts": parse_result}, repository_root=tmp_path)
        assert len(candidates) == 1
        assert candidates[0].route_hint == "GET /users"
        assert candidates[0].symbol.name == "listUsers"

    def test_detects_go_gin_and_net_http_routes(self, tmp_path: Path) -> None:
        src = (
            'package main\n\n'
            'func GetUser(c *gin.Context) {}\n'
            'func Health(w http.ResponseWriter, r *http.Request) {}\n'
            'func main() {\n'
            '    r.GET("/users/:id", GetUser)\n'
            '    http.HandleFunc("/health", Health)\n'
            '}\n'
        )
        (tmp_path / "main.go").write_text(src, encoding="utf-8")
        parse_result = parse_go_file("main.go", src.encode("utf-8"))
        candidates = detect_endpoints({"main.go": parse_result}, repository_root=tmp_path)
        hints = {c.route_hint for c in candidates}
        assert "GET /users/:id" in hints
        assert "HANDLE /health" in hints

    def test_detects_rust_actix_and_axum_routes(self, tmp_path: Path) -> None:
        src = (
            '#[get("/items")]\n'
            'async fn list_items() -> String { String::new() }\n'
            'async fn create_item() {}\n'
            'fn router() { app.route("/items", post(create_item)); }\n'
        )
        (tmp_path / "main.rs").write_text(src, encoding="utf-8")
        parse_result = parse_rust_file("main.rs", src.encode("utf-8"))
        candidates = detect_endpoints({"main.rs": parse_result}, repository_root=tmp_path)
        hints = {c.route_hint for c in candidates}
        assert "GET /items" in hints
        assert "POST /items" in hints


class TestBuildEndpointDigest:
    def test_includes_signature_parameters_return_type_and_docstring(self) -> None:
        src = b'@app.get("/x")\ndef foo(y: int) -> bool:\n    """Does a thing."""\n    return True\n'
        parse_result = parse_python_file("api.py", src)
        candidates = detect_endpoints({"api.py": parse_result})
        digest = build_endpoint_digest(candidates)
        assert "@app.get" in digest
        assert "def foo(y: int) -> bool" in digest
        assert "Parameters: y: int" in digest
        assert "Return type: bool" in digest
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
