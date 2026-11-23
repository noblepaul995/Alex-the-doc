"""Tests for the live HTTP endpoint smoke-tester (tools/api_tester.py)."""

from __future__ import annotations

import httpx
import pytest

from agents.api_agent import EndpointCandidate, extract_method_and_path
from parsers.metadata import Span, Symbol, SymbolKind
from parsers.python_parser import parse_python_file
from tools.api_tester import build_test_plan, parse_kv_flags, run_test_plan, substitute_path_params


def _sym(name: str, decorators: list[str] | None = None) -> Symbol:
    return Symbol(
        name=name,
        kind=SymbolKind.FUNCTION,
        language="python",
        span=Span(start_byte=0, end_byte=1, start_line=0, end_line=0),
        decorators=decorators or [],
    )


class TestExtractMethodAndPath:
    def test_extracts_python_fastapi_decorator(self) -> None:
        src = b'@app.get("/items/{item_id}")\ndef read_item(item_id: int):\n    pass\n'
        pr = parse_python_file("api.py", src)
        cand = EndpointCandidate(file_path="api.py", symbol=pr.symbols[0])
        assert extract_method_and_path(cand) == ("GET", "/items/{item_id}")

    def test_extracts_route_hint(self) -> None:
        cand = EndpointCandidate(file_path="routes.ts", symbol=_sym("listUsers"), route_hint="POST /users")
        assert extract_method_and_path(cand) == ("POST", "/users")


class TestBuildTestPlan:
    def test_skips_writes_by_default_and_includes_with_flag(self) -> None:
        cands = [
            EndpointCandidate(file_path="api.py", symbol=_sym("list_items"), route_hint="GET /items"),
            EndpointCandidate(file_path="api.py", symbol=_sym("create_item"), route_hint="POST /items"),
        ]
        safe_plan = build_test_plan(cands, include_writes=False)
        by_method = {c.method: c for c in safe_plan}
        assert by_method["GET"].status == "PENDING"
        assert by_method["POST"].status == "SKIPPED"
        assert "write method" in (by_method["POST"].skip_reason or "")

        full_plan = build_test_plan(cands, include_writes=True)
        assert all(c.status == "PENDING" for c in full_plan)

    def test_path_params_substitute_or_skip(self) -> None:
        cands = [
            EndpointCandidate(file_path="api.py", symbol=_sym("get_item"), route_hint="GET /items/{item_id}"),
            EndpointCandidate(file_path="go.go", symbol=_sym("GetUser"), route_hint="GET /users/:id"),
        ]
        skipped_plan = build_test_plan(cands)
        assert all(c.status == "SKIPPED" for c in skipped_plan)
        assert "item_id" in (skipped_plan[0].skip_reason or "")
        assert "id" in (skipped_plan[1].skip_reason or "")

        resolved_plan = build_test_plan(cands, params={"item_id": "42", "id": "7"})
        assert resolved_plan[0].status == "PENDING"
        assert resolved_plan[0].resolved_path == "/items/42"
        assert resolved_plan[1].resolved_path == "/users/7"


class TestHelpers:
    def test_parse_kv_flags(self) -> None:
        assert parse_kv_flags(["item_id=42", "x=a=b"], sep="=") == {"item_id": "42", "x": "a=b"}
        assert parse_kv_flags(["Authorization: Bearer tok"], sep=":") == {"Authorization": "Bearer tok"}
        with pytest.raises(ValueError):
            parse_kv_flags(["bad_flag"], sep="=")

    def test_substitute_path_params_partial_missing(self) -> None:
        resolved, missing = substitute_path_params("/orgs/{org}/items/:id", {"org": "acme"})
        assert resolved is None
        assert missing == ["id"]


class TestRunTestPlan:
    @pytest.mark.asyncio
    async def test_runs_requests_and_forwards_auth_headers(self) -> None:
        seen_auth: list[str | None] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen_auth.append(request.headers.get("authorization"))
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(500, text="boom")

        cands = [
            EndpointCandidate(file_path="api.py", symbol=_sym("health"), route_hint="GET /health"),
            EndpointCandidate(file_path="api.py", symbol=_sym("broken"), route_hint="GET /broken"),
            EndpointCandidate(file_path="api.py", symbol=_sym("mutate"), route_hint="POST /items"),
        ]
        plan = build_test_plan(cands, include_writes=False)
        results = await run_test_plan(
            plan,
            "http://localhost:8000",
            headers={"Authorization": "Bearer secret"},
            transport=httpx.MockTransport(_handler),
        )
        by_route = {r.case.route_template: r for r in results}
        assert by_route["/health"].outcome == "PASS"
        assert by_route["/health"].status_code == 200
        assert by_route["/broken"].outcome == "FAIL"
        assert by_route["/broken"].status_code == 500
        assert by_route["/items"].outcome == "SKIPPED"
        assert seen_auth == ["Bearer secret", "Bearer secret"]
