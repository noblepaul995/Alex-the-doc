"""
API Agent — documents HTTP endpoints detected from Python route
decorators, Next.js App Router exports, Express/Fastify route calls,
Go (`net/http`, Gin, Echo) handlers, and Rust (Actix-web, Axum) routes.

A repository with no matching endpoints produces no API document at all,
rather than asking the LLM to invent an API surface that doesn't exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config.prompts import render_prompt
from config.providers import resolve_provider_config
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from parsers.metadata import ParseResult, Span, Symbol, SymbolKind
from utils.helpers import with_retry
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)

_ROUTE_MARKERS = (".route(", ".get(", ".post(", ".put(", ".delete(", ".patch(", "api_view", "api_route")
_NEXTJS_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_NEXTJS_APP_ROUTE_RE = re.compile(r"(?:^|/)app/(.*/)?route\.(?:js|jsx|ts|tsx)$")

_JS_ROUTE_RE = re.compile(
    r"""\b(?:app|router|fastify|server)\.(get|post|put|delete|patch|all)\(\s*['"]([^'"]+)['"](?:\s*,\s*([A-Za-z_$][\w$]*))?""",
    re.IGNORECASE,
)
_GO_HTTP_HANDLE_RE = re.compile(
    r"""HandleFunc\(\s*"([^"]+)"\s*,\s*([A-Za-z_][\w.]*)"""
)
_GO_FRAMEWORK_ROUTE_RE = re.compile(
    r"""\.(GET|POST|PUT|DELETE|PATCH|Get|Post|Put|Delete|Patch)\(\s*"([^"]+)"\s*,\s*([A-Za-z_][\w.]*)"""
)
_RUST_ATTR_ROUTE_RE = re.compile(
    r"""#\[(get|post|put|delete|patch)\(\s*"([^"]+)"\s*\)\]\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)""",
    re.IGNORECASE,
)
_RUST_AXUM_ROUTE_RE = re.compile(
    r"""\.route\(\s*"([^"]+)"\s*,\s*(?:[\w:]+::)?(get|post|put|delete|patch)\(\s*([A-Za-z_][\w:]*)\s*\)"""
)


@dataclass
class EndpointCandidate:
    file_path: str
    symbol: Symbol
    route_hint: str | None = None


def _looks_like_route(decorator: str) -> bool:
    lowered = decorator.lower()
    return any(marker in lowered for marker in _ROUTE_MARKERS)


def _read_source(repository_root: Path | None, file_path: str) -> str | None:
    if repository_root is None:
        return None
    full_path = repository_root / file_path
    if not full_path.is_file():
        return None
    try:
        return full_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _synthetic_symbol(name: str, language: str) -> Symbol:
    return Symbol(
        name=name,
        kind=SymbolKind.FUNCTION,
        language=language,
        span=Span(start_byte=0, end_byte=0, start_line=0, end_line=0),
    )


def _find_symbol_by_name(result: ParseResult, raw_name: str | None) -> Symbol | None:
    if not raw_name:
        return None
    short = raw_name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    for sym in result.symbols:
        if sym.name == raw_name or sym.name == short:
            return sym
    return None


def detect_endpoints(
    parse_results: dict[str, ParseResult],
    *,
    repository_root: Path | None = None,
) -> list[EndpointCandidate]:
    """
    Find HTTP endpoint candidates across Python, JS/TS, Go, and Rust files.
    When `repository_root` is provided, also inspects JS/TS, Go, and Rust
    source files for framework route registrations (Express, Fastify, Gin,
    Echo, `net/http`, Actix-web, Axum).
    """
    candidates: list[EndpointCandidate] = []
    seen_keys: set[tuple[str, str, str | None]] = set()

    def _add(file_path: str, symbol: Symbol, route_hint: str | None = None) -> None:
        key = (file_path, symbol.name, route_hint)
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(EndpointCandidate(file_path=file_path, symbol=symbol, route_hint=route_hint))

    for file_path, result in parse_results.items():
        lang = result.language.lower()

        # 1. Decorator-based (Python + any Rust/TS decorators populated by parsers)
        for symbol in result.symbols:
            if any(_looks_like_route(d) for d in symbol.decorators):
                _add(file_path, symbol)

        # 2. Next.js App Router (`app/.../route.ts` exporting GET/POST/...)
        if lang in ("javascript", "typescript", "tsx"):
            normalized = file_path.replace("\\", "/")
            app_match = _NEXTJS_APP_ROUTE_RE.search(normalized)
            if app_match:
                route_path = "/" + (app_match.group(1) or "").rstrip("/")
                for symbol in result.symbols:
                    if symbol.name in _NEXTJS_HTTP_METHODS:
                        _add(file_path, symbol, route_hint=f"{symbol.name} {route_path}")

            source = _read_source(repository_root, file_path)
            if source:
                for match in _JS_ROUTE_RE.finditer(source):
                    method, route_path, handler_name = match.group(1).upper(), match.group(2), match.group(3)
                    sym = _find_symbol_by_name(result, handler_name) or _synthetic_symbol(
                        handler_name or f"{method.lower()}_{route_path}", result.language
                    )
                    _add(file_path, sym, route_hint=f"{method} {route_path}")

        # 3. Go (`net/http`, Gin, Echo, Chi)
        elif lang == "go":
            source = _read_source(repository_root, file_path)
            if source:
                for match in _GO_HTTP_HANDLE_RE.finditer(source):
                    route_path, handler_name = match.group(1), match.group(2)
                    sym = _find_symbol_by_name(result, handler_name) or _synthetic_symbol(handler_name, "go")
                    _add(file_path, sym, route_hint=f"HANDLE {route_path}")
                for match in _GO_FRAMEWORK_ROUTE_RE.finditer(source):
                    method, route_path, handler_name = match.group(1).upper(), match.group(2), match.group(3)
                    sym = _find_symbol_by_name(result, handler_name) or _synthetic_symbol(handler_name, "go")
                    _add(file_path, sym, route_hint=f"{method} {route_path}")

        # 4. Rust (Actix-web / Rocket attributes & Axum `.route(...)`)
        elif lang == "rust":
            source = _read_source(repository_root, file_path)
            if source:
                for match in _RUST_ATTR_ROUTE_RE.finditer(source):
                    method, route_path, fn_name = match.group(1).upper(), match.group(2), match.group(3)
                    sym = _find_symbol_by_name(result, fn_name) or _synthetic_symbol(fn_name, "rust")
                    _add(file_path, sym, route_hint=f"{method} {route_path}")
                for match in _RUST_AXUM_ROUTE_RE.finditer(source):
                    route_path, method, handler_name = match.group(1), match.group(2).upper(), match.group(3)
                    sym = _find_symbol_by_name(result, handler_name) or _synthetic_symbol(handler_name, "rust")
                    _add(file_path, sym, route_hint=f"{method} {route_path}")

    return candidates


async def api_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate `state.generated_docs["api"]`, if any endpoint candidates exist."""
    candidates = detect_endpoints(state.parse_results, repository_root=state.repository_path)
    if not candidates:
        log.info("API Agent: no HTTP endpoints detected; skipping.")
        return {"generated_docs": state.generated_docs, "statistics": state.statistics, "errors": state.errors}

    digest = build_endpoint_digest(candidates)
    config = resolve_provider_config()

    with Stopwatch("api") as sw:
        async with create_provider(config) as provider:
            try:
                doc, prompt_tokens, completion_tokens = await generate_api_reference(digest, provider)
            except ProviderError as exc:
                state.record_error("api", f"Failed to generate API documentation: {exc}", exception=exc)
                return {"generated_docs": state.generated_docs, "statistics": state.statistics, "errors": state.errors}

    state.statistics.tokens_used += (prompt_tokens or 0) + (completion_tokens or 0)
    state.generated_docs["api"] = doc

    log.info("API Agent: document generated (%d endpoint candidate(s)) in %.2fs", len(candidates), sw.elapsed_seconds)

    return {"generated_docs": state.generated_docs, "statistics": state.statistics, "errors": state.errors}


_DECORATOR_METHOD_RE = re.compile(
    r"""\.(get|post|put|delete|patch|head|options|route)\(\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)
_ROUTE_METHODS_KW_RE = re.compile(r"""methods\s*=\s*\[([^\]]+)\]""", re.IGNORECASE)


def extract_method_and_path(candidate: EndpointCandidate) -> tuple[str | None, str | None]:
    """
    Extract `(HTTP_METHOD, route_path)` from an `EndpointCandidate`'s
    `route_hint` or Python route decorators. Returns `(None, None)` if
    neither yields a recognizable route.
    """
    if candidate.route_hint:
        parts = candidate.route_hint.split(None, 1)
        if len(parts) == 2:
            method, path = parts[0].upper(), parts[1]
            if method in ("HANDLE", "ALL"):
                method = "GET"
            return method, path

    for dec in candidate.symbol.decorators:
        match = _DECORATOR_METHOD_RE.search(dec)
        if not match:
            continue
        verb, path = match.group(1).upper(), match.group(2)
        if verb == "ROUTE":
            kw = _ROUTE_METHODS_KW_RE.search(dec)
            if kw:
                first = re.search(r"""['"]([A-Za-z]+)['"]""", kw.group(1))
                verb = first.group(1).upper() if first else "GET"
            else:
                verb = "GET"
        return verb, path

    return None, None


def build_endpoint_digest(candidates: list[EndpointCandidate]) -> str:
    """Format endpoint candidates with route/decorator, signature, parameters, return type, and docstring."""
    lines = []
    for c in sorted(candidates, key=lambda c: (c.file_path, c.symbol.name, c.route_hint or "")):
        route_parts = list(c.symbol.decorators)
        if c.route_hint:
            route_parts.append(c.route_hint)
        route_str = " ".join(route_parts) if route_parts else "(route)"

        signature = c.symbol.signature or c.symbol.name
        params_str = ", ".join(
            f"{p.name}: {p.type_annotation}" if p.type_annotation else p.name
            for p in c.symbol.parameters
        ) or "(none)"
        return_str = c.symbol.return_type or "(not annotated)"
        docstring = c.symbol.docstring or "(no docstring)"

        lines.append(
            f"- {c.file_path}::{c.symbol.name} ({c.symbol.language})\n"
            f"  Route/Decorator: {route_str}\n"
            f"  Signature: {signature}\n"
            f"  Parameters: {params_str}\n"
            f"  Return type: {return_str}\n"
            f"  Docstring: {docstring}"
        )
    return "\n".join(lines)


async def generate_api_reference(digest: str, provider: BaseProvider) -> tuple[str, int | None, int | None]:
    """Generate the API reference document for `digest`. Raises `ProviderError` after retries are exhausted."""
    prompt = render_prompt("api", endpoint_digest=digest)
    messages = [ChatMessage(role=Role.USER, content=prompt)]

    async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
        with attempt:
            result = await provider.generate(messages, temperature=0.1, max_tokens=1500)

    return result.text.strip(), result.prompt_tokens, result.completion_tokens
