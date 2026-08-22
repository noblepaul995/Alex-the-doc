"""
API Agent — documents HTTP endpoints, detected heuristically from
decorators.

Scoped honestly to what the pipeline can actually detect: a function is
treated as an endpoint candidate if one of its decorators matches a
common route-decorator pattern (`@app.route(...)`, `@router.get(...)`,
`@api_view(...)`, and similar Flask/FastAPI/Django-REST-style forms).
This is a real, documented limitation, not an oversight — decorators
are currently only extracted for Python (`parsers/python_parser.py`);
`parsers/ts_parser.py` doesn't populate `Symbol.decorators` at all, so
Express/NestJS-style JS/TS route handlers are invisible to this agent
regardless of how they're written, and Go has no decorator concept at
all. A repository with no matching decorators (including this one —
`alex` is a CLI tool, not an API) produces no API document at all,
rather than asking the LLM to invent an API surface that doesn't exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.prompts import render_prompt
from config.providers import resolve_provider_config
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, ProviderError, Role
from llm.factory import create_provider
from parsers.metadata import ParseResult, Symbol
from utils.helpers import with_retry
from utils.logger import get_logger
from utils.timers import Stopwatch

log = get_logger(__name__)

_ROUTE_MARKERS = (".route(", ".get(", ".post(", ".put(", ".delete(", ".patch(", "api_view", "api_route")


@dataclass
class EndpointCandidate:
    file_path: str
    symbol: Symbol


def detect_endpoints(parse_results: dict[str, ParseResult]) -> list[EndpointCandidate]:
    """Find every symbol whose decorators match a common route-decorator pattern."""
    candidates: list[EndpointCandidate] = []
    for file_path, result in parse_results.items():
        for symbol in result.symbols:
            if any(_looks_like_route(d) for d in symbol.decorators):
                candidates.append(EndpointCandidate(file_path=file_path, symbol=symbol))
    return candidates


def _looks_like_route(decorator: str) -> bool:
    lowered = decorator.lower()
    return any(marker in lowered for marker in _ROUTE_MARKERS)


async def api_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate `state.generated_docs["api"]`, if any endpoint candidates exist."""
    candidates = detect_endpoints(state.parse_results)
    if not candidates:
        log.info("API Agent: no route-decorated endpoints detected; skipping.")
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


def build_endpoint_digest(candidates: list[EndpointCandidate]) -> str:
    lines = []
    for c in sorted(candidates, key=lambda c: (c.file_path, c.symbol.name)):
        decorators = " ".join(c.symbol.decorators)
        signature = c.symbol.signature or c.symbol.name
        docstring = c.symbol.docstring or "(no docstring)"
        lines.append(f"- {c.file_path}::{c.symbol.name}\n  {decorators}\n  {signature}\n  {docstring}")
    return "\n".join(lines)


async def generate_api_reference(digest: str, provider: BaseProvider) -> tuple[str, int | None, int | None]:
    """Generate the API reference document for `digest`. Raises `ProviderError` after retries are exhausted."""
    prompt = render_prompt("api", endpoint_digest=digest)
    messages = [ChatMessage(role=Role.USER, content=prompt)]

    async for attempt in with_retry(max_attempts=3, exceptions=(ProviderError,)):
        with attempt:
            result = await provider.generate(messages, temperature=0.1, max_tokens=800)

    return result.text.strip(), result.prompt_tokens, result.completion_tokens
