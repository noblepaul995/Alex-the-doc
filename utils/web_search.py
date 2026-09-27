"""
Standalone web search. Deliberately independent of any one agent or use
case — the Q&A investigation loop is the first caller
(`agents/qa_tools.py`), but nothing here assumes that; a future
documentation agent checking whether an API a repository uses is still
current, for instance, can call this directly the same way.

Two backends, tried in this order:

1. open-websearch (https://github.com/Aas-ee/open-webSearch) — a small,
   self-hosted Node daemon that scrapes free search engines (Bing,
   DuckDuckGo, ...), no API key required. This module can run the
   daemon itself: it checks whether one is already reachable at
   `Settings.open_websearch_endpoint` (or `http://localhost:{open_websearch_port}`),
   and if not, and `open_websearch_autostart` is on, offers to start
   one with `npx -y open-websearch@latest` — asking for confirmation
   first when attached to an interactive terminal, and simply declining
   (falling through to Tavily, or reporting unavailable) otherwise, so
   nothing ever launches a background process the person didn't agree
   to.
2. Tavily (https://tavily.com) — hosted, requires `TAVILY_API_KEY`.
   Used only if open-websearch isn't reachable and can't be started.

Nothing in this pipeline calls this automatically — same principle
`prompts/qa_agent.md` states explicitly for the Q&A loop: web search is
a tool a caller (usually an LLM deciding it needs current or external
information) chooses to use, not something that runs by default. The
open-websearch autostart above is the one exception to "nothing runs
automatically," and only because the person is asked before it happens.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import httpx

from config.settings import Settings, get_settings
from utils.logger import get_logger

log = get_logger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_TIMEOUT_SECONDS = 15

_PING_TIMEOUT_SECONDS = 3
"""Short on purpose: this fires on every availability check, so a daemon that isn't there shouldn't stall the caller."""

_STARTUP_TIMEOUT_SECONDS = 25
"""How long to wait for a freshly-spawned open-websearch daemon to answer its health check before giving up."""
_STARTUP_POLL_INTERVAL_SECONDS = 0.5

# Process-wide state (this module is a singleton — one daemon per `alex` run).
_owned_process: subprocess.Popen | None = None
"""The daemon subprocess this module itself started, if any — checked so a second caller in the same run reuses it instead of spawning another."""
_verified_endpoint: str | None = None
"""Endpoint we've already confirmed reachable this run, so a healthy daemon isn't re-pinged on every single call."""
_declined_autostart = False
"""Set once the person says no (or isn't there to ask) — remembered so we don't prompt again every time a tool call needs search."""


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    published_date: str | None = None


class WebSearchError(Exception):
    """Raised on any failure to complete a search — no backend available, a network error, an invalid key, a rate limit. Callers dispatching this as an agent tool should catch this and treat it the same as "search unavailable right now" (report that plainly to the model) rather than letting it crash the calling loop."""


def _candidate_open_websearch_endpoint(settings: Settings) -> str:
    return (settings.open_websearch_endpoint or f"http://localhost:{settings.open_websearch_port}").rstrip("/")


async def _ping_open_websearch(endpoint: str) -> bool:
    """Hit the daemon's own documented health check (`GET /api/health`) rather than just opening a TCP connection, so we don't mistake some unrelated service on the same port for it."""
    try:
        async with httpx.AsyncClient(timeout=_PING_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{endpoint}/api/health")
            return response.status_code < 500
    except httpx.HTTPError:
        return False


def _npx_available() -> bool:
    return shutil.which("npx") is not None


def _confirm(prompt: str) -> bool:
    """Ask the person a yes/no question on the real terminal. Returns False (never asks, never blocks) when there isn't an interactive terminal to ask on — e.g. running in a script, a CI job, or with stdin piped from a file — since silently spawning a process in that situation would be a surprise, not a convenience."""
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # tidy up the dangling prompt line
        return False
    return answer in ("y", "yes")


def _spawn_open_websearch(settings: Settings, port: int) -> subprocess.Popen:
    log.info("Starting open-websearch (`npx -y open-websearch@latest`) on port %d ...", port)
    env = {**os.environ, "PORT": str(port), "MODE": "http"}
    return subprocess.Popen(  # noqa: S603 — argv is a fixed literal list, nothing from user input reaches it
        ["npx", "-y", "open-websearch@latest"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


async def ensure_open_websearch(settings: Settings | None = None, *, interactive: bool = True) -> str | None:
    """
    Return a reachable open-websearch base URL, starting the daemon if
    needed and allowed. Order of attempts:

    1. Already verified reachable this run → return it immediately, no network call.
    2. `open_websearch_endpoint` / default localhost port → ping it; if it answers, someone (or
       something) is already running it — use that.
    3. We already spawned it ourselves earlier this run and it's still alive → give it a little
       more time to finish starting rather than launching a second copy.
    4. Otherwise, if autostart is enabled and Node's `npx` is on PATH: ask for confirmation (when
       `interactive` and attached to a real terminal) before running `npx -y open-websearch@latest`,
       then poll its health check until it comes up or `_STARTUP_TIMEOUT_SECONDS` elapses.

    Returns `None` (never raises) at any point search isn't available this way — callers fall back
    to Tavily or report search as unavailable, exactly as if no backend were configured at all.
    """
    global _owned_process, _verified_endpoint, _declined_autostart

    settings = settings or get_settings()

    if _verified_endpoint and await _ping_open_websearch(_verified_endpoint):
        return _verified_endpoint
    _verified_endpoint = None

    endpoint = _candidate_open_websearch_endpoint(settings)
    if await _ping_open_websearch(endpoint):
        _verified_endpoint = endpoint
        return endpoint

    if _owned_process is not None and _owned_process.poll() is None:
        deadline_ticks = int(_STARTUP_TIMEOUT_SECONDS / _STARTUP_POLL_INTERVAL_SECONDS)
        for _ in range(deadline_ticks):
            await asyncio.sleep(_STARTUP_POLL_INTERVAL_SECONDS)
            if await _ping_open_websearch(endpoint):
                _verified_endpoint = endpoint
                return endpoint
        log.warning("open-websearch didn't come up within %ds — leaving it running in case it's just slow, but treating it as unavailable for now.", _STARTUP_TIMEOUT_SECONDS)
        return None

    if not settings.open_websearch_autostart or _declined_autostart:
        return None

    if not _npx_available():
        log.info(
            "open-websearch isn't running and Node/npx isn't on PATH to start it — "
            "install Node.js, point OPEN_WEBSEARCH_ENDPOINT at a daemon already running elsewhere, "
            "or configure TAVILY_API_KEY instead."
        )
        _declined_autostart = True
        return None

    if interactive:
        if not _confirm(
            "Web search isn't running. open-websearch (free, no API key, "
            f"via `npx open-websearch@latest`) can be installed and started now on port {settings.open_websearch_port} — proceed?"
        ):
            _declined_autostart = True
            return None
    else:
        # No one to ask right now (e.g. a non-interactive tool call deep in the agent loop);
        # don't spawn a process behind the person's back without at least one explicit yes.
        _declined_autostart = True
        return None

    _owned_process = _spawn_open_websearch(settings, settings.open_websearch_port)

    deadline_ticks = int(_STARTUP_TIMEOUT_SECONDS / _STARTUP_POLL_INTERVAL_SECONDS)
    for _ in range(deadline_ticks):
        await asyncio.sleep(_STARTUP_POLL_INTERVAL_SECONDS)
        if _owned_process.poll() is not None:
            log.warning("open-websearch exited immediately (exit code %s) — falling back.", _owned_process.returncode)
            _owned_process = None
            return None
        if await _ping_open_websearch(endpoint):
            _verified_endpoint = endpoint
            log.info("open-websearch is up at %s", endpoint)
            return endpoint

    log.warning("open-websearch didn't respond within %ds of starting it — falling back.", _STARTUP_TIMEOUT_SECONDS)
    return None


async def is_web_search_available(settings: Settings | None = None, *, interactive: bool = True) -> bool:
    """
    Whether a call to `web_search()` right now would actually have a
    backend to use — checked (and, for open-websearch, potentially
    started) rather than assumed, so a caller building a tool-listing
    prompt doesn't offer `web_search` only for it to error immediately.

    `interactive=False` skips ever prompting to install/start
    open-websearch (used for the first check that decides whether to
    even mention the tool exists, so the person isn't asked about
    installing something before they've asked a question that needs
    it) — pass `interactive=True` (the default) at the point the tool
    is actually about to run.
    """
    settings = settings or get_settings()
    if await ensure_open_websearch(settings, interactive=interactive) is not None:
        return True
    return bool(settings.tavily_api_key)


def is_web_search_configured(settings: Settings | None = None) -> bool:
    """
    Cheap, synchronous, no-network, never-prompts check: is there any
    path to search working at all (a key set, an endpoint explicitly
    configured, or autostart possible)? For deciding whether to mention
    `web_search` as a tool in a prompt built before any actual search is
    attempted — the real availability (and any install prompt) is
    resolved lazily, only once the model actually tries to call it, via
    `is_web_search_available`.
    """
    settings = settings or get_settings()
    if settings.tavily_api_key:
        return True
    if settings.open_websearch_endpoint:
        return True
    return settings.open_websearch_autostart and _npx_available()


async def _search_open_websearch(endpoint: str, query: str, max_results: int) -> list[WebSearchResult]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{endpoint}/api/search", params={"q": query, "limit": max_results})
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise WebSearchError(f"open-websearch request failed: {exc}") from exc

    # Documented shape is `{"success": bool, "data": {"results": [...]}, ...}` (see REST-API.md),
    # with `data.results` possibly missing on error; some deployments have also been seen returning
    # a bare `[...]` for `data` instead of the `{"results": [...]}` envelope, so accept either rather
    # than assuming the richer shape and crashing on the simpler one.
    if isinstance(payload, dict) and payload.get("success") is False:
        raise WebSearchError(f"open-websearch reported an error: {payload.get('error', 'unknown error')}")

    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    raw_results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

    results = []
    for item in raw_results[:max_results]:
        results.append(
            WebSearchResult(
                title=item.get("title") or "(untitled)",
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                published_date=item.get("published_date") or None,
            )
        )
    return results


async def _search_tavily(settings: Settings, query: str, max_results: int) -> list[WebSearchResult]:
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(_TAVILY_ENDPOINT, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise WebSearchError(f"Tavily request failed: {exc}") from exc

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            WebSearchResult(
                title=item.get("title") or "(untitled)",
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                published_date=item.get("published_date") or None,
            )
        )
    return results


async def web_search(query: str, *, max_results: int = 5, settings: Settings | None = None, interactive: bool = True) -> list[WebSearchResult]:
    """
    Run a real web search and return up to `max_results` results. Tries
    open-websearch first (starting it, with confirmation, if it isn't
    running and `interactive` allows asking), then Tavily if that's
    configured instead. Raises `WebSearchError` — never returns a
    silently-empty list to mean "failed," only to mean "genuinely no
    results" — so a caller can tell the two apart.
    """
    settings = settings or get_settings()

    endpoint = await ensure_open_websearch(settings, interactive=interactive)
    if endpoint is not None:
        return await _search_open_websearch(endpoint, query, max_results)

    if settings.tavily_api_key:
        return await _search_tavily(settings, query, max_results)

    raise WebSearchError(
        "Web search isn't configured — no open-websearch daemon is running or startable "
        "(see OPEN_WEBSEARCH_ENDPOINT / OPEN_WEBSEARCH_AUTOSTART in .env.example), and no "
        "TAVILY_API_KEY is set."
    )


def render_results(results: list[WebSearchResult]) -> str:
    """Format results as plain text ready to drop into an LLM prompt or tool-result transcript — no caller should need to know `WebSearchResult`'s field names to display them."""
    if not results:
        return "No web results found."
    lines = []
    for result in results:
        date_part = f" ({result.published_date})" if result.published_date else ""
        lines.append(f"- {result.title}{date_part} — {result.url}\n  {result.snippet}")
    return "\n".join(lines)