"""
Tests for `utils/web_search.py` — focused on the parts that are easy to
get subtly wrong: the open-websearch autostart/install-prompt gating
(never spawns anything without a yes from an interactive terminal),
reuse of an already-running or already-spawned daemon, and parsing both
the documented response envelope and its error shape. Real network
calls and real subprocesses are mocked throughout.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from utils import web_search as ws


def _settings(**overrides) -> SimpleNamespace:
    defaults = dict(
        open_websearch_endpoint=None,
        open_websearch_port=3000,
        open_websearch_autostart=True,
        tavily_api_key=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Every test gets a clean slate — these are process-wide singletons by design."""
    ws._owned_process = None
    ws._verified_endpoint = None
    ws._declined_autostart = False
    yield
    ws._owned_process = None
    ws._verified_endpoint = None
    ws._declined_autostart = False


def _mock_response(status_code: int = 200, json_data=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in whose .get/.post are swappable per test."""

    def __init__(self, *, get=None, post=None, **_):
        self._get = get or AsyncMock(side_effect=httpx.ConnectError("refused"))
        self._post = post or AsyncMock(side_effect=httpx.ConnectError("refused"))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *a, **kw):
        return await self._get(*a, **kw)

    async def post(self, *a, **kw):
        return await self._post(*a, **kw)


# --- ensure_open_websearch: reuse / reachability -----------------------------------------------


@pytest.mark.asyncio
async def test_already_running_daemon_is_used_without_spawning():
    settings = _settings()
    healthy_get = AsyncMock(return_value=_mock_response(200))
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(get=healthy_get)), patch.object(
        ws, "_spawn_open_websearch"
    ) as spawn:
        endpoint = await ws.ensure_open_websearch(settings, interactive=True)

    assert endpoint == "http://localhost:3000"
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_verified_endpoint_is_cached_and_not_reponged():
    settings = _settings()
    ws._verified_endpoint = "http://localhost:3000"
    ping_calls = AsyncMock(return_value=_mock_response(200))
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(get=ping_calls)):
        endpoint = await ws.ensure_open_websearch(settings, interactive=True)
    assert endpoint == "http://localhost:3000"
    assert ping_calls.await_count == 1  # one re-verify ping, not a full re-discovery


# --- ensure_open_websearch: autostart / install-prompt gating ----------------------------------


@pytest.mark.asyncio
async def test_declines_to_start_when_npx_missing():
    settings = _settings()
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient()), patch.object(
        ws, "_npx_available", return_value=False
    ), patch.object(ws, "_spawn_open_websearch") as spawn:
        endpoint = await ws.ensure_open_websearch(settings, interactive=True)

    assert endpoint is None
    spawn.assert_not_called()
    assert ws._declined_autostart is True


@pytest.mark.asyncio
async def test_asks_before_starting_and_respects_no():
    settings = _settings()
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient()), patch.object(
        ws, "_npx_available", return_value=True
    ), patch.object(ws, "_confirm", return_value=False) as confirm, patch.object(
        ws, "_spawn_open_websearch"
    ) as spawn:
        endpoint = await ws.ensure_open_websearch(settings, interactive=True)

    assert endpoint is None
    confirm.assert_called_once()
    spawn.assert_not_called()
    assert ws._declined_autostart is True


@pytest.mark.asyncio
async def test_never_prompts_or_spawns_when_not_interactive():
    settings = _settings()
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient()), patch.object(
        ws, "_npx_available", return_value=True
    ), patch.object(ws, "_confirm") as confirm, patch.object(ws, "_spawn_open_websearch") as spawn:
        endpoint = await ws.ensure_open_websearch(settings, interactive=False)

    assert endpoint is None
    confirm.assert_not_called()
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_autostart_disabled_never_prompts():
    settings = _settings(open_websearch_autostart=False)
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient()), patch.object(
        ws, "_npx_available", return_value=True
    ), patch.object(ws, "_confirm") as confirm:
        endpoint = await ws.ensure_open_websearch(settings, interactive=True)

    assert endpoint is None
    confirm.assert_not_called()


@pytest.mark.asyncio
async def test_confirmed_yes_spawns_and_polls_until_healthy():
    settings = _settings()
    fake_process = MagicMock()
    fake_process.poll.return_value = None  # still running throughout

    # First two health pings fail (still booting), third succeeds.
    call_count = {"n": 0}

    async def flaky_get(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise httpx.ConnectError("not up yet")
        return _mock_response(200)

    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(get=AsyncMock(side_effect=flaky_get))), patch.object(
        ws, "_npx_available", return_value=True
    ), patch.object(ws, "_confirm", return_value=True), patch.object(
        ws, "_spawn_open_websearch", return_value=fake_process
    ) as spawn, patch.object(ws, "_STARTUP_POLL_INTERVAL_SECONDS", 0.001):
        endpoint = await ws.ensure_open_websearch(settings, interactive=True)

    assert endpoint == "http://localhost:3000"
    spawn.assert_called_once()
    assert ws._owned_process is fake_process


@pytest.mark.asyncio
async def test_second_call_reuses_process_already_spawned_this_run():
    settings = _settings()
    fake_process = MagicMock()
    fake_process.poll.return_value = None
    ws._owned_process = fake_process  # as if a prior call already started it

    healthy_get = AsyncMock(return_value=_mock_response(200))
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(get=healthy_get)), patch.object(
        ws, "_confirm"
    ) as confirm, patch.object(ws, "_spawn_open_websearch") as spawn, patch.object(
        ws, "_STARTUP_POLL_INTERVAL_SECONDS", 0.001
    ):
        endpoint = await ws.ensure_open_websearch(settings, interactive=True)

    assert endpoint == "http://localhost:3000"
    confirm.assert_not_called()  # already running (or starting) — nothing new to ask about
    spawn.assert_not_called()


# --- is_web_search_configured: no network, no prompt --------------------------------------------


def test_is_configured_true_with_tavily_key_only():
    assert ws.is_web_search_configured(_settings(tavily_api_key="x")) is True


def test_is_configured_true_with_explicit_endpoint():
    assert ws.is_web_search_configured(_settings(open_websearch_endpoint="http://x:1")) is True


def test_is_configured_true_with_autostart_and_npx():
    with patch.object(ws, "_npx_available", return_value=True):
        assert ws.is_web_search_configured(_settings()) is True


def test_is_configured_false_with_nothing_available():
    with patch.object(ws, "_npx_available", return_value=False):
        assert ws.is_web_search_configured(_settings(open_websearch_autostart=False)) is False


# --- search parsing: real documented envelope + error path + bare-array fallback ---------------


@pytest.mark.asyncio
async def test_search_parses_documented_envelope():
    settings = _settings()
    payload = {
        "success": True,
        "data": {
            "query": "typescript",
            "engines": ["bing"],
            "totalResults": 1,
            "results": [{"title": "TS Docs", "url": "https://ts.example", "description": "A language.", "source": "bing", "engine": "bing"}],
        },
        "timestamp": "2024-01-01T00:00:00.000Z",
    }
    get_mock = AsyncMock(return_value=_mock_response(200, payload))
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(get=get_mock)):
        results = await ws._search_open_websearch("http://localhost:3000", "typescript", 5)

    assert len(results) == 1
    assert results[0].title == "TS Docs"
    assert results[0].url == "https://ts.example"
    assert results[0].snippet == "A language."


@pytest.mark.asyncio
async def test_search_handles_bare_array_data():
    payload = {"success": True, "data": [{"title": "Bare", "url": "https://x", "description": "d"}]}
    get_mock = AsyncMock(return_value=_mock_response(200, payload))
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(get=get_mock)):
        results = await ws._search_open_websearch("http://localhost:3000", "q", 5)
    assert len(results) == 1
    assert results[0].title == "Bare"


@pytest.mark.asyncio
async def test_search_surfaces_daemon_error_message():
    payload = {"success": False, "error": "engine rate-limited", "timestamp": "..."}
    get_mock = AsyncMock(return_value=_mock_response(200, payload))
    with patch.object(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(get=get_mock)):
        with pytest.raises(ws.WebSearchError, match="engine rate-limited"):
            await ws._search_open_websearch("http://localhost:3000", "q", 5)


# --- web_search: end-to-end backend selection ---------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_prefers_open_websearch_over_tavily():
    settings = _settings(tavily_api_key="unused-should-not-be-called")
    with patch.object(ws, "ensure_open_websearch", AsyncMock(return_value="http://localhost:3000")), patch.object(
        ws, "_search_open_websearch", AsyncMock(return_value=[ws.WebSearchResult("t", "u", "s")])
    ) as ows, patch.object(ws, "_search_tavily", AsyncMock()) as tavily:
        results = await ws.web_search("q", settings=settings)

    ows.assert_called_once()
    tavily.assert_not_called()
    assert results[0].title == "t"


@pytest.mark.asyncio
async def test_web_search_falls_back_to_tavily_when_open_websearch_unavailable():
    settings = _settings(tavily_api_key="key")
    with patch.object(ws, "ensure_open_websearch", AsyncMock(return_value=None)), patch.object(
        ws, "_search_tavily", AsyncMock(return_value=[ws.WebSearchResult("t", "u", "s")])
    ) as tavily:
        results = await ws.web_search("q", settings=settings)

    tavily.assert_called_once()
    assert results[0].title == "t"


@pytest.mark.asyncio
async def test_web_search_raises_when_nothing_available():
    settings = _settings()
    with patch.object(ws, "ensure_open_websearch", AsyncMock(return_value=None)):
        with pytest.raises(ws.WebSearchError):
            await ws.web_search("q", settings=settings)