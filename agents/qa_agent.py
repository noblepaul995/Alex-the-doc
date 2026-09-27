"""
The agentic Q&A loop: given a question, repeatedly lets the model choose
between investigating further (via `agents/qa_tools.py`'s tools) or
answering, instead of the older single-shot "retrieve once, then
answer" approach. See `prompts/qa_agent.md` for the exact ReAct-style
protocol (`ACTION: tool(arg)` / `FINAL ANSWER: ...`) this parses.

Text-based, not native provider tool-calling — see this module's
motivation in `app.py`'s `github` command docstring / the conversation
that led here: `llm/base.py`'s `BaseProvider` has no function-calling
support at all, and adding it properly across every provider this
pipeline already supports (OpenAI, Anthropic, Gemini, Ollama,
OpenRouter, any OpenAI-compatible endpoint) would each need separate,
provider-specific work. A plain-text protocol works identically with
every one of them today, at the cost of needing to parse the model's
own text instead of a structured tool-call field — the same tradeoff
this codebase already made for the critique/revise loops.
"""

from __future__ import annotations

import re

from config.prompts import render_prompt
from embeddings.lancedb import LanceVectorStore
from graph.state import RepositoryState
from llm.base import BaseProvider, ChatMessage, Role
from utils.logger import get_logger

from agents.qa_tools import get_file, get_symbol, list_dependencies, search_repository, web_search_tool
from utils.web_search import is_web_search_configured

log = get_logger(__name__)

MAX_INVESTIGATION_STEPS = 6
"""Cap on tool calls before forcing a final answer — bounds worst-case cost/latency on a question the model can't resolve, same reasoning as MAX_CRITIQUE_ROUNDS elsewhere in this pipeline."""

_BASE_TOOLS = [
    "search_repository(query) — semantic search over the repository's indexed summaries",
    "get_file(path) — read a file's real content (or its summary if the raw file isn't available)",
    "get_symbol(name) — look up a function/class/symbol's definition",
    "list_dependencies(path) — a file's real, resolved imports and importers",
]
"""Always-available investigation tools, rendered into `prompts/qa_agent.md`'s `{tools_list}` placeholder."""


def _build_tools_list() -> str:
    """
    Tools description for the prompt, including `web_search` only when
    it's actually configured (a Tavily key, an explicit open-websearch
    endpoint, or autostart-with-Node available) — checked via the
    cheap, never-prompting `is_web_search_configured`, since this runs
    once per investigation step just to build the prompt, long before
    the model has decided it wants to use it. The real "is it up right
    now" check (and, for open-websearch, the install prompt) happens
    lazily inside `web_search_tool` itself, only if/when the model
    actually calls it.
    """
    tools = list(_BASE_TOOLS)
    if is_web_search_configured():
        tools.append("web_search(query) — search the web for current or external information the repository can't contain")
    return "\n".join(f"- {tool}" for tool in tools)

_FINAL_ANSWER_PREFIX = "final answer:"
_ACTION_PATTERN = re.compile(r"ACTION:\s*(\w+)\((.*)\)", re.DOTALL)


def _parse_response(text: str) -> tuple[str, str]:
    """Returns `("final", answer)` or `("action", "tool_name(raw_arg)")`. Anything that doesn't parse as either is treated as a final answer verbatim — a malformed response shouldn't crash the loop, and the raw text is still more useful to the user than nothing."""
    stripped = text.strip()
    if stripped.lower().startswith(_FINAL_ANSWER_PREFIX):
        return "final", stripped[len(_FINAL_ANSWER_PREFIX) :].strip()

    match = _ACTION_PATTERN.search(stripped)
    if match:
        return "action", f"{match.group(1)}({match.group(2)})"

    return "final", stripped


def _split_action(action: str) -> tuple[str, str]:
    match = _ACTION_PATTERN.search(action) or re.match(r"(\w+)\((.*)\)", action, re.DOTALL)
    if not match:
        return action.strip(), ""
    return match.group(1), _clean_arg(match.group(2))


def _clean_arg(raw: str) -> str:
    arg = raw.strip()
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
        arg = arg[1:-1]
    return arg


async def _dispatch_tool(
    tool_name: str, arg: str, state: RepositoryState, provider: BaseProvider, store: LanceVectorStore
) -> str:
    if tool_name == "search_repository":
        return await search_repository(arg, state, provider, store)
    if tool_name == "get_file":
        return get_file(arg, state)
    if tool_name == "get_symbol":
        return get_symbol(arg, state)
    if tool_name == "list_dependencies":
        return list_dependencies(arg, state)
    if tool_name == "web_search":
        return await web_search_tool(arg)
    return f"Unknown tool '{tool_name}'. Available tools: search_repository, get_file, get_symbol, list_dependencies, web_search."


async def investigate(
    question: str,
    state: RepositoryState,
    repo_summary: str | None,
    provider: BaseProvider,
    store: LanceVectorStore,
) -> str:
    """
    Run the investigation loop for `question`, returning the final
    answer text. Logs each step at INFO level (tool + argument only, not
    the full result) so a long investigation is visible in the same
    console output every other pipeline stage already uses, rather than
    looking like a silent hang.
    """
    transcript_lines = [f"Question: {question}"]
    tools_list = _build_tools_list()

    for step in range(1, MAX_INVESTIGATION_STEPS + 1):
        prompt = render_prompt(
            "qa_agent",
            repo_summary=repo_summary or "(not available)",
            tools_list=tools_list,
            transcript="\n".join(transcript_lines),
        )
        result = await provider.generate([ChatMessage(role=Role.USER, content=prompt)], temperature=0.1, max_tokens=800)
        kind, payload = _parse_response(result.text)

        if kind == "final":
            return payload

        tool_name, arg = _split_action(payload)
        log.info('Q&A step %d/%d: %s("%s")', step, MAX_INVESTIGATION_STEPS, tool_name, arg)
        tool_result = await _dispatch_tool(tool_name, arg, state, provider, store)
        transcript_lines.append(f'\nStep {step}:\nACTION: {tool_name}("{arg}")\nResult: {tool_result}')

    log.info("Q&A: hit the %d-step investigation cap, forcing a final answer", MAX_INVESTIGATION_STEPS)
    forced_prompt = render_prompt(
        "qa_agent",
        repo_summary=repo_summary or "(not available)",
        tools_list=tools_list,
        transcript="\n".join(transcript_lines)
        + "\n\nYou're out of investigation steps. Give your FINAL ANSWER now, based on what you've found so far.",
    )
    result = await provider.generate([ChatMessage(role=Role.USER, content=forced_prompt)], temperature=0.1, max_tokens=800)
    kind, payload = _parse_response(result.text)
    return payload if kind == "final" else result.text.strip()