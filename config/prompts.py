"""
Prompt template loading.

Prompt bodies themselves live as Markdown files under `prompts/` and are
added when each documentation agent is implemented (chunk, file,
repository, architecture, api, readme, review). This module only provides the
generic, reusable loading/rendering mechanism so agents don't each
reinvent file I/O and templating.

Templates use simple `{placeholder}` substitution via `str.format_map`
with a permissive mapping (missing keys render as an empty string rather
than raising), which is intentionally low-tech: prompts should stay easy
to hand-edit by non-engineers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


@lru_cache(maxsize=32)
def _read_template(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template '{name}' not found at {path}. "
            "It will be added when the corresponding agent is implemented."
        )
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, /, **variables: object) -> str:
    """
    Load `prompts/{name}.md` and substitute `{variable}` placeholders.

    Example:
        render_prompt("chunk", code=snippet, language="python")
    """
    template = _read_template(name)
    return template.format_map(_SafeDict(**variables))


def clear_prompt_cache() -> None:
    """Useful in tests / hot-reload scenarios after editing a prompt file."""
    _read_template.cache_clear()
