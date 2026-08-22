"""
GitHub README Agent — produces `state.generated_docs["githubReadme"]`, a
GitHub-flavored presentation of the README the README Agent already
generated and fact-checked (draft -> critique -> revise, see
`agents/readme_agent.py`).

Deliberately *not* a second LLM-drafted document. Two agents independently
drafting "the README" from the same grounding material would risk the two
documents quietly drifting apart — different phrasing of the same claim,
or worse, one catching an invented claim the other's critique pass missed
this run. Instead, this stage runs after `readme` in the graph and
transforms its already-verified text: a centered title, a row of
badges, a divider, and an emoji on each of the three fixed section
headings — the prose, the sections, and every factual claim in the body
are untouched and exactly what already passed the README Agent's own
critique loop. Every transform here is presentation only, never content:
nothing added here is a claim that could be right or wrong.

The badges themselves follow the same rule as everywhere else in this
pipeline: real, verified data or nothing.
    - **License** — only shown if `utils/license_detect.py` finds an
      actual `LICENSE`-shaped file in the repository root; the badge
      names a specific license only if the file's text matches a known
      signature, and links to the real file either way. No license file,
      no badge — never a guessed license.
    - **Primary language** — computed the same way `repo_agent.build_digest`
      computes `language_breakdown` (a count over `state.parse_results`,
      not an inference), so it's exactly what the rest of the pipeline
      already considers the repository's dominant language.
    - **Files documented** — `len(state.file_docs.docs)` with no error,
      a plain count already available on state, not a claim about
      quality or coverage.
No CI, build, coverage, or version badges: this pipeline has no stage
that produces any of that data, and a badge implying otherwise (a green
"build passing" pill with nothing behind it) would be exactly the kind
of confident-but-unverifiable claim every other stage here goes out of
its way to avoid.

Runs after `readme` (needs its output) and before `review` in the graph
— see `graph/graph.py` and `graph/stages.py`. Not sent through the LLM
review pass: there's no new LLM-authored content here for a reviewer to
fact-check against grounding, only a deterministic transform of text
that's already been reviewed once as `readme`. It does still pass through
`_apply_deterministic_checks` in `agents/review_agent.py` like every
other document, since that scan runs over all of `state.generated_docs`
regardless of doc name — a banned phrase or a bad file reference in the
underlying README text would still get flagged under this key too.
"""

from __future__ import annotations

import re

from graph.state import RepositoryState
from utils.license_detect import detect_license
from utils.logger import get_logger

log = get_logger(__name__)

# Query params, not a different shields.io endpoint: `for-the-badge` is a
# purely visual style shields.io itself supports on the exact same badge
# URLs — bigger, bolder, all-caps pills, the look most current GitHub
# READMEs use for their badge row. Doesn't change what a badge says, only
# how it's drawn.
_BADGE_STYLE = "for-the-badge"

_LANGUAGE_BADGE_COLORS = {
    "python": "3776AB",
    "typescript": "3178C6",
    "javascript": "F7DF1E",
    "go": "00ADD8",
    "rust": "DEA584",
}
_DEFAULT_BADGE_COLOR = "555555"

# Display casing for the canonical (lowercase) language names in
# `parsers/tree_sitter.py`'s registry — a plain `.title()` gets three of
# these wrong (`javascript` -> "Javascript" instead of "JavaScript",
# `typescript` -> "Typescript" instead of "TypeScript", `tsx`/`jsx` ->
# "Tsx"/"Jsx" instead of "TSX"/"JSX"), so the real names are spelled out
# explicitly rather than guessed from a casing rule. `.title()` is still
# used as the fallback for any language this pipeline adds parser support
# for later and this dict hasn't been updated for yet — a generic
# capitalization is a reasonable default; asserting a *specific* wrong
# casing with confidence would not be.
_LANGUAGE_DISPLAY_NAMES = {
    "python": "Python",
    "javascript": "JavaScript",
    "jsx": "JSX",
    "typescript": "TypeScript",
    "tsx": "TSX",
    "go": "Go",
    "rust": "Rust",
}

# Purely decorative — an emoji prefix on a section heading, nothing more.
# Since `prompts/readme.md` now lets the model choose its own section
# names, count, and order per repository (deliberately — a README's
# structure should reflect what that repository's material actually
# looks like, not a fixed template repeated identically for every
# project), this can no longer match a small fixed set of exact heading
# strings. Instead it matches keywords likely to appear *within*
# whatever heading the model picked — checked in order, first match
# wins per heading — falling back to a neutral default so every H2
# section still gets some visual marker instead of an inconsistent mix
# of decorated and undecorated headings.
_HEADING_KEYWORD_EMOJI: list[tuple[str, str]] = [
    ("overview", "🔍"),
    ("summary", "🔍"),
    ("architecture", "🏗️"),
    ("design", "🏗️"),
    ("component", "🧩"),
    ("module", "🧩"),
    ("subsystem", "🧩"),
    ("api", "🔌"),
    ("interface", "🔌"),
    ("config", "⚙️"),
    ("setting", "⚙️"),
    ("database", "💾"),
    ("storage", "💾"),
    ("persist", "💾"),
    ("memory", "💾"),
    ("security", "🔐"),
    ("auth", "🔐"),
    ("performance", "⚡"),
    ("test", "🧪"),
    ("deploy", "🚀"),
    ("infrastructure", "🚀"),
    ("decision", "🎯"),
    ("pattern", "🎯"),
    ("pipeline", "🔄"),
    ("workflow", "🔄"),
    ("stage", "🔄"),
    ("util", "🛠️"),
    ("helper", "🛠️"),
    ("tool", "🛠️"),
    ("provider", "🔗"),
    ("backend", "🔗"),
    ("integration", "🔗"),
    ("dependency", "🔗"),
    ("dependencies", "🔗"),
    ("scan", "🔎"),
    ("language", "🌐"),
    ("lexer", "🌐"),
    ("syntax", "🌳"),
    ("symbol", "🌳"),
    ("parsing", "🌳"),
    ("parser", "🌳"),
    ("graph", "🕸️"),
    ("state", "🧠"),
    ("orchestrat", "🎼"),
    ("export", "📤"),
    ("output", "📤"),
    ("render", "📤"),
    ("format", "📤"),
    ("knowledge", "📚"),
    ("review", "🔬"),
    ("embed", "🧬"),
    ("vector", "🧬"),
    ("semantic", "🧬"),
    ("retriev", "🧭"),
    ("search", "🧭"),
    ("changelog", "📝"),
    ("version", "📝"),
    ("release", "📝"),
    ("cache", "🗃️"),
    ("log", "📜"),
    ("hash", "🔑"),
    ("error", "⚠️"),
    ("exception", "⚠️"),
    ("cli", "💻"),
    ("command", "💻"),
    ("queue", "⏱️"),
    ("schedul", "⏱️"),
    ("document", "📖"),
    ("data", "📊"),
]
_DEFAULT_HEADING_EMOJI = "📄"


def github_readme_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate `state.generated_docs["githubReadme"]` from the already-generated `readme` doc."""
    base_readme = state.generated_docs.get("readme")
    if not base_readme:
        # Nothing to wrap — same "nothing invented" stance as the README
        # Agent itself: nothing here should generate a document from
        # scratch, since anything not already fact-checked as `readme`
        # would be new, unverified content.
        return {"generated_docs": state.generated_docs}

    badges = _build_badges(state)
    styled = _apply_github_header(base_readme, badges)
    styled = _add_section_emoji(styled)
    state.generated_docs["githubReadme"] = styled

    log.info("GitHub README Agent: wrapped README with %d badge(s)", len(badges))
    return {"generated_docs": state.generated_docs}


def _primary_language(state: RepositoryState) -> tuple[str, int] | None:
    """The most common language across `state.parse_results`, or `None` if nothing was parsed. Same count basis as `repo_agent.build_digest`'s `language_breakdown`, just kept as data instead of pre-rendered into a string."""
    counts: dict[str, int] = {}
    for parse_result in state.parse_results.values():
        counts[parse_result.language] = counts.get(parse_result.language, 0) + 1
    if not counts:
        return None
    language = max(counts, key=lambda lang: counts[lang])
    return language, counts[language]


def _build_badges(state: RepositoryState) -> list[str]:
    """
    Build shields.io badges as real `<img>` HTML tags from real, verified
    pipeline data only — see this module's docstring for what's
    deliberately excluded.

    HTML, not Markdown image syntax (`![alt](url)`): the badge row gets
    wrapped in a raw `<p align="center">` block by `_apply_github_header`
    so it renders centered, and CommonMark (what GitHub, and most other
    Markdown renderers, actually implement) does not parse Markdown
    syntax inside a raw HTML block — content there is passed through
    verbatim. A `![...](...)`  inside that block would show up as the
    literal text `![Primary language](https://...)` instead of an image,
    which is exactly what happened before this was HTML. A real `<img>`
    tag has no such problem, since it's already the HTML the block is
    raw-passing-through in the first place.
    """
    badges: list[str] = []

    license_info = detect_license(state.repository_path)
    if license_info is not None:
        spdx_id, filename = license_info
        label = spdx_id if spdx_id is not None else "see%20LICENSE"
        img = f'<img alt="License" src="https://img.shields.io/badge/license-{label}-blue.svg?style={_BADGE_STYLE}">'
        badges.append(f'<a href="{filename}">{img}</a>')

    language = _primary_language(state)
    if language is not None:
        name, _count = language
        color = _LANGUAGE_BADGE_COLORS.get(name.lower(), _DEFAULT_BADGE_COLOR)
        display_name = _LANGUAGE_DISPLAY_NAMES.get(name.lower(), name.title())
        display = display_name.replace(" ", "%20")
        badges.append(
            f'<img alt="Primary language" src="https://img.shields.io/badge/language-{display}-{color}.svg?style={_BADGE_STYLE}">'
        )

    documented = sum(1 for doc in state.file_docs.docs if doc.error is None)
    if documented:
        badges.append(
            f'<img alt="Files documented" '
            f'src="https://img.shields.io/badge/files%20documented-{documented}-informational.svg?style={_BADGE_STYLE}">'
        )

    return badges


def _apply_github_header(readme_text: str, badges: list[str]) -> str:
    """Center the title and insert a badge row under it, followed by a divider, leaving every other line of `readme_text` untouched. `badges` must already be HTML (see `_build_badges`) since this wraps them in a raw `<p>` block."""
    stripped = readme_text.lstrip("\n")
    badge_row = " ".join(badges)

    if not stripped.startswith("# "):
        # No title line to center (e.g. the minimal fallback README) —
        # just place the badge row on its own, above the existing text.
        if not badge_row:
            return readme_text
        return f'<p align="center">\n{badge_row}\n</p>\n\n---\n\n{stripped}'

    title_line, _, rest = stripped.partition("\n")
    title = title_line[2:].strip()
    rest = rest.lstrip("\n")

    header_lines = [f'<h1 align="center">{title}</h1>']
    if badge_row:
        header_lines.append(f'<p align="center">\n{badge_row}\n</p>')

    return "\n".join(header_lines) + "\n\n---\n\n" + rest


def _add_section_emoji(readme_text: str) -> str:
    """Prefix every `##` (level-2) section heading with an emoji chosen by keyword match against the heading's own text, or a neutral default if nothing matches. Purely cosmetic — see `_HEADING_KEYWORD_EMOJI`'s comment for why keyword matching, not an exact heading list, is needed now that section names are model-chosen per repository. Only matches `##` (exactly two hashes), so a `###` subsystem heading (if the model used one) or the `**Bold**` subsystem-name lines the prompt's optional format uses are left untouched."""

    def _emoji_for(heading_text: str) -> str:
        lowered = heading_text.lower()
        for keyword, emoji in _HEADING_KEYWORD_EMOJI:
            if keyword in lowered:
                return emoji
        return _DEFAULT_HEADING_EMOJI

    def _replace(match: re.Match[str]) -> str:
        heading_text = match.group(1)
        return f"## {_emoji_for(heading_text)} {heading_text}"

    return re.sub(r"^## (?!#)(.+)$", _replace, readme_text, flags=re.MULTILINE)