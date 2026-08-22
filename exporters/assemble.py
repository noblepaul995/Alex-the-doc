"""
Format-independent document assembly for the Export stage.

Every exporter (`markdown.py`, `html.py`, `json.py`, `pdf.py`) writes
the *same* logical document set in a different format — this module
decides what that set is, once, so the decision isn't duplicated (and
can't drift) across four format-specific writers.

Five documents are assembled, each grounded directly in pipeline state
with nothing invented here:
    - README: `state.generated_docs["readme"]` if the README Agent
      produced one, otherwise a minimal fallback built from just
      `state.repo_summary` — never blank, but never fabricated either.
      Gets a deterministically-built Table of Contents inserted right
      after its title, linking to its own real section headers plus
      whichever sibling documents below actually exist in this export
      set (skipped entirely if there's nothing to link to — the minimal
      fallback README has no sections, so no ToC gets added to it).
    - ARCHITECTURE: `state.generated_docs["architecture"]`, if present.
    - API: `state.generated_docs["api"]`, if present (recall: the API
      Agent only produces this when it actually found route-decorated
      endpoints, so its absence here is expected for most repositories).
    - STRUCTURE: `state.generated_docs["structure"]`, if present — a
      deterministic folder/file table produced by the Structure Agent
      with no LLM call involved, so it's always exact.
    - CHANGELOG: `state.generated_docs["changelog"]`, if present — a
      deterministic recent-commits section produced by the Changelog
      Agent straight from `git log`, no LLM call involved. Absent for
      repositories that aren't under git (or have no commits yet).
    - DEPENDENCIES: `state.generated_docs["dependencies"]`, if present —
      a deterministic dependency/install-command listing produced by the
      Dependencies Agent straight from the repository's own package
      manifest(s) (`pyproject.toml`, `package.json`, `Cargo.toml`,
      `go.mod`, `requirements.txt`), no LLM call involved. Absent for
      repositories with no manifest this pipeline recognizes.
    - githubReadme: `state.generated_docs["githubReadme"]`, if present —
      the same README content, wrapped with a centered title and real,
      verified badges (license, primary language, files documented) by
      the GitHub README Agent. No independent drafting, so nothing new
      here for the Review Agent to fact-check that `readme` didn't
      already cover.
    - REVIEW: a rendering of `state.review`'s findings, if any review
      ran — a genuinely useful "documentation health" artifact, built
      entirely from data the Review Agent already produced.
"""

from __future__ import annotations

import re

from graph.review import ReviewCollection
from graph.state import RepositoryState

# Labels for sibling documents in the exported set, keyed by the doc_name
# `build_export_documents` uses. Order here is the order they'll appear in
# the "other documents" part of the Table of Contents.
_SIBLING_DOC_LABELS: dict[str, str] = {
    "ARCHITECTURE": "Architecture",
    "API": "API Reference",
    "STRUCTURE": "Folder Structure",
    "CHANGELOG": "Recent Changes",
    "DEPENDENCIES": "Dependencies & Installation",
    "githubReadme": "GitHub README",
    "REVIEW": "Documentation Review",
}


def build_export_documents(state: RepositoryState) -> dict[str, str]:
    """Assemble the logical document set: `{doc_name: markdown_text}`, doc_name without extension."""
    documents: dict[str, str] = {}

    if "readme" in state.generated_docs:
        documents["README"] = state.generated_docs["readme"]
    elif state.repo_summary:
        documents["README"] = f"# Repository Overview\n\n{state.repo_summary}\n"

    if "architecture" in state.generated_docs:
        documents["ARCHITECTURE"] = state.generated_docs["architecture"]

    if "api" in state.generated_docs:
        documents["API"] = state.generated_docs["api"]

    # Key is "githubReadme", not "GITHUBREADME" — every other doc here is
    # ALL_CAPS by convention, but this one deliberately breaks it: the dict
    # key is also the output filename (see markdown.py's `write_markdown`),
    # and the file is meant to be named `githubReadme.md`.
    if "githubReadme" in state.generated_docs:
        documents["githubReadme"] = state.generated_docs["githubReadme"]

    if "structure" in state.generated_docs:
        documents["STRUCTURE"] = state.generated_docs["structure"]

    if "changelog" in state.generated_docs:
        documents["CHANGELOG"] = state.generated_docs["changelog"]

    if "dependencies" in state.generated_docs:
        documents["DEPENDENCIES"] = state.generated_docs["dependencies"]

    if state.review.results:
        documents["REVIEW"] = render_review_markdown(state.review)

    if "README" in documents:
        documents["README"] = _insert_table_of_contents(documents["README"], documents)

    return documents


def _extract_headers(markdown_text: str) -> list[tuple[int, str]]:
    """Return `(level, text)` for each `##`/`###` Markdown header found, in the order they appear."""
    headers = []
    for line in markdown_text.splitlines():
        match = re.match(r"^(#{2,3})\s+(.+)$", line.strip())
        if match:
            headers.append((len(match.group(1)), match.group(2).strip()))
    return headers


def _slugify(text: str) -> str:
    """GitHub-style header anchor slug: lowercase, spaces to hyphens, strip punctuation."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s+", "-", slug)


def _build_table_of_contents(readme_text: str, documents: dict[str, str]) -> str | None:
    """
    Deterministically build a Table of Contents linking to README's own real
    section headers, plus whichever sibling documents actually exist in
    this export set. Nothing here is invented or LLM-generated — every
    entry comes directly from a header or filename that's actually present.
    Returns None if there's nothing worth a ToC for (e.g. the minimal
    fallback README with no sections).
    """
    headers = _extract_headers(readme_text)
    sibling_links = [f"- [{label}]({name}.md)" for name, label in _SIBLING_DOC_LABELS.items() if name in documents]

    if not headers and not sibling_links:
        return None

    lines = ["## Table of Contents", ""]
    for level, text in headers:
        indent = "  " * (level - 2)
        lines.append(f"{indent}- [{text}](#{_slugify(text)})")
    if sibling_links:
        lines.append("- Other documents in this set:")
        lines.extend(f"  {link}" for link in sibling_links)
    lines.append("")
    return "\n".join(lines)


def _insert_table_of_contents(readme_text: str, documents: dict[str, str]) -> str:
    """Insert the Table of Contents right after README's title line, or prepend it if there's no title."""
    toc = _build_table_of_contents(readme_text, documents)
    if toc is None:
        return readme_text

    stripped = readme_text.lstrip("\n")
    if not stripped.startswith("# "):
        return f"{toc}\n{readme_text}"

    title_line, _, rest = stripped.partition("\n")
    return f"{title_line}\n\n{toc}\n{rest.lstrip(chr(10))}"


def render_review_markdown(review: ReviewCollection) -> str:
    lines = ["# Documentation Review\n"]
    for result in review.results:
        status = "PASS" if result.passed else "ISSUES FOUND"
        lines.append(f"## {result.doc_name}\n")
        lines.append(f"**Status:** {status}\n")
        if result.error:
            lines.append(f"_Review could not complete: {result.error}_\n")
        elif result.issues:
            lines.extend(f"- {issue}" for issue in result.issues)
            lines.append("")
    return "\n".join(lines)