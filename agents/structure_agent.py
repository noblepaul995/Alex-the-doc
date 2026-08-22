"""
Structure Agent — deterministically renders `state.generated_docs["structure"]`:
a folder-grouped table of every documented file paired with a one-line
description of its purpose.

Deliberately *not* an LLM call. Every line here is taken directly from the
File Documentation Agent's already-produced, already-grounded per-file
summaries (`state.file_docs`) — this agent only reformats verified data into
a table, so it's free, instant, and structurally can't hallucinate a file
that doesn't exist or misdescribe one that does. It runs alongside
`architecture`/`api`/`readme` (see graph/graph.py's fan-out) even though it
doesn't need `repo_summary`, since running it earlier would only save a few
milliseconds and consistency with its siblings keeps the graph simple.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from graph.state import RepositoryState
from utils.logger import get_logger

log = get_logger(__name__)

_SENTENCE_TERMINATORS = (". ", ".\n", "! ", "!\n", "? ", "?\n")


def _first_sentence(summary: str) -> str:
    """Take just the first sentence of a (2-5 sentence) file summary for a compact table cell."""
    text = summary.strip().replace("\n", " ")
    if not text:
        return ""
    earliest = -1
    for terminator in _SENTENCE_TERMINATORS:
        idx = text.find(terminator)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest != -1:
        return text[: earliest + 1].strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def structure_node(state: RepositoryState) -> dict[str, object]:
    """LangGraph node: generate `state.generated_docs["structure"]`."""
    docs_by_dir: dict[str, list[tuple[str, str]]] = {}
    for doc in sorted(state.file_docs.docs, key=lambda d: d.file_path):
        if doc.error is not None:
            continue
        parent = str(PurePosixPath(doc.file_path).parent)
        directory = "(repository root)" if parent in ("", ".") else parent
        docs_by_dir.setdefault(directory, []).append((doc.file_path, _first_sentence(doc.summary)))

    if not docs_by_dir:
        log.info("Structure Agent: no documented files available, skipping")
        return {"generated_docs": state.generated_docs}

    lines = [
        "# Folder Structure",
        "",
        "Every entry below is taken directly from a file that was actually parsed "
        "and documented in this run — nothing here is inferred or assumed.",
        "",
    ]
    for directory in sorted(docs_by_dir):
        lines.append(f"### `{directory}`" if directory != "(repository root)" else "### Repository root")
        lines.append("")
        lines.append("| File | Purpose |")
        lines.append("|---|---|")
        for file_path, blurb in docs_by_dir[directory]:
            name = PurePosixPath(file_path).name
            lines.append(f"| `{name}` | {blurb or '_no summary available_'} |")
        lines.append("")

    state.generated_docs["structure"] = "\n".join(lines).rstrip() + "\n"
    log.info(f"Structure Agent: {len(docs_by_dir)} directory(ies), {len(state.file_docs.docs)} file(s) tabulated")
    return {"generated_docs": state.generated_docs}
