"""
JSON export — a single structured file combining everything the
pipeline produced, for programmatic consumption rather than reading.

Unlike the other three exporters, this one isn't built from
`exporters.assemble.build_export_documents()` — that assembly is
specifically about the *narrative* documents (README/ARCHITECTURE/API/
REVIEW), whereas the JSON export is meant to expose the underlying data
those narratives were built from (repo summary, per-document review
findings, run statistics) in a form a script could parse, which the
narrative Markdown documents aren't meant for.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from graph.state import RepositoryState


def build_json_export(state: RepositoryState) -> dict[str, object]:
    """Build the structured export payload from `state`."""
    return {
        "repository_path": str(state.repository_path),
        "repo_summary": state.repo_summary,
        "generated_docs": dict(state.generated_docs),
        "review": [
            {"doc_name": r.doc_name, "passed": r.passed, "issues": r.issues, "error": r.error} for r in state.review.results
        ],
        "statistics": {
            "files_total": state.statistics.files_total,
            "files_changed": state.statistics.files_changed,
            "chunks_documented": state.statistics.chunks_documented,
            "embeddings_generated": state.statistics.embeddings_generated,
            "vectors_persisted": state.statistics.vectors_persisted,
            "tokens_used": state.statistics.tokens_used,
            "duration_seconds": state.statistics.duration_seconds,
        },
    }


def write_json(state: RepositoryState, output_dir: Path, *, filename: str = "documentation.json") -> Path:
    """Write the structured export to `{output_dir}/{filename}`. Returns the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = build_json_export(state)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path
