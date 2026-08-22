"""
Export orchestration — writes the assembled document set to disk in
one or more formats.

Deliberately *not* wired into `build_graph()` as an automatic pipeline
node, unlike every stage before it. The CLI already has a separate
`export` command distinct from `document` (`app.py`), and writing files
to disk is a meaningfully different kind of side effect than producing
in-memory state — a `document` run shouldn't silently start writing
files a user didn't ask for. `export_documentation` is a plain function
the `export` CLI command calls explicitly, after running the pipeline
itself.
"""

from __future__ import annotations

from pathlib import Path

from exporters.assemble import build_export_documents
from exporters.html import write_html
from exporters.json import write_json
from exporters.markdown import write_markdown
from exporters.pdf import write_pdf
from graph.state import RepositoryState
from utils.logger import get_logger

log = get_logger(__name__)

_WRITERS = {
    "markdown": lambda documents, state, output_dir: write_markdown(documents, output_dir),
    "html": lambda documents, state, output_dir: write_html(documents, output_dir),
    "pdf": lambda documents, state, output_dir: write_pdf(documents, output_dir),
    "json": lambda documents, state, output_dir: [write_json(state, output_dir)],
}

SUPPORTED_FORMATS = frozenset(_WRITERS)


def export_documentation(state: RepositoryState, formats: list[str], output_dir: Path) -> dict[str, list[Path]]:
    """
    Write the document set assembled from `state` in each of `formats`
    (any of `SUPPORTED_FORMATS`) to `output_dir`. Returns
    `{format: [paths written]}`. Raises `ValueError` for an unknown
    format — fails loudly rather than silently skipping a typo'd format
    name the caller might not notice went unwritten.
    """
    unknown = set(formats) - SUPPORTED_FORMATS
    if unknown:
        raise ValueError(f"Unknown export format(s): {sorted(unknown)}. Supported: {sorted(SUPPORTED_FORMATS)}")

    documents = build_export_documents(state)
    if not documents:
        log.info("Export: nothing to export (no generated docs or repository summary).")
        return {fmt: [] for fmt in formats}

    written: dict[str, list[Path]] = {}
    for fmt in formats:
        paths = _WRITERS[fmt](documents, state, output_dir)
        written[fmt] = paths
        log.info("Export: wrote %d file(s) as %s to %s", len(paths), fmt, output_dir)

    return written
