"""Markdown export of generated documentation — the simplest format, since every generated doc already is Markdown."""

from __future__ import annotations

from pathlib import Path


def write_markdown(documents: dict[str, str], output_dir: Path) -> list[Path]:
    """Write each document as `{output_dir}/{name}.md`. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in documents.items():
        path = output_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
