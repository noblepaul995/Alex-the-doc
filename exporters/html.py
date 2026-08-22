"""HTML export of generated documentation, via the standard `markdown` library (fenced code + tables extensions)."""

from __future__ import annotations

from pathlib import Path

import markdown as markdown_lib

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.6; color: #1a1a1a; }}
  pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; border-radius: 4px; }}
  code {{ background: #f4f4f4; padding: 0.15em 0.35em; border-radius: 3px; }}
  pre code {{ background: none; padding: 0; }}
  h1, h2, h3 {{ border-bottom: 1px solid #e0e0e0; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_html(title: str, markdown_text: str) -> str:
    """Render a single document's Markdown as a complete, styled HTML page."""
    body = markdown_lib.markdown(markdown_text, extensions=["fenced_code", "tables"])
    return _TEMPLATE.format(title=title, body=body)


def write_html(documents: dict[str, str], output_dir: Path) -> list[Path]:
    """Write each document as `{output_dir}/{name}.html`. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in documents.items():
        path = output_dir / f"{name}.html"
        path.write_text(render_html(name, content), encoding="utf-8")
        written.append(path)
    return written
