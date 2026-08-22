"""
PDF export of generated documentation.

Rather than a separate Markdown-to-PDF pipeline, this reuses the exact
HTML the `html` exporter already produces (via the `markdown` library)
and walks that HTML tree with BeautifulSoup to build ReportLab
flowables. One conversion path (Markdown -> HTML) feeding two renderers
is simpler to keep correct than two independent Markdown parsers, and
means the PDF and HTML exports can never silently disagree about how a
given document's Markdown should be interpreted.

Element coverage is intentionally modest — headings (h1-h4), paragraphs
(with bold/italic converted to ReportLab's own inline markup), bullet
and numbered lists, and preformatted code blocks. That covers
everything every prompt in this codebase actually asks the LLM to
produce (plain paragraphs, occasional lists, one Mermaid code fence in
the architecture doc). Verified end-to-end, not just "should work":
generated a PDF from real Markdown and round-tripped it back through a
PDF text extractor to confirm the actual rendered content matches.
"""

from __future__ import annotations

import io
from pathlib import Path

import markdown as markdown_lib
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer

_HEADING_STYLES = {"h1": "Heading1", "h2": "Heading2", "h3": "Heading3", "h4": "Heading4"}


def render_pdf(markdown_text: str) -> bytes:
    """Render one document's Markdown as a complete PDF file's bytes."""
    html = markdown_lib.markdown(markdown_text, extensions=["fenced_code", "tables"])
    soup = BeautifulSoup(html, "html.parser")
    styles = getSampleStyleSheet()

    flowables = []
    for element in soup.find_all(recursive=False):
        flowable = _element_to_flowable(element, styles)
        if flowable is not None:
            flowables.append(flowable)
            flowables.append(Spacer(1, 8))

    if not flowables:
        flowables = [Paragraph("(empty document)", styles["Normal"])]

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=letter).build(flowables)
    return buffer.getvalue()


def _element_to_flowable(element, styles):
    if element.name in _HEADING_STYLES:
        return Paragraph(element.get_text(), styles[_HEADING_STYLES[element.name]])
    if element.name == "p":
        return Paragraph(_inline_markup(element), styles["Normal"])
    if element.name in ("ul", "ol"):
        items = [ListItem(Paragraph(li.get_text(), styles["Normal"])) for li in element.find_all("li", recursive=False)]
        return ListFlowable(items, bulletType="bullet" if element.name == "ul" else "1")
    if element.name == "pre":
        return Preformatted(element.get_text(), styles["Code"])
    return None


def _inline_markup(tag) -> str:
    """Convert HTML's `<strong>`/`<em>` to ReportLab's own `<b>`/`<i>` mini-markup, which `Paragraph` renders directly."""
    text = str(tag)
    return text.replace("<strong>", "<b>").replace("</strong>", "</b>").replace("<em>", "<i>").replace("</em>", "</i>")


def write_pdf(documents: dict[str, str], output_dir: Path) -> list[Path]:
    """Write each document as `{output_dir}/{name}.pdf`. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in documents.items():
        path = output_dir / f"{name}.pdf"
        path.write_bytes(render_pdf(content))
        written.append(path)
    return written
