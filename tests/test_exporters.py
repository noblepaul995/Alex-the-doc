"""Tests for the Export stage: exporters/assemble.py, markdown.py, html.py, json.py, pdf.py, agents/export_agent.py.

Real files are written and read back (no mocking) — every exporter here
is local/offline, so there's nothing to avoid by mocking, and mocking
would just re-describe the API instead of verifying it actually works.
"""

from __future__ import annotations

import orjson
import pytest
from pypdf import PdfReader

from agents.export_agent import SUPPORTED_FORMATS, export_documentation
from exporters.assemble import build_export_documents, render_review_markdown
from exporters.html import render_html, write_html
from exporters.json import build_json_export, write_json
from exporters.markdown import write_markdown
from exporters.pdf import render_pdf, write_pdf
from graph.review import ReviewCollection, ReviewResult
from graph.state import RepositoryState


def _state(tmp_path, **kwargs):
    state = RepositoryState(repository_path=tmp_path)
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


class TestBuildExportDocuments:
    def test_readme_from_generated_docs(self, tmp_path):
        state = _state(tmp_path, generated_docs={"readme": "# Real README"})
        docs = build_export_documents(state)
        assert docs["README"] == "# Real README"

    def test_readme_fallback_from_repo_summary(self, tmp_path):
        state = _state(tmp_path, repo_summary="A summary.")
        docs = build_export_documents(state)
        assert "A summary." in docs["README"]

    def test_no_readme_key_when_nothing_available(self, tmp_path):
        state = _state(tmp_path)
        docs = build_export_documents(state)
        assert "README" not in docs

    def test_architecture_and_api_only_when_present(self, tmp_path):
        state = _state(tmp_path, generated_docs={"architecture": "Arch doc."})
        docs = build_export_documents(state)
        assert docs["ARCHITECTURE"] == "Arch doc."

    def test_toc_links_readmes_own_headers(self, tmp_path):
        readme = "# My Project\n\nIntro.\n\n## Overview\n\nText.\n\n## Key Design Decisions\n\n- One.\n"
        state = _state(tmp_path, generated_docs={"readme": readme})
        docs = build_export_documents(state)
        assert "## Table of Contents" in docs["README"]
        assert "[Overview](#overview)" in docs["README"]
        assert "[Key Design Decisions](#key-design-decisions)" in docs["README"]
        # ToC must come after the title and before the rest of the content.
        assert docs["README"].index("# My Project") < docs["README"].index("## Table of Contents") < docs["README"].index("Intro.")

    def test_toc_links_sibling_documents_that_exist(self, tmp_path):
        readme = "# My Project\n\n## Overview\n\nText.\n"
        state = _state(
            tmp_path,
            generated_docs={"readme": readme, "architecture": "Arch.", "structure": "Struct."},
        )
        docs = build_export_documents(state)
        assert "[Architecture](ARCHITECTURE.md)" in docs["README"]
        assert "[Folder Structure](STRUCTURE.md)" in docs["README"]
        assert "API Reference" not in docs["README"]  # not present in this state, must not be linked

    def test_toc_skipped_for_headerless_fallback_readme(self, tmp_path):
        state = _state(tmp_path, repo_summary="A summary.")
        docs = build_export_documents(state)
        assert "Table of Contents" not in docs["README"]
        assert "API" not in docs

    def test_review_included_when_review_ran(self, tmp_path):
        state = _state(tmp_path, review=ReviewCollection(results=[ReviewResult(doc_name="readme", passed=True)]))
        docs = build_export_documents(state)
        assert "REVIEW" in docs
        assert "readme" in docs["REVIEW"]

    def test_empty_state_produces_no_documents(self, tmp_path):
        state = _state(tmp_path)
        assert build_export_documents(state) == {}


class TestRenderReviewMarkdown:
    def test_passed_review_shows_pass_status(self):
        review = ReviewCollection(results=[ReviewResult(doc_name="readme", passed=True)])
        text = render_review_markdown(review)
        assert "PASS" in text
        assert "readme" in text

    def test_flagged_review_lists_issues(self):
        review = ReviewCollection(results=[ReviewResult(doc_name="api", passed=False, issues=["unsupported claim"])])
        text = render_review_markdown(review)
        assert "ISSUES FOUND" in text
        assert "unsupported claim" in text

    def test_review_error_is_shown(self):
        review = ReviewCollection(results=[ReviewResult(doc_name="readme", passed=False, error="provider timeout")])
        text = render_review_markdown(review)
        assert "provider timeout" in text


class TestMarkdownExport:
    def test_writes_one_file_per_document(self, tmp_path):
        paths = write_markdown({"README": "# Hi", "API": "# API"}, tmp_path)
        assert len(paths) == 2
        assert (tmp_path / "README.md").read_text() == "# Hi"
        assert (tmp_path / "API.md").read_text() == "# API"

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "nested" / "docs"
        write_markdown({"README": "# Hi"}, nested)
        assert (nested / "README.md").exists()


class TestHtmlExport:
    def test_render_html_produces_valid_page(self):
        html = render_html("README", "# Title\n\nSome **bold** text.")
        assert "<h1>Title</h1>" in html
        assert "<strong>bold</strong>" in html
        assert "<!DOCTYPE html>" in html

    def test_write_html_creates_files(self, tmp_path):
        paths = write_html({"README": "# Hi"}, tmp_path)
        assert len(paths) == 1
        assert (tmp_path / "README.html").exists()


class TestJsonExport:
    def test_build_json_export_includes_key_fields(self, tmp_path):
        state = _state(tmp_path, repo_summary="A summary.", generated_docs={"readme": "# R"})
        payload = build_json_export(state)
        assert payload["repo_summary"] == "A summary."
        assert payload["generated_docs"] == {"readme": "# R"}
        assert "statistics" in payload

    def test_write_json_round_trips(self, tmp_path):
        state = _state(tmp_path, repo_summary="A summary.")
        path = write_json(state, tmp_path)
        loaded = orjson.loads(path.read_bytes())
        assert loaded["repo_summary"] == "A summary."


class TestPdfExport:
    def test_render_pdf_produces_valid_file(self):
        data = render_pdf("# Title\n\nSome text.")
        assert data[:4] == b"%PDF"

    def test_pdf_content_round_trips_through_extraction(self, tmp_path):
        md = "# My Title\n\nSome **bold** text and *italic* text.\n\n- item one\n- item two\n"
        data = render_pdf(md)
        path = tmp_path / "out.pdf"
        path.write_bytes(data)
        text = PdfReader(str(path)).pages[0].extract_text()
        assert "My Title" in text
        assert "bold text" in text
        assert "item one" in text
        assert "item two" in text

    def test_empty_document_does_not_crash(self):
        data = render_pdf("")
        assert data[:4] == b"%PDF"

    def test_write_pdf_creates_files(self, tmp_path):
        paths = write_pdf({"README": "# Hi"}, tmp_path)
        assert len(paths) == 1
        assert (tmp_path / "README.pdf").exists()


class TestExportDocumentation:
    def test_unknown_format_raises(self, tmp_path):
        state = _state(tmp_path, repo_summary="A summary.")
        with pytest.raises(ValueError):
            export_documentation(state, ["not_a_real_format"], tmp_path)

    def test_writes_all_requested_formats(self, tmp_path):
        state = _state(tmp_path, repo_summary="A summary.", generated_docs={"readme": "# R"})
        written = export_documentation(state, ["markdown", "html", "json", "pdf"], tmp_path)
        assert set(written) == {"markdown", "html", "json", "pdf"}
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / "README.html").exists()
        assert (tmp_path / "README.pdf").exists()
        assert (tmp_path / "documentation.json").exists()

    def test_empty_state_writes_nothing(self, tmp_path):
        state = _state(tmp_path)
        written = export_documentation(state, ["markdown"], tmp_path)
        assert written == {"markdown": []}

    def test_supported_formats_matches_writers(self):
        assert SUPPORTED_FORMATS == {"markdown", "html", "pdf", "json"}
