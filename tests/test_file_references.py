"""Tests for utils/file_references.py."""

from __future__ import annotations

from utils.file_references import find_nonexistent_file_references, render_missing_file_findings


class TestFindNonexistentFileReferences:
    def test_real_full_path_not_flagged(self):
        text = "See `agents/repo_agent.py` for the digest builder."
        assert find_nonexistent_file_references(text, {"agents/repo_agent.py"}) == []

    def test_real_bare_filename_not_flagged(self):
        text = "See `repo_agent.py` for details."
        assert find_nonexistent_file_references(text, {"agents/repo_agent.py"}) == []

    def test_fabricated_path_is_flagged(self):
        text = "See `agents/nonexistent_agent.py` for details."
        assert find_nonexistent_file_references(text, {"agents/repo_agent.py"}) == ["agents/nonexistent_agent.py"]

    def test_subtle_pluralization_mismatch_is_flagged(self):
        text = "Utility functions live in `utils/helper.py`."
        assert find_nonexistent_file_references(text, {"utils/helpers.py"}) == ["utils/helper.py"]

    def test_code_reference_without_slash_or_extension_not_flagged(self):
        text = "Calls `state.generated_docs` and `ChatMessage(role=Role.USER)`."
        assert find_nonexistent_file_references(text, {"agents/repo_agent.py"}) == []

    def test_known_output_filenames_not_flagged(self):
        text = "See `README.md` and `ARCHITECTURE.md` for more."
        assert find_nonexistent_file_references(text, set()) == []

    def test_deduplicates_repeated_references(self):
        text = "Bad `agents/fake.py` appears here and again `agents/fake.py` here."
        assert find_nonexistent_file_references(text, set()) == ["agents/fake.py"]

    def test_no_backticks_means_nothing_checked(self):
        text = "agents/fake.py is mentioned with no backticks at all."
        assert find_nonexistent_file_references(text, set()) == []

    def test_real_directory_not_flagged_when_repository_root_given(self, tmp_path):
        (tmp_path / "graph").mkdir()
        text = "**Location:** `graph/`"
        assert find_nonexistent_file_references(text, set(), repository_root=tmp_path) == []

    def test_fabricated_directory_is_flagged_when_repository_root_given(self, tmp_path):
        text = "**Location:** `nonexistent_dir/`"
        assert find_nonexistent_file_references(text, set(), repository_root=tmp_path) == ["nonexistent_dir/"]

    def test_directory_reference_skipped_without_repository_root(self):
        # Can't verify a directory claim without the real filesystem — must not guess either way.
        text = "**Location:** `some_dir/`"
        assert find_nonexistent_file_references(text, set()) == []

    def test_real_directory_with_no_scanned_source_files_still_not_flagged(self, tmp_path):
        # e.g. prompts/ contains only .md templates, which aren't a parsed source language,
        # so nothing under it ends up in real_paths — but the directory itself is still real.
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "readme.md").write_text("template")
        text = "loads templates from the `prompts/` directory"
        assert find_nonexistent_file_references(text, set(), repository_root=tmp_path) == []

    def test_api_path_fragment_not_flagged_as_a_file(self):
        # Contains a slash but isn't a file/directory reference at all — no extension,
        # no trailing slash. Must not be treated as a path candidate in the first place.
        text = "a direct HTTP client for the Anthropic `/v1/messages` API"
        assert find_nonexistent_file_references(text, {"agents/repo_agent.py"}) == []

    def test_url_like_fragment_not_flagged(self):
        text = "see `api/v2/users` for the old endpoint shape"
        assert find_nonexistent_file_references(text, set()) == []


class TestRenderMissingFileFindings:
    def test_renders_one_line_per_missing_file(self):
        rendered = render_missing_file_findings(["agents/fake.py", "utils/helper.py"])
        assert "agents/fake.py" in rendered
        assert "utils/helper.py" in rendered
        assert rendered.count("\n") == 1  # two lines total
