"""Tests for tools/eval_harness.py's compute_metrics() — the pure, model-free half of the harness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from eval_harness import compute_metrics  # noqa: E402
from graph.review import ReviewCollection, ReviewResult  # noqa: E402
from graph.state import RepositoryState  # noqa: E402

_GOOD_README = (
    "# Demo\n\n## Table of Contents\n\n- [Overview](#overview)\n\n"
    "## Overview\n\nWords go here explaining the system in real detail with several sentences.\n\n"
    "## Core Components\n\nMore words.\n\n## Key Design Decisions\n\n- A real decision.\n"
)


class TestComputeMetrics:
    def test_detects_present_sections_and_toc(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": _GOOD_README}
        metrics = compute_metrics(state)
        assert metrics.readme_has_overview is True
        assert metrics.readme_has_core_components is True
        assert metrics.readme_has_key_design_decisions is True
        assert metrics.readme_has_toc is True
        assert metrics.readme_generated is True

    def test_detects_missing_sections(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": "# Demo\n\nJust a title, no sections at all.\n"}
        metrics = compute_metrics(state)
        assert metrics.readme_has_overview is False
        assert metrics.readme_has_core_components is False
        assert metrics.readme_has_key_design_decisions is False
        assert metrics.readme_has_toc is False

    def test_word_count_is_real(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": "one two three four five"}
        metrics = compute_metrics(state)
        assert metrics.readme_word_count == 5

    def test_no_readme_gives_zero_word_count_and_false_flags(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        metrics = compute_metrics(state)
        assert metrics.readme_generated is False
        assert metrics.readme_word_count == 0
        assert metrics.readme_has_overview is False

    def test_categorizes_banned_phrase_issues(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": _GOOD_README}
        state.review = ReviewCollection(
            results=[
                ReviewResult(
                    doc_name="readme",
                    passed=False,
                    issues=['Contains banned generic phrase: "promotes flexibility"'],
                    model="deterministic-scan",
                )
            ]
        )
        metrics = compute_metrics(state)
        assert metrics.banned_phrase_issue_count == 1
        assert metrics.nonexistent_file_issue_count == 0
        assert metrics.flagged_document_count == 1

    def test_categorizes_nonexistent_file_issues(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": _GOOD_README}
        state.review = ReviewCollection(
            results=[
                ReviewResult(
                    doc_name="readme",
                    passed=False,
                    issues=['References a file that doesn\'t exist in this repository: "agents/fake.py"'],
                    model="deterministic-scan",
                )
            ]
        )
        metrics = compute_metrics(state)
        assert metrics.nonexistent_file_issue_count == 1
        assert metrics.banned_phrase_issue_count == 0

    def test_categorizes_other_llm_found_issues(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": _GOOD_README}
        state.review = ReviewCollection(
            results=[ReviewResult(doc_name="readme", passed=False, issues=["Some other LLM-found issue"], model="fake")]
        )
        metrics = compute_metrics(state)
        assert metrics.other_issue_count == 1
        assert metrics.banned_phrase_issue_count == 0
        assert metrics.nonexistent_file_issue_count == 0

    def test_clean_review_gives_zero_issue_counts(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": _GOOD_README}
        state.review = ReviewCollection(results=[ReviewResult(doc_name="readme", passed=True, model="fake")])
        metrics = compute_metrics(state)
        assert metrics.flagged_document_count == 0
        assert metrics.banned_phrase_issue_count == 0
        assert metrics.nonexistent_file_issue_count == 0
        assert metrics.other_issue_count == 0

    def test_metadata_fields_recorded(self, tmp_path: Path) -> None:
        state = RepositoryState(repository_path=tmp_path)
        state.generated_docs = {"readme": _GOOD_README, "structure": "x", "changelog": "y"}
        metrics = compute_metrics(state, repo="my-repo", run_index=3)
        assert metrics.repo == "my-repo"
        assert metrics.run_index == 3
        assert metrics.structure_generated is True
        assert metrics.changelog_generated is True
        assert metrics.architecture_generated is False
