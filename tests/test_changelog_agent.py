"""Tests for agents/changelog_agent.py — deterministic, so tested against real git repos, no mocking."""

from __future__ import annotations

import subprocess

from agents.changelog_agent import _read_git_log, changelog_node
from graph.state import RepositoryState


def _init_repo(path, *, commits: list[str] = ()):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    for message in commits:
        (path / "file.txt").write_text(message)
        subprocess.run(["git", "add", "."], cwd=path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=path, check=True)


class TestReadGitLog:
    def test_returns_none_for_non_git_directory(self, tmp_path):
        assert _read_git_log(tmp_path, max_commits=20, timeout_seconds=10.0) is None

    def test_returns_empty_list_for_git_repo_with_no_commits(self, tmp_path):
        _init_repo(tmp_path)
        assert _read_git_log(tmp_path, max_commits=20, timeout_seconds=10.0) == []

    def test_returns_commits_in_recency_order(self, tmp_path):
        _init_repo(tmp_path, commits=["first commit", "second commit", "third commit"])
        entries = _read_git_log(tmp_path, max_commits=20, timeout_seconds=10.0)
        assert entries is not None
        assert len(entries) == 3
        subjects = [subject for _, _, subject in entries]
        assert subjects == ["third commit", "second commit", "first commit"]

    def test_respects_max_commits_cap(self, tmp_path):
        _init_repo(tmp_path, commits=[f"commit {i}" for i in range(5)])
        entries = _read_git_log(tmp_path, max_commits=2, timeout_seconds=10.0)
        assert entries is not None
        assert len(entries) == 2

    def test_each_entry_has_hash_date_and_subject(self, tmp_path):
        _init_repo(tmp_path, commits=["only commit"])
        entries = _read_git_log(tmp_path, max_commits=20, timeout_seconds=10.0)
        assert entries is not None
        commit_hash, date, subject = entries[0]
        assert len(commit_hash) > 0
        assert len(date) == 10  # YYYY-MM-DD
        assert subject == "only commit"


class TestChangelogNode:
    def test_skips_cleanly_for_non_git_repository(self, tmp_path):
        state = RepositoryState(repository_path=tmp_path)
        result = changelog_node(state)
        assert "changelog" not in result.get("generated_docs", {})

    def test_skips_cleanly_for_git_repo_with_no_commits(self, tmp_path):
        _init_repo(tmp_path)
        state = RepositoryState(repository_path=tmp_path)
        result = changelog_node(state)
        assert "changelog" not in result.get("generated_docs", {})

    def test_generates_changelog_from_real_commits(self, tmp_path):
        _init_repo(tmp_path, commits=["add feature X", "fix bug Y"])
        state = RepositoryState(repository_path=tmp_path)
        result = changelog_node(state)
        changelog = result["generated_docs"]["changelog"]
        assert "# Recent Changes" in changelog
        assert "add feature X" in changelog
        assert "fix bug Y" in changelog
        # More recent commit should appear first.
        assert changelog.index("fix bug Y") < changelog.index("add feature X")

    def test_never_invents_content_beyond_real_commits(self, tmp_path):
        _init_repo(tmp_path, commits=["only real commit"])
        state = RepositoryState(repository_path=tmp_path)
        result = changelog_node(state)
        changelog = result["generated_docs"]["changelog"]
        assert changelog.count("- `") == 1  # exactly one commit line, nothing fabricated
