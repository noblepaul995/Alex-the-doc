"""Tests for the Scanner Agent (scanner/*.py + agents/scanner.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scanner.hashes import ChangeStatus, diff_hashes
from scanner.ignore import is_binary
from scanner.language import detect_language
from scanner.scanner import scan_repository


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "assets").mkdir()

    (tmp_path / ".gitignore").write_text("*.log\nsecret.txt\n")
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    (tmp_path / "src" / "utils.ts").write_text("export const x = 1;\n")
    (tmp_path / "secret.txt").write_text("nope\n")
    (tmp_path / "debug.log").write_text("log line\n")
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("module.exports = {};\n")
    (tmp_path / "assets" / "logo.png").write_bytes(b"\x89PNG\x00\x01\x02binarydata")
    return tmp_path


def test_scan_always_skips_secret_files_regardless_of_gitignore(tmp_path: Path) -> None:
    # Deliberately empty .gitignore, like the real repo that surfaced this bug —
    # secrets must be skipped even when the repo's own .gitignore says nothing.
    (tmp_path / ".gitignore").write_text("")
    (tmp_path / ".env").write_text("API_KEY=supersecret\n")
    (tmp_path / ".env.example").write_text("API_KEY=\n")
    (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    (tmp_path / "id_rsa.pub").write_text("ssh-rsa AAAA...\n")
    (tmp_path / "credentials.json").write_text('{"key": "secret"}')
    (tmp_path / "normal.py").write_text("print('fine')\n")

    result = scan_repository(tmp_path)
    relative_paths = {f.relative_path for f in result.files}

    assert ".env" not in relative_paths
    assert "id_rsa" not in relative_paths
    assert "credentials.json" not in relative_paths
    # Safe templates and public keys should NOT be treated as secrets.
    assert ".env.example" in relative_paths
    assert "id_rsa.pub" in relative_paths
    assert "normal.py" in relative_paths
    assert result.skipped_secret == 3


def test_is_secret_file_patterns() -> None:
    from scanner.ignore import is_secret_file

    assert is_secret_file(".env")
    assert is_secret_file("nested/dir/.env")
    assert is_secret_file(".env.local")
    assert is_secret_file("service-account-prod.json")
    assert not is_secret_file(".env.example")
    assert not is_secret_file("id_rsa.pub")
    assert not is_secret_file("src/main.py")


def test_scan_respects_gitignore_and_always_ignore_dirs(sample_repo: Path) -> None:
    result = scan_repository(sample_repo)
    relative_paths = {f.relative_path for f in result.files}

    assert "src/main.py" in relative_paths
    assert "src/utils.ts" in relative_paths
    assert "secret.txt" not in relative_paths
    assert "debug.log" not in relative_paths
    assert not any(p.startswith("node_modules/") for p in relative_paths)


def test_scan_skips_binary_files(sample_repo: Path) -> None:
    result = scan_repository(sample_repo)
    relative_paths = {f.relative_path for f in result.files}
    assert "assets/logo.png" not in relative_paths
    assert result.skipped_binary == 1


def test_scan_detects_language(sample_repo: Path) -> None:
    result = scan_repository(sample_repo)
    by_path = {f.relative_path: f for f in result.files}
    assert by_path["src/main.py"].language == "python"
    assert by_path["src/utils.ts"].language == "typescript"


def test_scan_computes_stable_hash(sample_repo: Path) -> None:
    first = scan_repository(sample_repo)
    second = scan_repository(sample_repo)
    first_hashes = {f.relative_path: f.content_hash for f in first.files}
    second_hashes = {f.relative_path: f.content_hash for f in second.files}
    assert first_hashes == second_hashes
    assert all(h is not None and len(h) == 64 for h in first_hashes.values())


def test_scan_nonexistent_path_reports_error(tmp_path: Path) -> None:
    result = scan_repository(tmp_path / "does_not_exist")
    assert result.files == []
    assert len(result.errors) == 1


def test_is_binary_detects_null_bytes(tmp_path: Path) -> None:
    text_file = tmp_path / "text.txt"
    text_file.write_text("just text")
    binary_file = tmp_path / "bin.dat"
    binary_file.write_bytes(b"\x00\x01\x02")

    assert not is_binary(text_file)
    assert is_binary(binary_file)


def test_is_binary_uses_extension_fast_path(tmp_path: Path) -> None:
    fake_png = tmp_path / "image.png"
    fake_png.write_text("not actually binary content")  # no NUL byte
    assert is_binary(fake_png)  # caught by extension, not content


@pytest.mark.parametrize(
    ("filename", "expected_language"),
    [
        ("main.py", "python"),
        ("component.tsx", "tsx"),
        ("Dockerfile", "dockerfile"),
        ("README.md", "markdown"),
        ("data.unknownext", None),
    ],
)
def test_detect_language_by_extension(tmp_path: Path, filename: str, expected_language: str | None) -> None:
    path = tmp_path / filename
    assert detect_language(path) == expected_language


def test_detect_language_by_shebang(tmp_path: Path) -> None:
    path = tmp_path / "run"  # no extension
    peek = b"#!/usr/bin/env python3\nprint('hi')\n"
    assert detect_language(path, peek_bytes=peek) == "python"


def test_diff_hashes_first_run_reports_everything_added(sample_repo: Path) -> None:
    result = scan_repository(sample_repo)
    current = {f.relative_path: f for f in result.files}
    change_set = diff_hashes(current=current, previous=None)
    assert set(change_set.added) == set(current.keys())
    assert change_set.modified == []
    assert change_set.status_of("src/main.py") == ChangeStatus.ADDED


def test_diff_hashes_detects_modifications_and_deletions() -> None:
    from graph.state import ProjectFile
    from pathlib import Path as P

    current = {
        "a.py": ProjectFile(path=P("a.py"), relative_path="a.py", size_bytes=1, content_hash="newhash"),
        "c.py": ProjectFile(path=P("c.py"), relative_path="c.py", size_bytes=1, content_hash="samehash"),
    }
    previous = {"a.py": "oldhash", "b.py": "somehash", "c.py": "samehash"}

    change_set = diff_hashes(current=current, previous=previous)
    assert change_set.modified == ["a.py"]
    assert change_set.deleted == ["b.py"]
    assert change_set.unchanged == ["c.py"]
    assert change_set.changed == ["a.py"]
