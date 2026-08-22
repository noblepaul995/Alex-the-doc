"""Tests for the Memory System (memory/repository.py + database/*), against a real SQLite database — no mocking, it's local and offline."""

from __future__ import annotations

from pathlib import Path

from graph.chunk import ChunkDocumentation
from graph.file_doc import FileDocumentation
from memory.repository import RepositoryMemory
from parsers.python_parser import parse_python_file


def _memory(tmp_path: Path) -> RepositoryMemory:
    return RepositoryMemory(tmp_path / "memory.sqlite3")


class TestFileHashes:
    def test_first_run_returns_none(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        assert mem.load_previous_hashes("/repo") is None

    def test_saved_hashes_are_loaded(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        mem.save_file_hash("/repo", "a.py", "hash1", "python")
        mem.save_file_hash("/repo", "b.py", "hash2", "python")
        assert mem.load_previous_hashes("/repo") == {"a.py": "hash1", "b.py": "hash2"}

    def test_saving_again_updates_in_place(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        mem.save_file_hash("/repo", "a.py", "hash1", "python")
        mem.save_file_hash("/repo", "a.py", "hash2", "python")
        assert mem.load_previous_hashes("/repo") == {"a.py": "hash2"}

    def test_repositories_are_isolated(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        mem.save_file_hash("/repo-a", "a.py", "hash1", "python")
        assert mem.load_previous_hashes("/repo-b") is None


class TestParseResultCache:
    def test_matching_hash_returns_cached_result(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        result = parse_python_file("a.py", b"def foo():\n    pass\n")
        mem.save_parse_result("/repo", "a.py", "hashX", result)
        loaded = mem.load_parse_result("/repo", "a.py", "hashX")
        assert loaded is not None
        assert loaded.symbols[0].name == "foo"

    def test_mismatched_hash_is_a_miss(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        result = parse_python_file("a.py", b"def foo():\n    pass\n")
        mem.save_parse_result("/repo", "a.py", "hashX", result)
        assert mem.load_parse_result("/repo", "a.py", "hashY") is None

    def test_no_prior_entry_is_a_miss(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        assert mem.load_parse_result("/repo", "never-saved.py", "anyhash") is None


class TestChunkDocsCache:
    def test_saved_list_round_trips(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        docs = [
            ChunkDocumentation(chunk_id="a.py:1-2", file_path="a.py", summary="s1", model="fake"),
            ChunkDocumentation(chunk_id="a.py:3-4", file_path="a.py", summary="s2", model="fake"),
        ]
        mem.save_chunk_docs("/repo", "a.py", "hashX", docs)
        loaded = mem.load_chunk_docs("/repo", "a.py", "hashX")
        assert loaded is not None
        assert [d.summary for d in loaded] == ["s1", "s2"]

    def test_stale_hash_is_a_miss(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        docs = [ChunkDocumentation(chunk_id="a.py:1-2", file_path="a.py", summary="s1", model="fake")]
        mem.save_chunk_docs("/repo", "a.py", "hashX", docs)
        assert mem.load_chunk_docs("/repo", "a.py", "different-hash") is None


class TestFileDocCache:
    def test_saved_doc_round_trips(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        doc = FileDocumentation(file_path="a.py", language="python", summary="whole file", model="fake")
        mem.save_file_doc("/repo", "a.py", "hashX", doc)
        loaded = mem.load_file_doc("/repo", "a.py", "hashX")
        assert loaded is not None
        assert loaded.summary == "whole file"

    def test_stale_hash_is_a_miss(self, tmp_path: Path) -> None:
        mem = _memory(tmp_path)
        doc = FileDocumentation(file_path="a.py", language="python", summary="whole file", model="fake")
        mem.save_file_doc("/repo", "a.py", "hashX", doc)
        assert mem.load_file_doc("/repo", "a.py", "wrong-hash") is None


class TestPersistenceAcrossConnections:
    def test_data_survives_reconnecting(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.sqlite3"
        RepositoryMemory(db_path).save_file_hash("/repo", "a.py", "hash1", "python")
        # Fresh connection, same file — simulates a second CLI invocation.
        reconnected = RepositoryMemory(db_path)
        assert reconnected.load_previous_hashes("/repo") == {"a.py": "hash1"}
