"""Tests for the Chunk Agent (graph/chunk.py + agents/chunker.py)."""

from __future__ import annotations

from agents.chunker import _chunk_file
from parsers.go_parser import parse_go_file
from parsers.python_parser import parse_python_file
from parsers.rust_parser import parse_rust_file


def _all_symbol_names(chunks) -> list[str]:
    return [n for c in chunks for n in c.symbol_names]


class TestSmallFile:
    def test_small_file_packs_into_one_chunk(self) -> None:
        src = b"import os\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        result = parse_python_file("m.py", src)
        chunks = _chunk_file(result, src, max_tokens=1500)
        assert len(chunks) == 1
        assert chunks[0].is_partial is False
        assert set(chunks[0].symbol_names) == {"foo", "bar"}

    def test_module_level_gap_is_included(self) -> None:
        src = b"import os\nimport sys\n\ndef foo():\n    return 1\n"
        result = parse_python_file("m.py", src)
        chunks = _chunk_file(result, src, max_tokens=1500)
        joined = "".join(c.content for c in chunks)
        assert "import os" in joined
        assert "import sys" in joined


class TestByteCoverage:
    def test_no_gaps_or_overlaps_across_chunks(self) -> None:
        src = b"import os\n\ndef foo():\n    return 1\n\nclass Bar:\n    def m(self):\n        pass\n\nx = 1\n"
        result = parse_python_file("m.py", src)
        chunks = sorted(_chunk_file(result, src, max_tokens=1500), key=lambda c: c.start_byte)
        for a, b in zip(chunks, chunks[1:]):
            assert a.end_byte <= b.start_byte

    def test_every_symbol_appears_exactly_once(self) -> None:
        methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(10))
        src = ("class C:\n" + methods).encode()
        result = parse_python_file("m.py", src)
        # Tiny budget forces the class to split by method; a large budget would
        # keep it whole (tagged just ['C']), which is correct — no reason to
        # split something that already fits.
        chunks = _chunk_file(result, src, max_tokens=15)
        names = _all_symbol_names(chunks)
        assert sorted(names) == sorted(f"m{i}" for i in range(10))

    def test_small_class_stays_whole_not_split_by_method(self) -> None:
        methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(10))
        src = ("class C:\n" + methods).encode()
        result = parse_python_file("m.py", src)
        chunks = _chunk_file(result, src, max_tokens=1500)
        assert len(chunks) == 1
        assert chunks[0].is_partial is False
        assert chunks[0].symbol_names == ["C"]


class TestOversizedClassSplitting:
    def test_large_class_splits_into_partial_chunks(self) -> None:
        methods = "\n".join(f'    def method_{i}(self):\n        """Padding docstring to inflate size."""\n        return {i}\n' for i in range(40))
        src = ("class Big:\n    \"\"\"A large class.\"\"\"\n\n" + methods).encode()
        result = parse_python_file("big.py", src)
        chunks = _chunk_file(result, src, max_tokens=100)

        assert len(chunks) > 1
        assert all(c.is_partial for c in chunks)
        assert all(c.parent_symbol == "Big" for c in chunks)
        assert sorted(_all_symbol_names(chunks)) == sorted(f"method_{i}" for i in range(40))

    def test_partial_chunks_carry_context_header(self) -> None:
        methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(30))
        src = ("class Big:\n" + methods).encode()
        result = parse_python_file("big.py", src)
        chunks = _chunk_file(result, src, max_tokens=50)
        assert all(c.context_header is not None for c in chunks if c.is_partial)
        assert all("Big" in (c.context_header or "") for c in chunks)


class TestOversizedFunctionFallback:
    def test_giant_function_without_children_uses_raw_split(self) -> None:
        lines = "\n".join(f"    x{i} = {i}" for i in range(200))
        src = f"def giant():\n{lines}\n    return x0\n".encode()
        result = parse_python_file("giant.py", src)
        chunks = _chunk_file(result, src, max_tokens=50)

        assert len(chunks) > 1
        assert all(c.is_partial for c in chunks)
        assert all(c.parent_symbol == "giant" for c in chunks)

    def test_raw_split_has_no_gaps(self) -> None:
        lines = "\n".join(f"    x{i} = {i}" for i in range(100))
        src = f"def giant():\n{lines}\n".encode()
        result = parse_python_file("giant.py", src)
        chunks = sorted(_chunk_file(result, src, max_tokens=50), key=lambda c: c.start_byte)
        for a, b in zip(chunks, chunks[1:]):
            assert a.end_byte == b.start_byte


class TestGoDisjointMethods:
    """Go/Rust methods aren't lexically nested in their struct's span — regression coverage for that geometry."""

    def test_struct_and_disjoint_methods_all_covered(self) -> None:
        src = b"package main\n\ntype Point struct {\n\tX int\n}\n\nfunc (p *Point) Move() {}\n\nfunc (p *Point) String() string {\n\treturn \"\"\n}\n"
        result = parse_go_file("main.go", src)
        chunks = _chunk_file(result, src, max_tokens=1500)
        names = _all_symbol_names(chunks)
        assert "Point" in names
        assert "Move" in names
        assert "String" in names


class TestRustDisjointMethods:
    def test_struct_and_impl_methods_all_covered(self) -> None:
        src = b"pub struct Point {\n    x: i32,\n}\n\nimpl Point {\n    pub fn new() -> Self {\n        Point { x: 0 }\n    }\n}\n"
        result = parse_rust_file("lib.rs", src)
        chunks = _chunk_file(result, src, max_tokens=1500)
        names = _all_symbol_names(chunks)
        assert "Point" in names
        assert "new" in names


class TestChunkIds:
    def test_chunk_id_reflects_line_range(self) -> None:
        src = b"def foo():\n    return 1\n"
        result = parse_python_file("m.py", src)
        chunks = _chunk_file(result, src, max_tokens=1500)
        assert chunks[0].id.startswith("m.py:")
        assert chunks[0].id == f"m.py:{chunks[0].start_line}-{chunks[0].end_line}"
