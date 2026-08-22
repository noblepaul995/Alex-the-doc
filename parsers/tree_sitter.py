"""
Tree-sitter engine wrapper — grammar loading, parsing, and query
execution shared by every language-specific extractor.

Architecture note (verified against the actually-installed versions
before writing this file, not assumed from memory/tutorials):

`tree-sitter-languages` — the bundled-grammar package many tutorials
still reference — has been unmaintained since early 2024 and hard-pins
`tree-sitter==0.21.3`, which predates the current query API. This
project instead uses the per-language grammar packages
(`tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`,
`tree-sitter-go`, `tree-sitter-rust`), each maintained independently and
compatible with current `tree-sitter` (>=0.24).

That current API differs from older `tree-sitter` in a way that matters
for this file: `Language.query()` no longer exists. Queries are now
constructed as standalone `Query(language, source)` objects and run
through a `QueryCursor(query)`, which returns `dict[str, list[Node]]`
(capture name -> matching nodes) rather than the old flat list of
`(node, capture_name)` tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree

from utils.logger import get_logger

log = get_logger(__name__)


class UnsupportedLanguageError(ValueError):
    """Raised when no Tree-sitter grammar is registered for a language."""


@dataclass(frozen=True)
class GrammarSpec:
    """How to obtain a `Language` object for one canonical language name."""

    module_name: str
    loader_attr: str = "language"
    """Name of the zero-arg function on the grammar module that returns the raw language pointer."""


# Canonical language name (matches `scanner/language.py`) -> where to get its grammar.
# Extend this map as new grammar packages are added to pyproject.toml.
_GRAMMAR_REGISTRY: dict[str, GrammarSpec] = {
    "python": GrammarSpec("tree_sitter_python"),
    "javascript": GrammarSpec("tree_sitter_javascript"),
    "jsx": GrammarSpec("tree_sitter_javascript"),
    "typescript": GrammarSpec("tree_sitter_typescript", "language_typescript"),
    "tsx": GrammarSpec("tree_sitter_typescript", "language_tsx"),
    "go": GrammarSpec("tree_sitter_go"),
    "rust": GrammarSpec("tree_sitter_rust"),
}


def supported_languages() -> frozenset[str]:
    """Canonical language names this Parser Agent can currently parse."""
    return frozenset(_GRAMMAR_REGISTRY)


@lru_cache(maxsize=None)
def get_language(language: str) -> Language:
    """
    Return the (cached) `Language` object for a canonical language name.

    Loading a grammar module and constructing its `Language` wrapper is
    the expensive part; `Parser` objects are cheap, so only the
    `Language` lookup is memoized here.
    """
    spec = _GRAMMAR_REGISTRY.get(language)
    if spec is None:
        raise UnsupportedLanguageError(
            f"No Tree-sitter grammar registered for language {language!r}. "
            f"Supported: {sorted(_GRAMMAR_REGISTRY)}"
        )

    import importlib

    module = importlib.import_module(spec.module_name)
    loader = getattr(module, spec.loader_attr)
    return Language(loader())


def get_parser(language: str) -> Parser:
    """Construct a fresh `Parser` bound to `language`'s grammar."""
    return Parser(get_language(language))


def parse_source(language: str, source: bytes) -> Tree:
    """Parse raw source bytes for `language` into a Tree-sitter `Tree`."""
    return get_parser(language).parse(source)


def run_query(language: str, tree: Tree, query_source: str) -> dict[str, list[Node]]:
    """
    Run a Tree-sitter query string against `tree` and return captures
    grouped by capture name, e.g. `{"func.name": [Node, Node, ...]}`.

    This is the one place `QueryCursor` is used, so every extractor
    goes through the same (version-verified) call shape.
    """
    query = Query(get_language(language), query_source)
    cursor = QueryCursor(query)
    return cursor.captures(tree.root_node)


def node_text(node: Node, source: bytes) -> str:
    """Decode the source slice a node spans, as UTF-8 (replacing invalid bytes)."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def has_error(tree: Tree) -> bool:
    """
    True if Tree-sitter's error-recovery produced any ERROR/MISSING node,
    i.e. the source wasn't fully well-formed per the grammar.
    """
    return tree.root_node.has_error


def find_error_locations(tree: Tree, source: bytes, *, limit: int = 3) -> list[str]:
    """
    Find up to `limit` ERROR/MISSING nodes in `tree` and describe each as
    `line <N>: <the offending source line, trimmed>`, for diagnostics.

    Walks the whole tree rather than stopping at the first error, since
    Tree-sitter's error recovery can resynchronize and produce several
    independent error nodes in one file — seeing all of them (up to
    `limit`) is much faster to debug from than just the first.
    """
    locations: list[str] = []
    stack: list[Node] = [tree.root_node]

    while stack and len(locations) < limit:
        node = stack.pop()
        if node.type in ("ERROR", "MISSING") or node.is_missing:
            line_no = node.start_point[0] + 1
            line_start = source.rfind(b"\n", 0, node.start_byte) + 1
            line_end = source.find(b"\n", node.start_byte)
            if line_end == -1:
                line_end = len(source)
            snippet = source[line_start:line_end].decode("utf-8", errors="replace").strip()
            locations.append(f"line {line_no}: {snippet[:120]}")
            continue  # don't descend into an already-flagged error subtree
        stack.extend(reversed(node.children))

    return locations
