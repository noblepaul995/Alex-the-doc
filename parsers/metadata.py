"""
Shared parser metadata models — the common output shape every
language-specific extractor (`python_parser.py`, `ts_parser.py`, ...)
produces, regardless of which Tree-sitter grammar it wraps.

Keeping this shape language-agnostic is what lets `agents/parser.py`
and every stage downstream (Dependency Agent, Chunker, Documentation
Agents) treat a parsed Python module and a parsed Go package
identically. Language-specific nuance still exists — it just lives in
`Symbol.language`, `Symbol.kind`, and free-form `metadata` dicts rather
than in the shape of the model itself.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SymbolKind(StrEnum):
    """Coarse-grained symbol categories, shared across all languages."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    VARIABLE = "variable"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"
    MODULE = "module"


class Span(BaseModel):
    """A byte/line range within a source file, 0-indexed, end-exclusive."""

    start_byte: int
    end_byte: int
    start_line: int
    end_line: int


class Parameter(BaseModel):
    """A single function/method parameter."""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None


class Symbol(BaseModel):
    """
    A single named construct extracted from a source file: a function,
    class, method, struct, etc.

    `signature` is the extractor's best-effort rendering of how the
    symbol would be declared (e.g. `def foo(x: int) -> str`) — useful
    for documentation agents that want the shape without re-deriving it
    from `parameters`/`return_type` themselves.
    """

    name: str
    kind: SymbolKind
    language: str
    span: Span
    signature: str | None = None
    docstring: str | None = None
    parameters: list[Parameter] = Field(default_factory=list)
    return_type: str | None = None
    decorators: list[str] = Field(default_factory=list)
    parent: str | None = None
    """Name of the enclosing symbol, e.g. the class a method belongs to."""
    children: list[str] = Field(default_factory=list)
    """Names of directly nested symbols (methods within a class, etc.)."""
    is_exported: bool = True
    """Best-effort public/private inference (leading underscore, `export`, casing, ...)."""


class Import(BaseModel):
    """A single import/require/use statement, not yet resolved to a file."""

    module: str
    """The raw module path/name as written, e.g. `os.path`, `./utils`, `crate::foo`."""
    imported_names: list[str] = Field(default_factory=list)
    """Specific names imported, e.g. `["Path"]` for `from pathlib import Path`. Empty for whole-module imports."""
    alias: str | None = None
    is_relative: bool = False
    span: Span | None = None


class ParseError(BaseModel):
    """A recoverable error encountered while parsing a single file."""

    message: str
    span: Span | None = None


class ParseResult(BaseModel):
    """
    The complete output of parsing a single source file: every symbol
    and import found, plus enough metadata for the caller to judge
    parse quality without re-inspecting the raw tree.
    """

    file_path: str
    language: str
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[Import] = Field(default_factory=list)
    has_syntax_errors: bool = False
    """True if Tree-sitter's error-recovery kicked in anywhere in the tree."""
    errors: list[ParseError] = Field(default_factory=list)
