"""Tests for the Parser Agent (parsers/*.py + agents/parser.py)."""

from __future__ import annotations

import pytest

from parsers.go_parser import parse_go_file
from parsers.metadata import SymbolKind
from parsers.python_parser import parse_python_file
from parsers.rust_parser import parse_rust_file
from parsers.tree_sitter import UnsupportedLanguageError, get_language, supported_languages
from parsers.ts_parser import parse_javascript_file, parse_typescript_file


def _symbol(result, name: str):
    return next(s for s in result.symbols if s.name == name)


class TestSupportedLanguages:
    def test_python_is_supported(self) -> None:
        assert "python" in supported_languages()

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(UnsupportedLanguageError):
            get_language("cobol")

    def test_grammar_is_cached(self) -> None:
        assert get_language("python") is get_language("python")


class TestImports:
    def test_plain_import(self) -> None:
        result = parse_python_file("m.py", b"import os\n")
        assert result.imports[0].module == "os"
        assert result.imports[0].imported_names == []

    def test_from_import_with_alias(self) -> None:
        result = parse_python_file("m.py", b"from pathlib import Path as P\n")
        imp = result.imports[0]
        assert imp.module == "pathlib"
        assert imp.imported_names == ["Path"]

    def test_multi_name_from_import(self) -> None:
        result = parse_python_file("m.py", b"from typing import List, Dict\n")
        assert result.imports[0].imported_names == ["List", "Dict"]

    def test_relative_import(self) -> None:
        result = parse_python_file("m.py", b"from ..pkg import thing\n")
        imp = result.imports[0]
        assert imp.is_relative is True
        assert imp.imported_names == ["thing"]

    def test_wildcard_import(self) -> None:
        result = parse_python_file("m.py", b"from mod import *\n")
        assert result.imports[0].imported_names == ["*"]


class TestFunctions:
    def test_basic_function(self) -> None:
        result = parse_python_file("m.py", b"def foo(x: int) -> bool:\n    return True\n")
        fn = _symbol(result, "foo")
        assert fn.kind == SymbolKind.FUNCTION
        assert fn.parameters[0].name == "x"
        assert fn.parameters[0].type_annotation == "int"
        assert fn.return_type == "bool"
        assert fn.signature == "def foo(x: int) -> bool"

    def test_docstring_extraction(self) -> None:
        result = parse_python_file("m.py", b'def foo():\n    """Does a thing."""\n    pass\n')
        assert _symbol(result, "foo").docstring == "Does a thing."

    def test_decorators(self) -> None:
        result = parse_python_file("m.py", b"@staticmethod\n@another\ndef foo():\n    pass\n")
        assert _symbol(result, "foo").decorators == ["@staticmethod", "@another"]

    def test_default_and_splat_parameters(self) -> None:
        result = parse_python_file("m.py", b'def foo(y: str = "a", *args, **kwargs):\n    pass\n')
        params = {p.name: p for p in _symbol(result, "foo").parameters}
        assert params["y"].default_value == '"a"'
        assert "*args" in params
        assert "**kwargs" in params

    def test_private_function_not_exported(self) -> None:
        result = parse_python_file("m.py", b"def _hidden():\n    pass\n")
        assert _symbol(result, "_hidden").is_exported is False


class TestClasses:
    def test_class_with_bases_and_docstring(self) -> None:
        result = parse_python_file("m.py", b'class Bar(Base):\n    """A class."""\n    pass\n')
        cls = _symbol(result, "Bar")
        assert cls.kind == SymbolKind.CLASS
        assert cls.signature == "class Bar(Base)"
        assert cls.docstring == "A class."

    def test_methods_are_nested_with_parent(self) -> None:
        src = b"class Bar:\n    def method(self):\n        pass\n"
        result = parse_python_file("m.py", src)
        cls = _symbol(result, "Bar")
        method = _symbol(result, "method")
        assert method.kind == SymbolKind.METHOD
        assert method.parent == "Bar"
        assert "method" in cls.children


class TestErrorRecovery:
    def test_syntax_error_is_flagged_not_raised(self) -> None:
        result = parse_python_file("broken.py", b"def foo(:\n    pass\n")
        assert result.has_syntax_errors is True
        assert result.errors

    def test_valid_file_has_no_errors(self) -> None:
        result = parse_python_file("m.py", b"def foo():\n    pass\n")
        assert result.has_syntax_errors is False
        assert result.errors == []


class TestJavaScript:
    def test_default_and_named_imports(self) -> None:
        src = b'import React from "react";\nimport { useState } from "react";\n'
        result = parse_javascript_file("a.js", src)
        assert result.imports[0].alias == "React"
        assert result.imports[1].imported_names == ["useState"]

    def test_exported_function(self) -> None:
        result = parse_javascript_file("a.js", b"export function foo(x, y = 1) {\n  return x;\n}\n")
        fn = _symbol(result, "foo")
        assert fn.is_exported is True
        assert fn.parameters[1].default_value == "1"

    def test_unexported_function(self) -> None:
        result = parse_javascript_file("a.js", b"function foo() {}\n")
        assert _symbol(result, "foo").is_exported is False

    def test_arrow_function_const(self) -> None:
        result = parse_javascript_file("a.js", b"export const bar = (a, b) => a + b;\n")
        bar = _symbol(result, "bar")
        assert bar.kind == SymbolKind.FUNCTION
        assert bar.is_exported is True

    def test_class_with_methods(self) -> None:
        src = b"class Baz extends Base {\n  method(z) {\n    return z;\n  }\n}\n"
        result = parse_javascript_file("a.js", src)
        cls = _symbol(result, "Baz")
        method = _symbol(result, "method")
        assert method.parent == "Baz"
        assert "method" in cls.children
        assert "Base" in cls.signature

    def test_private_method_not_exported(self) -> None:
        src = b"class Baz {\n  #private() {}\n}\n"
        result = parse_javascript_file("a.js", src)
        assert _symbol(result, "#private").is_exported is False


class TestTypeScript:
    def test_interface(self) -> None:
        result = parse_typescript_file("a.ts", b"interface Foo {\n  x: number;\n}\n")
        iface = _symbol(result, "Foo")
        assert iface.kind == SymbolKind.INTERFACE

    def test_type_alias(self) -> None:
        result = parse_typescript_file("a.ts", b"type Bar = string | number;\n")
        alias = _symbol(result, "Bar")
        assert alias.kind == SymbolKind.TYPE_ALIAS
        assert "string | number" in alias.signature

    def test_typed_function_params_and_return(self) -> None:
        src = b'export function foo(x: number, y: string = "a"): boolean {\n  return true;\n}\n'
        result = parse_typescript_file("a.ts", src)
        fn = _symbol(result, "foo")
        assert fn.parameters[0].type_annotation == "number"
        assert fn.return_type == "boolean"

    def test_optional_parameter(self) -> None:
        result = parse_typescript_file("a.ts", b"function foo(b?: string) {}\n")
        param = _symbol(result, "foo").parameters[0]
        assert param.type_annotation == "string | undefined"

    def test_known_limitation_bare_ampersand_in_jsx_text(self) -> None:
        """
        Documents an upstream tree-sitter-typescript grammar gap: a bare `&`
        in JSX text (vs. the escaped `&amp;`) produces a small ERROR node.
        Symbol extraction around it is unaffected — this is cosmetic, not a
        parse failure — which is what this test actually asserts.
        """
        from parsers.ts_parser import parse_tsx_file

        src = b"export default function Foo() {\n  return <h1>Image & Asset Tools</h1>;\n}\n"
        result = parse_tsx_file("page.tsx", src)
        assert result.has_syntax_errors is True
        assert _symbol(result, "Foo") is not None


class TestGo:
    def test_imports(self) -> None:
        src = b'package main\n\nimport (\n\t"fmt"\n\talias "path/filepath"\n)\n'
        result = parse_go_file("m.go", src)
        modules = {i.module: i.alias for i in result.imports}
        assert modules["fmt"] is None
        assert modules["path/filepath"] == "alias"

    def test_exported_vs_unexported_by_capitalization(self) -> None:
        result = parse_go_file("m.go", b"package main\n\nfunc Foo() {}\nfunc bar() {}\n")
        assert _symbol(result, "Foo").is_exported is True
        assert _symbol(result, "bar").is_exported is False

    def test_method_attached_to_receiver_struct(self) -> None:
        src = b"package main\n\ntype Point struct {\n\tX int\n}\n\nfunc (p *Point) Move() {}\n"
        result = parse_go_file("m.go", src)
        struct = _symbol(result, "Point")
        method = _symbol(result, "Move")
        assert method.parent == "Point"
        assert "Move" in struct.children

    def test_struct_vs_interface(self) -> None:
        src = b"package main\n\ntype S struct{}\ntype I interface{}\n"
        result = parse_go_file("m.go", src)
        assert _symbol(result, "S").kind == SymbolKind.STRUCT
        assert _symbol(result, "I").kind == SymbolKind.INTERFACE


class TestRust:
    def test_use_declarations(self) -> None:
        src = b"use std::fmt;\nuse std::collections::{HashMap, HashSet};\nuse std::io::Result as IoResult;\n"
        result = parse_rust_file("m.rs", src)
        assert result.imports[0].module == "std::fmt"
        assert result.imports[1].imported_names == ["HashMap", "HashSet"]
        assert result.imports[2].alias == "IoResult"

    def test_pub_visibility(self) -> None:
        result = parse_rust_file("m.rs", b"pub fn foo() {}\nfn bar() {}\n")
        assert _symbol(result, "foo").is_exported is True
        assert _symbol(result, "bar").is_exported is False

    def test_impl_methods_attached_to_struct(self) -> None:
        src = b"pub struct Point {\n    x: i32,\n}\n\nimpl Point {\n    pub fn new() -> Self {\n        Point { x: 0 }\n    }\n}\n"
        result = parse_rust_file("m.rs", src)
        struct = _symbol(result, "Point")
        method = _symbol(result, "new")
        assert method.parent == "Point"
        assert "new" in struct.children

    def test_enum_and_trait(self) -> None:
        src = b"pub enum Shape {\n    Circle(f64),\n}\n\npub trait Drawable {\n    fn draw(&self);\n}\n"
        result = parse_rust_file("m.rs", src)
        assert _symbol(result, "Shape").kind == SymbolKind.ENUM
        assert _symbol(result, "Drawable").kind == SymbolKind.INTERFACE
