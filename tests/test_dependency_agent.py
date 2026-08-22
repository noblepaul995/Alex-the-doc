"""Tests for the Dependency Agent (graph/dependency.py + agents/dependency_agent.py)."""

from __future__ import annotations

from pathlib import Path

from agents.dependency_agent import dependency_graph_node
from graph.dependency import DependencyEdge, DependencyGraph
from graph.state import RepositoryState
from parsers.go_parser import parse_go_file
from parsers.python_parser import parse_python_file
from parsers.rust_parser import parse_rust_file
from parsers.ts_parser import parse_javascript_file


def _state_with_files(tmp_path: Path, files: dict[str, bytes]) -> RepositoryState:
    from graph.state import ProjectFile

    state = RepositoryState(repository_path=tmp_path)
    for relative_path, content in files.items():
        full_path = tmp_path / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)
        state.file_metadata[relative_path] = ProjectFile(
            path=full_path, relative_path=relative_path, size_bytes=len(content), language=_lang_for(relative_path)
        )
    return state


def _lang_for(relative_path: str) -> str | None:
    ext = relative_path.rsplit(".", 1)[-1] if "." in relative_path else ""
    return {"py": "python", "js": "javascript", "go": "go", "rs": "rust"}.get(ext)


class TestDependencyEdge:
    def test_is_external_when_no_targets(self) -> None:
        assert DependencyEdge(source="a.py", module="os").is_external is True

    def test_is_external_false_when_resolved(self) -> None:
        assert DependencyEdge(source="a.py", module="b", targets=["b.py"]).is_external is False


class TestDependencyGraph:
    def test_dependencies_and_dependents(self) -> None:
        graph = DependencyGraph(
            edges=[
                DependencyEdge(source="a.py", module="b", targets=["b.py"]),
                DependencyEdge(source="c.py", module="b", targets=["b.py"]),
            ]
        )
        assert graph.dependencies_of("a.py") == ["b.py"]
        assert sorted(graph.dependents_of("b.py")) == ["a.py", "c.py"]

    def test_internal_vs_external_split(self) -> None:
        graph = DependencyGraph(
            edges=[
                DependencyEdge(source="a.py", module="os"),
                DependencyEdge(source="a.py", module="b", targets=["b.py"]),
            ]
        )
        assert len(graph.internal_edges()) == 1
        assert len(graph.external_edges()) == 1

    def test_to_adjacency(self) -> None:
        graph = DependencyGraph(edges=[DependencyEdge(source="a.py", module="b", targets=["b.py"])])
        assert graph.to_adjacency() == {"a.py": ["b.py"]}


class TestPythonResolution:
    def test_absolute_package_import(self, tmp_path: Path) -> None:
        state = _state_with_files(
            tmp_path,
            {
                "pkg/__init__.py": b"",
                "pkg/utils.py": b"",
                "main.py": b"from pkg import utils\n",
            },
        )
        state.parse_results["main.py"] = parse_python_file("main.py", b"from pkg import utils\n")
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.dependencies_of("main.py") == ["pkg/utils.py"]

    def test_relative_sibling_import(self, tmp_path: Path) -> None:
        state = _state_with_files(
            tmp_path,
            {
                "pkg/__init__.py": b"",
                "pkg/a.py": b"",
                "pkg/b.py": b"from . import a\n",
            },
        )
        state.parse_results["pkg/b.py"] = parse_python_file("pkg/b.py", b"from . import a\n")
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.dependencies_of("pkg/b.py") == ["pkg/a.py"]

    def test_stdlib_import_is_external(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, {"main.py": b"import os\n"})
        state.parse_results["main.py"] = parse_python_file("main.py", b"import os\n")
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.edges[0].is_external is True


class TestJavaScriptResolution:
    def test_relative_import_with_extension_lookup(self, tmp_path: Path) -> None:
        state = _state_with_files(
            tmp_path,
            {
                "src/main.js": b'import { helper } from "./utils";\n',
                "src/utils.js": b"",
            },
        )
        state.parse_results["src/main.js"] = parse_javascript_file("src/main.js", b'import { helper } from "./utils";\n')
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.dependencies_of("src/main.js") == ["src/utils.js"]

    def test_bare_specifier_is_external(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, {"main.js": b'import React from "react";\n'})
        state.parse_results["main.js"] = parse_javascript_file("main.js", b'import React from "react";\n')
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.edges[0].is_external is True


class TestGoResolution:
    def test_internal_package_import(self, tmp_path: Path) -> None:
        state = _state_with_files(
            tmp_path,
            {
                "go.mod": b"module example.com/proj\n\ngo 1.22\n",
                "main.go": b'package main\n\nimport "example.com/proj/utils"\n',
                "utils/helper.go": b"package utils\n",
            },
        )
        state.parse_results["main.go"] = parse_go_file("main.go", b'package main\n\nimport "example.com/proj/utils"\n')
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.dependencies_of("main.go") == ["utils/helper.go"]

    def test_stdlib_import_is_external(self, tmp_path: Path) -> None:
        state = _state_with_files(
            tmp_path, {"go.mod": b"module example.com/proj\n", "main.go": b'package main\n\nimport "fmt"\n'}
        )
        state.parse_results["main.go"] = parse_go_file("main.go", b'package main\n\nimport "fmt"\n')
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.edges[0].is_external is True


class TestRustResolution:
    def test_crate_path_import(self, tmp_path: Path) -> None:
        state = _state_with_files(
            tmp_path,
            {
                "src/lib.rs": b"",
                "src/foo.rs": b"use crate::foo;\n",
            },
        )
        state.parse_results["src/foo.rs"] = parse_rust_file("src/foo.rs", b"use crate::foo;\n")
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        # `use crate::foo` from within src/foo.rs resolves to src/foo.rs itself per our module-path rule.
        assert graph.dependencies_of("src/foo.rs") == ["src/foo.rs"]

    def test_external_crate_is_external(self, tmp_path: Path) -> None:
        state = _state_with_files(tmp_path, {"src/lib.rs": b"use serde::Serialize;\n"})
        state.parse_results["src/lib.rs"] = parse_rust_file("src/lib.rs", b"use serde::Serialize;\n")
        result = dependency_graph_node(state)
        graph: DependencyGraph = result["dependency_graph"]
        assert graph.edges[0].is_external is True
