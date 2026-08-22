"""Tests for the Knowledge Graph Agent (graph/knowledge.py + agents/knowledge_graph.py)."""

from __future__ import annotations

from pathlib import Path

from agents.knowledge_graph import knowledge_graph_node
from graph.chunk import ChunkDocumentation, ChunkDocumentationCollection
from graph.dependency import DependencyEdge, DependencyGraph
from graph.file_doc import FileDocumentation, FileDocumentationCollection
from graph.knowledge import EdgeType, NodeType
from graph.state import RepositoryState
from parsers.python_parser import parse_python_file


def _state_with(files: dict[str, bytes], tmp_path: Path) -> RepositoryState:
    from graph.state import ProjectFile

    state = RepositoryState(repository_path=tmp_path)
    for relative_path, content in files.items():
        full_path = tmp_path / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)
        state.file_metadata[relative_path] = ProjectFile(
            path=full_path, relative_path=relative_path, size_bytes=len(content), language="python"
        )
        state.parse_results[relative_path] = parse_python_file(relative_path, content)
    return state


class TestFileAndSymbolNodes:
    def test_file_node_created_for_every_parsed_file(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"def foo():\n    pass\n"}, tmp_path)
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        assert kg.node("a.py") is not None
        assert kg.node("a.py").node_type == NodeType.FILE

    def test_top_level_function_gets_defines_edge(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"def foo():\n    pass\n"}, tmp_path)
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        defines = kg.edges_by_type(EdgeType.DEFINES)
        assert len(defines) == 1
        assert defines[0].source == "a.py"
        assert defines[0].target == "a.py::foo"

    def test_method_gets_member_of_edge_not_defines(self, tmp_path: Path) -> None:
        src = b"class Bar:\n    def method(self):\n        pass\n"
        state = _state_with({"a.py": src}, tmp_path)
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        member_edges = kg.edges_by_type(EdgeType.MEMBER_OF)
        assert len(member_edges) == 1
        assert member_edges[0].source == "a.py::Bar.method"
        assert member_edges[0].target == "a.py::Bar"

    def test_same_named_methods_across_classes_do_not_collide(self, tmp_path: Path) -> None:
        src = b"class A:\n    def get(self):\n        pass\n\nclass B:\n    def get(self):\n        pass\n"
        state = _state_with({"a.py": src}, tmp_path)
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        symbol_ids = {n.id for n in kg.nodes_by_type(NodeType.SYMBOL)}
        assert "a.py::A.get" in symbol_ids
        assert "a.py::B.get" in symbol_ids
        assert len(symbol_ids) == 4  # A, B, A.get, B.get all distinct


class TestImportEdges:
    def test_internal_dependency_edge_becomes_imports_edge(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"import b\n", "b.py": b"x = 1\n"}, tmp_path)
        state.dependency_graph = DependencyGraph(edges=[DependencyEdge(source="a.py", module="b", targets=["b.py"])])
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        imports = kg.edges_by_type(EdgeType.IMPORTS)
        assert len(imports) == 1
        assert imports[0].source == "a.py"
        assert imports[0].target == "b.py"

    def test_import_to_unparsed_file_is_skipped(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"import missing\n"}, tmp_path)
        state.dependency_graph = DependencyGraph(edges=[DependencyEdge(source="a.py", module="missing", targets=["missing.py"])])
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        assert kg.edges_by_type(EdgeType.IMPORTS) == []  # missing.py was never a real FILE node

    def test_external_dependency_edges_never_become_imports_edges(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"import os\n"}, tmp_path)
        state.dependency_graph = DependencyGraph(edges=[DependencyEdge(source="a.py", module="os")])  # no targets = external
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        assert kg.edges_by_type(EdgeType.IMPORTS) == []


class TestSummaryEnrichment:
    def test_summary_attached_when_file_docs_available(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"def foo():\n    pass\n"}, tmp_path)
        state.file_docs = FileDocumentationCollection(
            docs=[FileDocumentation(file_path="a.py", language="python", summary="Does a thing.", model="fake")]
        )
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        assert kg.node("a.py").summary == "Does a thing."

    def test_no_summary_when_file_docs_absent(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"def foo():\n    pass\n"}, tmp_path)
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        assert kg.node("a.py").summary is None

    def test_failed_file_doc_not_used_as_summary(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"def foo():\n    pass\n"}, tmp_path)
        state.file_docs = FileDocumentationCollection(
            docs=[FileDocumentation(file_path="a.py", language="python", summary="", model="fake", error="boom")]
        )
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        assert kg.node("a.py").summary is None


class TestNeighborsHelper:
    def test_neighbors_combines_edge_types(self, tmp_path: Path) -> None:
        state = _state_with({"a.py": b"import b\ndef foo():\n    pass\n", "b.py": b"x = 1\n"}, tmp_path)
        state.dependency_graph = DependencyGraph(edges=[DependencyEdge(source="a.py", module="b", targets=["b.py"])])
        result = knowledge_graph_node(state)
        kg = result["knowledge_graph"]
        neighbors = kg.neighbors("a.py")
        assert "b.py" in neighbors  # imports edge
        assert "a.py::foo" in neighbors  # defines edge
