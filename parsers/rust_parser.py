"""
Rust-specific Tree-sitter extraction rules.

Follows the shape established by `python_parser.py`, with one
Rust-specific wrinkle: methods live inside `impl_item` blocks rather
than directly inside `struct_item`/`enum_item`, so — like Go's
receiver-based methods, but structurally rather than positionally —
`parent` is set from the `impl` block's target type name (`impl Point
{ ... }` -> methods get `parent="Point"`), and struct/enum symbols get
their `children` back-filled from every `impl` block matching their
name, including multiple `impl` blocks for the same type.

Visibility (`is_exported`) is read directly from the grammar's
`visibility_modifier` node (`pub`, `pub(crate)`, etc.) rather than
inferred from naming, since Rust — unlike Go — has an explicit keyword
for it.
"""

from __future__ import annotations

from tree_sitter import Node

from parsers.metadata import Import, Parameter, ParseError, ParseResult, Span, Symbol, SymbolKind
from parsers.tree_sitter import find_error_locations, has_error, node_text, parse_source

LANGUAGE = "rust"


def parse_rust_file(file_path: str, source: bytes) -> ParseResult:
    """Parse a single Rust source file into a language-agnostic `ParseResult`."""
    tree = parse_source(LANGUAGE, source)
    root = tree.root_node

    result = ParseResult(file_path=file_path, language=LANGUAGE, has_syntax_errors=has_error(tree))
    impl_children: dict[str, list[str]] = {}

    for child in root.children:
        _visit_top_level(child, source, result, impl_children)

    for type_name, children in impl_children.items():
        for symbol in result.symbols:
            if symbol.name == type_name and symbol.kind in (SymbolKind.STRUCT, SymbolKind.ENUM):
                symbol.children.extend(children)

    if result.has_syntax_errors:
        for location in find_error_locations(tree, source):
            result.errors.append(ParseError(message=f"Syntax error near {location}"))

    return result


def _visit_top_level(node: Node, source: bytes, result: ParseResult, impl_children: dict[str, list[str]]) -> None:
    if node.type == "function_item":
        result.symbols.append(_extract_function(node, source, parent=None))

    elif node.type == "struct_item":
        result.symbols.append(_extract_struct(node, source))

    elif node.type == "enum_item":
        result.symbols.append(_extract_enum(node, source))

    elif node.type == "trait_item":
        result.symbols.append(_extract_trait(node, source))

    elif node.type == "impl_item":
        _visit_impl(node, source, result, impl_children)

    elif node.type == "use_declaration":
        result.imports.extend(_extract_imports(node, source))


def _visit_impl(node: Node, source: bytes, result: ParseResult, impl_children: dict[str, list[str]]) -> None:
    type_node = node.child_by_field_name("type")
    type_name = _base_type_name(type_node, source) if type_node is not None else None

    body = node.child_by_field_name("body")
    if body is None or type_name is None:
        return

    for member in body.children:
        if member.type == "function_item":
            method = _extract_function(member, source, parent=type_name)
            result.symbols.append(method)
            impl_children.setdefault(type_name, []).append(method.name)


def _base_type_name(node: Node, source: bytes) -> str | None:
    if node.type == "type_identifier":
        return node_text(node, source)
    if node.type == "generic_type":
        inner = node.child_by_field_name("type")
        return _base_type_name(inner, source) if inner is not None else None
    return node_text(node, source)


def _extract_function(node: Node, source: bytes, *, parent: str | None) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    params_node = node.child_by_field_name("parameters")
    parameters = _extract_parameters(params_node, source) if params_node else []

    return_type_node = node.child_by_field_name("return_type")
    return_type = node_text(return_type_node, source) if return_type_node else None

    kind = SymbolKind.METHOD if parent is not None else SymbolKind.FUNCTION

    return Symbol(
        name=name,
        kind=kind,
        language=LANGUAGE,
        span=_span(node),
        signature=_function_signature(name, parameters, return_type),
        parameters=parameters,
        return_type=return_type,
        parent=parent,
        is_exported=_is_public(node),
    )


def _extract_struct(node: Node, source: bytes) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"
    return Symbol(
        name=name,
        kind=SymbolKind.STRUCT,
        language=LANGUAGE,
        span=_span(node),
        signature=f"struct {name}",
        is_exported=_is_public(node),
    )


def _extract_enum(node: Node, source: bytes) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"
    return Symbol(
        name=name,
        kind=SymbolKind.ENUM,
        language=LANGUAGE,
        span=_span(node),
        signature=f"enum {name}",
        is_exported=_is_public(node),
    )


def _extract_trait(node: Node, source: bytes) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"
    return Symbol(
        name=name,
        kind=SymbolKind.INTERFACE,
        language=LANGUAGE,
        span=_span(node),
        signature=f"trait {name}",
        is_exported=_is_public(node),
    )


def _extract_parameters(params_node: Node, source: bytes) -> list[Parameter]:
    parameters: list[Parameter] = []
    for child in params_node.children:
        if child.type == "parameter":
            name_node = child.child_by_field_name("pattern")
            type_node = child.child_by_field_name("type")
            if name_node is not None:
                parameters.append(
                    Parameter(
                        name=node_text(name_node, source),
                        type_annotation=node_text(type_node, source) if type_node else None,
                    )
                )
        elif child.type == "self_parameter":
            parameters.append(Parameter(name=node_text(child, source)))
    return parameters


def _extract_imports(node: Node, source: bytes) -> list[Import]:
    imports: list[Import] = []
    span = _span(node)

    # `use a::b::{c, d};` / `use a::b::c;` / `use a::b::c as d;` / `use a;`
    target = next((c for c in node.children if c.type not in ("use", ";", "visibility_modifier")), None)
    if target is None:
        return imports

    if target.type == "use_as_clause":
        path_node = target.child_by_field_name("path")
        alias_node = target.child_by_field_name("alias")
        module = node_text(path_node, source) if path_node else node_text(target, source)
        imports.append(Import(module=module, alias=node_text(alias_node, source) if alias_node else None, span=span))
        return imports

    if target.type == "scoped_use_list":
        base = next((c for c in target.children if c.type in ("scoped_identifier", "identifier")), None)
        base_path = node_text(base, source) if base else ""
        use_list = next((c for c in target.children if c.type == "use_list"), None)
        names = [node_text(c, source) for c in use_list.children if c.type == "identifier"] if use_list else []
        imports.append(Import(module=base_path, imported_names=names, span=span))
        return imports

    # plain `scoped_identifier` or `identifier`
    module = node_text(target, source)
    imports.append(Import(module=module, span=span))
    return imports


def _is_public(node: Node) -> bool:
    return any(c.type == "visibility_modifier" for c in node.children)


def _function_signature(name: str, parameters: list[Parameter], return_type: str | None) -> str:
    rendered = []
    for p in parameters:
        piece = p.name
        if p.type_annotation:
            piece += f": {p.type_annotation}"
        rendered.append(piece)
    signature = f"fn {name}({', '.join(rendered)})"
    if return_type:
        signature += f" -> {return_type}"
    return signature


def _span(node: Node) -> Span:
    return Span(
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_line=node.start_point[0],
        end_line=node.end_point[0],
    )
