"""
Go-specific Tree-sitter extraction rules.

Follows the shape established by `python_parser.py`. Go has no
class/method nesting like Python or JS — methods are free functions
with a receiver parameter (`func (r *Receiver) Method(...)`) — so
"parent" here is derived from the receiver's type name rather than
lexical nesting, and structs/interfaces never contain their methods as
child nodes the way a Python/JS class body does. `_attach_child` is
therefore a post-pass keyed on receiver type name rather than a walk
inside a body, unlike the other two extractors.

Exported/unexported is purely capitalization-driven in Go (verified
against the language spec, not the grammar): an identifier is exported
iff its first rune is uppercase — no keyword or decorator marks it.
"""

from __future__ import annotations

from tree_sitter import Node

from parsers.metadata import Import, Parameter, ParseError, ParseResult, Span, Symbol, SymbolKind
from parsers.tree_sitter import find_error_locations, has_error, node_text, parse_source

LANGUAGE = "go"


def parse_go_file(file_path: str, source: bytes) -> ParseResult:
    """Parse a single Go source file into a language-agnostic `ParseResult`."""
    tree = parse_source(LANGUAGE, source)
    root = tree.root_node

    result = ParseResult(file_path=file_path, language=LANGUAGE, has_syntax_errors=has_error(tree))
    receiver_children: dict[str, list[str]] = {}

    for child in root.children:
        _visit_top_level(child, source, result, receiver_children)

    for struct_name, children in receiver_children.items():
        for symbol in result.symbols:
            if symbol.name == struct_name and symbol.kind in (SymbolKind.STRUCT, SymbolKind.TYPE_ALIAS):
                symbol.children.extend(children)

    if result.has_syntax_errors:
        for location in find_error_locations(tree, source):
            result.errors.append(ParseError(message=f"Syntax error near {location}"))

    return result


def _visit_top_level(node: Node, source: bytes, result: ParseResult, receiver_children: dict[str, list[str]]) -> None:
    if node.type == "function_declaration":
        result.symbols.append(_extract_function(node, source))

    elif node.type == "method_declaration":
        symbol, receiver_type = _extract_method(node, source)
        result.symbols.append(symbol)
        if receiver_type:
            receiver_children.setdefault(receiver_type, []).append(symbol.name)

    elif node.type == "type_declaration":
        for spec in (c for c in node.children if c.type == "type_spec"):
            result.symbols.append(_extract_type_spec(spec, source))

    elif node.type == "const_declaration":
        result.symbols.extend(_extract_var_or_const(node, source, kind=SymbolKind.CONSTANT))

    elif node.type == "var_declaration":
        result.symbols.extend(_extract_var_or_const(node, source, kind=SymbolKind.VARIABLE))

    elif node.type == "import_declaration":
        result.imports.extend(_extract_imports(node, source))


def _extract_function(node: Node, source: bytes) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    params_node = node.child_by_field_name("parameters")
    parameters = _extract_parameters(params_node, source) if params_node else []

    result_node = node.child_by_field_name("result")
    return_type = node_text(result_node, source) if result_node else None

    return Symbol(
        name=name,
        kind=SymbolKind.FUNCTION,
        language=LANGUAGE,
        span=_span(node),
        signature=_signature("func", name, parameters, return_type),
        parameters=parameters,
        return_type=return_type,
        is_exported=_is_exported(name),
    )


def _extract_method(node: Node, source: bytes) -> tuple[Symbol, str | None]:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    receiver_node = node.child_by_field_name("receiver")
    receiver_type = _receiver_type_name(receiver_node, source) if receiver_node else None

    params_node = node.child_by_field_name("parameters")
    parameters = _extract_parameters(params_node, source) if params_node else []

    result_node = node.child_by_field_name("result")
    return_type = node_text(result_node, source) if result_node else None

    receiver_text = node_text(receiver_node, source) if receiver_node else ""
    symbol = Symbol(
        name=name,
        kind=SymbolKind.METHOD,
        language=LANGUAGE,
        span=_span(node),
        signature=f"func {receiver_text} {name}({_params_str(parameters)})" + (f" {return_type}" if return_type else ""),
        parameters=parameters,
        return_type=return_type,
        parent=receiver_type,
        is_exported=_is_exported(name),
    )
    return symbol, receiver_type


def _receiver_type_name(receiver_node: Node, source: bytes) -> str | None:
    decl = next((c for c in receiver_node.children if c.type == "parameter_declaration"), None)
    if decl is None:
        return None
    type_node = decl.child_by_field_name("type")
    if type_node is None:
        return None
    if type_node.type == "pointer_type":
        inner = next((c for c in type_node.children if c.type == "type_identifier"), None)
        return node_text(inner, source) if inner else None
    if type_node.type == "type_identifier":
        return node_text(type_node, source)
    return None


def _extract_type_spec(spec: Node, source: bytes) -> Symbol:
    name_node = spec.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"
    type_node = spec.child_by_field_name("type")

    if type_node is not None and type_node.type == "interface_type":
        kind = SymbolKind.INTERFACE
        signature = f"type {name} interface"
    elif type_node is not None and type_node.type == "struct_type":
        kind = SymbolKind.STRUCT
        signature = f"type {name} struct"
    else:
        kind = SymbolKind.TYPE_ALIAS
        underlying = node_text(type_node, source) if type_node else ""
        signature = f"type {name} = {underlying}"

    return Symbol(
        name=name,
        kind=kind,
        language=LANGUAGE,
        span=_span(spec),
        signature=signature,
        is_exported=_is_exported(name),
    )


def _extract_var_or_const(node: Node, source: bytes, *, kind: SymbolKind) -> list[Symbol]:
    symbols: list[Symbol] = []
    specs = [c for c in node.children if c.type in ("const_spec", "var_spec")]
    if not specs:
        specs = [c for c in node.children if c.type == "const_spec_list" or c.type == "var_spec_list"]
        specs = [s for spec_list in specs for s in spec_list.children if s.type in ("const_spec", "var_spec")]

    for spec in specs:
        name_node = next((c for c in spec.children if c.type == "identifier"), None)
        if name_node is None:
            continue
        name = node_text(name_node, source)
        symbols.append(
            Symbol(
                name=name,
                kind=kind,
                language=LANGUAGE,
                span=_span(spec),
                signature=node_text(spec, source),
                is_exported=_is_exported(name),
            )
        )
    return symbols


def _extract_parameters(params_node: Node, source: bytes) -> list[Parameter]:
    parameters: list[Parameter] = []
    for decl in (c for c in params_node.children if c.type == "parameter_declaration"):
        name_node = decl.child_by_field_name("name")
        type_node = decl.child_by_field_name("type")
        parameters.append(
            Parameter(
                name=node_text(name_node, source) if name_node else "_",
                type_annotation=node_text(type_node, source) if type_node else None,
            )
        )
    return parameters


def _extract_imports(node: Node, source: bytes) -> list[Import]:
    specs = [c for c in node.children if c.type == "import_spec"]
    spec_list = next((c for c in node.children if c.type == "import_spec_list"), None)
    if spec_list is not None:
        specs = [c for c in spec_list.children if c.type == "import_spec"]

    imports: list[Import] = []
    for spec in specs:
        string_node = next((c for c in spec.children if "string_literal" in c.type), None)
        if string_node is None:
            continue
        content = next((c for c in string_node.children if "content" in c.type), None)
        module = node_text(content, source) if content is not None else node_text(string_node, source).strip('"')

        alias_node = next((c for c in spec.children if c.type == "package_identifier"), None)
        alias = node_text(alias_node, source) if alias_node else None

        imports.append(Import(module=module, alias=alias, span=_span(spec)))
    return imports


def _is_exported(name: str) -> bool:
    return bool(name) and name[0].isupper()


def _params_str(parameters: list[Parameter]) -> str:
    return ", ".join(f"{p.name} {p.type_annotation}" if p.type_annotation else p.name for p in parameters)


def _signature(keyword: str, name: str, parameters: list[Parameter], return_type: str | None) -> str:
    signature = f"{keyword} {name}({_params_str(parameters)})"
    if return_type:
        signature += f" {return_type}"
    return signature


def _span(node: Node) -> Span:
    return Span(
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_line=node.start_point[0],
        end_line=node.end_point[0],
    )
