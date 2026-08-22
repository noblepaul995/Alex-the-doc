"""
TypeScript/JavaScript-specific Tree-sitter extraction rules.

Follows the shape established by `python_parser.py`: recursive descent
over the concrete syntax tree, emitting `Import`/`Symbol` in the shared
`parsers.metadata` shape. One module covers `javascript`, `jsx`,
`typescript`, and `tsx` since the JS grammar is a strict subset of the
constructs handled here (TS-only nodes like `interface_declaration`,
`type_alias_declaration`, and typed parameters simply never appear in
plain JS/JSX source, so the same walker handles both without branching
on language).

Notable differences from the Python extractor, driven by the grammar
(verified against installed `tree-sitter-javascript`/`-typescript`,
not assumed):
    - Declarations are frequently wrapped in `export_statement`
      (and `export_statement` + `default` for default exports) rather
      than carrying an "exported" flag on the declaration itself.
    - Arrow functions assigned via `const foo = (...) => ...` are a
      `lexical_declaration` wrapping a `variable_declarator`, not a
      `function_declaration` — both are extracted as FUNCTION symbols.
    - TS parameters are `required_parameter` / `optional_parameter`
      nodes (vs. Python's `identifier` / `typed_parameter`), and may
      themselves wrap a `rest_pattern` for `...rest` args.
"""

from __future__ import annotations

from tree_sitter import Node

from parsers.metadata import Import, Parameter, ParseError, ParseResult, Span, Symbol, SymbolKind
from parsers.tree_sitter import find_error_locations, has_error, node_text, parse_source

_FUNCTION_LIKE = {"function_declaration", "generator_function_declaration"}
_CLASS_LIKE = {"class_declaration"}


def _make_parser(language: str):
    def parse_file(file_path: str, source: bytes) -> ParseResult:
        tree = parse_source(language, source)
        root = tree.root_node

        result = ParseResult(file_path=file_path, language=language, has_syntax_errors=has_error(tree))

        for child in root.children:
            _visit_top_level(child, source, result, language=language, parent=None, is_exported=None)

        if result.has_syntax_errors:
            for location in find_error_locations(tree, source):
                result.errors.append(ParseError(message=f"Syntax error near {location}"))

        return result

    return parse_file


def _visit_top_level(node: Node, source: bytes, result: ParseResult, *, language: str, parent: str | None, is_exported: bool | None) -> None:
    """Dispatch a single statement node, unwrapping `export_statement` first."""
    exported = is_exported
    target = node

    if node.type == "export_statement":
        exported = True
        inner = next((c for c in node.children if c.type not in ("export", "default")), None)
        if inner is None:
            return
        target = inner

    if target.type in _FUNCTION_LIKE:
        symbol = _extract_function(target, source, language=language, parent=parent, is_exported=exported)
        result.symbols.append(symbol)
        _attach_child(result, parent, symbol.name)

    elif target.type in _CLASS_LIKE:
        class_symbol = _extract_class(target, source, language=language, parent=parent, is_exported=exported)
        result.symbols.append(class_symbol)
        _attach_child(result, parent, class_symbol.name)

        body = target.child_by_field_name("body")
        if body is not None:
            for member in body.children:
                if member.type == "method_definition":
                    method = _extract_method(member, source, language=language, parent=class_symbol.name)
                    result.symbols.append(method)
                    class_symbol.children.append(method.name)

    elif target.type == "interface_declaration":
        result.symbols.append(_extract_interface(target, source, language=language, parent=parent, is_exported=exported))

    elif target.type == "type_alias_declaration":
        result.symbols.append(_extract_type_alias(target, source, language=language, parent=parent, is_exported=exported))

    elif target.type == "lexical_declaration":
        for declarator in (c for c in target.children if c.type == "variable_declarator"):
            symbol = _extract_arrow_function(declarator, source, language=language, parent=parent, is_exported=exported)
            if symbol is not None:
                result.symbols.append(symbol)
                _attach_child(result, parent, symbol.name)

    elif target.type == "import_statement":
        imp = _extract_import(target, source)
        if imp is not None:
            result.imports.append(imp)


def _attach_child(result: ParseResult, parent_name: str | None, child_name: str) -> None:
    if parent_name is None:
        return
    for symbol in result.symbols:
        if symbol.name == parent_name and symbol.kind == SymbolKind.CLASS:
            symbol.children.append(child_name)
            return


def _extract_function(node: Node, source: bytes, *, language: str, parent: str | None, is_exported: bool | None) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    params_node = node.child_by_field_name("parameters")
    parameters = _extract_parameters(params_node, source) if params_node else []

    return_type_node = node.child_by_field_name("return_type")
    return_type = _clean_type_annotation(node_text(return_type_node, source)) if return_type_node else None

    return Symbol(
        name=name,
        kind=SymbolKind.FUNCTION,
        language=language,
        span=_span(node),
        signature=_function_signature(name, parameters, return_type),
        parameters=parameters,
        return_type=return_type,
        parent=parent,
        is_exported=bool(is_exported),
    )


def _extract_arrow_function(declarator: Node, source: bytes, *, language: str, parent: str | None, is_exported: bool | None) -> Symbol | None:
    value = declarator.child_by_field_name("value")
    if value is None or value.type not in ("arrow_function", "function_expression"):
        return None

    name_node = declarator.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    params_node = value.child_by_field_name("parameters")
    parameters = _extract_parameters(params_node, source) if params_node else []

    return_type_node = value.child_by_field_name("return_type")
    return_type = _clean_type_annotation(node_text(return_type_node, source)) if return_type_node else None

    return Symbol(
        name=name,
        kind=SymbolKind.FUNCTION,
        language=language,
        span=_span(declarator),
        signature=_function_signature(name, parameters, return_type, arrow=True),
        parameters=parameters,
        return_type=return_type,
        parent=parent,
        is_exported=bool(is_exported),
    )


def _extract_method(node: Node, source: bytes, *, language: str, parent: str) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    params_node = node.child_by_field_name("parameters")
    parameters = _extract_parameters(params_node, source) if params_node else []

    return_type_node = node.child_by_field_name("return_type")
    return_type = _clean_type_annotation(node_text(return_type_node, source)) if return_type_node else None

    return Symbol(
        name=name,
        kind=SymbolKind.METHOD,
        language=language,
        span=_span(node),
        signature=_function_signature(name, parameters, return_type),
        parameters=parameters,
        return_type=return_type,
        parent=parent,
        is_exported=not name.startswith("#"),
    )


def _extract_class(node: Node, source: bytes, *, language: str, parent: str | None, is_exported: bool | None) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    heritage_node = next((c for c in node.children if c.type == "class_heritage"), None)
    heritage = f" {node_text(heritage_node, source)}" if heritage_node else ""

    return Symbol(
        name=name,
        kind=SymbolKind.CLASS,
        language=language,
        span=_span(node),
        signature=f"class {name}{heritage}",
        parent=parent,
        is_exported=bool(is_exported),
    )


def _extract_interface(node: Node, source: bytes, *, language: str, parent: str | None, is_exported: bool | None) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    return Symbol(
        name=name,
        kind=SymbolKind.INTERFACE,
        language=language,
        span=_span(node),
        signature=f"interface {name}",
        parent=parent,
        is_exported=bool(is_exported),
    )


def _extract_type_alias(node: Node, source: bytes, *, language: str, parent: str | None, is_exported: bool | None) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"
    value_node = node.child_by_field_name("value")
    value = node_text(value_node, source) if value_node else ""

    return Symbol(
        name=name,
        kind=SymbolKind.TYPE_ALIAS,
        language=language,
        span=_span(node),
        signature=f"type {name} = {value}",
        parent=parent,
        is_exported=bool(is_exported),
    )


def _extract_parameters(params_node: Node, source: bytes) -> list[Parameter]:
    parameters: list[Parameter] = []
    for child in params_node.children:
        if child.type in ("(", ")", ","):
            continue

        if child.type == "identifier":
            parameters.append(Parameter(name=node_text(child, source)))
            continue

        if child.type == "assignment_pattern":
            left = child.child_by_field_name("left")
            right = child.child_by_field_name("right")
            if left is not None:
                parameters.append(
                    Parameter(
                        name=node_text(left, source),
                        default_value=node_text(right, source) if right else None,
                    )
                )
            continue

        if child.type in ("required_parameter", "optional_parameter"):
            pattern = next((c for c in child.children if c.type in ("identifier", "rest_pattern")), None)
            if pattern is None:
                continue
            name = node_text(pattern, source)
            type_node = child.child_by_field_name("type")
            value_node = child.child_by_field_name("value")
            type_annotation = _clean_type_annotation(node_text(type_node, source)) if type_node else None
            if child.type == "optional_parameter" and type_annotation:
                type_annotation += " | undefined"
            parameters.append(
                Parameter(
                    name=name,
                    type_annotation=type_annotation,
                    default_value=node_text(value_node, source) if value_node else None,
                )
            )
            continue

        if child.type == "rest_pattern":
            parameters.append(Parameter(name=node_text(child, source)))

    return parameters


def _extract_import(node: Node, source: bytes) -> Import | None:
    source_node = node.child_by_field_name("source")
    if source_node is None:
        return None
    module = node_text(source_node, source).strip("\"'")
    is_relative = module.startswith(".")

    clause = next((c for c in node.children if c.type == "import_clause"), None)
    imported_names: list[str] = []
    alias: str | None = None

    if clause is not None:
        for part in clause.children:
            if part.type == "identifier":
                # Default import — represent as the alias, module stays whole-import.
                alias = node_text(part, source)
            elif part.type == "named_imports":
                for spec in (c for c in part.children if c.type == "import_specifier"):
                    idents = [c for c in spec.children if c.type == "identifier"]
                    if idents:
                        imported_names.append(node_text(idents[0], source))
            elif part.type == "namespace_import":
                ident = next((c for c in part.children if c.type == "identifier"), None)
                if ident is not None:
                    alias = node_text(ident, source)
                    imported_names.append("*")

    return Import(module=module, imported_names=imported_names, alias=alias, is_relative=is_relative, span=_span(node))


def _function_signature(name: str, parameters: list[Parameter], return_type: str | None, *, arrow: bool = False) -> str:
    rendered = []
    for p in parameters:
        piece = p.name
        if p.type_annotation:
            piece += f": {p.type_annotation}"
        if p.default_value:
            piece += f" = {p.default_value}"
        rendered.append(piece)
    params = f"({', '.join(rendered)})"
    if arrow:
        signature = f"const {name} = {params}"
    else:
        signature = f"function {name}{params}"
    if return_type:
        signature += f": {return_type}"
    return signature


def _clean_type_annotation(raw: str) -> str:
    return raw.removeprefix(":").strip()


def _span(node: Node) -> Span:
    return Span(
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_line=node.start_point[0],
        end_line=node.end_point[0],
    )


parse_javascript_file = _make_parser("javascript")
parse_jsx_file = _make_parser("jsx")
parse_typescript_file = _make_parser("typescript")
parse_tsx_file = _make_parser("tsx")
