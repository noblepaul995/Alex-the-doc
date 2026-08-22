"""
Python-specific Tree-sitter extraction rules.

This is the reference implementation for the Parser Agent: every other
language extractor (`ts_parser.py`, `go_parser.py`, `rust_parser.py`)
follows the same shape — walk the module-level tree, emit `Import` for
each import statement, emit a `Symbol` for each top-level function and
class (with methods nested as children carrying `parent` set to the
class name), and populate `ParseResult.has_syntax_errors` from
Tree-sitter's own error recovery rather than raising.

Deliberately uses recursive descent over the concrete syntax tree
rather than a single Tree-sitter query: parent/child relationships
(which class a method belongs to, docstring-adjacent-to-definition)
are much simpler to track with an explicit walk than to reconstruct
from a flat capture list.
"""

from __future__ import annotations

from tree_sitter import Node

from parsers.metadata import Import, Parameter, ParseError, ParseResult, Span, Symbol, SymbolKind
from parsers.tree_sitter import find_error_locations, has_error, node_text, parse_source

LANGUAGE = "python"


def parse_python_file(file_path: str, source: bytes) -> ParseResult:
    """Parse a single Python source file into a language-agnostic `ParseResult`."""
    tree = parse_source(LANGUAGE, source)
    root = tree.root_node

    result = ParseResult(file_path=file_path, language=LANGUAGE, has_syntax_errors=has_error(tree))

    for child in root.children:
        _visit_top_level(child, source, result, parent=None)

    if result.has_syntax_errors:
        for location in find_error_locations(tree, source):
            result.errors.append(ParseError(message=f"Syntax error near {location}"))

    return result


def _visit_top_level(node: Node, source: bytes, result: ParseResult, *, parent: str | None) -> None:
    """Dispatch a single statement node to the right extractor, if any."""
    decorators: list[str] = []
    target = node

    if node.type == "decorated_definition":
        decorators = _extract_decorators(node, source)
        definition = node.child_by_field_name("definition")
        if definition is None:
            return
        target = definition

    if target.type == "function_definition":
        symbol = _extract_function(target, source, decorators=decorators, parent=parent)
        result.symbols.append(symbol)
        if parent is not None:
            _attach_child(result, parent, symbol.name)

    elif target.type == "class_definition":
        class_symbol = _extract_class(target, source, decorators=decorators, parent=parent)
        result.symbols.append(class_symbol)
        if parent is not None:
            _attach_child(result, parent, class_symbol.name)

        body = target.child_by_field_name("body")
        if body is not None:
            for member in body.children:
                _visit_top_level(member, source, result, parent=class_symbol.name)

    elif target.type in ("import_statement", "import_from_statement"):
        result.imports.extend(_extract_imports(target, source))


def _attach_child(result: ParseResult, parent_name: str, child_name: str) -> None:
    for symbol in result.symbols:
        if symbol.name == parent_name and symbol.kind == SymbolKind.CLASS:
            symbol.children.append(child_name)
            return


def _extract_decorators(node: Node, source: bytes) -> list[str]:
    return [node_text(child, source) for child in node.children if child.type == "decorator"]


def _extract_function(node: Node, source: bytes, *, decorators: list[str], parent: str | None) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    params_node = node.child_by_field_name("parameters")
    parameters = _extract_parameters(params_node, source) if params_node else []

    return_type_node = node.child_by_field_name("return_type")
    return_type = node_text(return_type_node, source) if return_type_node else None

    body = node.child_by_field_name("body")
    docstring = _extract_docstring(body, source) if body else None

    kind = SymbolKind.METHOD if parent is not None else SymbolKind.FUNCTION

    return Symbol(
        name=name,
        kind=kind,
        language=LANGUAGE,
        span=_span(node),
        signature=_function_signature(name, parameters, return_type),
        docstring=docstring,
        parameters=parameters,
        return_type=return_type,
        decorators=decorators,
        parent=parent,
        is_exported=not name.startswith("_"),
    )


def _extract_class(node: Node, source: bytes, *, decorators: list[str], parent: str | None) -> Symbol:
    name_node = node.child_by_field_name("name")
    name = node_text(name_node, source) if name_node else "<anonymous>"

    body = node.child_by_field_name("body")
    docstring = _extract_docstring(body, source) if body else None

    bases_node = node.child_by_field_name("superclasses")
    bases = node_text(bases_node, source) if bases_node else ""

    return Symbol(
        name=name,
        kind=SymbolKind.CLASS,
        language=LANGUAGE,
        span=_span(node),
        signature=f"class {name}{bases}",
        docstring=docstring,
        decorators=decorators,
        parent=parent,
        is_exported=not name.startswith("_"),
    )


def _extract_parameters(params_node: Node, source: bytes) -> list[Parameter]:
    parameters: list[Parameter] = []
    for child in params_node.children:
        if child.type == "identifier":
            parameters.append(Parameter(name=node_text(child, source)))
        elif child.type == "typed_parameter":
            ident = next((c for c in child.children if c.type == "identifier"), None)
            type_node = child.child_by_field_name("type")
            if ident is not None:
                parameters.append(
                    Parameter(
                        name=node_text(ident, source),
                        type_annotation=node_text(type_node, source) if type_node else None,
                    )
                )
        elif child.type in ("default_parameter", "typed_default_parameter"):
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            value_node = child.child_by_field_name("value")
            if name_node is not None:
                parameters.append(
                    Parameter(
                        name=node_text(name_node, source),
                        type_annotation=node_text(type_node, source) if type_node else None,
                        default_value=node_text(value_node, source) if value_node else None,
                    )
                )
        elif child.type in ("list_splat_pattern", "dictionary_splat_pattern"):
            # *args / **kwargs — keep the sigil, drop the wrapping node type.
            parameters.append(Parameter(name=node_text(child, source)))
    return parameters


def _extract_docstring(body_node: Node, source: bytes) -> str | None:
    if not body_node.children:
        return None
    first = body_node.children[0]
    if first.type != "expression_statement" or not first.children:
        return None
    string_node = first.children[0]
    if string_node.type != "string":
        return None
    content = next((c for c in string_node.children if c.type == "string_content"), None)
    if content is None:
        return None
    return node_text(content, source).strip()


def _extract_imports(node: Node, source: bytes) -> list[Import]:
    span = _span(node)

    if node.type == "import_statement":
        imports: list[Import] = []
        for child in node.children:
            if child.type == "dotted_name":
                imports.append(Import(module=node_text(child, source), span=span))
            elif child.type == "aliased_import":
                dotted = child.child_by_field_name("name")
                alias = child.child_by_field_name("alias")
                if dotted is not None:
                    imports.append(
                        Import(
                            module=node_text(dotted, source),
                            alias=node_text(alias, source) if alias else None,
                            span=span,
                        )
                    )
        return imports

    # import_from_statement
    module_node = node.child_by_field_name("module_name")
    is_relative = module_node is not None and module_node.type == "relative_import"
    module = node_text(module_node, source) if module_node is not None else "."

    module_span = (module_node.start_byte, module_node.end_byte) if module_node is not None else None

    imported_names: list[str] = []
    for child in node.children:
        if child.type == "dotted_name" and (child.start_byte, child.end_byte) != module_span:
            imported_names.append(node_text(child, source))
        elif child.type == "aliased_import":
            dotted = child.child_by_field_name("name")
            if dotted is not None:
                imported_names.append(node_text(dotted, source))
        elif child.type == "wildcard_import":
            imported_names.append("*")

    return [Import(module=module, imported_names=imported_names, is_relative=is_relative, span=span)]


def _function_signature(name: str, parameters: list[Parameter], return_type: str | None) -> str:
    rendered_params = []
    for p in parameters:
        piece = p.name
        if p.type_annotation:
            piece += f": {p.type_annotation}"
        if p.default_value:
            piece += f" = {p.default_value}"
        rendered_params.append(piece)
    signature = f"def {name}({', '.join(rendered_params)})"
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
