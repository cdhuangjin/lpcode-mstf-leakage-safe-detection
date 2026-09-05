"""Deterministic, test-only surface-style attacks for the T4 protocol.

The transforms in this module deliberately prefer a no-op to an unsafe edit.
They never call a formatter or repair tool.  Parser provenance is recorded on
both sides of every transform, and a clean parse which regresses is surfaced as
an explicit attack failure rather than silently entering the evaluation set.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import token
import tokenize

from .features_enhanced import (
    _c_prefixed_quoted_end,
    _cpp_raw_string_end,
    _java_text_block_end,
    _quoted_end,
    _tree_sitter_parser,
    analyze_enhanced,
)


LANGUAGES = ("c", "cpp", "java", "py")
ATTACKS = (
    "comment_removal",
    "identifier_rename",
    "format_normalization",
    "comment_injection",
    "combined",
)
ATTACK_VERSION = "t4-style-attacks-v1"
INJECTED_COMMENTS = {
    "c": "/* deterministic T4 style probe */",
    "cpp": "/* deterministic T4 style probe */",
    "java": "/* deterministic T4 style probe */",
    "py": "# deterministic T4 style probe",
}


@dataclass(frozen=True)
class AttackResult:
    code: str
    attack: str
    language: str
    input_sha256: str
    output_sha256: str
    changed: bool
    transform_count: int
    parse_ok_before: bool
    parse_ok_after: bool
    backend_before: str
    backend_after: str
    failure_reason: str | None


Transform = Callable[[str, str], tuple[str, int]]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attack_source_sha256() -> str:
    """Return the hash of the exact implementation source used by a run."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _line_offsets(code: str) -> list[int]:
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _python_spans(code: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return comment and string character spans, retaining partial tokens."""

    comments: list[tuple[int, int]] = []
    strings: list[tuple[int, int]] = []
    offsets = _line_offsets(code)

    def absolute(position: tuple[int, int]) -> int:
        row, column = position
        if row - 1 >= len(offsets):
            return len(code)
        return min(len(code), offsets[row - 1] + column)

    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for item in stream:
            span = (absolute(item.start), absolute(item.end))
            if item.type == token.COMMENT:
                comments.append(span)
            elif item.type == token.STRING:
                strings.append(span)
    except (IndentationError, tokenize.TokenError):
        # The complete prefix remains useful.  Attacks on malformed snippets
        # are audited, but conservative no-ops are preferable to guessed edits.
        pass
    return comments, strings


def _c_like_spans(
    code: str, language: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    comments: list[tuple[int, int]] = []
    strings: list[tuple[int, int]] = []
    index = 0
    while index < len(code):
        if code.startswith("//", index):
            end = code.find("\n", index + 2)
            end = len(code) if end < 0 else end
            comments.append((index, end))
            index = end
            continue
        if code.startswith("/*", index):
            closing = code.find("*/", index + 2)
            end = len(code) if closing < 0 else closing + 2
            comments.append((index, end))
            index = end
            continue

        special_end = (
            _cpp_raw_string_end(code, index)
            if language == "cpp"
            else _java_text_block_end(code, index)
            if language == "java"
            else None
        )
        if special_end is None and language in {"c", "cpp"}:
            special_end = _c_prefixed_quoted_end(code, index)
        if special_end is not None:
            strings.append((index, special_end))
            index = special_end
            continue
        if code[index] in {'"', "'"}:
            end = _quoted_end(code, index, code[index])
            strings.append((index, end))
            index = end
            continue
        index += 1
    return comments, strings


def _protected_spans(
    code: str, language: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    return _python_spans(code) if language == "py" else _c_like_spans(code, language)


def _mask_span(buffer: list[str], start: int, end: int) -> None:
    """Blank a comment without deleting physical lines or joining tokens."""

    for index in range(start, end):
        if buffer[index] not in "\r\n":
            buffer[index] = " "


def _comment_removal(code: str, language: str) -> tuple[str, int]:
    comments, _ = _protected_spans(code, language)
    if not comments:
        return code, 0
    buffer = list(code)
    for start, end in comments:
        _mask_span(buffer, start, end)
    return "".join(buffer), len(comments)


def _byte_line_offsets(code: str) -> list[int]:
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    return offsets


def _apply_byte_replacements(
    code: str, replacements: list[tuple[int, int, str]]
) -> tuple[str, int]:
    if not replacements:
        return code, 0
    source = code.encode("utf-8")
    unique = sorted(set(replacements), key=lambda item: (item[0], item[1]))
    for left, right in zip(unique, unique[1:]):
        if left[1] > right[0]:
            raise ValueError("overlapping identifier replacements")
    for start, end, replacement in reversed(unique):
        source = source[:start] + replacement.encode("utf-8") + source[end:]
    return source.decode("utf-8"), len(unique)


_PY_NESTED_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _walk_python_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Walk one lexical function scope, excluding nested lexical scopes."""

    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _PY_NESTED_SCOPES):
            continue
        yield from _walk_python_scope(child)


def _nested_python_names(function: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(function):
        if child is function:
            continue
        if isinstance(child, _PY_NESTED_SCOPES):
            names.update(item.id for item in ast.walk(child) if isinstance(item, ast.Name))
    return names


def _python_identifier_rename(code: str) -> tuple[str, int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0

    all_names = {
        item.id for item in ast.walk(tree) if isinstance(item, ast.Name)
    } | {
        item.arg for item in ast.walk(tree) if isinstance(item, ast.arg)
    }
    replacements: list[tuple[int, int, str]] = []
    byte_offsets = _byte_line_offsets(code)
    next_index = 0

    def fresh_name() -> str:
        nonlocal next_index
        while True:
            candidate = f"_t4_local_{next_index}"
            next_index += 1
            if candidate not in all_names:
                all_names.add(candidate)
                return candidate

    functions = [
        item
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for function in sorted(functions, key=lambda item: (item.lineno, item.col_offset)):
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if function.args.vararg is not None:
            arguments.append(function.args.vararg)
        if function.args.kwarg is not None:
            arguments.append(function.args.kwarg)

        scoped = list(_walk_python_scope(function))
        global_names = {
            name
            for item in scoped
            if isinstance(item, (ast.Global, ast.Nonlocal))
            for name in item.names
        }
        declared_order: list[tuple[int, int, str]] = [
            (item.lineno, item.col_offset, item.arg) for item in arguments
        ]
        declared_order.extend(
            (item.lineno, item.col_offset, item.id)
            for item in scoped
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        )
        blocked = global_names | _nested_python_names(function)
        ordered_names: list[str] = []
        for _, _, name in sorted(declared_order):
            if name not in blocked and name not in ordered_names:
                ordered_names.append(name)
        mapping = {name: fresh_name() for name in ordered_names}

        for item in scoped:
            if isinstance(item, ast.Name) and item.id in mapping:
                start = byte_offsets[item.lineno - 1] + item.col_offset
                end = byte_offsets[item.end_lineno - 1] + item.end_col_offset
                replacements.append((start, end, mapping[item.id]))
        for item in arguments:
            if item.arg in mapping:
                start = byte_offsets[item.lineno - 1] + item.col_offset
                end = start + len(item.arg.encode("utf-8"))
                replacements.append((start, end, mapping[item.arg]))

    return _apply_byte_replacements(code, replacements)


def _walk_tree(node) -> Iterator[object]:
    yield node
    for child in node.children:
        yield from _walk_tree(child)


def _first_identifier(node):
    if node is None:
        return None
    if node.type == "identifier":
        return node
    declarator = node.child_by_field_name("declarator")
    if declarator is not None and declarator is not node:
        found = _first_identifier(declarator)
        if found is not None:
            return found
    for child in node.named_children:
        found = _first_identifier(child)
        if found is not None:
            return found
    return None


def _c_like_declarations(function, language: str) -> list[object]:
    body = function.child_by_field_name("body")
    if body is None:
        return []
    declarations: list[object] = []
    parameter_types = {"parameter_declaration"} if language in {"c", "cpp"} else {
        "formal_parameter",
        "spread_parameter",
        "catch_formal_parameter",
    }
    local_types = {"declaration"} if language in {"c", "cpp"} else {
        "local_variable_declaration"
    }

    for item in _walk_tree(function):
        if item.type in parameter_types:
            name = item.child_by_field_name("name") or _first_identifier(
                item.child_by_field_name("declarator")
            )
            if name is not None:
                declarations.append(name)
    for item in _walk_tree(body):
        if item.type not in local_types:
            continue
        if language == "java":
            for child in item.named_children:
                if child.type == "variable_declarator":
                    name = child.child_by_field_name("name") or _first_identifier(child)
                    if name is not None:
                        declarations.append(name)
            continue
        declarators = list(item.children_by_field_name("declarator"))
        for child in item.named_children:
            if child.type == "init_declarator":
                declarators.append(child.child_by_field_name("declarator"))
        for declarator in declarators:
            if declarator is not None and any(
                descendant.type == "function_declarator"
                for descendant in _walk_tree(declarator)
            ):
                continue
            name = _first_identifier(declarator)
            if name is not None:
                declarations.append(name)
    return declarations


def _has_ancestor_type(node, types: set[str], stop) -> bool:
    parent = node.parent
    while parent is not None and parent != stop:
        if parent.type in types or parent.type.startswith("preproc_"):
            return True
        parent = parent.parent
    return False


def _is_unsafe_member_name(node) -> bool:
    parent = node.parent
    if parent is None:
        return False
    if parent.type in {"field_access", "field_expression"}:
        field = parent.child_by_field_name("field")
        return field is not None and field.start_byte == node.start_byte
    if parent.type in {"method_invocation", "method_reference"}:
        name = parent.child_by_field_name("name")
        return name is not None and name.start_byte == node.start_byte
    return False


def _nearest_ancestor_of_type(node, types: set[str]):
    parent = node.parent
    while parent is not None:
        if parent.type in types:
            return parent
        parent = parent.parent
    return None


def _c_like_identifier_rename(code: str, language: str) -> tuple[str, int]:
    root = _tree_sitter_parser(language).parse(code.encode("utf-8")).root_node
    if root.has_error:
        return code, 0
    function_types = (
        {"function_definition"}
        if language in {"c", "cpp"}
        else {"method_declaration", "constructor_declaration"}
    )
    functions = [item for item in _walk_tree(root) if item.type in function_types]
    all_identifier_text = {
        code.encode("utf-8")[item.start_byte : item.end_byte].decode("utf-8")
        for item in _walk_tree(root)
        if item.type == "identifier"
    }
    source = code.encode("utf-8")
    replacements: list[tuple[int, int, str]] = []
    next_index = 0

    def fresh_name() -> str:
        nonlocal next_index
        while True:
            candidate = f"_t4_local_{next_index}"
            next_index += 1
            if candidate not in all_identifier_text:
                all_identifier_text.add(candidate)
                return candidate

    for function in sorted(functions, key=lambda item: item.start_byte):
        nested_functions = [
            item
            for item in _walk_tree(function)
            if item != function and item.type in function_types
        ]
        blocked_names = {
            source[item.start_byte : item.end_byte].decode("utf-8")
            for nested in nested_functions
            for item in _walk_tree(nested)
            if item.type == "identifier"
        }
        declarations = sorted(
            {
                item
                for item in _c_like_declarations(function, language)
                if _nearest_ancestor_of_type(item, function_types) == function
            },
            key=lambda item: item.start_byte,
        )
        names: list[str] = []
        for item in declarations:
            name = source[item.start_byte : item.end_byte].decode("utf-8")
            if name not in names:
                names.append(name)
        mapping = {name: fresh_name() for name in names if name not in blocked_names}
        if not mapping:
            continue

        body = function.child_by_field_name("body")
        scan_roots = [body] if body is not None else []
        parameters = function.child_by_field_name("declarator") or function.child_by_field_name(
            "parameters"
        )
        if parameters is not None:
            scan_roots.append(parameters)
        excluded_ancestors = {
            *function_types,
            "lambda_expression",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        }
        for scan_root in scan_roots:
            for item in _walk_tree(scan_root):
                if item.type != "identifier":
                    continue
                name = source[item.start_byte : item.end_byte].decode("utf-8")
                if name not in mapping:
                    continue
                if _is_unsafe_member_name(item) or _has_ancestor_type(
                    item, excluded_ancestors, function
                ):
                    continue
                replacements.append((item.start_byte, item.end_byte, mapping[name]))

    return _apply_byte_replacements(code, replacements)


def _identifier_rename(code: str, language: str) -> tuple[str, int]:
    return (
        _python_identifier_rename(code)
        if language == "py"
        else _c_like_identifier_rename(code, language)
    )


def _preprocessor_spans(code: str, language: str) -> list[tuple[int, int]]:
    if language not in {"c", "cpp"}:
        return []
    lines = code.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    offset = 0
    continuing = False
    for line in lines:
        body = line.rstrip("\r\n")
        directive = continuing or body.lstrip(" \t").startswith("#")
        if directive:
            spans.append((offset, offset + len(line)))
            continuing = body.rstrip(" \t").endswith("\\")
        else:
            continuing = False
        offset += len(line)
    return spans


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _format_normalization(code: str, language: str) -> tuple[str, int]:
    comments, strings = _protected_spans(code, language)
    protected = comments + strings + _preprocessor_spans(code, language)
    lines = code.splitlines(keepends=True)
    normalized: list[tuple[str, bool]] = []
    offset = 0
    count = 0

    for line in lines:
        if line.endswith("\r\n"):
            body, ending = line[:-2], "\r\n"
        elif line.endswith(("\n", "\r")):
            body, ending = line[:-1], line[-1]
        else:
            body, ending = line, ""
        line_start, line_end = offset, offset + len(body)
        is_protected = _overlaps(line_start, max(line_start + 1, line_end), protected)
        if not is_protected:
            trimmed = body.rstrip(" \t")
            if trimmed != body:
                body = trimmed
                count += 1
        is_blank = not body.strip() and not is_protected
        if is_blank and normalized and normalized[-1][1]:
            count += 1
        else:
            normalized.append((body + ending, is_blank))
        offset += len(line)

    while normalized and normalized[0][1]:
        normalized.pop(0)
        count += 1
    while normalized and normalized[-1][1]:
        normalized.pop()
        count += 1
    output = "".join(line for line, _ in normalized)
    return output, count


_ENCODING_COOKIE = re.compile(r"coding[:=][ \t]*[-_.a-zA-Z0-9]+")


def _comment_injection(code: str, language: str) -> tuple[str, int]:
    comment = INJECTED_COMMENTS[language] + "\n"
    if language != "py":
        return comment + code, 1

    lines = code.splitlines(keepends=True)
    insert_after = 0
    if lines and lines[0].startswith("#!"):
        insert_after = 1
    for index in range(min(2, len(lines))):
        if _ENCODING_COOKIE.search(lines[index]):
            insert_after = max(insert_after, index + 1)
    if insert_after == 0:
        return comment + code, 1
    prefix = "".join(lines[:insert_after])
    suffix = "".join(lines[insert_after:])
    if prefix and not prefix.endswith(("\n", "\r")):
        prefix += "\n"
    return prefix + comment + suffix, 1


_TRANSFORMS: dict[str, Transform] = {
    "comment_removal": _comment_removal,
    "identifier_rename": _identifier_rename,
    "format_normalization": _format_normalization,
    "comment_injection": _comment_injection,
}


def _run_transform(code: str, language: str, attack: str) -> tuple[str, int]:
    if attack != "combined":
        return _TRANSFORMS[attack](code, language)
    transformed = code
    count = 0
    for component in (
        "comment_removal",
        "identifier_rename",
        "format_normalization",
    ):
        transformed, component_count = _TRANSFORMS[component](transformed, language)
        count += component_count
    return transformed, count


def apply_attack(code: str, language: str, attack: str) -> AttackResult:
    """Apply one frozen T4 attack and return its complete parse/hash audit."""

    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if not isinstance(language, str) or language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language!r}")
    if not isinstance(attack, str) or attack not in ATTACKS:
        raise ValueError(f"unsupported attack: {attack!r}")

    before = analyze_enhanced(code, language)
    output, transform_count = _run_transform(code, language, attack)
    after = analyze_enhanced(output, language)
    failure_reason = "parse-regression" if before.parse_ok and not after.parse_ok else None
    return AttackResult(
        code=output,
        attack=attack,
        language=language,
        input_sha256=_sha256(code),
        output_sha256=_sha256(output),
        changed=output != code,
        transform_count=transform_count,
        parse_ok_before=before.parse_ok,
        parse_ok_after=after.parse_ok,
        backend_before=before.backend,
        backend_after=after.backend,
        failure_reason=failure_reason,
    )


__all__ = [
    "ATTACKS",
    "ATTACK_VERSION",
    "INJECTED_COMMENTS",
    "LANGUAGES",
    "AttackResult",
    "attack_source_sha256",
    "apply_attack",
]
