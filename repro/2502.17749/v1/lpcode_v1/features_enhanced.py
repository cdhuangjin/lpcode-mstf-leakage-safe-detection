"""Deterministic multi-view features added to LPcode's official ten.

The returned vector has the order given by :data:`FEATURE_NAMES`.  Lexical
densities use the number of effective lexical items (identifier, keyword,
operator, or literal) as their denominator.  Comments contribute no lexical
items; a quoted string/character is one literal and its contents are never
tokenized as code.  Identifier statistics use identifier occurrences, not
unique spellings, and entropy is Shannon entropy in bits.

Branch/loop/return/exception densities use structural statement units as their
denominator.  Parsed units are leaf/control statement nodes, Python conditional
expressions, and declaration or handler/label clauses, excluding block/compound
containers.  Fallback units are semicolon (C-family) or effective-line (Python)
leaf slots plus explicit
branch, loop, function (Python), and exception-control clauses; unterminated
return/raise/throw tokens provide a leaf floor.  Every density numerator is a
subset of these units, so densities stay in ``[0, 1]`` and an unrelated missing
closing delimiter does not change the denominator.  Statement density is units
per non-blank physical line.  AST depth counts named structural edges from the
root (root depth zero), ignoring Python context/operator marker nodes.
Cyclomatic complexity is one plus decision points when at least one effective
lexical item or counted structural construct is present.  Comment-only and
whitespace-only input therefore has zero complexity.

Blank-line ratio uses all physical lines.  Line-length mean/std and indentation
entropy use non-blank physical lines; tabs expand to four-column stops.  Empty
input has no physical lines and returns eighteen zeros.  Parse failures retain
the lexical/format features and use conservative token-based structural
estimates.  No feature may be NaN or infinite.

C, C++, and Java parsers come from pinned, offline tree-sitter grammar wheels;
feature extraction never downloads parser data at runtime.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import io
import keyword
import math
import re
import token as py_token
import tokenize
from typing import Iterable, Iterator

import numpy as np


FEATURE_NAMES = (
    "identifier_entropy",
    "identifier_length_mean",
    "identifier_length_std",
    "keyword_density",
    "operator_density",
    "literal_density",
    "ast_depth",
    "branch_density",
    "loop_density",
    "function_count",
    "statement_density",
    "cyclomatic_complexity",
    "return_density",
    "exception_density",
    "blank_line_ratio",
    "line_length_mean",
    "line_length_std",
    "indentation_entropy",
)

LANGUAGES = ("c", "cpp", "java", "py")


@dataclass(frozen=True)
class EnhancedAnalysis:
    """One enhanced vector and auditable parser provenance."""

    values: np.ndarray
    parse_ok: bool
    backend: str
    language: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class _Lexeme:
    kind: str
    text: str


@dataclass(frozen=True)
class _Structure:
    depth: int = 0
    branches: int = 0
    loops: int = 0
    functions: int = 0
    statements: int = 0
    decisions: int = 0
    returns: int = 0
    exceptions: int = 0


_PYTHON_OPERATORS = {
    "+", "-", "*", "**", "/", "//", "%", "@", "<<", ">>", "&", "|", "^", "~",
    ":=", "<", ">", "<=", ">=", "==", "!=", "=", "+=", "-=", "*=", "/=", "//=",
    "%=", "@=", "&=", "|=", "^=", ">>=", "<<=", "**=", "->",
}
_PYTHON_OPERATORS_LONGEST_FIRST = tuple(
    sorted(_PYTHON_OPERATORS, key=lambda item: (-len(item), item))
)

_C_OPERATORS = tuple(
    sorted(
        {
            ">>=", "<<=", "->*", "...", "::", "++", "--", "->", ".*", "&&", "||",
            "<=", ">=", "==", "!=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
            "<<", ">>", "+", "-", "*", "/", "%", "=", "<", ">", "!", "~", "&", "|",
            "^", "?", ".",
        },
        key=lambda item: (-len(item), item),
    )
)

_C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
    "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct",
    "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Alignas",
    "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn",
    "_Static_assert", "_Thread_local",
}

_CPP_KEYWORDS = _C_KEYWORDS | {
    "alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor", "bool", "catch",
    "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "consteval",
    "constexpr", "constinit", "const_cast", "co_await", "co_return", "co_yield", "decltype",
    "delete", "dynamic_cast", "explicit", "export", "false", "friend", "mutable", "namespace",
    "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
    "protected", "public", "reflexpr", "reinterpret_cast", "requires", "static_assert",
    "static_cast", "template", "this", "thread_local", "throw", "true", "try", "typeid",
    "typename", "using", "virtual", "wchar_t", "xor", "xor_eq",
}

_JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
    "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false", "null", "record", "sealed",
    "permits", "non-sealed", "var", "yield",
}

_KEYWORDS = {
    "c": _C_KEYWORDS,
    "cpp": _CPP_KEYWORDS,
    "java": _JAVA_KEYWORDS,
    "py": set(keyword.kwlist),
}
_PYTHON_LITERAL_WORDS = {"True", "False", "None"}
_C_LIKE_LITERAL_WORDS = {
    "c": set(),
    "cpp": {"true", "false", "nullptr"},
    "java": {"true", "false", "null"},
}


def _python_lexemes(code: str) -> list[_Lexeme]:
    result: list[_Lexeme] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for item in stream:
            kind, text = item.type, item.string
            if kind == py_token.NAME:
                kind_name = (
                    "literal"
                    if text in _PYTHON_LITERAL_WORDS
                    else "keyword"
                    if keyword.iskeyword(text)
                    else "identifier"
                )
                result.append(_Lexeme(kind_name, text))
            elif kind in {py_token.NUMBER, py_token.STRING}:
                result.append(_Lexeme("literal", text))
            elif kind == py_token.OP and text == "...":
                result.append(_Lexeme("literal", text))
            elif kind == py_token.OP and text in _PYTHON_OPERATORS:
                result.append(_Lexeme("operator", text))
    except (IndentationError, tokenize.TokenError):
        return _python_fallback_lexemes(code)
    return result


_PYTHON_STRING_PREFIXES = ("fr", "rf", "br", "rb", "r", "u", "b", "f", "")


def _quoted_end(code: str, quote_start: int, quote: str) -> int:
    """Return one-past a quoted literal, consuming to EOF if unclosed."""

    triple = len(quote) == 3
    index = quote_start + len(quote)
    while index < len(code):
        if code.startswith(quote, index):
            return index + len(quote)
        if not triple and code[index] in "\r\n":
            return index
        index = min(len(code), index + 2) if code[index] == "\\" else index + 1
    return len(code)


def _python_string_end(code: str, index: int) -> int | None:
    for prefix in _PYTHON_STRING_PREFIXES:
        if prefix and code[index : index + len(prefix)].lower() != prefix:
            continue
        quote_start = index + len(prefix)
        if quote_start >= len(code):
            continue
        for quote in ('"""', "'''", '"', "'"):
            if code.startswith(quote, quote_start):
                return _quoted_end(code, quote_start, quote)
    return None


def _python_fallback_lexemes(code: str) -> list[_Lexeme]:
    """Recover Python tokens without applying C/C++ comment/operator rules."""

    result: list[_Lexeme] = []
    index = 0
    while index < len(code):
        char = code[index]
        if char.isspace():
            index += 1
            continue
        if char == "#":
            newline = code.find("\n", index + 1)
            index = len(code) if newline < 0 else newline + 1
            continue
        string_end = _python_string_end(code, index)
        if string_end is not None:
            result.append(_Lexeme("literal", code[index:string_end]))
            index = string_end
            continue
        identifier = re.match(r"[A-Za-z_]\w*", code[index:], flags=re.UNICODE)
        if identifier:
            text = identifier.group(0)
            kind_name = (
                "literal"
                if text in _PYTHON_LITERAL_WORDS
                else "keyword"
                if keyword.iskeyword(text)
                else "identifier"
            )
            result.append(_Lexeme(kind_name, text))
            index += len(text)
            continue
        number = re.match(
            r"(?:0[xX][0-9A-Fa-f_]+|0[bB][01_]+|0[oO][0-7_]+|"
            r"(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?[\d_]+)?)[jJ]?",
            code[index:],
        )
        if number:
            text = number.group(0)
            result.append(_Lexeme("literal", text))
            index += len(text)
            continue
        if code.startswith("...", index):
            result.append(_Lexeme("literal", "..."))
            index += 3
            continue
        operator = next(
            (
                candidate
                for candidate in _PYTHON_OPERATORS_LONGEST_FIRST
                if code.startswith(candidate, index)
            ),
            None,
        )
        if operator is not None:
            result.append(_Lexeme("operator", operator))
            index += len(operator)
            continue
        if char in "{}()[];,:":
            result.append(_Lexeme("punctuation", char))
        index += 1
    return result


_CPP_RAW_PREFIXES = ('u8R"', 'uR"', 'UR"', 'LR"', 'R"')
_C_ORDINARY_PREFIXES = ("u8", "L", "u", "U")


def _cpp_raw_string_end(code: str, index: int) -> int | None:
    prefix = next((item for item in _CPP_RAW_PREFIXES if code.startswith(item, index)), None)
    if prefix is None:
        return None
    delimiter_start = index + len(prefix)
    opening = code.find("(", delimiter_start, delimiter_start + 17)
    if opening < 0 or any(
        char.isspace() or char in "\\()" for char in code[delimiter_start:opening]
    ):
        return len(code)
    delimiter = code[delimiter_start:opening]
    marker = ")" + delimiter + '"'
    closing = code.find(marker, opening + 1)
    return len(code) if closing < 0 else closing + len(marker)


def _java_text_block_end(code: str, index: int) -> int | None:
    if not code.startswith('"""', index):
        return None
    closing = code.find('"""', index + 3)
    return len(code) if closing < 0 else closing + 3


def _c_prefixed_quoted_end(code: str, index: int) -> int | None:
    prefix = next(
        (item for item in _C_ORDINARY_PREFIXES if code.startswith(item, index)),
        None,
    )
    if prefix is None:
        return None
    quote_start = index + len(prefix)
    if quote_start >= len(code) or code[quote_start] not in {'"', "'"}:
        return None
    return _quoted_end(code, quote_start, code[quote_start])


def _c_like_lexemes(code: str, language: str) -> list[_Lexeme]:
    keywords = _KEYWORDS[language]
    result: list[_Lexeme] = []
    index = 0
    length = len(code)
    while index < length:
        char = code[index]
        if char.isspace():
            index += 1
            continue
        if code.startswith("//", index):
            newline = code.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if code.startswith("/*", index):
            end = code.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        special_string_end = (
            _cpp_raw_string_end(code, index)
            if language == "cpp"
            else _java_text_block_end(code, index)
            if language == "java"
            else None
        )
        if special_string_end is None and language in {"c", "cpp"}:
            special_string_end = _c_prefixed_quoted_end(code, index)
        if special_string_end is not None:
            result.append(_Lexeme("literal", code[index:special_string_end]))
            index = special_string_end
            continue
        if char in {'"', "'"}:
            quote = char
            start = index
            index += 1
            while index < length:
                if code[index] == "\\":
                    index += 2
                elif code[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            result.append(_Lexeme("literal", code[start:index]))
            continue
        identifier = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", code[index:])
        if identifier:
            text = identifier.group(0)
            kind = (
                "literal"
                if text in _C_LIKE_LITERAL_WORDS[language]
                else "keyword"
                if text in keywords
                else "identifier"
            )
            result.append(_Lexeme(kind, text))
            index += len(text)
            continue
        number = re.match(r"(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[A-Za-z]*", code[index:])
        if number:
            text = number.group(0)
            result.append(_Lexeme("literal", text))
            index += len(text)
            continue
        operator = next((candidate for candidate in _C_OPERATORS if code.startswith(candidate, index)), None)
        if operator is not None:
            result.append(_Lexeme("operator", operator))
            index += len(operator)
            continue
        if char in "{}()[];,:":
            result.append(_Lexeme("punctuation", char))
        index += 1
    return result


def _entropy(items: Iterable[object]) -> float:
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


_PY_IGNORED_DEPTH_NODES = (
    ast.expr_context,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)


def _python_depth(node: ast.AST, depth: int = 0) -> int:
    children = [child for child in ast.iter_child_nodes(node) if not isinstance(child, _PY_IGNORED_DEPTH_NODES)]
    if not children:
        return depth
    return max(_python_depth(child, depth + 1) for child in children)


def _python_structure(tree: ast.AST) -> _Structure:
    nodes = list(ast.walk(tree))
    finally_clauses = sum(
        isinstance(node, (ast.Try, ast.TryStar)) and bool(node.finalbody)
        for node in nodes
    )
    conditional_expressions = sum(isinstance(node, ast.IfExp) for node in nodes)
    statements = (
        sum(isinstance(node, (ast.stmt, ast.ExceptHandler)) for node in nodes)
        + finally_clauses
        + conditional_expressions
    )
    branches = sum(isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in nodes)
    loops = sum(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in nodes)
    functions = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in nodes)
    returns = sum(isinstance(node, ast.Return) for node in nodes)
    exceptions = sum(isinstance(node, (ast.Raise, ast.Try, ast.TryStar, ast.ExceptHandler)) for node in nodes)
    boolean_decisions = sum(max(0, len(node.values) - 1) for node in nodes if isinstance(node, ast.BoolOp))
    exception_decisions = sum(isinstance(node, ast.ExceptHandler) for node in nodes)
    return _Structure(
        depth=_python_depth(tree),
        branches=branches,
        loops=loops,
        functions=functions,
        statements=statements,
        decisions=branches + loops + boolean_decisions + exception_decisions,
        returns=returns,
        exceptions=exceptions,
    )


@lru_cache(maxsize=3)
def _tree_sitter_parser(language: str):
    from tree_sitter import Language, Parser

    if language == "c":
        import tree_sitter_c as grammar
    elif language == "cpp":
        import tree_sitter_cpp as grammar
    elif language == "java":
        import tree_sitter_java as grammar
    else:  # validated by the public entry point
        raise ValueError(f"unsupported language: {language!r}")
    return Parser(Language(grammar.language()))


def _walk_named_tree(node, depth: int = 0) -> Iterator[tuple[object, int]]:
    yield node, depth
    for child in node.named_children:
        yield from _walk_named_tree(child, depth + 1)


_TREE_BRANCHES = {
    "if_statement", "switch_statement", "switch_expression", "conditional_expression",
    "case_statement", "switch_label", "switch_rule",
}
_TREE_LOOPS = {"for_statement", "enhanced_for_statement", "while_statement", "do_statement"}
_TREE_FUNCTIONS = {"function_definition", "method_declaration", "constructor_declaration"}
_TREE_RETURNS = {"return_statement", "co_return_statement"}
_TREE_EXCEPTIONS = {"throw_statement", "try_statement", "catch_clause"}
_TREE_DECLARATIONS = {"declaration", "local_variable_declaration"}
_TREE_CLAUSES = {"finally_clause"}
_TREE_STATEMENT_CONTAINERS = {"compound_statement"}


def _tree_structure(root, lexemes: list[_Lexeme]) -> _Structure:
    walked = list(_walk_named_tree(root))
    types = [node.type for node, _ in walked]
    depth = max((depth for _, depth in walked), default=0)
    branches = sum(node_type in _TREE_BRANCHES for node_type in types)
    loops = sum(node_type in _TREE_LOOPS for node_type in types)
    functions = sum(node_type in _TREE_FUNCTIONS for node_type in types)
    returns = sum(node_type in _TREE_RETURNS for node_type in types)
    exceptions = sum(node_type in _TREE_EXCEPTIONS for node_type in types)
    structural_unit_types = _TREE_BRANCHES | _TREE_LOOPS | _TREE_EXCEPTIONS | _TREE_CLAUSES
    statements = sum(
        (
            node_type.endswith("_statement")
            and node_type not in _TREE_STATEMENT_CONTAINERS
        )
        or node_type in _TREE_DECLARATIONS
        or node_type in structural_unit_types
        for node_type in types
    )
    logical_decisions = sum(lexeme.kind == "operator" and lexeme.text in {"&&", "||"} for lexeme in lexemes)
    catches = sum(node_type == "catch_clause" for node_type in types)
    return _Structure(
        depth=depth,
        branches=branches,
        loops=loops,
        functions=functions,
        statements=statements,
        decisions=branches + loops + logical_decisions + catches,
        returns=returns,
        exceptions=exceptions,
    )


def _fallback_c_like_function_count(lexemes: list[_Lexeme]) -> int:
    """Count definition-shaped tokens after comments and literals are protected."""

    count = 0
    for name_index, lexeme in enumerate(lexemes[:-2]):
        if lexeme.kind != "identifier" or lexemes[name_index + 1].text != "(":
            continue
        depth = 0
        for close_index in range(name_index + 1, len(lexemes)):
            symbol = lexemes[close_index].text
            if symbol == "(":
                depth += 1
            elif symbol == ")":
                depth -= 1
                if depth == 0:
                    if close_index + 1 < len(lexemes) and lexemes[close_index + 1].text == "{":
                        count += 1
                    break
            elif symbol in {";", "{"} and depth == 1:
                break
    return count


def _python_fallback_line_slots(code: str) -> int:
    """Count semicolon-separated effective statement slots per physical line."""

    slots = 0
    for line in code.splitlines():
        segment_has_effective = False
        for lexeme in _python_fallback_lexemes(line):
            if lexeme.text == ";":
                slots += int(segment_has_effective)
                segment_has_effective = False
            elif lexeme.kind != "punctuation":
                segment_has_effective = True
        slots += int(segment_has_effective)
    return slots


def _fallback_structure(code: str, language: str, lexemes: list[_Lexeme]) -> _Structure:
    keywords = [lexeme.text for lexeme in lexemes if lexeme.kind == "keyword"]
    punctuation = [lexeme.text for lexeme in lexemes if lexeme.kind == "punctuation"]
    branch_words = {"if", "elif", "case", "switch"}
    loop_words = {"for", "while", "do"}
    exception_words = {"raise", "throw", "try", "except", "catch"}
    exception_control_words = {"try", "except", "catch", "finally"}
    exception_transfer_words = {"raise", "throw"}
    branches = sum(word in branch_words for word in keywords)
    loops = sum(word in loop_words for word in keywords)
    returns = sum(word in {"return", "co_return"} for word in keywords)
    exceptions = sum(word in exception_words for word in keywords)
    logical = sum(
        (lexeme.kind == "operator" and lexeme.text in {"&&", "||"})
        or (lexeme.kind == "keyword" and lexeme.text in {"and", "or"})
        for lexeme in lexemes
    )
    if language == "py":
        functions = keywords.count("def")
        control_units = (
            branches
            + loops
            + functions
            + sum(word in exception_control_words for word in keywords)
        )
        leaf_estimate = max(0, _python_fallback_line_slots(code) - control_units)
        leaf_floor = returns + sum(word in exception_transfer_words for word in keywords)
        statements = max(leaf_estimate, leaf_floor) + control_units
    else:
        functions = _fallback_c_like_function_count(lexemes)
        control_units = (
            branches
            + loops
            + sum(word in exception_control_words for word in keywords)
        )
        leaf_floor = returns + sum(word in exception_transfer_words for word in keywords)
        statements = max(punctuation.count(";"), leaf_floor) + control_units
    statements = max(statements, branches, loops, returns, exceptions)
    depth = 0
    current = 0
    for symbol in punctuation:
        if symbol in {"{", "(", "["}:
            current += 1
            depth = max(depth, current)
        elif symbol in {"}", ")", "]"}:
            current = max(0, current - 1)
    return _Structure(
        depth=depth,
        branches=branches,
        loops=loops,
        functions=functions,
        statements=statements,
        decisions=branches + loops + logical + sum(word in {"except", "catch"} for word in keywords),
        returns=returns,
        exceptions=exceptions,
    )


def _format_features(code: str) -> tuple[float, float, float, float]:
    lines = code.splitlines()
    if not lines:
        return 0.0, 0.0, 0.0, 0.0
    nonblank = [line for line in lines if line.strip()]
    blank_ratio = (len(lines) - len(nonblank)) / len(lines)
    if not nonblank:
        return float(blank_ratio), 0.0, 0.0, 0.0
    lengths = np.asarray([len(line) for line in nonblank], dtype=np.float64)
    indent_widths = [len(line[: len(line) - len(line.lstrip(" \t"))].expandtabs(4)) for line in nonblank]
    return float(blank_ratio), float(lengths.mean()), float(lengths.std()), _entropy(indent_widths)


def _normalize_parser_source(code: str) -> str:
    """Trim inert ``#include`` tails only in the parser's private copy.

    Backslash-terminated directives are deliberately untouched because making
    the backslash the final byte would create a semantic line splice.
    """

    normalized: list[str] = []
    for line in code.splitlines(keepends=True):
        if line.endswith("\r\n"):
            body, ending = line[:-2], "\r\n"
        elif line.endswith(("\n", "\r")):
            body, ending = line[:-1], line[-1]
        else:
            body, ending = line, ""
        trimmed = body.rstrip(" \t")
        if (
            re.match(r"^[ \t]*#[ \t]*include\b", body)
            and not trimmed.endswith("\\")
        ):
            body = trimmed
        normalized.append(body + ending)
    return "".join(normalized)


def analyze_enhanced(code: str, language: str) -> EnhancedAnalysis:
    """Return the enhanced 18-vector and parser provenance for one snippet."""

    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if not isinstance(language, str) or language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language!r}")

    lexemes = _python_lexemes(code) if language == "py" else _c_like_lexemes(code, language)
    if language == "py":
        try:
            structure = _python_structure(ast.parse(code))
            parse_ok, backend, reason = True, "python-ast", None
        except SyntaxError:
            structure = _fallback_structure(code, language, lexemes)
            parse_ok, backend, reason = False, "lexical-fallback", "syntax-error"
    else:
        parser_source = _normalize_parser_source(code)
        root = _tree_sitter_parser(language).parse(parser_source.encode("utf-8")).root_node
        if root.has_error:
            structure = _fallback_structure(code, language, lexemes)
            parse_ok, backend, reason = False, "lexical-fallback", "syntax-error"
        else:
            structure = _tree_structure(root, lexemes)
            parse_ok, backend, reason = True, "tree-sitter", None

    effective = [lexeme for lexeme in lexemes if lexeme.kind != "punctuation"]
    identifiers = [lexeme.text for lexeme in effective if lexeme.kind == "identifier"]
    lengths = np.asarray([len(name) for name in identifiers], dtype=np.float64)
    denominator = len(effective)
    keyword_count = sum(lexeme.kind == "keyword" for lexeme in effective)
    operator_count = sum(lexeme.kind == "operator" for lexeme in effective)
    literal_count = sum(lexeme.kind == "literal" for lexeme in effective)
    blank_ratio, line_mean, line_std, indent_entropy = _format_features(code)
    nonblank_lines = sum(bool(line.strip()) for line in code.splitlines())
    statement_denominator = structure.statements
    has_structure = any(
        (
            structure.statements,
            structure.functions,
            structure.branches,
            structure.loops,
            structure.returns,
            structure.exceptions,
        )
    )
    complexity = 1 + structure.decisions if effective or has_structure else 0

    values = np.asarray(
        [
            _entropy(identifiers),
            float(lengths.mean()) if lengths.size else 0.0,
            float(lengths.std()) if lengths.size else 0.0,
            keyword_count / denominator if denominator else 0.0,
            operator_count / denominator if denominator else 0.0,
            literal_count / denominator if denominator else 0.0,
            float(structure.depth),
            structure.branches / statement_denominator if statement_denominator else 0.0,
            structure.loops / statement_denominator if statement_denominator else 0.0,
            float(structure.functions),
            structure.statements / nonblank_lines if nonblank_lines else 0.0,
            float(complexity),
            structure.returns / statement_denominator if statement_denominator else 0.0,
            structure.exceptions / statement_denominator if statement_denominator else 0.0,
            blank_ratio,
            line_mean,
            line_std,
            indent_entropy,
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("enhanced feature extraction produced a non-finite or invalid vector")
    values.setflags(write=False)
    return EnhancedAnalysis(values, parse_ok, backend, language, reason)


__all__ = ["EnhancedAnalysis", "FEATURE_NAMES", "LANGUAGES", "analyze_enhanced"]
