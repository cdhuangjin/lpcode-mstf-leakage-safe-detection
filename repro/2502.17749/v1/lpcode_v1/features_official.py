"""Exact compatibility implementation of LPcode's official ten features."""

from __future__ import annotations

import re

import numpy as np


FEATURE_NAMES = (
    "function_naming_consistency",
    "variable_naming_consistency",
    "class_naming_consistency",
    "constant_naming_consistency",
    "indentation_consistency",
    "avg_function_length",
    "avg_nesting_depth",
    "comment_ratio",
    "avg_function_name_length",
    "avg_variable_name_length",
)


def classify_naming(name: str) -> str:
    if re.match(r"^[a-z]+(?:[A-Z][a-z]*)*$", name):
        return "camelCase"
    if re.match(r"^[a-z]+(?:_[a-z]+)+$", name):
        return "snake_case"
    if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]*)*$", name):
        return "PascalCase"
    if re.match(r"^[A-Z]+(?:_[A-Z]+)+$", name):
        return "UPPER_SNAKE_CASE"
    return "Other"


def extract_function_names(code: str, language: str) -> list[str]:
    if language == "py":
        pattern = r"^\s*def\s+(\w+)\s*\("
    elif language in ["c", "cpp", "java"]:
        pattern = r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:[\w<>\[\]\*&]+\s+)+(\w+)\s*\("
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)


def extract_variable_names(code: str, language: str) -> list[str]:
    if language == "py":
        pattern = r"^\s*(\w+)\s*=\s*[^=]"
    elif language in ["c", "cpp", "java"]:
        pattern = r"^\s*(?:[\w<>\[\]\*&]+\s+)+(\w+)\s*(?:=|;|\[|\()"
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)


def extract_class_names(code: str, language: str) -> list[str]:
    if language == "py":
        pattern = r"^\s*class\s+(\w+)\s*(?:\(|:)?"
    elif language in ["c", "cpp", "java"]:
        pattern = r"^\s*(?:public|private|protected)?\s*(?:class|struct|interface)\s+(\w+)"
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)


def extract_constant_names(code: str, language: str) -> list[str]:
    if language == "py":
        pattern = r"^\s*([A-Z][A-Z_0-9]*)\s*=\s*"
    elif language in ["c", "cpp"]:
        pattern = r"^\s*#define\s+([A-Z][A-Z_0-9]*)\b"
    elif language == "java":
        pattern = r"^\s*(?:public|private|protected)?\s*static\s+final\s+[\w<>\[\]]+\s+([A-Z][A-Z_0-9]*)\s*="
    else:
        return []
    return re.findall(pattern, code, re.MULTILINE)


def _naming_consistency(names: list[str]) -> float:
    naming_counts = {
        "camelCase": 0,
        "snake_case": 0,
        "PascalCase": 0,
        "UPPER_SNAKE_CASE": 0,
        "Other": 0,
    }
    for name in names:
        naming_counts[classify_naming(name)] += 1
    total_names = sum(naming_counts.values())
    if total_names > 0:
        return max(naming_counts.values()) / total_names
    return 0.0


def _indentation_consistency(lines: list[str]) -> float:
    indent_unit_counts: dict[int, int] = {}
    total_indented_lines = 0
    for line in lines:
        stripped_line = line.lstrip()
        if not stripped_line or stripped_line.startswith(("#", "//", "/*", "*")):
            continue
        indent = line[: len(line) - len(stripped_line)]
        if indent:
            total_indented_lines += 1
            indent_length = len(indent.replace("\t", "    "))
            indent_unit_counts[indent_length] = indent_unit_counts.get(indent_length, 0) + 1
    if total_indented_lines == 0:
        return 1.0
    return max(indent_unit_counts.values()) / total_indented_lines


def _average_function_length(lines: list[str], language: str) -> float:
    function_pattern = {
        "py": r"^\s*def\s+\w+\s*\(.*\):",
        "c": r"^\s*(?:[\w<>\[\]\*&]+\s+)+\w+\s*\(.*\)\s*\{",
        "cpp": r"^\s*(?:[\w<>\[\]\*&]+\s+)+\w+\s*\(.*\)\s*\{",
        "java": r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:[\w<>\[\]\.&]+\s+)+\w+\s*\(.*\)\s*\{",
    }.get(language)
    if not function_pattern:
        return 0.0
    function_lengths: list[int] = []
    function_starts = [i for i, line in enumerate(lines) if re.match(function_pattern, line)]
    for start_line in function_starts:
        length = 0
        nesting_level = 0
        i = start_line
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            if language == "py":
                start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                if i > start_line and stripped_line and len(line) - len(line.lstrip()) <= start_indent:
                    break
            else:
                nesting_level += line.count("{") - line.count("}")
                if i > start_line and nesting_level <= 0:
                    break
            length += 1
            i += 1
        function_lengths.append(length)
    return sum(function_lengths) / len(function_lengths) if function_lengths else 0.0


def _average_nesting_depth(lines: list[str], language: str) -> float:
    nesting_depths: list[int] = []
    if language == "py":
        indent_levels: list[int] = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            current_indent = len(line) - len(line.lstrip())
            while indent_levels and current_indent < indent_levels[-1]:
                indent_levels.pop()
            if indent_levels and current_indent == indent_levels[-1]:
                pass
            elif current_indent > (indent_levels[-1] if indent_levels else 0):
                indent_levels.append(current_indent)
            nesting_depths.append(len(indent_levels))
    else:
        nesting_level = 0
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith(("//", "/*", "*")):
                continue
            nesting_level += line.count("{") - line.count("}")
            nesting_depths.append(max(nesting_level, 0))
    return sum(nesting_depths) / len(nesting_depths) if nesting_depths else 0.0


def _comment_ratio(lines: list[str], language: str) -> float:
    comment_lines = 0
    code_lines = 0
    in_block_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if language == "py":
            if stripped_line.startswith("#"):
                comment_lines += 1
            elif re.match(r"('''|\"\"\")", stripped_line):
                comment_lines += 1
                if stripped_line.count("'''") % 2 == 1 or stripped_line.count('"""') % 2 == 1:
                    in_block_comment = not in_block_comment
            elif in_block_comment:
                comment_lines += 1
            else:
                code_lines += 1
        else:
            if in_block_comment:
                comment_lines += 1
                if "*/" in stripped_line:
                    in_block_comment = False
            elif stripped_line.startswith("/*"):
                comment_lines += 1
                if "*/" not in stripped_line:
                    in_block_comment = True
            elif stripped_line.startswith("//"):
                comment_lines += 1
            else:
                code_lines += 1
    total_code_lines = code_lines + comment_lines
    return comment_lines / total_code_lines if total_code_lines > 0 else 0.0


def analyze_code(code: str, language: str) -> np.ndarray:
    """Analyze one snippet using the exact official feature semantics."""
    lines = code.splitlines()
    function_names = extract_function_names(code, language)
    variable_names = extract_variable_names(code, language)
    class_names = extract_class_names(code, language)
    constant_names = extract_constant_names(code, language)
    feature = [
        _naming_consistency(function_names),
        _naming_consistency(variable_names),
        _naming_consistency(class_names),
        _naming_consistency(constant_names),
        _indentation_consistency(lines),
        _average_function_length(lines, language),
        _average_nesting_depth(lines, language),
        _comment_ratio(lines, language),
        sum(len(name) for name in function_names) / len(function_names) if function_names else 0.0,
        sum(len(name) for name in variable_names) / len(variable_names) if variable_names else 0.0,
    ]
    return np.asarray(feature, dtype=np.float64)
