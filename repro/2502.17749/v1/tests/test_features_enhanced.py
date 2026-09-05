from __future__ import annotations

import math
import importlib.util

import numpy as np
import pytest

from lpcode_v1.features_enhanced import (
    FEATURE_NAMES,
    EnhancedAnalysis,
    analyze_enhanced,
)


EXPECTED_FEATURE_NAMES = (
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


def _features(result: EnhancedAnalysis) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, result.values, strict=True))


def test_feature_contract_empty_input_and_determinism() -> None:
    assert FEATURE_NAMES == EXPECTED_FEATURE_NAMES

    first = analyze_enhanced("", "py")
    second = analyze_enhanced("", "py")

    assert isinstance(first, EnhancedAnalysis)
    assert first.values.shape == (18,)
    assert first.values.dtype == np.float64
    assert np.array_equal(first.values, np.zeros(18, dtype=np.float64))
    assert np.array_equal(first.values, second.values)
    assert first.parse_ok is True
    assert first.backend == "python-ast"
    assert first.language == "py"
    assert first.fallback_reason is None


def test_python_ast_exact_eighteen_feature_semantics() -> None:
    code = (
        "def f(x):\n"
        "    if x > 0:\n"
        "        return \"while + fake\"\n"
        "    return 0\n"
    )

    result = analyze_enhanced(code, "py")
    values = _features(result)

    assert result.parse_ok is True
    assert result.backend == "python-ast"
    assert values["identifier_entropy"] == pytest.approx(
        -(1 / 3) * math.log2(1 / 3) - (2 / 3) * math.log2(2 / 3)
    )
    assert values["identifier_length_mean"] == 1.0
    assert values["identifier_length_std"] == 0.0
    assert values["keyword_density"] == pytest.approx(4 / 11)
    assert values["operator_density"] == pytest.approx(1 / 11)
    assert values["literal_density"] == pytest.approx(3 / 11)
    assert values["ast_depth"] == 4.0
    assert values["branch_density"] == pytest.approx(1 / 4)
    assert values["loop_density"] == 0.0
    assert values["function_count"] == 1.0
    assert values["statement_density"] == 1.0
    assert values["cyclomatic_complexity"] == 2.0
    assert values["return_density"] == pytest.approx(2 / 4)
    assert values["exception_density"] == 0.0
    assert values["blank_line_ratio"] == 0.0
    line_lengths = np.asarray([len(line) for line in code.splitlines()], dtype=np.float64)
    assert values["line_length_mean"] == pytest.approx(float(line_lengths.mean()))
    assert values["line_length_std"] == pytest.approx(float(line_lengths.std()))
    assert values["indentation_entropy"] == 1.5
    assert np.isfinite(result.values).all()


@pytest.mark.parametrize("suffix", ["", "\n("])
def test_python_literal_like_tokens_are_literals_in_ast_and_fallback(suffix: str) -> None:
    code = "a = True; b = False; c = None; d = ..." + suffix
    result = analyze_enhanced(code, "py")
    values = _features(result)

    assert result.parse_ok is (suffix == "")
    # Four identifiers, four assignments, and four literal-like tokens.
    assert values["keyword_density"] == 0.0
    assert values["operator_density"] == pytest.approx(4 / 12)
    assert values["literal_density"] == pytest.approx(4 / 12)


def test_cpp_literal_like_keywords_are_literals() -> None:
    result = analyze_enhanced("auto a=true; auto b=false; auto c=nullptr;", "cpp")
    values = _features(result)

    assert result.parse_ok is True
    assert values["keyword_density"] == pytest.approx(3 / 12)
    assert values["operator_density"] == pytest.approx(3 / 12)
    assert values["literal_density"] == pytest.approx(3 / 12)


def test_java_literal_like_keywords_are_literals() -> None:
    code = "class A { void f() { boolean a=true; Object b=null; boolean c=false; } }"
    values = _features(analyze_enhanced(code, "java"))

    # class/void/boolean/boolean, six identifiers, three assignments, three literals.
    assert values["keyword_density"] == pytest.approx(4 / 16)
    assert values["operator_density"] == pytest.approx(3 / 16)
    assert values["literal_density"] == pytest.approx(3 / 16)


def test_c_macro_like_names_remain_identifiers_not_literals() -> None:
    code = "int true = false; void *p = NULL;"
    values = _features(analyze_enhanced(code, "c"))

    assert values["literal_density"] == 0.0
    assert values["identifier_length_mean"] == pytest.approx((4 + 5 + 1 + 4) / 4)


@pytest.mark.parametrize(
    ("language", "code", "minimum_functions", "minimum_branches", "minimum_loops"),
    [
        ("c", "int f(int x) { while (x) { if (x > 2) return x; x--; } return 0; }", 1, 1, 1),
        ("cpp", "int f(int x) { for (int i=0; i<x; ++i) { if (i) x--; } return x; }", 1, 1, 1),
        ("java", "class A { int f(int x) { while (x > 0) { if (x == 2) break; x--; } return x; } }", 1, 1, 1),
    ],
)
def test_c_family_uses_tree_sitter_and_extracts_structure(
    language: str,
    code: str,
    minimum_functions: int,
    minimum_branches: int,
    minimum_loops: int,
) -> None:
    result = analyze_enhanced(code, language)
    values = _features(result)

    assert result.parse_ok is True
    assert result.backend == "tree-sitter"
    assert result.language == language
    assert values["function_count"] >= minimum_functions
    statement_count = values["statement_density"] * len(code.splitlines())
    assert values["branch_density"] * statement_count >= minimum_branches
    assert values["loop_density"] * statement_count >= minimum_loops
    assert values["ast_depth"] > 0
    assert values["cyclomatic_complexity"] >= 3
    assert np.isfinite(result.values).all()


@pytest.mark.parametrize("language", ["c", "cpp"])
def test_parser_only_trailing_space_normalization_repairs_include_directive(
    language: str,
) -> None:
    raw = (
        "#include <stdio.h>   \n"
        "int f(void) { return 2; }"
    )
    normalized = (
        "#include <stdio.h>\n"
        "int f(void) { return 2; }"
    )
    raw_result = analyze_enhanced(raw, language)
    normalized_result = analyze_enhanced(normalized, language)

    assert raw_result.parse_ok is True
    assert normalized_result.parse_ok is True
    np.testing.assert_allclose(raw_result.values[:15], normalized_result.values[:15])
    assert _features(raw_result)["line_length_mean"] == pytest.approx(
        np.mean([len(line) for line in raw.splitlines()])
    )
    assert _features(raw_result)["line_length_mean"] > _features(normalized_result)["line_length_mean"]


@pytest.mark.parametrize("language", ["c", "cpp"])
def test_parser_normalization_does_not_repair_macro_splice_with_trailing_spaces(
    language: str,
) -> None:
    invalid = (
        "#define TWICE(x) ((x) + \\   \n"
        "                  (x))\n"
        "int f(void) { return TWICE(2); }"
    )
    valid = (
        "#define TWICE(x) ((x) + \\\n"
        "                  (x))\n"
        "int f(void) { return TWICE(2); }"
    )

    assert analyze_enhanced(invalid, language).parse_ok is False
    assert analyze_enhanced(valid, language).parse_ok is True


def test_java_parser_normalization_preserves_raw_format_features() -> None:
    raw = "class A {   \n    int f() { return 1; }\t  \n}  "
    result = analyze_enhanced(raw, "java")

    assert result.parse_ok is True
    assert _features(result)["line_length_mean"] == pytest.approx(
        np.mean([len(line) for line in raw.splitlines()])
    )


@pytest.mark.parametrize(
    ("language", "code"),
    [
        ("py", "def broken(:\n    if True:\n"),
        ("c", "int main( { if (x) return 1;"),
        ("cpp", "template <typename T int f(T x) { return x;"),
        ("java", "class A { void f( { throw new Error();"),
    ],
)
def test_malformed_code_uses_finite_deterministic_lexical_fallback(
    language: str, code: str
) -> None:
    first = analyze_enhanced(code, language)
    second = analyze_enhanced(code, language)

    assert first.parse_ok is False
    assert first.backend == "lexical-fallback"
    assert first.fallback_reason == "syntax-error"
    assert np.array_equal(first.values, second.values)
    assert first.values.shape == (18,)
    assert first.values.dtype == np.float64
    assert np.isfinite(first.values).all()


@pytest.mark.parametrize(
    ("language", "valid", "malformed"),
    [
        (
            "c",
            "int f(int x) { if (x) { while (x) { return 1; } } }",
            "int f(int x) { if (x) { while (x) { return 1; } }",
        ),
        (
            "cpp",
            "int f(int x) { if (x) { while (x) { return 1; } } }",
            "int f(int x) { if (x) { while (x) { return 1; } }",
        ),
        (
            "java",
            "class A { int f(int x) { if (x > 0) { while (x > 0) { return 1; } } } }",
            "class A { int f(int x) { if (x > 0) { while (x > 0) { return 1; } } }",
        ),
    ],
)
def test_fallback_control_units_match_parsed_structural_denominator(
    language: str, valid: str, malformed: str
) -> None:
    parsed = _features(analyze_enhanced(valid, language))
    fallback_result = analyze_enhanced(malformed, language)
    fallback = _features(fallback_result)

    assert fallback_result.parse_ok is False
    for feature in ("branch_density", "loop_density", "return_density"):
        assert fallback[feature] == pytest.approx(parsed[feature])
        assert 0.0 <= fallback[feature] <= 1.0


def test_fallback_exception_clause_units_match_parsed_denominator_and_are_bounded() -> None:
    valid = (
        "class A { int f() { try { throw new RuntimeException(); } "
        "catch (RuntimeException e) { return 1; } } }"
    )
    malformed = valid[:-1]
    parsed = _features(analyze_enhanced(valid, "java"))
    fallback_result = analyze_enhanced(malformed, "java")
    fallback = _features(fallback_result)

    assert fallback_result.parse_ok is False
    assert fallback["exception_density"] == pytest.approx(parsed["exception_density"])
    assert 0.0 <= fallback["exception_density"] <= 1.0


def test_python_unrelated_punctuation_error_does_not_change_structural_densities() -> None:
    valid = "def f(x):\n    if x:\n        while x:\n            return 1"
    malformed = valid + "\n("
    parsed = _features(analyze_enhanced(valid, "py"))
    fallback_result = analyze_enhanced(malformed, "py")
    fallback = _features(fallback_result)

    assert fallback_result.parse_ok is False
    for feature in ("branch_density", "loop_density", "return_density"):
        assert fallback[feature] == pytest.approx(parsed[feature])
        assert 0.0 <= fallback[feature] <= 1.0


def test_nested_python_conditional_expressions_are_structural_units() -> None:
    result = analyze_enhanced("x = a if b else c if d else e", "py")
    values = _features(result)

    assert result.parse_ok is True
    assert values["branch_density"] == pytest.approx(2 / 3)
    assert 0.0 <= values["branch_density"] <= 1.0


def test_python_tokenizer_fallback_uses_python_not_cpp_keywords() -> None:
    result = analyze_enhanced("x = int(1", "py")
    values = _features(result)

    assert result.parse_ok is False
    assert values["keyword_density"] == 0.0
    assert values["identifier_length_mean"] == pytest.approx((1 + 3) / 2)


def test_python_malformed_fallback_preserves_floor_division_and_hash_in_string() -> None:
    code = 'x = "inside # not comment"\ny = 8 // 2\nz = (1'
    result = analyze_enhanced(code, "py")
    values = _features(result)

    assert result.parse_ok is False
    # Three identifiers, four operators (=, =, //, =), and four literals.
    assert values["keyword_density"] == 0.0
    assert values["operator_density"] == pytest.approx(4 / 11)
    assert values["literal_density"] == pytest.approx(4 / 11)
    assert values["identifier_length_mean"] == 1.0


@pytest.mark.parametrize(
    "prefix",
    ["R", "U", "B", "F", "Br", "rB", "Fr", "rF", "RF", "FR"],
)
def test_python_malformed_fallback_protects_case_insensitive_string_prefixes(
    prefix: str,
) -> None:
    code = f'x = {prefix}"if # hidden" + ('
    result = analyze_enhanced(code, "py")
    values = _features(result)

    assert result.parse_ok is False
    # x, =, +, and one prefixed literal; the prefix and contents add no names.
    assert values["keyword_density"] == 0.0
    assert values["operator_density"] == pytest.approx(2 / 4)
    assert values["literal_density"] == pytest.approx(1 / 4)
    assert values["identifier_length_mean"] == 1.0


def test_cpp_raw_string_is_one_protected_literal_in_fallback() -> None:
    code = 'int broken( { auto s = R"tag(if fake() { return 9; } // x)tag";'
    result = analyze_enhanced(code, "cpp")
    values = _features(result)

    assert result.parse_ok is False
    # int, broken, auto, s, =, and one raw-string literal.
    assert values["keyword_density"] == pytest.approx(2 / 6)
    assert values["operator_density"] == pytest.approx(1 / 6)
    assert values["literal_density"] == pytest.approx(1 / 6)
    assert values["identifier_length_mean"] == pytest.approx((6 + 1) / 2)
    assert values["function_count"] == 0.0


@pytest.mark.parametrize("language", ["c", "cpp"])
@pytest.mark.parametrize(
    "literal",
    [
        'L"fake() {"',
        'u"fake() {"',
        'U"fake() {"',
        'u8"fake() {"',
        "L'f'",
        "u'f'",
        "U'f'",
        "u8'f'",
    ],
)
def test_c_family_prefixed_ordinary_literals_are_one_protected_token(
    language: str, literal: str
) -> None:
    code = f"int broken( {{ const char *s = {literal};"
    result = analyze_enhanced(code, language)
    values = _features(result)

    assert result.parse_ok is False
    # int, broken, const, char, *, s, =, and one prefixed literal.
    assert values["keyword_density"] == pytest.approx(3 / 8)
    assert values["operator_density"] == pytest.approx(2 / 8)
    assert values["literal_density"] == pytest.approx(1 / 8)
    assert values["identifier_length_mean"] == pytest.approx((6 + 1) / 2)
    assert values["function_count"] == 0.0


@pytest.mark.parametrize("prefix", ["u8R", "uR", "UR", "LR"])
def test_cpp_prefixed_raw_literals_are_one_protected_token(prefix: str) -> None:
    code = f'int broken( {{ auto s = {prefix}"tag(fake() {{ return 1; }})tag";'
    result = analyze_enhanced(code, "cpp")
    values = _features(result)

    assert result.parse_ok is False
    assert values["keyword_density"] == pytest.approx(2 / 6)
    assert values["operator_density"] == pytest.approx(1 / 6)
    assert values["literal_density"] == pytest.approx(1 / 6)
    assert values["identifier_length_mean"] == pytest.approx((6 + 1) / 2)
    assert values["function_count"] == 0.0


def test_java_text_block_is_one_protected_literal_in_fallback() -> None:
    code = 'class Broken { void x( { String s = """\nfake() { if (x) return; }\n""";'
    result = analyze_enhanced(code, "java")
    values = _features(result)

    assert result.parse_ok is False
    # class, Broken, void, x, String, s, =, and one text-block literal.
    assert values["keyword_density"] == pytest.approx(2 / 8)
    assert values["operator_density"] == pytest.approx(1 / 8)
    assert values["literal_density"] == pytest.approx(1 / 8)
    assert values["identifier_length_mean"] == pytest.approx((6 + 1 + 6 + 1) / 4)
    assert values["function_count"] == 0.0


def test_fallback_function_count_uses_protected_tokens_not_raw_text() -> None:
    code = (
        'int real() { return 0; } int broken( { const char *s = "fake() {"; '
        "/* ghost() { */ return 0;"
    )
    result = analyze_enhanced(code, "c")

    assert result.parse_ok is False
    assert _features(result)["function_count"] == 1.0


def test_comments_and_string_contents_are_not_counted_as_code_tokens() -> None:
    code = (
        'int main() { const char *s = "if + while 99"; '
        "// for fake + 2\n"
        "return 0; }"
    )
    result = analyze_enhanced(code, "c")
    values = _features(result)

    # Effective lexical items are four keywords, two identifiers, two operators,
    # and two literals. Text inside the string/comment is never re-tokenized.
    assert values["keyword_density"] == pytest.approx(4 / 10)
    assert values["operator_density"] == pytest.approx(2 / 10)
    assert values["literal_density"] == pytest.approx(2 / 10)
    assert values["identifier_length_mean"] == pytest.approx((4 + 1) / 2)


def test_formatting_denominators_ignore_blank_lines_for_length_and_indent() -> None:
    code = "x\n\n    yy\n"
    result = analyze_enhanced(code, "py")
    values = _features(result)

    assert values["blank_line_ratio"] == pytest.approx(1 / 3)
    assert values["line_length_mean"] == pytest.approx((1 + 6) / 2)
    assert values["line_length_std"] == pytest.approx(2.5)
    assert values["indentation_entropy"] == 1.0


def test_returned_feature_vector_is_read_only() -> None:
    result = analyze_enhanced("x = 1", "py")

    assert result.values.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        result.values[0] = 99.0


@pytest.mark.parametrize(
    ("language", "code"),
    [
        ("py", "  \n# only a comment\n\t"),
        ("c", "  \n/* only a comment */\n\t"),
        ("cpp", "// only a comment\n"),
        ("java", "  // only a comment\n"),
    ],
)
def test_comment_or_whitespace_only_code_has_zero_complexity(language: str, code: str) -> None:
    assert _features(analyze_enhanced(code, language))["cyclomatic_complexity"] == 0.0


@pytest.mark.parametrize(("language", "code"), [("py", '"doc"'), ("c", ";")])
def test_lexical_or_structural_non_comment_code_has_base_complexity_one(
    language: str, code: str
) -> None:
    assert _features(analyze_enhanced(code, language))["cyclomatic_complexity"] == 1.0


def test_tree_sitter_backends_are_offline_wheels_without_language_pack() -> None:
    assert importlib.util.find_spec("tree_sitter_language_pack") is None
    for language, code in (
        ("c", "int f(void) { return 0; }"),
        ("cpp", "int f() { return 0; }"),
        ("java", "class A { int f() { return 0; } }"),
    ):
        assert analyze_enhanced(code, language).backend == "tree-sitter"


@pytest.mark.parametrize("value", [None, 1, b"x", ["x"]])
def test_rejects_non_string_code(value: object) -> None:
    with pytest.raises(TypeError, match="code must be a string"):
        analyze_enhanced(value, "py")  # type: ignore[arg-type]


@pytest.mark.parametrize("language", ["python", "C", "javascript", "", None])
def test_rejects_unsupported_language(language: object) -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        analyze_enhanced("x = 1", language)  # type: ignore[arg-type]
