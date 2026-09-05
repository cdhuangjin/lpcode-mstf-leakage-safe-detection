from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from lpcode_v1 import attacks
from lpcode_v1.attacks import ATTACKS, AttackResult, apply_attack


SAMPLES = {
    "c": (
        "#define KEEP_NAME(x) ((x) + 1)\\\n"
        "    + 0\n"
        "int external = 2; // remove me\n"
        "int f(int value) {\n"
        "    int local = value; /* remove me too */\n"
        '    const char *text = "// local value";\n'
        "    return local + external;   \n"
        "}\n"
    ),
    "cpp": (
        "#define KEEP_NAME(x) ((x) + 1)\n"
        "struct Box { int field; }; // remove me\n"
        "int f(int value) {\n"
        "    int local = value;\n"
        '    const char *text = R"tag(/* local */)tag";\n'
        "    Box object{local};\n"
        "    return object.field + local;   \n"
        "}\n"
    ),
    "java": (
        "class Demo {\n"
        "    int field; // remove me\n"
        "    int f(int value) {\n"
        "        int local = value;\n"
        '        String text = "// local value";\n'
        "        return this.field + local;   \n"
        "    }\n"
        "}\n"
    ),
    "py": (
        "external = 2  # keep the declaration external\n"
        "def f(value):\n"
        "    local = value  # remove me\n"
        '    text = "# local value"\n'
        "    return holder.field + local + external   \n"
    ),
}


def test_public_attack_contract_is_closed_and_frozen() -> None:
    assert ATTACKS == (
        "comment_removal",
        "identifier_rename",
        "format_normalization",
        "comment_injection",
        "combined",
    )
    assert attacks.LANGUAGES == ("c", "cpp", "java", "py")
    assert isinstance(attacks.ATTACK_VERSION, str) and attacks.ATTACK_VERSION

    result = apply_attack("x = 1\n", "py", "format_normalization")
    assert isinstance(result, AttackResult)
    with pytest.raises(FrozenInstanceError):
        result.code = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("language", attacks.LANGUAGES)
@pytest.mark.parametrize("attack", ATTACKS)
def test_every_attack_is_deterministic_and_records_complete_audit(
    language: str, attack: str
) -> None:
    code = SAMPLES[language]

    first = apply_attack(code, language, attack)
    second = apply_attack(code, language, attack)

    assert first == second
    assert first.attack == attack
    assert first.language == language
    assert first.input_sha256 == hashlib.sha256(code.encode("utf-8")).hexdigest()
    assert first.output_sha256 == hashlib.sha256(first.code.encode("utf-8")).hexdigest()
    assert first.changed is (first.code != code)
    assert type(first.transform_count) is int and first.transform_count >= 0
    assert first.parse_ok_before is True
    assert first.parse_ok_after is True
    assert first.backend_before == ("python-ast" if language == "py" else "tree-sitter")
    assert first.backend_after == first.backend_before
    assert first.failure_reason is None


@pytest.mark.parametrize(
    ("language", "code", "protected"),
    [
        (
            "c",
            'int f(void) { const char *s = "// not a comment"; /* gone\ncontinued */ return 0; }\n',
            '"// not a comment"',
        ),
        (
            "cpp",
            'int f() { auto s = R"tag(/* not a comment */)tag"; // gone\nreturn 0; }\n',
            'R"tag(/* not a comment */)tag"',
        ),
        (
            "java",
            'class A { String s = "// not a comment"; /* gone\ncontinued */ }\n',
            '"// not a comment"',
        ),
        (
            "py",
            's = "# not a comment"\ntext = """# still not\na comment"""\n# gone\n',
            '"""# still not\na comment"""',
        ),
    ],
)
def test_comment_removal_preserves_literals_lines_and_token_separation(
    language: str, code: str, protected: str
) -> None:
    result = apply_attack(code, language, "comment_removal")

    assert protected in result.code
    assert result.code.count("\n") == code.count("\n")
    assert result.parse_ok_after is True
    assert result.transform_count >= 1
    if language == "py":
        assert "# gone" not in result.code
    else:
        assert "/* gone" not in result.code and "// gone" not in result.code

    adjacent = apply_attack(
        "value/**/other\n" if language != "py" else "value # gone\nother\n",
        language,
        "comment_removal",
    )
    assert "valueother" not in adjacent.code


@pytest.mark.parametrize("language", attacks.LANGUAGES)
def test_identifier_rename_changes_only_safe_locals_and_parameters(language: str) -> None:
    code = SAMPLES[language]
    result = apply_attack(code, language, "identifier_rename")

    assert result.changed is True
    assert result.transform_count >= 2
    assert result.parse_ok_after is True
    if language in {"c", "cpp"}:
        assert "KEEP_NAME" in result.code
    if language in {"c", "py"}:
        assert "external" in result.code
    protected_text = "/* local */" if language == "cpp" else "local value"
    assert protected_text in result.code  # protected literal/comment text
    if language != "c":
        assert ".field" in result.code
    assert result.code == apply_attack(code, language, "identifier_rename").code

    if language == "py":
        assert "def f(" in result.code
        assert "holder.field" in result.code
    else:
        assert " f(" in result.code
        if language == "cpp":
            assert "object.field" not in result.code  # the local base may be renamed


def test_identifier_rename_handles_nested_c_function_scopes_without_overlap() -> None:
    code = (
        "void outer(int *value) {\n"
        "    void inner(int *value) { int local = value[0]; }\n"
        "    inner(value);\n"
        "}\n"
    )

    result = apply_attack(code, "c", "identifier_rename")

    assert result.parse_ok_before is True
    assert result.parse_ok_after is True
    assert result.failure_reason is None
    assert result.changed is True


@pytest.mark.parametrize(
    ("language", "code", "literal_line"),
    [
        ("py", 'text = """inside   \nstill inside"""\nvalue = 1   \n\n\n', "inside   \n"),
        (
            "cpp",
            'auto s = R"tag(inside   \nstill inside)tag";   \nint value = 1;   \n\n\n',
            "inside   \n",
        ),
        (
            "java",
            'class A { String s = """\ninside   \n""";\nint value = 1;   \n}\n\n\n',
            "inside   \n",
        ),
        ("c", "int f(void) { return 0; }   \n\n\n", "return 0;"),
    ],
)
def test_format_normalization_is_conservative_around_multiline_literals(
    language: str, code: str, literal_line: str
) -> None:
    result = apply_attack(code, language, "format_normalization")

    assert literal_line in result.code
    if "value = 1" in code:
        assert "value = 1;   \n" not in result.code
    assert not result.code.endswith("\n\n")
    assert result.parse_ok_after is True
    assert result.transform_count >= 1


def test_format_normalization_preserves_preprocessor_continuation_verbatim() -> None:
    code = "#define SUM(x) ((x) + \\\n+    1)   \nint f(void) { return SUM(1); }   \n"
    result = apply_attack(code, "c", "format_normalization")

    assert result.code.startswith("#define SUM(x) ((x) + \\\n+    1)   \n")
    assert "return SUM(1); }\n" in result.code
    assert result.parse_ok_after is True


def test_python_comment_injection_preserves_shebang_and_encoding_cookie() -> None:
    code = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nvalue = 1\n"
    result = apply_attack(code, "py", "comment_injection")

    lines = result.code.splitlines()
    assert lines[:2] == code.splitlines()[:2]
    assert lines[2] == attacks.INJECTED_COMMENTS["py"]
    assert result.transform_count == 1
    assert result.parse_ok_after is True


@pytest.mark.parametrize("language", ("c", "cpp", "java"))
def test_c_family_comment_injection_is_a_safe_deterministic_prefix(language: str) -> None:
    code = SAMPLES[language]
    result = apply_attack(code, language, "comment_injection")

    assert result.code.startswith(attacks.INJECTED_COMMENTS[language] + "\n")
    assert result.code.endswith(code)
    assert result.transform_count == 1


@pytest.mark.parametrize("language", attacks.LANGUAGES)
def test_combined_has_frozen_order_and_excludes_injection(language: str) -> None:
    code = SAMPLES[language]
    combined = apply_attack(code, language, "combined")
    removal = apply_attack(code, language, "comment_removal")
    renamed = apply_attack(removal.code, language, "identifier_rename")
    normalized = apply_attack(renamed.code, language, "format_normalization")

    assert combined.code == normalized.code
    assert combined.transform_count == (
        removal.transform_count + renamed.transform_count + normalized.transform_count
    )
    assert attacks.INJECTED_COMMENTS[language] not in combined.code


def test_parse_regression_is_a_declared_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(attacks._TRANSFORMS, "format_normalization", lambda code, language: ("(", 1))

    result = apply_attack("value = 1\n", "py", "format_normalization")

    assert result.parse_ok_before is True
    assert result.parse_ok_after is False
    assert result.backend_before == "python-ast"
    assert result.backend_after == "lexical-fallback"
    assert result.failure_reason == "parse-regression"


def test_original_parse_failure_is_audited_but_not_misreported_as_attack_failure() -> None:
    result = apply_attack("def broken(:\n", "py", "comment_injection")

    assert result.parse_ok_before is False
    assert result.parse_ok_after is False
    assert result.failure_reason is None


@pytest.mark.parametrize("value", [None, 1, b"x", ["x"]])
def test_rejects_non_string_code(value: object) -> None:
    with pytest.raises(TypeError, match="code must be a string"):
        apply_attack(value, "py", "comment_removal")  # type: ignore[arg-type]


@pytest.mark.parametrize("language", ["python", "C", "", None])
def test_rejects_unsupported_language(language: object) -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        apply_attack("x = 1", language, "comment_removal")  # type: ignore[arg-type]


@pytest.mark.parametrize("attack", ["clean", "rename", "", None])
def test_rejects_unsupported_attack(attack: object) -> None:
    with pytest.raises(ValueError, match="unsupported attack"):
        apply_attack("x = 1", "py", attack)  # type: ignore[arg-type]
