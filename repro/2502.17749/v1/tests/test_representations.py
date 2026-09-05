import numpy as np
import pytest

from lpcode_v1.representations import build_representation


HUMAN = np.array([[1.0, 0.0], [2.0, -4.0]])
LLM = np.array([[3.0, 2.0], [1.0, -2.0]])
DELTA = np.array([[2.0, 2.0], [-1.0, 2.0]])


def test_concat_representation_has_expected_values_and_dtype() -> None:
    result = build_representation(HUMAN, LLM, "concat")
    np.testing.assert_array_equal(result, np.hstack([HUMAN, LLM]))
    assert result.shape == (2, 4)
    assert result.dtype == np.float64


def test_delta_representation() -> None:
    result = build_representation(HUMAN, LLM, "delta")
    np.testing.assert_array_equal(result, DELTA)
    assert result.shape == (2, 2)


def test_concat_delta_representation() -> None:
    result = build_representation(HUMAN, LLM, "concat_delta")
    np.testing.assert_array_equal(result, np.hstack([HUMAN, LLM, DELTA]))
    assert result.shape == (2, 6)


def test_full_representation_includes_relative_delta_with_zero_denominator() -> None:
    result = build_representation(HUMAN, LLM, "full")
    relative = np.array([[2.0, 2e8], [-0.5, 0.5]])
    np.testing.assert_allclose(result, np.hstack([HUMAN, LLM, DELTA, relative]))
    assert result.shape == (2, 8)
    assert result.dtype == np.float64


@pytest.mark.parametrize("name", ["concat", "delta", "concat_delta", "full"])
def test_all_modes_coerce_integer_and_float32_inputs_to_float64(name: str) -> None:
    result = build_representation(HUMAN.astype(np.float32), LLM.astype(np.int32), name)
    assert result.dtype == np.float64


def test_custom_epsilon_changes_relative_delta_columns() -> None:
    result = build_representation(HUMAN, LLM, "full", epsilon=1.0)
    expected_relative = DELTA / (np.abs(HUMAN) + 1.0)
    np.testing.assert_allclose(result[:, -2:], expected_relative)


def test_rejects_unknown_representation_name() -> None:
    with pytest.raises(ValueError, match="unknown representation"):
        build_representation(HUMAN, LLM, "other")


@pytest.mark.parametrize("name", [None, [], {}])
def test_rejects_non_string_representation_name(name) -> None:
    with pytest.raises(ValueError, match="representation name"):
        build_representation(HUMAN, LLM, name)


@pytest.mark.parametrize(
    ("human", "llm", "message"),
    [
        (np.ones((2, 2)), np.ones((3, 2)), "shape mismatch"),
        (np.ones(2), np.ones(2), "two-dimensional"),
    ],
)
def test_rejects_mismatched_or_non_2d_inputs(human, llm, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_representation(human, llm, "delta")


@pytest.mark.parametrize(
    "human,llm",
    [
        (np.array([[np.nan]]), np.array([[1.0]])),
        (np.array([[np.inf]]), np.array([[1.0]])),
        (np.array([[1.0]]), np.array([[np.inf]])),
        ([["not numeric"]], [[1.0]]),
    ],
)
def test_rejects_non_finite_or_non_numeric_inputs(human, llm) -> None:
    with pytest.raises(ValueError, match="finite numeric"):
        build_representation(human, llm, "delta")


@pytest.mark.parametrize("epsilon", [0, -1, np.nan, np.inf, -np.inf])
def test_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        build_representation(HUMAN, LLM, "full", epsilon=epsilon)


def test_rejects_non_finite_output_from_overflow() -> None:
    human = np.array([[-1e308]])
    llm = np.array([[1e308]])
    with pytest.raises(ValueError, match="finite"):
        build_representation(human, llm, "full")
