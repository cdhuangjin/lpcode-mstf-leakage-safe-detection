"""Transition representations used by LPCode V1."""

from __future__ import annotations

from typing import Any

import numpy as np


_REPRESENTATION_NAMES = {"concat", "delta", "concat_delta", "full"}


def _as_matrix(value: Any, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite numeric matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite numeric matrix")
    return matrix


def build_representation(
    human: Any,
    llm: Any,
    name: str,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Build a finite float64 transition representation from two matrices.

    ``human`` and ``llm`` must be matching, finite, two-dimensional numeric
    matrices. ``concat`` lays out ``[human, llm]``; ``delta`` is ``llm-human``;
    ``concat_delta`` lays out ``[human, llm, delta]``; and ``full`` additionally
    appends ``delta / (abs(human) + epsilon)``. A ``ValueError`` is raised for
    invalid inputs, names, epsilon values, or non-finite results.
    """
    if not isinstance(name, str):
        raise ValueError("representation name must be a string")
    if name not in _REPRESENTATION_NAMES:
        raise ValueError(f"unknown representation: {name!r}")

    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("epsilon must be positive and finite") from exc
    if not np.isfinite(epsilon_value) or epsilon_value <= 0:
        raise ValueError("epsilon must be positive and finite")

    human_matrix = _as_matrix(human, "human")
    llm_matrix = _as_matrix(llm, "llm")
    if human_matrix.shape != llm_matrix.shape:
        raise ValueError("human and llm shape mismatch")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if name == "concat":
            result = np.hstack((human_matrix, llm_matrix))
        else:
            delta = llm_matrix - human_matrix
            if name == "delta":
                result = delta
            elif name == "concat_delta":
                result = np.hstack((human_matrix, llm_matrix, delta))
            else:
                relative = delta / (np.abs(human_matrix) + epsilon_value)
                result = np.hstack((human_matrix, llm_matrix, delta, relative))

    result = np.asarray(result, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("representation output must be finite")
    return result
