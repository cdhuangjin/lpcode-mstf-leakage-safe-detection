"""Deterministic grouped cross-validation without human-source leakage."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def assert_no_group_leakage(
    rows: Sequence[dict[str, object]],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    """Raise when a human source occurs on both sides of a split."""
    train_groups = {str(rows[int(index)]["human_source_id"]) for index in train_idx}
    test_groups = {str(rows[int(index)]["human_source_id"]) for index in test_idx}
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"human-source leakage: {sorted(overlap)[:3]}")


def grouped_folds(
    rows: Sequence[dict[str, object]],
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic stratified folds grouped by human source."""
    labels = np.asarray([int(row["label"]) for row in rows])
    groups = np.asarray([str(row["human_source_id"]) for row in rows])
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = list(splitter.split(np.zeros(len(rows)), labels, groups))
    for train_idx, test_idx in folds:
        assert_no_group_leakage(rows, train_idx, test_idx)
    return folds
