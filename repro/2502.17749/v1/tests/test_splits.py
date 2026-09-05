import pytest

from lpcode_v1.splits import assert_no_group_leakage, grouped_folds


def test_grouped_folds_never_overlap_sources() -> None:
    rows = [
        {"human_source_id": f"g{i}", "label": label}
        for i in range(12)
        for label in (0, 1)
    ]
    folds = grouped_folds(rows, n_splits=3, seed=42)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        train_groups = {rows[i]["human_source_id"] for i in train_idx}
        test_groups = {rows[i]["human_source_id"] for i in test_idx}
        assert train_groups.isdisjoint(test_groups)


def test_grouped_folds_are_deterministic() -> None:
    rows = [
        {"human_source_id": f"g{i}", "label": i % 2}
        for i in range(20)
    ]
    first = grouped_folds(rows, n_splits=5, seed=123)
    second = grouped_folds(rows, n_splits=5, seed=123)
    assert [test.tolist() for _, test in first] == [test.tolist() for _, test in second]


def test_explicit_leakage_assertion_rejects_overlap() -> None:
    rows = [
        {"human_source_id": "same", "label": 0},
        {"human_source_id": "same", "label": 1},
    ]
    with pytest.raises(AssertionError, match="human-source leakage"):
        assert_no_group_leakage(rows, [0], [1])
