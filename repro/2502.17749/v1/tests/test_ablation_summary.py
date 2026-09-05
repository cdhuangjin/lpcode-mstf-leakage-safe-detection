"""Statistical contracts for orthogonal-ablation summaries."""

import pytest

from lpcode_v1.ablation import summarize_contrast


def test_cluster_bootstrap_uses_seed_as_resampling_unit() -> None:
    records = [
        {"environment": "clean", "method": method, "language": "c", "heldout_llm": None, "seed": seed, "fold": fold, "f1": base + (0.01 if method == "A5" else 0.0)}
        for seed, base in ((42, 0.8), (123, 0.7), (2024, 0.9))
        for fold in range(5)
        for method in ("A0", "A5")
    ]

    result = summarize_contrast(records, "A5", "A0", replicates=100, rng_seed=250217749)

    assert result["cluster_unit"] == "seed"
    assert result["n_seeds"] == 3
    assert result["n_folds"] == 15
    assert result["mean_delta_f1"] == pytest.approx(0.01)


def test_contrast_reports_available_secondary_metric_deltas() -> None:
    records = [
        {"environment": "clean", "method": method, "language": "c", "heldout_llm": None, "seed": 42, "fold": 0, "f1": 0.8 + (0.1 if method == "A5" else 0), "precision": 0.7 + (0.1 if method == "A5" else 0)}
        for method in ("A0", "A5")
    ]

    result = summarize_contrast(records, "A5", "A0", replicates=10)

    assert result["metric_deltas"]["precision"]["mean"] == pytest.approx(0.1)
