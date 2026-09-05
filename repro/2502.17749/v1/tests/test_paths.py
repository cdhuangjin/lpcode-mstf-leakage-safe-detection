from pathlib import Path

import pytest

from lpcode_v1.paths import OFFICIAL_EXPERIMENT_DIR, resolve_output_path


def test_rejects_output_inside_official_experiment() -> None:
    with pytest.raises(ValueError, match="official experiment tree"):
        resolve_output_path(OFFICIAL_EXPERIMENT_DIR / "task1" / "new.pkl")


def test_accepts_output_under_results(tmp_path: Path) -> None:
    target = tmp_path / "results" / "run.json"
    assert resolve_output_path(target) == target.resolve()
