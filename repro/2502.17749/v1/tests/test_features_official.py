import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from lpcode_v1.features_official import FEATURE_NAMES, analyze_code


@pytest.mark.parametrize("task", ["task1", "task2"])
@pytest.mark.parametrize("language", ["c", "cpp", "java", "py"])
def test_matches_official_on_first_pair(task: str, language: str) -> None:
    root = Path(__file__).resolve().parents[2]
    main_path = root / "code" / "experiment" / task / "main.py"
    spec = importlib.util.spec_from_file_location(f"official_{task}", main_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row_path = root / "code" / "experiment" / task / "dataset" / f"{language}.jsonl"
    with row_path.open(encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    for field in ("human_src", "llm_src"):
        expected = module.analyze_code(row[field], language)
        actual = analyze_code(row[field], language)
        np.testing.assert_array_equal(actual, expected)


def test_feature_contract() -> None:
    assert len(FEATURE_NAMES) == 10
    assert analyze_code("", "py").shape == (10,)
