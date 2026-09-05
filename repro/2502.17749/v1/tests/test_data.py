import json

import pytest

from lpcode_v1.data import load_jsonl, normalize_record


def test_task1_negative_uses_human_file_name() -> None:
    row = {"file_name": "pair", "human_file_name": "human.c", "label": 0}
    assert normalize_record(row, task="task1")["human_source_id"] == "human.c"


def test_task1_positive_and_task2_use_file_name() -> None:
    row = {"file_name": "human.c", "label": 1}
    assert normalize_record(row, task="task1")["human_source_id"] == "human.c"
    assert normalize_record(row, task="task2")["human_source_id"] == "human.c"


def test_load_jsonl_normalizes_complete_record(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    row = {
        "file_name": "human.py",
        "human_src": "x = 1",
        "llm_src": "x=1",
        "label": 1,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_jsonl(path, task="task1")[0]["human_source_id"] == "human.py"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("\n", "blank line"),
        ("not-json\n", "invalid JSON"),
        ('{"file_name":"x","label":1}\n', "missing fields"),
    ],
)
def test_load_jsonl_rejects_invalid_rows(tmp_path, content: str, message: str) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_jsonl(path, task="task1")


def test_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unsupported task"):
        normalize_record({"file_name": "x", "label": 1}, task="task3")
