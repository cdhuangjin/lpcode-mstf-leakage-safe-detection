"""Strict LPcode JSONL loading and canonical source identities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TASKS = {"task1", "task2"}
REQUIRED_CONTENT_FIELDS = {"human_src", "llm_src", "label"}


def normalize_record(row: Mapping[str, object], task: str) -> dict[str, object]:
    """Return a record with its leakage-control human source identifier."""
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    try:
        label = int(row["label"])
        if task == "task1" and label == 0:
            source_id = str(row["human_file_name"])
        else:
            source_id = str(row["file_name"])
    except KeyError as exc:
        raise ValueError(f"missing source-id field: {exc.args[0]}") from exc
    normalized = dict(row)
    normalized["human_source_id"] = source_id
    return normalized


def load_jsonl(path: str | Path, task: str) -> list[dict[str, Any]]:
    """Load and validate every JSON object in an LPcode data file."""
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    records: list[dict[str, Any]] = []
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line at {source}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSON row is not an object at {source}:{line_number}")
            missing = REQUIRED_CONTENT_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"missing fields at {source}:{line_number}: {', '.join(sorted(missing))}"
                )
            try:
                records.append(normalize_record(row, task))
            except ValueError as exc:
                raise ValueError(f"{exc} at {source}:{line_number}") from exc
    return records
