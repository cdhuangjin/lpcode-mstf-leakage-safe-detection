"""Bounded end-to-end validation for the LPcode V1 foundation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_jsonl
from .features_official import analyze_code
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .splits import grouped_folds


LANGUAGES = ("c", "cpp", "java", "py")


def _complete_group_sample(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit_per_language must be positive")
    group_sizes = Counter(str(row["human_source_id"]) for row in rows)
    selected: set[str] = set()
    selected_rows = 0
    for row in rows:
        group = str(row["human_source_id"])
        if group not in selected:
            selected.add(group)
            selected_rows += group_sizes[group]
            if selected_rows >= limit and len(selected) >= 3:
                break
    return [row for row in rows if str(row["human_source_id"]) in selected]


def run_smoke(limit_per_language: int = 40) -> dict[str, Any]:
    """Validate loading, grouping, and official feature extraction on bounded data."""
    language_reports: dict[str, dict[str, Any]] = {}
    dataset_root = REPRO_ROOT / "code" / "experiment" / "task1" / "dataset"
    for language in LANGUAGES:
        rows = load_jsonl(dataset_root / f"{language}.jsonl", task="task1")
        sample = _complete_group_sample(rows, limit_per_language)
        folds = grouped_folds(sample, n_splits=3, seed=42)
        matrix = np.vstack(
            [
                np.concatenate(
                    (
                        analyze_code(str(row["human_src"]), language),
                        analyze_code(str(row["llm_src"]), language),
                    )
                )
                for row in sample
            ]
        )
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite official features for language {language}")
        leakage_count = 0
        for train_idx, test_idx in folds:
            train_groups = {str(sample[int(i)]["human_source_id"]) for i in train_idx}
            test_groups = {str(sample[int(i)]["human_source_id"]) for i in test_idx}
            leakage_count += len(train_groups & test_groups)
        language_reports[language] = {
            "rows": len(sample),
            "feature_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "label_counts": {
                str(label): count
                for label, count in sorted(Counter(int(row["label"]) for row in sample).items())
            },
            "unique_sources": len({str(row["human_source_id"]) for row in sample}),
            "folds": len(folds),
            "leakage_count": leakage_count,
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "limit_per_language": limit_per_language,
        "languages": language_reports,
    }


def write_smoke_report(output: str | Path, limit_per_language: int = 40) -> Path:
    """Atomically write a smoke report to a protected output path."""
    destination = resolve_output_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(run_smoke(limit_per_language), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-language", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "00_official_baseline" / "foundation_smoke.json",
    )
    args = parser.parse_args()
    print(write_smoke_report(args.output, args.limit_per_language))


if __name__ == "__main__":
    main()
