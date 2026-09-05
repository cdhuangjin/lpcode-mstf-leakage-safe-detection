"""Summarise precision, recall and AUROC from a completed robustness ledger."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "negative_pair_robustness" / "raw_results.json"
OUT = ROOT / "results" / "negative_pair_robustness" / "additional_metrics.csv"


def main() -> None:
    records = json.loads(RAW.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for record in records:
        groups[(record["negative_pair_mode"], record["method"])].append(record)
    rows = []
    for mode in ("current", "random", "hard"):
        for metric in ("precision", "recall", "auroc", "mcc"):
            row = {"negative_pairing": mode, "metric": metric}
            for method in ("baseline", "mstf"):
                values = [float(record[metric]) for record in groups[(mode, method)]]
                row[f"{method}_mean"] = sum(values) / len(values)
            rows.append(row)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
