"""Deterministic Task 5 summaries and the pre-registered Gate D."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import t5
from .paths import resolve_output_path
from .t1 import _exclusive_output_lock


SUMMARY_SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 250_217_749
BOOTSTRAP_REPLICATES = 10_000
BASELINE_METHOD = "lpcode_original"
CANDIDATE_METHOD = "mstf"
METRICS = ("f1", "precision", "recall", "auroc", "mcc")


def _strict_delta(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite non-boolean number")
    result = float(value)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return result


def gate_d(
    heldout_mean_deltas: Mapping[str, float],
    overall_equal_language_mean_delta_f1: float,
) -> dict[str, Any]:
    """Apply the frozen cross-language Gate D decision rule."""

    if not isinstance(heldout_mean_deltas, Mapping) or not heldout_mean_deltas:
        raise ValueError("heldout mean deltas must be a non-empty mapping")
    deltas = {
        str(language): _strict_delta(value, f"{language} mean delta")
        for language, value in heldout_mean_deltas.items()
    }
    overall = _strict_delta(
        overall_equal_language_mean_delta_f1, "overall equal-language mean delta"
    )
    positive = sum(value > 0.0 for value in deltas.values())
    mean_002 = overall >= 0.02 or math.isclose(overall, 0.02, rel_tol=0.0, abs_tol=1e-12)
    mean_003 = overall >= 0.03 or math.isclose(overall, 0.03, rel_tol=0.0, abs_tol=1e-12)
    wins = positive >= 3
    return {
        "passed": bool(wins and mean_002),
        "heldout_mean_delta_f1": deltas,
        "positive_holdouts": positive,
        "total_holdouts": len(deltas),
        "at_least_3_of_4_positive": wins,
        "overall_equal_language_mean_delta_f1": overall,
        "mean_delta_at_least_0_02": mean_002,
        "stronger_mean_delta_at_least_0_03": mean_003,
        "thresholds": {
            "positive_holdouts": ">= 3 of 4",
            "overall_equal_language_mean_delta_f1": ">= 0.02",
            "stronger_reporting_threshold": ">= 0.03",
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty T5 cell")
    return float(np.mean(values)), 0.0 if len(values) == 1 else float(np.std(values, ddof=1))


def _cluster_ci(by_seed: Mapping[int, list[float]]) -> dict[str, Any]:
    seeds = sorted(by_seed)
    if not seeds or any(not by_seed[seed] for seed in seeds):
        raise ValueError("invalid T5 paired seed clusters")
    cluster_means = np.asarray(
        [float(np.mean(by_seed[seed])) for seed in seeds], dtype=np.float64
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = generator.integers(0, len(seeds), size=(BOOTSTRAP_REPLICATES, len(seeds)))
    means = np.mean(cluster_means[sampled], axis=1)
    return {
        "method": "seed_cluster_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "degenerate_single_seed": len(seeds) == 1,
        "cluster_unit": "seed",
    }


RecordKey = tuple[str, str, int]


def _load_completed_run(output: Path) -> tuple[dict[str, Any], dict[RecordKey, dict[str, Any]]]:
    config_path = output / "config.json"
    ledger_path = output / "folds.jsonl"
    if not config_path.is_file() or not ledger_path.is_file():
        raise ValueError("completed T5 run requires config.json and folds.jsonl")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid T5 run config") from exc
    t5._validate_config(config)
    records = t5._load_records(ledger_path, config)
    expected = set(product(config["heldout_languages"], config["methods"], config["seeds"]))
    if set(records) != expected:
        raise ValueError("incomplete T5 record matrix")
    for heldout, seed in product(config["heldout_languages"], config["seeds"]):
        signatures = {
            (
                record["train_index_sha256"],
                record["test_index_sha256"],
                json.dumps(record["train_bank_sha256"], sort_keys=True, separators=(",", ":")),
                record["test_bank_sha256"],
            )
            for method in config["methods"]
            for record in (records[(heldout, method, seed)],)
        }
        if len(signatures) != 1:
            raise ValueError("paired T5 hashes differ across methods")
    return config, records


def _gate_axis_reasons(config: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected = (
        ("languages", config["languages"], list(t5.LANGUAGES)),
        ("heldout_languages", config["heldout_languages"], list(t5.LANGUAGES)),
        ("methods", config["methods"], list(t5.METHODS)),
        ("seeds", config["seeds"], list(t5.DEFAULT_SEEDS)),
    )
    for name, actual, required in expected:
        if actual != required:
            reasons.append(f"Gate D requires {name}={required}, got {actual}")
    if config["n_pair_folds"] != t5.T5_PAIR_FOLDS:
        reasons.append(f"Gate D requires n_pair_folds={t5.T5_PAIR_FOLDS}, got {config['n_pair_folds']}")
    if config.get("limit_origins") is not None:
        reasons.append("Gate D requires limit_origins=null")
    if config.get("full_matrix") is not True:
        reasons.append("Gate D requires full_matrix=true")
    configured = len(config["heldout_languages"]) * len(config["methods"]) * len(config["seeds"])
    if configured != 48:
        reasons.append(f"Gate D requires exactly 48 records, configured {configured}")
    return reasons


def _cell_summaries(config: Mapping[str, Any], records: Mapping[RecordKey, Mapping[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for heldout, method in product(config["heldout_languages"], config["methods"]):
        selected = [records[(heldout, method, seed)] for seed in config["seeds"]]
        entry: dict[str, Any] = {"n": len(selected)}
        for metric in METRICS:
            mean, std = _mean_std([float(record[metric]) for record in selected])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        cells.setdefault(heldout, {})[method] = entry
    return cells


def _paired_summaries(
    config: Mapping[str, Any], records: Mapping[RecordKey, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    heldout: dict[str, Any] = {}
    for language in config["heldout_languages"]:
        values = [
            float(records[(language, CANDIDATE_METHOD, seed)]["f1"])
            - float(records[(language, BASELINE_METHOD, seed)]["f1"])
            for seed in config["seeds"]
        ]
        mean, std = _mean_std(values)
        heldout[language] = {
            "n": len(values),
            "mean_delta_f1": mean,
            "std_delta_f1": std,
            "ci_95": _cluster_ci({seed: [value] for seed, value in zip(config["seeds"], values)}),
        }
    by_seed: dict[int, list[float]] = {}
    directions: dict[str, Any] = {}
    counts = {"positive": 0, "negative": 0, "zero": 0}
    for seed in config["seeds"]:
        values = [
            float(records[(language, CANDIDATE_METHOD, seed)]["f1"])
            - float(records[(language, BASELINE_METHOD, seed)]["f1"])
            for language in config["heldout_languages"]
        ]
        by_seed[seed] = values
        mean = float(np.mean(values))
        direction = "positive" if mean > 0 else "negative" if mean < 0 else "zero"
        counts[direction] += 1
        directions[str(seed)] = {
            "equal_language_mean_delta_f1": mean,
            "direction": direction,
            "heldout_languages": len(values),
        }
    all_values = [value for seed in config["seeds"] for value in by_seed[seed]]
    mean, std = _mean_std(all_values)
    return {
        "definition": "mstf minus lpcode_original on identical language-bank and split hashes",
        "aggregation": "equal-weight macro average across heldout programming languages",
        "statistical_claim": "descriptive",
        "by_heldout_language": heldout,
        "overall_equal_language_mean_delta_f1": mean,
        "overall_equal_language_std_delta_f1": std,
        "n_paired_records": len(all_values),
        "ci_95": _cluster_ci(by_seed),
    }, directions, counts


def _display_method(method: str) -> str:
    return {
        "lpcode_original": "LPcodedec Original",
        "xgb_original": "XGB Original",
        "best_transition": "Best Transition",
        "mstf": "MSTF",
    }.get(method, method)


def _table_contents(config: Mapping[str, Any], cells: Mapping[str, Any]) -> tuple[str, str]:
    labels = {"c": "C", "cpp": "C++", "java": "Java", "py": "Python"}
    headers = ["Method", *[labels.get(language, language) for language in config["heldout_languages"]], "Macro Avg"]
    rows: list[list[str]] = []
    for method in config["methods"]:
        values = [
            f"{100 * cells[language][method]['f1_mean']:.2f}% ± {100 * cells[language][method]['f1_std']:.2f}%"
            for language in config["heldout_languages"]
        ]
        macro_values = [cells[language][method]["f1_mean"] for language in config["heldout_languages"]]
        macro_mean, macro_std = _mean_std(macro_values)
        rows.append([_display_method(method), *values, f"{100 * macro_mean:.2f}% ± {100 * macro_std:.2f}%"])
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    markdown = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]
    return csv_buffer.getvalue(), "\n".join(markdown) + "\n"


def _summarize_t5_locked(output: Path) -> dict[str, Any]:
    if not output.is_dir():
        raise ValueError(f"T5 output root does not exist: {output}")
    config, records = _load_completed_run(output)
    reasons = _gate_axis_reasons(config)
    cells = _cell_summaries(config, records)
    paired = directions = direction_counts = None
    if BASELINE_METHOD in config["methods"] and CANDIDATE_METHOD in config["methods"]:
        paired, directions, direction_counts = _paired_summaries(config, records)
    else:
        reasons.append("Gate D requires lpcode_original and mstf methods")
    if reasons:
        gate: dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "not_evaluable",
            "reasons": reasons,
            "comparison": "mstf versus lpcode_original",
            "strict": None,
        }
    else:
        assert paired is not None
        strict = gate_d(
            {
                language: paired["by_heldout_language"][language]["mean_delta_f1"]
                for language in config["heldout_languages"]
            },
            paired["overall_equal_language_mean_delta_f1"],
        )
        gate = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "evaluable",
            "reasons": [],
            "comparison": "mstf versus lpcode_original",
            "strict": strict,
        }
    expected_records = len(config["heldout_languages"]) * len(config["methods"]) * len(config["seeds"])
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config": {
            key: config[key]
            for key in (
                "config_id", "languages", "heldout_languages", "methods", "seeds",
                "n_pair_folds", "limit_origins", "full_matrix", "method_contract",
            )
        },
        "gate_c_binding": config["gate_c_binding"],
        "matrix": {
            "expected_records": expected_records,
            "observed_records": len(records),
            "official_gate_expected_records": 48,
            "complete_cartesian_product": len(records) == expected_records,
        },
        "methodology": {
            "cell_standard_deviation": "sample standard deviation (ddof=1; zero for n=1)",
            "language_aggregation": "equal-weight macro average across heldout programming languages",
            "paired_delta": "mstf minus lpcode_original on identical language-bank and split hashes",
            "bootstrap": {
                "method": "seed_cluster_bootstrap", "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED, "retains": ["heldout_languages"],
            },
        },
        "cell_summaries": cells,
        "paired_mstf_minus_lpcode": paired,
        "direction_consistency": directions,
        "direction_counts": direction_counts,
    }
    table_csv, table_md = _table_contents(config, cells)
    _write_json(output / "summary.json", summary)
    _atomic_write(output / "table_d.csv", table_csv.encode("utf-8"))
    _atomic_write(output / "table_d.md", table_md.encode("utf-8"))
    _write_json(output / "gate_d.json", gate)
    artifact_names = (
        "config.json", "folds.jsonl", "summary.json", "table_d.csv", "table_d.md", "gate_d.json"
    )
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "method_versions": {
            "summary": "t5-summary-v1",
            "bootstrap": "seed-cluster-v1",
            "gate_d": "cross-language-mstf-vs-lpcode-v1",
        },
        "gate_c_binding": config["gate_c_binding"],
        "protocol_binding": {
            key: config[key]
            for key in ("split_protocol", "bank_protocol", "pair_protocol", "component_protocol")
        },
        "files": {
            name: {"sha256": _sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in artifact_names
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "output_root": str(output),
        "summary_path": str(output / "summary.json"),
        "gate_d_path": str(output / "gate_d.json"),
        "manifest_path": str(output / "manifest.json"),
        "verdict": "not_evaluable" if gate["status"] != "evaluable" else gate["strict"]["passed"],
    }


def summarize_t5(output_root: str | Path) -> dict[str, Any]:
    """Validate and summarize one completed T5 ledger under its output lock."""

    output = resolve_output_path(output_root)
    if not output.is_dir():
        raise ValueError(f"T5 output root does not exist: {output}")
    with _exclusive_output_lock(output):
        return _summarize_t5_locked(output)


__all__ = ["BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED", "gate_d", "summarize_t5"]
