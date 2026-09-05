"""Deterministic Task 4 summaries and the pre-registered Gate C.

All comparisons are paired on the frozen split and, for attacks, on the exact
attack-success set.  Languages are macro-averaged with equal weight before
seed-cluster bootstrap uncertainty is calculated.
"""

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

from . import t4
from .paths import resolve_output_path
from .t1 import _exclusive_output_lock


SUMMARY_SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 250_217_749
BOOTSTRAP_REPLICATES = 10_000
BASELINE_METHOD = "lpcode_original"
CANDIDATE_METHOD = "mstf"
METRICS = ("f1", "precision", "recall", "auroc", "mcc")


def _strict_f1(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite non-boolean number")
    result = float(value)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return result


def gate_c(
    *,
    candidate_attacked_f1: float,
    baseline_attacked_f1: float,
    candidate_drop: float,
    baseline_drop: float,
) -> dict[str, Any]:
    """Apply the frozen combined-condition Gate C decision rule."""

    candidate = _strict_f1(candidate_attacked_f1, "candidate attacked F1")
    baseline = _strict_f1(baseline_attacked_f1, "baseline attacked F1")
    candidate_loss = _strict_f1(candidate_drop, "candidate drop")
    baseline_loss = _strict_f1(baseline_drop, "baseline drop")
    advantage = candidate - baseline
    higher = candidate > baseline
    advantage_branch = advantage >= 0.05 or math.isclose(
        advantage, 0.05, rel_tol=0.0, abs_tol=1e-12
    )
    reduction_evaluable = baseline_loss > 0.0
    reduction = (
        (baseline_loss - candidate_loss) / abs(baseline_loss)
        if reduction_evaluable
        else None
    )
    reduction_branch = reduction is not None and (
        reduction >= 0.30
        or math.isclose(reduction, 0.30, rel_tol=0.0, abs_tol=1e-12)
    )
    return {
        "passed": bool(higher and (advantage_branch or reduction_branch)),
        "candidate_higher": higher,
        "candidate_attacked_f1": candidate,
        "baseline_attacked_f1": baseline,
        "attacked_f1_advantage": advantage,
        "candidate_drop": candidate_loss,
        "baseline_drop": baseline_loss,
        "relative_drop_branch_evaluable": reduction_evaluable,
        "relative_drop_reduction": reduction,
        "advantage_at_least_0_05": advantage_branch,
        "relative_drop_reduction_at_least_0_30": bool(reduction_branch),
        "dual_criterion": bool(advantage_branch and reduction_branch),
        "thresholds": {
            "candidate_comparison": "> baseline",
            "attacked_f1_advantage": ">= 0.05",
            "relative_drop_reduction": ">= 0.30",
            "baseline_drop_for_reduction_branch": "> 0",
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
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty T4 cell")
    mean = float(np.mean(values))
    std = 0.0 if len(values) == 1 else float(np.std(values, ddof=1))
    return mean, std


def _cluster_ci(by_seed: Mapping[int, list[float]]) -> dict[str, Any]:
    seeds = sorted(by_seed)
    if not seeds or any(not by_seed[seed] for seed in seeds):
        raise ValueError("invalid T4 paired seed clusters")
    lengths = {len(by_seed[seed]) for seed in seeds}
    if len(lengths) != 1:
        raise ValueError("T4 seed clusters must have equal observation counts")
    cluster_means = np.asarray(
        [float(np.mean(by_seed[seed])) for seed in seeds], dtype=np.float64
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = generator.integers(
        0, len(seeds), size=(BOOTSTRAP_REPLICATES, len(seeds))
    )
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


RecordKey = tuple[str, str, int, int, str]


def _load_completed_run(
    output: Path,
) -> tuple[dict[str, Any], dict[RecordKey, dict[str, Any]]]:
    config_path = output / "config.json"
    folds_path = output / "folds.jsonl"
    if not config_path.is_file() or not folds_path.is_file():
        raise ValueError("completed T4 run requires config.json and folds.jsonl")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid T4 run config") from exc
    t4._validate_t4_config(config)
    records = t4._load_t4_records(folds_path, config)
    expected = set(
        product(
            config["languages"],
            config["methods"],
            config["seeds"],
            range(config["n_splits"]),
            config["conditions"],
        )
    )
    if set(records) != expected:
        raise ValueError("incomplete T4 fold matrix")

    for language, method, seed, fold in product(
        config["languages"],
        config["methods"],
        config["seeds"],
        range(config["n_splits"]),
    ):
        pair_hashes = {
            (
                records[(language, method, seed, fold, condition)][
                    "train_index_sha256"
                ],
                records[(language, method, seed, fold, condition)][
                    "test_index_sha256"
                ],
            )
            for condition in config["conditions"]
        }
        if len(pair_hashes) != 1:
            raise ValueError("T4 train/test pair hashes differ across conditions")

    for language, seed, fold, condition in product(
        config["languages"],
        config["seeds"],
        range(config["n_splits"]),
        config["conditions"],
    ):
        success_hashes = {
            records[(language, method, seed, fold, condition)][
                "attack_success_set_sha256"
            ]
            for method in config["methods"]
        }
        if len(success_hashes) != 1:
            raise ValueError("T4 success-set hashes differ across methods")
    return config, records


def _gate_axis_reasons(config: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected = (
        ("languages", config["languages"], list(t4.LANGUAGES)),
        ("methods", config["methods"], list(t4.METHODS)),
        ("seeds", config["seeds"], list(t4.DEFAULT_SEEDS)),
        ("conditions", config["conditions"], list(t4.CONDITIONS)),
    )
    for name, actual, required in expected:
        if actual != required:
            reasons.append(f"Gate C requires {name}={required}, got {actual}")
    if config["n_splits"] != 5:
        reasons.append(f"Gate C requires n_splits=5, got {config['n_splits']}")
    if config.get("full_matrix") is not True:
        reasons.append("Gate C requires full_matrix=true")
    configured = (
        len(config["languages"])
        * len(config["methods"])
        * len(config["seeds"])
        * config["n_splits"]
        * len(config["conditions"])
    )
    if configured != 1440:
        reasons.append(f"Gate C requires exactly 1440 records, configured {configured}")
    return reasons


def _cell_summaries(
    config: Mapping[str, Any], records: Mapping[RecordKey, Mapping[str, Any]]
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for condition, language, method in product(
        config["conditions"], config["languages"], config["methods"]
    ):
        selected = [
            records[(language, method, seed, fold, condition)]
            for seed, fold in product(config["seeds"], range(config["n_splits"]))
        ]
        entry: dict[str, Any] = {
            "n": len(selected),
            "attack_failures": int(
                sum(int(record["attack_failures"]) for record in selected)
            ),
        }
        for metric in METRICS:
            mean, std = _mean_std([float(record[metric]) for record in selected])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        cells.setdefault(condition, {}).setdefault(language, {})[method] = entry
    return cells


def _macro_language_summaries(
    config: Mapping[str, Any], records: Mapping[RecordKey, Mapping[str, Any]]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for condition, method in product(config["conditions"], config["methods"]):
        entry: dict[str, Any] = {
            "n_seed_fold_macro_observations": len(config["seeds"])
            * config["n_splits"]
        }
        for metric in METRICS:
            values = [
                float(
                    np.mean(
                        [
                            records[(language, method, seed, fold, condition)][metric]
                            for language in config["languages"]
                        ]
                    )
                )
                for seed, fold in product(
                    config["seeds"], range(config["n_splits"])
                )
            ]
            mean, std = _mean_std(values)
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        summaries.setdefault(condition, {})[method] = entry
    return summaries


def _check_paired_records(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    for field in (
        "train_index_sha256",
        "test_index_sha256",
        "attack_success_set_sha256",
    ):
        if candidate[field] != baseline[field]:
            raise ValueError("paired MSTF and LPcodedec hashes differ")


def _paired_summaries(
    config: Mapping[str, Any], records: Mapping[RecordKey, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, int]]]:
    by_condition_language: dict[str, Any] = {}
    by_condition: dict[str, Any] = {}
    directions: dict[str, Any] = {}
    direction_counts: dict[str, dict[str, int]] = {}
    for condition in config["conditions"]:
        for language in config["languages"]:
            by_seed: dict[int, list[float]] = {}
            values: list[float] = []
            for seed in config["seeds"]:
                cluster: list[float] = []
                for fold in range(config["n_splits"]):
                    candidate = records[
                        (language, CANDIDATE_METHOD, seed, fold, condition)
                    ]
                    baseline = records[
                        (language, BASELINE_METHOD, seed, fold, condition)
                    ]
                    _check_paired_records(candidate, baseline)
                    cluster.append(float(candidate["f1"]) - float(baseline["f1"]))
                by_seed[seed] = cluster
                values.extend(cluster)
            mean, std = _mean_std(values)
            by_condition_language.setdefault(condition, {})[language] = {
                "n": len(values),
                "mean_delta_f1": mean,
                "std_delta_f1": std,
                "ci_95": _cluster_ci(by_seed),
            }

        macro_by_seed: dict[int, list[float]] = {
            seed: [] for seed in config["seeds"]
        }
        counts = {"negative": 0, "positive": 0, "zero": 0}
        for seed in config["seeds"]:
            for fold in range(config["n_splits"]):
                macro_by_seed[seed].append(
                    float(
                        np.mean(
                            [
                                float(
                                    records[
                                        (
                                            language,
                                            CANDIDATE_METHOD,
                                            seed,
                                            fold,
                                            condition,
                                        )
                                    ]["f1"]
                                )
                                - float(
                                    records[
                                        (
                                            language,
                                            BASELINE_METHOD,
                                            seed,
                                            fold,
                                            condition,
                                        )
                                    ]["f1"]
                                )
                                for language in config["languages"]
                            ]
                        )
                    )
                )
            seed_mean = float(np.mean(macro_by_seed[seed]))
            direction = (
                "positive"
                if seed_mean > 0.0
                else "negative" if seed_mean < 0.0 else "zero"
            )
            counts[direction] += 1
            directions.setdefault(condition, {})[str(seed)] = {
                "mean_delta_f1": seed_mean,
                "direction": direction,
                "folds": config["n_splits"],
            }
        macro_values = [
            value for seed in config["seeds"] for value in macro_by_seed[seed]
        ]
        mean, std = _mean_std(macro_values)
        by_condition[condition] = {
            "n_paired_records": len(config["languages"]) * len(macro_values),
            "n_macro_observations": len(macro_values),
            "macro_language_mean_delta_f1": mean,
            "macro_language_std_delta_f1": std,
            "ci_95": _cluster_ci(macro_by_seed),
            "language_means": {
                language: by_condition_language[condition][language]["mean_delta_f1"]
                for language in config["languages"]
            },
        }
        direction_counts[condition] = counts
    return (
        {
            "definition": "mstf minus lpcode_original on identical split and success-set hashes",
            "by_condition_language": by_condition_language,
            "by_condition": by_condition,
        },
        directions,
        direction_counts,
    )


def _drop_summaries(
    config: Mapping[str, Any], records: Mapping[RecordKey, Mapping[str, Any]]
) -> dict[str, Any]:
    by_condition_language: dict[str, Any] = {}
    by_condition: dict[str, Any] = {}
    attacks = [condition for condition in config["conditions"] if condition != "clean"]
    for condition, method in product(attacks, config["methods"]):
        macro_by_seed: dict[int, list[float]] = {
            seed: [] for seed in config["seeds"]
        }
        for language in config["languages"]:
            by_seed: dict[int, list[float]] = {}
            values: list[float] = []
            for seed in config["seeds"]:
                cluster = [
                    float(
                        records[(language, method, seed, fold, condition)][
                            "clean_reference_f1"
                        ]
                    )
                    - float(records[(language, method, seed, fold, condition)]["f1"])
                    for fold in range(config["n_splits"])
                ]
                by_seed[seed] = cluster
                values.extend(cluster)
            mean, std = _mean_std(values)
            by_condition_language.setdefault(condition, {}).setdefault(language, {})[
                method
            ] = {
                "n": len(values),
                "mean_drop_f1": mean,
                "std_drop_f1": std,
                "ci_95": _cluster_ci(by_seed),
            }
        for seed in config["seeds"]:
            for fold in range(config["n_splits"]):
                macro_by_seed[seed].append(
                    float(
                        np.mean(
                            [
                                float(
                                    records[
                                        (language, method, seed, fold, condition)
                                    ]["clean_reference_f1"]
                                )
                                - float(
                                    records[
                                        (language, method, seed, fold, condition)
                                    ]["f1"]
                                )
                                for language in config["languages"]
                            ]
                        )
                    )
                )
        macro_values = [
            value for seed in config["seeds"] for value in macro_by_seed[seed]
        ]
        mean, std = _mean_std(macro_values)
        by_condition.setdefault(condition, {})[method] = {
            "n_macro_observations": len(macro_values),
            "macro_language_mean_drop_f1": mean,
            "macro_language_std_drop_f1": std,
            "ci_95": _cluster_ci(macro_by_seed),
            "language_means": {
                language: by_condition_language[condition][language][method][
                    "mean_drop_f1"
                ]
                for language in config["languages"]
            },
        }
    return {
        "definition": "clean_reference_f1 minus attacked f1 on each attack-success set",
        "by_condition_language": by_condition_language,
        "by_condition": by_condition,
    }


def _display_method(method: str) -> str:
    return {
        "lpcode_original": "LPcodedec Original",
        "xgb_original": "XGB Original",
        "best_transition": "Best Transition",
        "mstf": "MSTF",
    }.get(method, method)


def _table_contents(
    config: Mapping[str, Any],
    cells: Mapping[str, Any],
    macro: Mapping[str, Any],
) -> tuple[str, str]:
    labels = {"c": "C", "cpp": "C++", "java": "Java", "py": "Python"}
    headers = [
        "Condition",
        "Method",
        *[labels.get(language, language) for language in config["languages"]],
        "Macro Avg",
    ]
    rows: list[list[str]] = []
    for condition, method in product(config["conditions"], config["methods"]):
        values = [
            f"{100 * cells[condition][language][method]['f1_mean']:.2f}% ± "
            f"{100 * cells[condition][language][method]['f1_std']:.2f}%"
            for language in config["languages"]
        ]
        macro_cell = macro[condition][method]
        rows.append(
            [
                condition,
                _display_method(method),
                *values,
                f"{100 * macro_cell['f1_mean']:.2f}% ± "
                f"{100 * macro_cell['f1_std']:.2f}%",
            ]
        )
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


def _summarize_t4_locked(output: Path) -> dict[str, Any]:
    if not output.is_dir():
        raise ValueError(f"T4 output root does not exist: {output}")
    config, records = _load_completed_run(output)
    reasons = _gate_axis_reasons(config)
    cells = _cell_summaries(config, records)
    macro = _macro_language_summaries(config, records)

    paired: dict[str, Any] | None = None
    directions: dict[str, Any] | None = None
    direction_counts: dict[str, dict[str, int]] | None = None
    drops: dict[str, Any] | None = None
    if BASELINE_METHOD in config["methods"] and CANDIDATE_METHOD in config["methods"]:
        paired, directions, direction_counts = _paired_summaries(config, records)
        drops = _drop_summaries(config, records)
    else:
        reasons.append("Gate C requires lpcode_original and mstf methods")

    if reasons:
        gate: dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "not_evaluable",
            "reasons": reasons,
            "condition": "combined",
            "comparison": "mstf versus lpcode_original",
            "strict": None,
        }
    else:
        assert drops is not None
        combined = "combined"
        verdict = gate_c(
            candidate_attacked_f1=macro[combined][CANDIDATE_METHOD]["f1_mean"],
            baseline_attacked_f1=macro[combined][BASELINE_METHOD]["f1_mean"],
            candidate_drop=drops["by_condition"][combined][CANDIDATE_METHOD][
                "macro_language_mean_drop_f1"
            ],
            baseline_drop=drops["by_condition"][combined][BASELINE_METHOD][
                "macro_language_mean_drop_f1"
            ],
        )
        gate = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "evaluable",
            "reasons": [],
            "condition": combined,
            "comparison": "mstf versus lpcode_original",
            "strict": verdict,
        }

    expected_records = (
        len(config["languages"])
        * len(config["methods"])
        * len(config["seeds"])
        * config["n_splits"]
        * len(config["conditions"])
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config": {
            key: config[key]
            for key in (
                "config_id",
                "languages",
                "methods",
                "seeds",
                "n_splits",
                "conditions",
                "full_matrix",
                "method_contract",
            )
        },
        "gate_a_binding": config["gate_a_binding"],
        "gate_b_binding": config["gate_b_binding"],
        "matrix": {
            "expected_records": expected_records,
            "observed_records": len(records),
            "official_gate_expected_records": 1440,
            "complete_cartesian_product": len(records) == expected_records,
        },
        "methodology": {
            "cell_standard_deviation": "sample standard deviation (ddof=1; zero for n=1)",
            "language_aggregation": "equal-weight macro average across configured languages at each seed/fold",
            "paired_delta": "mstf minus lpcode_original on identical split and success-set hashes",
            "clean_to_attack_drop": "record clean_reference_f1 minus attacked f1 on the same success set",
            "bootstrap": {
                "method": "seed_cluster_bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "retains": ["folds", "languages"],
            },
        },
        "cell_summaries": cells,
        "macro_language_summaries": macro,
        "paired_mstf_minus_lpcode": paired,
        "clean_to_attack_drops": drops,
        "direction_consistency": directions,
        "direction_counts": direction_counts,
    }
    table_csv, table_md = _table_contents(config, cells, macro)
    _write_json(output / "summary.json", summary)
    _atomic_write(output / "table_c.csv", table_csv.encode("utf-8"))
    _atomic_write(output / "table_c.md", table_md.encode("utf-8"))
    _write_json(output / "gate_c.json", gate)
    artifact_names = (
        "config.json",
        "folds.jsonl",
        "summary.json",
        "table_c.csv",
        "table_c.md",
        "gate_c.json",
    )
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "method_versions": {
            "summary": "t4-summary-v1",
            "bootstrap": "seed-cluster-v1",
            "gate_c": "combined-mstf-vs-lpcode-v1",
        },
        "gate_a_binding": config["gate_a_binding"],
        "gate_b_binding": config["gate_b_binding"],
        "files": {
            name: {
                "sha256": _sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in artifact_names
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "output_root": str(output),
        "summary_path": str(output / "summary.json"),
        "gate_c_path": str(output / "gate_c.json"),
        "manifest_path": str(output / "manifest.json"),
        "verdict": (
            "not_evaluable"
            if gate["status"] != "evaluable"
            else gate["strict"]["passed"]
        ),
    }


def summarize_t4(output_root: str | Path) -> dict[str, Any]:
    """Validate and summarize one completed T4 ledger under its output lock."""

    output = resolve_output_path(output_root)
    if not output.is_dir():
        raise ValueError(f"T4 output root does not exist: {output}")
    with _exclusive_output_lock(output):
        return _summarize_t4_locked(output)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "gate_c",
    "summarize_t4",
]
