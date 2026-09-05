"""Deterministic summaries and the pre-registered Gate B for Task 3.

Gate B is deliberately bound to ``mstf - lpcode_original``.  All descriptive
statistics retain the fold ledger, but uncertainty resamples seeds as clusters;
folds, languages, and held-out generators are never treated as independent
datasets by the bootstrap.
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

from . import t3
from .paths import resolve_output_path
from .t1 import _exclusive_output_lock


SUMMARY_SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 250_217_749
BOOTSTRAP_REPLICATES = 10_000
BASELINE_METHOD = "lpcode_original"
CANDIDATE_METHOD = "mstf"
METRICS = ("f1", "precision", "recall", "auroc", "mcc")
LEAKAGE_DEFINITION = {
    "version": "dual-endpoint-exact-code-v2",
    "train_test_disjoint_on": ["origin_endpoint_id", "exact_code_sha256"],
    "heldout_source_rule": "train excludes held-out LLM; test contains only held-out LLM",
    "negative_pair_constraint": "human_component_id != candidate_component_id",
    "required_zero_fields": [
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
    ],
}


def _strict_number(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Gate B delta must be a finite non-boolean number")
    result = float(value)
    if not -1.0 <= result <= 1.0:
        raise ValueError("Gate B delta must be an F1 fraction in [-1, 1]")
    return result


def gate_b(holdout_deltas: Mapping[str, float]) -> dict[str, Any]:
    """Apply the strict and relaxed pre-registered Gate B rules."""

    if not isinstance(holdout_deltas, Mapping) or set(holdout_deltas) != set(
        t3.LLM_SOURCES
    ):
        raise ValueError("Gate B requires exactly the four pre-registered LLM sources")
    deltas = {
        source: _strict_number(holdout_deltas[source]) for source in t3.LLM_SOURCES
    }
    mean = float(np.mean([deltas[source] for source in t3.LLM_SOURCES]))
    won = sum(deltas[source] > 0.0 for source in t3.LLM_SOURCES)
    strict_thresholds = {
        "minimum_positive_holdouts": 3,
        "overall_macro_mean_delta_f1": 0.03,
    }
    relaxed_thresholds = {
        "minimum_positive_holdouts": 3,
        "overall_macro_mean_delta_f1": 0.0,
    }
    passed = won >= 3 and mean >= 0.03
    relaxed = won >= 3 and mean > 0.0
    return {
        "passed": passed,
        "relaxed_passed": relaxed,
        "authorizes_t4": passed,
        "holdouts_won": won,
        "overall_macro_mean_delta_f1": mean,
        "holdout_deltas": deltas,
        "thresholds": {
            "strict": strict_thresholds,
            "relaxed": relaxed_thresholds,
            "positive_holdout_comparison": "> 0",
            "strict_mean_comparison": ">= 0.03",
            "relaxed_mean_comparison": "> 0",
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
        raise ValueError("cannot summarize an empty T3 cell")
    mean = float(np.mean(values))
    std = 0.0 if len(values) == 1 else float(np.std(values, ddof=1))
    return mean, std


def _cluster_ci(by_seed: dict[int, list[float]]) -> dict[str, Any]:
    seeds = sorted(by_seed)
    if not seeds or any(not values for values in by_seed.values()):
        raise ValueError("invalid T3 paired seed clusters")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = generator.choice(seeds, size=len(seeds), replace=True)
        observations = [
            value for sampled_seed in sampled for value in by_seed[int(sampled_seed)]
        ]
        means[index] = float(np.mean(observations))
    return {
        "method": "seed_cluster_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "degenerate_single_seed": len(seeds) == 1,
        "cluster_unit": "seed",
    }


def _validate_gate_binding(config: Mapping[str, Any]) -> None:
    binding = config.get("gate_a_binding")
    required = {
        "gate_a_path",
        "gate_a_sha256",
        "manifest_path",
        "manifest_sha256",
        "strict_config_sha256",
        "strict_config_id",
        "source_jsonl_sha256",
        "protocol_version",
        "strict_passed",
        "selected_candidate",
    }
    is_hash = t3._is_sha256
    valid = isinstance(binding, dict) and set(binding) == required
    if valid:
        selected = binding["selected_candidate"]
        valid = (
            binding["gate_a_path"] == str(t3.DEFAULT_GATE_A_PATH.resolve())
            and binding["manifest_path"]
            == str(t3.DEFAULT_GATE_A_PATH.resolve().with_name("manifest.json"))
            and all(
                is_hash(binding[field])
                for field in (
                    "gate_a_sha256",
                    "manifest_sha256",
                    "strict_config_sha256",
                    "strict_config_id",
                )
            )
            and binding["protocol_version"] == t3.STRICT_GATE_PROTOCOL_VERSION
            and binding["strict_passed"] is True
            and isinstance(selected, dict)
            and set(selected) == {"representation", "model"}
            and selected["representation"]
            in ("concat", "delta", "concat_delta", "full")
            and selected["model"] in ("mlp", "xgb")
            and isinstance(binding["source_jsonl_sha256"], dict)
            and set(binding["source_jsonl_sha256"]) == set(t3.LANGUAGES)
            and all(is_hash(value) for value in binding["source_jsonl_sha256"].values())
            and config["method_contract"]
            == {
                method: t3._method_contract(selected)[method]
                for method in config["methods"]
            }
            and all(
                config["source_jsonl_sha256"][language]
                == binding["source_jsonl_sha256"][language]
                for language in config["languages"]
            )
        )
    if not valid:
        raise ValueError("invalid or non-passing strict Gate A binding")


def _load_completed_run(
    output: Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str, str, int, int], dict[str, Any]],
]:
    config_path = output / "config.json"
    folds_path = output / "folds.jsonl"
    if not config_path.is_file() or not folds_path.is_file():
        raise ValueError("completed T3 run requires config.json and folds.jsonl")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid T3 run config") from exc
    t3._validate_t3_config(config)
    _validate_gate_binding(config)
    records = t3._load_t3_records(folds_path, config)
    expected = set(
        product(
            config["languages"],
            config["heldout_llms"],
            config["methods"],
            config["seeds"],
            range(config["n_splits"]),
        )
    )
    if set(records) != expected:
        raise ValueError("incomplete T3 fold matrix")
    for language, heldout, seed, fold in product(
        config["languages"],
        config["heldout_llms"],
        config["seeds"],
        range(config["n_splits"]),
    ):
        hashes = {
            (
                records[(language, heldout, method, seed, fold)][
                    "train_index_sha256"
                ],
                records[(language, heldout, method, seed, fold)][
                    "test_index_sha256"
                ],
            )
            for method in config["methods"]
        }
        if len(hashes) != 1:
            raise ValueError("T3 split hashes differ across methods")
    return config, records


def _gate_axis_reasons(config: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected = (
        ("languages", config["languages"], list(t3.LANGUAGES)),
        ("heldout_llms", config["heldout_llms"], list(t3.LLM_SOURCES)),
        ("seeds", config["seeds"], list(t3.DEFAULT_SEEDS)),
        ("methods", config["methods"], list(t3.T3_METHODS)),
    )
    for name, actual, required in expected:
        if actual != required:
            reasons.append(f"Gate B requires {name}={required}, got {actual}")
    if config["n_splits"] != 5:
        reasons.append(f"Gate B requires n_splits=5, got {config['n_splits']}")
    if config["limit_origins"] is not None:
        reasons.append("Gate B requires limit_origins=null")
    expected_count = (
        len(t3.LANGUAGES)
        * len(t3.LLM_SOURCES)
        * len(t3.T3_METHODS)
        * len(t3.DEFAULT_SEEDS)
        * 5
    )
    actual_count = (
        len(config["languages"])
        * len(config["heldout_llms"])
        * len(config["methods"])
        * len(config["seeds"])
        * config["n_splits"]
    )
    if actual_count != expected_count:
        reasons.append(
            f"Gate B requires exactly {expected_count} records, configured {actual_count}"
        )
    return reasons


def _cell_summaries(
    config: Mapping[str, Any],
    records: Mapping[tuple[str, str, str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for heldout, language, method in product(
        config["heldout_llms"], config["languages"], config["methods"]
    ):
        selected = [
            records[(language, heldout, method, seed, fold)]
            for seed, fold in product(config["seeds"], range(config["n_splits"]))
        ]
        entry: dict[str, Any] = {"n": len(selected)}
        for metric in METRICS:
            mean, std = _mean_std([float(record[metric]) for record in selected])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        cells.setdefault(heldout, {}).setdefault(language, {})[method] = entry
    return cells


def _macro_language_summaries(
    config: Mapping[str, Any],
    records: Mapping[tuple[str, str, str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for heldout, method in product(config["heldout_llms"], config["methods"]):
        entry: dict[str, Any] = {
            "n_seed_fold_macro_observations": len(config["seeds"])
            * config["n_splits"]
        }
        for metric in METRICS:
            values = [
                float(
                    np.mean(
                        [
                            records[(language, heldout, method, seed, fold)][metric]
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
        summaries.setdefault(heldout, {})[method] = entry
    return summaries


def _paired_summaries(
    config: Mapping[str, Any],
    records: Mapping[tuple[str, str, str, int, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    by_holdout_language: dict[str, Any] = {}
    by_holdout: dict[str, Any] = {}
    directions: dict[str, Any] = {}
    direction_counts = {"negative": 0, "positive": 0, "zero": 0}
    overall_by_seed: dict[int, list[float]] = {
        seed: [] for seed in config["seeds"]
    }
    all_raw: list[float] = []

    for heldout in config["heldout_llms"]:
        holdout_raw: list[float] = []
        holdout_by_seed: dict[int, list[float]] = {
            seed: [] for seed in config["seeds"]
        }
        for language in config["languages"]:
            values: list[float] = []
            by_seed: dict[int, list[float]] = {}
            for seed in config["seeds"]:
                cluster: list[float] = []
                for fold in range(config["n_splits"]):
                    candidate = records[
                        (language, heldout, CANDIDATE_METHOD, seed, fold)
                    ]
                    baseline = records[
                        (language, heldout, BASELINE_METHOD, seed, fold)
                    ]
                    if any(
                        candidate[field] != baseline[field]
                        for field in ("train_index_sha256", "test_index_sha256")
                    ):
                        raise ValueError(
                            "paired MSTF and LPcodedec split hashes differ"
                        )
                    cluster.append(float(candidate["f1"]) - float(baseline["f1"]))
                by_seed[seed] = cluster
                values.extend(cluster)
                seed_mean = float(np.mean(cluster))
                if seed_mean > 0.0:
                    direction = "positive"
                elif seed_mean < 0.0:
                    direction = "negative"
                else:
                    direction = "zero"
                direction_counts[direction] += 1
                directions.setdefault(heldout, {}).setdefault(language, {})[
                    str(seed)
                ] = {
                    "mean_delta_f1": seed_mean,
                    "direction": direction,
                    "folds": config["n_splits"],
                }
            mean, std = _mean_std(values)
            by_holdout_language.setdefault(heldout, {})[language] = {
                "n": len(values),
                "mean_delta_f1": mean,
                "std_delta_f1": std,
                "ci_95": _cluster_ci(by_seed),
            }
            holdout_raw.extend(values)

        for seed in config["seeds"]:
            for fold in range(config["n_splits"]):
                macro_value = float(
                    np.mean(
                        [
                            float(
                                records[
                                    (
                                        language,
                                        heldout,
                                        CANDIDATE_METHOD,
                                        seed,
                                        fold,
                                    )
                                ]["f1"]
                            )
                            - float(
                                records[
                                    (
                                        language,
                                        heldout,
                                        BASELINE_METHOD,
                                        seed,
                                        fold,
                                    )
                                ]["f1"]
                            )
                            for language in config["languages"]
                        ]
                    )
                )
                holdout_by_seed[seed].append(macro_value)
                overall_by_seed[seed].append(macro_value)
        macro_values = [
            value for seed in config["seeds"] for value in holdout_by_seed[seed]
        ]
        macro_mean, macro_std = _mean_std(macro_values)
        by_holdout[heldout] = {
            "n_paired_records": len(holdout_raw),
            "n_macro_observations": len(macro_values),
            "macro_language_mean_delta_f1": macro_mean,
            "macro_language_std_delta_f1": macro_std,
            "ci_95": _cluster_ci(holdout_by_seed),
            "language_means": {
                language: by_holdout_language[heldout][language]["mean_delta_f1"]
                for language in config["languages"]
            },
        }
        all_raw.extend(holdout_raw)

    overall_values = [
        value for seed in config["seeds"] for value in overall_by_seed[seed]
    ]
    overall_mean, overall_std = _mean_std(overall_values)
    paired = {
        "definition": "mstf minus lpcode_original on identical language/heldout_llm/seed/fold splits",
        "by_holdout_language": by_holdout_language,
        "by_holdout": by_holdout,
        "overall": {
            "n_paired_records": len(all_raw),
            "n_macro_observations": len(overall_values),
            "macro_holdout_language_mean_delta_f1": overall_mean,
            "macro_holdout_language_std_delta_f1": overall_std,
            "ci_95": _cluster_ci(overall_by_seed),
        },
    }
    return paired, directions, direction_counts


def _display_method(method: str) -> str:
    return {
        "lpcode_original": "LPcodedec Original",
        "xgb_original": "XGB Original",
        "best_transition": "Best Transition",
        "mstf": "MSTF",
    }[method]


def _table_contents(
    config: Mapping[str, Any],
    cells: Mapping[str, Any],
    macro: Mapping[str, Any],
) -> tuple[str, str]:
    language_labels = {"c": "C", "cpp": "C++", "java": "Java", "py": "Python"}
    headers = [
        "Held-out LLM",
        "Method",
        *[language_labels[language] for language in config["languages"]],
        "Macro Avg",
    ]
    rows: list[list[str]] = []
    for heldout, method in product(config["heldout_llms"], config["methods"]):
        values = [
            f"{100 * cells[heldout][language][method]['f1_mean']:.2f}% ± "
            f"{100 * cells[heldout][language][method]['f1_std']:.2f}%"
            for language in config["languages"]
        ]
        macro_cell = macro[heldout][method]
        rows.append(
            [
                heldout,
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


def _summarize_t3_locked(output: Path) -> dict[str, Any]:
    if not output.is_dir():
        raise ValueError(f"T3 output root does not exist: {output}")
    config, records = _load_completed_run(output)
    reasons = _gate_axis_reasons(config)
    cells = _cell_summaries(config, records)
    macro = _macro_language_summaries(config, records)

    paired: dict[str, Any] | None = None
    directions: dict[str, Any] | None = None
    direction_counts: dict[str, int] | None = None
    if BASELINE_METHOD in config["methods"] and CANDIDATE_METHOD in config["methods"]:
        paired, directions, direction_counts = _paired_summaries(config, records)
    else:
        reasons.append("Gate B requires lpcode_original and mstf methods")

    if reasons:
        gate: dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "not_evaluable",
            "reasons": reasons,
            "comparison": "mstf - lpcode_original",
            "strict": None,
            "relaxed": None,
        }
    else:
        assert paired is not None
        holdout_deltas = {
            source: paired["by_holdout"][source][
                "macro_language_mean_delta_f1"
            ]
            for source in t3.LLM_SOURCES
        }
        verdict = gate_b(holdout_deltas)
        gate = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "evaluable",
            "reasons": [],
            "comparison": "mstf - lpcode_original",
            "strict": verdict,
            "relaxed": {
                "passed": verdict["relaxed_passed"],
                "authorizes_t4": False,
                "thresholds": verdict["thresholds"]["relaxed"],
            },
        }

    expected_records = (
        len(config["languages"])
        * len(config["heldout_llms"])
        * len(config["methods"])
        * len(config["seeds"])
        * config["n_splits"]
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config": {
            key: config[key]
            for key in (
                "config_id",
                "languages",
                "heldout_llms",
                "seeds",
                "n_splits",
                "methods",
                "limit_origins",
                "split_protocol",
                "pair_protocol",
                "component_protocol",
                "cache_content_sha256",
            )
        },
        "gate_a_binding": {
            key: config["gate_a_binding"][key]
            for key in (
                "gate_a_sha256",
                "manifest_sha256",
                "strict_config_sha256",
                "strict_config_id",
                "protocol_version",
                "selected_candidate",
            )
        },
        "matrix": {
            "expected_records": expected_records,
            "observed_records": len(records),
            "official_gate_expected_records": 960,
            "complete_cartesian_product": len(records) == expected_records,
        },
        "methodology": {
            "cell_standard_deviation": "sample standard deviation (ddof=1; zero for n=1)",
            "language_aggregation": "equal-weight macro average across configured languages at each seed/fold",
            "paired_delta": "mstf minus lpcode_original on identical train/test hashes",
            "bootstrap": {
                "method": "seed_cluster_bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "retains": ["folds", "languages", "heldout_llms"],
            },
        },
        "leakage_definition": LEAKAGE_DEFINITION,
        "cell_summaries": cells,
        "macro_language_summaries": macro,
        "paired_mstf_minus_lpcode": paired,
        "direction_consistency": directions,
        "direction_counts": direction_counts,
    }
    table_csv, table_md = _table_contents(config, cells, macro)
    _write_json(output / "summary.json", summary)
    _atomic_write(output / "table_b.csv", table_csv.encode("utf-8"))
    _atomic_write(output / "table_b.md", table_md.encode("utf-8"))
    _write_json(output / "gate_b.json", gate)
    artifact_names = (
        "config.json",
        "folds.jsonl",
        "summary.json",
        "table_b.csv",
        "table_b.md",
        "gate_b.json",
    )
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "method_versions": {
            "summary": "t3-summary-v1",
            "bootstrap": "seed-cluster-v1",
            "gate_b": "mstf-vs-lpcode-v1",
        },
        "protocol": {
            "split": config["split_protocol"],
            "pair": config["pair_protocol"],
            "component": config["component_protocol"],
        },
        "leakage": LEAKAGE_DEFINITION,
        "gate_a_binding": summary["gate_a_binding"],
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
        "gate_b_path": str(output / "gate_b.json"),
        "manifest_path": str(output / "manifest.json"),
        "verdict": (
            "not_evaluable"
            if gate["status"] != "evaluable"
            else gate["strict"]["passed"]
        ),
    }


def summarize_t3(output_root: str | Path) -> dict[str, Any]:
    """Validate and summarize one completed T3 ledger under its output lock."""

    output = resolve_output_path(output_root)
    if not output.is_dir():
        raise ValueError(f"T3 output root does not exist: {output}")
    with _exclusive_output_lock(output):
        return _summarize_t3_locked(output)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "LEAKAGE_DEFINITION",
    "gate_b",
    "summarize_t3",
]
