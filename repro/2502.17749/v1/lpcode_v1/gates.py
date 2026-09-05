"""Task 1 descriptive summaries and the pre-registered Gate A verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .paths import RESULTS_ROOT, resolve_output_path


SUMMARY_SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 250217749
BOOTSTRAP_REPLICATES = 10_000
GATE_LANGUAGES = ("c", "cpp", "java", "py")
TRANSITION_ORDER = ("delta", "concat_delta", "full")
MODEL_ORDER = ("mlp", "xgb")
LEGACY_PROTOCOL_VERSION = "human-source-grouped-v1"
LEGACY_LEAKAGE_DEFINITION = {
    "version": "single-human-source-id-v1",
    "formula": "leakage_count = size(train human_source_id intersect test human_source_id)",
    "train_test_disjoint_on": ["human_source_id"],
    "required_zero_fields": ["leakage_count"],
}
STRICT_LEAKAGE_DEFINITION = {
    "version": "dual-endpoint-exact-code-v2",
    "formula": "leakage_count = endpoint_leakage_count + content_leakage_count",
    "train_test_disjoint_on": ["origin_endpoint_id", "exact_code_sha256"],
    "negative_pair_constraint": "human_component_id != candidate_component_id",
    "required_zero_fields": [
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
    ],
}


def _strict_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("delta must be a finite non-boolean number")
    numeric = float(value)
    if not -1.0 <= numeric <= 1.0:
        raise ValueError("delta must be expressed as an F1 fraction in [-1, 1]")
    return numeric


def gate_a(language_deltas: Mapping[str, float]) -> dict[str, Any]:
    """Apply the strict and relaxed Gate A rules to four fractional F1 deltas."""
    if not isinstance(language_deltas, Mapping) or set(language_deltas) != set(GATE_LANGUAGES):
        raise ValueError("Gate A requires exactly c, cpp, java, and py deltas")
    deltas = {language: _strict_number(language_deltas[language]) for language in GATE_LANGUAGES}
    mean = float(np.mean(list(deltas.values())))
    worst = min(deltas.values())
    won = sum(value >= 0.0 for value in deltas.values())
    strict = {"mean_delta_f1": -0.005, "minimum_languages_won": 3, "worst_language_delta_f1": -0.01}
    relaxed = {"mean_delta_f1": -0.01, "minimum_languages_won": 3, "worst_language_delta_f1": -0.015}
    return {
        "passed": mean >= strict["mean_delta_f1"] and won >= strict["minimum_languages_won"] and worst >= strict["worst_language_delta_f1"],
        "relaxed_passed": mean >= relaxed["mean_delta_f1"] and won >= relaxed["minimum_languages_won"] and worst >= relaxed["worst_language_delta_f1"],
        "mean_delta_f1": mean,
        "worst_language_delta_f1": worst,
        "languages_won": won,
        "language_deltas": deltas,
        "thresholds": {"strict": strict, "relaxed": relaxed},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(contents)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty summary cell")
    return float(np.mean(values)), 0.0 if len(values) == 1 else float(np.std(values, ddof=1))


def _protocol_metadata(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if config["schema_version"] == 2 and config["task"] == "task1_strict_origins":
        return str(config["split_protocol"]), dict(STRICT_LEAKAGE_DEFINITION)
    return LEGACY_PROTOCOL_VERSION, dict(LEGACY_LEAKAGE_DEFINITION)


def _strict_gate_axis_reasons(config: Mapping[str, Any]) -> list[str]:
    if config["schema_version"] != 2:
        return []
    from . import t1_strict

    reasons: list[str] = []
    expected_axes = (
        ("languages", tuple(config["languages"]), t1_strict.LANGUAGES),
        ("seeds", tuple(config["seeds"]), t1_strict.DEFAULT_SEEDS),
        ("representations", tuple(config["representations"]), t1_strict.DEFAULT_REPRESENTATIONS),
        ("models", tuple(config["models"]), t1_strict.DEFAULT_MODELS),
    )
    for name, actual, expected in expected_axes:
        if actual != expected:
            reasons.append(
                f"strict-origin Gate A requires {name}={list(expected)}, got {list(actual)}"
            )
    if config["n_splits"] != 5:
        reasons.append(
            f"strict-origin Gate A requires n_splits=5, got {config['n_splits']}"
        )
    if config["limit_origins"] is not None:
        reasons.append("strict-origin Gate A requires limit_origins=null")
    return reasons


def _cluster_ci(by_seed: dict[int, list[float]]) -> dict[str, Any]:
    seeds = sorted(by_seed)
    if not seeds or any(not values for values in by_seed.values()):
        raise ValueError("invalid paired seed clusters")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = generator.choice(seeds, size=len(seeds), replace=True)
        observations = [value for seed in sampled for value in by_seed[int(seed)]]
        means[index] = float(np.mean(observations))
    return {
        "method": "seed_cluster_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "degenerate_single_seed": len(seeds) == 1,
    }


def _load_completed_run(output: Path) -> tuple[dict[str, Any], dict[tuple[str, str, str, int, int], dict[str, Any]]]:
    config_path, folds_path = output / "config.json", output / "folds.jsonl"
    if not config_path.is_file() or not folds_path.is_file():
        raise ValueError("completed Task 1 run requires config.json and folds.jsonl")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid run config") from exc
    if isinstance(config, dict) and config.get("schema_version") == 1 and config.get("task") == "task1":
        from .t1 import _validate_existing_records, _validate_run_config
    elif isinstance(config, dict) and config.get("schema_version") == 2 and config.get("task") == "task1_strict_origins":
        from .t1_strict import _validate_existing_records, _validate_run_config
    else:
        raise ValueError("unsupported Task 1 run protocol")
    _validate_run_config(config)
    records = _validate_existing_records(folds_path, config)
    expected = set(product(config["languages"], config["representations"], config["models"], config["seeds"], range(config["n_splits"])))
    if set(records) != expected:
        raise ValueError("incomplete completed Task 1 fold matrix")
    if config["schema_version"] == 2:
        required_zero_fields = STRICT_LEAKAGE_DEFINITION["required_zero_fields"]
        if any(
            record[field] != 0
            for record in records.values()
            for field in required_zero_fields
        ):
            raise ValueError("strict-origin leakage or negative component violation is nonzero")
        for language, seed, fold in product(
            config["languages"], config["seeds"], range(config["n_splits"])
        ):
            hashes = {
                (
                    records[(language, representation, model, seed, fold)]["train_index_sha256"],
                    records[(language, representation, model, seed, fold)]["test_index_sha256"],
                )
                for representation, model in product(config["representations"], config["models"])
            }
            if len(hashes) != 1:
                raise ValueError("strict-origin pair split hashes differ across configured methods")
    return config, records


def _cell_summaries(config: dict[str, Any], records: Mapping[tuple[str, str, str, int, int], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = ("f1", "precision", "recall", "auroc", "mcc")
    for language, representation, model in product(config["languages"], config["representations"], config["models"]):
        selected = [records[(language, representation, model, seed, fold)] for seed, fold in product(config["seeds"], range(config["n_splits"]))]
        entry: dict[str, Any] = {"n": len(selected)}
        for metric in metrics:
            mean, std = _mean_std([float(item[metric]) for item in selected])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        result.setdefault(language, {}).setdefault(representation, {})[model] = entry
    return result


def _paired_deltas(config: dict[str, Any], records: Mapping[tuple[str, str, str, int, int], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "concat" not in config["representations"]:
        return result
    for language, representation, model in product(config["languages"], config["representations"], config["models"]):
        if representation == "concat":
            continue
        by_seed: dict[int, list[float]] = {}
        values: list[float] = []
        for seed in config["seeds"]:
            cluster = []
            for fold in range(config["n_splits"]):
                candidate_record = records[(language, representation, model, seed, fold)]
                baseline_record = records[(language, "concat", model, seed, fold)]
                if any(
                    candidate_record[field] != baseline_record[field]
                    for field in ("train_index_sha256", "test_index_sha256")
                ):
                    raise ValueError("paired split hashes differ for transition and concat counterpart")
                candidate = float(candidate_record["f1"])
                baseline = float(baseline_record["f1"])
                cluster.append(candidate - baseline)
            by_seed[seed] = cluster
            values.extend(cluster)
        mean, std = _mean_std(values)
        result.setdefault(language, {}).setdefault(representation, {})[model] = {
            "n": len(values), "mean_delta_f1": mean, "std_delta_f1": std, "ci_95": _cluster_ci(by_seed),
            "paired_keys": [[language, model, seed, fold] for seed, fold in product(config["seeds"], range(config["n_splits"]))],
        }
    return result


def _candidate_ranking(config: dict[str, Any], cells: dict[str, Any], paired: dict[str, Any]) -> list[dict[str, Any]]:
    if "concat" not in config["representations"] or set(config["languages"]) != set(GATE_LANGUAGES):
        return []
    candidates: list[dict[str, Any]] = []
    for representation, model in product(TRANSITION_ORDER, config["models"]):
        if representation not in config["representations"]:
            continue
        language_f1 = {language: cells[language][representation][model]["f1_mean"] for language in GATE_LANGUAGES}
        language_deltas = {language: paired[language][representation][model]["mean_delta_f1"] for language in GATE_LANGUAGES}
        candidates.append({
            "representation": representation, "model": model, "language_f1": language_f1,
            "language_deltas": language_deltas, "macro_f1": float(np.mean(list(language_f1.values()))),
            "macro_paired_delta_f1": float(np.mean(list(language_deltas.values()))),
            "per_seed_macro_deltas": {}, "languages_nonnegative": sum(value >= 0.0 for value in language_deltas.values()),
            "seeds_nonnegative": 0,
        })
    return candidates


def _populate_seed_macro_deltas(candidates: list[dict[str, Any]], config: dict[str, Any], records: Mapping[tuple[str, str, str, int, int], dict[str, Any]]) -> None:
    for candidate in candidates:
        per_seed: dict[str, float] = {}
        for seed in config["seeds"]:
            values = [
                float(records[(language, candidate["representation"], candidate["model"], seed, fold)]["f1"])
                - float(records[(language, "concat", candidate["model"], seed, fold)]["f1"])
                for language, fold in product(GATE_LANGUAGES, range(config["n_splits"]))
            ]
            per_seed[str(seed)] = float(np.mean(values))
        candidate["per_seed_macro_deltas"] = per_seed
        candidate["seeds_nonnegative"] = sum(value >= 0.0 for value in per_seed.values())


def _rank_key(candidate: dict[str, Any]) -> tuple[float, int, int, str]:
    representation = candidate["representation"]
    model = candidate["model"]
    return (-float(candidate["macro_f1"]), TRANSITION_ORDER.index(representation), MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), str(model))


def _table_contents(config: dict[str, Any], cells: dict[str, Any]) -> tuple[str, str]:
    labels = {"c": "C", "cpp": "C++", "java": "Java", "py": "Python"}
    headers = ["Model", "Representation", "C", "C++", "Java", "Python", "Avg"]
    csv_rows = [",".join(headers)]
    markdown_rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for model in config["models"]:
        for representation in config["representations"]:
            values: list[float] = []
            displayed: list[str] = []
            for language in GATE_LANGUAGES:
                if language not in config["languages"]:
                    displayed.append("—")
                    continue
                cell = cells[language][representation][model]
                values.append(float(cell["f1_mean"]))
                displayed.append(f"{100 * cell['f1_mean']:.2f}% ± {100 * cell['f1_std']:.2f}%")
            average = "—" if not values else f"{100 * float(np.mean(values)):.2f}%"
            row = [model.upper(), representation, *displayed, average]
            csv_rows.append(",".join('"' + value.replace('"', '""') + '"' if any(token in value for token in (",", "±")) else value for value in row))
            markdown_rows.append("| " + " | ".join(row) + " |")
    return "\n".join(csv_rows) + "\n", "\n".join(markdown_rows) + "\n"


def _summarize_t1_locked(output: Path) -> dict[str, Any]:
    if not output.is_dir():
        raise ValueError(f"Task 1 output root does not exist: {output}")
    config, records = _load_completed_run(output)
    protocol_version, leakage_definition = _protocol_metadata(config)
    cells = _cell_summaries(config, records)
    paired = _paired_deltas(config, records)
    candidates = _candidate_ranking(config, cells, paired)
    _populate_seed_macro_deltas(candidates, config, records)
    candidates.sort(key=_rank_key)
    selected = {"representation": candidates[0]["representation"], "model": candidates[0]["model"]} if candidates else None
    reasons: list[str] = []
    reasons.extend(_strict_gate_axis_reasons(config))
    if set(config["languages"]) != set(GATE_LANGUAGES):
        reasons.append("official Gate A requires all four configured languages: c, cpp, java, py")
    if "concat" not in config["representations"]:
        reasons.append("Gate A requires concat as the paired baseline")
    if not candidates:
        reasons.append("no configured transition candidate is evaluable")
    if reasons:
        gate: dict[str, Any] = {"schema_version": SUMMARY_SCHEMA_VERSION, "protocol_version": protocol_version, "leakage_definition": leakage_definition, "status": "not_evaluable", "reasons": reasons, "selected_candidate": selected, "strict": None, "relaxed": None, "language_deltas": None, "per_seed_deltas": None, "ci_summary": None}
    else:
        chosen = candidates[0]
        strict = gate_a(chosen["language_deltas"])
        gate = {
            "schema_version": SUMMARY_SCHEMA_VERSION, "protocol_version": protocol_version, "leakage_definition": leakage_definition, "status": "evaluable", "reasons": [], "selected_candidate": selected,
            "strict": strict, "relaxed": {"passed": strict["relaxed_passed"], "thresholds": strict["thresholds"]["relaxed"]},
            "language_deltas": chosen["language_deltas"], "per_seed_deltas": chosen["per_seed_macro_deltas"],
            "ci_summary": {language: paired[language][selected["representation"]][selected["model"]]["ci_95"] for language in GATE_LANGUAGES},
        }
    methodology = {
        "paired_delta": "non-concat minus concat on identical language/model/seed/fold keys",
        "standard_deviation": "sample standard deviation (ddof=1; zero for n=1)",
        "bootstrap": {"method": "seed_cluster_bootstrap", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "selection": "one global candidate by macro F1; ties: delta, concat_delta, full then mlp, xgb",
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION, "protocol_version": protocol_version, "leakage_definition": leakage_definition, "methodology": methodology, "config": {key: config[key] for key in ("config_id", "languages", "seeds", "n_splits", "representations", "models")},
        "cell_summaries": cells, "paired_deltas": paired, "candidate_ranking": candidates,
        "seed_language_consistency": {"languages": config["languages"], "seeds": config["seeds"], "n_splits": config["n_splits"], "complete_cartesian_product": True},
    }
    table_csv, table_md = _table_contents(config, cells)
    _write_json(output / "summary.json", summary)
    _atomic_write(output / "table_a.csv", table_csv.encode("utf-8"))
    _atomic_write(output / "table_a.md", table_md.encode("utf-8"))
    _write_json(output / "gate_a.json", gate)
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "method_versions": {"summary": SUMMARY_SCHEMA_VERSION, "bootstrap": "seed-cluster-v1", "gate_a": "v1", "protocol": protocol_version, "leakage_definition": leakage_definition["version"]},
        "files": {name: {"sha256": _sha256(output / name), "bytes": (output / name).stat().st_size} for name in ("config.json", "folds.jsonl", "summary.json", "table_a.csv", "table_a.md", "gate_a.json")},
    }
    _write_json(output / "manifest.json", manifest)
    return {"output_root": str(output), "summary_path": str(output / "summary.json"), "gate_a_path": str(output / "gate_a.json"), "manifest_path": str(output / "manifest.json"), "selected_candidate": selected, "verdict": gate["status"] if gate["status"] != "evaluable" else gate["strict"]["passed"]}


def summarize_t1(output_root: str | Path) -> dict[str, Any]:
    """Validate one complete Task 1 run and write descriptive, Gate A artifacts."""
    from .t1 import _exclusive_output_lock

    output = resolve_output_path(output_root)
    if not output.is_dir():
        raise ValueError(f"Task 1 output root does not exist: {output}")
    with _exclusive_output_lock(output):
        return _summarize_t1_locked(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT / "01_t1")
    args = parser.parse_args()
    print(json.dumps(summarize_t1(args.output_root), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
