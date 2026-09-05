"""Leakage-safe negative-pair sensitivity checks for strict-clean Gate A.

This module is deliberately separate from the frozen Gate A runner.  It uses
the same split builder, features and fixed classifiers, but records a new,
explicit negative-pair mode for supplementary robustness analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .experiment import evaluate_fold
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .representations import build_representation
from .t1_strict import _select_positive_bank, _split_metadata
from .t3 import T1PairSplit, build_t1_pair_splits, load_or_build_enhanced_cache


LANGUAGES = ("c", "cpp", "java", "py")
SEEDS = (42, 123, 2024)
N_SPLITS = 5
MODES = ("current", "random", "hard")
REPRESENTATIONS = {"baseline": "concat", "mstf": "concat_delta"}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _pair_keys(pairs: Sequence[Any], label: int) -> tuple[str, ...]:
    return tuple(pair.pair_sha256 for pair in pairs if pair.label == label)


def pairing_audit(
    variants: Mapping[str, Sequence[T1PairSplit]],
    *,
    classifier_config: Mapping[str, Any],
    feature_hash: str,
) -> dict[str, Any]:
    """Audit that variants change only fold-local negative-pair construction."""

    if set(variants) != {"current", "random", "hard"}:
        raise ValueError("audit requires current, random and hard variants")
    if not isinstance(feature_hash, str) or len(feature_hash) != 64:
        raise ValueError("feature_hash must be a SHA-256 digest")
    reference = variants["current"]
    if not reference:
        raise ValueError("at least one split is required")
    result: dict[str, Any] = {
        "classifier_config_sha256": hashlib.sha256(
            _canonical_json(dict(classifier_config))
        ).hexdigest(),
        "feature_extraction_sha256": feature_hash,
        "same_positive_pairs": True,
        "variants": {},
    }
    for mode, splits in variants.items():
        if len(splits) != len(reference):
            raise ValueError("variants must share fold count")
        mode_pass = True
        for expected, split in zip(reference, splits):
            if split.fold != expected.fold:
                raise ValueError("variants must share fold identities")
            metadata = _split_metadata(split)
            positives_fixed = (
                _pair_keys(expected.train_pairs, 1) == _pair_keys(split.train_pairs, 1)
                and _pair_keys(expected.test_pairs, 1) == _pair_keys(split.test_pairs, 1)
            )
            result["same_positive_pairs"] &= positives_fixed
            no_duplicates = all(
                len({pair.pair_sha256 for pair in pairs}) == len(pairs)
                for pairs in (split.train_pairs, split.test_pairs)
            )
            balanced = all(
                metadata[f"{side}_class_counts"]["0"]
                == metadata[f"{side}_class_counts"]["1"]
                for side in ("train", "test")
            )
            isolated = all(
                metadata[field] == 0
                for field in (
                    "leakage_count",
                    "endpoint_leakage_count",
                    "content_leakage_count",
                    "negative_component_violation_count",
                )
            )
            mode_pass &= positives_fixed and no_duplicates and balanced and isolated
        result["variants"][mode] = {
            "fold_count": len(splits),
            "positives_equal_negatives": mode_pass,
            "endpoint_and_content_isolation": mode_pass,
            "no_duplicate_pairs": mode_pass,
        }
    result["pass"] = bool(result["same_positive_pairs"]) and all(
        item["positives_equal_negatives"]
        for item in result["variants"].values()
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(cache: Any, pairs: Sequence[Any], representation: str) -> tuple[np.ndarray, np.ndarray]:
    human = np.asarray([pair.human_positive_row_idx for pair in pairs], dtype=np.int64)
    candidate = np.asarray([pair.candidate_positive_row_idx for pair in pairs], dtype=np.int64)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    return build_representation(cache.human[human, :10], cache.llm[candidate, :10], representation), labels


def _seed_cluster_ci(deltas: Mapping[int, Sequence[float]], *, replicates: int = 10_000) -> dict[str, Any]:
    seed_means = np.asarray([np.mean(deltas[seed]) for seed in sorted(deltas)], dtype=float)
    rng = np.random.default_rng(250217749)
    sample = np.asarray(
        [np.mean(rng.choice(seed_means, size=len(seed_means), replace=True)) for _ in range(replicates)],
        dtype=float,
    )
    return {
        "method": "seed_cluster_bootstrap",
        "seed": 250217749,
        "replicates": replicates,
        "low": float(np.quantile(sample, 0.025)),
        "high": float(np.quantile(sample, 0.975)),
    }


def _summaries(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[tuple[str, int, int], Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["negative_pair_mode"]), str(record["method"]))][
            (str(record["language"]), int(record["seed"]), int(record["fold"]))
        ] = record
    rows: list[dict[str, Any]] = []
    per_language: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    for mode in MODES:
        baseline = grouped[(mode, "baseline")]
        mstf = grouped[(mode, "mstf")]
        if not baseline or set(baseline) != set(mstf):
            raise ValueError("baseline and MSTF records must be exactly paired")
        keys = sorted(baseline)
        deltas = [float(mstf[key]["f1"]) - float(baseline[key]["f1"]) for key in keys]
        by_seed: dict[int, list[float]] = defaultdict(list)
        for key, delta in zip(keys, deltas):
            by_seed[key[1]].append(delta)
        ci = _seed_cluster_ci(by_seed)
        bootstrap[mode] = ci
        rows.append({
            "negative_pairing": mode,
            "baseline_f1_mean": float(np.mean([float(baseline[key]["f1"]) for key in keys])),
            "baseline_f1_std": float(np.std([float(baseline[key]["f1"]) for key in keys], ddof=1)),
            "mstf_f1_mean": float(np.mean([float(mstf[key]["f1"]) for key in keys])),
            "mstf_f1_std": float(np.std([float(mstf[key]["f1"]) for key in keys], ddof=1)),
            "delta_f1_mean": float(np.mean(deltas)),
            "delta_f1_std": float(np.std(deltas, ddof=1)),
            "ci_95_low": ci["low"],
            "ci_95_high": ci["high"],
            "n_fold_records": len(keys),
        })
        for language in LANGUAGES:
            language_keys = [key for key in keys if key[0] == language]
            per_language.append({
                "negative_pairing": mode,
                "language": language,
                "baseline_f1_mean": float(np.mean([float(baseline[key]["f1"]) for key in language_keys])),
                "mstf_f1_mean": float(np.mean([float(mstf[key]["f1"]) for key in language_keys])),
                "delta_f1_mean": float(np.mean([float(mstf[key]["f1"]) - float(baseline[key]["f1"]) for key in language_keys])),
                "n_fold_records": len(language_keys),
            })
    return rows, per_language, bootstrap


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_negative_pair_robustness(
    output_root: str | Path,
    *,
    frozen_n0_root: str | Path = RESULTS_ROOT / "01_transition_test_strict_origins",
    cache_root: str | Path = RESULTS_ROOT / "01_transition_test_strict_origins" / "cache",
) -> dict[str, Any]:
    """Run N1/N2 only and combine them with frozen N0 Gate-A records."""

    output = resolve_output_path(output_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("negative-pair output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen_root = Path(frozen_n0_root).resolve()
    frozen_config_path = frozen_root / "config.json"
    frozen_folds_path = frozen_root / "folds.jsonl"
    if not frozen_config_path.is_file() or not frozen_folds_path.is_file():
        raise ValueError("frozen N0 artifacts are incomplete")
    frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    frozen_records = [json.loads(line) for line in frozen_folds_path.read_text(encoding="utf-8").splitlines()]
    n0_records = [
        {**record, "negative_pair_mode": "current", "method": method}
        for method, representation in REPRESENTATIONS.items()
        for record in frozen_records
        if record["model"] == "xgb" and record["representation"] == representation
    ]
    expected_n0 = len(LANGUAGES) * len(SEEDS) * N_SPLITS * len(REPRESENTATIONS)
    if len(n0_records) != expected_n0:
        raise ValueError("frozen N0 records do not cover the required Gate-A cells")
    classifier_config = {"model": "xgb", "representations": REPRESENTATIONS}
    feature_hash = str(frozen_config["feature_contract"]["official_feature_contract_sha256"])
    records: list[dict[str, Any]] = list(n0_records)
    audit_cells: dict[str, Any] = {}
    pair_counts: dict[str, int] = {"current": 0, "random": 0, "hard": 0}
    data_paths = {
        language: REPRO_ROOT / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl"
        for language in LANGUAGES
    }
    for language in LANGUAGES:
        cache = load_or_build_enhanced_cache(language, data_paths[language], cache_root, cache_root)
        cache = _select_positive_bank(cache, None)
        for seed in SEEDS:
            variants = {
                mode: build_t1_pair_splits(cache, language=language, n_splits=N_SPLITS, seed=seed, negative_pair_mode=mode)
                for mode in MODES
            }
            audit = pairing_audit(variants, classifier_config=classifier_config, feature_hash=feature_hash)
            audit_cells[f"{language}:{seed}"] = audit
            for mode, splits in variants.items():
                pair_counts[mode] += sum(len(split.train_pairs) + len(split.test_pairs) for split in splits)
                if mode == "current":
                    continue
                for split in splits:
                    metadata = _split_metadata(split)
                    if any(metadata[field] != 0 for field in ("leakage_count", "endpoint_leakage_count", "content_leakage_count", "negative_component_violation_count")):
                        raise ValueError("pair audit failed before evaluation")
                    for method, representation in REPRESENTATIONS.items():
                        x_train, y_train = _matrix(cache, split.train_pairs, representation)
                        x_test, y_test = _matrix(cache, split.test_pairs, representation)
                        metrics = evaluate_fold(x_train, y_train, x_test, y_test, "xgb", seed)
                        records.append({
                            **metrics,
                            "negative_pair_mode": mode,
                            "method": method,
                            "representation": representation,
                            "model": "xgb",
                            "language": language,
                            "seed": seed,
                            "fold": split.fold,
                            "feature_dimensions": int(x_train.shape[1]),
                            **metadata,
                        })
    summary, per_language, bootstrap = _summaries(records)
    config = {
        "task": "negative_pair_robustness_gate_a",
        "modes": list(MODES),
        "languages": list(LANGUAGES),
        "seeds": list(SEEDS),
        "n_splits": N_SPLITS,
        "classifier_config": classifier_config,
        "feature_extraction_sha256": feature_hash,
        "frozen_n0_config_sha256": _sha256(frozen_config_path),
        "frozen_n0_folds_sha256": _sha256(frozen_folds_path),
        "n0_reused_not_rerun": True,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "raw_results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "pairing_audit.json").write_text(json.dumps({"pass": all(value["pass"] for value in audit_cells.values()), "cells": audit_cells, "pair_counts": pair_counts}, indent=2) + "\n", encoding="utf-8")
    (output / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "per_language.csv", per_language)
    return {"output_root": str(output), "pair_counts": pair_counts, "summary": summary, "audit_pass": all(value["pass"] for value in audit_cells.values())}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frozen-n0-root", default=str(RESULTS_ROOT / "01_transition_test_strict_origins"))
    parser.add_argument("--cache-root", default=str(RESULTS_ROOT / "01_transition_test_strict_origins" / "cache"))
    args = parser.parse_args()
    print(json.dumps(run_negative_pair_robustness(args.output_root, frozen_n0_root=args.frozen_n0_root, cache_root=args.cache_root), indent=2))


if __name__ == "__main__":
    main()
