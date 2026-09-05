"""Read-only mechanism analyses derived from frozen formal ledgers."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from .experiment import build_model


class MechanismError(RuntimeError):
    """Raised when a read-only mechanism analysis has incomplete evidence."""


ATTACK_CONDITIONS = (
    "clean",
    "comment_removal",
    "identifier_rename",
    "format_normalization",
    "comment_injection",
    "combined",
)
ATTACK_METHODS = ("lpcode_original", "xgb_original", "best_transition", "mstf")
FEATURE_FAMILIES = ("original_style", "lexical", "structural_syntax", "formatting_layout")
FULL_BLOCKS = ("human_absolute", "candidate_absolute", "delta", "relative_delta")


def full_mstf_feature_groups() -> dict[str, tuple[int, ...]]:
    """Return disjoint, named indices for every 28x4 full-MSTF coordinate."""
    # First 10 official features are original-style. Enhanced dimensions follow
    # a fixed 6 lexical, 8 structural/syntax, and 4 formatting/layout layout.
    family_ranges = {
        "original_style": range(0, 10),
        "lexical": range(10, 16),
        "structural_syntax": range(16, 24),
        "formatting_layout": range(24, 28),
    }
    groups: dict[str, tuple[int, ...]] = {}
    for block_index, block in enumerate(FULL_BLOCKS):
        offset = 28 * block_index
        for family in FEATURE_FAMILIES:
            groups[f"{block}:{family}"] = tuple(offset + index for index in family_ranges[family])
    return groups


def mstf_feature_names() -> tuple[str, ...]:
    """Return stable, human-readable names for the fixed 112-D MSTF vector."""
    base = (
        *(f"original_style_{index:02d}" for index in range(10)),
        *(f"lexical_{index:02d}" for index in range(6)),
        *(f"structural_syntax_{index:02d}" for index in range(8)),
        *(f"formatting_layout_{index:02d}" for index in range(4)),
    )
    return tuple(f"{block}:{feature}" for block in FULL_BLOCKS for feature in base)


def jaccard_top_k(left: list[str], right: list[str], k: int) -> float:
    """Compute Jaccard similarity of bounded ranked feature-name lists."""
    if not isinstance(k, int) or k <= 0:
        raise MechanismError("top-k must be a positive integer")
    left_set, right_set = set(left[:k]), set(right[:k])
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def aggregate_importance(
    gain_folds: list[list[float]],
    permutation_folds: list[list[float]],
    feature_names: tuple[str, ...],
    groups: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Aggregate fold-level gain and held-out permutation importance.

    The function intentionally keeps raw permutation F1 decreases: a negative
    value means shuffling the coordinate improved that held-out fold. Gain is
    normalized within each fitted model before aggregation, so its scale is
    comparable across the environments.
    """
    width = len(feature_names)
    if not gain_folds or len(gain_folds) != len(permutation_folds) or width == 0:
        raise MechanismError("importance folds must be nonempty and paired")
    gain = np.asarray(gain_folds, dtype=float)
    permutation = np.asarray(permutation_folds, dtype=float)
    if gain.shape != permutation.shape or gain.ndim != 2 or gain.shape[1] != width:
        raise MechanismError("importance fold arrays have incompatible dimensions")
    if not np.isfinite(gain).all() or not np.isfinite(permutation).all():
        raise MechanismError("importance values must be finite")
    expected = set(range(width))
    observed = [index for indices in groups.values() for index in indices]
    if sorted(observed) != sorted(expected):
        raise MechanismError("feature groups must partition the feature coordinates")
    totals = gain.sum(axis=1, keepdims=True)
    normalized_gain = np.divide(gain, totals, out=np.zeros_like(gain), where=totals > 0)
    feature_rows = [
        {
            "feature": name,
            "gain_mean": float(normalized_gain[:, index].mean()),
            "gain_sd": float(normalized_gain[:, index].std(ddof=0)),
            "permutation_mean": float(permutation[:, index].mean()),
            "permutation_sd": float(permutation[:, index].std(ddof=0)),
            "n_folds": int(gain.shape[0]),
        }
        for index, name in enumerate(feature_names)
    ]
    group_rows = []
    for group, indices in groups.items():
        group_rows.append(
            {
                "group": group,
                "gain_mean": float(normalized_gain[:, indices].sum(axis=1).mean()),
                "gain_sd": float(normalized_gain[:, indices].sum(axis=1).std(ddof=0)),
                "permutation_mean": float(permutation[:, indices].sum(axis=1).mean()),
                "permutation_sd": float(permutation[:, indices].sum(axis=1).std(ddof=0)),
                "n_folds": int(gain.shape[0]),
            }
        )
    return {"feature_rows": feature_rows, "group_rows": group_rows}


def _xgb_gain(model: Any, width: int) -> list[float]:
    estimator = model.named_steps["model"]
    raw = estimator.get_booster().get_score(importance_type="gain")
    values = np.zeros(width, dtype=float)
    for key, value in raw.items():
        if key.startswith("f") and key[1:].isdigit() and int(key[1:]) < width:
            values[int(key[1:])] = float(value)
    return values.tolist()


def _permutation_f1_decrease(model: Any, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> list[float]:
    baseline = float(f1_score(y_test, model.predict(x_test), zero_division=0))
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for index in range(x_test.shape[1]):
        permuted = x_test.copy()
        permuted[:, index] = generator.permutation(permuted[:, index])
        score = float(f1_score(y_test, model.predict(permuted), zero_division=0))
        values.append(baseline - score)
    return values


@dataclass(frozen=True)
class EnvironmentData:
    """Named, already reconstructed train/test folds for one saved protocol."""

    folds: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], ...]


def analyze_feature_mechanism(environments: dict[str, EnvironmentData]) -> dict[str, Any]:
    """Fit fresh fixed XGBoost models only on reconstructed saved splits."""
    names = mstf_feature_names()
    groups = full_mstf_feature_groups()
    rankings: dict[str, dict[str, Any]] = {}
    for environment, data in sorted(environments.items()):
        gains: list[list[float]] = []
        permutations: list[list[float]] = []
        for train_x, train_y, test_x, test_y, seed in data.folds:
            if train_x.shape[1] != 112 or test_x.shape[1] != 112:
                raise MechanismError("importance analysis requires full 112-D MSTF matrices")
            model = build_model("xgb", seed).fit(train_x, train_y)
            gains.append(_xgb_gain(model, 112))
            permutations.append(_permutation_f1_decrease(model, test_x, test_y, seed))
        rankings[environment] = aggregate_importance(gains, permutations, names, groups)
    stability: list[dict[str, Any]] = []
    for left in sorted(rankings):
        for right in sorted(rankings):
            if left >= right:
                continue
            left_ranked = [row["feature"] for row in sorted(rankings[left]["feature_rows"], key=lambda item: item["permutation_mean"], reverse=True)]
            right_ranked = [row["feature"] for row in sorted(rankings[right]["feature_rows"], key=lambda item: item["permutation_mean"], reverse=True)]
            stability.append({"left_environment": left, "right_environment": right, "metric": "jaccard_top_10_permutation", "value": jaccard_top_k(left_ranked, right_ranked, 10)})
    return {"schema_version": 1, "rankings": rankings, "rank_stability": stability}


def _full_mstf_spec() -> dict[str, Any]:
    return {"feature_count": 28, "feature_dimensions": 112, "feature_family": "enhanced28", "representation": "full", "model": "xgb"}


def _dataset_paths() -> dict[str, Path]:
    # The frozen source corpus lives at the reproduction root; V1 is a child
    # implementation directory rather than the dataset owner.
    root = Path(__file__).resolve().parents[2]
    return {language: root / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl" for language in ("c", "cpp", "java", "py")}


def _saved_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MechanismError(f"required frozen ledger is absent: {path}")
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MechanismError(f"invalid frozen ledger: {path}") from exc
    if not records:
        raise MechanismError(f"frozen ledger is empty: {path}")
    return records


def _assert_saved_split(records: list[dict[str, Any]], **binding: Any) -> None:
    matches = [record for record in records if all(record.get(key) == value for key, value in binding.items())]
    if len(matches) != 1:
        raise MechanismError(f"saved ledger does not uniquely bind reconstructed split: {binding}")


def reconstruct_importance_environments(results_root: Path) -> dict[str, EnvironmentData]:
    """Reconstruct exactly the saved T1/T3/T4/T5 split families for MSTF.

    This creates no Gate records and writes no caches.  Each reconstructed fold
    is matched to precisely one pre-existing `mstf` ledger entry before it can
    enter the importance analysis.
    """
    from . import t3, t4, t5

    datasets = _dataset_paths()
    official_cache = results_root / "01_transition_test_strict_origins" / "cache"
    enhanced_cache = results_root / "02_unseen_llm" / "cache"
    attack_cache = results_root / "03_style_attack" / "cache"
    clean_records = _saved_records(results_root / "01_transition_test_strict_origins" / "folds.jsonl")
    unseen_records = _saved_records(results_root / "02_unseen_llm" / "folds.jsonl")
    attack_records = _saved_records(results_root / "03_style_attack" / "folds.jsonl")
    # Gate D is intentionally stored under the V1 project, unlike A--C which
    # are workspace-level result roots.  This is the canonical location named
    # by the frozen registry; do not copy it into the workspace results root.
    cross_records = _saved_records(Path(__file__).resolve().parents[1] / "results" / "04_cross_language" / "folds.jsonl")
    caches = {language: t3.load_or_build_enhanced_cache(language, datasets[language], enhanced_cache, official_cache) for language in datasets}
    spec = _full_mstf_spec()
    clean: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    unseen: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    attack: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    cross: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    seeds = (42, 123, 2024)
    for language, cache in caches.items():
        attacked = t4.load_or_build_attack_cache(language, datasets[language], attack_cache)
        for seed in seeds:
            for split in t3.build_t1_pair_splits(cache, language, 5, seed):
                _assert_saved_split(clean_records, language=language, representation="concat_delta", model="xgb", seed=seed, fold=split.fold, train_index_sha256=split.train_pair_sha256, test_index_sha256=split.test_pair_sha256)
                train_x, train_y = t3._t3_pair_matrix(cache, split.train_pairs, spec)
                test_x, test_y = t3._t3_pair_matrix(cache, split.test_pairs, spec)
                clean.append((train_x, train_y, test_x, test_y, seed))
                _assert_saved_split(attack_records, language=language, method="mstf", condition="combined", seed=seed, fold=split.fold, train_index_sha256=split.train_pair_sha256, test_index_sha256=split.test_pair_sha256)
                matrices = t4._condition_matrices(cache, attacked, split, spec)
                attack_x, _clean_reference, attack_y, audit = matrices["combined"]
                if audit["parse_regressions"] != 0:
                    raise MechanismError("combined attack has parse regressions")
                attack.append((train_x, train_y, attack_x, attack_y, seed))
            for heldout in t3.LLM_SOURCES:
                for split in t3.build_t3_splits(cache, language, heldout, 5, seed):
                    _assert_saved_split(unseen_records, language=language, heldout_llm=heldout, method="mstf", seed=seed, fold=split.fold, train_index_sha256=split.train_pair_sha256, test_index_sha256=split.test_pair_sha256)
                    train_x, train_y = t3._t3_pair_matrix(cache, split.train_pairs, spec)
                    test_x, test_y = t3._t3_pair_matrix(cache, split.test_pairs, spec)
                    unseen.append((train_x, train_y, test_x, test_y, seed))
    for seed in seeds:
        banks = {language: t5.build_language_pair_bank(cache, language, seed, 5) for language, cache in caches.items()}
        for heldout, bank in banks.items():
            _assert_saved_split(cross_records, heldout_language=heldout, method="mstf", seed=seed)
            train_parts = [t3._t3_pair_matrix(caches[language], banks[language].pairs, spec) for language in sorted(caches) if language != heldout]
            train_x = np.vstack([part[0] for part in train_parts])
            train_y = np.concatenate([part[1] for part in train_parts])
            test_x, test_y = t3._t3_pair_matrix(caches[heldout], bank.pairs, spec)
            cross.append((train_x, train_y, test_x, test_y, seed))
    return {"clean": EnvironmentData(tuple(clean)), "unseen_llm": EnvironmentData(tuple(unseen)), "combined_attack": EnvironmentData(tuple(attack)), "cross_language": EnvironmentData(tuple(cross))}


def write_importance_analysis(report: dict[str, Any], output_root: Path, registry_digest: str) -> None:
    """Publish registry-bound importance CSV/JSON/Markdown outside Gate roots."""
    output_root.mkdir(parents=True, exist_ok=True)
    feature_rows = [dict(environment=name, frozen_registry_sha256=registry_digest, **row) for name, ranking in report["rankings"].items() for row in ranking["feature_rows"]]
    group_rows = [dict(environment=name, frozen_registry_sha256=registry_digest, **row) for name, ranking in report["rankings"].items() for row in ranking["group_rows"]]
    for filename, rows in (("gain_importance.csv", feature_rows), ("permutation_importance.csv", feature_rows), ("grouped_importance.csv", group_rows), ("rank_stability.csv", [{**row, "frozen_registry_sha256": registry_digest} for row in report["rank_stability"]])):
        selected = rows
        if filename == "gain_importance.csv":
            selected = [{key: value for key, value in row.items() if not key.startswith("permutation_")} for row in rows]
        elif filename == "permutation_importance.csv":
            selected = [{key: value for key, value in row.items() if not key.startswith("gain_")} for row in rows]
        if not selected:
            raise MechanismError(f"no rows produced for {filename}")
        with (output_root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    payload = {**report, "frozen_registry_sha256": registry_digest}
    (output_root / "mechanism_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Feature-importance mechanism analysis", "", f"Frozen registry SHA-256: `{registry_digest}`.", "", "This is descriptive: importance is computed from fixed XGBoost fits on reconstructed saved splits, and does not establish a causal feature effect.", ""]
    for environment, ranking in report["rankings"].items():
        leading = max(ranking["group_rows"], key=lambda row: row["permutation_mean"])
        lines.append(f"- `{environment}`: highest grouped held-out permutation decrease is `{leading['group']}` ({leading['permutation_mean']:.6f}; {leading['n_folds']} reconstructed folds).")
    (output_root / "mechanism_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decompose_attack_ledger(records: list[dict[str, Any]], registry_digest: str) -> dict[str, Any]:
    """Summarize Gate C only; absent conditions/methods are hard failures."""
    if not isinstance(registry_digest, str) or len(registry_digest) != 64:
        raise MechanismError("invalid frozen registry digest")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise MechanismError("attack ledger contains a non-object record")
        if record.get("attack_parse_regressions", 0) != 0:
            raise MechanismError("attack ledger has nonzero parse regression")
        grouped[(record.get("language"), record.get("condition"), record.get("method"))].append(record)
    languages = tuple(sorted({key[0] for key in grouped if isinstance(key[0], str)}))
    missing = [
        (language, condition, method)
        for language in languages or ("<none>",)
        for condition in ATTACK_CONDITIONS
        for method in ATTACK_METHODS
        if not grouped.get((language, condition, method))
    ]
    if missing:
        raise MechanismError("missing required attack conditions or methods")
    rows: list[dict[str, Any]] = []
    for language in languages:
        for condition in ATTACK_CONDITIONS:
            for method in ATTACK_METHODS:
                values = grouped[(language, condition, method)]
                clean = grouped[(language, "clean", method)]
                by_pair = {(item["seed"], item["fold"]): item for item in clean}
                attacked_by_pair = {(item["seed"], item["fold"]): item for item in values}
                if set(by_pair) != set(attacked_by_pair):
                    raise MechanismError("unpaired clean/attack fold records")
                drops = [by_pair[key]["f1"] - attacked_by_pair[key]["f1"] for key in by_pair]
                rows.append(
                    {
                        "language": language,
                        "condition": condition,
                        "method": method,
                        "n_paired_records": len(drops),
                        "clean_f1": mean(item["f1"] for item in clean),
                        "attacked_f1": mean(item["f1"] for item in values),
                        "absolute_drop": mean(drops),
                        "relative_drop": mean(drops) / mean(item["f1"] for item in clean),
                        "changed_snippet_ratio": mean(item.get("attack_changed", 0) / item.get("attack_attempted", 1) for item in values),
                        "attack_success_ratio": mean(item.get("attack_successes", 0) / item.get("attack_attempted", 1) for item in values),
                        "parse_failure_ratio": mean((item.get("test_human_parse_failures", 0) + item.get("test_candidate_parse_failures", 0)) / (2 * item["test_rows"]) for item in values),
                        "frozen_registry_sha256": registry_digest,
                    }
                )
    return {"schema_version": 1, "conditions": ATTACK_CONDITIONS, "rows": rows, "frozen_registry_sha256": registry_digest}


def write_attack_decomposition(report: dict[str, Any], output_root: Path) -> None:
    """Write CSV, Markdown and JSON outputs outside all Gate roots."""
    output_root.mkdir(parents=True, exist_ok=True)
    rows = report["rows"]
    columns = list(rows[0])
    with (output_root / "attack_decomposition.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    markdown = ["# Style-attack decomposition", "", f"Frozen registry SHA-256: `{report['frozen_registry_sha256']}`.", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    markdown.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    (output_root / "attack_decomposition.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (output_root / "attack_drop_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("attack", "importance"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--t4-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if not args.registry.is_file():
        raise MechanismError("frozen registry does not exist")
    registry_digest = hashlib.sha256(args.registry.read_bytes()).hexdigest()
    results_root = args.registry.parent.parent
    if args.command == "attack":
        if args.t4_root is None:
            raise MechanismError("attack analysis requires --t4-root")
        records = _saved_records(args.t4_root / "folds.jsonl")
        report = decompose_attack_ledger(records, registry_digest)
        write_attack_decomposition(report, args.output_root or results_root / "05_mechanism_analysis")
        return
    environments = reconstruct_importance_environments(results_root)
    report = analyze_feature_mechanism(environments)
    write_importance_analysis(report, args.output_root or results_root / "05_mechanism_analysis" / "feature_importance", registry_digest)


if __name__ == "__main__":
    main()
