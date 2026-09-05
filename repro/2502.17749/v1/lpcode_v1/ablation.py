"""Registry-bound A0--A5 orthogonal ablation utilities and runner.

The runner is deliberately additive: it reconstructs and verifies the formal
T1/T3 fold hashes before evaluating any fixed XGBoost ablation cell and never
writes below a canonical Gate root.
"""

from __future__ import annotations

import hashlib
import json
import argparse
import csv
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import evaluate_fold
from .gates_ablation import METHOD_CONTRACT, expected_feature_dimension, record_key, validate_ablation_records
from .representations import build_representation
from . import t1_strict, t3
from .paths import REPRO_ROOT, RESULTS_ROOT


class AblationContractError(RuntimeError):
    """Raised when an ablation input is not tied to the frozen protocol."""


ABLATION_METHODS = METHOD_CONTRACT
_PAIR_KEYS = ("language", "seed", "fold", "train_index_sha256", "test_index_sha256")
CONTRASTS = {"C1": ("A1", "A0"), "C2": ("A2", "A0"), "C3": ("A4", "A1"), "C4": ("A5", "A4"), "C5": ("A5", "A0")}
NEGATIVE_PAIR_ARTIFACTS = (
    "config.json",
    "pairing_audit.json",
    "raw_results.json",
    "summary.csv",
    "bootstrap.json",
    "per_language.csv",
    "pair_difficulty_summary.csv",
    "additional_metrics.csv",
)


def registry_sha256(path: str | Path) -> str:
    """Return the digest only after verifying a usable frozen registry exists."""
    path = Path(path).resolve()
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationContractError(f"unreadable frozen registry: {path}") from exc
    if not isinstance(registry, dict) or set(registry.get("bundles", ())) != {
        "gate_a", "gate_b", "gate_c", "gate_d"
    }:
        raise AblationContractError("frozen registry does not bind all four formal gates")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_registry_files(path: str | Path) -> dict[str, dict[str, str]]:
    """Rehash every registry-controlled file and fail on any frozen mutation."""
    registry_path = Path(path).resolve()
    registry = _read_json(registry_path)
    snapshot: dict[str, dict[str, str]] = {}
    bundles = registry.get("bundles")
    if not isinstance(bundles, dict):
        raise AblationContractError("frozen registry has no bundles map")
    for bundle_name, bundle in bundles.items():
        files = bundle.get("files") if isinstance(bundle, dict) else None
        if not isinstance(files, dict):
            raise AblationContractError(f"frozen registry bundle has no files: {bundle_name}")
        snapshot[str(bundle_name)] = {}
        for filename, entry in files.items():
            if not isinstance(entry, dict):
                raise AblationContractError(f"invalid frozen registry entry: {bundle_name}/{filename}")
            controlled = Path(str(entry.get("path", ""))).resolve()
            if not controlled.is_file():
                raise AblationContractError(f"PROTOCOL / ARTIFACT MUTATION DETECTED: missing {controlled}")
            actual = hashlib.sha256(controlled.read_bytes()).hexdigest()
            if actual != entry.get("sha256") or controlled.stat().st_size != entry.get("bytes"):
                raise AblationContractError(f"PROTOCOL / ARTIFACT MUTATION DETECTED: {controlled}")
            snapshot[str(bundle_name)][str(filename)] = actual
    return snapshot


def negative_pair_artifact_hashes(root: str | Path) -> dict[str, str]:
    """Return a complete read-only digest map for the registered sensitivity check."""
    directory = Path(root).resolve()
    hashes: dict[str, str] = {}
    for filename in NEGATIVE_PAIR_ARTIFACTS:
        path = directory / filename
        if not path.is_file():
            raise AblationContractError(f"missing negative-pair artifact: {path}")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = _read_json(directory / "pairing_audit.json")
    if audit.get("pass") is not True:
        raise AblationContractError("negative-pair robustness audit is not PASS")
    return hashes


def assert_pair_contract(source: dict[str, Any], record: dict[str, Any]) -> None:
    """Require each ablation cell to retain the source fold identity exactly."""
    if not isinstance(source, dict) or not isinstance(record, dict):
        raise AblationContractError("source and ablation records must be objects")
    for key in _PAIR_KEYS:
        if source.get(key) != record.get(key):
            label = "pair hash" if key.endswith("sha256") else key
            raise AblationContractError(f"ablation {label} differs from source split")


def build_ablation_matrix(
    human: np.ndarray, candidate: np.ndarray, method: str
) -> np.ndarray:
    """Build one locked A0--A5 representation from matching endpoint features."""
    try:
        spec = ABLATION_METHODS[method]
    except KeyError as exc:
        raise AblationContractError(f"unknown ablation method: {method}") from exc
    count = int(spec["feature_count"])
    matrix = build_representation(human[:, :count], candidate[:, :count], spec["representation"])
    expected = count * {"concat": 2, "delta": 1, "concat_delta": 3, "full": 4}[spec["representation"]]
    if matrix.shape[1] != expected:
        raise AblationContractError("ablation feature dimension mismatch")
    return matrix


def evaluate_ablation_cell(
    source_record: dict[str, Any],
    *,
    human_train: np.ndarray,
    candidate_train: np.ndarray,
    y_train: np.ndarray,
    human_test: np.ndarray,
    candidate_test: np.ndarray,
    y_test: np.ndarray,
    method: str,
    frozen_registry_sha256: str,
) -> dict[str, Any]:
    """Score one fixed-XGBoost cell while preserving every source split digest."""
    if not isinstance(frozen_registry_sha256, str) or len(frozen_registry_sha256) != 64:
        raise AblationContractError("invalid frozen registry digest")
    train = build_ablation_matrix(human_train, candidate_train, method)
    test = build_ablation_matrix(human_test, candidate_test, method)
    metrics = evaluate_fold(train, y_train, test, y_test, "xgb", int(source_record["seed"]))
    result = {
        **metrics,
        **{key: source_record[key] for key in _PAIR_KEYS},
        "environment": source_record.get("environment"),
        "heldout_llm": source_record.get("heldout_llm"),
        "method": method,
        **ABLATION_METHODS[method],
        "feature_dimensions": int(train.shape[1]),
        "frozen_registry_sha256": frozen_registry_sha256,
    }
    assert_pair_contract(source_record, result)
    return result


def summarize_contrast(
    records: list[dict[str, Any]], left_method: str, right_method: str, *, replicates: int = 10_000, rng_seed: int = 250217749
) -> dict[str, Any]:
    """Pair methods on identical records and bootstrap *seeds*, not folds."""
    if left_method not in ABLATION_METHODS or right_method not in ABLATION_METHODS:
        raise AblationContractError("unknown ablation contrast method")
    left = {tuple(record.get(key) for key in ("environment", "language", "heldout_llm", "seed", "fold")): record for record in records if record.get("method") == left_method}
    right = {tuple(record.get(key) for key in ("environment", "language", "heldout_llm", "seed", "fold")): record for record in records if record.get("method") == right_method}
    if not left or set(left) != set(right):
        raise AblationContractError("contrast methods are not exactly paired")
    deltas = {key: float(left[key]["f1"]) - float(right[key]["f1"]) for key in left}
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key, value in deltas.items():
        by_seed[int(key[3])].append(value)
    seed_means = np.asarray([np.mean(values) for _seed, values in sorted(by_seed.items())], dtype=float)
    rng = np.random.default_rng(rng_seed)
    bootstrap = np.asarray([np.mean(rng.choice(seed_means, size=len(seed_means), replace=True)) for _ in range(replicates)], dtype=float)
    metric_deltas = {}
    for metric in ("f1", "precision", "recall", "auroc", "mcc"):
        if all(metric in left[key] and metric in right[key] for key in left):
            values = [float(left[key][metric]) - float(right[key][metric]) for key in left]
            metric_deltas[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return {
        "left": left_method,
        "right": right_method,
        "cluster_unit": "seed",
        "n_seeds": len(seed_means),
        "n_folds": len(deltas),
        "mean_delta_f1": float(np.mean(list(deltas.values()))),
        "metric_deltas": metric_deltas,
        "per_seed_delta_f1": {str(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items())},
        "direction_counts": {"positive": int(sum(value > 0 for value in seed_means)), "zero": int(sum(value == 0 for value in seed_means)), "negative": int(sum(value < 0 for value in seed_means))},
        "ci_95": {"method": "seed_cluster_bootstrap", "replicates": replicates, "seed": rng_seed, "low": float(np.quantile(bootstrap, 0.025)), "high": float(np.quantile(bootstrap, 0.975))},
    }


def summarize_ablation(output_root: str | Path) -> dict[str, Any]:
    """Create the required clean/unseen summary files from a complete ledger."""
    output_root = Path(output_root).resolve()
    config = _read_json(output_root / "config.json")
    records = _records(output_root / "folds.jsonl")
    expected = int(config["expected_clean"]) + int(config["expected_unseen"])
    if len(records) != expected:
        raise AblationContractError(f"ablation ledger incomplete: expected {expected}, got {len(records)}")
    summaries: dict[str, Any] = {"schema_version": 1, "frozen_registry_sha256": config["frozen_registry_sha256"], "environments": {}}
    for environment in ("clean", "unseen"):
        scoped = [record for record in records if record.get("environment") == environment]
        contrasts = {name: summarize_contrast(scoped, *methods) for name, methods in CONTRASTS.items()}
        by_language = {
            language: {name: summarize_contrast([record for record in scoped if record["language"] == language], *methods) for name, methods in CONTRASTS.items()}
            for language in sorted({record["language"] for record in scoped})
        }
        by_heldout = {
            heldout: {name: summarize_contrast([record for record in scoped if record.get("heldout_llm") == heldout], *methods) for name, methods in CONTRASTS.items()}
            for heldout in sorted({record.get("heldout_llm") for record in scoped if record.get("heldout_llm") is not None})
        }
        summaries["environments"][environment] = {"overall": contrasts, "by_language": by_language, "by_heldout_llm": by_heldout}
        csv_path = output_root / f"ablation_{environment}.csv"
        rows = [{"contrast": name, **summary} for name, summary in contrasts.items()]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["contrast", "left", "right", "cluster_unit", "n_seeds", "n_folds", "mean_delta_f1", "metric_deltas", "per_seed_delta_f1", "direction_counts", "ci_95"])
            writer.writeheader()
            writer.writerows(rows)
        lines = [f"# Orthogonal ablation: {environment}", "", f"Frozen registry SHA-256: `{config['frozen_registry_sha256']}`.", "", "| Contrast | Comparison | Mean paired ΔF1 | 95% seed-cluster CI |", "| --- | --- | ---: | --- |"]
        lines.extend(f"| {name} | {value['left']} − {value['right']} | {value['mean_delta_f1']:.6f} | [{value['ci_95']['low']:.6f}, {value['ci_95']['high']:.6f}] |" for name, value in contrasts.items())
        (output_root / f"ablation_{environment}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "ablation_paired_deltas.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "ablation_summary.md").write_text("# Orthogonal ablation summary\n\n" + "This report is generated from paired A0–A5 records with seed-cluster bootstrap intervals.\n", encoding="utf-8")
    (output_root / "ablation_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summaries


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationContractError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AblationContractError(f"JSON root must be an object: {path}")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationContractError(f"unreadable fold ledger: {path}") from exc


def _pairs_matrix(cache: Any, pairs: tuple[Any, ...], method: str) -> tuple[np.ndarray, np.ndarray]:
    human_indices = np.asarray([pair.human_positive_row_idx for pair in pairs], dtype=np.int64)
    candidate_indices = np.asarray([pair.candidate_positive_row_idx for pair in pairs], dtype=np.int64)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    return build_ablation_matrix(cache.human[human_indices], cache.llm[candidate_indices], method), labels


def _source_index(records: list[dict[str, Any]], *, unseen: bool) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (record.get("language"), record.get("heldout_llm") if unseen else None, record.get("seed"), record.get("fold"))
        required_key = key if unseen else (key[0], key[2], key[3])
        if not all(value is not None for value in required_key):
            raise AblationContractError("source fold record has incomplete key")
        prior = indexed.get(key)
        if prior is None:
            indexed[key] = record
        else:
            for digest in ("train_index_sha256", "test_index_sha256"):
                if prior.get(digest) != record.get(digest):
                    raise AblationContractError("source methods disagree on pair hash")
    return indexed


def _load_cache(language: str, dataset: Path, enhanced_root: Path, official_root: Path) -> Any:
    archive, metadata = t3._cache_paths(language, enhanced_root)
    if not archive.is_file() or not metadata.is_file():
        raise AblationContractError(f"missing frozen enhanced cache for {language}")
    # Existing cache is loaded through the established validator; no cache creation
    # or publication path is permitted here.
    return t3._load_or_build_enhanced_cache_unlocked(language, dataset, enhanced_root, official_root)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _replace_with_retry(temporary, path)


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a deterministic config or require an existing one to match exactly."""
    path = path.resolve()
    if path.exists():
        if _read_json(path) != payload:
            raise AblationContractError(f"immutable config mismatch: {path}")
        return
    _atomic_json(path, payload)


def assert_smoke_output_root(output_root: str | Path, protected_roots: tuple[Path, ...]) -> Path:
    """Require an explicit smoke path that cannot overlap protected results."""
    output = Path(output_root).resolve()
    if "smoke" not in output.name.lower():
        raise AblationContractError("smoke output path must contain the token 'smoke'")
    for root in (Path(value).resolve() for value in protected_roots):
        if output == root or output in root.parents or root in output.parents:
            raise AblationContractError(f"smoke output overlaps protected root: {root}")
    return output


def _completed_index(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    completed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in completed:
            raise AblationContractError(f"duplicate record key: {key}")
        completed[key] = record
    return completed


def _class_counts_from_pairs(pairs: tuple[Any, ...]) -> dict[str, int]:
    counts = Counter(int(pair.label) for pair in pairs)
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def _audit_pair_split(split: Any) -> dict[str, Any]:
    """Compute leakage and fairness counters directly from immutable pair specs."""
    train_pairs, test_pairs = tuple(split.train_pairs), tuple(split.test_pairs)
    train_endpoints = {
        endpoint for pair in train_pairs for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
    }
    test_endpoints = {
        endpoint for pair in test_pairs for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
    }
    train_codes = {
        digest for pair in train_pairs for digest in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    test_codes = {
        digest for pair in test_pairs for digest in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    duplicate_count = sum(
        len(pairs) - len({pair.pair_sha256 for pair in pairs})
        for pairs in (train_pairs, test_pairs)
    )
    return {
        "endpoint_leakage_count": len(train_endpoints & test_endpoints),
        "content_leakage_count": len(train_codes & test_codes),
        "negative_component_violation_count": sum(
            int(pair.label == 0 and pair.human_component_id == pair.candidate_component_id)
            for pair in (*train_pairs, *test_pairs)
        ),
        "duplicate_pair_count": int(duplicate_count),
        "train_class_counts": _class_counts_from_pairs(train_pairs),
        "test_class_counts": _class_counts_from_pairs(test_pairs),
    }


def _assert_pair_audit_passes(audit: dict[str, Any]) -> None:
    """Apply leakage and balance stop criteria before fitting any model."""
    labels = {
        "endpoint_leakage_count": "endpoint leakage",
        "content_leakage_count": "content leakage",
        "negative_component_violation_count": "negative-component violation",
        "duplicate_pair_count": "duplicate pair",
    }
    for field, label in labels.items():
        if audit.get(field) != 0:
            raise AblationContractError(f"non-zero {label}")
    for side in ("train", "test"):
        counts = audit.get(f"{side}_class_counts")
        if not isinstance(counts, dict) or counts.get("0", 0) <= 0 or counts.get("0") != counts.get("1"):
            raise AblationContractError(f"invalid {side} class balance")


def _assert_resume_record(
    reconstructed: dict[str, Any],
    existing: dict[str, Any],
    *,
    method: str,
    frozen_registry_sha256: str,
) -> None:
    """Reject a completed cell unless a fresh split reconstruction still matches it."""
    for key in ("train_pair_sha256", "test_pair_sha256", "train_index_sha256", "test_index_sha256"):
        if reconstructed.get(key) != existing.get(key):
            raise AblationContractError(f"resume pair hash mismatch: {key}")
    if existing.get("method") != method or existing.get("feature_dimensions") != expected_feature_dimension(method):
        raise AblationContractError("resume method contract mismatch")
    if existing.get("frozen_registry_sha256") != frozen_registry_sha256:
        raise AblationContractError("resume registry mismatch")
    for key in (
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
        "duplicate_pair_count",
        "train_class_counts",
        "test_class_counts",
    ):
        if key in reconstructed and reconstructed.get(key) != existing.get(key):
            raise AblationContractError(f"resume audit mismatch: {key}")


def _build_smoke_config(
    *,
    frozen_registry_sha256: str,
    negative_pair_artifact_sha256: dict[str, str],
    language: str,
    seed: int,
    n_splits: int,
    limit_origins: int,
    heldout_llm: str,
) -> dict[str, Any]:
    """Build the deterministic, immutable 24-record smoke configuration."""
    expected_per_environment = len(ABLATION_METHODS) * n_splits
    return {
        "schema_version": 1,
        "run_kind": "smoke",
        "frozen_registry_sha256": frozen_registry_sha256,
        "negative_pair_artifact_sha256": dict(sorted(negative_pair_artifact_sha256.items())),
        "language": language,
        "seed": seed,
        "n_splits": n_splits,
        "limit_origins": limit_origins,
        "heldout_llm": heldout_llm,
        "methods": ABLATION_METHODS,
        "protocols": {
            "clean": "all-llm-strict-origin-v2-smoke",
            "unseen": "leave-one-llm-strict-origin-v1-smoke",
            "pair": t3.PAIR_PROTOCOL_VERSION,
            "component": t3.COMPONENT_PROTOCOL_VERSION,
        },
        "expected_clean": expected_per_environment,
        "expected_unseen": expected_per_environment,
        "expected_total": expected_per_environment * 2,
    }


def _load_cache_read_only(language: str, dataset: Path, enhanced_root: Path, official_root: Path) -> tuple[Any, dict[str, str]]:
    """Load validated caches and prove that the operation did not create or alter them."""
    enhanced_archive, enhanced_metadata = t3._cache_paths(language, enhanced_root)
    official_archive, official_metadata = t3._official_cache_paths(language, official_root)
    paths = (enhanced_archive, enhanced_metadata, official_archive, official_metadata)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise AblationContractError(f"missing frozen cache; refusing to build: {missing}")
    before = {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    try:
        cache = _load_cache(language, dataset, enhanced_root, official_root)
    except (OSError, ValueError) as exc:
        raise AblationContractError(f"stale or invalid frozen cache for {language}") from exc
    after = {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise AblationContractError("PROTOCOL / ARTIFACT MUTATION DETECTED: cache changed during load")
    return cache, after


def run_ablation_smoke(
    registry_path: str | Path,
    output_root: str | Path,
    *,
    negative_pair_root: str | Path = RESULTS_ROOT / "negative_pair_robustness",
    language: str = "c",
    seed: int = 42,
    n_splits: int = 2,
    limit_origins: int = 8,
    heldout_llm: str = "gpt3.5",
) -> dict[str, Any]:
    """Run/resume the isolated clean plus unseen A0--A5 smoke matrix only."""
    if n_splits != 2 or limit_origins != 8 or language != "c" or seed != 42 or heldout_llm != "gpt3.5":
        raise AblationContractError("smoke contract is fixed to c/42/2 folds/8 origins/gpt3.5")
    registry_path = Path(registry_path).resolve()
    digest = registry_sha256(registry_path)
    before_registry = verify_registry_files(registry_path)
    registry = _read_json(registry_path)
    protected_roots = tuple(Path(bundle["root"]).resolve() for bundle in registry["bundles"].values())
    output = assert_smoke_output_root(output_root, protected_roots)
    negative_hashes = negative_pair_artifact_hashes(negative_pair_root)
    config = _build_smoke_config(
        frozen_registry_sha256=digest,
        negative_pair_artifact_sha256=negative_hashes,
        language=language,
        seed=seed,
        n_splits=n_splits,
        limit_origins=limit_origins,
        heldout_llm=heldout_llm,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(output / "config.json", config)
    ledger_path = output / "folds.jsonl"
    completed_records = _records(ledger_path) if ledger_path.exists() else []
    completed = _completed_index(completed_records)
    if any(record.get("run_kind") != "smoke" for record in completed.values()):
        raise AblationContractError("formal record found in smoke ledger")

    gate_a_root = Path(registry["bundles"]["gate_a"]["root"]).resolve()
    gate_b_root = Path(registry["bundles"]["gate_b"]["root"]).resolve()
    dataset = (REPRO_ROOT / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl").resolve()
    cache, cache_hashes = _load_cache_read_only(language, dataset, gate_b_root / "cache", gate_a_root / "cache")
    cache = t1_strict._select_positive_bank(cache, limit_origins)
    environment_splits = (
        ("clean", None, t3.build_t1_pair_splits(cache, language, n_splits, seed)),
        ("unseen", heldout_llm, t3.build_t3_splits(cache, language, heldout_llm, n_splits, seed)),
    )
    new_count = 0
    skipped = 0
    for environment, heldout, splits in environment_splits:
        for split in splits:
            pair_audit = _audit_pair_split(split)
            _assert_pair_audit_passes(pair_audit)
            if environment == "unseen":
                if any(pair.llm_source == heldout_llm for pair in split.train_pairs) or any(
                    pair.llm_source != heldout_llm for pair in split.test_pairs
                ):
                    raise AblationContractError("held-out LLM exclusivity violation")
            reconstructed = {
                "environment": environment,
                "language": language,
                "heldout_llm": heldout,
                "seed": seed,
                "fold": int(split.fold),
                "train_pair_sha256": split.train_pair_sha256,
                "test_pair_sha256": split.test_pair_sha256,
                "train_index_sha256": split.train_pair_sha256,
                "test_index_sha256": split.test_pair_sha256,
                **pair_audit,
            }
            for method in ABLATION_METHODS:
                key = record_key({**reconstructed, "method": method})
                if key in completed:
                    _assert_resume_record(reconstructed, completed[key], method=method, frozen_registry_sha256=digest)
                    skipped += 1
                    continue
                human_train = cache.human[[pair.human_positive_row_idx for pair in split.train_pairs]]
                candidate_train = cache.llm[[pair.candidate_positive_row_idx for pair in split.train_pairs]]
                human_test = cache.human[[pair.human_positive_row_idx for pair in split.test_pairs]]
                candidate_test = cache.llm[[pair.candidate_positive_row_idx for pair in split.test_pairs]]
                y_train = np.asarray([pair.label for pair in split.train_pairs], dtype=np.int64)
                y_test = np.asarray([pair.label for pair in split.test_pairs], dtype=np.int64)
                result = evaluate_ablation_cell(
                    reconstructed,
                    human_train=human_train,
                    candidate_train=candidate_train,
                    y_train=y_train,
                    human_test=human_test,
                    candidate_test=candidate_test,
                    y_test=y_test,
                    method=method,
                    frozen_registry_sha256=digest,
                )
                result.update(
                    {
                        "run_kind": "smoke",
                        "split_protocol": config["protocols"][environment],
                        "pair_protocol": config["protocols"]["pair"],
                        "component_protocol": config["protocols"]["component"],
                        "train_pair_sha256": split.train_pair_sha256,
                        "test_pair_sha256": split.test_pair_sha256,
                        **pair_audit,
                    }
                )
                completed[key] = result
                _atomic_jsonl(ledger_path, list(completed.values()))
                new_count += 1

        stage_records = [record for record in completed.values() if record.get("environment") == environment]
        validate_ablation_records(
            stage_records,
            expected_count=int(config[f"expected_{environment}"]),
            frozen_registry_sha256=digest,
        )

    records = list(completed.values())
    audit = validate_ablation_records(
        records,
        expected_count=int(config["expected_total"]),
        frozen_registry_sha256=digest,
    )
    audit.update(
        {
            "run_kind": "smoke",
            "expected_clean": config["expected_clean"],
            "expected_unseen": config["expected_unseen"],
            "cache_sha256": cache_hashes,
            "negative_pair_artifact_sha256": negative_hashes,
            "registry_files_unchanged": before_registry == verify_registry_files(registry_path),
        }
    )
    if audit["registry_files_unchanged"] is not True:
        raise AblationContractError("PROTOCOL / ARTIFACT MUTATION DETECTED after smoke")
    _atomic_json(output / "audit.json", audit)
    manifest_files = (output / "config.json", ledger_path, output / "audit.json")
    manifest = {
        "schema_version": 1,
        "run_kind": "smoke",
        "frozen_registry_sha256": digest,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in manifest_files
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return {
        "completed": new_count,
        "skipped": skipped,
        "expected": config["expected_total"],
        "audit_status": audit["status"],
        "output_root": output.as_posix(),
    }


def _replace_with_retry(source: Path, destination: Path, *, attempts: int = 12, delay_seconds: float = 0.1) -> None:
    """Tolerate short-lived Windows readers without sacrificing atomic publish."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay_seconds * (attempt + 1))


def run_ablation(registry_path: str | Path, output_root: str | Path) -> dict[str, int]:
    """Run/resume all A0--A5 cells on Gate A/B reconstructed formal splits."""
    registry_path = Path(registry_path).resolve()
    digest = registry_sha256(registry_path)
    registry = _read_json(registry_path)
    roots = {name: Path(value["root"]).resolve() for name, value in registry["bundles"].items()}
    gate_a_root, gate_b_root = roots["gate_a"], roots["gate_b"]
    output_root = Path(output_root).resolve()
    if output_root in {gate_a_root, gate_b_root, roots["gate_c"], roots["gate_d"]} or any(
        gate in output_root.parents for gate in roots.values()
    ):
        raise AblationContractError("refusing to write ablation output inside a frozen Gate root")
    clean_config, unseen_config = _read_json(gate_a_root / "config.json"), _read_json(gate_b_root / "config.json")
    clean_sources = _source_index(_records(gate_a_root / "folds.jsonl"), unseen=False)
    unseen_sources = _source_index(_records(gate_b_root / "folds.jsonl"), unseen=True)
    expected_clean = len(clean_config["languages"]) * len(clean_config["seeds"]) * clean_config["n_splits"] * len(ABLATION_METHODS)
    expected_unseen = len(unseen_config["languages"]) * len(unseen_config["heldout_llms"]) * len(unseen_config["seeds"]) * unseen_config["n_splits"] * len(ABLATION_METHODS)
    if (len(clean_sources), len(unseen_sources)) != (expected_clean // len(ABLATION_METHODS), expected_unseen // len(ABLATION_METHODS)):
        raise AblationContractError("canonical source ledgers are incomplete")
    ledger_path = output_root / "folds.jsonl"
    completed_records = _records(ledger_path) if ledger_path.exists() else []
    completed = _completed_index(completed_records)
    if any(record.get("frozen_registry_sha256") != digest for record in completed.values()):
        raise AblationContractError("existing ablation ledger belongs to a different frozen registry")
    new_count, skipped = 0, 0
    enhanced_root = gate_b_root / "cache"
    official_root = gate_a_root / "cache"
    for environment, config, sources, unseen in (("clean", clean_config, clean_sources, False), ("unseen", unseen_config, unseen_sources, True)):
        for language in config["languages"]:
            dataset = (REPRO_ROOT / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl").resolve()
            cache = _load_cache(language, dataset, enhanced_root, official_root)
            if environment == "clean":
                cache = t1_strict._select_positive_bank(cache, config["limit_origins"])
            for source_key, source in sources.items():
                if source_key[0] != language:
                    continue
                splits = (
                    t3.build_t3_splits(cache, language, source_key[1], config["n_splits"], source_key[2])
                    if unseen
                    else t3.build_t1_pair_splits(cache, language, config["n_splits"], source_key[2])
                )
                split = splits[int(source_key[3])]
                rebuilt = {**source, "environment": environment, "train_index_sha256": split.train_pair_sha256, "test_index_sha256": split.test_pair_sha256}
                assert_pair_contract(source, rebuilt)
                for method in ABLATION_METHODS:
                    key = (environment, method, language, source.get("heldout_llm"), source["seed"], source["fold"])
                    if key in completed:
                        assert_pair_contract(rebuilt, completed[key])
                        skipped += 1
                        continue
                    train, y_train = _pairs_matrix(cache, split.train_pairs, method)
                    test, y_test = _pairs_matrix(cache, split.test_pairs, method)
                    result = evaluate_ablation_cell(rebuilt, human_train=cache.human[[pair.human_positive_row_idx for pair in split.train_pairs]], candidate_train=cache.llm[[pair.candidate_positive_row_idx for pair in split.train_pairs]], y_train=y_train, human_test=cache.human[[pair.human_positive_row_idx for pair in split.test_pairs]], candidate_test=cache.llm[[pair.candidate_positive_row_idx for pair in split.test_pairs]], y_test=y_test, method=method, frozen_registry_sha256=digest)
                    if result["feature_dimensions"] != train.shape[1] or test.shape[1] != train.shape[1]:
                        raise AblationContractError("reconstructed ablation matrix differs from method contract")
                    completed[key] = result
                    _atomic_jsonl(ledger_path, list(completed.values()))
                    new_count += 1
    config_payload = {"schema_version": 1, "frozen_registry_sha256": digest, "expected_clean": expected_clean, "expected_unseen": expected_unseen, "methods": ABLATION_METHODS}
    (output_root / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "frozen_registry_sha256": digest, "files": {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (output_root / "config.json", ledger_path)}}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"completed": new_count, "skipped": skipped, "expected": expected_clean + expected_unseen}


def main() -> None:
    """Execute the fixed full ablation matrix from command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("run", "summarize", "smoke"), default="run")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--negative-pair-root", type=Path, default=RESULTS_ROOT / "negative_pair_robustness")
    args = parser.parse_args()
    if args.mode == "summarize":
        print(json.dumps(summarize_ablation(args.output_root), sort_keys=True))
    elif args.mode == "smoke":
        print(json.dumps(run_ablation_smoke(args.registry, args.output_root, negative_pair_root=args.negative_pair_root), sort_keys=True))
    else:
        print(json.dumps(run_ablation(args.registry, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
