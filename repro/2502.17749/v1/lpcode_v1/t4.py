"""Strict-origin, test-only style-attack experiment (T4)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features_enhanced import analyze_enhanced
from .features_official import analyze_code
from .experiment import build_model
from .cache_io import (
    atomic_write_bytes,
    exclusive_cache_lock,
    unique_temporary_path,
)
from .data import load_jsonl
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .representations import build_representation
from .t1 import _exclusive_output_lock
from .t3 import (
    COMPONENT_PROTOCOL_VERSION,
    DEFAULT_GATE_A_PATH,
    DEFAULT_OFFICIAL_CACHE_ROOT,
    EnhancedFeatureCache,
    LANGUAGES,
    PAIR_PROTOCOL_VERSION,
    T3_METHODS,
    _canonical_json,
    _cache_as_arrays,
    _digest_json,
    _load_strict_gate_a,
    _method_contract,
    _select_t3_positive_bank,
    _sha256,
    _row_sha256,
    _t3_pair_matrix,
    _validate_arrays,
    build_t1_pair_splits,
    load_or_build_enhanced_cache,
)
from .t1_strict import _split_metadata


ATTACKS = (
    "comment_removal",
    "identifier_rename",
    "format_normalization",
    "comment_injection",
    "combined",
)
CONDITIONS = ("clean", *ATTACKS)
METHODS = T3_METHODS
DEFAULT_SEEDS = (42, 123, 2024)
DEFAULT_GATE_B_PATH = RESULTS_ROOT / "02_unseen_llm" / "gate_b.json"
DEFAULT_ATTACK_CACHE_ROOT = RESULTS_ROOT / "03_style_attack" / "cache"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "03_style_attack"
ATTACK_CACHE_VERSION = "style-attack28-v1"
T4_SPLIT_PROTOCOL_VERSION = "all-llm-strict-origin-attack-v1"
T4_SCHEMA_VERSION = 1
T4_SMOKE_ORIGINS = 8


@dataclass(frozen=True)
class AttackFeatureCache:
    language: str
    row_sha256: np.ndarray
    features: np.ndarray
    success: np.ndarray
    output_sha256: np.ndarray
    changed: np.ndarray
    transform_count: np.ndarray
    parse_ok_before: np.ndarray
    parse_ok_after: np.ndarray
    backend_before: np.ndarray
    backend_after: np.ndarray
    failure_reason: np.ndarray
    semantic_content_sha256: str


T4_FOLD_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "config_id",
        "language",
        "method",
        "condition",
        "feature_family",
        "representation",
        "model",
        "feature_dimensions",
        "seed",
        "fold",
        "split_protocol",
        "pair_protocol",
        "component_protocol",
        "f1",
        "precision",
        "recall",
        "auroc",
        "mcc",
        "fit_seconds",
        "predict_seconds",
        "train_rows",
        "test_rows",
        "train_class_counts",
        "test_class_counts",
        "clean_reference_f1",
        "attack_attempted",
        "attack_successes",
        "attack_failures",
        "attack_changed",
        "attack_transform_count",
        "attack_parse_regressions",
        "attack_success_set_sha256",
        "attack_output_set_sha256",
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
        "train_index_sha256",
        "test_index_sha256",
        "train_unique_sources",
        "test_unique_sources",
        "train_unique_code_hashes",
        "test_unique_code_hashes",
        "train_unique_components",
        "test_unique_components",
        "train_llm_label_counts",
        "test_llm_label_counts",
        "source_jsonl_sha256",
        "clean_cache_content_sha256",
        "attack_cache_content_sha256",
        "gate_a_sha256",
        "gate_b_sha256",
        "record_sha256",
    }
)


def _evaluation_count(
    languages: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
) -> int:
    return len(languages) * len(methods) * len(seeds) * n_splits * len(CONDITIONS)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _load_strict_gate_b(path: str | Path) -> dict[str, Any]:
    requested = Path(path).resolve()
    expected = DEFAULT_GATE_B_PATH.resolve()
    if requested != expected or not requested.is_file():
        raise ValueError("T4 requires the exact Gate B artifact")
    root = requested.parent
    manifest_path = root / "manifest.json"
    config_path = root / "config.json"
    summary_path = root / "summary.json"
    for item in (manifest_path, config_path, summary_path):
        if not item.is_file():
            raise ValueError("T4 exact Gate B artifact is incomplete")
    gate = _load_json(requested)
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    summary = _load_json(summary_path)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ValueError("T4 exact Gate B manifest is malformed")
    for name in ("gate_b.json", "config.json", "summary.json", "folds.jsonl"):
        item = root / name
        spec = files.get(name)
        if (
            not item.is_file()
            or not isinstance(spec, dict)
            or spec.get("sha256") != _sha256(item)
            or spec.get("bytes") != item.stat().st_size
        ):
            raise ValueError("T4 exact Gate B manifest hash mismatch")
    strict = gate.get("strict") if isinstance(gate, dict) else None
    matrix = summary.get("matrix") if isinstance(summary, dict) else None
    if (
        gate.get("status") != "evaluable"
        or not isinstance(strict, dict)
        or strict.get("passed") is not True
        or strict.get("holdouts_won") != 4
        or strict.get("overall_macro_mean_delta_f1", -1) < 0.03
        or not isinstance(matrix, dict)
        or matrix.get("complete_cartesian_product") is not True
        or matrix.get("observed_records") != 960
        or config.get("languages") != list(LANGUAGES)
        or config.get("methods") != list(METHODS)
        or config.get("seeds") != list(DEFAULT_SEEDS)
        or config.get("n_splits") != 5
    ):
        raise ValueError("T4 requires a strict passing exact Gate B artifact")
    gate_a = manifest.get("gate_a_binding")
    if not isinstance(gate_a, dict):
        raise ValueError("T4 Gate B lacks strict Gate A provenance")
    return {
        "strict_passed": True,
        "authorizes_t4": True,
        "holdouts_won": int(strict["holdouts_won"]),
        "overall_macro_mean_delta_f1": float(
            strict["overall_macro_mean_delta_f1"]
        ),
        "gate_b_path": str(requested),
        "gate_b_sha256": _sha256(requested),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": _sha256(config_path),
        "config_id": config["config_id"],
        "gate_a_binding": gate_a,
        "source_jsonl_sha256": config["source_jsonl_sha256"],
        "method_contract": config["method_contract"],
    }


def _validate_attack_cache(cache: AttackFeatureCache) -> AttackFeatureCache:
    if cache.language not in LANGUAGES:
        raise ValueError("unsupported attack-cache language")
    n = len(cache.row_sha256)
    if (
        cache.row_sha256.shape != (n,)
        or cache.features.shape != (n, len(ATTACKS), 28)
        or cache.success.shape != (n, len(ATTACKS))
        or not np.isfinite(cache.features).all()
        or len(set(map(str, cache.row_sha256.tolist()))) != n
        or not isinstance(cache.semantic_content_sha256, str)
        or len(cache.semantic_content_sha256) != 64
    ):
        raise ValueError("invalid attack feature cache")
    for array in (
        cache.output_sha256,
        cache.changed,
        cache.transform_count,
        cache.parse_ok_before,
        cache.parse_ok_after,
        cache.backend_before,
        cache.backend_after,
        cache.failure_reason,
    ):
        if array.shape != (n, len(ATTACKS)):
            raise ValueError("invalid attack feature cache audit shape")
    return cache


def _attack_cache_paths(language: str, cache_root: str | Path) -> tuple[Path, Path]:
    root = resolve_output_path(cache_root) / ATTACK_CACHE_VERSION
    return root / f"{language}.npz", root / f"{language}.json"


def _attack_cache_arrays(cache: AttackFeatureCache) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(cache, name))
        for name in (
            "row_sha256",
            "features",
            "success",
            "output_sha256",
            "changed",
            "transform_count",
            "parse_ok_before",
            "parse_ok_after",
            "backend_before",
            "backend_after",
            "failure_reason",
        )
    }


def _semantic_attack_sha256(language: str, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(language.encode("utf-8"))
    digest.update(b"\0")
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        if np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype(bool):
            normalized = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
            digest.update(normalized.tobytes())
        else:
            digest.update(_canonical_json(array.astype(str).tolist()))
    return digest.hexdigest()


def _attack_cache_expected(language: str, dataset_path: Path) -> dict[str, Any]:
    from . import attacks

    return {
        "schema_version": 1,
        "cache_version": ATTACK_CACHE_VERSION,
        "language": language,
        "attacks": list(ATTACKS),
        "feature_count": 28,
        "source_jsonl_sha256": _sha256(dataset_path),
        "attack_version": attacks.ATTACK_VERSION,
        "attack_source_sha256": attacks.attack_source_sha256(),
        "official_feature_source_sha256": _sha256(
            Path(analyze_code.__code__.co_filename).resolve()
        ),
        "enhanced_feature_source_sha256": _sha256(
            Path(analyze_enhanced.__code__.co_filename).resolve()
        ),
    }


def _cache_from_arrays(
    language: str, arrays: Mapping[str, np.ndarray], semantic: str
) -> AttackFeatureCache:
    return _validate_attack_cache(
        AttackFeatureCache(
            language=language,
            semantic_content_sha256=semantic,
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
    )


def _load_attack_cache(
    language: str, archive: Path, metadata_path: Path, expected: dict[str, Any]
) -> AttackFeatureCache | None:
    if not archive.exists() and not metadata_path.exists():
        return None
    if not archive.is_file() or not metadata_path.is_file():
        raise ValueError("incomplete T4 attack cache publication")
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("T4 attack cache provenance mismatch")
    if metadata.get("npz_sha256") != _sha256(archive):
        raise ValueError("T4 attack cache archive hash mismatch")
    try:
        with np.load(archive, allow_pickle=False) as stored:
            if set(stored.files) != {
                "row_sha256",
                "features",
                "success",
                "output_sha256",
                "changed",
                "transform_count",
                "parse_ok_before",
                "parse_ok_after",
                "backend_before",
                "backend_after",
                "failure_reason",
            }:
                raise ValueError("T4 attack cache fields mismatch")
            arrays = {name: stored[name] for name in stored.files}
    except (OSError, ValueError) as exc:
        raise ValueError("invalid T4 attack cache archive") from exc
    semantic = _semantic_attack_sha256(language, arrays)
    if metadata.get("semantic_content_sha256") != semantic:
        raise ValueError("T4 attack cache semantic hash mismatch")
    return _cache_from_arrays(language, arrays, semantic)


def _build_attack_cache(
    language: str,
    dataset_path: Path,
    archive: Path,
    metadata_path: Path,
    expected: dict[str, Any],
) -> AttackFeatureCache:
    from .attacks import apply_attack

    rows = load_jsonl(dataset_path, task="task1")
    positive_rows = [row for row in rows if int(row["label"]) == 1]
    count = len(positive_rows)
    shape = (count, len(ATTACKS))
    features = np.empty((count, len(ATTACKS), 28), dtype=np.float64)
    success = np.empty(shape, dtype=np.bool_)
    output_sha256 = np.empty(shape, dtype="<U64")
    changed = np.empty(shape, dtype=np.bool_)
    transform_count = np.empty(shape, dtype=np.int64)
    parse_ok_before = np.empty(shape, dtype=np.bool_)
    parse_ok_after = np.empty(shape, dtype=np.bool_)
    backend_before = np.empty(shape, dtype="<U64")
    backend_after = np.empty(shape, dtype="<U64")
    failure_reason = np.empty(shape, dtype="<U128")
    memo: dict[tuple[str, str], tuple[Any, np.ndarray]] = {}
    for row_index, row in enumerate(positive_rows):
        code = str(row["llm_src"])
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        for attack_index, attack in enumerate(ATTACKS):
            key = (code_hash, attack)
            cached = memo.get(key)
            if cached is None:
                result = apply_attack(code, language, attack)
                enhanced = analyze_enhanced(result.code, language)
                vector = np.concatenate(
                    [analyze_code(result.code, language), enhanced.values]
                ).astype(np.float64, copy=False)
                if vector.shape != (28,) or not np.isfinite(vector).all():
                    raise ValueError("T4 attack produced invalid feature vector")
                if enhanced.parse_ok != result.parse_ok_after:
                    raise ValueError("T4 attack parser audit disagrees with features")
                cached = (result, vector)
                memo[key] = cached
            result, vector = cached
            features[row_index, attack_index] = vector
            success[row_index, attack_index] = result.failure_reason is None
            output_sha256[row_index, attack_index] = result.output_sha256
            changed[row_index, attack_index] = result.changed
            transform_count[row_index, attack_index] = result.transform_count
            parse_ok_before[row_index, attack_index] = result.parse_ok_before
            parse_ok_after[row_index, attack_index] = result.parse_ok_after
            backend_before[row_index, attack_index] = result.backend_before
            backend_after[row_index, attack_index] = result.backend_after
            failure_reason[row_index, attack_index] = result.failure_reason or ""
    arrays = {
        "row_sha256": np.asarray([_row_sha256(row) for row in positive_rows], dtype=str),
        "features": features,
        "success": success,
        "output_sha256": output_sha256,
        "changed": changed,
        "transform_count": transform_count,
        "parse_ok_before": parse_ok_before,
        "parse_ok_after": parse_ok_after,
        "backend_before": backend_before,
        "backend_after": backend_after,
        "failure_reason": failure_reason,
    }
    semantic = _semantic_attack_sha256(language, arrays)
    cache = _cache_from_arrays(language, arrays, semantic)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = unique_temporary_path(archive, suffix=".npz")
    try:
        np.savez(temporary, **arrays)
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        **expected,
        "rows": count,
        "failures": int((~success).sum()),
        "parse_regressions": int((parse_ok_before & ~parse_ok_after).sum()),
        "semantic_content_sha256": semantic,
        "npz_sha256": _sha256(archive),
    }
    atomic_write_bytes(metadata_path, _canonical_json(metadata))
    return cache


def load_or_build_attack_cache(
    language: str,
    dataset_path: str | Path,
    cache_root: str | Path = DEFAULT_ATTACK_CACHE_ROOT,
) -> AttackFeatureCache:
    if language not in LANGUAGES:
        raise ValueError("unsupported T4 attack-cache language")
    dataset = Path(dataset_path).resolve()
    if not dataset.is_file():
        raise ValueError("T4 attack-cache dataset does not exist")
    archive, metadata = _attack_cache_paths(language, cache_root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    expected = _attack_cache_expected(language, dataset)
    with exclusive_cache_lock(archive.parent / f".{language}.lock"):
        cached = _load_attack_cache(language, archive, metadata, expected)
        return cached or _build_attack_cache(
            language, dataset, archive, metadata, expected
        )


def _attack_rows(
    cache: AttackFeatureCache, row_hashes: np.ndarray, attack: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _validate_attack_cache(cache)
    if attack not in ATTACKS:
        raise ValueError("unsupported T4 attack")
    lookup = {str(value): index for index, value in enumerate(cache.row_sha256)}
    try:
        indices = np.asarray([lookup[str(value)] for value in row_hashes], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("attack cache row hash is missing") from exc
    attack_index = ATTACKS.index(attack)
    success = cache.success[indices, attack_index].astype(np.bool_, copy=True)
    failures = int((~success).sum())
    return (
        cache.features[indices, attack_index, :].copy(),
        success,
        {
            "attempted": int(len(indices)),
            "successes": int(success.sum()),
            "failures": failures,
            "changed": int(cache.changed[indices, attack_index].sum()),
            "transform_count": int(cache.transform_count[indices, attack_index].sum()),
            "parse_regressions": int(
                (
                    cache.parse_ok_before[indices, attack_index]
                    & ~cache.parse_ok_after[indices, attack_index]
                ).sum()
            ),
            "success_set_sha256": _digest_json(
                [
                    str(row_hash)
                    for row_hash, ok in zip(row_hashes.tolist(), success.tolist())
                    if ok
                ]
            ),
            "output_set_sha256": _digest_json(
                cache.output_sha256[indices, attack_index].tolist()
            ),
        },
    )


def _score_model(model: Any, x_test: Any, y_test: Any) -> dict[str, Any]:
    features = np.asarray(x_test)
    labels = np.asarray(y_test)
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or labels.shape != (features.shape[0],)
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(features).all()
    ):
        raise ValueError("T4 scoring requires finite nonempty binary data")
    started = time.perf_counter()
    predictions = np.asarray(model.predict(features))
    probabilities = np.asarray(model.predict_proba(features))
    elapsed = time.perf_counter() - started
    classes = np.asarray(model.classes_)
    positive = int(np.flatnonzero(classes == 1)[0])
    scores = probabilities[:, positive]
    metrics = {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("T4 model produced non-finite metrics")
    return {
        **metrics,
        "predict_seconds": float(max(0.0, elapsed)),
        "test_rows": int(len(labels)),
        "test_class_counts": {
            str(label): int((labels == label).sum()) for label in (0, 1)
        },
    }


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(record.get("language")),
        str(record.get("method")),
        int(record.get("seed")),
        int(record.get("fold")),
        str(record.get("condition")),
    )


def _t4_record_sha256(record: Mapping[str, Any]) -> str:
    return _digest_json({key: value for key, value in record.items() if key != "record_sha256"})


def _validate_metric_record(record: Mapping[str, Any]) -> None:
    for name in ("f1", "precision", "recall", "auroc", "clean_reference_f1"):
        value = record.get(name)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid T4 metric")
    mcc = record.get("mcc")
    if type(mcc) not in (int, float) or not np.isfinite(mcc) or not -1 <= mcc <= 1:
        raise ValueError("invalid T4 metric")
    for name in ("fit_seconds", "predict_seconds"):
        value = record.get(name)
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("invalid T4 timing")


def _validate_t4_record(record: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != T4_FOLD_RECORD_FIELDS:
        raise ValueError("T4 fold record schema mismatch")
    _validate_metric_record(record)
    language, method, seed, fold, condition = _record_key(record)
    spec = config["method_contract"].get(method)
    if (
        record["schema_version"] != T4_SCHEMA_VERSION
        or record["config_id"] != config["config_id"]
        or language not in config["languages"]
        or method not in config["methods"]
        or seed not in config["seeds"]
        or type(fold) is not int
        or fold not in range(config["n_splits"])
        or condition not in config["conditions"]
        or not isinstance(spec, dict)
        or record["feature_family"] != spec["feature_family"]
        or record["representation"] != spec["representation"]
        or record["model"] != spec["model"]
        or record["feature_dimensions"] != spec["feature_dimensions"]
        or record["split_protocol"] != T4_SPLIT_PROTOCOL_VERSION
        or record["pair_protocol"] != PAIR_PROTOCOL_VERSION
        or record["component_protocol"] != COMPONENT_PROTOCOL_VERSION
        or record["source_jsonl_sha256"] != config["source_jsonl_sha256"][language]
        or record["clean_cache_content_sha256"] != config["clean_cache_content_sha256"][language]
        or record["attack_cache_content_sha256"] != config["attack_cache_content_sha256"][language]
        or record["gate_a_sha256"] != config["gate_a_binding"]["gate_a_sha256"]
        or record["gate_b_sha256"] != config["gate_b_binding"]["gate_b_sha256"]
        or record["record_sha256"] != _t4_record_sha256(record)
    ):
        raise ValueError("T4 fold record schema/config mismatch")
    for name in (
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
        "attack_failures",
        "attack_parse_regressions",
    ):
        if type(record[name]) is not int or record[name] < 0:
            raise ValueError("invalid T4 audit count")
    if any(type(record[name]) is not int or record[name] < 0 for name in (
        "attack_attempted", "attack_successes", "attack_changed", "attack_transform_count",
        "train_rows", "test_rows", "train_unique_sources", "test_unique_sources",
        "train_unique_code_hashes", "test_unique_code_hashes",
        "train_unique_components", "test_unique_components",
    )):
        raise ValueError("invalid T4 count")
    if record["attack_successes"] + record["attack_failures"] != record["attack_attempted"]:
        raise ValueError("invalid T4 attack accounting")
    if record["test_rows"] != record["attack_successes"]:
        raise ValueError("T4 test rows disagree with attack successes")
    if not all(
        isinstance(record[name], str) and len(record[name]) == 64
        for name in (
            "train_index_sha256", "test_index_sha256", "attack_success_set_sha256",
            "attack_output_set_sha256", "source_jsonl_sha256",
            "clean_cache_content_sha256", "attack_cache_content_sha256",
            "gate_a_sha256", "gate_b_sha256", "record_sha256",
        )
    ):
        raise ValueError("invalid T4 digest")
    return record


def _validate_t4_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("invalid T4 config")
    required = {
        "schema_version", "config_id", "task", "languages", "methods", "seeds",
        "n_splits", "conditions", "limit_origins", "full_matrix", "split_protocol",
        "pair_protocol", "component_protocol", "method_contract",
        "source_jsonl_sha256", "clean_cache_content_sha256",
        "attack_cache_content_sha256", "gate_a_binding", "gate_b_binding",
        "attack_contract", "implementation_contract", "package_versions",
    }
    if set(config) != required:
        raise ValueError("invalid T4 config fields")
    if (
        config["schema_version"] != T4_SCHEMA_VERSION
        or config["task"] != "task4_style_attack"
        or config["conditions"] != list(CONDITIONS)
        or config["split_protocol"] != T4_SPLIT_PROTOCOL_VERSION
        or config["pair_protocol"] != PAIR_PROTOCOL_VERSION
        or config["component_protocol"] != COMPONENT_PROTOCOL_VERSION
        or type(config["full_matrix"]) is not bool
        or config["config_id"]
        != _digest_json({key: value for key, value in config.items() if key != "config_id"})
    ):
        raise ValueError("invalid T4 config binding")
    return config


def _load_t4_records(path: str | Path, config: Mapping[str, Any]) -> dict[tuple[str, str, int, int, str], dict[str, Any]]:
    ledger = Path(path)
    if not ledger.exists():
        return {}
    records: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("invalid T4 fold ledger") from exc
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid T4 fold ledger JSON") from exc
        validated = _validate_t4_record(record, config)
        key = _record_key(validated)
        if key in records:
            raise ValueError("duplicate T4 fold record key")
        records[key] = validated
    return records


def _atomic_write_t4_records(
    path: Path,
    records: Mapping[tuple[str, str, int, int, str], Mapping[str, Any]],
) -> None:
    contents = b"".join(_canonical_json(records[key]) for key in sorted(records))
    for attempt in range(15):
        try:
            atomic_write_bytes(path, contents)
            return
        except PermissionError:
            if attempt == 14:
                raise
            time.sleep(min(1.0, 0.02 * (2**attempt)))


def _package_versions() -> dict[str, str]:
    values = {"python": platform.python_version(), "numpy": np.__version__}
    for distribution, key in (
        ("scikit-learn", "scikit_learn"),
        ("xgboost", "xgboost"),
        ("tree-sitter", "tree-sitter"),
        ("tree-sitter-c", "tree-sitter-c"),
        ("tree-sitter-cpp", "tree-sitter-cpp"),
        ("tree-sitter-java", "tree-sitter-java"),
    ):
        values[key] = importlib.metadata.version(distribution)
    return values


def _build_t4_config(
    languages: tuple[str, ...], methods: tuple[str, ...], seeds: tuple[int, ...],
    n_splits: int, limit_origins: int | None, paths: Mapping[str, Path],
    clean_caches: Mapping[str, EnhancedFeatureCache],
    attack_caches: Mapping[str, AttackFeatureCache],
    gate_a: Mapping[str, Any], gate_b: Mapping[str, Any],
) -> dict[str, Any]:
    from . import attacks

    full_matrix = (
        languages == LANGUAGES and methods == METHODS and seeds == DEFAULT_SEEDS
        and n_splits == 5 and limit_origins is None
    )
    payload = {
        "schema_version": T4_SCHEMA_VERSION,
        "task": "task4_style_attack",
        "languages": list(languages),
        "methods": list(methods),
        "seeds": list(seeds),
        "n_splits": n_splits,
        "conditions": list(CONDITIONS),
        "limit_origins": limit_origins,
        "full_matrix": full_matrix,
        "split_protocol": T4_SPLIT_PROTOCOL_VERSION,
        "pair_protocol": PAIR_PROTOCOL_VERSION,
        "component_protocol": COMPONENT_PROTOCOL_VERSION,
        "method_contract": {name: gate_b["method_contract"][name] for name in methods},
        "source_jsonl_sha256": {language: _sha256(paths[language]) for language in languages},
        "clean_cache_content_sha256": {
            language: _digest_json({
                "row_sha256": clean_caches[language].row_sha256.tolist(),
                "human": hashlib.sha256(np.ascontiguousarray(clean_caches[language].human).tobytes()).hexdigest(),
                "llm": hashlib.sha256(np.ascontiguousarray(clean_caches[language].llm).tobytes()).hexdigest(),
            }) for language in languages
        },
        "attack_cache_content_sha256": {
            language: attack_caches[language].semantic_content_sha256 for language in languages
        },
        "gate_a_binding": dict(gate_a),
        "gate_b_binding": dict(gate_b),
        "attack_contract": {
            "version": attacks.ATTACK_VERSION,
            "source_sha256": attacks.attack_source_sha256(),
            "conditions": list(ATTACKS),
            "scope": "test-candidate-endpoint-only",
            "combined_order": ["comment_removal", "identifier_rename", "format_normalization"],
        },
        "implementation_contract": {
            "t4_source_sha256": _sha256(Path(__file__).resolve()),
            "t3_source_sha256": _sha256(Path(build_t1_pair_splits.__code__.co_filename).resolve()),
            "experiment_source_sha256": _sha256(Path(build_model.__code__.co_filename).resolve()),
            "representations_source_sha256": _sha256(Path(build_representation.__code__.co_filename).resolve()),
            "official_features_source_sha256": _sha256(Path(analyze_code.__code__.co_filename).resolve()),
            "enhanced_features_source_sha256": _sha256(Path(analyze_enhanced.__code__.co_filename).resolve()),
        },
        "package_versions": _package_versions(),
    }
    payload["config_id"] = _digest_json(payload)
    return _validate_t4_config(payload)


def _load_or_write_config(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        actual = _validate_t4_config(_load_json(path))
        if actual != expected:
            raise ValueError("existing T4 config does not match requested run")
        return actual
    atomic_write_bytes(path, _canonical_json(expected))
    return expected


def _clean_audit(split: Any) -> dict[str, Any]:
    pairs = split.test_pairs
    return {
        "attempted": len(pairs),
        "successes": len(pairs),
        "failures": 0,
        "changed": 0,
        "transform_count": 0,
        "parse_regressions": 0,
        "success_set_sha256": _digest_json([pair.pair_sha256 for pair in pairs]),
        "output_set_sha256": _digest_json([pair.candidate_code_sha256 for pair in pairs]),
    }


def _condition_matrices(
    clean_cache: EnhancedFeatureCache,
    attack_cache: AttackFeatureCache,
    split: Any,
    method_spec: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    clean_matrix, labels = _t3_pair_matrix(clean_cache, split.test_pairs, dict(method_spec))
    result = {
        "clean": (clean_matrix, clean_matrix, labels, _clean_audit(split))
    }
    human_indices = np.asarray([pair.human_positive_row_idx for pair in split.test_pairs], dtype=np.int64)
    candidate_indices = np.asarray([pair.candidate_positive_row_idx for pair in split.test_pairs], dtype=np.int64)
    row_hashes = clean_cache.row_sha256[candidate_indices]
    for attack in ATTACKS:
        attacked, mask, audit = _attack_rows(attack_cache, row_hashes, attack)
        audit["success_set_sha256"] = _digest_json([
            pair.pair_sha256 for pair, ok in zip(split.test_pairs, mask.tolist()) if ok
        ])
        count = method_spec["feature_count"]
        attacked_matrix = build_representation(
            clean_cache.human[human_indices, :count], attacked[:, :count],
            method_spec["representation"],
        )
        result[attack] = (
            attacked_matrix[mask], clean_matrix[mask], labels[mask], audit
        )
    return result


def _expected_record_binding(
    config: Mapping[str, Any], language: str, method: str, seed: int, fold: int,
    condition: str, split_metadata: Mapping[str, Any], audit: Mapping[str, Any],
) -> dict[str, Any]:
    spec = config["method_contract"][method]
    return {
        "schema_version": T4_SCHEMA_VERSION,
        "config_id": config["config_id"],
        "language": language,
        "method": method,
        "condition": condition,
        "feature_family": spec["feature_family"],
        "representation": spec["representation"],
        "model": spec["model"],
        "feature_dimensions": spec["feature_dimensions"],
        "seed": seed,
        "fold": fold,
        "split_protocol": T4_SPLIT_PROTOCOL_VERSION,
        "pair_protocol": PAIR_PROTOCOL_VERSION,
        "component_protocol": COMPONENT_PROTOCOL_VERSION,
        "attack_attempted": audit["attempted"],
        "attack_successes": audit["successes"],
        "attack_failures": audit["failures"],
        "attack_changed": audit["changed"],
        "attack_transform_count": audit["transform_count"],
        "attack_parse_regressions": audit["parse_regressions"],
        "attack_success_set_sha256": audit["success_set_sha256"],
        "attack_output_set_sha256": audit["output_set_sha256"],
        **{name: split_metadata[name] for name in (
            "leakage_count", "endpoint_leakage_count", "content_leakage_count",
            "negative_component_violation_count", "train_index_sha256", "test_index_sha256",
            "train_unique_sources", "test_unique_sources", "train_unique_code_hashes",
            "test_unique_code_hashes", "train_unique_components", "test_unique_components",
            "train_llm_label_counts", "test_llm_label_counts",
        )},
        "source_jsonl_sha256": config["source_jsonl_sha256"][language],
        "clean_cache_content_sha256": config["clean_cache_content_sha256"][language],
        "attack_cache_content_sha256": config["attack_cache_content_sha256"][language],
        "gate_a_sha256": config["gate_a_binding"]["gate_a_sha256"],
        "gate_b_sha256": config["gate_b_binding"]["gate_b_sha256"],
    }


def _run_t4_locked(
    output_root: str | Path,
    languages: tuple[str, ...] = LANGUAGES,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    clean_cache_root: str | Path = RESULTS_ROOT / "02_unseen_llm" / "cache",
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    attack_cache_root: str | Path = DEFAULT_ATTACK_CACHE_ROOT,
    gate_a_path: str | Path = DEFAULT_GATE_A_PATH,
    gate_b_path: str | Path = DEFAULT_GATE_B_PATH,
) -> dict[str, Any]:
    if (
        not languages or any(language not in LANGUAGES for language in languages)
        or len(set(languages)) != len(languages)
        or not methods or any(method not in METHODS for method in methods)
        or len(set(methods)) != len(methods)
        or not seeds or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or type(n_splits) is not int or n_splits < 2
        or (limit_origins is not None and (type(limit_origins) is not int or limit_origins < n_splits * 2))
    ):
        raise ValueError("invalid T4 axes")
    if dataset_paths is not None and set(dataset_paths) != set(languages):
        raise ValueError("T4 dataset paths must exactly match configured languages")
    paths = {
        language: (
            Path(dataset_paths[language]).resolve() if dataset_paths else
            (REPRO_ROOT / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl").resolve()
        ) for language in languages
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("T4 dataset does not exist")
    gate_a = _load_strict_gate_a(gate_a_path)
    gate_b = _load_strict_gate_b(gate_b_path)
    gate_b_gate_a = gate_b["gate_a_binding"]
    if (
        gate_a["gate_a_sha256"] != gate_b_gate_a.get("gate_a_sha256")
        or gate_a["manifest_sha256"] != gate_b_gate_a.get("manifest_sha256")
        or gate_a["strict_config_id"] != gate_b_gate_a.get("strict_config_id")
    ):
        raise ValueError("T4 Gate A and Gate B provenance mismatch")
    if any(_sha256(paths[language]) != gate_b["source_jsonl_sha256"][language] for language in languages):
        raise ValueError("T4 dataset does not match strict Gate B data")
    clean_caches = {
        language: _select_t3_positive_bank(
            load_or_build_enhanced_cache(language, paths[language], clean_cache_root, official_cache_root),
            limit_origins,
        ) for language in languages
    }
    attack_caches = {
        language: load_or_build_attack_cache(language, paths[language], attack_cache_root)
        for language in languages
    }
    config = _build_t4_config(
        languages, methods, seeds, n_splits, limit_origins, paths,
        clean_caches, attack_caches, gate_a, gate_b,
    )
    output = resolve_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    _load_or_write_config(output / "config.json", config)
    ledger = output / "folds.jsonl"
    records = _load_t4_records(ledger, config)
    completed = skipped = 0
    for language in languages:
        clean_cache = clean_caches[language]
        attack_cache = attack_caches[language]
        for seed in seeds:
            for split in build_t1_pair_splits(clean_cache, language, n_splits, seed):
                split_meta = _split_metadata(split)
                if any(split_meta[name] != 0 for name in (
                    "leakage_count", "endpoint_leakage_count", "content_leakage_count",
                    "negative_component_violation_count",
                )):
                    raise ValueError("T4 strict split leakage")
                for method in methods:
                    spec = config["method_contract"][method]
                    train_matrix, train_labels = _t3_pair_matrix(clean_cache, split.train_pairs, spec)
                    conditions = _condition_matrices(clean_cache, attack_cache, split, spec)
                    existing = True
                    for condition, (_attacked, _clean_ref, _labels, audit) in conditions.items():
                        key = (language, method, seed, split.fold, condition)
                        if key not in records:
                            existing = False
                            continue
                        binding = _expected_record_binding(
                            config, language, method, seed, split.fold, condition, split_meta, audit
                        )
                        if any(records[key][name] != value for name, value in binding.items()):
                            raise ValueError("completed T4 record reconstruction mismatch")
                    if existing:
                        skipped += len(CONDITIONS)
                        continue
                    model = build_model(spec["model"], seed)
                    fit_started = time.perf_counter()
                    model.fit(train_matrix, train_labels)
                    fit_seconds = float(max(0.0, time.perf_counter() - fit_started))
                    for condition, (attacked_matrix, clean_reference_matrix, labels, audit) in conditions.items():
                        key = (language, method, seed, split.fold, condition)
                        if key in records:
                            skipped += 1
                            continue
                        attacked_metrics = _score_model(model, attacked_matrix, labels)
                        clean_reference = _score_model(model, clean_reference_matrix, labels)
                        binding = _expected_record_binding(
                            config, language, method, seed, split.fold, condition, split_meta, audit
                        )
                        record = {
                            **attacked_metrics,
                            **binding,
                            "fit_seconds": fit_seconds,
                            "train_rows": int(len(train_labels)),
                            "train_class_counts": {
                                str(label): int((train_labels == label).sum()) for label in (0, 1)
                            },
                            "clean_reference_f1": clean_reference["f1"],
                        }
                        record["record_sha256"] = _t4_record_sha256(record)
                        _validate_t4_record(record, config)
                        current = _load_t4_records(ledger, config)
                        if key in current:
                            raise ValueError("duplicate T4 fold record key")
                        current[key] = record
                        _atomic_write_t4_records(ledger, current)
                        records = current
                        completed += 1
    expected = _evaluation_count(languages, methods, seeds, n_splits)
    return {
        "schema_version": T4_SCHEMA_VERSION,
        "config_id": config["config_id"],
        "expected": expected,
        "completed": completed,
        "skipped": skipped,
        "output_root": str(output),
    }


def run_t4(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    languages: tuple[str, ...] = LANGUAGES,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    clean_cache_root: str | Path = RESULTS_ROOT / "02_unseen_llm" / "cache",
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    attack_cache_root: str | Path = DEFAULT_ATTACK_CACHE_ROOT,
    gate_a_path: str | Path = DEFAULT_GATE_A_PATH,
    gate_b_path: str | Path = DEFAULT_GATE_B_PATH,
) -> dict[str, Any]:
    output = resolve_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        return _run_t4_locked(
            output, languages, methods, seeds, n_splits, limit_origins,
            dataset_paths, clean_cache_root, official_cache_root,
            attack_cache_root, gate_a_path, gate_b_path,
        )


def run_t4_smoke(
    output_root: str | Path,
    dataset_paths: dict[str, str | Path] | None = None,
    clean_cache_root: str | Path = RESULTS_ROOT / "02_unseen_llm" / "cache",
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    attack_cache_root: str | Path = DEFAULT_ATTACK_CACHE_ROOT,
) -> dict[str, Any]:
    return run_t4(
        output_root,
        languages=("c",),
        methods=METHODS,
        seeds=(42,),
        n_splits=2,
        limit_origins=T4_SMOKE_ORIGINS,
        dataset_paths=dataset_paths,
        clean_cache_root=clean_cache_root,
        official_cache_root=official_cache_root,
        attack_cache_root=attack_cache_root,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--clean-cache-root", type=Path, default=RESULTS_ROOT / "02_unseen_llm" / "cache")
    parser.add_argument("--official-cache-root", type=Path, default=DEFAULT_OFFICIAL_CACHE_ROOT)
    parser.add_argument("--attack-cache-root", type=Path, default=DEFAULT_ATTACK_CACHE_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.summarize_only:
        parser.error("--smoke and --summarize-only cannot be combined")
    if args.summarize_only:
        from .gates_t4 import summarize_t4

        print(json.dumps(summarize_t4(args.output_root), sort_keys=True, separators=(",", ":")))
        return
    runner = run_t4_smoke if args.smoke else run_t4
    result = runner(
        args.output_root,
        clean_cache_root=args.clean_cache_root,
        official_cache_root=args.official_cache_root,
        attack_cache_root=args.attack_cache_root,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTACKS",
    "CONDITIONS",
    "METHODS",
    "AttackFeatureCache",
    "run_t4",
    "run_t4_smoke",
]
