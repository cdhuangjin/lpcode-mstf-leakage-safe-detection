"""Resumable Task 1 evaluation runner and versioned official-feature cache."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_jsonl
from .cache_io import (
    atomic_write_bytes as _atomic_write_bytes,
    exclusive_cache_lock as _exclusive_cache_lock,
    unique_temporary_path as _unique_temporary_path,
)
from .features_official import FEATURE_NAMES, analyze_code
from .experiment import evaluate_fold
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .representations import build_representation
from .splits import assert_no_group_leakage, grouped_folds


CACHE_VERSION = "official10-v2"
FEATURE_COUNT = 10
LANGUAGES = ("c", "cpp", "java", "py")
DEFAULT_SEEDS = (42, 123, 2024)
DEFAULT_REPRESENTATIONS = ("concat", "delta", "concat_delta", "full")
DEFAULT_MODELS = ("mlp", "xgb")
FOLD_SCHEMA_VERSION = 1
METRIC_FIELDS = (
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
)
FOLD_RECORD_FIELDS = frozenset(
    {
        "schema_version", "config_id", "language", "representation", "model", "seed", "fold",
        "leakage_count", "train_index_sha256", "test_index_sha256", "feature_dimensions",
        "train_unique_sources", "test_unique_sources", *METRIC_FIELDS,
    }
)


def _feature_contract() -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": FEATURE_COUNT,
        "official_feature_source_sha256": _sha256(
            Path(__file__).with_name("features_official.py")
        ),
    }


def _feature_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(_feature_contract(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _is_strict_int(value: Any) -> bool:
    return type(value) is int


@dataclass(frozen=True)
class FeatureCache:
    """Validated in-memory contents of a Task 1 feature cache."""

    human: np.ndarray
    llm: np.ndarray
    labels: np.ndarray
    source_ids: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


@contextmanager
def _exclusive_output_lock(output: Path):
    """Hold an advisory OS lock without treating a stale lockfile as a lock."""
    lock_path = output / ".t1.lock"
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise ValueError(f"output root is already locked: {output}") from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _cache_paths(language: str, cache_root: str | Path) -> tuple[Path, Path]:
    if language not in LANGUAGES:
        raise ValueError(f"unsupported Task 1 language: {language!r}")
    root = resolve_output_path(cache_root)
    cache_path = root / CACHE_VERSION / f"{language}.npz"
    return cache_path, cache_path.with_suffix(".json")


def _validate_cache_metadata(metadata: Any) -> dict[str, Any]:
    required = {
        "schema_version", "cache_version", "language", "rows", "feature_names",
        "source_jsonl_sha256", "official_feature_source_sha256",
        "feature_contract_sha256", "npz_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("invalid feature cache metadata")
    if (
        not _is_strict_int(metadata["schema_version"])
        or metadata["schema_version"] != 1
        or type(metadata["cache_version"]) is not str
        or type(metadata["language"]) is not str
        or not _is_strict_int(metadata["rows"])
        or metadata["rows"] < 0
        or type(metadata["feature_names"]) is not list
        or any(type(name) is not str for name in metadata["feature_names"])
        or any(
            not _is_sha256(metadata[field])
            for field in (
                "source_jsonl_sha256",
                "official_feature_source_sha256",
                "feature_contract_sha256",
                "npz_sha256",
            )
        )
    ):
        raise ValueError("invalid feature cache metadata")
    return metadata


def _validate_cache_arrays(
    human: Any, llm: Any, labels: Any, source_ids: Any, expected_rows: int | None = None
) -> FeatureCache:
    human_array = np.asarray(human, dtype=np.float64)
    llm_array = np.asarray(llm, dtype=np.float64)
    labels_array = np.asarray(labels)
    source_ids_array = np.asarray(source_ids)
    if (
        human_array.ndim != 2
        or llm_array.ndim != 2
        or human_array.shape != llm_array.shape
        or human_array.shape[1:] != (FEATURE_COUNT,)
        or not np.isfinite(human_array).all()
        or not np.isfinite(llm_array).all()
    ):
        raise ValueError("invalid feature cache matrices")
    if (
        labels_array.ndim != 1
        or labels_array.shape[0] != human_array.shape[0]
        or labels_array.dtype.kind not in "iu"
        or not set(np.unique(labels_array)).issubset({0, 1})
        or source_ids_array.ndim != 1
        or source_ids_array.shape[0] != human_array.shape[0]
        or source_ids_array.dtype.kind not in "US"
    ):
        raise ValueError("invalid feature cache labels or source ids")
    if expected_rows is not None and human_array.shape[0] != expected_rows:
        raise ValueError("stale cache row count")
    return FeatureCache(
        human=np.asarray(human_array, dtype=np.float64),
        llm=np.asarray(llm_array, dtype=np.float64),
        labels=np.asarray(labels_array, dtype=np.int64),
        source_ids=np.asarray(source_ids_array, dtype=str),
    )


def _load_existing_cache(
    cache_path: Path, metadata_path: Path, language: str, source_hash: str, source_rows: int
) -> FeatureCache:
    if not cache_path.exists() and not metadata_path.exists():
        raise FileNotFoundError
    if not cache_path.exists() or not metadata_path.exists():
        raise ValueError("incomplete feature cache")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid feature cache metadata") from exc
    metadata = _validate_cache_metadata(metadata)
    expected_metadata = {
        "schema_version": 1,
        "cache_version": CACHE_VERSION,
        "language": language,
        "rows": source_rows,
        "feature_names": list(FEATURE_NAMES),
        "source_jsonl_sha256": source_hash,
        "official_feature_source_sha256": _feature_contract()[
            "official_feature_source_sha256"
        ],
        "feature_contract_sha256": _feature_contract_sha256(),
    }
    if any(metadata[key] != value for key, value in expected_metadata.items()):
        raise ValueError("stale cache metadata")
    if not isinstance(metadata.get("npz_sha256"), str) or metadata["npz_sha256"] != _sha256(cache_path):
        raise ValueError("corrupt feature cache")
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != {"human", "llm", "labels", "source_ids"}:
                raise ValueError("invalid feature cache archive")
            if archive["human"].dtype != np.dtype(np.float64) or archive["llm"].dtype != np.dtype(np.float64):
                raise ValueError("invalid feature cache archive dtype")
            return _validate_cache_arrays(
                archive["human"], archive["llm"], archive["labels"], archive["source_ids"], source_rows
            )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("invalid feature cache", "stale cache")):
            raise
        raise ValueError("corrupt feature cache") from exc


def _load_or_build_feature_cache_unlocked(
    language: str, dataset_path: str | Path | None = None, cache_root: str | Path = "results"
) -> FeatureCache:
    """Load one verified official-feature cache or build it once from Task 1 JSONL.

    A cache whose source JSONL hash, metadata, archive digest, or array contract
    differs from the current source is rejected instead of being silently reused.
    """
    dataset = (
        REPRO_ROOT / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl"
        if dataset_path is None
        else Path(dataset_path)
    ).resolve()
    if not dataset.is_file():
        raise ValueError(f"Task 1 dataset does not exist: {dataset}")
    source_hash = _sha256(dataset)
    rows = load_jsonl(dataset, task="task1")
    if not rows:
        raise ValueError(f"Task 1 dataset is empty: {dataset}")
    if _sha256(dataset) != source_hash:
        raise ValueError("Task 1 dataset changed during load")
    cache_path, metadata_path = _cache_paths(language, cache_root)
    try:
        cached = _load_existing_cache(cache_path, metadata_path, language, source_hash, len(rows))
        if _sha256(dataset) != source_hash:
            raise ValueError("Task 1 dataset changed during cache load")
        return cached
    except FileNotFoundError:
        pass

    human = np.vstack([np.asarray(analyze_code(str(row["human_src"]), language), dtype=np.float64) for row in rows])
    llm = np.vstack([np.asarray(analyze_code(str(row["llm_src"]), language), dtype=np.float64) for row in rows])
    if _sha256(dataset) != source_hash:
        raise ValueError("Task 1 dataset changed during feature extraction")
    cache = _validate_cache_arrays(
        human,
        llm,
        np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        np.asarray([str(row["human_source_id"]) for row in rows], dtype=str),
        len(rows),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(cache_path, suffix=".npz")
    try:
        np.savez(temporary, human=cache.human, llm=cache.llm, labels=cache.labels, source_ids=cache.source_ids)
        temporary.replace(cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": 1,
            "cache_version": CACHE_VERSION,
            "language": language,
            "rows": len(rows),
            "feature_names": list(FEATURE_NAMES),
            "source_jsonl_sha256": source_hash,
            "official_feature_source_sha256": _feature_contract()[
                "official_feature_source_sha256"
            ],
            "feature_contract_sha256": _feature_contract_sha256(),
            "npz_sha256": _sha256(cache_path),
        },
    )
    return cache


def load_or_build_feature_cache(
    language: str,
    dataset_path: str | Path | None = None,
    cache_root: str | Path = "results",
) -> FeatureCache:
    """Serialize all read/build/publication work for one cache path/language."""

    cache_path, _metadata_path = _cache_paths(language, cache_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_cache_lock(cache_path.parent / f".{language}.lock"):
        return _load_or_build_feature_cache_unlocked(
            language, dataset_path, cache_root
        )


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _index_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _validate_axes(
    languages: tuple[str, ...], seeds: tuple[int, ...], n_splits: int,
    representations: tuple[str, ...], models: tuple[str, ...], limit_groups: int | None,
) -> None:
    if not languages or any(language not in LANGUAGES for language in languages) or len(set(languages)) != len(languages):
        raise ValueError("languages must be a nonempty unique subset of c cpp java py")
    if not seeds or any(not _is_strict_int(seed) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty unique integers")
    if not _is_strict_int(n_splits) or n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not representations or any(name not in DEFAULT_REPRESENTATIONS for name in representations):
        raise ValueError("unknown representation")
    if len(set(representations)) != len(representations):
        raise ValueError("representations must be unique")
    if not models or any(name not in DEFAULT_MODELS for name in models) or len(set(models)) != len(models):
        raise ValueError("unknown or duplicate model")
    if limit_groups is not None and (not _is_strict_int(limit_groups) or limit_groups < 1):
        raise ValueError("limit_groups must be a positive integer")


def _select_complete_groups(cache: FeatureCache, limit_groups: int | None) -> FeatureCache:
    if limit_groups is None:
        return cache
    selected: list[str] = []
    known: set[str] = set()
    for source_id in cache.source_ids.tolist():
        if source_id not in known:
            known.add(source_id)
            selected.append(source_id)
            if len(selected) == limit_groups:
                break
    keep = np.isin(cache.source_ids, np.asarray(selected, dtype=str))
    return _validate_cache_arrays(cache.human[keep], cache.llm[keep], cache.labels[keep], cache.source_ids[keep])


def _package_versions() -> dict[str, str]:
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def _build_config(
    languages: tuple[str, ...], seeds: tuple[int, ...], n_splits: int,
    representations: tuple[str, ...], models: tuple[str, ...], limit_groups: int | None,
    dataset_paths: dict[str, Path],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "task": "task1",
        "fold_index_base": 0,
        "languages": list(languages),
        "seeds": list(seeds),
        "n_splits": n_splits,
        "representations": list(representations),
        "models": list(models),
        "limit_groups": limit_groups,
        "feature_contract": _feature_contract(),
        "source_jsonl_sha256": {language: _sha256(path) for language, path in dataset_paths.items()},
        "package_versions": _package_versions(),
    }
    payload["config_id"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_run_config(config: Any) -> dict[str, Any]:
    required = {
        "schema_version", "task", "fold_index_base", "languages", "seeds", "n_splits",
        "representations", "models", "limit_groups", "feature_contract", "source_jsonl_sha256",
        "package_versions", "config_id",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("invalid run config")
    list_fields = ("languages", "seeds", "representations", "models")
    if any(type(config[field]) is not list for field in list_fields):
        raise ValueError("invalid run config")
    if (
        not _is_strict_int(config["schema_version"])
        or config["schema_version"] != FOLD_SCHEMA_VERSION
        or type(config["task"]) is not str
        or config["task"] != "task1"
        or not _is_strict_int(config["fold_index_base"])
        or config["fold_index_base"] != 0
        or not _is_strict_int(config["n_splits"])
        or (config["limit_groups"] is not None and not _is_strict_int(config["limit_groups"]))
        or type(config["feature_contract"]) is not dict
        or config["feature_contract"] != _feature_contract()
        or type(config["source_jsonl_sha256"]) is not dict
        or type(config["package_versions"]) is not dict
        or not _is_sha256(config["config_id"])
    ):
        raise ValueError("invalid run config")
    if any(type(value) is not str for value in config["languages"] + config["representations"] + config["models"]):
        raise ValueError("invalid run config")
    if any(not _is_strict_int(seed) for seed in config["seeds"]):
        raise ValueError("invalid run config")
    try:
        _validate_axes(
            tuple(config["languages"]), tuple(config["seeds"]), config["n_splits"],
            tuple(config["representations"]), tuple(config["models"]), config["limit_groups"],
        )
    except ValueError as exc:
        raise ValueError("invalid run config") from exc
    if set(config["source_jsonl_sha256"]) != set(config["languages"]):
        raise ValueError("invalid run config")
    if any(type(language) is not str or not _is_sha256(source_hash) for language, source_hash in config["source_jsonl_sha256"].items()):
        raise ValueError("invalid run config")
    package_fields = {"python", "numpy", "scikit_learn", "xgboost"}
    if set(config["package_versions"]) != package_fields or any(type(version) is not str or not version for version in config["package_versions"].values()):
        raise ValueError("invalid run config")
    payload = {key: value for key, value in config.items() if key != "config_id"}
    if config["config_id"] != hashlib.sha256(_canonical_json(payload)).hexdigest():
        raise ValueError("invalid run config")
    return config


def _load_or_write_config(path: Path, current: dict[str, Any]) -> None:
    _validate_run_config(current)
    if not path.exists():
        _atomic_write_json(path, current)
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid run config") from exc
    _validate_run_config(existing)
    if existing != current:
        raise ValueError("config mismatch: refusing to combine incomparable Task 1 folds")


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    values = (record.get("language"), record.get("representation"), record.get("model"), record.get("seed"), record.get("fold"))
    if (
        not all(isinstance(value, str) for value in values[:3])
        or any(not _is_strict_int(value) for value in values[3:])
    ):
        raise ValueError("invalid fold record key")
    return values  # type: ignore[return-value]


def _validate_metric_value(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("non-finite or invalid fold metric")


def _validate_class_counts(value: Any, rows: int) -> bool:
    if not isinstance(value, dict) or set(value) != {"0", "1"}:
        return False
    counts = list(value.values())
    return all(isinstance(count, int) and not isinstance(count, bool) and count > 0 for count in counts) and sum(counts) == rows


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_fold_record(record: Any, config: dict[str, Any]) -> tuple[str, str, str, int, int]:
    if not isinstance(record, dict) or set(record) != FOLD_RECORD_FIELDS:
        raise ValueError("malformed fold record")
    if (
        not _is_strict_int(record["schema_version"])
        or record["schema_version"] != FOLD_SCHEMA_VERSION
        or type(record["config_id"]) is not str
        or record["config_id"] != config["config_id"]
        or not _is_strict_int(record["leakage_count"])
        or record["leakage_count"] != 0
    ):
        raise ValueError("fold record schema/config mismatch")
    key = _record_key(record)
    language, representation, model, seed, fold = key
    if (
        language not in config["languages"] or representation not in config["representations"]
        or model not in config["models"] or seed not in config["seeds"]
        or fold < 0 or fold >= config["n_splits"]
        or not _is_strict_int(record["feature_dimensions"]) or record["feature_dimensions"] <= 0
    ):
        raise ValueError("fold record key is outside current config")
    expected_dimensions = {"concat": 20, "delta": 10, "concat_delta": 30, "full": 40}
    integer_fields = ("train_rows", "test_rows", "train_unique_sources", "test_unique_sources")
    if (
        record["feature_dimensions"] != expected_dimensions[representation]
        or any(not _is_strict_int(record[field]) or record[field] <= 0 for field in integer_fields)
        or not _validate_class_counts(record["train_class_counts"], record["train_rows"])
        or not _validate_class_counts(record["test_class_counts"], record["test_rows"])
        or not _is_sha256(record["train_index_sha256"])
        or not _is_sha256(record["test_index_sha256"])
    ):
        raise ValueError("invalid fold record schema")
    for metric in ("f1", "precision", "recall", "auroc", "mcc", "fit_seconds", "predict_seconds"):
        _validate_metric_value(record[metric])
    if (
        any(not 0.0 <= float(record[metric]) <= 1.0 for metric in ("f1", "precision", "recall", "auroc"))
        or not -1.0 <= float(record["mcc"]) <= 1.0
        or float(record["fit_seconds"]) < 0.0
        or float(record["predict_seconds"]) < 0.0
    ):
        raise ValueError("invalid fold metric range")
    return key


def _validate_existing_records(path: Path, config: dict[str, Any]) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"malformed fold record at line {number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed fold record at line {number}") from exc
        try:
            key = _validate_fold_record(record, config)
        except ValueError as exc:
            raise ValueError(f"{exc} at line {number}") from exc
        if key in records:
            raise ValueError("duplicate fold record key")
        records[key] = record
    return records


def _atomic_write_records(path: Path, records: dict[tuple[str, str, str, int, int], dict[str, Any]]) -> None:
    ordered = [records[key] for key in sorted(records)]
    _atomic_write_bytes(path, b"".join(_canonical_json(record) for record in ordered))


def _validate_record_split(
    record: dict[str, Any], cache: FeatureCache, matrix: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    train_groups = {str(cache.source_ids[int(index)]) for index in train_idx}
    test_groups = {str(cache.source_ids[int(index)]) for index in test_idx}
    expected = {
        "train_index_sha256": _index_sha256(train_idx),
        "test_index_sha256": _index_sha256(test_idx),
        "feature_dimensions": int(matrix.shape[1]),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_class_counts": _class_counts(cache.labels[train_idx]),
        "test_class_counts": _class_counts(cache.labels[test_idx]),
        "train_unique_sources": len(train_groups),
        "test_unique_sources": len(test_groups),
        "leakage_count": len(train_groups & test_groups),
    }
    if any(record[field] != value for field, value in expected.items()):
        raise ValueError("completed fold record split hash or split metadata mismatch")


def _run_t1_locked(
    output_root: str | Path,
    languages: tuple[str, ...] = LANGUAGES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS,
    models: tuple[str, ...] = DEFAULT_MODELS,
    limit_groups: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed Task 1 matrix while its output root is locked."""
    language_axis, seed_axis = tuple(languages), tuple(seeds)
    representation_axis, model_axis = tuple(representations), tuple(models)
    _validate_axes(language_axis, seed_axis, n_splits, representation_axis, model_axis, limit_groups)
    output = resolve_output_path(output_root)
    paths = {
        language: Path(dataset_paths[language]).resolve() if dataset_paths and language in dataset_paths
        else (REPRO_ROOT / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl").resolve()
        for language in language_axis
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Task 1 dataset does not exist")
    config = _build_config(language_axis, seed_axis, n_splits, representation_axis, model_axis, limit_groups, paths)
    output.mkdir(parents=True, exist_ok=True)
    _load_or_write_config(output / "config.json", config)
    folds_path = output / "folds.jsonl"
    records = _validate_existing_records(folds_path, config)
    completed = 0
    skipped = 0
    total = len(language_axis) * len(seed_axis) * n_splits * len(representation_axis) * len(model_axis)
    for language in language_axis:
        cache = _select_complete_groups(load_or_build_feature_cache(language, paths[language], output / "cache"), limit_groups)
        if _sha256(paths[language]) != config["source_jsonl_sha256"][language]:
            raise ValueError("Task 1 dataset changed since run config")
        rows = [{"human_source_id": str(source), "label": int(label)} for source, label in zip(cache.source_ids, cache.labels)]
        feature_sets = {name: build_representation(cache.human, cache.llm, name) for name in representation_axis}
        for seed in seed_axis:
            folds = grouped_folds(rows, n_splits=n_splits, seed=seed)
            if len(folds) != n_splits:
                raise ValueError("grouped folds did not return requested split count")
            for fold_number, (train_idx, test_idx) in enumerate(folds):
                assert_no_group_leakage(rows, train_idx, test_idx)
                train_labels, test_labels = cache.labels[train_idx], cache.labels[test_idx]
                if np.unique(train_labels).size != 2 or np.unique(test_labels).size != 2:
                    raise ValueError("each Task 1 fold must contain both labels")
                train_groups = {str(cache.source_ids[int(index)]) for index in train_idx}
                test_groups = {str(cache.source_ids[int(index)]) for index in test_idx}
                for representation, matrix in feature_sets.items():
                    for model in model_axis:
                        key = (language, representation, model, seed, fold_number)
                        if key in records:
                            _validate_record_split(records[key], cache, matrix, train_idx, test_idx)
                            skipped += 1
                            continue
                        metrics = evaluate_fold(matrix[train_idx], train_labels, matrix[test_idx], test_labels, model, seed)
                        if not isinstance(metrics, dict) or set(metrics) != set(METRIC_FIELDS):
                            raise ValueError("evaluator result schema does not match Task 1 contract")
                        for metric in ("f1", "precision", "recall", "auroc", "mcc", "fit_seconds", "predict_seconds"):
                            if metric not in metrics:
                                raise ValueError(f"evaluator did not return {metric}")
                            _validate_metric_value(metrics[metric])
                        record = {
                            **metrics,
                            "schema_version": FOLD_SCHEMA_VERSION,
                            "config_id": config["config_id"],
                            "language": language,
                            "representation": representation,
                            "model": model,
                            "seed": seed,
                            "fold": fold_number,
                            "train_unique_sources": len(train_groups),
                            "test_unique_sources": len(test_groups),
                            "leakage_count": len(train_groups & test_groups),
                            "train_index_sha256": _index_sha256(train_idx),
                            "test_index_sha256": _index_sha256(test_idx),
                            "feature_dimensions": int(matrix.shape[1]),
                            "train_rows": int(len(train_idx)),
                            "test_rows": int(len(test_idx)),
                            "train_class_counts": _class_counts(train_labels),
                            "test_class_counts": _class_counts(test_labels),
                        }
                        if record["leakage_count"] != 0:
                            raise AssertionError("human-source leakage")
                        _validate_fold_record(record, config)
                        _validate_record_split(record, cache, matrix, train_idx, test_idx)
                        # Re-read the small JSONL immediately before each replace so a
                        # damaged or concurrently changed file is never overwritten.
                        records = _validate_existing_records(folds_path, config)
                        if key in records:
                            raise ValueError("duplicate fold record key")
                        records[key] = record
                        _atomic_write_records(folds_path, records)
                        completed += 1
    return {"schema_version": FOLD_SCHEMA_VERSION, "config_id": config["config_id"], "expected": total, "completed": completed, "skipped": skipped, "output_root": str(output)}


def run_t1(
    output_root: str | Path,
    languages: tuple[str, ...] = LANGUAGES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS,
    models: tuple[str, ...] = DEFAULT_MODELS,
    limit_groups: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed Task 1 matrix under an exclusive output-root lock."""
    output = resolve_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        return _run_t1_locked(
            output, languages, seeds, n_splits, representations, models, limit_groups, dataset_paths
        )


def run_smoke(output_root: str | Path) -> dict[str, Any]:
    """Run the bounded, complete-group C smoke matrix (16 evaluations)."""
    return run_t1(output_root, ("c",), (42,), 2, DEFAULT_REPRESENTATIONS, DEFAULT_MODELS, limit_groups=20)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT / "01_t1")
    parser.add_argument("--smoke", action="store_true", help="run C/seed-42/two-fold bounded smoke matrix")
    parser.add_argument("--summarize-only", action="store_true", help="write Task 1 summaries without evaluating folds")
    args = parser.parse_args()
    if args.summarize_only:
        if args.smoke:
            parser.error("--smoke and --summarize-only cannot be combined")
        from .gates import summarize_t1

        report = summarize_t1(args.output_root)
    else:
        report = run_smoke(args.output_root) if args.smoke else run_t1(args.output_root)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
