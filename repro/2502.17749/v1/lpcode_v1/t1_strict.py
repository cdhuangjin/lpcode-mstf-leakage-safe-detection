"""Schema-v2 strict-origin Task 1 evaluation runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from . import data, experiment, representations, t1, t3
from .experiment import evaluate_fold
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .representations import build_representation
from .t1 import (
    METRIC_FIELDS,
    _atomic_write_json,
    _atomic_write_records,
    _exclusive_output_lock,
    _package_versions,
    _sha256,
)
from .t3 import (
    EnhancedFeatureCache,
    T1PairSplit,
    T3PairSpec,
    build_t1_pair_splits,
    load_or_build_enhanced_cache,
)


LANGUAGES = ("c", "cpp", "java", "py")
DEFAULT_SEEDS = (42, 123, 2024)
DEFAULT_REPRESENTATIONS = ("concat", "delta", "concat_delta", "full")
DEFAULT_MODELS = ("mlp", "xgb")
FOLD_SCHEMA_VERSION = 2
SPLIT_PROTOCOL_VERSION = "all-llm-strict-origin-v2"
FEATURE_COUNT = 10
SMOKE_ORIGINS = 8
FOLD_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "config_id",
        "language",
        "representation",
        "model",
        "seed",
        "fold",
        "split_protocol",
        "pair_protocol",
        "component_protocol",
        "record_sha256",
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
        "train_index_sha256",
        "test_index_sha256",
        "feature_dimensions",
        "train_unique_sources",
        "test_unique_sources",
        "train_unique_code_hashes",
        "test_unique_code_hashes",
        "train_unique_components",
        "test_unique_components",
        "train_llm_label_counts",
        "test_llm_label_counts",
        *METRIC_FIELDS,
    }
)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_strict_int(value: Any) -> bool:
    return type(value) is int


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_axes(
    languages: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    representations: tuple[str, ...],
    models: tuple[str, ...],
    limit_origins: int | None,
) -> None:
    if (
        not languages
        or len(set(languages)) != len(languages)
        or any(language not in LANGUAGES for language in languages)
    ):
        raise ValueError("languages must be a nonempty unique supported subset")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not _is_strict_int(seed) for seed in seeds)
    ):
        raise ValueError("seeds must be nonempty unique integers")
    if not _is_strict_int(n_splits) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if (
        not representations
        or len(set(representations)) != len(representations)
        or any(name not in DEFAULT_REPRESENTATIONS for name in representations)
    ):
        raise ValueError("representations must be a nonempty unique supported subset")
    if (
        not models
        or len(set(models)) != len(models)
        or any(name not in DEFAULT_MODELS for name in models)
    ):
        raise ValueError("models must be a nonempty unique supported subset")
    if limit_origins is not None and (
        not _is_strict_int(limit_origins) or limit_origins < 2 * n_splits
    ):
        raise ValueError("limit_origins must provide at least two origins per fold")


def _evaluation_count(
    languages: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    representations: tuple[str, ...],
    models: tuple[str, ...],
) -> int:
    return (
        len(languages)
        * len(seeds)
        * n_splits
        * len(representations)
        * len(models)
    )


def _feature_contract() -> dict[str, Any]:
    enhanced_contract = t3._feature_contract()
    official_contract = t1._feature_contract()
    return {
        "enhanced_cache_version": t3.CACHE_VERSION,
        "enhanced_feature_contract": enhanced_contract,
        "enhanced_feature_contract_sha256": hashlib.sha256(
            _canonical_json(enhanced_contract)
        ).hexdigest(),
        "official_cache_version": t1.CACHE_VERSION,
        "official_feature_contract": official_contract,
        "official_feature_contract_sha256": hashlib.sha256(
            _canonical_json(official_contract)
        ).hexdigest(),
        "selected_columns": [0, FEATURE_COUNT],
        "selected_feature_count": FEATURE_COUNT,
        "selected_feature_names": list(t3.OFFICIAL_FEATURE_NAMES),
    }


def _bound_package_versions() -> dict[str, str]:
    return {**t3._package_contract(), **_package_versions()}


def _implementation_contract() -> dict[str, str]:
    return {
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "experiment_source_sha256": _sha256(Path(experiment.__file__).resolve()),
        "representations_source_sha256": _sha256(
            Path(representations.__file__).resolve()
        ),
        "pair_builder_source_sha256": _sha256(Path(t3.__file__).resolve()),
        "data_normalization_source_sha256": _sha256(Path(data.__file__).resolve()),
        "official_cache_runner_source_sha256": _sha256(Path(t1.__file__).resolve()),
    }


def _build_config(
    languages: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    representations: tuple[str, ...],
    models: tuple[str, ...],
    limit_origins: int | None,
    dataset_paths: dict[str, Path],
) -> dict[str, Any]:
    _validate_axes(
        languages, seeds, n_splits, representations, models, limit_origins
    )
    if set(dataset_paths) != set(languages):
        raise ValueError("dataset paths must exactly match configured languages")
    payload: dict[str, Any] = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "task": "task1_strict_origins",
        "fold_index_base": 0,
        "languages": list(languages),
        "seeds": list(seeds),
        "n_splits": n_splits,
        "representations": list(representations),
        "models": list(models),
        "limit_origins": limit_origins,
        "split_protocol": SPLIT_PROTOCOL_VERSION,
        "pair_protocol": t3.PAIR_PROTOCOL_VERSION,
        "component_protocol": t3.COMPONENT_PROTOCOL_VERSION,
        "implementation_contract": _implementation_contract(),
        "feature_contract": _feature_contract(),
        "source_jsonl_sha256": {
            language: _sha256(dataset_paths[language]) for language in languages
        },
        "package_versions": _bound_package_versions(),
    }
    payload["config_id"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_run_config(config: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "task",
        "fold_index_base",
        "languages",
        "seeds",
        "n_splits",
        "representations",
        "models",
        "limit_origins",
        "split_protocol",
        "pair_protocol",
        "component_protocol",
        "implementation_contract",
        "feature_contract",
        "source_jsonl_sha256",
        "package_versions",
        "config_id",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("invalid strict-origin run config")
    if any(
        type(config[field]) is not list
        for field in ("languages", "seeds", "representations", "models")
    ):
        raise ValueError("invalid strict-origin run config")
    if (
        not _is_strict_int(config["schema_version"])
        or config["schema_version"] != FOLD_SCHEMA_VERSION
        or config["task"] != "task1_strict_origins"
        or not _is_strict_int(config["fold_index_base"])
        or config["fold_index_base"] != 0
        or not _is_strict_int(config["n_splits"])
        or (
            config["limit_origins"] is not None
            and not _is_strict_int(config["limit_origins"])
        )
        or config["split_protocol"] != SPLIT_PROTOCOL_VERSION
        or config["pair_protocol"] != t3.PAIR_PROTOCOL_VERSION
        or config["component_protocol"] != t3.COMPONENT_PROTOCOL_VERSION
        or config["implementation_contract"] != _implementation_contract()
        or config["feature_contract"] != _feature_contract()
        or not isinstance(config["source_jsonl_sha256"], dict)
        or not isinstance(config["package_versions"], dict)
        or config["package_versions"] != _bound_package_versions()
        or not _is_sha256(config["config_id"])
    ):
        raise ValueError("invalid strict-origin run config")
    try:
        _validate_axes(
            tuple(config["languages"]),
            tuple(config["seeds"]),
            config["n_splits"],
            tuple(config["representations"]),
            tuple(config["models"]),
            config["limit_origins"],
        )
    except ValueError as exc:
        raise ValueError("invalid strict-origin run config") from exc
    if (
        set(config["source_jsonl_sha256"]) != set(config["languages"])
        or any(
            not _is_sha256(value)
            for value in config["source_jsonl_sha256"].values()
        )
        or any(
            type(key) is not str or type(value) is not str or not value
            for key, value in config["package_versions"].items()
        )
    ):
        raise ValueError("invalid strict-origin run config")
    payload = {key: value for key, value in config.items() if key != "config_id"}
    if config["config_id"] != hashlib.sha256(_canonical_json(payload)).hexdigest():
        raise ValueError("invalid strict-origin run config")
    return config


def _load_or_write_config(path: Path, current: dict[str, Any]) -> None:
    _validate_run_config(current)
    if not path.exists():
        _atomic_write_json(path, current)
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid strict-origin run config") from exc
    _validate_run_config(existing)
    if existing != current:
        raise ValueError(
            "config mismatch: refusing to combine incomparable strict-origin folds"
        )


def _assert_dataset_unchanged(path: Path, expected_sha256: str) -> None:
    if _sha256(path) != expected_sha256:
        raise ValueError("Task 1 dataset changed since strict-origin run config")


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    values = (
        record.get("language"),
        record.get("representation"),
        record.get("model"),
        record.get("seed"),
        record.get("fold"),
    )
    if (
        any(type(value) is not str for value in values[:3])
        or any(not _is_strict_int(value) for value in values[3:])
    ):
        raise ValueError("invalid strict-origin fold record key")
    return values  # type: ignore[return-value]


def _record_sha256(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_class_counts(value: Any, rows: int) -> bool:
    if not isinstance(value, dict) or set(value) != {"0", "1"}:
        return False
    return (
        all(_is_strict_int(count) and count > 0 for count in value.values())
        and sum(value.values()) == rows
        and value["0"] == value["1"]
    )


def _validate_llm_label_counts(value: Any, rows: int) -> bool:
    if not isinstance(value, dict) or set(value) != set(t3.LLM_SOURCES):
        return False
    if any(
        not isinstance(counts, dict)
        or set(counts) != {"0", "1"}
        or any(not _is_strict_int(count) or count <= 0 for count in counts.values())
        or counts["0"] != counts["1"]
        for counts in value.values()
    ):
        return False
    return sum(sum(counts.values()) for counts in value.values()) == rows


def _validate_metric_value(value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError("non-finite or invalid strict-origin fold metric")


def _validate_evaluator_metrics(metrics: dict[str, Any]) -> None:
    for field in (
        "f1",
        "precision",
        "recall",
        "auroc",
        "mcc",
        "fit_seconds",
        "predict_seconds",
    ):
        _validate_metric_value(metrics[field])
    if (
        any(
            not 0.0 <= float(metrics[field]) <= 1.0
            for field in ("f1", "precision", "recall", "auroc")
        )
        or not -1.0 <= float(metrics["mcc"]) <= 1.0
        or float(metrics["fit_seconds"]) < 0.0
        or float(metrics["predict_seconds"]) < 0.0
    ):
        raise ValueError("invalid strict-origin fold metric range")


def _validate_fold_record(
    record: Any, config: dict[str, Any]
) -> tuple[str, str, str, int, int]:
    if not isinstance(record, dict) or set(record) != FOLD_RECORD_FIELDS:
        raise ValueError("malformed strict-origin fold record schema")
    if (
        not _is_strict_int(record["schema_version"])
        or record["schema_version"] != FOLD_SCHEMA_VERSION
        or type(record["config_id"]) is not str
        or record["config_id"] != config["config_id"]
        or record["split_protocol"] != SPLIT_PROTOCOL_VERSION
        or record["pair_protocol"] != t3.PAIR_PROTOCOL_VERSION
        or record["component_protocol"] != t3.COMPONENT_PROTOCOL_VERSION
    ):
        raise ValueError("strict-origin fold record schema/config mismatch")
    key = _record_key(record)
    language, representation, model, seed, fold = key
    expected_dimensions = {
        "concat": 20,
        "delta": 10,
        "concat_delta": 30,
        "full": 40,
    }
    if (
        language not in config["languages"]
        or representation not in config["representations"]
        or model not in config["models"]
        or seed not in config["seeds"]
        or fold < 0
        or fold >= config["n_splits"]
        or not _is_strict_int(record["feature_dimensions"])
        or record["feature_dimensions"] != expected_dimensions[representation]
        or not _is_sha256(record["train_index_sha256"])
        or not _is_sha256(record["test_index_sha256"])
    ):
        raise ValueError("invalid strict-origin fold record schema")
    zero_count_fields = (
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
    )
    positive_integer_fields = (
        "train_rows",
        "test_rows",
        "train_unique_sources",
        "test_unique_sources",
        "train_unique_code_hashes",
        "test_unique_code_hashes",
        "train_unique_components",
        "test_unique_components",
    )
    if (
        any(not _is_strict_int(record[field]) for field in zero_count_fields)
        or any(record[field] < 0 for field in zero_count_fields)
        or any(
            not _is_strict_int(record[field]) or record[field] <= 0
            for field in positive_integer_fields
        )
        or not _validate_class_counts(
            record["train_class_counts"], record["train_rows"]
        )
        or not _validate_class_counts(
            record["test_class_counts"], record["test_rows"]
        )
        or not _validate_llm_label_counts(
            record["train_llm_label_counts"], record["train_rows"]
        )
        or not _validate_llm_label_counts(
            record["test_llm_label_counts"], record["test_rows"]
        )
    ):
        raise ValueError("invalid strict-origin fold record schema")
    for field in (
        "f1",
        "precision",
        "recall",
        "auroc",
        "mcc",
        "fit_seconds",
        "predict_seconds",
    ):
        _validate_metric_value(record[field])
    if (
        any(
            not 0.0 <= float(record[field]) <= 1.0
            for field in ("f1", "precision", "recall", "auroc")
        )
        or not -1.0 <= float(record["mcc"]) <= 1.0
        or float(record["fit_seconds"]) < 0.0
        or float(record["predict_seconds"]) < 0.0
    ):
        raise ValueError("invalid strict-origin fold metric range")
    if (
        not _is_sha256(record["record_sha256"])
        or record["record_sha256"] != _record_sha256(record)
    ):
        raise ValueError("invalid strict-origin fold record digest")
    return key


def _validate_existing_records(
    path: Path, config: dict[str, Any]
) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"malformed strict-origin fold record at line {number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed strict-origin fold record at line {number}"
            ) from exc
        try:
            key = _validate_fold_record(record, config)
        except ValueError as exc:
            raise ValueError(f"{exc} at line {number}") from exc
        if key in records:
            raise ValueError("duplicate strict-origin fold record key")
        records[key] = record
    return records


def _select_positive_bank(
    cache: EnhancedFeatureCache, limit_origins: int | None
) -> EnhancedFeatureCache:
    positive = cache.labels == 1
    origins = sorted(set(cache.human_origin_ids[positive].tolist()))
    if limit_origins is not None:
        origins = origins[:limit_origins]
    keep = positive & np.isin(cache.human_origin_ids, np.asarray(origins, dtype=str))
    if not keep.any():
        raise ValueError("strict-origin positive bank is empty")
    fields = {
        field: np.asarray(getattr(cache, field))[keep]
        for field in cache.__dataclass_fields__
        if field != "language"
    }
    return replace(cache, **fields)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(label): int((labels == label).sum()) for label in (0, 1)}


def _llm_label_counts(pairs: tuple[T3PairSpec, ...]) -> dict[str, dict[str, int]]:
    return {
        source: {
            str(label): sum(
                pair.llm_source == source and pair.label == label for pair in pairs
            )
            for label in (0, 1)
        }
        for source in t3.LLM_SOURCES
    }


def _side_metadata(prefix: str, pairs: tuple[T3PairSpec, ...]) -> dict[str, Any]:
    endpoints = {
        endpoint
        for pair in pairs
        for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
    }
    code_hashes = {
        code_hash
        for pair in pairs
        for code_hash in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    components = {
        component
        for pair in pairs
        for component in (pair.human_component_id, pair.candidate_component_id)
    }
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    return {
        f"{prefix}_rows": len(pairs),
        f"{prefix}_class_counts": _class_counts(labels),
        f"{prefix}_unique_sources": len(endpoints),
        f"{prefix}_unique_code_hashes": len(code_hashes),
        f"{prefix}_unique_components": len(components),
        f"{prefix}_llm_label_counts": _llm_label_counts(pairs),
    }


def _split_metadata(split: T1PairSplit) -> dict[str, Any]:
    train_endpoints = {
        endpoint
        for pair in split.train_pairs
        for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
    }
    test_endpoints = {
        endpoint
        for pair in split.test_pairs
        for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
    }
    train_hashes = {
        code_hash
        for pair in split.train_pairs
        for code_hash in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    test_hashes = {
        code_hash
        for pair in split.test_pairs
        for code_hash in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    negative_violations = sum(
        pair.human_component_id == pair.candidate_component_id
        for pair in (*split.train_pairs, *split.test_pairs)
        if pair.label == 0
    )
    endpoint_leakage = len(train_endpoints & test_endpoints)
    content_leakage = len(train_hashes & test_hashes)
    return {
        "leakage_count": endpoint_leakage + content_leakage,
        "endpoint_leakage_count": endpoint_leakage,
        "content_leakage_count": content_leakage,
        "negative_component_violation_count": negative_violations,
        "train_index_sha256": split.train_pair_sha256,
        "test_index_sha256": split.test_pair_sha256,
        **_side_metadata("train", split.train_pairs),
        **_side_metadata("test", split.test_pairs),
    }


def _pair_matrices(
    cache: EnhancedFeatureCache,
    pairs: tuple[T3PairSpec, ...],
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    human_indices = np.asarray(
        [pair.human_positive_row_idx for pair in pairs], dtype=np.int64
    )
    candidate_indices = np.asarray(
        [pair.candidate_positive_row_idx for pair in pairs], dtype=np.int64
    )
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    matrix = build_representation(
        cache.human[human_indices, :FEATURE_COUNT],
        cache.llm[candidate_indices, :FEATURE_COUNT],
        representation,
    )
    return matrix, labels


def _validate_record_split(
    record: dict[str, Any], split: T1PairSplit, feature_dimensions: int
) -> None:
    expected = {**_split_metadata(split), "feature_dimensions": feature_dimensions}
    digest_fields = ("train_index_sha256", "test_index_sha256")
    if any(record[field] != expected[field] for field in digest_fields):
        raise ValueError("completed strict-origin fold pair digest mismatch")
    if any(
        record[field] != value
        for field, value in expected.items()
        if field not in digest_fields
    ):
        raise ValueError("completed strict-origin fold split metadata mismatch")


def _run_t1_strict_locked(
    output_root: str | Path,
    languages: tuple[str, ...] = LANGUAGES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS,
    models: tuple[str, ...] = DEFAULT_MODELS,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path | None = None,
    official_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    language_axis, seed_axis = tuple(languages), tuple(seeds)
    representation_axis, model_axis = tuple(representations), tuple(models)
    _validate_axes(
        language_axis,
        seed_axis,
        n_splits,
        representation_axis,
        model_axis,
        limit_origins,
    )
    output = resolve_output_path(output_root)
    paths = {
        language: (
            Path(dataset_paths[language]).resolve()
            if dataset_paths and language in dataset_paths
            else (
                REPRO_ROOT
                / "code"
                / "experiment"
                / "task1"
                / "dataset"
                / f"{language}.jsonl"
            ).resolve()
        )
        for language in language_axis
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Task 1 dataset does not exist")
    config = _build_config(
        language_axis,
        seed_axis,
        n_splits,
        representation_axis,
        model_axis,
        limit_origins,
        paths,
    )
    output.mkdir(parents=True, exist_ok=True)
    _load_or_write_config(output / "config.json", config)
    folds_path = output / "folds.jsonl"
    records = _validate_existing_records(folds_path, config)
    completed = 0
    skipped = 0
    selected_cache_root = Path(cache_root).resolve() if cache_root else output / "cache"
    selected_official_root = (
        Path(official_cache_root).resolve()
        if official_cache_root
        else output / "cache"
    )
    for language in language_axis:
        cache = load_or_build_enhanced_cache(
            language,
            paths[language],
            selected_cache_root,
            selected_official_root,
        )
        cache = _select_positive_bank(cache, limit_origins)
        source_sha256 = config["source_jsonl_sha256"][language]
        _assert_dataset_unchanged(paths[language], source_sha256)
        for seed in seed_axis:
            splits = build_t1_pair_splits(
                cache, language=language, n_splits=n_splits, seed=seed
            )
            if len(splits) != n_splits:
                raise ValueError("strict-origin pair builder returned wrong fold count")
            for split in splits:
                split_metadata = _split_metadata(split)
                if any(
                    split_metadata[field] != 0
                    for field in (
                        "leakage_count",
                        "endpoint_leakage_count",
                        "content_leakage_count",
                        "negative_component_violation_count",
                    )
                ):
                    raise ValueError("strict-origin pair builder violated isolation")
                for representation in representation_axis:
                    train_matrix, train_labels = _pair_matrices(
                        cache, split.train_pairs, representation
                    )
                    test_matrix, test_labels = _pair_matrices(
                        cache, split.test_pairs, representation
                    )
                    for model in model_axis:
                        key = (language, representation, model, seed, split.fold)
                        if key in records:
                            _validate_record_split(
                                records[key], split, int(train_matrix.shape[1])
                            )
                            _assert_dataset_unchanged(paths[language], source_sha256)
                            skipped += 1
                            continue
                        _assert_dataset_unchanged(paths[language], source_sha256)
                        metrics = evaluate_fold(
                            train_matrix,
                            train_labels,
                            test_matrix,
                            test_labels,
                            model,
                            seed,
                        )
                        _assert_dataset_unchanged(paths[language], source_sha256)
                        if not isinstance(metrics, dict) or set(metrics) != set(
                            METRIC_FIELDS
                        ):
                            raise ValueError(
                                "evaluator result schema does not match strict-origin contract"
                            )
                        _validate_evaluator_metrics(metrics)
                        record = {
                            **metrics,
                            "schema_version": FOLD_SCHEMA_VERSION,
                            "config_id": config["config_id"],
                            "language": language,
                            "representation": representation,
                            "model": model,
                            "seed": seed,
                            "fold": split.fold,
                            "split_protocol": SPLIT_PROTOCOL_VERSION,
                            "pair_protocol": t3.PAIR_PROTOCOL_VERSION,
                            "component_protocol": t3.COMPONENT_PROTOCOL_VERSION,
                            "feature_dimensions": int(train_matrix.shape[1]),
                            **split_metadata,
                        }
                        record["record_sha256"] = _record_sha256(record)
                        _validate_fold_record(record, config)
                        _validate_record_split(
                            record, split, int(train_matrix.shape[1])
                        )
                        records = _validate_existing_records(folds_path, config)
                        if key in records:
                            raise ValueError("duplicate strict-origin fold record key")
                        records[key] = record
                        _assert_dataset_unchanged(paths[language], source_sha256)
                        _atomic_write_records(folds_path, records)
                        _assert_dataset_unchanged(paths[language], source_sha256)
                        completed += 1
        _assert_dataset_unchanged(paths[language], source_sha256)
    expected = _evaluation_count(
        language_axis,
        seed_axis,
        n_splits,
        representation_axis,
        model_axis,
    )
    return {
        "schema_version": FOLD_SCHEMA_VERSION,
        "config_id": config["config_id"],
        "expected": expected,
        "completed": completed,
        "skipped": skipped,
        "output_root": str(output),
    }


def run_t1_strict(
    output_root: str | Path,
    languages: tuple[str, ...] = LANGUAGES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS,
    models: tuple[str, ...] = DEFAULT_MODELS,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path | None = None,
    official_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate strict-origin Task 1 under one exclusive output-root lock."""

    output = resolve_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        return _run_t1_strict_locked(
            output,
            languages,
            seeds,
            n_splits,
            representations,
            models,
            limit_origins,
            dataset_paths,
            cache_root,
            official_cache_root,
        )


def run_smoke(
    output_root: str | Path,
    dataset_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path | None = None,
    official_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the bounded 16-evaluation strict-origin C smoke matrix."""

    return run_t1_strict(
        output_root,
        ("c",),
        (42,),
        2,
        DEFAULT_REPRESENTATIONS,
        DEFAULT_MODELS,
        SMOKE_ORIGINS,
        dataset_paths,
        cache_root,
        official_cache_root,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT / "01_transition_test_strict_origins",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="run bounded C/seed-42/two-fold matrix"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="reuse or publish the enhanced28-v3 cache at this root",
    )
    parser.add_argument(
        "--official-cache-root",
        type=Path,
        help="reuse or publish the official10-v2 cache at this root",
    )
    args = parser.parse_args()
    if args.smoke:
        report = run_smoke(
            args.output_root,
            cache_root=args.cache_root,
            official_cache_root=args.official_cache_root,
        )
    else:
        report = run_t1_strict(
            args.output_root,
            cache_root=args.cache_root,
            official_cache_root=args.official_cache_root,
        )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_REPRESENTATIONS",
    "DEFAULT_SEEDS",
    "FOLD_SCHEMA_VERSION",
    "LANGUAGES",
    "run_smoke",
    "run_t1_strict",
]


if __name__ == "__main__":
    main()
