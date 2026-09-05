"""Strict leave-one-programming-language-out experiment (T5)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .cache_io import atomic_write_bytes
from .experiment import evaluate_fold
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .t1 import _exclusive_output_lock
from .t3 import (
    COMPONENT_PROTOCOL_VERSION,
    DEFAULT_OFFICIAL_CACHE_ROOT,
    EnhancedFeatureCache,
    LANGUAGES,
    LLM_SOURCES,
    PAIR_PROTOCOL_VERSION,
    T3_METHODS,
    T3PairSpec,
    _canonical_json,
    _digest_json,
    _method_contract,
    _select_t3_positive_bank,
    _sha256,
    _t3_pair_matrix,
    build_t1_pair_splits,
    load_or_build_enhanced_cache,
)


METHODS = T3_METHODS
DEFAULT_SEEDS = (42, 123, 2024)
DEFAULT_GATE_C_PATH = RESULTS_ROOT / "03_style_attack" / "gate_c.json"
DEFAULT_CLEAN_CACHE_ROOT = RESULTS_ROOT / "02_unseen_llm" / "cache"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "04_cross_language"
T5_SCHEMA_VERSION = 1
T5_SPLIT_PROTOCOL_VERSION = "leave-one-language-strict-origin-v1"
T5_BANK_PROTOCOL_VERSION = "complete-language-bank-from-strict-folds-v1"
T5_PAIR_FOLDS = 5
T5_SMOKE_ORIGINS = 10


T5_RECORD_FIELDS = frozenset(
    {
        "schema_version", "config_id", "heldout_language", "method",
        "feature_family", "representation", "model", "feature_dimensions",
        "seed", "split_protocol", "bank_protocol", "pair_protocol",
        "component_protocol", "f1", "precision", "recall", "auroc", "mcc",
        "fit_seconds", "predict_seconds", "train_rows", "test_rows",
        "train_class_counts", "test_class_counts", "train_languages",
        "test_language", "train_unique_code_hashes", "test_unique_code_hashes",
        "content_leakage_count", "train_index_sha256", "test_index_sha256",
        "train_bank_sha256", "test_bank_sha256", "source_jsonl_sha256",
        "cache_content_sha256", "gate_c_sha256", "gate_c_manifest_sha256",
        "record_sha256",
    }
)


@dataclass(frozen=True)
class LanguagePairBank:
    language: str
    seed: int
    n_pair_folds: int
    pairs: tuple[T3PairSpec, ...]
    pair_sha256: str
    audit: dict[str, Any]


def _evaluation_count(
    heldout_languages: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> int:
    return len(heldout_languages) * len(methods) * len(seeds)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def build_language_pair_bank(
    cache: EnhancedFeatureCache,
    language: str | None = None,
    seed: int = 42,
    n_pair_folds: int = T5_PAIR_FOLDS,
) -> LanguagePairBank:
    if not isinstance(cache, EnhancedFeatureCache):
        raise TypeError("cache must be an EnhancedFeatureCache")
    requested = cache.language if language is None else language
    if requested != cache.language or requested not in LANGUAGES:
        raise ValueError("language-pair bank language mismatch")
    if type(seed) is not int or type(n_pair_folds) is not int or n_pair_folds < 2:
        raise ValueError("invalid language-pair bank seed/fold count")
    splits = build_t1_pair_splits(cache, requested, n_pair_folds, seed)
    if len(splits) != n_pair_folds:
        raise ValueError("strict pair builder returned wrong fold count")
    pairs = tuple(pair for split in sorted(splits, key=lambda item: item.fold) for pair in split.test_pairs)
    cells = [(pair.human_origin_id, pair.llm_source, pair.label) for pair in pairs]
    if len(cells) != len(set(cells)):
        raise ValueError("language-pair bank repeats an anchor/source/label cell")
    origins = set(map(str, cache.human_origin_ids.tolist()))
    positive_origins = {
        str(cache.human_origin_ids[index])
        for index in np.flatnonzero(cache.labels == 1).tolist()
    }
    if {pair.human_origin_id for pair in pairs} != positive_origins or positive_origins != origins:
        raise ValueError("language-pair bank does not use every selected origin")
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    class_counts = {str(label): int((labels == label).sum()) for label in (0, 1)}
    if class_counts["0"] != class_counts["1"]:
        raise ValueError("language-pair bank is not class balanced")
    llm_label_counts = {
        source: {
            str(label): sum(
                pair.llm_source == source and pair.label == label for pair in pairs
            )
            for label in (0, 1)
        }
        for source in LLM_SOURCES
    }
    if any(counts["0"] != counts["1"] for counts in llm_label_counts.values()):
        raise ValueError("language-pair bank is not balanced per LLM")
    violations = sum(
        pair.human_component_id == pair.candidate_component_id
        for pair in pairs
        if pair.label == 0
    )
    if violations:
        raise ValueError("language-pair bank has within-component negative")
    pair_sha = _digest_json([pair.pair_sha256 for pair in pairs])
    return LanguagePairBank(
        language=requested,
        seed=seed,
        n_pair_folds=n_pair_folds,
        pairs=pairs,
        pair_sha256=pair_sha,
        audit={
            "rows": len(pairs),
            "class_counts": class_counts,
            "unique_origins": len(positive_origins),
            "unique_code_hashes": len(
                {
                    code_hash
                    for pair in pairs
                    for code_hash in (pair.human_code_sha256, pair.candidate_code_sha256)
                }
            ),
            "unique_components": len(
                {
                    component
                    for pair in pairs
                    for component in (pair.human_component_id, pair.candidate_component_id)
                }
            ),
            "llm_label_counts": llm_label_counts,
            "negative_component_violation_count": violations,
        },
    )


def _load_strict_gate_c(path: str | Path) -> dict[str, Any]:
    requested = Path(path).resolve()
    expected = DEFAULT_GATE_C_PATH.resolve()
    if requested != expected or not requested.is_file():
        raise ValueError("T5 requires the exact Gate C artifact")
    root = requested.parent
    manifest_path = root / "manifest.json"
    config_path = root / "config.json"
    summary_path = root / "summary.json"
    for item in (manifest_path, config_path, summary_path, root / "folds.jsonl"):
        if not item.is_file():
            raise ValueError("T5 exact Gate C artifact is incomplete")
    gate = _load_json(requested)
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    summary = _load_json(summary_path)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ValueError("T5 exact Gate C manifest is malformed")
    for name in ("gate_c.json", "config.json", "summary.json", "folds.jsonl"):
        item = root / name
        spec = files.get(name)
        if (
            not isinstance(spec, dict)
            or spec.get("sha256") != _sha256(item)
            or spec.get("bytes") != item.stat().st_size
        ):
            raise ValueError("T5 exact Gate C manifest hash mismatch")
    strict = gate.get("strict") if isinstance(gate, dict) else None
    matrix = summary.get("matrix") if isinstance(summary, dict) else None
    if (
        gate.get("status") != "evaluable"
        or not isinstance(strict, dict)
        or strict.get("passed") is not True
        or strict.get("dual_criterion") is not True
        or strict.get("attacked_f1_advantage", -1) < 0.05
        or strict.get("relative_drop_reduction", -1) < 0.30
        or not isinstance(matrix, dict)
        or matrix.get("complete_cartesian_product") is not True
        or matrix.get("observed_records") != 1440
        or config.get("languages") != list(LANGUAGES)
        or config.get("methods") != list(METHODS)
        or config.get("seeds") != list(DEFAULT_SEEDS)
    ):
        raise ValueError("T5 requires a strict passing exact Gate C artifact")
    return {
        "strict_passed": True,
        "authorizes_t5": True,
        "dual_criterion": True,
        "attacked_f1_advantage": float(strict["attacked_f1_advantage"]),
        "relative_drop_reduction": float(strict["relative_drop_reduction"]),
        "gate_c_path": str(requested),
        "gate_c_sha256": _sha256(requested),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": _sha256(config_path),
        "config_id": config["config_id"],
        "source_jsonl_sha256": config["source_jsonl_sha256"],
        "clean_cache_content_sha256": config["clean_cache_content_sha256"],
        "method_contract": config["method_contract"],
        "gate_a_binding": manifest.get("gate_a_binding"),
        "gate_b_binding": manifest.get("gate_b_binding"),
    }


def _cross_language_split_metadata(
    train: Mapping[str, tuple[EnhancedFeatureCache, LanguagePairBank]],
    test_cache: EnhancedFeatureCache,
    test_bank: LanguagePairBank,
) -> dict[str, Any]:
    if not train or test_bank.language != test_cache.language or test_bank.language in train:
        raise ValueError("invalid T5 train/test language partition")
    train_pairs = [pair for _language, (_cache, bank) in sorted(train.items()) for pair in bank.pairs]
    test_pairs = list(test_bank.pairs)
    train_hashes = {
        code_hash
        for pair in train_pairs
        for code_hash in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    test_hashes = {
        code_hash
        for pair in test_pairs
        for code_hash in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError("T5 exact code content leakage across languages")
    train_labels = np.asarray([pair.label for pair in train_pairs], dtype=np.int64)
    test_labels = np.asarray([pair.label for pair in test_pairs], dtype=np.int64)
    return {
        "train_languages": sorted(train),
        "test_language": test_bank.language,
        "train_rows": len(train_pairs),
        "test_rows": len(test_pairs),
        "train_class_counts": {
            str(label): int((train_labels == label).sum()) for label in (0, 1)
        },
        "test_class_counts": {
            str(label): int((test_labels == label).sum()) for label in (0, 1)
        },
        "train_unique_code_hashes": len(train_hashes),
        "test_unique_code_hashes": len(test_hashes),
        "content_leakage_count": 0,
        "train_index_sha256": _digest_json(
            {language: bank.pair_sha256 for language, (_cache, bank) in sorted(train.items())}
        ),
        "test_index_sha256": test_bank.pair_sha256,
        "train_bank_sha256": {
            language: bank.pair_sha256 for language, (_cache, bank) in sorted(train.items())
        },
        "test_bank_sha256": test_bank.pair_sha256,
    }


def _cache_digest(cache: EnhancedFeatureCache) -> str:
    return _digest_json(
        {
            "row_sha256": cache.row_sha256.tolist(),
            "human": hashlib.sha256(np.ascontiguousarray(cache.human).tobytes()).hexdigest(),
            "llm": hashlib.sha256(np.ascontiguousarray(cache.llm).tobytes()).hexdigest(),
        }
    )


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record.get("heldout_language")),
        str(record.get("method")),
        int(record.get("seed")),
    )


def _record_sha256(record: Mapping[str, Any]) -> str:
    return _digest_json({key: value for key, value in record.items() if key != "record_sha256"})


def _validate_metrics(record: Mapping[str, Any]) -> None:
    for name in ("f1", "precision", "recall", "auroc"):
        value = record.get(name)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid T5 metric")
    value = record.get("mcc")
    if type(value) not in (int, float) or not np.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("invalid T5 metric")
    for name in ("fit_seconds", "predict_seconds"):
        value = record.get(name)
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("invalid T5 timing")


def _validate_record(record: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != T5_RECORD_FIELDS:
        raise ValueError("T5 record schema mismatch")
    _validate_metrics(record)
    heldout, method, seed = _record_key(record)
    spec = config["method_contract"].get(method)
    if (
        record["schema_version"] != T5_SCHEMA_VERSION
        or record["config_id"] != config["config_id"]
        or heldout not in config["heldout_languages"]
        or method not in config["methods"]
        or seed not in config["seeds"]
        or not isinstance(spec, dict)
        or record["feature_family"] != spec["feature_family"]
        or record["representation"] != spec["representation"]
        or record["model"] != spec["model"]
        or record["feature_dimensions"] != spec["feature_dimensions"]
        or record["split_protocol"] != T5_SPLIT_PROTOCOL_VERSION
        or record["bank_protocol"] != T5_BANK_PROTOCOL_VERSION
        or record["pair_protocol"] != PAIR_PROTOCOL_VERSION
        or record["component_protocol"] != COMPONENT_PROTOCOL_VERSION
        or record["source_jsonl_sha256"] != config["source_jsonl_sha256"]
        or record["cache_content_sha256"] != config["cache_content_sha256"]
        or record["gate_c_sha256"] != config["gate_c_binding"]["gate_c_sha256"]
        or record["gate_c_manifest_sha256"] != config["gate_c_binding"]["manifest_sha256"]
        or record["record_sha256"] != _record_sha256(record)
    ):
        raise ValueError("T5 record schema/config mismatch")
    if record["content_leakage_count"] != 0:
        raise ValueError("T5 record contains cross-language leakage")
    if record["train_languages"] != sorted(language for language in config["languages"] if language != heldout):
        raise ValueError("T5 record language partition mismatch")
    if record["test_language"] != heldout:
        raise ValueError("T5 record heldout language mismatch")
    if not all(
        isinstance(record[name], str) and len(record[name]) == 64
        for name in (
            "train_index_sha256", "test_index_sha256", "test_bank_sha256",
            "gate_c_sha256", "gate_c_manifest_sha256", "record_sha256",
        )
    ):
        raise ValueError("invalid T5 digest")
    return record


def _validate_config(config: Any) -> dict[str, Any]:
    required = {
        "schema_version", "config_id", "task", "languages", "heldout_languages",
        "methods", "seeds", "n_pair_folds", "limit_origins", "full_matrix",
        "split_protocol", "bank_protocol", "pair_protocol", "component_protocol",
        "method_contract", "source_jsonl_sha256", "cache_content_sha256",
        "gate_c_binding", "implementation_contract", "package_versions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("invalid T5 config fields")
    if (
        config["schema_version"] != T5_SCHEMA_VERSION
        or config["task"] != "task5_cross_language"
        or config["split_protocol"] != T5_SPLIT_PROTOCOL_VERSION
        or config["bank_protocol"] != T5_BANK_PROTOCOL_VERSION
        or config["pair_protocol"] != PAIR_PROTOCOL_VERSION
        or config["component_protocol"] != COMPONENT_PROTOCOL_VERSION
        or type(config["full_matrix"]) is not bool
        or config["config_id"]
        != _digest_json({key: value for key, value in config.items() if key != "config_id"})
    ):
        raise ValueError("invalid T5 config binding")
    return config


def _load_records(path: str | Path, config: Mapping[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    ledger = Path(path)
    if not ledger.exists():
        return {}
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("invalid T5 fold ledger") from exc
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid T5 fold ledger JSON") from exc
        validated = _validate_record(record, config)
        key = _record_key(validated)
        if key in records:
            raise ValueError("duplicate T5 record key")
        records[key] = validated
    return records


def _atomic_write_records(path: Path, records: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> None:
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


def _build_config(
    languages: tuple[str, ...],
    heldout_languages: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    n_pair_folds: int,
    limit_origins: int | None,
    paths: Mapping[str, Path],
    caches: Mapping[str, EnhancedFeatureCache],
    gate_c: Mapping[str, Any],
) -> dict[str, Any]:
    full = (
        languages == LANGUAGES
        and heldout_languages == LANGUAGES
        and methods == METHODS
        and seeds == DEFAULT_SEEDS
        and n_pair_folds == T5_PAIR_FOLDS
        and limit_origins is None
    )
    payload = {
        "schema_version": T5_SCHEMA_VERSION,
        "task": "task5_cross_language",
        "languages": list(languages),
        "heldout_languages": list(heldout_languages),
        "methods": list(methods),
        "seeds": list(seeds),
        "n_pair_folds": n_pair_folds,
        "limit_origins": limit_origins,
        "full_matrix": full,
        "split_protocol": T5_SPLIT_PROTOCOL_VERSION,
        "bank_protocol": T5_BANK_PROTOCOL_VERSION,
        "pair_protocol": PAIR_PROTOCOL_VERSION,
        "component_protocol": COMPONENT_PROTOCOL_VERSION,
        "method_contract": {method: gate_c["method_contract"][method] for method in methods},
        "source_jsonl_sha256": {language: _sha256(paths[language]) for language in languages},
        "cache_content_sha256": {language: _cache_digest(caches[language]) for language in languages},
        "gate_c_binding": dict(gate_c),
        "implementation_contract": {
            "t5_source_sha256": _sha256(Path(__file__).resolve()),
            "pair_builder_source_sha256": _sha256(Path(build_t1_pair_splits.__code__.co_filename).resolve()),
            "experiment_source_sha256": _sha256(Path(evaluate_fold.__code__.co_filename).resolve()),
        },
        "package_versions": _package_versions(),
    }
    payload["config_id"] = _digest_json(payload)
    return _validate_config(payload)


def _expected_binding(
    config: Mapping[str, Any], heldout: str, method: str, seed: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    spec = config["method_contract"][method]
    return {
        "schema_version": T5_SCHEMA_VERSION,
        "config_id": config["config_id"],
        "heldout_language": heldout,
        "method": method,
        "feature_family": spec["feature_family"],
        "representation": spec["representation"],
        "model": spec["model"],
        "feature_dimensions": spec["feature_dimensions"],
        "seed": seed,
        "split_protocol": T5_SPLIT_PROTOCOL_VERSION,
        "bank_protocol": T5_BANK_PROTOCOL_VERSION,
        "pair_protocol": PAIR_PROTOCOL_VERSION,
        "component_protocol": COMPONENT_PROTOCOL_VERSION,
        **metadata,
        "source_jsonl_sha256": config["source_jsonl_sha256"],
        "cache_content_sha256": config["cache_content_sha256"],
        "gate_c_sha256": config["gate_c_binding"]["gate_c_sha256"],
        "gate_c_manifest_sha256": config["gate_c_binding"]["manifest_sha256"],
    }


def _run_t5_locked(
    output_root: str | Path,
    languages: tuple[str, ...] = LANGUAGES,
    heldout_languages: tuple[str, ...] = LANGUAGES,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_pair_folds: int = T5_PAIR_FOLDS,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    clean_cache_root: str | Path = DEFAULT_CLEAN_CACHE_ROOT,
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    gate_c_path: str | Path = DEFAULT_GATE_C_PATH,
) -> dict[str, Any]:
    if (
        not languages or len(set(languages)) != len(languages)
        or any(language not in LANGUAGES for language in languages)
        or not heldout_languages or any(language not in languages for language in heldout_languages)
        or len(set(heldout_languages)) != len(heldout_languages)
        or not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods)
        or not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds)
        or type(n_pair_folds) is not int or n_pair_folds < 2
        or (limit_origins is not None and (type(limit_origins) is not int or limit_origins < n_pair_folds * 2))
    ):
        raise ValueError("invalid T5 axes")
    if dataset_paths is not None and set(dataset_paths) != set(languages):
        raise ValueError("T5 dataset paths must exactly match configured languages")
    paths = {
        language: (
            Path(dataset_paths[language]).resolve()
            if dataset_paths
            else (REPRO_ROOT / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl").resolve()
        )
        for language in languages
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("T5 dataset does not exist")
    gate_c = _load_strict_gate_c(gate_c_path)
    if any(_sha256(paths[language]) != gate_c["source_jsonl_sha256"][language] for language in languages):
        raise ValueError("T5 dataset does not match strict Gate C data")
    caches = {
        language: _select_t3_positive_bank(
            load_or_build_enhanced_cache(
                language, paths[language], clean_cache_root, official_cache_root
            ),
            limit_origins,
        )
        for language in languages
    }
    config = _build_config(
        languages, heldout_languages, methods, seeds, n_pair_folds,
        limit_origins, paths, caches, gate_c,
    )
    output = resolve_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        if _validate_config(_load_json(config_path)) != config:
            raise ValueError("existing T5 config does not match requested run")
    else:
        atomic_write_bytes(config_path, _canonical_json(config))
    ledger = output / "folds.jsonl"
    records = _load_records(ledger, config)
    completed = skipped = 0
    for seed in seeds:
        banks = {
            language: build_language_pair_bank(
                caches[language], language, seed, n_pair_folds
            )
            for language in languages
        }
        for heldout in heldout_languages:
            train = {
                language: (caches[language], banks[language])
                for language in languages
                if language != heldout
            }
            metadata = _cross_language_split_metadata(
                train, caches[heldout], banks[heldout]
            )
            for method in methods:
                spec = config["method_contract"][method]
                train_parts = []
                train_label_parts = []
                for language in sorted(train):
                    matrix, labels = _t3_pair_matrix(
                        caches[language], banks[language].pairs, spec
                    )
                    train_parts.append(matrix)
                    train_label_parts.append(labels)
                train_matrix = np.vstack(train_parts)
                train_labels = np.concatenate(train_label_parts)
                test_matrix, test_labels = _t3_pair_matrix(
                    caches[heldout], banks[heldout].pairs, spec
                )
                key = (heldout, method, seed)
                binding = _expected_binding(
                    config, heldout, method, seed, metadata
                )
                if key in records:
                    if any(records[key][name] != value for name, value in binding.items()):
                        raise ValueError("completed T5 record reconstruction mismatch")
                    skipped += 1
                    continue
                metrics = evaluate_fold(
                    train_matrix, train_labels, test_matrix, test_labels,
                    spec["model"], seed,
                )
                if (
                    metrics["train_rows"] != metadata["train_rows"]
                    or metrics["test_rows"] != metadata["test_rows"]
                    or metrics["train_class_counts"] != metadata["train_class_counts"]
                    or metrics["test_class_counts"] != metadata["test_class_counts"]
                ):
                    raise ValueError("T5 evaluator rows/classes disagree with language banks")
                record = {**metrics, **binding}
                record["record_sha256"] = _record_sha256(record)
                _validate_record(record, config)
                current = _load_records(ledger, config)
                if key in current:
                    raise ValueError("duplicate T5 record key")
                current[key] = record
                _atomic_write_records(ledger, current)
                records = current
                completed += 1
    return {
        "schema_version": T5_SCHEMA_VERSION,
        "config_id": config["config_id"],
        "expected": _evaluation_count(heldout_languages, methods, seeds),
        "completed": completed,
        "skipped": skipped,
        "output_root": str(output),
    }


def run_t5(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    languages: tuple[str, ...] = LANGUAGES,
    heldout_languages: tuple[str, ...] = LANGUAGES,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_pair_folds: int = T5_PAIR_FOLDS,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    clean_cache_root: str | Path = DEFAULT_CLEAN_CACHE_ROOT,
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    gate_c_path: str | Path = DEFAULT_GATE_C_PATH,
) -> dict[str, Any]:
    output = resolve_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        return _run_t5_locked(
            output, languages, heldout_languages, methods, seeds, n_pair_folds,
            limit_origins, dataset_paths, clean_cache_root,
            official_cache_root, gate_c_path,
        )


def run_t5_smoke(
    output_root: str | Path,
    dataset_paths: dict[str, str | Path] | None = None,
    clean_cache_root: str | Path = DEFAULT_CLEAN_CACHE_ROOT,
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
) -> dict[str, Any]:
    return run_t5(
        output_root,
        languages=LANGUAGES,
        heldout_languages=LANGUAGES,
        methods=METHODS,
        seeds=(42,),
        n_pair_folds=2,
        limit_origins=T5_SMOKE_ORIGINS,
        dataset_paths=dataset_paths,
        clean_cache_root=clean_cache_root,
        official_cache_root=official_cache_root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or summarize the strict leave-one-language-out T5 matrix."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean-cache-root", type=Path, default=DEFAULT_CLEAN_CACHE_ROOT)
    parser.add_argument(
        "--official-cache-root", type=Path, default=DEFAULT_OFFICIAL_CACHE_ROOT
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    from .gates_t5 import summarize_t5

    if args.summarize_only:
        report: dict[str, Any] = summarize_t5(args.output_root)
    else:
        if args.smoke:
            run_report = run_t5_smoke(
                args.output_root,
                clean_cache_root=args.clean_cache_root,
                official_cache_root=args.official_cache_root,
            )
        else:
            run_report = run_t5(
                args.output_root,
                clean_cache_root=args.clean_cache_root,
                official_cache_root=args.official_cache_root,
            )
        report = {"run": run_report, "summary": summarize_t5(args.output_root)}
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


__all__ = [
    "LanguagePairBank",
    "build_language_pair_bank",
    "run_t5",
    "run_t5_smoke",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
