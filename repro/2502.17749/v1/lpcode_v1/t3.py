"""Task 3 cache, strict unseen-LLM splits, and resumable evaluation runner.

The cache appends the pre-registered enhanced 18-vector to the frozen official
10-vector without changing row order. Its semantic digest detects uncoordinated
cache/metadata corruption. The runner binds every record to a passing strict-
origin Gate A artifact and reconstructs cache, pairs, and matrices before a
completed record may be skipped.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_jsonl
from .cache_io import (
    acquire_windows_file_lock as _acquire_windows_cache_lock,
    atomic_write_bytes as _atomic_write_bytes,
    exclusive_cache_lock as _exclusive_cache_lock,
    unique_temporary_path as _unique_temporary_path,
)
from .features_enhanced import FEATURE_NAMES as ENHANCED_FEATURE_NAMES
from .features_enhanced import analyze_enhanced
from .features_official import FEATURE_NAMES as OFFICIAL_FEATURE_NAMES
from .experiment import evaluate_fold
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .representations import build_representation
from .t1 import _cache_paths as _official_cache_paths
from .t1 import (
    METRIC_FIELDS,
    _atomic_write_records,
    _exclusive_output_lock,
    _package_versions,
    load_or_build_feature_cache,
)


CACHE_VERSION = "enhanced28-v3"
OFFICIAL_FEATURE_COUNT = 10
ENHANCED_FEATURE_COUNT = 18
FEATURE_COUNT = OFFICIAL_FEATURE_COUNT + ENHANCED_FEATURE_COUNT
LANGUAGES = ("c", "cpp", "java", "py")
LLM_SOURCES = (
    "gpt3.5",
    "gemini-pro",
    "wizardcoder:33b-v1.1",
    "deepseek-coder:33b-instruct",
)
DEFAULT_OFFICIAL_CACHE_ROOT = (
    RESULTS_ROOT / "01_transition_test_strict_origins" / "cache"
)
DEFAULT_GATE_A_PATH = (
    RESULTS_ROOT / "01_transition_test_strict_origins" / "gate_a.json"
)
DEFAULT_T3_CACHE_ROOT = RESULTS_ROOT / "02_unseen_llm" / "cache"
DEFAULT_SEEDS = (42, 123, 2024)
T3_METHODS = (
    "lpcode_original",
    "xgb_original",
    "best_transition",
    "mstf",
)
T3_SPLIT_PROTOCOL_VERSION = "leave-one-llm-strict-origin-v1"
STRICT_GATE_PROTOCOL_VERSION = "all-llm-strict-origin-v2"
T3_SCHEMA_VERSION = 1
T3_SMOKE_ORIGINS = 8
_PACKAGE_DISTRIBUTIONS = (
    "numpy",
    "tree-sitter",
    "tree-sitter-c",
    "tree-sitter-cpp",
    "tree-sitter-java",
)


@dataclass(frozen=True)
class EnhancedFeatureCache:
    """Validated row-aligned official-10 plus enhanced-18 feature arrays."""

    language: str
    human: np.ndarray
    llm: np.ndarray
    labels: np.ndarray
    source_ids: np.ndarray
    human_origin_ids: np.ndarray
    candidate_origin_ids: np.ndarray
    human_code_sha256: np.ndarray
    candidate_code_sha256: np.ndarray
    llm_sources: np.ndarray
    row_sha256: np.ndarray
    human_parse_ok: np.ndarray
    llm_parse_ok: np.ndarray
    human_backends: np.ndarray
    llm_backends: np.ndarray
    human_fallback_reasons: np.ndarray
    llm_fallback_reasons: np.ndarray


PAIR_PROTOCOL_VERSION = "positive-bank-cross-component-derangement-v2"
COMPONENT_PROTOCOL_VERSION = "exact-content-components-v1"
NEGATIVE_PAIR_MODES = ("current", "random", "hard")


@dataclass(frozen=True)
class T3PairSpec:
    """One immutable pair built exclusively from positive-bank rows."""

    human_positive_row_idx: int
    candidate_positive_row_idx: int
    label: int
    human_origin_id: str
    candidate_origin_id: str
    human_code_sha256: str
    candidate_code_sha256: str
    human_component_id: str
    candidate_component_id: str
    llm_source: str
    pair_sha256: str


@dataclass(frozen=True)
class T3PairSplit:
    """One leakage-free fold whose pair tuples are shared by every method."""

    fold: int
    heldout_llm: str
    train_pairs: tuple[T3PairSpec, ...]
    test_pairs: tuple[T3PairSpec, ...]
    train_pair_sha256: str
    test_pair_sha256: str
    leakage_count: int = 0

    def pairs_for_methods(
        self, method_names: tuple[str, ...]
    ) -> dict[str, tuple[tuple[T3PairSpec, ...], tuple[T3PairSpec, ...]]]:
        """Bind all named methods to these exact immutable tuple objects."""

        if (
            not method_names
            or any(type(name) is not str or not name for name in method_names)
            or len(set(method_names)) != len(method_names)
        ):
            raise ValueError("method names must be nonempty unique strings")
        return {
            name: (self.train_pairs, self.test_pairs) for name in method_names
        }


@dataclass(frozen=True)
class T1PairSplit:
    """One all-LLM strict-origin fold shared by every T1 method."""

    fold: int
    train_pairs: tuple[T3PairSpec, ...]
    test_pairs: tuple[T3PairSpec, ...]
    train_pair_sha256: str
    test_pair_sha256: str
    leakage_count: int = 0

    def pairs_for_methods(
        self, method_names: tuple[str, ...]
    ) -> dict[str, tuple[tuple[T3PairSpec, ...], tuple[T3PairSpec, ...]]]:
        """Bind all named methods to these exact immutable tuple objects."""

        if (
            not method_names
            or any(type(name) is not str or not name for name in method_names)
            or len(set(method_names)) != len(method_names)
        ):
            raise ValueError("method names must be nonempty unique strings")
        return {
            name: (self.train_pairs, self.test_pairs) for name in method_names
        }


@dataclass(frozen=True)
class _PairSplitData:
    fold: int
    train_pairs: tuple[T3PairSpec, ...]
    test_pairs: tuple[T3PairSpec, ...]
    train_pair_sha256: str
    test_pair_sha256: str


T3Split = T3PairSplit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(row)).hexdigest()


def _index_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _semantic_content_sha256(arrays: dict[str, Any], language: str) -> str:
    """Hash cached enhanced values and their exact row-level provenance.

    Float columns are canonical contiguous little-endian float64.  Variable
    length sections use explicit name and payload lengths, so concatenation is
    unambiguous.  This intentionally excludes official columns 0:10, which are
    independently bound to the validated official cache digest and exact array
    equality during loading.
    """

    digest = hashlib.sha256()
    digest.update(b"lpcode-v1/enhanced28-semantic-v2\0")

    def add(name: str, payload: bytes) -> None:
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    add("language", language.encode("ascii"))
    add("row_sha256", _canonical_json(np.asarray(arrays["row_sha256"]).tolist()))
    for name in (
        "human_origin_ids",
        "candidate_origin_ids",
        "human_code_sha256",
        "candidate_code_sha256",
        "llm_sources",
    ):
        add(name, _canonical_json(np.asarray(arrays[name]).tolist()))
    for name in ("human", "llm"):
        enhanced = np.ascontiguousarray(
            np.asarray(arrays[name])[:, OFFICIAL_FEATURE_COUNT:], dtype="<f8"
        )
        add(f"{name}_enhanced18_shape", _canonical_json(list(enhanced.shape)))
        add(f"{name}_enhanced18", enhanced.tobytes(order="C"))
    for name in ("human_parse_ok", "llm_parse_ok"):
        status = np.ascontiguousarray(np.asarray(arrays[name]), dtype=np.bool_)
        add(name, status.view(np.uint8).tobytes(order="C"))
    for name in (
        "human_backends",
        "llm_backends",
        "human_fallback_reasons",
        "llm_fallback_reasons",
    ):
        add(name, _canonical_json(np.asarray(arrays[name]).tolist()))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_strict_gate_a(gate_a_path: str | Path) -> dict[str, Any]:
    """Validate and bind the strict-origin Gate A artifact and its manifest."""

    gate_path = Path(gate_a_path).resolve()
    if gate_path != DEFAULT_GATE_A_PATH.resolve():
        raise ValueError("Gate A must use the exact strict-origin artifact path")
    root = gate_path.parent
    manifest_path = root / "manifest.json"
    config_path = root / "config.json"
    summary_path = root / "summary.json"
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        strict_config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid strict-origin Gate A artifact") from exc

    files = manifest.get("files") if isinstance(manifest, dict) else None
    expected_manifest_files = {
        "config.json",
        "folds.jsonl",
        "gate_a.json",
        "summary.json",
        "table_a.csv",
        "table_a.md",
    }
    if not isinstance(files, dict) or set(files) != expected_manifest_files:
        raise ValueError("invalid strict-origin Gate A manifest")
    for name, expected in files.items():
        candidate = (root / name).resolve()
        if (
            type(name) is not str
            or Path(name).name != name
            or candidate.parent != root
            or not candidate.is_file()
            or not isinstance(expected, dict)
            or set(expected) != {"sha256", "bytes"}
            or not _is_sha256(expected["sha256"])
            or type(expected["bytes"]) is not int
            or expected["bytes"] < 0
            or candidate.stat().st_size != expected["bytes"]
            or _sha256(candidate) != expected["sha256"]
        ):
            raise ValueError("invalid strict-origin Gate A manifest")

    selected = gate.get("selected_candidate") if isinstance(gate, dict) else None
    strict = gate.get("strict") if isinstance(gate, dict) else None
    method_versions = manifest.get("method_versions", {})
    strict_config_fields = {
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
    strict_config_payload = (
        {key: value for key, value in strict_config.items() if key != "config_id"}
        if isinstance(strict_config, dict)
        else {}
    )
    ranking = summary.get("candidate_ranking") if isinstance(summary, dict) else None
    summary_config = summary.get("config") if isinstance(summary, dict) else None
    strict_feature = (
        strict_config.get("feature_contract", {})
        if isinstance(strict_config, dict)
        else {}
    )
    strict_official_feature = (
        strict_feature.get("official_feature_contract", {})
        if isinstance(strict_feature, dict)
        else {}
    )
    strict_enhanced_feature = (
        strict_feature.get("enhanced_feature_contract", {})
        if isinstance(strict_feature, dict)
        else {}
    )
    expected_implementation_keys = {
        "runner_source_sha256",
        "experiment_source_sha256",
        "representations_source_sha256",
        "pair_builder_source_sha256",
        "data_normalization_source_sha256",
        "official_cache_runner_source_sha256",
    }
    expected_package_keys = {
        "python",
        "numpy",
        "scikit_learn",
        "xgboost",
        "tree-sitter",
        "tree-sitter-c",
        "tree-sitter-cpp",
        "tree-sitter-java",
    }
    expected_feature_keys = {
        "enhanced_cache_version",
        "enhanced_feature_contract",
        "enhanced_feature_contract_sha256",
        "official_cache_version",
        "official_feature_contract",
        "official_feature_contract_sha256",
        "selected_columns",
        "selected_feature_count",
        "selected_feature_names",
    }
    expected_enhanced_feature = _feature_contract()
    expected_official_feature = {
        "cache_version": "official10-v2",
        "feature_count": OFFICIAL_FEATURE_COUNT,
        "feature_names": list(OFFICIAL_FEATURE_NAMES),
        "official_feature_source_sha256": expected_enhanced_feature[
            "official_feature_source_sha256"
        ],
    }
    if (
        gate.get("status") != "evaluable"
        or gate.get("protocol_version") != STRICT_GATE_PROTOCOL_VERSION
        or not isinstance(strict, dict)
        or strict.get("passed") is not True
        or not isinstance(selected, dict)
        or set(selected) != {"representation", "model"}
        or selected["representation"] not in ("concat", "delta", "concat_delta", "full")
        or selected["model"] not in ("mlp", "xgb")
        or not isinstance(method_versions, dict)
        or method_versions.get("protocol") != STRICT_GATE_PROTOCOL_VERSION
        or set(strict_config) != strict_config_fields
        or type(strict_config.get("schema_version")) is not int
        or strict_config.get("schema_version") != 2
        or strict_config.get("task") != "task1_strict_origins"
        or type(strict_config.get("fold_index_base")) is not int
        or strict_config.get("fold_index_base") != 0
        or strict_config.get("languages") != list(LANGUAGES)
        or strict_config.get("seeds") != list(DEFAULT_SEEDS)
        or type(strict_config.get("n_splits")) is not int
        or strict_config.get("n_splits") != 5
        or strict_config.get("representations")
        != ["concat", "delta", "concat_delta", "full"]
        or strict_config.get("models") != ["mlp", "xgb"]
        or strict_config.get("limit_origins") is not None
        or strict_config.get("split_protocol") != STRICT_GATE_PROTOCOL_VERSION
        or strict_config.get("pair_protocol") != PAIR_PROTOCOL_VERSION
        or strict_config.get("component_protocol") != COMPONENT_PROTOCOL_VERSION
        or not _is_sha256(strict_config.get("config_id"))
        or strict_config.get("config_id") != _digest_json(strict_config_payload)
        or not isinstance(strict_config.get("implementation_contract"), dict)
        or set(strict_config.get("implementation_contract", {}))
        != expected_implementation_keys
        or any(
            not _is_sha256(value)
            for value in strict_config.get("implementation_contract", {}).values()
        )
        or not isinstance(strict_config.get("feature_contract"), dict)
        or set(strict_feature) != expected_feature_keys
        or strict_feature.get("selected_columns") != [0, 10]
        or strict_feature.get("selected_feature_count") != 10
        or strict_feature.get("selected_feature_names")
        != list(OFFICIAL_FEATURE_NAMES)
        or strict_feature.get("official_cache_version") != "official10-v2"
        or strict_feature.get("enhanced_cache_version") != CACHE_VERSION
        or set(strict_official_feature)
        != {"cache_version", "feature_count", "feature_names", "official_feature_source_sha256"}
        or strict_official_feature.get("cache_version") != "official10-v2"
        or strict_official_feature.get("feature_count") != 10
        or strict_official_feature.get("feature_names")
        != list(OFFICIAL_FEATURE_NAMES)
        or strict_official_feature != expected_official_feature
        or not _is_sha256(
            strict_official_feature.get("official_feature_source_sha256")
        )
        or strict_feature.get("official_feature_contract_sha256")
        != _digest_json(strict_official_feature)
        or set(strict_enhanced_feature)
        != {
            "cache_version",
            "feature_count",
            "official_feature_names",
            "enhanced_feature_names",
            "official_feature_source_sha256",
            "enhanced_feature_source_sha256",
        }
        or strict_enhanced_feature.get("cache_version") != CACHE_VERSION
        or strict_enhanced_feature.get("feature_count") != FEATURE_COUNT
        or strict_enhanced_feature.get("official_feature_names")
        != list(OFFICIAL_FEATURE_NAMES)
        or strict_enhanced_feature.get("enhanced_feature_names")
        != list(ENHANCED_FEATURE_NAMES)
        or strict_enhanced_feature != expected_enhanced_feature
        or strict_enhanced_feature.get("official_feature_source_sha256")
        != strict_official_feature.get("official_feature_source_sha256")
        or not _is_sha256(
            strict_enhanced_feature.get("enhanced_feature_source_sha256")
        )
        or strict_feature.get("enhanced_feature_contract_sha256")
        != _digest_json(strict_enhanced_feature)
        or not isinstance(strict_config.get("package_versions"), dict)
        or set(strict_config.get("package_versions", {})) != expected_package_keys
        or strict_config.get("package_versions") != _runner_package_versions()
        or any(
            type(key) is not str
            or type(value) is not str
            or not key
            or not value
            for key, value in strict_config.get("package_versions", {}).items()
        )
        or not isinstance(strict_config.get("source_jsonl_sha256"), dict)
        or set(strict_config.get("source_jsonl_sha256", {})) != set(LANGUAGES)
        or any(
            not _is_sha256(value)
            for value in strict_config.get("source_jsonl_sha256", {}).values()
        )
        or selected["representation"] not in strict_config["representations"]
        or selected["model"] not in strict_config["models"]
        or not isinstance(ranking, list)
        or not ranking
        or not isinstance(ranking[0], dict)
        or {
            "representation": ranking[0].get("representation"),
            "model": ranking[0].get("model"),
        }
        != selected
        or not isinstance(summary_config, dict)
        or summary_config.get("config_id") != strict_config["config_id"]
    ):
        raise ValueError("Gate A is not a passing strict-origin artifact")
    return {
        "gate_a_path": str(gate_path),
        "gate_a_sha256": _sha256(gate_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "strict_config_sha256": _sha256(config_path),
        "strict_config_id": strict_config["config_id"],
        "source_jsonl_sha256": dict(strict_config["source_jsonl_sha256"]),
        "protocol_version": gate["protocol_version"],
        "strict_passed": True,
        "selected_candidate": {
            "representation": selected["representation"],
            "model": selected["model"],
        },
    }


def _representation_dimensions(feature_count: int, representation: str) -> int:
    multiplier = {"concat": 2, "delta": 1, "concat_delta": 3, "full": 4}
    if type(feature_count) is not int or feature_count <= 0 or representation not in multiplier:
        raise ValueError("invalid T3 representation dimensions")
    return feature_count * multiplier[representation]


def _method_contract(selected_candidate: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Return the closed, pre-registered four-method T3 contract."""

    if (
        not isinstance(selected_candidate, dict)
        or set(selected_candidate) != {"representation", "model"}
        or selected_candidate["representation"] not in ("concat", "delta", "concat_delta", "full")
        or selected_candidate["model"] not in ("mlp", "xgb")
    ):
        raise ValueError("invalid strict-origin Gate A candidate")
    specs: dict[str, dict[str, Any]] = {
        "lpcode_original": {
            "feature_family": "official10",
            "feature_count": 10,
            "representation": "concat",
            "model": "mlp",
        },
        "xgb_original": {
            "feature_family": "official10",
            "feature_count": 10,
            "representation": "concat",
            "model": "xgb",
        },
        "best_transition": {
            "feature_family": "official10",
            "feature_count": 10,
            "representation": selected_candidate["representation"],
            "model": selected_candidate["model"],
        },
        "mstf": {
            "feature_family": "enhanced28",
            "feature_count": 28,
            "representation": "full",
            "model": "xgb",
        },
    }
    for spec in specs.values():
        spec["feature_dimensions"] = _representation_dimensions(
            spec["feature_count"], spec["representation"]
        )
    return specs


def _evaluation_count(
    languages: tuple[str, ...],
    heldout_llms: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    methods: tuple[str, ...],
) -> int:
    return len(languages) * len(heldout_llms) * len(seeds) * n_splits * len(methods)


def _feature_contract() -> dict[str, Any]:
    enhanced_module = Path(__file__).with_name("features_enhanced.py")
    official_module = Path(__file__).with_name("features_official.py")
    return {
        "cache_version": CACHE_VERSION,
        "feature_count": FEATURE_COUNT,
        "official_feature_names": list(OFFICIAL_FEATURE_NAMES),
        "enhanced_feature_names": list(ENHANCED_FEATURE_NAMES),
        "official_feature_source_sha256": _sha256(official_module),
        "enhanced_feature_source_sha256": _sha256(enhanced_module),
    }


def _package_contract() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in _PACKAGE_DISTRIBUTIONS:
        versions[distribution] = importlib.metadata.version(distribution)
    return versions


def _cache_paths(language: str, cache_root: str | Path) -> tuple[Path, Path]:
    if language not in LANGUAGES:
        raise ValueError(f"unsupported Task 3 language: {language!r}")
    root = resolve_output_path(cache_root)
    archive = root / CACHE_VERSION / f"{language}.npz"
    return archive, archive.with_suffix(".json")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_bytes(path, _canonical_json(value))


def _as_string_array(value: Any, rows: int, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (rows,) or array.dtype.kind not in "US":
        raise ValueError(f"invalid enhanced cache {field}")
    return np.asarray(array, dtype=str)


def _validate_arrays(
    arrays: dict[str, Any], language: str, expected_rows: int | None = None
) -> EnhancedFeatureCache:
    if type(language) is not str or language not in LANGUAGES:
        raise ValueError(f"invalid enhanced cache language: {language!r}")
    required = {
        "human",
        "llm",
        "labels",
        "source_ids",
        "human_origin_ids",
        "candidate_origin_ids",
        "human_code_sha256",
        "candidate_code_sha256",
        "llm_sources",
        "row_sha256",
        "human_parse_ok",
        "llm_parse_ok",
        "human_backends",
        "llm_backends",
        "human_fallback_reasons",
        "llm_fallback_reasons",
    }
    if set(arrays) != required:
        raise ValueError("invalid enhanced cache archive schema")
    human = np.asarray(arrays["human"])
    llm = np.asarray(arrays["llm"])
    if (
        human.dtype != np.dtype(np.float64)
        or llm.dtype != np.dtype(np.float64)
        or human.ndim != 2
        or llm.shape != human.shape
        or human.shape[1:] != (FEATURE_COUNT,)
        or not np.isfinite(human).all()
        or not np.isfinite(llm).all()
    ):
        raise ValueError("invalid enhanced cache matrices")
    rows = human.shape[0]
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("stale enhanced cache row count")
    labels = np.asarray(arrays["labels"])
    if (
        labels.ndim != 1
        or labels.shape != (rows,)
        or labels.dtype != np.dtype(np.int64)
        or not set(np.unique(labels)).issubset({0, 1})
    ):
        raise ValueError("invalid enhanced cache labels")
    human_parse_ok = np.asarray(arrays["human_parse_ok"])
    llm_parse_ok = np.asarray(arrays["llm_parse_ok"])
    if (
        human_parse_ok.shape != (rows,)
        or llm_parse_ok.shape != (rows,)
        or human_parse_ok.dtype != np.dtype(np.bool_)
        or llm_parse_ok.dtype != np.dtype(np.bool_)
    ):
        raise ValueError("invalid enhanced cache parse status")
    source_ids = _as_string_array(arrays["source_ids"], rows, "source ids")
    human_origin_ids = _as_string_array(
        arrays["human_origin_ids"], rows, "human origin ids"
    )
    candidate_origin_ids = _as_string_array(
        arrays["candidate_origin_ids"], rows, "candidate origin ids"
    )
    human_code_sha256 = _as_string_array(
        arrays["human_code_sha256"], rows, "human code hashes"
    )
    candidate_code_sha256 = _as_string_array(
        arrays["candidate_code_sha256"], rows, "candidate code hashes"
    )
    if (
        any(not origin for origin in human_origin_ids.tolist())
        or any(not origin for origin in candidate_origin_ids.tolist())
        or not np.array_equal(source_ids, human_origin_ids)
        or any(not _is_sha256(value) for value in human_code_sha256.tolist())
        or any(not _is_sha256(value) for value in candidate_code_sha256.tolist())
    ):
        raise ValueError("invalid enhanced cache endpoint origins")
    llm_sources = _as_string_array(arrays["llm_sources"], rows, "LLM sources")
    if any(source not in LLM_SOURCES for source in llm_sources.tolist()):
        raise ValueError("invalid enhanced cache LLM sources")
    row_sha256 = _as_string_array(arrays["row_sha256"], rows, "row hashes")
    if any(not _is_sha256(digest) for digest in row_sha256.tolist()):
        raise ValueError("invalid enhanced cache row hashes")
    human_backends = _as_string_array(arrays["human_backends"], rows, "human backends")
    llm_backends = _as_string_array(arrays["llm_backends"], rows, "LLM backends")
    human_reasons = _as_string_array(
        arrays["human_fallback_reasons"], rows, "human fallback reasons"
    )
    llm_reasons = _as_string_array(
        arrays["llm_fallback_reasons"], rows, "LLM fallback reasons"
    )
    return EnhancedFeatureCache(
        language=language,
        human=np.asarray(human, dtype=np.float64),
        llm=np.asarray(llm, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        source_ids=source_ids,
        human_origin_ids=human_origin_ids,
        candidate_origin_ids=candidate_origin_ids,
        human_code_sha256=human_code_sha256,
        candidate_code_sha256=candidate_code_sha256,
        llm_sources=llm_sources,
        row_sha256=row_sha256,
        human_parse_ok=np.asarray(human_parse_ok, dtype=np.bool_),
        llm_parse_ok=np.asarray(llm_parse_ok, dtype=np.bool_),
        human_backends=human_backends,
        llm_backends=llm_backends,
        human_fallback_reasons=human_reasons,
        llm_fallback_reasons=llm_reasons,
    )


def _validate_parse_provenance(
    cache: EnhancedFeatureCache, language: str
) -> None:
    expected_backend = "python-ast" if language == "py" else "tree-sitter"
    for side in ("human", "llm"):
        statuses = getattr(cache, f"{side}_parse_ok")
        backends = getattr(cache, f"{side}_backends")
        reasons = getattr(cache, f"{side}_fallback_reasons")
        for row, (parse_ok, backend, reason) in enumerate(
            zip(statuses.tolist(), backends.tolist(), reasons.tolist())
        ):
            valid = (
                backend == expected_backend and reason == ""
                if parse_ok
                else backend == "lexical-fallback" and reason == "syntax-error"
            )
            if not valid:
                raise ValueError(
                    f"invalid enhanced cache parse provenance for {side} row {row}"
                )


_METADATA_FIELDS = {
    "schema_version",
    "cache_version",
    "language",
    "rows",
    "feature_names",
    "source_jsonl_sha256",
    "feature_contract_sha256",
    "package_contract",
    "package_contract_sha256",
    "official_cache_npz_sha256",
    "official_cache_metadata_sha256",
    "parse_failures",
    "semantic_content_sha256",
    "npz_sha256",
}


def _validate_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _METADATA_FIELDS:
        raise ValueError("invalid enhanced cache metadata")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["cache_version"]) is not str
        or type(value["language"]) is not str
        or type(value["rows"]) is not int
        or value["rows"] < 0
        or type(value["feature_names"]) is not list
        or any(type(name) is not str for name in value["feature_names"])
        or not isinstance(value["package_contract"], dict)
        or any(type(key) is not str or type(item) is not str for key, item in value["package_contract"].items())
        or not isinstance(value["parse_failures"], dict)
        or set(value["parse_failures"]) != {"human", "llm"}
        or any(type(count) is not int or count < 0 for count in value["parse_failures"].values())
        or any(
            not _is_sha256(value[field])
            for field in (
                "source_jsonl_sha256",
                "feature_contract_sha256",
                "package_contract_sha256",
                "official_cache_npz_sha256",
                "official_cache_metadata_sha256",
                "semantic_content_sha256",
                "npz_sha256",
            )
        )
    ):
        raise ValueError("invalid enhanced cache metadata")
    return value


def _expected_metadata(
    language: str,
    rows: int,
    source_hash: str,
    official_npz_hash: str,
    official_metadata_hash: str,
) -> dict[str, Any]:
    packages = _package_contract()
    return {
        "schema_version": 1,
        "cache_version": CACHE_VERSION,
        "language": language,
        "rows": rows,
        "feature_names": list(OFFICIAL_FEATURE_NAMES) + list(ENHANCED_FEATURE_NAMES),
        "source_jsonl_sha256": source_hash,
        "feature_contract_sha256": _digest_json(_feature_contract()),
        "package_contract": packages,
        "package_contract_sha256": _digest_json(packages),
        "official_cache_npz_sha256": official_npz_hash,
        "official_cache_metadata_sha256": official_metadata_hash,
    }


def _load_existing_cache(
    archive_path: Path,
    metadata_path: Path,
    expected: dict[str, Any],
    official_human: np.ndarray,
    official_llm: np.ndarray,
    expected_labels: np.ndarray,
    expected_source_ids: np.ndarray,
    expected_human_origin_ids: np.ndarray,
    expected_candidate_origin_ids: np.ndarray,
    expected_human_code_sha256: np.ndarray,
    expected_candidate_code_sha256: np.ndarray,
    expected_llm_sources: np.ndarray,
    expected_row_sha256: np.ndarray,
) -> EnhancedFeatureCache:
    if not archive_path.exists() and not metadata_path.exists():
        raise FileNotFoundError
    if not archive_path.exists() or not metadata_path.exists():
        raise ValueError("incomplete enhanced cache")
    try:
        metadata = _validate_metadata(json.loads(metadata_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid enhanced cache metadata") from exc
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError("stale enhanced cache metadata")
    if metadata["npz_sha256"] != _sha256(archive_path):
        raise ValueError("corrupt enhanced cache")
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            cache = _validate_arrays(
                {name: archive[name] for name in archive.files},
                language=expected["language"],
                expected_rows=expected["rows"],
            )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("invalid enhanced", "stale enhanced")):
            raise
        raise ValueError("corrupt enhanced cache") from exc
    if (
        not np.array_equal(cache.human[:, :OFFICIAL_FEATURE_COUNT], official_human)
        or not np.array_equal(cache.llm[:, :OFFICIAL_FEATURE_COUNT], official_llm)
        or not np.array_equal(cache.labels, expected_labels)
        or not np.array_equal(cache.source_ids, expected_source_ids)
        or not np.array_equal(cache.human_origin_ids, expected_human_origin_ids)
        or not np.array_equal(
            cache.candidate_origin_ids, expected_candidate_origin_ids
        )
        or not np.array_equal(cache.human_code_sha256, expected_human_code_sha256)
        or not np.array_equal(
            cache.candidate_code_sha256, expected_candidate_code_sha256
        )
        or not np.array_equal(cache.llm_sources, expected_llm_sources)
        or not np.array_equal(cache.row_sha256, expected_row_sha256)
    ):
        raise ValueError("enhanced cache row alignment mismatch")
    if metadata["semantic_content_sha256"] != _semantic_content_sha256(
        _cache_as_arrays(cache), cache.language
    ):
        raise ValueError("enhanced cache semantic content digest mismatch")
    _validate_parse_provenance(cache, expected["language"])
    expected_failures = {
        "human": int((~cache.human_parse_ok).sum()),
        "llm": int((~cache.llm_parse_ok).sum()),
    }
    if metadata["parse_failures"] != expected_failures:
        raise ValueError("enhanced cache parse-status mismatch")
    return cache


def _load_or_build_enhanced_cache_unlocked(
    language: str,
    dataset_path: str | Path | None = None,
    cache_root: str | Path = RESULTS_ROOT / "02_unseen_llm" / "cache",
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
) -> EnhancedFeatureCache:
    """Load or build a source-, contract-, package-, and row-bound cache."""

    if language not in LANGUAGES:
        raise ValueError(f"unsupported Task 3 language: {language!r}")
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
    llm_sources: list[str] = []
    human_origins: list[str] = []
    candidate_origins: list[str] = []
    for row_index, row in enumerate(rows):
        source = row.get("paraphrased_by")
        if type(source) is not str or source not in LLM_SOURCES:
            raise ValueError(f"invalid LLM source at row {row_index}")
        llm_sources.append(source)
        label = int(row["label"])
        human_field = "file_name" if label == 1 else "human_file_name"
        candidate_field = "file_name" if label == 1 else "llm_file_name"
        human_origin = row.get(human_field)
        candidate_origin = row.get(candidate_field)
        if (
            type(human_origin) is not str
            or not human_origin
            or type(candidate_origin) is not str
            or not candidate_origin
        ):
            raise ValueError(f"invalid endpoint origin at row {row_index}")
        human_origins.append(human_origin)
        candidate_origins.append(candidate_origin)
    if _sha256(dataset) != source_hash:
        raise ValueError("Task 1 dataset changed during load")

    official = load_or_build_feature_cache(
        language, dataset, official_cache_root
    )
    expected_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    expected_sources = np.asarray(
        [str(row["human_source_id"]) for row in rows], dtype=str
    )
    expected_human_origins = np.asarray(human_origins, dtype=str)
    expected_candidate_origins = np.asarray(candidate_origins, dtype=str)
    expected_human_code_sha256 = np.asarray(
        [hashlib.sha256(str(row["human_src"]).encode("utf-8")).hexdigest() for row in rows],
        dtype=str,
    )
    expected_candidate_code_sha256 = np.asarray(
        [hashlib.sha256(str(row["llm_src"]).encode("utf-8")).hexdigest() for row in rows],
        dtype=str,
    )
    if not np.array_equal(expected_sources, expected_human_origins):
        raise ValueError("normalized source id does not match human endpoint origin")
    expected_llm_sources = np.asarray(llm_sources, dtype=str)
    expected_row_sha256 = np.asarray([_row_sha256(row) for row in rows], dtype=str)
    if (
        not np.array_equal(official.labels, expected_labels)
        or not np.array_equal(official.source_ids, expected_sources)
    ):
        raise ValueError("official cache row alignment mismatch")
    official_archive, official_metadata = _official_cache_paths(
        language, official_cache_root
    )
    official_npz_hash = _sha256(official_archive)
    official_metadata_hash = _sha256(official_metadata)
    expected = _expected_metadata(
        language,
        len(rows),
        source_hash,
        official_npz_hash,
        official_metadata_hash,
    )
    archive_path, metadata_path = _cache_paths(language, cache_root)
    try:
        cached = _load_existing_cache(
            archive_path,
            metadata_path,
            expected,
            official.human,
            official.llm,
            expected_labels,
            expected_sources,
            expected_human_origins,
            expected_candidate_origins,
            expected_human_code_sha256,
            expected_candidate_code_sha256,
            expected_llm_sources,
            expected_row_sha256,
        )
        if _sha256(dataset) != source_hash:
            raise ValueError("Task 1 dataset changed during cache load")
        return cached
    except FileNotFoundError:
        pass

    analysis_by_code: dict[str, Any] = {}

    def analyze_once(code: str):
        if code not in analysis_by_code:
            analysis_by_code[code] = analyze_enhanced(code, language)
        return analysis_by_code[code]

    human_analyses = [analyze_once(str(row["human_src"])) for row in rows]
    llm_analyses = [analyze_once(str(row["llm_src"])) for row in rows]
    if _sha256(dataset) != source_hash:
        raise ValueError("Task 1 dataset changed during enhanced extraction")
    if (
        _sha256(official_archive) != official_npz_hash
        or _sha256(official_metadata) != official_metadata_hash
    ):
        raise ValueError("official feature cache changed during enhanced extraction")
    arrays: dict[str, Any] = {
        "human": np.column_stack(
            (official.human, np.vstack([analysis.values for analysis in human_analyses]))
        ).astype(np.float64, copy=False),
        "llm": np.column_stack(
            (official.llm, np.vstack([analysis.values for analysis in llm_analyses]))
        ).astype(np.float64, copy=False),
        "labels": expected_labels,
        "source_ids": expected_sources,
        "human_origin_ids": expected_human_origins,
        "candidate_origin_ids": expected_candidate_origins,
        "human_code_sha256": expected_human_code_sha256,
        "candidate_code_sha256": expected_candidate_code_sha256,
        "llm_sources": expected_llm_sources,
        "row_sha256": expected_row_sha256,
        "human_parse_ok": np.asarray(
            [analysis.parse_ok for analysis in human_analyses], dtype=np.bool_
        ),
        "llm_parse_ok": np.asarray(
            [analysis.parse_ok for analysis in llm_analyses], dtype=np.bool_
        ),
        "human_backends": np.asarray(
            [analysis.backend for analysis in human_analyses], dtype=str
        ),
        "llm_backends": np.asarray(
            [analysis.backend for analysis in llm_analyses], dtype=str
        ),
        "human_fallback_reasons": np.asarray(
            [analysis.fallback_reason or "" for analysis in human_analyses], dtype=str
        ),
        "llm_fallback_reasons": np.asarray(
            [analysis.fallback_reason or "" for analysis in llm_analyses], dtype=str
        ),
    }
    cache = _validate_arrays(arrays, language=language, expected_rows=len(rows))
    _validate_parse_provenance(cache, language)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(archive_path, suffix=".npz")
    try:
        np.savez(temporary, **arrays)
        temporary.replace(archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        **expected,
        "parse_failures": {
            "human": int((~cache.human_parse_ok).sum()),
            "llm": int((~cache.llm_parse_ok).sum()),
        },
        "semantic_content_sha256": _semantic_content_sha256(arrays, language),
        "npz_sha256": _sha256(archive_path),
    }
    _atomic_write_json(metadata_path, metadata)
    return cache


def load_or_build_enhanced_cache(
    language: str,
    dataset_path: str | Path | None = None,
    cache_root: str | Path = RESULTS_ROOT / "02_unseen_llm" / "cache",
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
) -> EnhancedFeatureCache:
    """Load/build one language while holding its publication lock."""

    archive_path, _metadata_path = _cache_paths(language, cache_root)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = archive_path.parent / f".{language}.lock"
    with _exclusive_cache_lock(lock_path):
        return _load_or_build_enhanced_cache_unlocked(
            language, dataset_path, cache_root, official_cache_root
        )


def _cache_as_arrays(cache: EnhancedFeatureCache) -> dict[str, Any]:
    return {
        field: getattr(cache, field)
        for field in cache.__dataclass_fields__
        if field != "language"
    }


def _build_pair_splits(
    cache: EnhancedFeatureCache,
    language: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
    *,
    train_sources: tuple[str, ...],
    test_sources: tuple[str, ...],
    negative_pair_mode: str = "current",
) -> list[_PairSplitData]:
    """Partition positive origins, then construct balanced fold-local pairs."""

    if not isinstance(cache, EnhancedFeatureCache):
        raise TypeError("cache must be an EnhancedFeatureCache")
    validated = _validate_arrays(_cache_as_arrays(cache), language=cache.language)
    if language is None:
        language = cache.language
    elif language != cache.language:
        raise ValueError(
            f"cache language mismatch: cache={cache.language!r}, requested={language!r}"
        )
    if type(language) is not str or language not in LANGUAGES:
        raise ValueError(f"unsupported Task 3 language: {language!r}")
    for side_name, sources in (("train", train_sources), ("test", test_sources)):
        if (
            type(sources) is not tuple
            or not sources
            or any(type(source) is not str or source not in LLM_SOURCES for source in sources)
            or len(set(sources)) != len(sources)
        ):
            raise ValueError(f"invalid {side_name} LLM sources")
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if negative_pair_mode not in NEGATIVE_PAIR_MODES:
        raise ValueError("unsupported negative-pair mode")

    bank: dict[tuple[str, str], int] = {}
    positive_indices = np.flatnonzero(validated.labels == 1)
    for raw_index in positive_indices.tolist():
        index = int(raw_index)
        human_origin = str(validated.human_origin_ids[index])
        candidate_origin = str(validated.candidate_origin_ids[index])
        source = str(validated.llm_sources[index])
        if human_origin != candidate_origin:
            raise ValueError("positive bank row has different endpoint origins")
        key = (human_origin, source)
        if key in bank:
            raise ValueError("positive bank requires exactly one positive per origin and LLM")
        bank[key] = index
    origins = sorted({origin for origin, _source in bank})
    expected_keys = {(origin, source) for origin in origins for source in LLM_SOURCES}
    if set(bank) != expected_keys:
        raise ValueError("positive bank requires exactly one positive per origin and LLM")
    parent = {origin: origin for origin in origins}

    def find(origin: str) -> str:
        root = origin
        while parent[root] != root:
            root = parent[root]
        while parent[origin] != origin:
            following = parent[origin]
            parent[origin] = root
            origin = following
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    content_origins: dict[str, set[str]] = {}
    for (origin, _source), index in bank.items():
        for code_hash in (
            str(validated.human_code_sha256[index]),
            str(validated.candidate_code_sha256[index]),
        ):
            content_origins.setdefault(code_hash, set()).add(origin)
    for linked_origins in content_origins.values():
        ordered_linked = sorted(linked_origins)
        for other in ordered_linked[1:]:
            union(ordered_linked[0], other)

    component_members: dict[str, set[str]] = {}
    for origin in origins:
        component_members.setdefault(find(origin), set()).add(origin)
    components = [tuple(sorted(members)) for members in component_members.values()]
    component_id_by_origin: dict[str, str] = {}
    for members in components:
        component_id = _digest_json(
            {
                "protocol_version": COMPONENT_PROTOCOL_VERSION,
                "language": language,
                "origins": list(members),
            }
        )
        for origin in members:
            component_id_by_origin[origin] = component_id

    def component_tie_hash(members: tuple[str, ...]) -> str:
        return _digest_json(
            {
                "protocol_version": COMPONENT_PROTOCOL_VERSION,
                "language": language,
                "seed": seed,
                "origins": list(members),
            }
        )

    ordered_components = sorted(
        components, key=lambda members: (-len(members), component_tie_hash(members))
    )
    test_chunks: list[list[str]] = [[] for _ in range(n_splits)]
    fold_counts = [0] * n_splits
    for members in ordered_components:
        target = min(range(n_splits), key=lambda fold: (fold_counts[fold], fold))
        test_chunks[target].extend(members)
        fold_counts[target] += len(members)
    if any(count < 2 for count in fold_counts):
        raise ValueError("content-component fold has fewer than two origins")

    def pair_spec(
        *,
        fold: int,
        side: str,
        source: str,
        human_origin: str,
        candidate_origin: str,
        label: int,
    ) -> T3PairSpec:
        human_index = bank[(human_origin, source)]
        candidate_index = bank[(candidate_origin, source)]
        human_code_hash = str(validated.human_code_sha256[human_index])
        candidate_code_hash = str(
            validated.candidate_code_sha256[candidate_index]
        )
        human_component_id = component_id_by_origin[human_origin]
        candidate_component_id = component_id_by_origin[candidate_origin]
        payload = {
            "protocol_version": PAIR_PROTOCOL_VERSION,
            "language": language,
            "seed": seed,
            "fold": fold,
            "side": side,
            "llm_source": source,
            "label": label,
            "human_origin_id": human_origin,
            "candidate_origin_id": candidate_origin,
            "human_code_sha256": human_code_hash,
            "candidate_code_sha256": candidate_code_hash,
            "human_component_id": human_component_id,
            "candidate_component_id": candidate_component_id,
            "human_positive_row_sha256": str(validated.row_sha256[human_index]),
            "candidate_positive_row_sha256": str(validated.row_sha256[candidate_index]),
        }
        if label == 0 and negative_pair_mode != "current":
            payload["negative_pair_mode"] = negative_pair_mode
        return T3PairSpec(
            human_positive_row_idx=human_index,
            candidate_positive_row_idx=candidate_index,
            label=label,
            human_origin_id=human_origin,
            candidate_origin_id=candidate_origin,
            human_code_sha256=human_code_hash,
            candidate_code_sha256=candidate_code_hash,
            human_component_id=human_component_id,
            candidate_component_id=candidate_component_id,
            llm_source=source,
            pair_sha256=_digest_json(payload),
        )

    def side_pairs(
        *, fold: int, side: str, side_origins: list[str], sources: tuple[str, ...]
    ) -> tuple[T3PairSpec, ...]:
        if len(side_origins) < 2:
            raise ValueError("each fold side needs at least two origins for derangement")
        pairs: list[T3PairSpec] = []
        for source in sources:
            ordered = sorted(
                side_origins,
                key=lambda origin: hashlib.sha256(
                    _canonical_json(
                        {
                            "protocol_version": PAIR_PROTOCOL_VERSION,
                            "language": language,
                            "seed": seed,
                            "fold": fold,
                            "side": side,
                            "llm_source": source,
                            "origin": origin,
                        }
                    )
                ).hexdigest(),
            )
            if negative_pair_mode == "current":
                offset = next(
                    (
                        candidate_offset
                        for candidate_offset in range(1, len(ordered))
                        if all(
                            component_id_by_origin[origin]
                            != component_id_by_origin[
                                ordered[(index + candidate_offset) % len(ordered)]
                            ]
                            for index, origin in enumerate(ordered)
                        )
                    ),
                    None,
                )
                if offset is None:
                    raise ValueError(
                        "no valid cross-component cyclic derangement for fold side"
                    )
                negative_candidates = {
                    human_origin: ordered[(index + offset) % len(ordered)]
                    for index, human_origin in enumerate(ordered)
                }
            else:
                negative_candidates: dict[str, str] = {}
                for human_origin in ordered:
                    eligible = [
                        candidate_origin
                        for candidate_origin in ordered
                        if component_id_by_origin[human_origin]
                        != component_id_by_origin[candidate_origin]
                    ]
                    if not eligible:
                        raise ValueError("no eligible cross-component negative")
                    if negative_pair_mode == "random":
                        candidate_origin = min(
                            eligible,
                            key=lambda candidate: _digest_json(
                                {
                                    "negative_pair_mode": negative_pair_mode,
                                    "language": language,
                                    "seed": seed,
                                    "fold": fold,
                                    "side": side,
                                    "llm_source": source,
                                    "human_origin": human_origin,
                                    "candidate_origin": candidate,
                                }
                            ),
                        )
                    else:
                        human_index = bank[(human_origin, source)]
                        candidate_origin = min(
                            eligible,
                            key=lambda candidate: (
                                float(
                                    np.linalg.norm(
                                        validated.human[human_index]
                                        - validated.llm[bank[(candidate, source)]]
                                    )
                                ),
                                _digest_json(
                                    {
                                        "negative_pair_mode": negative_pair_mode,
                                        "language": language,
                                        "seed": seed,
                                        "fold": fold,
                                        "side": side,
                                        "llm_source": source,
                                        "human_origin": human_origin,
                                        "candidate_origin": candidate,
                                    }
                                ),
                            ),
                        )
                    negative_candidates[human_origin] = candidate_origin
            for human_origin in ordered:
                candidate_origin = negative_candidates[human_origin]
                pairs.append(
                    pair_spec(
                        fold=fold,
                        side=side,
                        source=source,
                        human_origin=human_origin,
                        candidate_origin=human_origin,
                        label=1,
                    )
                )
                pairs.append(
                    pair_spec(
                        fold=fold,
                        side=side,
                        source=source,
                        human_origin=human_origin,
                        candidate_origin=candidate_origin,
                        label=0,
                    )
                )
        return tuple(pairs)

    splits: list[_PairSplitData] = []
    origin_set = set(origins)
    for fold, test_origins in enumerate(test_chunks):
        test_set = set(test_origins)
        train_origins = [origin for origin in origins if origin not in test_set]
        train_pairs = side_pairs(
            fold=fold,
            side="train",
            side_origins=train_origins,
            sources=train_sources,
        )
        test_pairs = side_pairs(
            fold=fold,
            side="test",
            side_origins=test_origins,
            sources=test_sources,
        )
        train_endpoints = {
            endpoint
            for pair in train_pairs
            for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
        }
        test_endpoints = {
            endpoint
            for pair in test_pairs
            for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
        }
        overlap = train_endpoints & test_endpoints
        if overlap or train_endpoints | test_endpoints != origin_set:
            raise ValueError(f"T3 pair endpoint leakage: {sorted(overlap)[:3]}")
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
        if train_hashes & test_hashes:
            raise ValueError("T3 exact code content leakage")
        splits.append(
            _PairSplitData(
                fold=fold,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                train_pair_sha256=_digest_json(
                    [pair.pair_sha256 for pair in train_pairs]
                ),
                test_pair_sha256=_digest_json(
                    [pair.pair_sha256 for pair in test_pairs]
                ),
            )
        )
    return splits


def build_t3_splits(
    cache: EnhancedFeatureCache,
    language: str | None = None,
    heldout_llm: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> list[T3PairSplit]:
    """Build strict folds trained on three LLMs and tested on one held-out LLM."""

    if type(heldout_llm) is not str or heldout_llm not in LLM_SOURCES:
        raise ValueError(f"unsupported held-out LLM: {heldout_llm!r}")
    train_sources = tuple(source for source in LLM_SOURCES if source != heldout_llm)
    raw_splits = _build_pair_splits(
        cache,
        language,
        n_splits,
        seed,
        train_sources=train_sources,
        test_sources=(heldout_llm,),
    )
    return [
        T3PairSplit(
            fold=split.fold,
            heldout_llm=heldout_llm,
            train_pairs=split.train_pairs,
            test_pairs=split.test_pairs,
            train_pair_sha256=split.train_pair_sha256,
            test_pair_sha256=split.test_pair_sha256,
        )
        for split in raw_splits
    ]


def build_t1_pair_splits(
    cache: EnhancedFeatureCache,
    language: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
    negative_pair_mode: str = "current",
) -> list[T1PairSplit]:
    """Build strict-origin folds under one registered negative-pair mode."""

    raw_splits = _build_pair_splits(
        cache,
        language,
        n_splits,
        seed,
        train_sources=LLM_SOURCES,
        test_sources=LLM_SOURCES,
        negative_pair_mode=negative_pair_mode,
    )
    return [
        T1PairSplit(
            fold=split.fold,
            train_pairs=split.train_pairs,
            test_pairs=split.test_pairs,
            train_pair_sha256=split.train_pair_sha256,
            test_pair_sha256=split.test_pair_sha256,
        )
        for split in raw_splits
    ]


T3_FOLD_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "config_id",
        "language",
        "heldout_llm",
        "method",
        "feature_family",
        "representation",
        "model",
        "seed",
        "fold",
        "split_protocol",
        "pair_protocol",
        "component_protocol",
        "gate_a_sha256",
        "gate_a_manifest_sha256",
        "cache_content_sha256",
        "source_jsonl_sha256",
        "record_sha256",
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
        "train_index_sha256",
        "test_index_sha256",
        "feature_dimensions",
        "train_unique_origins",
        "test_unique_origins",
        "train_unique_code_hashes",
        "test_unique_code_hashes",
        "train_unique_components",
        "test_unique_components",
        "train_llm_sources",
        "test_llm_sources",
        "train_llm_label_counts",
        "test_llm_label_counts",
        "train_human_parse_failures",
        "train_candidate_parse_failures",
        "test_human_parse_failures",
        "test_candidate_parse_failures",
        *METRIC_FIELDS,
    }
)


def _strict_int(value: Any) -> bool:
    return type(value) is int


def _cache_content_sha256(cache: EnhancedFeatureCache) -> str:
    """Hash all row-aligned cache values consumed by T3."""

    validated = _validate_arrays(_cache_as_arrays(cache), language=cache.language)
    digest = hashlib.sha256()
    digest.update(b"lpcode-v1/t3-runner-cache-v1\0")
    digest.update(cache.language.encode("ascii"))
    for name in cache.__dataclass_fields__:
        if name == "language":
            continue
        array = np.asarray(getattr(validated, name))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(_canonical_json(list(array.shape)))
        if array.dtype.kind == "f":
            payload = np.ascontiguousarray(array, dtype="<f8").tobytes()
        elif array.dtype.kind in "iu":
            payload = np.ascontiguousarray(array, dtype="<i8").tobytes()
        elif array.dtype.kind == "b":
            payload = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
        else:
            payload = _canonical_json(array.astype(str).tolist())
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _runner_feature_contract() -> dict[str, Any]:
    contract = _feature_contract()
    return {
        "cache_version": CACHE_VERSION,
        "cache_feature_contract": contract,
        "cache_feature_contract_sha256": _digest_json(contract),
        "official_feature_count": OFFICIAL_FEATURE_COUNT,
        "enhanced_feature_count": ENHANCED_FEATURE_COUNT,
        "combined_feature_count": FEATURE_COUNT,
    }


def _runner_package_versions() -> dict[str, str]:
    return {**_package_contract(), **_package_versions()}


def _runner_implementation_contract() -> dict[str, str]:
    module = Path(__file__).resolve()
    return {
        "runner_pair_cache_source_sha256": _sha256(module),
        "experiment_source_sha256": _sha256(module.with_name("experiment.py")),
        "representations_source_sha256": _sha256(
            module.with_name("representations.py")
        ),
        "data_normalization_source_sha256": _sha256(module.with_name("data.py")),
        "official_cache_runner_source_sha256": _sha256(module.with_name("t1.py")),
    }


def _validate_t3_axes(
    languages: tuple[str, ...],
    heldout_llms: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    methods: tuple[str, ...],
    limit_origins: int | None,
) -> None:
    if (
        not languages
        or len(set(languages)) != len(languages)
        or any(language not in LANGUAGES for language in languages)
    ):
        raise ValueError("languages must be a nonempty unique supported subset")
    if (
        not heldout_llms
        or len(set(heldout_llms)) != len(heldout_llms)
        or any(source not in LLM_SOURCES for source in heldout_llms)
    ):
        raise ValueError("heldout_llms must be a nonempty unique supported subset")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not _strict_int(seed) for seed in seeds)
    ):
        raise ValueError("seeds must be nonempty unique integers")
    if not _strict_int(n_splits) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(method not in T3_METHODS for method in methods)
    ):
        raise ValueError("methods must be a nonempty unique supported subset")
    if limit_origins is not None and (
        not _strict_int(limit_origins) or limit_origins < 2 * n_splits
    ):
        raise ValueError("limit_origins must provide at least two origins per fold")


def _build_t3_config(
    languages: tuple[str, ...],
    heldout_llms: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    methods: tuple[str, ...],
    limit_origins: int | None,
    dataset_paths: dict[str, Path],
    caches: dict[str, EnhancedFeatureCache],
    gate_a_path: str | Path,
    gate_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_t3_axes(
        languages, heldout_llms, seeds, n_splits, methods, limit_origins
    )
    if set(dataset_paths) != set(languages) or set(caches) != set(languages):
        raise ValueError("T3 datasets and caches must exactly match languages")
    if gate_binding is None:
        gate_binding = _load_strict_gate_a(gate_a_path)
    full_method_contract = _method_contract(gate_binding["selected_candidate"])
    current_source_hashes = {
        language: _sha256(dataset_paths[language]) for language in languages
    }
    if any(
        current_source_hashes[language]
        != gate_binding["source_jsonl_sha256"].get(language)
        for language in languages
    ):
        raise ValueError("T3 dataset does not match strict Gate A dataset hash")
    payload: dict[str, Any] = {
        "schema_version": T3_SCHEMA_VERSION,
        "task": "task3_unseen_llm",
        "fold_index_base": 0,
        "languages": list(languages),
        "heldout_llms": list(heldout_llms),
        "seeds": list(seeds),
        "n_splits": n_splits,
        "methods": list(methods),
        "method_contract": {
            method: full_method_contract[method] for method in methods
        },
        "limit_origins": limit_origins,
        "split_protocol": T3_SPLIT_PROTOCOL_VERSION,
        "pair_protocol": PAIR_PROTOCOL_VERSION,
        "component_protocol": COMPONENT_PROTOCOL_VERSION,
        "gate_a_binding": gate_binding,
        "feature_contract": _runner_feature_contract(),
        "source_jsonl_sha256": current_source_hashes,
        "cache_content_sha256": {
            language: _cache_content_sha256(caches[language])
            for language in languages
        },
        "package_versions": _runner_package_versions(),
        "implementation_contract": _runner_implementation_contract(),
    }
    payload["config_id"] = _digest_json(payload)
    return payload


def _validate_t3_config(config: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "task",
        "fold_index_base",
        "languages",
        "heldout_llms",
        "seeds",
        "n_splits",
        "methods",
        "method_contract",
        "limit_origins",
        "split_protocol",
        "pair_protocol",
        "component_protocol",
        "gate_a_binding",
        "feature_contract",
        "source_jsonl_sha256",
        "cache_content_sha256",
        "package_versions",
        "implementation_contract",
        "config_id",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("invalid T3 run config")
    if any(
        type(config[field]) is not list
        for field in ("languages", "heldout_llms", "seeds", "methods")
    ):
        raise ValueError("invalid T3 run config")
    try:
        _validate_t3_axes(
            tuple(config["languages"]),
            tuple(config["heldout_llms"]),
            tuple(config["seeds"]),
            config["n_splits"],
            tuple(config["methods"]),
            config["limit_origins"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid T3 run config") from exc
    gate_binding = config["gate_a_binding"]
    expected_methods = (
        _method_contract(gate_binding.get("selected_candidate", {}))
        if isinstance(gate_binding, dict)
        else {}
    )
    if (
        not _strict_int(config["schema_version"])
        or config["schema_version"] != T3_SCHEMA_VERSION
        or config["task"] != "task3_unseen_llm"
        or not _strict_int(config["fold_index_base"])
        or config["fold_index_base"] != 0
        or config["split_protocol"] != T3_SPLIT_PROTOCOL_VERSION
        or config["pair_protocol"] != PAIR_PROTOCOL_VERSION
        or config["component_protocol"] != COMPONENT_PROTOCOL_VERSION
        or config["method_contract"]
        != {method: expected_methods[method] for method in config["methods"]}
        or config["feature_contract"] != _runner_feature_contract()
        or config["package_versions"] != _runner_package_versions()
        or config["implementation_contract"] != _runner_implementation_contract()
        or not _is_sha256(config["config_id"])
    ):
        raise ValueError("invalid T3 run config")
    for field in ("source_jsonl_sha256", "cache_content_sha256"):
        values = config[field]
        if (
            not isinstance(values, dict)
            or set(values) != set(config["languages"])
            or any(not _is_sha256(value) for value in values.values())
        ):
            raise ValueError("invalid T3 run config")
    without_id = {key: value for key, value in config.items() if key != "config_id"}
    if config["config_id"] != _digest_json(without_id):
        raise ValueError("invalid T3 run config digest")
    return config


def _load_or_write_t3_config(path: Path, current: dict[str, Any]) -> None:
    _validate_t3_config(current)
    if not path.exists():
        _atomic_write_json(path, current)
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid T3 run config") from exc
    _validate_t3_config(existing)
    if existing != current:
        raise ValueError("config mismatch: refusing to combine incomparable T3 folds")


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    values = (
        record.get("language"),
        record.get("heldout_llm"),
        record.get("method"),
        record.get("seed"),
        record.get("fold"),
    )
    if (
        any(type(value) is not str for value in values[:3])
        or any(not _strict_int(value) for value in values[3:])
    ):
        raise ValueError("invalid T3 fold record key")
    return values  # type: ignore[return-value]


def _t3_record_sha256(record: dict[str, Any]) -> str:
    return _digest_json(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _valid_balanced_counts(value: Any, rows: int) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"0", "1"}
        and all(_strict_int(count) and count > 0 for count in value.values())
        and value["0"] == value["1"]
        and sum(value.values()) == rows
    )


def _valid_llm_counts(value: Any, sources: list[str], rows: int) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(sources)
        and all(
            isinstance(counts, dict)
            and set(counts) == {"0", "1"}
            and all(_strict_int(count) and count > 0 for count in counts.values())
            and counts["0"] == counts["1"]
            for counts in value.values()
        )
        and sum(sum(counts.values()) for counts in value.values()) == rows
    )


def _validate_metric_fields(record: dict[str, Any]) -> None:
    numeric = (
        "f1",
        "precision",
        "recall",
        "auroc",
        "mcc",
        "fit_seconds",
        "predict_seconds",
    )
    if any(
        isinstance(record[field], bool)
        or not isinstance(record[field], (int, float))
        or not np.isfinite(record[field])
        for field in numeric
    ):
        raise ValueError("non-finite or invalid T3 fold metric")
    if (
        any(not 0.0 <= float(record[field]) <= 1.0 for field in numeric[:4])
        or not -1.0 <= float(record["mcc"]) <= 1.0
        or float(record["fit_seconds"]) < 0.0
        or float(record["predict_seconds"]) < 0.0
    ):
        raise ValueError("invalid T3 fold metric range")


def _validate_t3_record(
    record: Any, config: dict[str, Any]
) -> tuple[str, str, str, int, int]:
    if not isinstance(record, dict) or set(record) != T3_FOLD_RECORD_FIELDS:
        raise ValueError("malformed T3 fold record schema")
    key = _record_key(record)
    language, heldout, method, seed, fold = key
    method_spec = config["method_contract"].get(method)
    expected_train_sources = [source for source in LLM_SOURCES if source != heldout]
    expected_test_sources = [heldout]
    if (
        not _strict_int(record["schema_version"])
        or record["schema_version"] != T3_SCHEMA_VERSION
        or record["config_id"] != config["config_id"]
        or language not in config["languages"]
        or heldout not in config["heldout_llms"]
        or method not in config["methods"]
        or seed not in config["seeds"]
        or not _strict_int(fold)
        or fold < 0
        or fold >= config["n_splits"]
        or not isinstance(method_spec, dict)
        or record["feature_family"] != method_spec["feature_family"]
        or record["representation"] != method_spec["representation"]
        or record["model"] != method_spec["model"]
        or record["feature_dimensions"] != method_spec["feature_dimensions"]
        or record["split_protocol"] != T3_SPLIT_PROTOCOL_VERSION
        or record["pair_protocol"] != PAIR_PROTOCOL_VERSION
        or record["component_protocol"] != COMPONENT_PROTOCOL_VERSION
        or record["gate_a_sha256"] != config["gate_a_binding"]["gate_a_sha256"]
        or record["gate_a_manifest_sha256"]
        != config["gate_a_binding"]["manifest_sha256"]
        or record["cache_content_sha256"] != config["cache_content_sha256"][language]
        or record["source_jsonl_sha256"] != config["source_jsonl_sha256"][language]
        or record["train_llm_sources"] != expected_train_sources
        or record["test_llm_sources"] != expected_test_sources
        or not _is_sha256(record["train_index_sha256"])
        or not _is_sha256(record["test_index_sha256"])
    ):
        raise ValueError("T3 fold record schema/config mismatch")
    strict_zero = (
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
    )
    positive_ints = (
        "train_rows",
        "test_rows",
        "train_unique_origins",
        "test_unique_origins",
        "train_unique_code_hashes",
        "test_unique_code_hashes",
        "train_unique_components",
        "test_unique_components",
    )
    parse_counts = (
        "train_human_parse_failures",
        "train_candidate_parse_failures",
        "test_human_parse_failures",
        "test_candidate_parse_failures",
    )
    if (
        any(not _strict_int(record[field]) or record[field] != 0 for field in strict_zero)
        or any(not _strict_int(record[field]) or record[field] <= 0 for field in positive_ints)
        or any(not _strict_int(record[field]) or record[field] < 0 for field in parse_counts)
        or not _valid_balanced_counts(record["train_class_counts"], record["train_rows"])
        or not _valid_balanced_counts(record["test_class_counts"], record["test_rows"])
        or not _valid_llm_counts(
            record["train_llm_label_counts"], expected_train_sources, record["train_rows"]
        )
        or not _valid_llm_counts(
            record["test_llm_label_counts"], expected_test_sources, record["test_rows"]
        )
    ):
        raise ValueError("invalid T3 fold record schema")
    _validate_metric_fields(record)
    if (
        not _is_sha256(record["record_sha256"])
        or record["record_sha256"] != _t3_record_sha256(record)
    ):
        raise ValueError("invalid T3 fold record digest")
    return key


def _load_t3_records(
    path: Path, config: dict[str, Any]
) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"malformed T3 fold record at line {line_number}")
        try:
            record = json.loads(line)
            key = _validate_t3_record(record, config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed T3 fold record at line {line_number}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"{exc} at line {line_number}") from exc
        if key in records:
            raise ValueError("duplicate T3 fold record key")
        records[key] = record
    return records


def _select_t3_positive_bank(
    cache: EnhancedFeatureCache, limit_origins: int | None
) -> EnhancedFeatureCache:
    positive = cache.labels == 1
    origins = sorted(set(cache.human_origin_ids[positive].tolist()))
    if limit_origins is not None:
        origins = origins[:limit_origins]
    keep = positive & np.isin(cache.human_origin_ids, np.asarray(origins, dtype=str))
    if not keep.any():
        raise ValueError("T3 positive bank is empty")
    fields = {
        field: np.asarray(getattr(cache, field))[keep]
        for field in cache.__dataclass_fields__
        if field != "language"
    }
    return replace(cache, **fields)


def _labels(pairs: tuple[T3PairSpec, ...]) -> np.ndarray:
    return np.asarray([pair.label for pair in pairs], dtype=np.int64)


def _llm_label_counts(
    pairs: tuple[T3PairSpec, ...], sources: list[str]
) -> dict[str, dict[str, int]]:
    return {
        source: {
            str(label): sum(
                pair.llm_source == source and pair.label == label for pair in pairs
            )
            for label in (0, 1)
        }
        for source in sources
    }


def _t3_side_metadata(
    cache: EnhancedFeatureCache,
    prefix: str,
    pairs: tuple[T3PairSpec, ...],
    sources: list[str],
) -> dict[str, Any]:
    endpoints = {
        endpoint
        for pair in pairs
        for endpoint in (pair.human_origin_id, pair.candidate_origin_id)
    }
    hashes = {
        code_hash
        for pair in pairs
        for code_hash in (pair.human_code_sha256, pair.candidate_code_sha256)
    }
    components = {
        component
        for pair in pairs
        for component in (pair.human_component_id, pair.candidate_component_id)
    }
    labels = _labels(pairs)
    human_failures = sum(
        not bool(cache.human_parse_ok[pair.human_positive_row_idx]) for pair in pairs
    )
    candidate_failures = sum(
        not bool(cache.llm_parse_ok[pair.candidate_positive_row_idx]) for pair in pairs
    )
    return {
        f"{prefix}_rows": len(pairs),
        f"{prefix}_class_counts": {
            str(label): int((labels == label).sum()) for label in (0, 1)
        },
        f"{prefix}_unique_origins": len(endpoints),
        f"{prefix}_unique_code_hashes": len(hashes),
        f"{prefix}_unique_components": len(components),
        f"{prefix}_llm_sources": sources,
        f"{prefix}_llm_label_counts": _llm_label_counts(pairs, sources),
        f"{prefix}_human_parse_failures": human_failures,
        f"{prefix}_candidate_parse_failures": candidate_failures,
    }


def _t3_split_metadata(
    cache: EnhancedFeatureCache, split: T3PairSplit
) -> dict[str, Any]:
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
    train_sources = [source for source in LLM_SOURCES if source != split.heldout_llm]
    test_sources = [split.heldout_llm]
    return {
        "leakage_count": endpoint_leakage + content_leakage,
        "endpoint_leakage_count": endpoint_leakage,
        "content_leakage_count": content_leakage,
        "negative_component_violation_count": negative_violations,
        "train_index_sha256": split.train_pair_sha256,
        "test_index_sha256": split.test_pair_sha256,
        **_t3_side_metadata(cache, "train", split.train_pairs, train_sources),
        **_t3_side_metadata(cache, "test", split.test_pairs, test_sources),
    }


def _t3_pair_matrix(
    cache: EnhancedFeatureCache,
    pairs: tuple[T3PairSpec, ...],
    method_spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    human_indices = np.asarray(
        [pair.human_positive_row_idx for pair in pairs], dtype=np.int64
    )
    candidate_indices = np.asarray(
        [pair.candidate_positive_row_idx for pair in pairs], dtype=np.int64
    )
    count = method_spec["feature_count"]
    matrix = build_representation(
        cache.human[human_indices, :count],
        cache.llm[candidate_indices, :count],
        method_spec["representation"],
    )
    return matrix, _labels(pairs)


def _validate_record_reconstruction(
    record: dict[str, Any],
    config: dict[str, Any],
    cache: EnhancedFeatureCache,
    split: T3PairSplit,
    matrix_dimensions: int,
) -> None:
    expected = {
        **_t3_split_metadata(cache, split),
        "feature_dimensions": matrix_dimensions,
        "cache_content_sha256": config["cache_content_sha256"][record["language"]],
        "source_jsonl_sha256": config["source_jsonl_sha256"][record["language"]],
        "gate_a_sha256": config["gate_a_binding"]["gate_a_sha256"],
        "gate_a_manifest_sha256": config["gate_a_binding"]["manifest_sha256"],
    }
    if any(record[field] != value for field, value in expected.items()):
        raise ValueError("completed T3 fold split/cache reconstruction mismatch")


def _run_t3_locked(
    output_root: str | Path,
    languages: tuple[str, ...] = LANGUAGES,
    heldout_llms: tuple[str, ...] = LLM_SOURCES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    methods: tuple[str, ...] = T3_METHODS,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path = DEFAULT_T3_CACHE_ROOT,
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    gate_a_path: str | Path = DEFAULT_GATE_A_PATH,
) -> dict[str, Any]:
    language_axis = tuple(languages)
    heldout_axis = tuple(heldout_llms)
    seed_axis = tuple(seeds)
    method_axis = tuple(methods)
    _validate_t3_axes(
        language_axis, heldout_axis, seed_axis, n_splits, method_axis, limit_origins
    )
    output = resolve_output_path(output_root)
    if dataset_paths is not None and set(dataset_paths) != set(language_axis):
        raise ValueError("T3 dataset paths must exactly match configured languages")
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
    if set(paths) != set(language_axis) or any(not path.is_file() for path in paths.values()):
        raise ValueError("T3 dataset does not exist")
    gate_binding = _load_strict_gate_a(gate_a_path)
    if any(
        _sha256(paths[language])
        != gate_binding["source_jsonl_sha256"].get(language)
        for language in language_axis
    ):
        raise ValueError("T3 dataset does not match strict Gate A dataset hash")
    caches = {
        language: _select_t3_positive_bank(
            load_or_build_enhanced_cache(
                language, paths[language], cache_root, official_cache_root
            ),
            limit_origins,
        )
        for language in language_axis
    }
    config = _build_t3_config(
        language_axis,
        heldout_axis,
        seed_axis,
        n_splits,
        method_axis,
        limit_origins,
        paths,
        caches,
        gate_a_path,
        gate_binding,
    )
    output.mkdir(parents=True, exist_ok=True)
    _load_or_write_t3_config(output / "config.json", config)
    folds_path = output / "folds.jsonl"
    records = _load_t3_records(folds_path, config)
    completed = 0
    skipped = 0
    for language in language_axis:
        cache = caches[language]
        source_digest = config["source_jsonl_sha256"][language]
        if _sha256(paths[language]) != source_digest:
            raise ValueError("T3 dataset changed since run config")
        for heldout in heldout_axis:
            for seed in seed_axis:
                splits = build_t3_splits(
                    cache,
                    language=language,
                    heldout_llm=heldout,
                    n_splits=n_splits,
                    seed=seed,
                )
                if len(splits) != n_splits:
                    raise ValueError("T3 pair builder returned wrong fold count")
                for split in splits:
                    split_metadata = _t3_split_metadata(cache, split)
                    if any(
                        split_metadata[field] != 0
                        for field in (
                            "leakage_count",
                            "endpoint_leakage_count",
                            "content_leakage_count",
                            "negative_component_violation_count",
                        )
                    ):
                        raise ValueError("T3 pair builder violated strict isolation")
                    bindings = split.pairs_for_methods(method_axis)
                    for method in method_axis:
                        method_spec = config["method_contract"][method]
                        train_pairs, test_pairs = bindings[method]
                        train_matrix, train_labels = _t3_pair_matrix(
                            cache, train_pairs, method_spec
                        )
                        test_matrix, test_labels = _t3_pair_matrix(
                            cache, test_pairs, method_spec
                        )
                        dimensions = int(train_matrix.shape[1])
                        if (
                            dimensions != method_spec["feature_dimensions"]
                            or test_matrix.shape[1] != dimensions
                        ):
                            raise ValueError("T3 method feature dimension mismatch")
                        key = (language, heldout, method, seed, split.fold)
                        if key in records:
                            _validate_record_reconstruction(
                                records[key], config, cache, split, dimensions
                            )
                            if _sha256(paths[language]) != source_digest:
                                raise ValueError("T3 dataset changed during resume")
                            skipped += 1
                            continue
                        if _sha256(paths[language]) != source_digest:
                            raise ValueError("T3 dataset changed before evaluation")
                        metrics = evaluate_fold(
                            train_matrix,
                            train_labels,
                            test_matrix,
                            test_labels,
                            method_spec["model"],
                            seed,
                        )
                        if not isinstance(metrics, dict) or set(metrics) != set(METRIC_FIELDS):
                            raise ValueError("evaluator result schema does not match T3 contract")
                        _validate_metric_fields(metrics)
                        if any(
                            metrics[field] != split_metadata[field]
                            for field in (
                                "train_rows",
                                "test_rows",
                                "train_class_counts",
                                "test_class_counts",
                            )
                        ):
                            raise ValueError(
                                "T3 evaluator row/class counts disagree with split"
                            )
                        if _sha256(paths[language]) != source_digest:
                            raise ValueError("T3 dataset changed during evaluation")
                        record = {
                            **metrics,
                            **split_metadata,
                            "schema_version": T3_SCHEMA_VERSION,
                            "config_id": config["config_id"],
                            "language": language,
                            "heldout_llm": heldout,
                            "method": method,
                            "feature_family": method_spec["feature_family"],
                            "representation": method_spec["representation"],
                            "model": method_spec["model"],
                            "seed": seed,
                            "fold": split.fold,
                            "split_protocol": T3_SPLIT_PROTOCOL_VERSION,
                            "pair_protocol": PAIR_PROTOCOL_VERSION,
                            "component_protocol": COMPONENT_PROTOCOL_VERSION,
                            "gate_a_sha256": config["gate_a_binding"]["gate_a_sha256"],
                            "gate_a_manifest_sha256": config["gate_a_binding"]["manifest_sha256"],
                            "cache_content_sha256": config["cache_content_sha256"][language],
                            "source_jsonl_sha256": source_digest,
                            "feature_dimensions": dimensions,
                        }
                        record["record_sha256"] = _t3_record_sha256(record)
                        _validate_t3_record(record, config)
                        _validate_record_reconstruction(
                            record, config, cache, split, dimensions
                        )
                        records = _load_t3_records(folds_path, config)
                        if key in records:
                            raise ValueError("duplicate T3 fold record key")
                        records[key] = record
                        _atomic_write_records(folds_path, records)
                        completed += 1
        if _sha256(paths[language]) != source_digest:
            raise ValueError("T3 dataset changed during language evaluation")
    expected = _evaluation_count(
        language_axis, heldout_axis, seed_axis, n_splits, method_axis
    )
    return {
        "schema_version": T3_SCHEMA_VERSION,
        "config_id": config["config_id"],
        "expected": expected,
        "completed": completed,
        "skipped": skipped,
        "output_root": str(output),
    }


def run_t3(
    output_root: str | Path = RESULTS_ROOT / "02_unseen_llm",
    languages: tuple[str, ...] = LANGUAGES,
    heldout_llms: tuple[str, ...] = LLM_SOURCES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = 5,
    methods: tuple[str, ...] = T3_METHODS,
    limit_origins: int | None = None,
    dataset_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path = DEFAULT_T3_CACHE_ROOT,
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    gate_a_path: str | Path = DEFAULT_GATE_A_PATH,
) -> dict[str, Any]:
    """Run T3 under one exclusive output-root lock."""

    output = resolve_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        return _run_t3_locked(
            output,
            languages,
            heldout_llms,
            seeds,
            n_splits,
            methods,
            limit_origins,
            dataset_paths,
            cache_root,
            official_cache_root,
            gate_a_path,
        )


def run_t3_smoke(
    output_root: str | Path,
    dataset_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path = DEFAULT_T3_CACHE_ROOT,
    official_cache_root: str | Path = DEFAULT_OFFICIAL_CACHE_ROOT,
    gate_a_path: str | Path = DEFAULT_GATE_A_PATH,
) -> dict[str, Any]:
    """Run the bounded 32-evaluation C/all-holdout T3 smoke matrix."""

    return run_t3(
        output_root,
        languages=("c",),
        heldout_llms=LLM_SOURCES,
        seeds=(42,),
        n_splits=2,
        methods=T3_METHODS,
        limit_origins=T3_SMOKE_ORIGINS,
        dataset_paths=dataset_paths,
        cache_root=cache_root,
        official_cache_root=official_cache_root,
        gate_a_path=gate_a_path,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT / "02_unseen_llm",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_T3_CACHE_ROOT)
    parser.add_argument(
        "--official-cache-root", type=Path, default=DEFAULT_OFFICIAL_CACHE_ROOT
    )
    args = parser.parse_args()
    if args.smoke and args.summarize_only:
        parser.error("--smoke and --summarize-only are mutually exclusive")
    if args.summarize_only:
        from .gates_t3 import summarize_t3

        report = summarize_t3(args.output_root)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return
    runner = run_t3_smoke if args.smoke else run_t3
    report = runner(
        args.output_root,
        cache_root=args.cache_root,
        official_cache_root=args.official_cache_root,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


__all__ = [
    "CACHE_VERSION",
    "COMPONENT_PROTOCOL_VERSION",
    "DEFAULT_GATE_A_PATH",
    "DEFAULT_SEEDS",
    "DEFAULT_T3_CACHE_ROOT",
    "EnhancedFeatureCache",
    "FEATURE_COUNT",
    "LLM_SOURCES",
    "PAIR_PROTOCOL_VERSION",
    "T3_FOLD_RECORD_FIELDS",
    "T3_METHODS",
    "T3_SCHEMA_VERSION",
    "T3_SPLIT_PROTOCOL_VERSION",
    "T1PairSplit",
    "T3PairSpec",
    "T3PairSplit",
    "T3Split",
    "build_t1_pair_splits",
    "build_t3_splits",
    "load_or_build_enhanced_cache",
    "run_t3",
    "run_t3_smoke",
]


if __name__ == "__main__":
    main()
