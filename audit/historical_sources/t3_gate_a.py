"""Task 3 enhanced-feature cache and strict unseen-LLM split primitives.

This module deliberately contains no model fitting, result ledger, CLI, or gate
logic.  The cache appends the pre-registered enhanced 18-vector to the frozen
official 10-vector without changing row order.  Its semantic digest detects
uncoordinated cache/metadata corruption; coordinated rewriting of every digest
and metadata field is outside the threat model without a trusted signature.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
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
from .paths import REPRO_ROOT, RESULTS_ROOT, resolve_output_path
from .t1 import _cache_paths as _official_cache_paths
from .t1 import load_or_build_feature_cache


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
            rotated = ordered[offset:] + ordered[:offset]
            for human_origin, candidate_origin in zip(ordered, rotated):
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
) -> list[T1PairSplit]:
    """Build strict-origin folds containing every LLM on both fold sides."""

    raw_splits = _build_pair_splits(
        cache,
        language,
        n_splits,
        seed,
        train_sources=LLM_SOURCES,
        test_sources=LLM_SOURCES,
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


__all__ = [
    "CACHE_VERSION",
    "COMPONENT_PROTOCOL_VERSION",
    "EnhancedFeatureCache",
    "FEATURE_COUNT",
    "LLM_SOURCES",
    "PAIR_PROTOCOL_VERSION",
    "T1PairSplit",
    "T3PairSpec",
    "T3PairSplit",
    "T3Split",
    "build_t1_pair_splits",
    "build_t3_splits",
    "load_or_build_enhanced_cache",
]
