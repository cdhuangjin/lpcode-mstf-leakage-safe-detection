from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


LLM_SOURCES = (
    "gpt3.5",
    "gemini-pro",
    "wizardcoder:33b-v1.1",
    "deepseek-coder:33b-instruct",
)


def _hold_t3_cache_lock(path: str, ready, release) -> None:
    from lpcode_v1.t3 import _exclusive_cache_lock

    with _exclusive_cache_lock(Path(path)):
        ready.set()
        release.wait(15)


def _observe_t3_cache_lock(path: str, acquired) -> None:
    from lpcode_v1.t3 import _exclusive_cache_lock

    with _exclusive_cache_lock(Path(path)):
        acquired.set()


def _hold_t3_cache_lock_for(path: str, ready, seconds: float) -> None:
    from lpcode_v1.t3 import _exclusive_cache_lock

    with _exclusive_cache_lock(Path(path)):
        ready.set()
        time.sleep(seconds)


def _hold_t3_output_lock(path: str, ready, release) -> None:
    from lpcode_v1.t3 import _exclusive_output_lock

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        ready.set()
        release.wait(15)


def _rows(group_count: int = 5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index in range(group_count):
        group = f"group-{group_index}.c"
        for llm_source in LLM_SOURCES:
            for label in (0, 1):
                row_index = len(rows)
                row = {
                    "human_src": f"int human_{group_index}(void) {{ return {group_index}; }}",
                    "llm_src": f"int llm_{row_index}(void) {{ return {row_index}; }}",
                    "paraphrased_by": llm_source,
                    "label": label,
                }
                if label == 1:
                    row["file_name"] = group
                else:
                    row["human_file_name"] = group
                    row["llm_file_name"] = f"group-{(group_index + 1) % group_count}.c"
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _patch_extractors(monkeypatch: pytest.MonkeyPatch) -> None:
    from lpcode_v1 import t1, t3
    from lpcode_v1.features_enhanced import EnhancedAnalysis

    monkeypatch.setattr(
        t1,
        "analyze_code",
        lambda code, language: np.arange(10, dtype=np.float64),
    )
    monkeypatch.setattr(
        t3,
        "analyze_enhanced",
        lambda code, language: EnhancedAnalysis(
            np.arange(18, dtype=np.float64), True, "tree-sitter", language
        ),
    )


def test_t3_method_contract_is_exact_960_matrix_and_strict_gate_bound(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t3

    binding = t3._load_strict_gate_a(t3.DEFAULT_GATE_A_PATH)
    methods = t3._method_contract(binding["selected_candidate"])

    assert t3.T3_METHODS == (
        "lpcode_original",
        "xgb_original",
        "best_transition",
        "mstf",
    )
    assert methods == {
        "lpcode_original": {
            "feature_family": "official10",
            "feature_count": 10,
            "representation": "concat",
            "model": "mlp",
            "feature_dimensions": 20,
        },
        "xgb_original": {
            "feature_family": "official10",
            "feature_count": 10,
            "representation": "concat",
            "model": "xgb",
            "feature_dimensions": 20,
        },
        "best_transition": {
            "feature_family": "official10",
            "feature_count": 10,
            "representation": "concat_delta",
            "model": "xgb",
            "feature_dimensions": 30,
        },
        "mstf": {
            "feature_family": "enhanced28",
            "feature_count": 28,
            "representation": "full",
            "model": "xgb",
            "feature_dimensions": 112,
        },
    }
    assert binding["protocol_version"] == "all-llm-strict-origin-v2"
    assert binding["strict_passed"] is True
    assert len(binding["gate_a_sha256"]) == 64
    assert len(binding["manifest_sha256"]) == 64
    assert t3._evaluation_count(
        t3.LANGUAGES,
        t3.LLM_SOURCES,
        t3.DEFAULT_SEEDS,
        5,
        t3.T3_METHODS,
    ) == 960

    with pytest.raises(ValueError, match="strict-origin"):
        t3._load_strict_gate_a(
            t3.RESULTS_ROOT / "01_transition_test" / "gate_a.json"
        )
    copied = tmp_path / "copied-strict-gate"
    shutil.copytree(t3.DEFAULT_GATE_A_PATH.parent, copied)
    with pytest.raises(ValueError, match="exact strict-origin"):
        t3._load_strict_gate_a(copied / "gate_a.json")


def test_t3_gate_rejects_self_consistent_manifest_with_malformed_strict_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    copied = tmp_path / "strict-gate"
    shutil.copytree(t3.DEFAULT_GATE_A_PATH.parent, copied)
    copied_gate = copied / "gate_a.json"
    monkeypatch.setattr(t3, "DEFAULT_GATE_A_PATH", copied_gate)
    config_path = copied / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["feature_contract"]["selected_feature_count"] = 28
    config["config_id"] = t3._digest_json(
        {key: value for key, value in config.items() if key != "config_id"}
    )
    config_path.write_bytes(t3._canonical_json(config))
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["config.json"] = {
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "bytes": config_path.stat().st_size,
    }
    manifest_path.write_bytes(t3._canonical_json(manifest))

    with pytest.raises(ValueError, match="strict-origin"):
        t3._load_strict_gate_a(copied_gate)


def test_t3_gate_rejects_empty_nested_provenance_with_all_digests_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    copied = tmp_path / "strict-gate"
    shutil.copytree(t3.DEFAULT_GATE_A_PATH.parent, copied)
    copied_gate = copied / "gate_a.json"
    monkeypatch.setattr(t3, "DEFAULT_GATE_A_PATH", copied_gate)
    config_path = copied / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["implementation_contract"] = {}
    config["config_id"] = t3._digest_json(
        {key: value for key, value in config.items() if key != "config_id"}
    )
    config_path.write_bytes(t3._canonical_json(config))
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config"]["config_id"] = config["config_id"]
    summary_path.write_bytes(t3._canonical_json(summary))
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, path in (("config.json", config_path), ("summary.json", summary_path)):
        manifest["files"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest_path.write_bytes(t3._canonical_json(manifest))

    with pytest.raises(ValueError, match="strict-origin"):
        t3._load_strict_gate_a(copied_gate)


def test_t3_gate_rejects_rewritten_package_and_feature_provenance_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    copied = tmp_path / "strict-gate"
    shutil.copytree(t3.DEFAULT_GATE_A_PATH.parent, copied)
    copied_gate = copied / "gate_a.json"
    monkeypatch.setattr(t3, "DEFAULT_GATE_A_PATH", copied_gate)
    config_path = copied / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["package_versions"] = {
        key: "99.99.99" for key in config["package_versions"]
    }
    enhanced = config["feature_contract"]["enhanced_feature_contract"]
    enhanced["enhanced_feature_source_sha256"] = "f" * 64
    config["feature_contract"]["enhanced_feature_contract_sha256"] = t3._digest_json(
        enhanced
    )
    config["config_id"] = t3._digest_json(
        {key: value for key, value in config.items() if key != "config_id"}
    )
    config_path.write_bytes(t3._canonical_json(config))
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config"]["config_id"] = config["config_id"]
    summary_path.write_bytes(t3._canonical_json(summary))
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, path in (("config.json", config_path), ("summary.json", summary_path)):
        manifest["files"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest_path.write_bytes(t3._canonical_json(manifest))

    with pytest.raises(ValueError, match="strict-origin"):
        t3._load_strict_gate_a(copied_gate)


def test_enhanced_cache_builds_aligned_float64_28_vectors_and_parse_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1, t3
    from lpcode_v1.features_enhanced import EnhancedAnalysis

    dataset = tmp_path / "c.jsonl"
    rows = _rows(2)
    _write_jsonl(dataset, rows)

    monkeypatch.setattr(
        t1,
        "analyze_code",
        lambda code, language: np.arange(10, dtype=np.float64),
    )
    calls: list[tuple[str, str]] = []

    def enhanced(code: str, language: str) -> EnhancedAnalysis:
        calls.append((code, language))
        parse_ok = "llm_1" not in code
        return EnhancedAnalysis(
            np.arange(18, dtype=np.float64),
            parse_ok,
            "tree-sitter" if parse_ok else "lexical-fallback",
            language,
            None if parse_ok else "syntax-error",
        )

    monkeypatch.setattr(t3, "analyze_enhanced", enhanced)
    cache_root = tmp_path / "enhanced-cache"
    official_root = tmp_path / "official-cache"
    cache = t3.load_or_build_enhanced_cache(
        "c", dataset, cache_root, official_root
    )

    assert cache.human.shape == (len(rows), 28)
    assert cache.language == "c"
    assert cache.llm.shape == (len(rows), 28)
    assert cache.human.dtype == np.float64
    assert cache.llm.dtype == np.float64
    np.testing.assert_array_equal(cache.human[:, :10], np.tile(np.arange(10), (len(rows), 1)))
    np.testing.assert_array_equal(cache.human[:, 10:], np.tile(np.arange(18), (len(rows), 1)))
    assert cache.labels.tolist() == [int(row["label"]) for row in rows]
    expected_human_origins = [
        str(row["file_name"] if row["label"] == 1 else row["human_file_name"])
        for row in rows
    ]
    expected_candidate_origins = [
        str(row["file_name"] if row["label"] == 1 else row["llm_file_name"])
        for row in rows
    ]
    assert cache.source_ids.tolist() == expected_human_origins
    assert cache.human_origin_ids.tolist() == expected_human_origins
    assert cache.candidate_origin_ids.tolist() == expected_candidate_origins
    assert cache.human_code_sha256.tolist() == [
        hashlib.sha256(str(row["human_src"]).encode("utf-8")).hexdigest()
        for row in rows
    ]
    assert cache.candidate_code_sha256.tolist() == [
        hashlib.sha256(str(row["llm_src"]).encode("utf-8")).hexdigest()
        for row in rows
    ]
    assert cache.llm_sources.tolist() == [str(row["paraphrased_by"]) for row in rows]
    assert cache.row_sha256.shape == (len(rows),)
    assert cache.row_sha256.dtype.kind in "US"
    assert all(len(digest) == 64 for digest in cache.row_sha256.tolist())
    assert cache.human_parse_ok.dtype == np.bool_
    assert cache.llm_parse_ok.dtype == np.bool_
    assert cache.human_backends.dtype.kind in "US"
    assert cache.llm_backends.dtype.kind in "US"
    unique_code_texts = {
        str(row[field]) for row in rows for field in ("human_src", "llm_src")
    }
    assert len(calls) == len(unique_code_texts)

    archive = cache_root / "enhanced28-v3" / "c.npz"
    metadata = json.loads(archive.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["cache_version"] == "enhanced28-v3"
    assert metadata["rows"] == len(rows)
    assert metadata["source_jsonl_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert metadata["npz_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert len(metadata["feature_contract_sha256"]) == 64
    assert len(metadata["package_contract_sha256"]) == 64
    assert len(metadata["official_cache_npz_sha256"]) == 64
    assert len(metadata["semantic_content_sha256"]) == 64

    monkeypatch.setattr(t3, "analyze_enhanced", lambda *args: pytest.fail("cache reuse extracted"))
    reused = t3.load_or_build_enhanced_cache("c", dataset, cache_root, official_root)
    np.testing.assert_array_equal(reused.human, cache.human)


def _memory_cache(group_count: int = 10):
    from lpcode_v1.t3 import EnhancedFeatureCache

    rows = _rows(group_count)
    count = len(rows)
    human_origins = np.asarray(
        [row["file_name"] if row["label"] == 1 else row["human_file_name"] for row in rows],
        dtype=str,
    )
    candidate_origins = np.asarray(
        [row["file_name"] if row["label"] == 1 else row["llm_file_name"] for row in rows],
        dtype=str,
    )
    return EnhancedFeatureCache(
        language="c",
        human=np.zeros((count, 28), dtype=np.float64),
        llm=np.ones((count, 28), dtype=np.float64),
        labels=np.asarray([row["label"] for row in rows], dtype=np.int64),
        source_ids=human_origins,
        human_origin_ids=human_origins,
        candidate_origin_ids=candidate_origins,
        human_code_sha256=np.asarray(
            [hashlib.sha256(str(row["human_src"]).encode()).hexdigest() for row in rows],
            dtype=str,
        ),
        candidate_code_sha256=np.asarray(
            [hashlib.sha256(str(row["llm_src"]).encode()).hexdigest() for row in rows],
            dtype=str,
        ),
        llm_sources=np.asarray([row["paraphrased_by"] for row in rows], dtype=str),
        row_sha256=np.asarray(
            [hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for row in rows],
            dtype=str,
        ),
        human_parse_ok=np.ones(count, dtype=np.bool_),
        llm_parse_ok=np.ones(count, dtype=np.bool_),
        human_backends=np.full(count, "tree-sitter", dtype=str),
        llm_backends=np.full(count, "tree-sitter", dtype=str),
        human_fallback_reasons=np.full(count, "", dtype=str),
        llm_fallback_reasons=np.full(count, "", dtype=str),
    )


def _runner_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    return {
        "f1": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "auroc": 0.5,
        "mcc": 0.0,
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_class_counts": {
            str(label): int((y_train == label).sum()) for label in (0, 1)
        },
        "test_class_counts": {
            str(label): int((y_test == label).sum()) for label in (0, 1)
        },
    }


def _bind_synthetic_gate(
    t3, monkeypatch: pytest.MonkeyPatch, dataset: Path, language: str = "c"
) -> None:
    binding = t3._load_strict_gate_a(t3.DEFAULT_GATE_A_PATH)
    binding["source_jsonl_sha256"] = dict(binding["source_jsonl_sha256"])
    binding["source_jsonl_sha256"][language] = hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(t3, "_load_strict_gate_a", lambda _path: binding)


def test_t3_runner_writes_closed_shared_split_records_and_validated_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("strict T3 synthetic dataset\n", encoding="utf-8")
    cache = _memory_cache(8)
    _bind_synthetic_gate(t3, monkeypatch, dataset)
    monkeypatch.setattr(
        t3, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )
    calls: list[tuple[int, str, int, np.ndarray, np.ndarray]] = []

    def evaluator(x_train, y_train, x_test, y_test, model_name, seed):
        calls.append(
            (
                int(x_train.shape[1]),
                model_name,
                seed,
                x_train.copy(),
                y_train.copy(),
            )
        )
        return _runner_metrics(y_train, y_test)

    monkeypatch.setattr(t3, "evaluate_fold", evaluator)
    output = tmp_path / "t3-run"
    heldout = (t3.LLM_SOURCES[0],)
    first = t3.run_t3(
        output,
        languages=("c",),
        heldout_llms=heldout,
        seeds=(42,),
        n_splits=2,
        methods=t3.T3_METHODS,
        limit_origins=8,
        dataset_paths={"c": dataset},
    )

    assert first["expected"] == 8
    assert first["completed"] == 8
    assert first["skipped"] == 0
    assert sorted(call[:3] for call in calls) == sorted(
        [(20, "mlp", 42), (20, "xgb", 42), (30, "xgb", 42), (112, "xgb", 42)]
        * 2
    )
    first_split = t3.build_t3_splits(
        cache,
        language="c",
        heldout_llm=heldout[0],
        n_splits=2,
        seed=42,
    )[0]
    human_indices = np.asarray(
        [pair.human_positive_row_idx for pair in first_split.train_pairs]
    )
    candidate_indices = np.asarray(
        [pair.candidate_positive_row_idx for pair in first_split.train_pairs]
    )
    expected_labels = np.asarray([pair.label for pair in first_split.train_pairs])
    expected_specs = [
        (10, "concat"),
        (10, "concat"),
        (10, "concat_delta"),
        (28, "full"),
    ]
    from lpcode_v1.representations import build_representation

    for call, (feature_count, representation) in zip(calls[:4], expected_specs):
        expected_matrix = build_representation(
            cache.human[human_indices, :feature_count],
            cache.llm[candidate_indices, :feature_count],
            representation,
        )
        np.testing.assert_array_equal(call[3], expected_matrix)
        np.testing.assert_array_equal(call[4], expected_labels)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["task"] == "task3_unseen_llm"
    assert config["split_protocol"] == t3.T3_SPLIT_PROTOCOL_VERSION
    assert config["pair_protocol"] == t3.PAIR_PROTOCOL_VERSION
    assert config["component_protocol"] == t3.COMPONENT_PROTOCOL_VERSION
    assert config["methods"] == list(t3.T3_METHODS)
    assert config["method_contract"]["best_transition"]["representation"] == "concat_delta"
    assert config["gate_a_binding"]["strict_passed"] is True
    assert len(config["gate_a_binding"]["gate_a_sha256"]) == 64
    assert len(config["cache_content_sha256"]["c"]) == 64
    assert len(config["source_jsonl_sha256"]["c"]) == 64
    assert all(len(value) == 64 for value in config["implementation_contract"].values())

    records = [
        json.loads(line)
        for line in (output / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 8
    assert all(set(record) == t3.T3_FOLD_RECORD_FIELDS for record in records)
    assert len({tuple(t3._record_key(record)) for record in records}) == 8
    assert [tuple(t3._record_key(record)) for record in records] == sorted(
        tuple(t3._record_key(record)) for record in records
    )
    for fold in (0, 1):
        matching = [record for record in records if record["fold"] == fold]
        assert {record["method"] for record in matching} == set(t3.T3_METHODS)
        assert len({record["train_index_sha256"] for record in matching}) == 1
        assert len({record["test_index_sha256"] for record in matching}) == 1
        assert {record["feature_dimensions"] for record in matching} == {20, 30, 112}
        for record in matching:
            assert record["heldout_llm"] == heldout[0]
            assert record["train_llm_sources"] == list(t3.LLM_SOURCES[1:])
            assert record["test_llm_sources"] == [heldout[0]]
            assert record["leakage_count"] == 0
            assert record["endpoint_leakage_count"] == 0
            assert record["content_leakage_count"] == 0
            assert record["negative_component_violation_count"] == 0
            assert record["train_class_counts"]["0"] == record["train_class_counts"]["1"]
            assert record["test_class_counts"]["0"] == record["test_class_counts"]["1"]
            assert record["cache_content_sha256"] == config["cache_content_sha256"]["c"]
            assert record["source_jsonl_sha256"] == config["source_jsonl_sha256"]["c"]
            assert len(record["record_sha256"]) == 64

    before = (output / "folds.jsonl").read_bytes()
    monkeypatch.setattr(
        t3,
        "evaluate_fold",
        lambda *args: pytest.fail("validated T3 resume fitted a completed record"),
    )
    second = t3.run_t3(
        output,
        languages=("c",),
        heldout_llms=heldout,
        seeds=(42,),
        n_splits=2,
        methods=t3.T3_METHODS,
        limit_origins=8,
        dataset_paths={"c": dataset},
    )
    assert second["completed"] == 0
    assert second["skipped"] == 8
    assert (output / "folds.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("f1", float("nan"), "metric"),
        ("schema_version", True, "schema/config"),
        ("train_index_sha256", "0" * 64, "reconstruction"),
        ("train_llm_sources", [LLM_SOURCES[0]], "schema/config"),
        ("cache_content_sha256", "1" * 64, "schema/config"),
    ],
)
def test_t3_resume_rejects_corrupt_or_incomparable_completed_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    from lpcode_v1 import t3

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("strict T3 synthetic dataset\n", encoding="utf-8")
    cache = _memory_cache(8)
    _bind_synthetic_gate(t3, monkeypatch, dataset)
    monkeypatch.setattr(t3, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(
        t3,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _runner_metrics(
            y_train, y_test
        ),
    )
    output = tmp_path / "run"
    kwargs = {
        "languages": ("c",),
        "heldout_llms": (LLM_SOURCES[0],),
        "seeds": (42,),
        "n_splits": 2,
        "methods": ("lpcode_original",),
        "limit_origins": 8,
        "dataset_paths": {"c": dataset},
    }
    t3.run_t3(output, **kwargs)
    ledger = output / "folds.jsonl"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    records[0][field] = value
    if field != "f1":
        records[0]["record_sha256"] = t3._t3_record_sha256(records[0])
    ledger.write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        t3.run_t3(output, **kwargs)


def test_t3_output_lock_refuses_concurrent_writer_and_preserves_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("strict T3 synthetic dataset\n", encoding="utf-8")
    cache = _memory_cache(8)
    _bind_synthetic_gate(t3, monkeypatch, dataset)
    monkeypatch.setattr(t3, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(
        t3,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _runner_metrics(
            y_train, y_test
        ),
    )
    output = tmp_path / "run"
    kwargs = {
        "languages": ("c",),
        "heldout_llms": (LLM_SOURCES[0],),
        "seeds": (42,),
        "n_splits": 2,
        "methods": ("lpcode_original",),
        "limit_origins": 8,
        "dataset_paths": {"c": dataset},
    }
    t3.run_t3(output, **kwargs)
    ledger = output / "folds.jsonl"
    before = ledger.read_bytes()
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_t3_output_lock, args=(str(output), ready, release)
    )
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(ValueError, match="already locked"):
            t3.run_t3(output, **kwargs)
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    assert ledger.read_bytes() == before


def test_t3_bounded_smoke_runs_exact_32_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("strict T3 synthetic dataset\n", encoding="utf-8")
    cache = _memory_cache(12)
    _bind_synthetic_gate(t3, monkeypatch, dataset)
    monkeypatch.setattr(t3, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(
        t3,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _runner_metrics(
            y_train, y_test
        ),
    )
    report = t3.run_t3_smoke(
        tmp_path / "smoke", dataset_paths={"c": dataset}
    )
    config = json.loads((tmp_path / "smoke" / "config.json").read_text(encoding="utf-8"))

    assert report["expected"] == 32
    assert report["completed"] == 32
    assert config["limit_origins"] == t3.T3_SMOKE_ORIGINS
    assert config["heldout_llms"] == list(t3.LLM_SOURCES)
    assert len((tmp_path / "smoke" / "folds.jsonl").read_text(encoding="utf-8").splitlines()) == 32


def test_t3_rejects_evaluator_row_or_class_counts_that_disagree_with_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("strict T3 synthetic dataset\n", encoding="utf-8")
    cache = _memory_cache(8)
    _bind_synthetic_gate(t3, monkeypatch, dataset)
    monkeypatch.setattr(t3, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache)

    def evaluator(_x_train, y_train, _x_test, y_test, *_args):
        metrics = _runner_metrics(y_train, y_test)
        metrics["train_rows"] = int(metrics["train_rows"]) + 2
        metrics["train_class_counts"] = {
            "0": int(metrics["train_class_counts"]["0"]) + 1,
            "1": int(metrics["train_class_counts"]["1"]) + 1,
        }
        return metrics

    monkeypatch.setattr(t3, "evaluate_fold", evaluator)
    with pytest.raises(ValueError, match="evaluator.*split"):
        t3.run_t3(
            tmp_path / "run",
            languages=("c",),
            heldout_llms=(LLM_SOURCES[0],),
            seeds=(42,),
            n_splits=2,
            methods=("lpcode_original",),
            limit_origins=8,
            dataset_paths={"c": dataset},
        )
    assert not (tmp_path / "run" / "folds.jsonl").exists()


def test_t3_rejects_dataset_not_identical_to_strict_gate_selection_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("different from strict Gate A data\n", encoding="utf-8")
    monkeypatch.setattr(
        t3,
        "load_or_build_enhanced_cache",
        lambda *args, **kwargs: pytest.fail(
            "Gate/data mismatch reached cache construction"
        ),
    )
    with pytest.raises(ValueError, match="Gate A dataset hash"):
        t3.run_t3(
            tmp_path / "run",
            languages=("c",),
            heldout_llms=(LLM_SOURCES[0],),
            seeds=(42,),
            n_splits=2,
            methods=("lpcode_original",),
            limit_origins=8,
            dataset_paths={"c": dataset},
        )


def test_t3_dataset_mapping_must_exactly_match_language_axis(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t3

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("data\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset paths.*exactly"):
        t3.run_t3(
            tmp_path / "run",
            languages=("c",),
            heldout_llms=(LLM_SOURCES[0],),
            seeds=(42,),
            n_splits=2,
            methods=("lpcode_original",),
            limit_origins=8,
            dataset_paths={"c": dataset, "cpp": dataset},
        )


def test_t3_positive_bank_splits_are_balanced_deranged_disjoint_and_shared() -> None:
    from lpcode_v1.t3 import build_t3_splits

    cache = _memory_cache()
    heldout = "gemini-pro"
    splits = build_t3_splits(
        cache, language="c", heldout_llm=heldout, n_splits=5, seed=42
    )

    assert len(splits) == 5
    for split in splits:
        assert len(split.train_pairs) == 8 * 3 * 2
        assert len(split.test_pairs) == 2 * 1 * 2
        train_sources = {pair.llm_source for pair in split.train_pairs}
        test_sources = {pair.llm_source for pair in split.test_pairs}
        assert train_sources == set(LLM_SOURCES) - {heldout}
        assert test_sources == {heldout}
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
        assert train_endpoints.isdisjoint(test_endpoints)
        for pairs in (split.train_pairs, split.test_pairs):
            labels = [pair.label for pair in pairs]
            assert labels.count(0) == labels.count(1)
            grouped: dict[tuple[str, str], list] = {}
            for pair in pairs:
                grouped.setdefault((pair.human_origin_id, pair.llm_source), []).append(pair)
                assert cache.labels[pair.human_positive_row_idx] == 1
                assert cache.labels[pair.candidate_positive_row_idx] == 1
                assert cache.llm_sources[pair.human_positive_row_idx] == pair.llm_source
                assert cache.llm_sources[pair.candidate_positive_row_idx] == pair.llm_source
                assert len(pair.pair_sha256) == 64
                if pair.label == 1:
                    assert pair.human_origin_id == pair.candidate_origin_id
                    assert pair.human_positive_row_idx == pair.candidate_positive_row_idx
                else:
                    assert pair.human_origin_id != pair.candidate_origin_id
            assert all(sorted(pair.label for pair in group) == [0, 1] for group in grouped.values())
            for source in {pair.llm_source for pair in pairs}:
                negatives = [pair for pair in pairs if pair.llm_source == source and pair.label == 0]
                assert {pair.human_origin_id for pair in negatives} == {
                    pair.candidate_origin_id for pair in negatives
                }
        assert split.leakage_count == 0
        assert len(split.train_pair_sha256) == 64
        assert len(split.test_pair_sha256) == 64

        per_method = split.pairs_for_methods(
            ("lpcode_original", "xgb_original", "best_transition", "mstf")
        )
        assert set(per_method) == {
            "lpcode_original",
            "xgb_original",
            "best_transition",
            "mstf",
        }
        assert all(pairs[0] is split.train_pairs for pairs in per_method.values())
        assert all(pairs[1] is split.test_pairs for pairs in per_method.values())


def test_t3_split_builder_requires_exactly_one_positive_per_origin_and_llm() -> None:
    from lpcode_v1.t3 import EnhancedFeatureCache, build_t3_splits

    cache = _memory_cache()
    keep = np.ones(len(cache.labels), dtype=bool)
    positive = np.flatnonzero(cache.labels == 1)
    keep[positive[0]] = False
    missing = EnhancedFeatureCache(
        **{
            field: (
                getattr(cache, field)
                if field == "language"
                else getattr(cache, field)[keep]
            )
            for field in cache.__dataclass_fields__
        }
    )
    with pytest.raises(ValueError, match="exactly one positive"):
        build_t3_splits(
            missing, language="c", heldout_llm="gemini-pro", n_splits=5, seed=42
        )

    duplicate_source = cache.llm_sources.copy()
    same_origin_positive = positive[
        cache.human_origin_ids[positive] == cache.human_origin_ids[positive[0]]
    ]
    duplicate_source[same_origin_positive[1]] = duplicate_source[same_origin_positive[0]]
    duplicate = EnhancedFeatureCache(
        **{
            field: (
                duplicate_source
                if field == "llm_sources"
                else getattr(cache, field)
            )
            for field in cache.__dataclass_fields__
        }
    )
    with pytest.raises(ValueError, match="exactly one positive"):
        build_t3_splits(
            duplicate, language="c", heldout_llm="gemini-pro", n_splits=5, seed=42
        )


def test_enhanced_cache_rejects_source_and_package_contract_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    _patch_extractors(monkeypatch)
    dataset = tmp_path / "c.jsonl"
    rows = _rows(2)
    _write_jsonl(dataset, rows)
    enhanced_root = tmp_path / "enhanced"
    official_root = tmp_path / "official"
    t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)

    original_package_contract = t3._package_contract
    monkeypatch.setattr(
        t3,
        "_package_contract",
        lambda: {**original_package_contract(), "tree-sitter": "changed"},
    )
    with pytest.raises(ValueError, match="stale enhanced cache"):
        t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)

    monkeypatch.setattr(t3, "_package_contract", original_package_contract)
    rows[0]["llm_src"] = "int changed(void) { return 999; }"
    _write_jsonl(dataset, rows)
    with pytest.raises(ValueError, match="stale cache"):
        t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)


@pytest.mark.parametrize("tamper", ["row_alignment", "parse_dtype", "raw_corruption"])
def test_enhanced_cache_rejects_corruption_even_when_digest_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    from lpcode_v1 import t3

    _patch_extractors(monkeypatch)
    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    enhanced_root = tmp_path / "enhanced"
    official_root = tmp_path / "official"
    t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)
    archive_path = enhanced_root / "enhanced28-v3" / "c.npz"
    metadata_path = archive_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if tamper == "raw_corruption":
        archive_path.write_bytes(b"not an npz")
    else:
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        if tamper == "row_alignment":
            arrays["source_ids"][0] = "wrong.c"
        else:
            arrays["human_parse_ok"] = arrays["human_parse_ok"].astype(np.int64)
        np.savez(archive_path, **arrays)
        metadata["npz_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt|alignment|parse status|endpoint origins"):
        t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)


@pytest.mark.parametrize("missing", ["npz", "metadata"])
def test_enhanced_cache_rejects_incomplete_atomic_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    from lpcode_v1 import t3

    _patch_extractors(monkeypatch)
    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    enhanced_root = tmp_path / "enhanced"
    official_root = tmp_path / "official"
    t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)
    archive_path = enhanced_root / "enhanced28-v3" / "c.npz"
    (archive_path if missing == "npz" else archive_path.with_suffix(".json")).unlink()

    with pytest.raises(ValueError, match="incomplete enhanced cache"):
        t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)


def test_enhanced_cache_cleans_temporary_archive_after_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    _patch_extractors(monkeypatch)
    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    enhanced_root = tmp_path / "enhanced"

    def interrupted_savez(path, **arrays):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(t3.np, "savez", interrupted_savez)
    with pytest.raises(OSError, match="interrupted"):
        t3.load_or_build_enhanced_cache(
            "c", dataset, enhanced_root, tmp_path / "official"
        )
    version_root = enhanced_root / "enhanced28-v3"
    assert not (version_root / "c.npz").exists()
    assert not (version_root / "c.json").exists()
    assert list(version_root.glob("*.tmp*")) == []


def test_enhanced_cache_refuses_any_output_in_frozen_official_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3
    from lpcode_v1.paths import OFFICIAL_EXPERIMENT_DIR

    _patch_extractors(monkeypatch)
    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    forbidden = OFFICIAL_EXPERIMENT_DIR / "t3-task2-should-not-exist"
    assert not forbidden.exists()
    with pytest.raises(ValueError, match="official experiment tree"):
        t3.load_or_build_enhanced_cache(
            "c", dataset, forbidden, tmp_path / "official"
        )
    assert not forbidden.exists()


def test_enhanced_cache_rejects_unknown_or_missing_llm_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    _patch_extractors(monkeypatch)
    for bad_source in (None, "other-model"):
        rows = _rows(2)
        if bad_source is None:
            rows[0].pop("paraphrased_by")
        else:
            rows[0]["paraphrased_by"] = bad_source
        dataset = tmp_path / f"{bad_source}.jsonl"
        _write_jsonl(dataset, rows)
        with pytest.raises(ValueError, match="invalid LLM source"):
            t3.load_or_build_enhanced_cache(
                "c", dataset, tmp_path / f"enhanced-{bad_source}", tmp_path / f"official-{bad_source}"
            )


def test_t3_pair_splits_are_deterministic_cover_origins_and_specs_are_immutable() -> None:
    from lpcode_v1.t3 import build_t3_splits

    cache = _memory_cache()
    heldout = "deepseek-coder:33b-instruct"
    first = build_t3_splits(cache, "c", heldout, n_splits=5, seed=2024)
    second = build_t3_splits(cache, "c", heldout, n_splits=5, seed=2024)
    for left, right in zip(first, second):
        assert left.train_pairs == right.train_pairs
        assert left.test_pairs == right.test_pairs
        assert left.train_pair_sha256 == right.train_pair_sha256
        assert left.test_pair_sha256 == right.test_pair_sha256
        with pytest.raises((AttributeError, TypeError)):
            left.test_pairs[0].label = 9
    observed = [
        pair.human_origin_id
        for split in first
        for pair in split.test_pairs
        if pair.label == 1
    ]
    expected = sorted(set(cache.human_origin_ids[cache.labels == 1].tolist()))
    assert sorted(observed) == expected
    assert len(observed) == len(set(observed))


def test_fold_local_pairs_remove_legacy_crossed_second_endpoint_leak() -> None:
    from lpcode_v1.t3 import build_t3_splits

    cache = _memory_cache()
    splits = build_t3_splits(cache, "c", "gpt3.5", n_splits=5, seed=42)
    legacy_crosses = 0
    for split in splits:
        train_origins = {pair.human_origin_id for pair in split.train_pairs}
        test_origins = {pair.human_origin_id for pair in split.test_pairs}
        legacy_crosses += sum(
            human in train_origins and candidate in test_origins
            for human, candidate, label in zip(
                cache.human_origin_ids, cache.candidate_origin_ids, cache.labels
            )
            if label == 0
        )
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
        assert train_endpoints.isdisjoint(test_endpoints)
    assert legacy_crosses > 0


def test_t3_split_derives_cache_language_and_rejects_c_as_java() -> None:
    from lpcode_v1.t3 import build_t3_splits

    cache = _memory_cache()
    derived = build_t3_splits(
        cache, heldout_llm="gpt3.5", n_splits=5, seed=42
    )
    assert len(derived) == 5
    with pytest.raises(ValueError, match="language mismatch"):
        build_t3_splits(
            cache,
            language="java",
            heldout_llm="gpt3.5",
            n_splits=5,
            seed=42,
        )


def test_content_components_are_transitive_fold_atomic_and_never_negative_pairs() -> None:
    from lpcode_v1.t3 import PAIR_PROTOCOL_VERSION, _canonical_json, build_t3_splits

    cache = _memory_cache(30)
    human_hashes = cache.human_code_sha256.copy()
    candidate_hashes = cache.candidate_code_sha256.copy()
    origin0, origin1, origin2 = "group-0.c", "group-1.c", "group-2.c"
    shared_human = "a" * 64
    human_hashes[np.isin(cache.human_origin_ids, [origin0, origin1])] = shared_human
    positive = np.flatnonzero(cache.labels == 1)
    row1 = next(i for i in positive if cache.human_origin_ids[i] == origin1)
    row2 = next(i for i in positive if cache.human_origin_ids[i] == origin2)
    shared_candidate = "b" * 64
    candidate_hashes[[row1, row2]] = shared_candidate
    connected = replace(
        cache,
        human_code_sha256=human_hashes,
        candidate_code_sha256=candidate_hashes,
    )

    splits = build_t3_splits(
        connected, heldout_llm="gpt3.5", n_splits=5, seed=3
    )
    component_origins = {origin0, origin1, origin2}
    for split in splits:
        train_origins = {pair.human_origin_id for pair in split.train_pairs}
        test_origins = {pair.human_origin_id for pair in split.test_pairs}
        assert component_origins <= train_origins or component_origins <= test_origins
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
        assert train_hashes.isdisjoint(test_hashes)
        assert all(
            pair.human_component_id != pair.candidate_component_id
            for pair in (*split.train_pairs, *split.test_pairs)
            if pair.label == 0
        )
        assert sum(pair.label == 0 for pair in split.train_pairs) == sum(
            pair.label == 1 for pair in split.train_pairs
        )
        assert sum(pair.label == 0 for pair in split.test_pairs) == sum(
            pair.label == 1 for pair in split.test_pairs
        )
        for side, pairs in (("train", split.train_pairs), ("test", split.test_pairs)):
            for source in {pair.llm_source for pair in pairs}:
                source_pairs = [pair for pair in pairs if pair.llm_source == source]
                component_by_origin = {
                    pair.human_origin_id: pair.human_component_id for pair in source_pairs
                }
                ordered = sorted(
                    component_by_origin,
                    key=lambda origin: hashlib.sha256(
                        _canonical_json(
                            {
                                "protocol_version": PAIR_PROTOCOL_VERSION,
                                "language": connected.language,
                                "seed": 3,
                                "fold": split.fold,
                                "side": side,
                                "llm_source": source,
                                "origin": origin,
                            }
                        )
                    ).hexdigest(),
                )
                valid_offsets = [
                    offset
                    for offset in range(1, len(ordered))
                    if all(
                        component_by_origin[origin]
                        != component_by_origin[ordered[(index + offset) % len(ordered)]]
                        for index, origin in enumerate(ordered)
                    )
                ]
                assert valid_offsets
                expected_mapping = {
                    origin: ordered[(index + min(valid_offsets)) % len(ordered)]
                    for index, origin in enumerate(ordered)
                }
                actual_mapping = {
                    pair.human_origin_id: pair.candidate_origin_id
                    for pair in source_pairs
                    if pair.label == 0
                }
                assert actual_mapping == expected_mapping


def test_content_component_derangement_rejects_impossible_single_component() -> None:
    from lpcode_v1.t3 import build_t3_splits

    cache = _memory_cache(10)
    same = np.full(len(cache.labels), "c" * 64, dtype="<U64")
    connected = replace(
        cache,
        human_code_sha256=same,
        candidate_code_sha256=same.copy(),
    )
    with pytest.raises(ValueError, match="component|derangement|fold"):
        build_t3_splits(
            connected, heldout_llm="gpt3.5", n_splits=5, seed=42
        )


@pytest.mark.parametrize("tamper", ["enhanced_value", "backend", "status"])
def test_enhanced_cache_semantic_digest_rejects_tamper_hidden_by_outer_hash_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    from lpcode_v1 import t3

    _patch_extractors(monkeypatch)
    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    enhanced_root = tmp_path / "enhanced"
    official_root = tmp_path / "official"
    t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)
    archive_path = enhanced_root / "enhanced28-v3" / "c.npz"
    metadata_path = archive_path.with_suffix(".json")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    if tamper == "enhanced_value":
        arrays["human"][0, 10] += 1.0
    elif tamper == "backend":
        arrays["human_backends"][0] = "bad-backend"
    else:
        arrays["human_parse_ok"][0] = False
        arrays["human_backends"] = arrays["human_backends"].astype("<U32")
        arrays["human_backends"][0] = "lexical-fallback"
        arrays["human_fallback_reasons"] = arrays["human_fallback_reasons"].astype("<U32")
        arrays["human_fallback_reasons"][0] = "syntax-error"

    np.savez(archive_path, **arrays)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["npz_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    metadata["parse_failures"] = {
        "human": int((~arrays["human_parse_ok"]).sum()),
        "llm": int((~arrays["llm_parse_ok"]).sum()),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic content digest"):
        t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)


def test_enhanced_cache_rejects_invalid_provenance_even_if_all_digests_are_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3

    _patch_extractors(monkeypatch)
    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    enhanced_root = tmp_path / "enhanced"
    official_root = tmp_path / "official"
    t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)
    archive_path = enhanced_root / "enhanced28-v3" / "c.npz"
    metadata_path = archive_path.with_suffix(".json")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["human_backends"][0] = "bad-backend"
    np.savez(archive_path, **arrays)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["npz_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    metadata["semantic_content_sha256"] = t3._semantic_content_sha256(arrays, "c")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        t3.load_or_build_enhanced_cache("c", dataset, enhanced_root, official_root)


def test_enhanced_cache_serializes_same_language_build_and_extracts_unique_text_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1, t3
    from lpcode_v1.features_enhanced import EnhancedAnalysis

    dataset = tmp_path / "c.jsonl"
    rows = _rows(2)
    _write_jsonl(dataset, rows)
    official_root = tmp_path / "official"
    monkeypatch.setattr(
        t1, "analyze_code", lambda code, language: np.arange(10, dtype=np.float64)
    )
    t1.load_or_build_feature_cache("c", dataset, official_root)
    calls: list[str] = []
    calls_lock = threading.Lock()
    first_started = threading.Event()
    release = threading.Event()

    def slow_analyzer(code: str, language: str) -> EnhancedAnalysis:
        with calls_lock:
            calls.append(code)
            first = len(calls) == 1
        if first:
            first_started.set()
            assert release.wait(10)
        return EnhancedAnalysis(
            np.arange(18, dtype=np.float64), True, "tree-sitter", language
        )

    monkeypatch.setattr(t3, "analyze_enhanced", slow_analyzer)
    enhanced_root = tmp_path / "enhanced"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            enhanced_root,
            official_root,
        )
        assert first_started.wait(10)
        second = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            enhanced_root,
            official_root,
        )
        time.sleep(0.2)
        release.set()
        left = first.result(timeout=10)
        right = second.result(timeout=10)

    unique_texts = {str(row[field]) for row in rows for field in ("human_src", "llm_src")}
    assert len(calls) == len(unique_texts)
    np.testing.assert_array_equal(left.human, right.human)


def test_enhanced_cache_lock_hides_half_published_npz_from_concurrent_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1, t3

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    official_root = tmp_path / "official"
    _patch_extractors(monkeypatch)
    t1.load_or_build_feature_cache("c", dataset, official_root)
    original_writer = t3._atomic_write_json
    metadata_pending = threading.Event()
    release = threading.Event()

    def blocking_metadata_write(path: Path, value: dict) -> None:
        metadata_pending.set()
        assert release.wait(10)
        original_writer(path, value)

    monkeypatch.setattr(t3, "_atomic_write_json", blocking_metadata_write)
    enhanced_root = tmp_path / "enhanced"
    with ThreadPoolExecutor(max_workers=2) as pool:
        publisher = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            enhanced_root,
            official_root,
        )
        assert metadata_pending.wait(10)
        reader = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            enhanced_root,
            official_root,
        )
        time.sleep(0.2)
        assert not reader.done()
        release.set()
        publisher.result(timeout=10)
        reader.result(timeout=10)


def test_different_enhanced_roots_serialize_shared_official_cache_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1, t3
    from lpcode_v1.features_enhanced import EnhancedAnalysis

    dataset = tmp_path / "c.jsonl"
    rows = _rows(2)
    _write_jsonl(dataset, rows)
    calls: list[str] = []
    calls_lock = threading.Lock()
    first_started = threading.Event()
    release = threading.Event()

    def slow_official(code: str, language: str) -> np.ndarray:
        with calls_lock:
            calls.append(code)
            first = len(calls) == 1
        if first:
            first_started.set()
            assert release.wait(10)
        return np.arange(10, dtype=np.float64)

    monkeypatch.setattr(t1, "analyze_code", slow_official)
    monkeypatch.setattr(
        t3,
        "analyze_enhanced",
        lambda code, language: EnhancedAnalysis(
            np.arange(18, dtype=np.float64), True, "tree-sitter", language
        ),
    )
    official_root = tmp_path / "shared-official"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            tmp_path / "enhanced-a",
            official_root,
        )
        assert first_started.wait(10)
        second = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            tmp_path / "enhanced-b",
            official_root,
        )
        time.sleep(0.2)
        release.set()
        first.result(timeout=10)
        second.result(timeout=10)

    assert len(calls) == 2 * len(rows)


def test_shared_official_lock_hides_half_publication_across_enhanced_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1, t3

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _rows(2))
    _patch_extractors(monkeypatch)
    original_writer = t1._atomic_write_json
    pending = threading.Event()
    release = threading.Event()

    def blocking_writer(path: Path, value: dict) -> None:
        pending.set()
        assert release.wait(10)
        original_writer(path, value)

    monkeypatch.setattr(t1, "_atomic_write_json", blocking_writer)
    official_root = tmp_path / "shared-official"
    with ThreadPoolExecutor(max_workers=2) as pool:
        publisher = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            tmp_path / "enhanced-a",
            official_root,
        )
        assert pending.wait(10)
        reader = pool.submit(
            t3.load_or_build_enhanced_cache,
            "c",
            dataset,
            tmp_path / "enhanced-b",
            official_root,
        )
        time.sleep(0.2)
        assert not reader.done()
        release.set()
        publisher.result(timeout=10)
        reader.result(timeout=10)


def test_cache_temporary_paths_are_unique_for_same_target(tmp_path: Path) -> None:
    from lpcode_v1.t3 import _unique_temporary_path

    target = tmp_path / "c.npz"
    first = _unique_temporary_path(target, suffix=".npz")
    second = _unique_temporary_path(target, suffix=".npz")
    assert first != second
    assert first.parent == target.parent == second.parent
    assert first.name.startswith(".c.npz.")
    assert second.name.startswith(".c.npz.")
    assert first.suffix == second.suffix == ".npz"


def test_cache_lock_serializes_independent_processes(tmp_path: Path) -> None:
    lock_path = tmp_path / "enhanced28-v3" / ".c.lock"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    acquired = context.Event()
    holder = context.Process(
        target=_hold_t3_cache_lock, args=(str(lock_path), ready, release)
    )
    observer = context.Process(
        target=_observe_t3_cache_lock, args=(str(lock_path), acquired)
    )
    holder.start()
    try:
        assert ready.wait(15)
        observer.start()
        assert not acquired.wait(0.5)
        release.set()
        assert acquired.wait(15)
    finally:
        release.set()
        holder.join(15)
        if observer.pid is not None:
            observer.join(15)
        for process in (holder, observer):
            if process.is_alive():
                process.terminate()
                process.join()
    assert holder.exitcode == 0
    assert observer.exitcode == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows msvcrt retry path")
def test_windows_cache_lock_retries_with_seek_and_backoff(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    import msvcrt
    from lpcode_v1 import cache_io, t3

    class Handle:
        def __init__(self) -> None:
            self.seeks: list[int] = []

        def seek(self, offset: int) -> None:
            self.seeks.append(offset)

        def fileno(self) -> int:
            return 123

    attempts = 0
    sleeps: list[float] = []

    def locking(fd: int, mode: int, count: int) -> None:
        nonlocal attempts
        attempts += 1
        assert (fd, mode, count) == (123, msvcrt.LK_NBLCK, 1)
        if attempts < 4:
            raise OSError(errno.EACCES, "held")

    monkeypatch.setattr(msvcrt, "locking", locking)
    monkeypatch.setattr(cache_io.time, "sleep", sleeps.append)
    handle = Handle()
    t3._acquire_windows_cache_lock(handle, retry_delay_seconds=0.05)

    assert attempts == 4
    assert handle.seeks == [0, 0, 0, 0]
    assert sleeps == [0.05, 0.05, 0.05]


@pytest.mark.skipif(os.name != "nt", reason="Windows msvcrt timeout regression")
def test_windows_cache_lock_waits_past_native_timeout_for_holder_exit(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "enhanced28-v3" / ".c.lock"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    acquired = context.Event()
    holder = context.Process(
        target=_hold_t3_cache_lock_for, args=(str(lock_path), ready, 11.5)
    )
    observer = context.Process(
        target=_observe_t3_cache_lock, args=(str(lock_path), acquired)
    )
    holder.start()
    try:
        assert ready.wait(15)
        started = time.monotonic()
        observer.start()
        assert not acquired.wait(10.2)
        assert observer.is_alive()
        assert acquired.wait(5)
        assert time.monotonic() - started >= 11.0
    finally:
        holder.join(15)
        if observer.pid is not None:
            observer.join(15)
        for process in (holder, observer):
            if process.is_alive():
                process.terminate()
                process.join()
    assert holder.exitcode == 0
    assert observer.exitcode == 0
