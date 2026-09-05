from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _positive_bank(origin_count: int = 12):
    from lpcode_v1.t3 import EnhancedFeatureCache, LLM_SOURCES

    origins: list[str] = []
    sources: list[str] = []
    for origin_index in range(origin_count):
        for source in LLM_SOURCES:
            origins.append(f"origin-{origin_index}.c")
            sources.append(source)
    rows = len(origins)
    return EnhancedFeatureCache(
        language="c",
        human=np.arange(rows * 28, dtype=np.float64).reshape(rows, 28),
        llm=np.arange(rows * 28, rows * 56, dtype=np.float64).reshape(rows, 28),
        labels=np.ones(rows, dtype=np.int64),
        source_ids=np.asarray(origins, dtype=str),
        human_origin_ids=np.asarray(origins, dtype=str),
        candidate_origin_ids=np.asarray(origins, dtype=str),
        human_code_sha256=np.asarray(
            [_digest(f"human:{origin}") for origin in origins], dtype=str
        ),
        candidate_code_sha256=np.asarray(
            [_digest(f"candidate:{origin}") for origin in origins], dtype=str
        ),
        llm_sources=np.asarray(sources, dtype=str),
        row_sha256=np.asarray(
            [_digest(f"row:{origin}:{source}") for origin, source in zip(origins, sources)],
            dtype=str,
        ),
        human_parse_ok=np.ones(rows, dtype=np.bool_),
        llm_parse_ok=np.ones(rows, dtype=np.bool_),
        human_backends=np.full(rows, "tree-sitter", dtype=str),
        llm_backends=np.full(rows, "tree-sitter", dtype=str),
        human_fallback_reasons=np.full(rows, "", dtype=str),
        llm_fallback_reasons=np.full(rows, "", dtype=str),
    )


def _metrics_for(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
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


def _dataset(path: Path, value: str = "strict-origin-v2") -> Path:
    path.write_text(value + "\n", encoding="utf-8")
    return path


def _hold_strict_output_lock(path: str, ready, release) -> None:
    from lpcode_v1.t1 import _exclusive_output_lock

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        ready.set()
        release.wait(15)


def test_strict_config_is_schema_v2_and_binds_every_reproducibility_contract(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t1_strict
    from lpcode_v1.t3 import COMPONENT_PROTOCOL_VERSION, PAIR_PROTOCOL_VERSION

    dataset = _dataset(tmp_path / "c.jsonl")
    config = t1_strict._build_config(
        ("c",), (42,), 2, ("concat",), ("mlp",), 8, {"c": dataset}
    )

    assert config["schema_version"] == 2
    assert config["task"] == "task1_strict_origins"
    assert config["pair_protocol"] == PAIR_PROTOCOL_VERSION
    assert config["component_protocol"] == COMPONENT_PROTOCOL_VERSION
    assert config["feature_contract"]["selected_columns"] == [0, 10]
    assert config["feature_contract"]["selected_feature_count"] == 10
    assert config["feature_contract"]["official_cache_version"] == "official10-v2"
    assert config["feature_contract"]["official_feature_contract"][
        "feature_count"
    ] == 10
    assert set(config["implementation_contract"]) == {
        "runner_source_sha256",
        "experiment_source_sha256",
        "representations_source_sha256",
        "pair_builder_source_sha256",
        "data_normalization_source_sha256",
        "official_cache_runner_source_sha256",
    }
    assert all(
        len(value) == 64 for value in config["implementation_contract"].values()
    )
    assert config["source_jsonl_sha256"]["c"] == hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()
    assert {"python", "numpy", "scikit_learn", "xgboost"} <= set(
        config["package_versions"]
    )
    assert len(config["config_id"]) == 64
    assert t1_strict._validate_run_config(config) is config

    mutated = json.loads(json.dumps(config))
    mutated["pair_protocol"] = "wrong"
    with pytest.raises(ValueError, match="config"):
        t1_strict._validate_run_config(mutated)

    mutated = json.loads(json.dumps(config))
    mutated["feature_contract"]["official_cache_version"] = "wrong"
    with pytest.raises(ValueError, match="config"):
        t1_strict._validate_run_config(mutated)


def test_strict_default_axes_are_the_exact_480_matrix() -> None:
    from lpcode_v1 import t1_strict

    assert t1_strict.LANGUAGES == ("c", "cpp", "java", "py")
    assert t1_strict.DEFAULT_SEEDS == (42, 123, 2024)
    assert t1_strict.DEFAULT_REPRESENTATIONS == (
        "concat",
        "delta",
        "concat_delta",
        "full",
    )
    assert t1_strict.DEFAULT_MODELS == ("mlp", "xgb")
    assert t1_strict._evaluation_count(
        t1_strict.LANGUAGES,
        t1_strict.DEFAULT_SEEDS,
        5,
        t1_strict.DEFAULT_REPRESENTATIONS,
        t1_strict.DEFAULT_MODELS,
    ) == 480


def test_strict_runner_uses_official_slice_reuses_pair_hashes_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict
    from lpcode_v1.t3 import LLM_SOURCES, build_t1_pair_splits

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    expected_splits = build_t1_pair_splits(cache, n_splits=2, seed=42)
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]] = []

    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )

    def evaluator(x_train, y_train, x_test, y_test, model_name, seed):
        calls.append((x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), model_name, seed))
        return _metrics_for(y_train, y_test)

    monkeypatch.setattr(t1_strict, "evaluate_fold", evaluator)
    output = tmp_path / "run"
    first = t1_strict.run_t1_strict(
        output,
        languages=("c",),
        seeds=(42,),
        n_splits=2,
        representations=("concat", "delta", "concat_delta", "full"),
        models=("mlp", "xgb"),
        dataset_paths={"c": dataset},
    )

    assert first["schema_version"] == 2
    assert first["expected"] == 16
    assert first["completed"] == 16
    assert first["skipped"] == 0
    assert len(calls) == 16
    assert {call[0].shape[1] for call in calls} == {10, 20, 30, 40}
    first_split = expected_splits[0]
    first_concat = next(call for call in calls if call[0].shape[1] == 20)
    human_indices = [pair.human_positive_row_idx for pair in first_split.train_pairs]
    candidate_indices = [
        pair.candidate_positive_row_idx for pair in first_split.train_pairs
    ]
    np.testing.assert_array_equal(first_concat[0][:, :10], cache.human[human_indices, :10])
    np.testing.assert_array_equal(first_concat[0][:, 10:], cache.llm[candidate_indices, :10])

    records = [
        json.loads(line)
        for line in (output / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 16
    for fold, split in enumerate(expected_splits):
        matching = [record for record in records if record["fold"] == fold]
        assert len(matching) == 8
        assert {record["train_index_sha256"] for record in matching} == {
            split.train_pair_sha256
        }
        assert {record["test_index_sha256"] for record in matching} == {
            split.test_pair_sha256
        }
        assert {record["feature_dimensions"] for record in matching} == {10, 20, 30, 40}
        assert all(record["schema_version"] == 2 for record in matching)
        assert all(record["endpoint_leakage_count"] == 0 for record in matching)
        assert all(record["content_leakage_count"] == 0 for record in matching)
        assert all(record["negative_component_violation_count"] == 0 for record in matching)
        for record in matching:
            assert set(record["train_llm_label_counts"]) == set(LLM_SOURCES)
            assert set(record["test_llm_label_counts"]) == set(LLM_SOURCES)
            assert all(
                counts["0"] == counts["1"] > 0
                for counts in record["train_llm_label_counts"].values()
            )
            assert all(
                counts["0"] == counts["1"] > 0
                for counts in record["test_llm_label_counts"].values()
            )

    monkeypatch.setattr(
        t1_strict,
        "evaluate_fold",
        lambda *args: pytest.fail("resume re-evaluated a completed strict fold"),
    )
    second = t1_strict.run_t1_strict(
        output,
        languages=("c",),
        seeds=(42,),
        n_splits=2,
        representations=("concat", "delta", "concat_delta", "full"),
        models=("mlp", "xgb"),
        dataset_paths={"c": dataset},
    )
    assert second["completed"] == 0
    assert second["skipped"] == 16


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("train_index_sha256", "0" * 64, "pair digest"),
        ("endpoint_leakage_count", 1, "split metadata"),
        ("content_leakage_count", False, "schema"),
        ("f1", 1.1, "metric range"),
    ],
)
def test_strict_resume_rejects_corrupt_closed_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )
    monkeypatch.setattr(
        t1_strict,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _metrics_for(
            y_train, y_test
        ),
    )
    output = tmp_path / "run"
    t1_strict.run_t1_strict(
        output,
        ("c",),
        (42,),
        2,
        ("concat",),
        ("mlp",),
        dataset_paths={"c": dataset},
    )
    folds = output / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    records[0][field] = value
    if field in {"train_index_sha256", "endpoint_leakage_count"}:
        records[0]["record_sha256"] = t1_strict._record_sha256(records[0])
    folds.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        t1_strict.run_t1_strict(
            output,
            ("c",),
            (42,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )


def test_strict_runner_locks_and_preserves_completed_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )
    monkeypatch.setattr(
        t1_strict,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _metrics_for(
            y_train, y_test
        ),
    )
    output = tmp_path / "run"
    t1_strict.run_t1_strict(
        output,
        ("c",),
        (42,),
        2,
        ("concat",),
        ("mlp",),
        dataset_paths={"c": dataset},
    )
    folds = output / "folds.jsonl"
    before = folds.read_bytes()
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_strict_output_lock, args=(str(output), ready, release)
    )
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(ValueError, match="already locked"):
            t1_strict.run_t1_strict(
                output,
                ("c",),
                (42,),
                2,
                ("concat",),
                ("mlp",),
                dataset_paths={"c": dataset},
            )
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    assert folds.read_bytes() == before


def test_strict_bounded_smoke_cli_entry_runs_16_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(12)
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )
    monkeypatch.setattr(
        t1_strict,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _metrics_for(
            y_train, y_test
        ),
    )
    report = t1_strict.run_smoke(
        tmp_path / "smoke", dataset_paths={"c": dataset}
    )
    config = json.loads((tmp_path / "smoke" / "config.json").read_text(encoding="utf-8"))

    assert report["expected"] == 16
    assert report["completed"] == 16
    assert config["limit_origins"] == t1_strict.SMOKE_ORIGINS
    assert len((tmp_path / "smoke" / "folds.jsonl").read_text(encoding="utf-8").splitlines()) == 16


def test_strict_runner_rejects_config_drift_and_extra_record_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )
    monkeypatch.setattr(
        t1_strict,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _metrics_for(
            y_train, y_test
        ),
    )
    output = tmp_path / "run"
    t1_strict.run_t1_strict(
        output,
        ("c",),
        (42,),
        2,
        ("concat",),
        ("mlp",),
        dataset_paths={"c": dataset},
    )
    with pytest.raises(ValueError, match="config mismatch"):
        t1_strict.run_t1_strict(
            output,
            ("c",),
            (123,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )

    folds = output / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    records[0]["unexpected"] = "not-closed"
    folds.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        t1_strict.run_t1_strict(
            output,
            ("c",),
            (42,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )


def test_strict_runner_revalidates_ledger_immediately_before_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    output = tmp_path / "run"
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )

    def evaluator(_x_train, y_train, _x_test, y_test, *_args):
        (output / "folds.jsonl").write_text("not json\n", encoding="utf-8")
        return _metrics_for(y_train, y_test)

    monkeypatch.setattr(t1_strict, "evaluate_fold", evaluator)
    with pytest.raises(ValueError, match="malformed strict-origin fold"):
        t1_strict.run_t1_strict(
            output,
            ("c",),
            (42,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )
    assert (output / "folds.jsonl").read_text(encoding="utf-8") == "not json\n"


def test_strict_runner_rejects_dataset_change_after_cache_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)

    def changing_loader(*args, **kwargs):
        _dataset(dataset, "changed-after-cache")
        return cache

    monkeypatch.setattr(t1_strict, "load_or_build_enhanced_cache", changing_loader)
    monkeypatch.setattr(
        t1_strict,
        "evaluate_fold",
        lambda *args: pytest.fail("changed dataset reached evaluator"),
    )
    with pytest.raises(ValueError, match="dataset changed"):
        t1_strict.run_t1_strict(
            tmp_path / "run",
            ("c",),
            (42,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )


def test_strict_runner_rejects_bad_evaluator_result_before_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )

    def evaluator(_x_train, y_train, _x_test, y_test, *_args):
        metrics = _metrics_for(y_train, y_test)
        metrics["auroc"] = float("nan")
        return metrics

    monkeypatch.setattr(t1_strict, "evaluate_fold", evaluator)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="metric"):
        t1_strict.run_t1_strict(
            output,
            ("c",),
            (42,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )
    assert not (output / "folds.jsonl").exists()


def test_strict_runner_rejects_dataset_mutation_during_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )

    def evaluator(_x_train, y_train, _x_test, y_test, *_args):
        _dataset(dataset, "changed-during-evaluation")
        return _metrics_for(y_train, y_test)

    monkeypatch.setattr(t1_strict, "evaluate_fold", evaluator)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="dataset changed"):
        t1_strict.run_t1_strict(
            output,
            ("c",),
            (42,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )
    assert not (output / "folds.jsonl").exists()


def test_strict_resume_rejects_in_range_metric_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict

    dataset = _dataset(tmp_path / "c.jsonl")
    cache = _positive_bank(8)
    monkeypatch.setattr(
        t1_strict, "load_or_build_enhanced_cache", lambda *args, **kwargs: cache
    )
    monkeypatch.setattr(
        t1_strict,
        "evaluate_fold",
        lambda _x_train, y_train, _x_test, y_test, *_args: _metrics_for(
            y_train, y_test
        ),
    )
    output = tmp_path / "run"
    t1_strict.run_t1_strict(
        output,
        ("c",),
        (42,),
        2,
        ("concat",),
        ("mlp",),
        dataset_paths={"c": dataset},
    )
    folds = output / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    records[0]["f1"] = 0.6
    folds.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="record digest"):
        t1_strict.run_t1_strict(
            output,
            ("c",),
            (42,),
            2,
            ("concat",),
            ("mlp",),
            dataset_paths={"c": dataset},
        )


@pytest.mark.parametrize(
    ("languages", "seeds", "n_splits", "representations", "models", "limit"),
    [
        (("c", "c"), (42,), 2, ("concat",), ("mlp",), None),
        (("c",), (True,), 2, ("concat",), ("mlp",), None),
        (("c",), (42,), 2.0, ("concat",), ("mlp",), None),
        (("c",), (42,), 2, ("concat", "concat"), ("mlp",), None),
        (("c",), (42,), 2, ("concat",), ("bad",), None),
        (("c",), (42,), 2, ("concat",), ("mlp",), 3),
    ],
)
def test_strict_axes_reject_ambiguous_or_unbounded_variants(
    languages, seeds, n_splits, representations, models, limit
) -> None:
    from lpcode_v1 import t1_strict

    with pytest.raises(ValueError):
        t1_strict._validate_axes(
            languages, seeds, n_splits, representations, models, limit
        )


def test_strict_cli_forwards_explicit_cache_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from lpcode_v1 import t1_strict

    output = tmp_path / "output"
    enhanced = tmp_path / "enhanced"
    official = tmp_path / "official"
    received: list[tuple[Path, Path, Path]] = []

    def smoke(output_root, dataset_paths=None, cache_root=None, official_cache_root=None):
        received.append((output_root, cache_root, official_cache_root))
        return {"completed": 16}

    monkeypatch.setattr(t1_strict, "run_smoke", smoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "t1_strict",
            "--smoke",
            "--output-root",
            str(output),
            "--cache-root",
            str(enhanced),
            "--official-cache-root",
            str(official),
        ],
    )
    t1_strict.main()

    assert received == [(output, enhanced, official)]
    assert json.loads(capsys.readouterr().out) == {"completed": 16}


def test_t1_pair_splits_use_all_llms_balanced_disjoint_and_shared() -> None:
    from lpcode_v1.t3 import LLM_SOURCES, build_t1_pair_splits

    cache = _positive_bank()
    splits = build_t1_pair_splits(cache, language="c", n_splits=5, seed=42)

    assert len(splits) == 5
    for split in splits:
        assert split.leakage_count == 0
        assert len(split.train_pair_sha256) == 64
        assert len(split.test_pair_sha256) == 64
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
        for pairs in (split.train_pairs, split.test_pairs):
            assert {pair.llm_source for pair in pairs} == set(LLM_SOURCES)
            for source in LLM_SOURCES:
                source_pairs = [pair for pair in pairs if pair.llm_source == source]
                assert sum(pair.label == 0 for pair in source_pairs) == sum(
                    pair.label == 1 for pair in source_pairs
                )
                assert {
                    pair.human_origin_id for pair in source_pairs if pair.label == 0
                } == {
                    pair.candidate_origin_id for pair in source_pairs if pair.label == 0
                }
            assert all(
                pair.human_component_id != pair.candidate_component_id
                for pair in pairs
                if pair.label == 0
            )

        bindings = split.pairs_for_methods(("concat", "delta", "full"))
        assert all(value[0] is split.train_pairs for value in bindings.values())
        assert all(value[1] is split.test_pairs for value in bindings.values())


@pytest.mark.parametrize("mode", ("random", "hard"))
def test_negative_pair_modes_preserve_positive_pairs_balance_and_isolation(
    mode: str,
) -> None:
    """Alternative negative sampling changes only negative pairs under a fixed split."""

    from lpcode_v1.t3 import build_t1_pair_splits

    cache = _positive_bank()
    current = build_t1_pair_splits(cache, language="c", n_splits=5, seed=42)
    alternative = build_t1_pair_splits(
        cache, language="c", n_splits=5, seed=42, negative_pair_mode=mode
    )

    assert len(current) == len(alternative) == 5
    for baseline_split, split in zip(current, alternative):
        for baseline_pairs, pairs in (
            (baseline_split.train_pairs, split.train_pairs),
            (baseline_split.test_pairs, split.test_pairs),
        ):
            baseline_positive = [pair.pair_sha256 for pair in baseline_pairs if pair.label == 1]
            positive = [pair.pair_sha256 for pair in pairs if pair.label == 1]
            negative = [pair for pair in pairs if pair.label == 0]
            assert positive == baseline_positive
            assert len(negative) == len(positive)
            assert len({pair.pair_sha256 for pair in negative}) == len(negative)
            assert all(
                pair.human_origin_id != pair.candidate_origin_id
                and pair.human_component_id != pair.candidate_component_id
                for pair in negative
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


def test_t1_pair_splits_are_deterministic_immutable_and_component_atomic() -> None:
    from lpcode_v1.t3 import LLM_SOURCES, build_t1_pair_splits

    cache = _positive_bank(30)
    human_hashes = cache.human_code_sha256.copy()
    candidate_hashes = cache.candidate_code_sha256.copy()
    origins = ("origin-0.c", "origin-1.c")
    human_hashes[np.isin(cache.human_origin_ids, origins)] = "a" * 64
    connected = replace(
        cache,
        human_code_sha256=human_hashes,
        candidate_code_sha256=candidate_hashes,
    )

    first = build_t1_pair_splits(connected, n_splits=5, seed=3)
    second = build_t1_pair_splits(connected, n_splits=5, seed=3)

    assert first == second
    for left, right in zip(first, second):
        assert left.train_pair_sha256 == right.train_pair_sha256
        assert left.test_pair_sha256 == right.test_pair_sha256
        train_origins = {pair.human_origin_id for pair in left.train_pairs}
        test_origins = {pair.human_origin_id for pair in left.test_pairs}
        assert set(origins) <= train_origins or set(origins) <= test_origins
        for pairs in (left.train_pairs, left.test_pairs):
            for source in LLM_SOURCES:
                source_pairs = [pair for pair in pairs if pair.llm_source == source]
                assert len({pair.pair_sha256 for pair in source_pairs}) == len(source_pairs)
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            left.train_pairs[0].label = 9
