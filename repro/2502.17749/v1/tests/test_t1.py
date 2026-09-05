from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest


def _record(group: str, label: int, index: int) -> dict[str, object]:
    return {
        "file_name": f"llm-{index}.c",
        "human_file_name": f"{group}.c",
        "human_src": f"int human_{index}(void) {{ return {index}; }}",
        "llm_src": f"int llm_{index}(void) {{ return {index + 1}; }}",
        "label": label,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _complete_groups(count: int = 4) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index in range(count):
        group = f"group-{group_index}.c"
        for label in (0, 1):
            index = len(rows)
            rows.append(
                {
                    "file_name": group,
                    "human_file_name": group,
                    "human_src": f"int human_{index}(void) {{ return {index}; }}",
                    "llm_src": f"int llm_{index}(void) {{ return {index + 1}; }}",
                    "label": label,
                }
            )
    return rows


def _metrics() -> dict[str, object]:
    return {
        "f1": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "auroc": 0.5,
        "mcc": 0.0,
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "train_rows": 4,
        "test_rows": 4,
        "train_class_counts": {"0": 2, "1": 2},
        "test_class_counts": {"0": 2, "1": 2},
    }


def _hold_output_lock(path: str, ready, release) -> None:
    from lpcode_v1.t1 import _exclusive_output_lock

    with _exclusive_output_lock(Path(path)):
        ready.set()
        release.wait(15)


def test_feature_cache_builds_strict_float64_contract_and_reuses_without_analyzing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, [_record("group-a", 0, 0), _record("group-a", 1, 1)])
    calls: list[tuple[str, str]] = []

    def analyzer(code: str, language: str) -> np.ndarray:
        calls.append((code, language))
        return np.arange(10, dtype=np.float64)

    monkeypatch.setattr(t1, "analyze_code", analyzer)
    cache = t1.load_or_build_feature_cache("c", dataset, tmp_path / "cache")

    assert cache.human.shape == (2, 10)
    assert cache.llm.shape == (2, 10)
    assert cache.human.dtype == np.float64
    assert cache.llm.dtype == np.float64
    assert cache.labels.dtype.kind in "iu"
    assert cache.labels.tolist() == [0, 1]
    assert cache.source_ids.tolist() == ["group-a.c", "llm-1.c"]
    assert len(calls) == 4
    cache_path = tmp_path / "cache" / "official10-v2" / "c.npz"
    metadata_path = cache_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["cache_version"] == "official10-v2"
    assert isinstance(metadata["feature_contract_sha256"], str)
    assert metadata["source_jsonl_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert metadata["npz_sha256"] == hashlib.sha256(cache_path.read_bytes()).hexdigest()

    monkeypatch.setattr(t1, "analyze_code", lambda *args: pytest.fail("cache reuse analyzed code"))
    reused = t1.load_or_build_feature_cache("c", dataset, tmp_path / "cache")
    np.testing.assert_array_equal(reused.human, cache.human)


def test_feature_cache_v2_binds_official_source_and_preserves_legacy_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1
    from lpcode_v1 import features_official

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, [_record("group-a", 0, 0), _record("group-a", 1, 1)])
    cache_root = tmp_path / "cache"
    legacy = cache_root / "official10-v1" / "c.npz"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-read-only")
    monkeypatch.setattr(
        t1, "analyze_code", lambda code, language: np.arange(10, dtype=np.float64)
    )

    t1.load_or_build_feature_cache("c", dataset, cache_root)

    assert legacy.read_bytes() == b"legacy-read-only"
    archive = cache_root / "official10-v2" / "c.npz"
    metadata = json.loads(archive.with_suffix(".json").read_text(encoding="utf-8"))
    expected_source_hash = hashlib.sha256(
        Path(features_official.__file__).read_bytes()
    ).hexdigest()
    assert archive.is_file()
    assert metadata["cache_version"] == "official10-v2"
    assert metadata["official_feature_source_sha256"] == expected_source_hash
    assert t1._feature_contract()["official_feature_source_sha256"] == expected_source_hash


def test_feature_cache_rejects_dataset_mutation_as_stale(tmp_path: Path) -> None:
    from lpcode_v1.t1 import load_or_build_feature_cache

    dataset = tmp_path / "c.jsonl"
    rows = [_record("group-a", 0, 0), _record("group-a", 1, 1)]
    _write_jsonl(dataset, rows)
    load_or_build_feature_cache("c", dataset, tmp_path / "cache")
    rows[0]["human_src"] = "int modified(void) { return 10; }"
    _write_jsonl(dataset, rows)

    with pytest.raises(ValueError, match="stale cache"):
        load_or_build_feature_cache("c", dataset, tmp_path / "cache")


def test_feature_cache_rejects_wrong_archive_dtype_even_with_matching_digest(tmp_path: Path) -> None:
    from lpcode_v1.t1 import load_or_build_feature_cache

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, [_record("group-a", 0, 0), _record("group-a", 1, 1)])
    cache_root = tmp_path / "cache"
    load_or_build_feature_cache("c", dataset, cache_root)
    archive_path = cache_root / "official10-v2" / "c.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        np.savez(
            archive_path,
            human=archive["human"].astype(np.float32),
            llm=archive["llm"],
            labels=archive["labels"],
            source_ids=archive["source_ids"],
        )
    metadata_path = archive_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["npz_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="feature cache"):
        load_or_build_feature_cache("c", dataset, cache_root)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("rows", 2.0)])
def test_feature_cache_rejects_bool_or_float_metadata_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    from lpcode_v1.t1 import load_or_build_feature_cache

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, [_record("group-a", 0, 0), _record("group-a", 1, 1)])
    cache_root = tmp_path / "cache"
    load_or_build_feature_cache("c", dataset, cache_root)
    metadata_path = cache_root / "official10-v2" / "c.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        load_or_build_feature_cache("c", dataset, cache_root)


@pytest.mark.parametrize("orphan", ["npz", "metadata"])
def test_feature_cache_rejects_interrupted_npz_metadata_pair(tmp_path: Path, orphan: str) -> None:
    from lpcode_v1.t1 import load_or_build_feature_cache

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, [_record("group-a", 0, 0), _record("group-a", 1, 1)])
    cache_root = tmp_path / "cache"
    load_or_build_feature_cache("c", dataset, cache_root)
    cache_path = cache_root / "official10-v2" / "c.npz"
    target = cache_path.with_suffix(".json") if orphan == "npz" else cache_path
    target.unlink()

    with pytest.raises(ValueError, match="incomplete feature cache"):
        load_or_build_feature_cache("c", dataset, cache_root)


def test_feature_cache_rejects_dataset_changed_during_feature_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    rows = [_record("group-a", 0, 0), _record("group-a", 1, 1)]
    _write_jsonl(dataset, rows)
    calls = 0

    def changing_analyzer(code: str, language: str) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            rows[0]["human_src"] = "int changed(void) { return 11; }"
            _write_jsonl(dataset, rows)
        return np.arange(10, dtype=np.float64)

    monkeypatch.setattr(t1, "analyze_code", changing_analyzer)
    with pytest.raises(ValueError, match="dataset changed"):
        t1.load_or_build_feature_cache("c", dataset, tmp_path / "cache")


def test_runner_rejects_dataset_changed_after_config_before_cache_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    rows = _complete_groups()
    _write_jsonl(dataset, rows)
    original_loader = t1.load_or_build_feature_cache

    def changing_loader(language, dataset_path, cache_root):
        cache = original_loader(language, dataset_path, cache_root)
        rows[0]["human_src"] = "int changed_after_config(void) { return 12; }"
        _write_jsonl(dataset, rows)
        return cache

    monkeypatch.setattr(t1, "load_or_build_feature_cache", changing_loader)
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: pytest.fail("changed dataset reached evaluator"))
    with pytest.raises(ValueError, match="dataset changed since run config"):
        t1.run_t1(tmp_path / "run", ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})


def test_runner_reuses_split_indices_and_resume_never_re_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    evaluations: list[tuple[str, int]] = []

    def evaluator(x_train, y_train, x_test, y_test, model_name, seed):
        evaluations.append((model_name, seed))
        assert x_train.shape[0] == 4 and x_test.shape[0] == 4
        return _metrics()

    monkeypatch.setattr(t1, "evaluate_fold", evaluator)
    first = t1.run_t1(
        output_root=tmp_path / "run",
        languages=("c",),
        seeds=(42,),
        n_splits=2,
        representations=("concat", "delta", "concat_delta", "full"),
        models=("mlp", "xgb"),
        dataset_paths={"c": dataset},
    )
    assert first["completed"] == 16
    assert first["skipped"] == 0
    assert len(evaluations) == 16
    fold_records = [json.loads(line) for line in (tmp_path / "run" / "folds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(fold_records) == 16
    for fold in (0, 1):
        matching = [record for record in fold_records if record["fold"] == fold]
        assert len(matching) == 8
        assert len({record["train_index_sha256"] for record in matching}) == 1
        assert len({record["test_index_sha256"] for record in matching}) == 1
        assert {record["feature_dimensions"] for record in matching} == {10, 20, 30, 40}
        assert all(record["leakage_count"] == 0 for record in matching)
    assert all(record["schema_version"] == 1 for record in fold_records)

    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: pytest.fail("resume evaluated completed fold"))
    second = t1.run_t1(
        output_root=tmp_path / "run",
        languages=("c",),
        seeds=(42,),
        n_splits=2,
        representations=("concat", "delta", "concat_delta", "full"),
        models=("mlp", "xgb"),
        dataset_paths={"c": dataset},
    )
    assert second["completed"] == 0
    assert second["skipped"] == 16
    assert len((tmp_path / "run" / "folds.jsonl").read_text(encoding="utf-8").splitlines()) == 16


@pytest.mark.parametrize("bad_contents", ["not json\n", '{"schema_version": 1}\n'])
def test_runner_rejects_malformed_or_wrong_fold_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_contents: str
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    output = tmp_path / "run"
    output.mkdir()
    (output / "folds.jsonl").write_text(bad_contents, encoding="utf-8")
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: _metrics())
    with pytest.raises(ValueError, match="fold"):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})


def test_runner_rejects_config_mismatch_and_duplicate_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: _metrics())
    output = tmp_path / "run"
    t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    with pytest.raises(ValueError, match="config mismatch"):
        t1.run_t1(output, ("c",), (123,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    folds = output / "folds.jsonl"
    folds.write_text(folds.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})


def test_runner_revalidates_fold_file_immediately_before_atomic_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    output = tmp_path / "run"

    def evaluator(*args):
        (output / "folds.jsonl").write_text("not json\n", encoding="utf-8")
        return _metrics()

    monkeypatch.setattr(t1, "evaluate_fold", evaluator)
    with pytest.raises(ValueError, match="malformed fold"):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})


def test_runner_rejects_completed_record_with_wrong_recomputed_split_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    output = tmp_path / "run"
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: _metrics())
    t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    folds = output / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    records[0]["train_index_sha256"] = "0" * 64
    folds.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="split hash"):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("leakage_count", False), ("fold", 0.0), ("train_rows", 4.0)],
)
def test_runner_rejects_bool_or_float_fold_record_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    output = tmp_path / "run"
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: _metrics())
    t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    folds = output / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    records[0][field] = value
    folds.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})


@pytest.mark.parametrize("field,value", [("schema_version", True), ("n_splits", 2.0), ("seeds", [42.0])])
def test_run_config_validator_rejects_bool_or_float_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    config = t1._build_config(("c",), (42,), 2, ("concat",), ("mlp",), None, {"c": dataset})
    config[field] = value

    with pytest.raises(ValueError, match="config"):
        t1._validate_run_config(config)


def test_runner_rejects_out_of_range_evaluator_metrics_before_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    metrics = _metrics()
    metrics["f1"] = 1.1
    output = tmp_path / "run"
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: metrics)

    with pytest.raises(ValueError, match="metric range"):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    assert not (output / "folds.jsonl").exists()


def test_runner_rejects_extra_evaluator_result_field_before_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    metrics = _metrics()
    metrics["unexpected"] = float("nan")
    output = tmp_path / "run"
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: metrics)

    with pytest.raises(ValueError, match="evaluator result schema"):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    assert not (output / "folds.jsonl").exists()


def test_runner_rejects_extra_completed_record_field_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    output = tmp_path / "run"
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: _metrics())
    t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    folds = output / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    records[0]["unexpected"] = float("nan")
    folds.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="malformed fold record"):
        t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})


def test_runner_rejects_second_process_holding_output_lock_without_losing_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1

    dataset = tmp_path / "c.jsonl"
    _write_jsonl(dataset, _complete_groups())
    output = tmp_path / "run"
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: _metrics())
    t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    folds = output / "folds.jsonl"
    completed_contents = folds.read_text(encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_output_lock, args=(str(output), ready, release))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(ValueError, match="already locked"):
            t1.run_t1(output, ("c",), (42,), 2, ("concat",), ("mlp",), dataset_paths={"c": dataset})
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    assert folds.read_text(encoding="utf-8") == completed_contents


def test_real_smoke_runs_all_models_and_writes_16_clean_records(tmp_path: Path) -> None:
    from lpcode_v1.t1 import run_smoke

    report = run_smoke(tmp_path / "smoke")
    records = [json.loads(line) for line in (tmp_path / "smoke" / "folds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert report["completed"] == 16
    assert len(records) == 16
    assert len({(item["language"], item["representation"], item["model"], item["seed"], item["fold"]) for item in records}) == 16
    assert {item["model"] for item in records} == {"mlp", "xgb"}
    assert all(item["leakage_count"] == 0 for item in records)
