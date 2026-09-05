from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


LLMS = (
    "gpt3.5",
    "gemini-pro",
    "wizardcoder:33b-v1.1",
    "deepseek-coder:33b-instruct",
)


def _memory_cache(group_count: int = 8):
    from lpcode_v1 import t3

    rows = []
    for origin_index in range(group_count):
        for source_index, source in enumerate(LLMS):
            rows.append(
                {
                    "file_name": f"origin-{origin_index}.c",
                    "human_src": f"int h{origin_index}(void) {{ return {origin_index}; }}",
                    "llm_src": f"int c{origin_index}_{source_index}(void) {{ return {source_index}; }}",
                    "paraphrased_by": source,
                    "label": 1,
                }
            )
    count = len(rows)
    origins = np.asarray([row["file_name"] for row in rows], dtype=str)
    return t3.EnhancedFeatureCache(
        language="c",
        human=np.zeros((count, 28), dtype=np.float64),
        llm=np.ones((count, 28), dtype=np.float64),
        labels=np.ones(count, dtype=np.int64),
        source_ids=origins,
        human_origin_ids=origins,
        candidate_origin_ids=origins,
        human_code_sha256=np.asarray(
            [hashlib.sha256(str(row["human_src"]).encode()).hexdigest() for row in rows]
        ),
        candidate_code_sha256=np.asarray(
            [hashlib.sha256(str(row["llm_src"]).encode()).hexdigest() for row in rows]
        ),
        llm_sources=np.asarray([row["paraphrased_by"] for row in rows], dtype=str),
        row_sha256=np.asarray([t3._row_sha256(row) for row in rows], dtype=str),
        human_parse_ok=np.ones(count, dtype=np.bool_),
        llm_parse_ok=np.ones(count, dtype=np.bool_),
        human_backends=np.full(count, "tree-sitter", dtype=str),
        llm_backends=np.full(count, "tree-sitter", dtype=str),
        human_fallback_reasons=np.full(count, "", dtype=str),
        llm_fallback_reasons=np.full(count, "", dtype=str),
    )


def _attack_cache(clean):
    from lpcode_v1 import t4

    count = len(clean.row_sha256)
    shape = (count, len(t4.ATTACKS))
    return t4.AttackFeatureCache(
        language="c",
        row_sha256=clean.row_sha256.copy(),
        features=np.full((count, len(t4.ATTACKS), 28), 2.0),
        success=np.ones(shape, dtype=np.bool_),
        output_sha256=np.full(shape, "e" * 64, dtype=str),
        changed=np.ones(shape, dtype=np.bool_),
        transform_count=np.ones(shape, dtype=np.int64),
        parse_ok_before=np.ones(shape, dtype=np.bool_),
        parse_ok_after=np.ones(shape, dtype=np.bool_),
        backend_before=np.full(shape, "tree-sitter", dtype=str),
        backend_after=np.full(shape, "tree-sitter", dtype=str),
        failure_reason=np.full(shape, "", dtype=str),
        semantic_content_sha256="f" * 64,
    )


def test_t4_closed_contract_and_exact_matrix() -> None:
    from lpcode_v1 import t3, t4

    assert t4.CONDITIONS == ("clean", *t4.ATTACKS)
    assert t4.METHODS == t3.T3_METHODS
    assert t4._evaluation_count(t4.LANGUAGES, t4.METHODS, t4.DEFAULT_SEEDS, 5) == 1440
    binding = t4._load_strict_gate_b(t4.DEFAULT_GATE_B_PATH)
    assert binding["strict_passed"] is True
    assert binding["authorizes_t4"] is True
    assert binding["holdouts_won"] == 4
    assert binding["overall_macro_mean_delta_f1"] >= 0.03
    assert len(binding["gate_b_sha256"]) == 64
    assert len(binding["manifest_sha256"]) == 64


def test_attack_cache_lookup_is_hash_bound_and_rejects_failed_rows() -> None:
    from lpcode_v1 import t4

    row_hashes = np.asarray(["a" * 64, "b" * 64], dtype=str)
    features = np.arange(2 * 5 * 28, dtype=np.float64).reshape(2, 5, 28)
    success = np.ones((2, 5), dtype=np.bool_)
    success[1, 2] = False
    cache = t4.AttackFeatureCache(
        language="c",
        row_sha256=row_hashes,
        features=features,
        success=success,
        output_sha256=np.full((2, 5), "c" * 64, dtype=str),
        changed=np.ones((2, 5), dtype=np.bool_),
        transform_count=np.ones((2, 5), dtype=np.int64),
        parse_ok_before=np.ones((2, 5), dtype=np.bool_),
        parse_ok_after=np.ones((2, 5), dtype=np.bool_),
        backend_before=np.full((2, 5), "tree-sitter", dtype=str),
        backend_after=np.full((2, 5), "tree-sitter", dtype=str),
        failure_reason=np.full((2, 5), "", dtype=str),
        semantic_content_sha256="d" * 64,
    )

    matrix, mask, audit = t4._attack_rows(cache, row_hashes, "format_normalization")
    np.testing.assert_array_equal(matrix, features[:, 2, :])
    np.testing.assert_array_equal(mask, np.asarray([True, False]))
    assert audit["attempted"] == 2
    assert audit["failures"] == 1
    with pytest.raises(ValueError, match="row hash"):
        t4._attack_rows(cache, np.asarray(["e" * 64]), "comment_removal")


def test_score_frozen_model_reports_binary_metrics_and_matched_clean_reference() -> None:
    from lpcode_v1 import t4

    class Model:
        classes_ = np.asarray([0, 1])

        def predict(self, values):
            return (np.asarray(values)[:, 0] > 0).astype(int)

        def predict_proba(self, values):
            positive = np.where(np.asarray(values)[:, 0] > 0, 0.9, 0.1)
            return np.column_stack([1 - positive, positive])

    x = np.asarray([[-1.0], [1.0], [-2.0], [2.0]])
    y = np.asarray([0, 1, 0, 1])
    metrics = t4._score_model(Model(), x, y)
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["test_rows"] == 4
    assert metrics["test_class_counts"] == {"0": 2, "1": 2}


def test_gate_b_binding_rejects_copied_or_nonpassing_artifact(tmp_path: Path) -> None:
    from lpcode_v1 import t4

    copied = tmp_path / "gate_b.json"
    copied.write_bytes(t4.DEFAULT_GATE_B_PATH.read_bytes())
    with pytest.raises(ValueError, match="exact Gate B"):
        t4._load_strict_gate_b(copied)

    payload = json.loads(t4.DEFAULT_GATE_B_PATH.read_text(encoding="utf-8"))
    payload["strict"]["passed"] = False
    copied.write_text(json.dumps(payload), encoding="utf-8")
    assert hashlib.sha256(copied.read_bytes()).hexdigest() != hashlib.sha256(
        t4.DEFAULT_GATE_B_PATH.read_bytes()
    ).hexdigest()


def test_t4_smoke_trains_once_per_method_fold_writes_48_and_validates_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t3, t4

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("synthetic T4 data\n", encoding="utf-8")
    clean = _memory_cache()
    attacked = _attack_cache(clean)
    source_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    gate_a = t3._load_strict_gate_a(t3.DEFAULT_GATE_A_PATH)
    gate_a["source_jsonl_sha256"] = dict(gate_a["source_jsonl_sha256"])
    gate_a["source_jsonl_sha256"]["c"] = source_hash
    gate_b = t4._load_strict_gate_b(t4.DEFAULT_GATE_B_PATH)
    gate_b["source_jsonl_sha256"] = dict(gate_b["source_jsonl_sha256"])
    gate_b["source_jsonl_sha256"]["c"] = source_hash
    monkeypatch.setattr(t4, "_load_strict_gate_a", lambda _path: gate_a)
    monkeypatch.setattr(t4, "_load_strict_gate_b", lambda _path: gate_b)
    monkeypatch.setattr(t4, "load_or_build_enhanced_cache", lambda *args: clean)
    monkeypatch.setattr(t4, "load_or_build_attack_cache", lambda *args: attacked)
    fit_calls = []

    class Model:
        classes_ = np.asarray([0, 1])

        def fit(self, x, y):
            fit_calls.append((x.shape, tuple(y.tolist())))
            return self

        def predict(self, x):
            return np.asarray([1 if index % 2 == 0 else 0 for index in range(len(x))])

        def predict_proba(self, x):
            pred = self.predict(x)
            positive = np.where(pred == 1, 0.9, 0.1)
            return np.column_stack([1 - positive, positive])

    monkeypatch.setattr(t4, "build_model", lambda *_args: Model())
    output = tmp_path / "run"
    report = t4.run_t4_smoke(output, dataset_paths={"c": dataset})

    assert report["expected"] == 48
    assert report["completed"] == 48
    assert report["skipped"] == 0
    assert len(fit_calls) == 8
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["full_matrix"] is False
    assert config["conditions"] == list(t4.CONDITIONS)
    records = [
        json.loads(line)
        for line in (output / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 48
    assert all(set(record) == t4.T4_FOLD_RECORD_FIELDS for record in records)
    assert all(record["attack_failures"] == 0 for record in records)
    for fold in (0, 1):
        selected = [record for record in records if record["fold"] == fold]
        assert len(selected) == 24
        assert len({record["train_index_sha256"] for record in selected}) == 1
        assert len({record["test_index_sha256"] for record in selected}) == 1
        for condition in t4.CONDITIONS:
            cell = [record for record in selected if record["condition"] == condition]
            assert len(cell) == 4
            assert len({record["attack_success_set_sha256"] for record in cell}) == 1

    before = (output / "folds.jsonl").read_bytes()
    monkeypatch.setattr(
        t4,
        "build_model",
        lambda *_args: pytest.fail("validated T4 resume retrained a completed fold"),
    )
    resumed = t4.run_t4_smoke(output, dataset_paths={"c": dataset})
    assert resumed["completed"] == 0
    assert resumed["skipped"] == 48
    assert (output / "folds.jsonl").read_bytes() == before


def test_attack_cache_loads_dataset_under_task1_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t4

    dataset = tmp_path / "c.jsonl"
    dataset.write_text("placeholder\n", encoding="utf-8")
    calls = []
    row = {
        "file_name": "one.c",
        "human_src": "int main(void) { return 0; }",
        "llm_src": "int main(void) { int value = 0; return value; }",
        "paraphrased_by": "gpt3.5",
        "label": 1,
    }

    def loader(path, *, task):
        calls.append((Path(path), task))
        return [row]

    monkeypatch.setattr(t4, "load_jsonl", loader)
    cache = t4.load_or_build_attack_cache("c", dataset, tmp_path / "cache")
    assert calls == [(dataset.resolve(), "task1")]
    assert cache.features.shape == (1, 5, 28)


def test_t4_ledger_atomic_publish_retries_transient_windows_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t4

    attempts = []
    written = []

    def flaky(path, contents):
        attempts.append(Path(path))
        if len(attempts) < 3:
            raise PermissionError(5, "transient Windows denial")
        written.append(contents)

    monkeypatch.setattr(t4, "atomic_write_bytes", flaky)
    monkeypatch.setattr(t4.time, "sleep", lambda _seconds: None)
    t4._atomic_write_t4_records(
        tmp_path / "folds.jsonl",
        {("c", "mstf", 42, 0, "clean"): {"value": 1}},
    )
    assert len(attempts) == 3
    assert written == [t4._canonical_json({"value": 1})]
