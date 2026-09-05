from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


LLMS = (
    "gpt3.5",
    "gemini-pro",
    "wizardcoder:33b-v1.1",
    "deepseek-coder:33b-instruct",
)


def _memory_cache(language: str = "c", origin_count: int = 10):
    from lpcode_v1 import t3

    rows = []
    for origin_index in range(origin_count):
        for source_index, source in enumerate(LLMS):
            rows.append(
                {
                    "file_name": f"{language}-origin-{origin_index}",
                    "human_src": f"{language} human {origin_index}",
                    "llm_src": f"{language} candidate {origin_index} {source_index}",
                    "paraphrased_by": source,
                    "label": 1,
                }
            )
    count = len(rows)
    origins = np.asarray([row["file_name"] for row in rows], dtype=str)
    return t3.EnhancedFeatureCache(
        language=language,
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


def test_language_pair_bank_uses_every_origin_balanced_and_is_deterministic() -> None:
    from lpcode_v1 import t5

    cache = _memory_cache(origin_count=10)
    first = t5.build_language_pair_bank(cache, "c", seed=42, n_pair_folds=5)
    second = t5.build_language_pair_bank(cache, "c", seed=42, n_pair_folds=5)

    assert first == second
    assert first.language == "c"
    assert first.seed == 42
    assert len(first.pairs) == 10 * 4 * 2
    assert len(first.pair_sha256) == 64
    labels = np.asarray([pair.label for pair in first.pairs])
    assert {int(label): int((labels == label).sum()) for label in (0, 1)} == {
        0: 40,
        1: 40,
    }
    cells = [
        (pair.human_origin_id, pair.llm_source, pair.label) for pair in first.pairs
    ]
    assert len(cells) == len(set(cells)) == 80
    assert {pair.human_origin_id for pair in first.pairs} == {
        f"c-origin-{index}" for index in range(10)
    }
    assert all(
        pair.human_component_id != pair.candidate_component_id
        for pair in first.pairs
        if pair.label == 0
    )
    assert first.audit["class_counts"] == {"0": 40, "1": 40}
    assert first.audit["negative_component_violation_count"] == 0


def test_language_pair_bank_changes_with_seed_but_not_membership() -> None:
    from lpcode_v1 import t5

    cache = _memory_cache()
    left = t5.build_language_pair_bank(cache, "c", 42)
    right = t5.build_language_pair_bank(cache, "c", 123)
    assert left.pair_sha256 != right.pair_sha256
    assert {
        (pair.human_origin_id, pair.llm_source, pair.label) for pair in left.pairs
    } == {
        (pair.human_origin_id, pair.llm_source, pair.label) for pair in right.pairs
    }


def test_t5_exact_gate_c_binding_authorizes_cross_language() -> None:
    from lpcode_v1 import t5

    binding = t5._load_strict_gate_c(t5.DEFAULT_GATE_C_PATH)
    assert binding["strict_passed"] is True
    assert binding["authorizes_t5"] is True
    assert binding["dual_criterion"] is True
    assert binding["attacked_f1_advantage"] >= 0.05
    assert binding["relative_drop_reduction"] >= 0.30
    assert len(binding["gate_c_sha256"]) == 64
    assert len(binding["manifest_sha256"]) == 64


def test_t5_gate_c_binding_rejects_copied_artifact(tmp_path: Path) -> None:
    from lpcode_v1 import t5

    copied = tmp_path / "gate_c.json"
    copied.write_bytes(t5.DEFAULT_GATE_C_PATH.read_bytes())
    with pytest.raises(ValueError, match="exact Gate C"):
        t5._load_strict_gate_c(copied)


def test_cross_language_metadata_rejects_exact_content_overlap() -> None:
    from lpcode_v1 import t5

    train_cache = _memory_cache("c")
    test_cache = _memory_cache("py")
    train_bank = t5.build_language_pair_bank(train_cache, "c", 42)
    test_bank = t5.build_language_pair_bank(test_cache, "py", 42)
    candidate_hashes = test_cache.candidate_code_sha256.copy()
    candidate_hashes[0] = train_cache.candidate_code_sha256[0]
    contaminated = replace(test_cache, candidate_code_sha256=candidate_hashes)
    contaminated_bank = t5.build_language_pair_bank(contaminated, "py", 42)

    with pytest.raises(ValueError, match="exact code content leakage"):
        t5._cross_language_split_metadata(
            {"c": (train_cache, train_bank)}, contaminated, contaminated_bank
        )


def test_t5_full_matrix_is_48() -> None:
    from lpcode_v1 import t5

    assert t5.METHODS == (
        "lpcode_original",
        "xgb_original",
        "best_transition",
        "mstf",
    )
    assert t5._evaluation_count(t5.LANGUAGES, t5.METHODS, t5.DEFAULT_SEEDS) == 48


def _metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    return {
        "f1": 0.75,
        "precision": 0.75,
        "recall": 0.75,
        "auroc": 0.8,
        "mcc": 0.5,
        "fit_seconds": 0.01,
        "predict_seconds": 0.01,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_class_counts": {
            str(label): int((y_train == label).sum()) for label in (0, 1)
        },
        "test_class_counts": {
            str(label): int((y_test == label).sum()) for label in (0, 1)
        },
    }


def test_t5_smoke_writes_16_shared_split_records_and_validates_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t5

    caches = {language: _memory_cache(language, 10) for language in t5.LANGUAGES}
    paths = {}
    for language in t5.LANGUAGES:
        path = tmp_path / f"{language}.jsonl"
        path.write_text(f"synthetic {language}\n", encoding="utf-8")
        paths[language] = path
    gate = t5._load_strict_gate_c(t5.DEFAULT_GATE_C_PATH)
    gate["source_jsonl_sha256"] = {
        language: hashlib.sha256(paths[language].read_bytes()).hexdigest()
        for language in t5.LANGUAGES
    }
    monkeypatch.setattr(t5, "_load_strict_gate_c", lambda _path: gate)
    monkeypatch.setattr(
        t5,
        "load_or_build_enhanced_cache",
        lambda language, *_args, **_kwargs: caches[language],
    )
    calls = []

    def evaluator(x_train, y_train, x_test, y_test, model, seed):
        calls.append((x_train.shape, x_test.shape, model, seed))
        return _metrics(y_train, y_test)

    monkeypatch.setattr(t5, "evaluate_fold", evaluator)
    output = tmp_path / "run"
    report = t5.run_t5_smoke(output, dataset_paths=paths)
    assert report == {
        "schema_version": 1,
        "config_id": report["config_id"],
        "expected": 16,
        "completed": 16,
        "skipped": 0,
        "output_root": str(output.resolve()),
    }
    assert len(calls) == 16
    assert {call[0][1] for call in calls} == {20, 30, 112}
    assert {call[1][1] for call in calls} == {20, 30, 112}
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["full_matrix"] is False
    assert config["heldout_languages"] == list(t5.LANGUAGES)
    assert config["methods"] == list(t5.METHODS)
    records = [
        json.loads(line)
        for line in (output / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 16
    assert all(set(record) == t5.T5_RECORD_FIELDS for record in records)
    assert len({tuple(t5._record_key(record)) for record in records}) == 16
    for heldout in t5.LANGUAGES:
        selected = [record for record in records if record["heldout_language"] == heldout]
        assert len(selected) == 4
        assert len({record["train_index_sha256"] for record in selected}) == 1
        assert len({record["test_index_sha256"] for record in selected}) == 1
        assert {record["feature_dimensions"] for record in selected} == {20, 30, 112}
        assert all(record["content_leakage_count"] == 0 for record in selected)
        assert all(record["train_class_counts"]["0"] == record["train_class_counts"]["1"] for record in selected)
        assert all(record["test_class_counts"]["0"] == record["test_class_counts"]["1"] for record in selected)

    before = (output / "folds.jsonl").read_bytes()
    monkeypatch.setattr(
        t5,
        "evaluate_fold",
        lambda *_args: pytest.fail("validated T5 resume refitted a record"),
    )
    resumed = t5.run_t5_smoke(output, dataset_paths=paths)
    assert resumed["completed"] == 0
    assert resumed["skipped"] == 16
    assert (output / "folds.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("f1", float("nan"), "metric"),
        ("test_index_sha256", "0" * 64, "reconstruction"),
        ("gate_c_sha256", "1" * 64, "schema/config"),
        ("cache_content_sha256", {}, "schema/config"),
    ],
)
def test_t5_resume_rejects_corrupt_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    from lpcode_v1 import t5

    caches = {language: _memory_cache(language, 10) for language in t5.LANGUAGES}
    paths = {}
    for language in t5.LANGUAGES:
        path = tmp_path / f"{language}.jsonl"
        path.write_text(f"synthetic {language}\n", encoding="utf-8")
        paths[language] = path
    gate = t5._load_strict_gate_c(t5.DEFAULT_GATE_C_PATH)
    gate["source_jsonl_sha256"] = {
        language: hashlib.sha256(paths[language].read_bytes()).hexdigest()
        for language in t5.LANGUAGES
    }
    monkeypatch.setattr(t5, "_load_strict_gate_c", lambda _path: gate)
    monkeypatch.setattr(t5, "load_or_build_enhanced_cache", lambda language, *_args, **_kwargs: caches[language])
    monkeypatch.setattr(t5, "evaluate_fold", lambda _xtr, ytr, _xte, yte, *_args: _metrics(ytr, yte))
    output = tmp_path / "run"
    t5.run_t5_smoke(output, dataset_paths=paths)
    ledger = output / "folds.jsonl"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    records[0][field] = value
    if field != "f1":
        records[0]["record_sha256"] = t5._record_sha256(records[0])
    ledger.write_text("".join(json.dumps(record, allow_nan=True) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        t5.run_t5_smoke(output, dataset_paths=paths)


def test_t5_cli_summary_only_never_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lpcode_v1 import gates_t5, t5

    output = tmp_path / "completed"
    output.mkdir()
    monkeypatch.setattr(
        t5,
        "run_t5",
        lambda *_args, **_kwargs: pytest.fail("summary-only refitted T5"),
    )
    monkeypatch.setattr(
        t5,
        "run_t5_smoke",
        lambda *_args, **_kwargs: pytest.fail("summary-only ran T5 smoke"),
    )
    monkeypatch.setattr(
        gates_t5,
        "summarize_t5",
        lambda root: {"output_root": str(Path(root).resolve()), "verdict": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["t5", "--summarize-only", "--output-root", str(output)],
    )
    assert t5.main() == 0
    assert json.loads(capsys.readouterr().out)["verdict"] is True
