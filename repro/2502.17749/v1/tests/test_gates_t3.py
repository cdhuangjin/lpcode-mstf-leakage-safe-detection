from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _gate_binding() -> dict[str, object]:
    from lpcode_v1 import t3

    return {
        "gate_a_path": str(t3.DEFAULT_GATE_A_PATH.resolve()),
        "gate_a_sha256": _hex("gate-a"),
        "manifest_path": str(t3.DEFAULT_GATE_A_PATH.resolve().with_name("manifest.json")),
        "manifest_sha256": _hex("gate-a-manifest"),
        "strict_config_sha256": _hex("strict-config"),
        "strict_config_id": _hex("strict-config-id"),
        "source_jsonl_sha256": {
            language: _hex(f"source:{language}") for language in t3.LANGUAGES
        },
        "protocol_version": t3.STRICT_GATE_PROTOCOL_VERSION,
        "strict_passed": True,
        "selected_candidate": {"representation": "full", "model": "xgb"},
    }


def _config(
    *,
    languages: tuple[str, ...],
    heldouts: tuple[str, ...],
    seeds: tuple[int, ...],
    folds: int,
    methods: tuple[str, ...],
    limit_origins: int | None,
) -> dict[str, object]:
    from lpcode_v1 import t3

    binding = _gate_binding()
    config: dict[str, object] = {
        "schema_version": t3.T3_SCHEMA_VERSION,
        "task": "task3_unseen_llm",
        "fold_index_base": 0,
        "languages": list(languages),
        "heldout_llms": list(heldouts),
        "seeds": list(seeds),
        "n_splits": folds,
        "methods": list(methods),
        "method_contract": {
            method: t3._method_contract(binding["selected_candidate"])[method]  # type: ignore[arg-type,index]
            for method in methods
        },
        "limit_origins": limit_origins,
        "split_protocol": t3.T3_SPLIT_PROTOCOL_VERSION,
        "pair_protocol": t3.PAIR_PROTOCOL_VERSION,
        "component_protocol": t3.COMPONENT_PROTOCOL_VERSION,
        "gate_a_binding": binding,
        "feature_contract": t3._runner_feature_contract(),
        "source_jsonl_sha256": {
            language: binding["source_jsonl_sha256"][language]  # type: ignore[index]
            for language in languages
        },
        "cache_content_sha256": {
            language: _hex(f"cache:{language}") for language in languages
        },
        "package_versions": t3._runner_package_versions(),
        "implementation_contract": t3._runner_implementation_contract(),
    }
    config["config_id"] = t3._digest_json(config)
    return config


def _write_t3_run(
    root: Path,
    *,
    languages: tuple[str, ...] | None = None,
    heldouts: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    folds: int = 5,
    methods: tuple[str, ...] | None = None,
    limit_origins: int | None = None,
    holdout_deltas: dict[str, float] | None = None,
) -> None:
    from lpcode_v1 import t3

    language_axis = languages or t3.LANGUAGES
    heldout_axis = heldouts or t3.LLM_SOURCES
    seed_axis = seeds or t3.DEFAULT_SEEDS
    method_axis = methods or t3.T3_METHODS
    deltas = holdout_deltas or {
        t3.LLM_SOURCES[0]: 0.05,
        t3.LLM_SOURCES[1]: 0.05,
        t3.LLM_SOURCES[2]: 0.04,
        t3.LLM_SOURCES[3]: -0.01,
    }
    config = _config(
        languages=language_axis,
        heldouts=heldout_axis,
        seeds=seed_axis,
        folds=folds,
        methods=method_axis,
        limit_origins=limit_origins,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    records: list[dict[str, object]] = []
    for language_index, language in enumerate(language_axis):
        for heldout_index, heldout in enumerate(heldout_axis):
            for method in method_axis:
                spec = config["method_contract"][method]  # type: ignore[index]
                for seed_index, seed in enumerate(seed_axis):
                    for fold in range(folds):
                        baseline = (
                            0.50
                            + 0.005 * language_index
                            + 0.003 * heldout_index
                            + 0.002 * seed_index
                            + 0.001 * fold
                        )
                        if method == "mstf":
                            f1 = baseline + deltas[heldout]
                        elif method == "xgb_original":
                            f1 = baseline + 0.01
                        elif method == "best_transition":
                            f1 = baseline + 0.015
                        else:
                            f1 = baseline
                        train_hash = _hex(f"train:{language}:{heldout}:{seed}:{fold}")
                        test_hash = _hex(f"test:{language}:{heldout}:{seed}:{fold}")
                        train_sources = [source for source in t3.LLM_SOURCES if source != heldout]
                        record: dict[str, object] = {
                            "schema_version": t3.T3_SCHEMA_VERSION,
                            "config_id": config["config_id"],
                            "language": language,
                            "heldout_llm": heldout,
                            "method": method,
                            "feature_family": spec["feature_family"],
                            "representation": spec["representation"],
                            "model": spec["model"],
                            "seed": seed,
                            "fold": fold,
                            "split_protocol": t3.T3_SPLIT_PROTOCOL_VERSION,
                            "pair_protocol": t3.PAIR_PROTOCOL_VERSION,
                            "component_protocol": t3.COMPONENT_PROTOCOL_VERSION,
                            "gate_a_sha256": config["gate_a_binding"]["gate_a_sha256"],  # type: ignore[index]
                            "gate_a_manifest_sha256": config["gate_a_binding"]["manifest_sha256"],  # type: ignore[index]
                            "cache_content_sha256": config["cache_content_sha256"][language],  # type: ignore[index]
                            "source_jsonl_sha256": config["source_jsonl_sha256"][language],  # type: ignore[index]
                            "leakage_count": 0,
                            "endpoint_leakage_count": 0,
                            "content_leakage_count": 0,
                            "negative_component_violation_count": 0,
                            "train_index_sha256": train_hash,
                            "test_index_sha256": test_hash,
                            "feature_dimensions": spec["feature_dimensions"],
                            "train_unique_origins": 12,
                            "test_unique_origins": 4,
                            "train_unique_code_hashes": 24,
                            "test_unique_code_hashes": 8,
                            "train_unique_components": 12,
                            "test_unique_components": 4,
                            "train_llm_sources": train_sources,
                            "test_llm_sources": [heldout],
                            "train_llm_label_counts": {
                                source: {"0": 2, "1": 2} for source in train_sources
                            },
                            "test_llm_label_counts": {heldout: {"0": 2, "1": 2}},
                            "train_human_parse_failures": 0,
                            "train_candidate_parse_failures": 0,
                            "test_human_parse_failures": 0,
                            "test_candidate_parse_failures": 0,
                            "f1": f1,
                            "precision": f1,
                            "recall": f1,
                            "auroc": f1,
                            "mcc": 2 * f1 - 1,
                            "fit_seconds": 0.1,
                            "predict_seconds": 0.01,
                            "train_rows": 12,
                            "test_rows": 4,
                            "train_class_counts": {"0": 6, "1": 6},
                            "test_class_counts": {"0": 2, "1": 2},
                        }
                        record["record_sha256"] = t3._t3_record_sha256(record)
                        records.append(record)
    (root / "folds.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_gate_b_strict_and_relaxed_boundaries() -> None:
    from lpcode_v1 import t3
    from lpcode_v1.gates_t3 import gate_b

    strict = gate_b(
        {
            t3.LLM_SOURCES[0]: 0.04,
            t3.LLM_SOURCES[1]: 0.04,
            t3.LLM_SOURCES[2]: 0.04,
            t3.LLM_SOURCES[3]: 0.0,
        }
    )
    assert strict["passed"] is True
    assert strict["relaxed_passed"] is True
    assert strict["holdouts_won"] == 3
    assert strict["overall_macro_mean_delta_f1"] == pytest.approx(0.03)

    relaxed = gate_b(
        {
            t3.LLM_SOURCES[0]: 0.01,
            t3.LLM_SOURCES[1]: 0.01,
            t3.LLM_SOURCES[2]: 0.01,
            t3.LLM_SOURCES[3]: -0.001,
        }
    )
    assert relaxed["passed"] is False
    assert relaxed["relaxed_passed"] is True
    assert relaxed["authorizes_t4"] is False


@pytest.mark.parametrize(
    "deltas",
    [
        {},
        {"unknown": 0.1},
        {"gpt3.5": True, "gemini-pro": 0.1, "wizardcoder:33b-v1.1": 0.1, "deepseek-coder:33b-instruct": 0.1},
        {"gpt3.5": math.nan, "gemini-pro": 0.1, "wizardcoder:33b-v1.1": 0.1, "deepseek-coder:33b-instruct": 0.1},
    ],
)
def test_gate_b_rejects_invalid_axes_and_values(deltas: dict[str, object]) -> None:
    from lpcode_v1.gates_t3 import gate_b

    with pytest.raises(ValueError):
        gate_b(deltas)  # type: ignore[arg-type]


def test_summary_validates_960_cells_pairs_macro_aggregation_and_artifacts(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t3
    from lpcode_v1.gates_t3 import summarize_t3

    root = tmp_path / "complete"
    _write_t3_run(root)
    inputs_before = {
        name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")
    }

    report = summarize_t3(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "gate_b.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert report["verdict"] is True
    assert gate["status"] == "evaluable"
    assert gate["strict"]["passed"] is True
    assert gate["strict"]["authorizes_t4"] is True
    assert gate["strict"]["holdouts_won"] == 3
    assert gate["strict"]["overall_macro_mean_delta_f1"] == pytest.approx(0.0325)
    assert summary["matrix"]["expected_records"] == 960
    assert summary["matrix"]["observed_records"] == 960
    assert summary["matrix"]["complete_cartesian_product"] is True

    cell = summary["cell_summaries"][t3.LLM_SOURCES[0]]["c"]["lpcode_original"]
    assert cell["n"] == 15
    assert cell["f1_mean"] == pytest.approx(0.504)
    assert cell["f1_std"] > 0.0

    holdout = summary["paired_mstf_minus_lpcode"]["by_holdout"][t3.LLM_SOURCES[0]]
    assert holdout["n_paired_records"] == 60
    assert holdout["n_macro_observations"] == 15
    assert holdout["macro_language_mean_delta_f1"] == pytest.approx(0.05)
    assert holdout["ci_95"]["replicates"] == 10_000
    assert holdout["ci_95"]["seed"] == 250_217_749
    assert holdout["ci_95"]["low"] == pytest.approx(0.05)
    assert holdout["ci_95"]["high"] == pytest.approx(0.05)
    overall = summary["paired_mstf_minus_lpcode"]["overall"]
    assert overall["macro_holdout_language_mean_delta_f1"] == pytest.approx(0.0325)
    assert overall["n_paired_records"] == 240
    assert overall["n_macro_observations"] == 60

    direction = summary["direction_consistency"][t3.LLM_SOURCES[3]]["py"]["42"]
    assert direction == {"direction": "negative", "folds": 5, "mean_delta_f1": pytest.approx(-0.01)}
    assert summary["direction_counts"] == {"negative": 12, "positive": 36, "zero": 0}

    assert inputs_before == {
        name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")
    }
    assert set(manifest["files"]) == {
        "config.json",
        "folds.jsonl",
        "summary.json",
        "table_b.csv",
        "table_b.md",
        "gate_b.json",
    }
    assert manifest["protocol"] == {
        "split": t3.T3_SPLIT_PROTOCOL_VERSION,
        "pair": t3.PAIR_PROTOCOL_VERSION,
        "component": t3.COMPONENT_PROTOCOL_VERSION,
    }
    assert manifest["leakage"]["required_zero_fields"] == [
        "leakage_count",
        "endpoint_leakage_count",
        "content_leakage_count",
        "negative_component_violation_count",
    ]
    for name, metadata in manifest["files"].items():
        payload = (root / name).read_bytes()
        assert metadata == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert "LPcodedec Original" in (root / "table_b.md").read_text(encoding="utf-8")
    assert "MSTF" in (root / "table_b.md").read_text(encoding="utf-8")


def test_summary_is_byte_deterministic_and_refuses_different_method_splits(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t3
    from lpcode_v1.gates_t3 import summarize_t3

    root = tmp_path / "complete"
    _write_t3_run(root)
    summarize_t3(root)
    generated = ("summary.json", "table_b.csv", "table_b.md", "gate_b.json", "manifest.json")
    before = {name: (root / name).read_bytes() for name in generated}
    summarize_t3(root)
    assert before == {name: (root / name).read_bytes() for name in generated}

    records = [
        json.loads(line)
        for line in (root / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(record for record in records if record["method"] == "xgb_original")
    target["train_index_sha256"] = _hex("wrong split")
    target["record_sha256"] = t3._t3_record_sha256(target)
    (root / "folds.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split hashes differ across methods"):
        summarize_t3(root)


def test_summary_rejects_incomplete_matrix_and_invalid_gate_binding(tmp_path: Path) -> None:
    from lpcode_v1 import t3
    from lpcode_v1.gates_t3 import summarize_t3

    root = tmp_path / "run"
    _write_t3_run(root)
    lines = (root / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    (root / "folds.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete T3 fold matrix"):
        summarize_t3(root)

    _write_t3_run(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["gate_a_binding"]["strict_passed"] = False
    config["config_id"] = t3._digest_json({key: value for key, value in config.items() if key != "config_id"})
    (root / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict Gate A binding"):
        summarize_t3(root)


def test_smoke_subset_is_descriptive_but_not_evaluable(tmp_path: Path) -> None:
    from lpcode_v1 import t3
    from lpcode_v1.gates_t3 import summarize_t3

    root = tmp_path / "smoke"
    _write_t3_run(
        root,
        languages=("c",),
        heldouts=t3.LLM_SOURCES,
        seeds=(42,),
        folds=2,
        methods=t3.T3_METHODS,
        limit_origins=8,
    )
    report = summarize_t3(root)
    gate = json.loads((root / "gate_b.json").read_text(encoding="utf-8"))

    assert report["verdict"] == "not_evaluable"
    assert gate["status"] == "not_evaluable"
    assert gate["strict"] is None
    assert gate["relaxed"] is None
    assert any("languages" in reason for reason in gate["reasons"])
    assert any("n_splits" in reason for reason in gate["reasons"])
    assert any("limit_origins" in reason for reason in gate["reasons"])


def test_t3_cli_summary_only_never_calls_runner(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from lpcode_v1 import gates_t3, t3

    called: list[Path] = []

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("summary-only must not fit or invoke the T3 runner")

    def fake_summary(path: Path) -> dict[str, object]:
        called.append(path)
        return {"verdict": True}

    monkeypatch.setattr(t3, "run_t3", fail_run)
    monkeypatch.setattr(t3, "run_t3_smoke", fail_run)
    monkeypatch.setattr(gates_t3, "summarize_t3", fake_summary)
    monkeypatch.setattr(sys, "argv", ["t3", "--summarize-only", "--output-root", "summary-root"])

    t3.main()

    assert called == [Path("summary-root")]
    assert json.loads(capsys.readouterr().out) == {"verdict": True}


def test_summarize_t3_holds_output_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lpcode_v1 import gates_t3, t3

    root = tmp_path / "smoke"
    _write_t3_run(
        root,
        languages=("c",),
        heldouts=(t3.LLM_SOURCES[0],),
        seeds=(42,),
        folds=2,
        methods=t3.T3_METHODS,
        limit_origins=8,
    )
    events: list[str] = []

    @contextmanager
    def lock(path: Path):
        events.append(f"enter:{path}")
        yield
        events.append(f"exit:{path}")

    real = gates_t3._summarize_t3_locked

    def wrapped(path: Path) -> dict[str, object]:
        events.append("summarize")
        return real(path)

    monkeypatch.setattr(gates_t3, "_exclusive_output_lock", lock)
    monkeypatch.setattr(gates_t3, "_summarize_t3_locked", wrapped)
    gates_t3.summarize_t3(root)

    assert events == [f"enter:{root.resolve()}", "summarize", f"exit:{root.resolve()}"]
