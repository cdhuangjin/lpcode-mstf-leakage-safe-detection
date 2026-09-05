from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import pytest


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config(
    *,
    languages: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    folds: int,
    conditions: tuple[str, ...],
    full_matrix: bool,
) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": 1,
        "languages": list(languages),
        "methods": list(methods),
        "seeds": list(seeds),
        "n_splits": folds,
        "conditions": list(conditions),
        "method_contract": {method: {"name": method} for method in methods},
        "full_matrix": full_matrix,
        "gate_a_binding": {"gate_a_sha256": _hex("gate-a")},
        "gate_b_binding": {"gate_b_sha256": _hex("gate-b")},
    }
    config["config_id"] = _digest(config)
    return config


def _records(config: dict[str, object]) -> dict[tuple[str, str, int, int, str], dict[str, object]]:
    records: dict[tuple[str, str, int, int, str], dict[str, object]] = {}
    languages = config["languages"]
    methods = config["methods"]
    seeds = config["seeds"]
    conditions = config["conditions"]
    assert isinstance(languages, list)
    assert isinstance(methods, list)
    assert isinstance(seeds, list)
    assert isinstance(conditions, list)
    for language_index, language in enumerate(languages):
        for method, seed, fold, condition in product(
            methods, seeds, range(int(config["n_splits"])), conditions
        ):
            baseline_clean = 0.70 + 0.01 * language_index + 0.001 * fold
            if method == "mstf":
                clean = baseline_clean + 0.01
            elif method == "xgb_original":
                clean = baseline_clean + 0.005
            elif method == "best_transition":
                clean = baseline_clean + 0.008
            else:
                clean = baseline_clean
            if condition == "clean":
                f1 = clean
                clean_reference = clean
            elif condition == "combined":
                if method == "mstf":
                    f1 = clean - 0.04
                elif method == "lpcode_original":
                    f1 = clean - 0.10
                else:
                    f1 = clean - 0.07
                clean_reference = clean
            else:
                attack_index = conditions.index(condition)
                f1 = clean - 0.01 * attack_index
                clean_reference = clean
            train_hash = _hex(f"train:{language}:{seed}:{fold}")
            test_hash = _hex(f"test:{language}:{seed}:{fold}")
            success_hash = _hex(f"success:{language}:{seed}:{fold}:{condition}")
            record: dict[str, object] = {
                "language": language,
                "method": method,
                "seed": seed,
                "fold": fold,
                "condition": condition,
                "f1": f1,
                "precision": f1,
                "recall": f1,
                "auroc": min(1.0, f1 + 0.05),
                "mcc": 2.0 * f1 - 1.0,
                "train_rows": 100,
                "test_rows": 40,
                "train_class_counts": {"0": 50, "1": 50},
                "test_class_counts": {"0": 20, "1": 20},
                "train_index_sha256": train_hash,
                "test_index_sha256": test_hash,
                "attack_success_set_sha256": success_hash,
                "attack_failures": 0,
                "clean_reference_f1": clean_reference,
            }
            record["record_sha256"] = _digest(record)
            records[(str(language), str(method), int(seed), fold, str(condition))] = record
    return records


def _write_run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    full: bool,
) -> tuple[dict[str, object], dict[tuple[str, str, int, int, str], dict[str, object]]]:
    from lpcode_v1 import gates_t4, t4

    languages = t4.LANGUAGES if full else (t4.LANGUAGES[0],)
    seeds = t4.DEFAULT_SEEDS if full else (t4.DEFAULT_SEEDS[0],)
    folds = 5 if full else 2
    config = _config(
        languages=languages,
        methods=t4.METHODS,
        seeds=seeds,
        folds=folds,
        conditions=t4.CONDITIONS,
        full_matrix=full,
    )
    records = _records(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (root / "folds.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records.values()
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(t4, "_validate_t4_config", lambda value: None)
    monkeypatch.setattr(
        t4,
        "_load_t4_records",
        lambda path, value: records,
    )
    return config, records


def test_gate_c_thresholds_and_nonpositive_baseline_drop() -> None:
    from lpcode_v1.gates_t4 import gate_c

    dual = gate_c(
        candidate_attacked_f1=0.66,
        baseline_attacked_f1=0.60,
        candidate_drop=0.04,
        baseline_drop=0.10,
    )
    assert dual["passed"] is True
    assert dual["candidate_higher"] is True
    assert dual["advantage_at_least_0_05"] is True
    assert dual["relative_drop_reduction"] == pytest.approx(0.60)
    assert dual["relative_drop_reduction_at_least_0_30"] is True
    assert dual["dual_criterion"] is True

    reduction_only = gate_c(
        candidate_attacked_f1=0.601,
        baseline_attacked_f1=0.60,
        candidate_drop=0.06,
        baseline_drop=0.10,
    )
    assert reduction_only["passed"] is True
    assert reduction_only["advantage_at_least_0_05"] is False
    assert reduction_only["dual_criterion"] is False

    unavailable = gate_c(
        candidate_attacked_f1=0.64,
        baseline_attacked_f1=0.60,
        candidate_drop=-0.02,
        baseline_drop=0.0,
    )
    assert unavailable["passed"] is False
    assert unavailable["relative_drop_branch_evaluable"] is False
    assert unavailable["relative_drop_reduction"] is None
    assert unavailable["relative_drop_reduction_at_least_0_30"] is False

    not_higher = gate_c(
        candidate_attacked_f1=0.60,
        baseline_attacked_f1=0.60,
        candidate_drop=0.01,
        baseline_drop=0.10,
    )
    assert not_higher["passed"] is False
    assert not_higher["candidate_higher"] is False

    exact_decimal_boundaries = gate_c(
        candidate_attacked_f1=0.60,
        baseline_attacked_f1=0.55,
        candidate_drop=0.07,
        baseline_drop=0.10,
    )
    assert exact_decimal_boundaries["advantage_at_least_0_05"] is True
    assert exact_decimal_boundaries["relative_drop_reduction_at_least_0_30"] is True
    assert exact_decimal_boundaries["dual_criterion"] is True


def test_summary_validates_1440_matrix_paired_drops_bootstrap_and_hash_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import gates_t4, t4

    root = tmp_path / "full"
    config, records = _write_run(root, monkeypatch, full=True)
    inputs_before = {
        name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")
    }

    report = gates_t4.summarize_t4(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "gate_c.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert report["verdict"] is True
    assert gate["status"] == "evaluable"
    assert gate["strict"]["passed"] is True
    assert gate["strict"]["dual_criterion"] is True
    assert gate["strict"]["attacked_f1_advantage"] == pytest.approx(0.07)
    assert gate["strict"]["baseline_drop"] == pytest.approx(0.10)
    assert gate["strict"]["candidate_drop"] == pytest.approx(0.04)
    assert gate["strict"]["relative_drop_reduction"] == pytest.approx(0.60)
    assert summary["matrix"] == {
        "complete_cartesian_product": True,
        "expected_records": 1440,
        "observed_records": 1440,
        "official_gate_expected_records": 1440,
    }

    cell = summary["cell_summaries"]["combined"][t4.LANGUAGES[0]]["mstf"]
    assert cell["n"] == 15
    assert cell["f1_mean"] == pytest.approx(0.672)
    assert cell["f1_std"] > 0.0
    combined = summary["paired_mstf_minus_lpcode"]["by_condition"]["combined"]
    assert combined["n_macro_observations"] == 15
    assert combined["macro_language_mean_delta_f1"] == pytest.approx(0.07)
    assert combined["ci_95"]["replicates"] == 10_000
    assert combined["ci_95"]["seed"] == gates_t4.BOOTSTRAP_SEED
    drops = summary["clean_to_attack_drops"]["by_condition"]["combined"]
    assert drops["lpcode_original"]["macro_language_mean_drop_f1"] == pytest.approx(0.10)
    assert drops["mstf"]["macro_language_mean_drop_f1"] == pytest.approx(0.04)
    assert summary["direction_counts"]["combined"] == {
        "negative": 0,
        "positive": 3,
        "zero": 0,
    }

    assert inputs_before == {
        name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")
    }
    assert set(manifest["files"]) == {
        "config.json",
        "folds.jsonl",
        "summary.json",
        "table_c.csv",
        "table_c.md",
        "gate_c.json",
    }
    for name, metadata in manifest["files"].items():
        payload = (root / name).read_bytes()
        assert metadata == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    assert "LPcodedec Original" in (root / "table_c.md").read_text(encoding="utf-8")
    assert "MSTF" in (root / "table_c.md").read_text(encoding="utf-8")
    assert len(records) == 1440
    assert config["full_matrix"] is True


def test_summary_rejects_condition_pair_hash_and_cross_method_success_set_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import gates_t4, t4

    root = tmp_path / "bad"
    _, records = _write_run(root, monkeypatch, full=False)
    condition_target = records[
        (t4.LANGUAGES[0], t4.METHODS[0], t4.DEFAULT_SEEDS[0], 0, t4.CONDITIONS[1])
    ]
    condition_target["test_index_sha256"] = _hex("wrong-test-pair")
    with pytest.raises(ValueError, match="train/test pair hashes differ across conditions"):
        gates_t4.summarize_t4(root)

    _, records = _write_run(root, monkeypatch, full=False)
    method_target = records[
        (t4.LANGUAGES[0], t4.METHODS[1], t4.DEFAULT_SEEDS[0], 0, t4.CONDITIONS[1])
    ]
    method_target["attack_success_set_sha256"] = _hex("wrong-success-set")
    with pytest.raises(ValueError, match="success-set hashes differ across methods"):
        gates_t4.summarize_t4(root)


def test_subset_is_descriptive_not_evaluable_and_outputs_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import gates_t4

    root = tmp_path / "smoke"
    _write_run(root, monkeypatch, full=False)
    first = gates_t4.summarize_t4(root)
    gate = json.loads((root / "gate_c.json").read_text(encoding="utf-8"))
    assert first["verdict"] == "not_evaluable"
    assert gate["status"] == "not_evaluable"
    assert gate["strict"] is None
    assert any("languages" in reason for reason in gate["reasons"])
    assert any("seeds" in reason for reason in gate["reasons"])
    assert any("n_splits" in reason for reason in gate["reasons"])
    assert any("full_matrix" in reason for reason in gate["reasons"])

    names = ("summary.json", "table_c.csv", "table_c.md", "gate_c.json", "manifest.json")
    before = {name: (root / name).read_bytes() for name in names}
    gates_t4.summarize_t4(root)
    assert before == {name: (root / name).read_bytes() for name in names}


def test_t4_cli_summary_only_never_calls_runner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lpcode_v1 import gates_t4, t4

    called: list[str | Path] = []

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("summary-only must not fit or invoke the T4 runner")

    def fake_summary(path: Path) -> dict[str, object]:
        called.append(path)
        return {"verdict": True}

    monkeypatch.setattr(t4, "run_t4", fail_run)
    monkeypatch.setattr(t4, "run_t4_smoke", fail_run)
    monkeypatch.setattr(gates_t4, "summarize_t4", fake_summary)
    monkeypatch.setattr(
        sys,
        "argv",
        ["t4", "--summarize-only", "--output-root", "summary-root"],
    )

    t4.main()

    assert called == ["summary-root"]
    assert json.loads(capsys.readouterr().out) == {"verdict": True}


def test_summarize_t4_holds_output_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lpcode_v1 import gates_t4

    root = tmp_path / "run"
    root.mkdir()
    events: list[str] = []

    @contextmanager
    def lock(path: Path):
        events.append(f"enter:{path}")
        yield
        events.append(f"exit:{path}")

    def summarize(path: Path) -> dict[str, object]:
        events.append("summarize")
        return {"verdict": True}

    monkeypatch.setattr(gates_t4, "_exclusive_output_lock", lock)
    monkeypatch.setattr(gates_t4, "_summarize_t4_locked", summarize)
    assert gates_t4.summarize_t4(root) == {"verdict": True}
    assert events == [f"enter:{root.resolve()}", "summarize", f"exit:{root.resolve()}"]
