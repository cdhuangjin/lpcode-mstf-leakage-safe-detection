from __future__ import annotations

import hashlib
import json
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


def _run_data(*, full: bool, deltas: tuple[float, ...] = (0.04, 0.03, 0.01, 0.02)):
    from lpcode_v1 import t5

    languages = t5.LANGUAGES if full else (t5.LANGUAGES[0],)
    seeds = t5.DEFAULT_SEEDS if full else (t5.DEFAULT_SEEDS[0],)
    methods = t5.METHODS
    config = {
        "schema_version": 1,
        "task": "task5_cross_language",
        "languages": list(languages),
        "heldout_languages": list(languages),
        "methods": list(methods),
        "seeds": list(seeds),
        "n_pair_folds": 5 if full else 2,
        "limit_origins": None if full else 10,
        "full_matrix": full,
        "split_protocol": t5.T5_SPLIT_PROTOCOL_VERSION,
        "bank_protocol": t5.T5_BANK_PROTOCOL_VERSION,
        "pair_protocol": t5.PAIR_PROTOCOL_VERSION,
        "component_protocol": t5.COMPONENT_PROTOCOL_VERSION,
        "method_contract": {method: {"name": method} for method in methods},
        "source_jsonl_sha256": {language: _hex(f"source:{language}") for language in languages},
        "cache_content_sha256": {language: _hex(f"cache:{language}") for language in languages},
        "gate_c_binding": {
            "gate_c_sha256": _hex("gate-c"),
            "manifest_sha256": _hex("gate-c-manifest"),
            "strict_passed": True,
        },
        "implementation_contract": {"t5_source_sha256": _hex("t5")},
        "package_versions": {"python": "test"},
    }
    config["config_id"] = _digest(config)
    records = {}
    for language_index, heldout in enumerate(languages):
        for method, seed in product(methods, seeds):
            baseline = 0.70 + 0.01 * language_index + 0.0001 * seeds.index(seed)
            if method == "mstf":
                f1 = baseline + deltas[language_index]
            elif method == "xgb_original":
                f1 = baseline + 0.005
            elif method == "best_transition":
                f1 = baseline + 0.012
            else:
                f1 = baseline
            train_hash = _hex(f"train:{heldout}:{seed}")
            test_hash = _hex(f"test:{heldout}:{seed}")
            train_banks = {
                language: _hex(f"bank:{language}:{seed}")
                for language in languages
                if language != heldout
            }
            record = {
                "heldout_language": heldout,
                "method": method,
                "seed": seed,
                "f1": f1,
                "precision": f1,
                "recall": f1,
                "auroc": min(1.0, f1 + 0.05),
                "mcc": 2 * f1 - 1,
                "train_index_sha256": train_hash,
                "test_index_sha256": test_hash,
                "train_bank_sha256": train_banks,
                "test_bank_sha256": _hex(f"bank:{heldout}:{seed}"),
            }
            records[(heldout, method, seed)] = record
    return config, records


def _write_run(root: Path, monkeypatch: pytest.MonkeyPatch, *, full: bool):
    from lpcode_v1 import t5

    config, records = _run_data(full=full)
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (root / "folds.jsonl").write_text("synthetic ledger\n", encoding="utf-8")
    monkeypatch.setattr(t5, "_validate_config", lambda value: value)
    monkeypatch.setattr(t5, "_load_records", lambda _path, _config: records)
    return config, records


def test_gate_d_frozen_thresholds() -> None:
    from lpcode_v1.gates_t5 import gate_d

    passed = gate_d({"c": 0.04, "cpp": 0.03, "java": 0.01, "py": 0.02}, 0.025)
    assert passed["passed"] is True
    assert passed["positive_holdouts"] == 4
    assert passed["mean_delta_at_least_0_02"] is True
    assert passed["stronger_mean_delta_at_least_0_03"] is False

    failed = gate_d({"c": 0.04, "cpp": 0.03, "java": -0.01, "py": -0.02}, 0.02)
    assert failed["passed"] is False
    assert failed["positive_holdouts"] == 2


def test_full_summary_is_paired_deterministic_and_manifest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import gates_t5

    root = tmp_path / "full"
    config, _records = _write_run(root, monkeypatch, full=True)
    first = gates_t5.summarize_t5(root)
    artifacts = {
        name: (root / name).read_bytes()
        for name in ("summary.json", "table_d.csv", "table_d.md", "gate_d.json", "manifest.json")
    }
    second = gates_t5.summarize_t5(root)
    assert first["verdict"] is True and second["verdict"] is True
    assert artifacts == {name: (root / name).read_bytes() for name in artifacts}

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "gate_d.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert summary["matrix"] == {
        "complete_cartesian_product": True,
        "expected_records": 48,
        "observed_records": 48,
        "official_gate_expected_records": 48,
    }
    assert gate["status"] == "evaluable"
    assert gate["strict"]["passed"] is True
    assert gate["strict"]["overall_equal_language_mean_delta_f1"] == pytest.approx(0.025)
    assert gate["strict"]["positive_holdouts"] == 4
    assert summary["direction_counts"] == {"positive": 3, "negative": 0, "zero": 0}
    assert summary["paired_mstf_minus_lpcode"]["ci_95"]["replicates"] == 10_000
    assert summary["paired_mstf_minus_lpcode"]["statistical_claim"] == "descriptive"
    assert manifest["gate_c_binding"] == config["gate_c_binding"]
    assert set(manifest["files"]) == {
        "config.json", "folds.jsonl", "summary.json", "table_d.csv", "table_d.md", "gate_d.json"
    }
    for name, spec in manifest["files"].items():
        assert spec["sha256"] == hashlib.sha256((root / name).read_bytes()).hexdigest()


def test_smoke_summary_is_not_gate_evaluable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import gates_t5

    root = tmp_path / "smoke"
    _write_run(root, monkeypatch, full=False)
    report = gates_t5.summarize_t5(root)
    gate = json.loads((root / "gate_d.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "not_evaluable"
    assert gate["status"] == "not_evaluable"
    assert gate["strict"] is None
    assert gate["reasons"]


def test_summary_rejects_cross_method_split_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import gates_t5, t5

    root = tmp_path / "mismatch"
    config, records = _write_run(root, monkeypatch, full=True)
    records[("c", "mstf", 42)]["test_index_sha256"] = _hex("wrong")
    monkeypatch.setattr(t5, "_load_records", lambda _path, _config: records)
    with pytest.raises(ValueError, match="paired T5 hashes differ"):
        gates_t5.summarize_t5(root)
