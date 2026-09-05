from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pytest


def _hold_gate_output_lock(path: str, ready, release) -> None:
    from lpcode_v1.t1 import _exclusive_output_lock

    with _exclusive_output_lock(Path(path)):
        ready.set()
        release.wait(15)


def _write_run(
    root: Path,
    *,
    languages: tuple[str, ...] = ("c", "cpp", "java", "py"),
    seeds: tuple[int, ...] = (11, 22),
    folds: int = 2,
    representations: tuple[str, ...] = ("concat", "delta", "concat_delta", "full"),
    models: tuple[str, ...] = ("mlp",),
    deltas: dict[tuple[str, str, str], float] | None = None,
    delta_values: dict[tuple[str, str, str, int, int], float] | None = None,
) -> None:
    """Write a complete, runner-schema-valid synthetic evaluation matrix."""
    from lpcode_v1 import t1

    root.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Path] = {}
    for language in languages:
        dataset = root / f"{language}.jsonl"
        dataset.write_text('{"label":0}\n', encoding="utf-8")
        datasets[language] = dataset
    config = t1._build_config(languages, seeds, folds, representations, models, None, datasets)
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    d = deltas or {}
    by_fold = delta_values or {}
    records: list[dict[str, object]] = []
    dimensions = {"concat": 20, "delta": 10, "concat_delta": 30, "full": 40}
    for language in languages:
        for representation in representations:
            for model in models:
                for seed_index, seed in enumerate(seeds):
                    for fold in range(folds):
                        base = 0.50 + seed_index * 0.02 + fold * 0.01
                        f1 = base if representation == "concat" else base + by_fold.get(
                            (representation, model, language, seed, fold), d.get((representation, model, language), 0.0)
                        )
                        records.append(
                            {
                                "schema_version": 1,
                                "config_id": config["config_id"],
                                "language": language,
                                "representation": representation,
                                "model": model,
                                "seed": seed,
                                "fold": fold,
                                "leakage_count": 0,
                                "train_index_sha256": "a" * 64,
                                "test_index_sha256": "b" * 64,
                                "feature_dimensions": dimensions[representation],
                                "train_unique_sources": 2,
                                "test_unique_sources": 2,
                                "f1": f1,
                                "precision": f1,
                                "recall": f1,
                                "auroc": f1,
                                "mcc": f1 * 2 - 1,
                                "fit_seconds": 0.0,
                                "predict_seconds": 0.0,
                                "train_rows": 4,
                                "test_rows": 4,
                                "train_class_counts": {"0": 2, "1": 2},
                                "test_class_counts": {"0": 2, "1": 2},
                            }
                        )
    (root / "folds.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _write_strict_run(
    root: Path,
    *,
    languages: tuple[str, ...] = ("c", "cpp", "java", "py"),
    seeds: tuple[int, ...] = (11, 22),
    folds: int = 2,
    representations: tuple[str, ...] = ("concat", "delta", "concat_delta", "full"),
    models: tuple[str, ...] = ("mlp", "xgb"),
    deltas: dict[tuple[str, str, str], float] | None = None,
) -> None:
    """Write a complete strict-origin schema-v2 matrix without fitting models."""
    from lpcode_v1 import t1_strict, t3

    root.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Path] = {}
    for language in languages:
        dataset = root / f"{language}.jsonl"
        dataset.write_text('{"label":1}\n', encoding="utf-8")
        datasets[language] = dataset
    config = t1_strict._build_config(
        languages, seeds, folds, representations, models, None, datasets
    )
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    d = deltas or {}
    dimensions = {"concat": 20, "delta": 10, "concat_delta": 30, "full": 40}
    llm_counts = {
        source: {"0": 2, "1": 2} for source in t3.LLM_SOURCES
    }
    records: list[dict[str, object]] = []
    for language in languages:
        for representation in representations:
            for model in models:
                for seed_index, seed in enumerate(seeds):
                    for fold in range(folds):
                        base = 0.50 + seed_index * 0.02 + fold * 0.01
                        f1 = base if representation == "concat" else base + d.get(
                            (representation, model, language), 0.0
                        )
                        record: dict[str, object] = {
                            "schema_version": t1_strict.FOLD_SCHEMA_VERSION,
                            "config_id": config["config_id"],
                            "language": language,
                            "representation": representation,
                            "model": model,
                            "seed": seed,
                            "fold": fold,
                            "split_protocol": t1_strict.SPLIT_PROTOCOL_VERSION,
                            "pair_protocol": t3.PAIR_PROTOCOL_VERSION,
                            "component_protocol": t3.COMPONENT_PROTOCOL_VERSION,
                            "leakage_count": 0,
                            "endpoint_leakage_count": 0,
                            "content_leakage_count": 0,
                            "negative_component_violation_count": 0,
                            "train_index_sha256": hashlib.sha256(
                                f"train:{language}:{seed}:{fold}".encode()
                            ).hexdigest(),
                            "test_index_sha256": hashlib.sha256(
                                f"test:{language}:{seed}:{fold}".encode()
                            ).hexdigest(),
                            "feature_dimensions": dimensions[representation],
                            "train_unique_sources": 8,
                            "test_unique_sources": 8,
                            "train_unique_code_hashes": 8,
                            "test_unique_code_hashes": 8,
                            "train_unique_components": 8,
                            "test_unique_components": 8,
                            "train_llm_label_counts": llm_counts,
                            "test_llm_label_counts": llm_counts,
                            "f1": f1,
                            "precision": f1,
                            "recall": f1,
                            "auroc": f1,
                            "mcc": f1 * 2 - 1,
                            "fit_seconds": 0.0,
                            "predict_seconds": 0.0,
                            "train_rows": 16,
                            "test_rows": 16,
                            "train_class_counts": {"0": 8, "1": 8},
                            "test_class_counts": {"0": 8, "1": 8},
                        }
                        record["record_sha256"] = t1_strict._record_sha256(record)
                        records.append(record)
    (root / "folds.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_gate_a_passes_and_reports_conservative_fields() -> None:
    from lpcode_v1.gates import gate_a

    result = gate_a({"c": 0.015, "cpp": 0.018, "java": 0.012, "py": 0.017})

    assert result["passed"] is True
    assert result["relaxed_passed"] is True
    assert result["languages_won"] == 4
    assert result["worst_language_delta_f1"] == 0.012
    assert result["mean_delta_f1"] == pytest.approx(0.0155)


def test_gate_a_threshold_ties_and_single_language_effect() -> None:
    from lpcode_v1.gates import gate_a

    assert gate_a({"c": -0.005, "cpp": 0.0, "java": 0.0, "py": 0.0})["passed"] is True
    failed = gate_a({"c": -0.01, "cpp": -0.01, "java": -0.01, "py": 0.03})
    assert failed["passed"] is False
    assert failed["relaxed_passed"] is False
    assert failed["languages_won"] == 1


def test_gate_a_exact_strict_and_relaxed_boundaries() -> None:
    from lpcode_v1.gates import gate_a

    strict_edge = gate_a({"c": -0.01, "cpp": 0.0, "java": 0.0, "py": 0.0})
    assert strict_edge["passed"] is True
    assert strict_edge["languages_won"] == 3
    assert strict_edge["thresholds"] == {
        "strict": {"mean_delta_f1": -0.005, "minimum_languages_won": 3, "worst_language_delta_f1": -0.01},
        "relaxed": {"mean_delta_f1": -0.01, "minimum_languages_won": 3, "worst_language_delta_f1": -0.015},
    }
    assert gate_a({"c": -0.0100000001, "cpp": 0.0, "java": 0.0, "py": 0.0})["passed"] is False

    relaxed_edge = gate_a({"c": -0.015, "cpp": 0.0, "java": 0.0, "py": 0.0})
    assert relaxed_edge["passed"] is False
    assert relaxed_edge["relaxed_passed"] is True
    assert relaxed_edge["languages_won"] == 3
    below_relaxed = gate_a({"c": -0.0150000001, "cpp": 0.0, "java": 0.0, "py": 0.0})
    assert below_relaxed["passed"] is False
    assert below_relaxed["relaxed_passed"] is False


@pytest.mark.parametrize(
    "values",
    [
        {"c": 0.0, "cpp": 0.0, "java": 0.0},
        {"c": 0.0, "cpp": 0.0, "java": 0.0, "py": 0.0, "go": 0.0},
        {"c": True, "cpp": 0.0, "java": 0.0, "py": 0.0},
        {"c": math.nan, "cpp": 0.0, "java": 0.0, "py": 0.0},
        {"c": 1.01, "cpp": 0.0, "java": 0.0, "py": 0.0},
    ],
)
def test_gate_a_rejects_invalid_axes_and_values(values: dict[str, object]) -> None:
    from lpcode_v1.gates import gate_a

    with pytest.raises(ValueError):
        gate_a(values)  # type: ignore[arg-type]


def test_summary_aggregates_pairs_bootstraps_and_writes_verifiable_artifacts(tmp_path: Path) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "complete"
    _write_run(
        root,
        deltas={
            **{("delta", "mlp", language): 0.015 for language in ("c", "cpp", "java", "py")},
            **{("concat_delta", "mlp", language): 0.010 for language in ("c", "cpp", "java", "py")},
            **{("full", "mlp", language): 0.012 for language in ("c", "cpp", "java", "py")},
        },
    )
    before = {name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")}

    report = summarize_t1(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    cell = summary["cell_summaries"]["c"]["concat"]["mlp"]
    assert cell["n"] == 4
    assert cell["f1_mean"] == pytest.approx(0.515)
    assert cell["f1_std"] == pytest.approx(math.sqrt(1 / 6000))
    paired = summary["paired_deltas"]["c"]["delta"]["mlp"]
    assert paired["n"] == 4
    assert paired["mean_delta_f1"] == pytest.approx(0.015)
    assert paired["std_delta_f1"] == 0.0
    assert paired["ci_95"]["method"] == "seed_cluster_bootstrap"
    assert paired["ci_95"]["replicates"] >= 10_000
    assert paired["paired_keys"] == [["c", "mlp", 11, 0], ["c", "mlp", 11, 1], ["c", "mlp", 22, 0], ["c", "mlp", 22, 1]]
    assert gate["selected_candidate"]["representation"] == "delta"
    assert gate["status"] == "evaluable"
    assert gate["strict"]["passed"] is True
    assert report["gate_a_path"] == str(root / "gate_a.json")
    csv_text = (root / "table_a.csv").read_text(encoding="utf-8")
    markdown_text = (root / "table_a.md").read_text(encoding="utf-8")
    assert "C" in csv_text and "51.50% ± 1.29%" in csv_text
    assert "MLP" in markdown_text and "51.50% ± 1.29%" in markdown_text
    assert manifest["schema_version"] == 1
    assert manifest["generated_at_utc"]
    assert set(manifest["method_versions"]) == {
        "summary",
        "bootstrap",
        "gate_a",
        "protocol",
        "leakage_definition",
    }
    assert set(manifest["files"]) == {"config.json", "folds.jsonl", "summary.json", "table_a.csv", "table_a.md", "gate_a.json"}
    for name, details in manifest["files"].items():
        assert name != "manifest.json"
        path = root / name
        assert details["bytes"] == path.stat().st_size
        assert details["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert {name: (root / name).read_bytes() for name in before} == before
    stable_bytes = {name: (root / name).read_bytes() for name in ("summary.json", "table_a.csv", "table_a.md", "gate_a.json")}
    second = summarize_t1(root)
    assert second["selected_candidate"] == report["selected_candidate"]
    second_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert second_summary["candidate_ranking"] == summary["candidate_ranking"]
    assert second_summary["paired_deltas"]["c"]["delta"]["mlp"]["ci_95"] == paired["ci_95"]
    assert {name: (root / name).read_bytes() for name in stable_bytes} == stable_bytes
    second_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert {key: value for key, value in second_manifest.items() if key != "generated_at_utc"} == {
        key: value for key, value in manifest.items() if key != "generated_at_utc"
    }
    assert not (root / "cache").exists()


def test_summary_dispatches_strict_origin_schema_v2_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "strict"
    _write_strict_run(root)
    before = {name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")}

    report = summarize_t1(root)

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "not_evaluable"
    assert summary["config"]["config_id"] == json.loads(
        (root / "config.json").read_text(encoding="utf-8")
    )["config_id"]
    assert {name: (root / name).read_bytes() for name in before} == before


def test_summary_records_strict_protocol_and_leakage_definition_in_all_artifacts(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t1_strict
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "strict-methodology"
    _write_strict_run(root)

    summarize_t1(root)

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_definition = {
        "version": "dual-endpoint-exact-code-v2",
        "formula": "leakage_count = endpoint_leakage_count + content_leakage_count",
        "train_test_disjoint_on": ["origin_endpoint_id", "exact_code_sha256"],
        "negative_pair_constraint": "human_component_id != candidate_component_id",
        "required_zero_fields": [
            "leakage_count",
            "endpoint_leakage_count",
            "content_leakage_count",
            "negative_component_violation_count",
        ],
    }
    for artifact in (summary, gate):
        assert artifact["protocol_version"] == t1_strict.SPLIT_PROTOCOL_VERSION
        assert artifact["leakage_definition"] == expected_definition
    assert manifest["method_versions"]["protocol"] == t1_strict.SPLIT_PROTOCOL_VERSION
    assert (
        manifest["method_versions"]["leakage_definition"]
        == expected_definition["version"]
    )


def test_summary_records_legacy_protocol_without_weakening_v1_validation(
    tmp_path: Path,
) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "legacy-methodology"
    _write_run(root)

    summarize_t1(root)

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for artifact in (summary, gate):
        assert artifact["protocol_version"] == "human-source-grouped-v1"
        assert artifact["leakage_definition"] == {
            "version": "single-human-source-id-v1",
            "formula": "leakage_count = size(train human_source_id intersect test human_source_id)",
            "train_test_disjoint_on": ["human_source_id"],
            "required_zero_fields": ["leakage_count"],
        }
    assert manifest["method_versions"]["protocol"] == "human-source-grouped-v1"
    assert manifest["method_versions"]["leakage_definition"] == "single-human-source-id-v1"


def test_strict_summary_keeps_global_selection_and_unchanged_gate_thresholds(
    tmp_path: Path,
) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "strict-global"
    _write_strict_run(
        root,
        seeds=(42, 123, 2024),
        folds=5,
        deltas={
            ("delta", "mlp", "c"): 0.15,
            **{
                ("full", "mlp", language): 0.04
                for language in ("c", "cpp", "java", "py")
            },
        },
    )

    result = summarize_t1(root)

    gate = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
    assert result["selected_candidate"] == {"representation": "full", "model": "mlp"}
    assert gate["strict"]["thresholds"] == {
        "strict": {
            "mean_delta_f1": -0.005,
            "minimum_languages_won": 3,
            "worst_language_delta_f1": -0.01,
        },
        "relaxed": {
            "mean_delta_f1": -0.01,
            "minimum_languages_won": 3,
            "worst_language_delta_f1": -0.015,
        },
    }


def test_strict_summary_marks_smoke_or_cherry_picked_axes_not_evaluable(
    tmp_path: Path,
) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "strict-smoke"
    _write_strict_run(
        root,
        seeds=(42,),
        folds=2,
        models=("mlp",),
    )

    report = summarize_t1(root)

    gate = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "not_evaluable"
    assert gate["status"] == "not_evaluable"
    assert any("strict-origin Gate A requires" in reason for reason in gate["reasons"])


def test_summary_cluster_bootstrap_and_seed_consistency_use_heterogeneous_fold_deltas(tmp_path: Path) -> None:
    from lpcode_v1.gates import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, summarize_t1

    root = tmp_path / "heterogeneous"
    values = {
        ("full", "mlp", "c", 11, 0): 0.01, ("full", "mlp", "c", 11, 1): 0.02,
        ("full", "mlp", "cpp", 11, 0): -0.005, ("full", "mlp", "cpp", 11, 1): 0.005,
        ("full", "mlp", "java", 11, 0): 0.03, ("full", "mlp", "java", 11, 1): 0.01,
        ("full", "mlp", "py", 11, 0): 0.015, ("full", "mlp", "py", 11, 1): 0.025,
        ("full", "mlp", "c", 22, 0): 0.04, ("full", "mlp", "c", 22, 1): 0.02,
        ("full", "mlp", "cpp", 22, 0): 0.0, ("full", "mlp", "cpp", 22, 1): -0.02,
        ("full", "mlp", "java", 22, 0): 0.05, ("full", "mlp", "java", 22, 1): 0.03,
        ("full", "mlp", "py", 22, 0): 0.02, ("full", "mlp", "py", 22, 1): 0.04,
    }
    _write_run(root, delta_values=values)

    summarize_t1(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
    chosen = summary["candidate_ranking"][0]
    c_clusters = {11: [0.01, 0.02], 22: [0.04, 0.02]}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    clustered = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.choice([11, 22], size=2, replace=True)
        clustered[index] = np.mean([value for seed in sampled_seeds for value in c_clusters[int(seed)]])
    expected_ci = (float(np.quantile(clustered, 0.025)), float(np.quantile(clustered, 0.975)))
    row_rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = [value for cluster in c_clusters.values() for value in cluster]
    rowwise = np.asarray([np.mean(row_rng.choice(rows, size=len(rows), replace=True)) for _ in range(BOOTSTRAP_REPLICATES)])
    ci = summary["paired_deltas"]["c"]["full"]["mlp"]["ci_95"]
    assert (ci["low"], ci["high"]) == pytest.approx(expected_ci)
    assert (ci["low"], ci["high"]) != pytest.approx((float(np.quantile(rowwise, 0.025)), float(np.quantile(rowwise, 0.975))))
    assert chosen["representation"] == "full"
    assert chosen["per_seed_macro_deltas"] == {"11": pytest.approx(0.01375), "22": pytest.approx(0.0225)}
    assert chosen["languages_nonnegative"] == 3
    assert chosen["seeds_nonnegative"] == 2
    assert gate["per_seed_deltas"] == chosen["per_seed_macro_deltas"]


def test_summary_chooses_one_global_candidate_not_per_language_cherrypick(tmp_path: Path) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "global"
    _write_run(
        root,
        deltas={
            ("delta", "mlp", "c"): 0.15,
            ("delta", "mlp", "cpp"): 0.0,
            ("delta", "mlp", "java"): 0.0,
            ("delta", "mlp", "py"): 0.0,
            **{("full", "mlp", language): 0.04 for language in ("c", "cpp", "java", "py")},
        },
    )

    result = summarize_t1(root)

    assert result["selected_candidate"] == {"representation": "full", "model": "mlp"}
    gate = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
    assert gate["language_deltas"] == {language: pytest.approx(0.04) for language in ("c", "cpp", "java", "py")}


def test_summary_uses_fixed_representation_then_model_tie_break(tmp_path: Path) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "ties"
    _write_run(root, models=("xgb", "mlp"))

    result = summarize_t1(root)

    assert result["selected_candidate"] == {"representation": "delta", "model": "mlp"}


def test_summary_rejects_incomplete_matrix_and_marks_subset_not_evaluable(tmp_path: Path) -> None:
    from lpcode_v1.gates import summarize_t1

    incomplete = tmp_path / "incomplete"
    _write_run(incomplete)
    lines = (incomplete / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    (incomplete / "folds.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        summarize_t1(incomplete)

    subset = tmp_path / "subset"
    _write_run(subset, languages=("c",))
    summarize_t1(subset)
    gate = json.loads((subset / "gate_a.json").read_text(encoding="utf-8"))
    assert gate["status"] == "not_evaluable"
    assert gate["reasons"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: (root / "folds.jsonl").write_text((root / "folds.jsonl").read_text(encoding="utf-8") * 2, encoding="utf-8"), "duplicate"),
        (lambda root: (root / "folds.jsonl").write_text("not-json\n", encoding="utf-8"), "malformed"),
        (lambda root: (root / "folds.jsonl").write_text((root / "folds.jsonl").read_text(encoding="utf-8").replace('"language": "c"', '"language": "go"', 1), encoding="utf-8"), "outside"),
    ],
)
def test_summary_rejects_duplicate_malformed_or_out_of_axis_records(tmp_path: Path, mutate, message: str) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "invalid"
    _write_run(root)
    mutate(root)

    with pytest.raises(ValueError, match=message):
        summarize_t1(root)


def test_summary_rejects_schema_valid_pair_with_different_split_hashes(tmp_path: Path) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "split-mismatch"
    _write_run(root)
    folds = root / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if (record["language"], record["representation"], record["model"], record["seed"], record["fold"]) == ("c", "full", "mlp", 11, 0):
            record["train_index_sha256"] = "c" * 64
            break
    folds.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="paired split hashes"):
        summarize_t1(root)
    assert not (root / "summary.json").exists()


def test_strict_summary_requires_pair_hash_reuse_across_all_models_and_representations(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t1_strict
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "strict-pair-hash-mismatch"
    _write_strict_run(root)
    folds = root / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if (
            record["language"] == "c"
            and record["model"] == "xgb"
            and record["seed"] == 11
            and record["fold"] == 0
        ):
            record["train_index_sha256"] = "c" * 64
            record["test_index_sha256"] = "d" * 64
            record["record_sha256"] = t1_strict._record_sha256(record)
    folds.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict-origin pair split hashes"):
        summarize_t1(root)
    assert not (root / "summary.json").exists()


def test_strict_summary_rejects_nonzero_declared_leakage(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t1_strict
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "strict-leakage"
    _write_strict_run(root)
    folds = root / "folds.jsonl"
    records = [json.loads(line) for line in folds.read_text(encoding="utf-8").splitlines()]
    records[0]["leakage_count"] = 1
    records[0]["endpoint_leakage_count"] = 1
    records[0]["record_sha256"] = t1_strict._record_sha256(records[0])
    folds.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict-origin leakage"):
        summarize_t1(root)
    assert not (root / "summary.json").exists()


def test_strict_summary_rejects_protocol_config_and_record_digest_mismatches(
    tmp_path: Path,
) -> None:
    from lpcode_v1 import t1_strict
    from lpcode_v1.gates import summarize_t1

    protocol_root = tmp_path / "protocol"
    _write_strict_run(protocol_root)
    protocol_folds = protocol_root / "folds.jsonl"
    protocol_records = [
        json.loads(line)
        for line in protocol_folds.read_text(encoding="utf-8").splitlines()
    ]
    protocol_records[0]["pair_protocol"] = "wrong-pair-protocol"
    protocol_records[0]["record_sha256"] = t1_strict._record_sha256(
        protocol_records[0]
    )
    protocol_folds.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in protocol_records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema/config mismatch"):
        summarize_t1(protocol_root)

    config_protocol_root = tmp_path / "config-protocol"
    _write_strict_run(config_protocol_root)
    config_protocol_path = config_protocol_root / "config.json"
    config_protocol = json.loads(
        config_protocol_path.read_text(encoding="utf-8")
    )
    config_protocol["split_protocol"] = "wrong-split-protocol"
    config_protocol["config_id"] = hashlib.sha256(
        t1_strict._canonical_json(
            {key: value for key, value in config_protocol.items() if key != "config_id"}
        )
    ).hexdigest()
    config_protocol_path.write_text(
        json.dumps(config_protocol, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid strict-origin run config"):
        summarize_t1(config_protocol_root)

    config_root = tmp_path / "config"
    _write_strict_run(config_root)
    config_path = config_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["config_id"] = "0" * 64
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid strict-origin run config"):
        summarize_t1(config_root)

    digest_root = tmp_path / "record-digest"
    _write_strict_run(digest_root)
    digest_folds = digest_root / "folds.jsonl"
    digest_records = [
        json.loads(line)
        for line in digest_folds.read_text(encoding="utf-8").splitlines()
    ]
    digest_records[0]["f1"] = 0.6
    digest_folds.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in digest_records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="record digest"):
        summarize_t1(digest_root)

    for root in (protocol_root, config_protocol_root, config_root, digest_root):
        assert not (root / "summary.json").exists()


def test_strict_summary_only_never_calls_evaluator_or_mutates_run_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpcode_v1 import t1_strict
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "strict-summary-only"
    _write_strict_run(root)
    before = {name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")}
    monkeypatch.setattr(
        t1_strict, "evaluate_fold", lambda *args: pytest.fail("summary fit a model")
    )

    summarize_t1(root)

    assert {name: (root / name).read_bytes() for name in before} == before
    assert not (root / "cache").exists()


def test_summary_respects_runner_output_lock_and_writes_no_partial_artifacts(
    tmp_path: Path,
) -> None:
    from lpcode_v1.gates import summarize_t1

    root = tmp_path / "locked-summary"
    _write_run(root)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_gate_output_lock, args=(str(root), ready, release)
    )
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(ValueError, match="already locked"):
            summarize_t1(root)
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    assert not (root / "summary.json").exists()
    assert not (root / "manifest.json").exists()


def test_t1_summarize_only_never_calls_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lpcode_v1 import t1

    root = tmp_path / "run"
    _write_run(root, languages=("c",))
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: pytest.fail("summary fit a model"))
    monkeypatch.setattr(sys, "argv", ["t1", "--summarize-only", "--output-root", str(root)])

    t1.main()

    assert (root / "summary.json").is_file()


def test_gates_main_never_calls_evaluator_or_mutates_run_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lpcode_v1 import gates, t1

    root = tmp_path / "gates-cli"
    _write_run(root, languages=("c",))
    before = {name: (root / name).read_bytes() for name in ("config.json", "folds.jsonl")}
    monkeypatch.setattr(t1, "evaluate_fold", lambda *args: pytest.fail("gates summary fit a model"))
    monkeypatch.setattr(sys, "argv", ["gates", "--output-root", str(root)])

    gates.main()

    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "not_evaluable"
    assert {name: (root / name).read_bytes() for name in before} == before
    assert not (root / "cache").exists()
