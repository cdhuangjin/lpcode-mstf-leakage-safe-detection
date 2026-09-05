"""Registry-bound paper asset contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lpcode_v1.paper_assets import build_paper_assets


def test_numeric_assets_embed_registry_digest_and_source_paths(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"bundles": {}, "schema_version": 1}), encoding="utf-8")
    digest = hashlib.sha256(registry.read_bytes()).hexdigest()
    assets = build_paper_assets(registry, tmp_path / "assets", rows={"table_01_official_reproduction": [{"value": 1.0}]})

    assert all(item["frozen_registry_sha256"] == digest for item in assets["numeric"])
    assert not any("smoke" in item["source"].lower() for item in assets["numeric"])


def test_mstf_figure_contract_creates_png_and_pdf(tmp_path: Path) -> None:
    """The five manuscript figures must be emitted in screen and print formats."""
    registry = Path(__file__).resolve().parents[4] / "results" / "06_paper_assets" / "frozen_result_registry.json"
    output = tmp_path / "assets"
    build_paper_assets(registry, output)

    for stem in (
        "mstf_fig_01_architecture",
        "mstf_fig_02_protocol",
        "mstf_fig_03_main_results",
        "mstf_fig_04_ablation_mechanism",
        "mstf_fig_05_robustness_boundary",
    ):
        assert (output / "figures" / f"{stem}.png").is_file()
        assert (output / "figures" / f"{stem}.pdf").is_file()
