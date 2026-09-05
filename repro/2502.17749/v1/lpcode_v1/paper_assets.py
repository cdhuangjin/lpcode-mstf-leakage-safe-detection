"""Build registry-bound tables and publication figures from formal result artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from .release_paths import resolve_recorded_path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from .release_figure_qa import require_matplotlib_panel_alignment


TABLES = ("table_01_official_reproduction", "table_02_strict_clean", "table_03_ablation", "table_04_unseen_llm", "table_05_style_attack", "table_06_cross_language", "table_07_mechanism")
FIGURES = ("fig_01_framework", "fig_02_strict_clean", "fig_03_ablation", "fig_04_unseen_llm", "fig_05_style_attack", "fig_06_cross_language", "fig_07_feature_importance")
MSTF_FIGURES = (
    "mstf_fig_01_architecture",
    "mstf_fig_02_protocol",
    "mstf_fig_03_main_results",
    "mstf_fig_04_ablation_mechanism",
    "mstf_fig_05_robustness_boundary",
)
COLORS = ("#26547C", "#E5823A", "#1E8A78", "#A56B93")
NAVY, TEAL, ORANGE, GRAY = "#26547C", "#1E8A78", "#E5823A", "#6B7787"
PALE_NAVY, PALE_TEAL, PALE_ORANGE, PAPER = "#EAF1F8", "#E8F5F2", "#FDF1E5", "#FBFCFE"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pp(value: float) -> str:
    return f"{100 * float(value):+.3f}"


def _write_table(stem: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    markdown = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    markdown.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    stem.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    tex = ["\\begin{tabular}{" + "l" * len(columns) + "}", " \\ ".join(columns) + " \\\\ \\hline"]
    tex.extend(" \\ ".join(str(row[column]) for column in columns) + " \\\\" for row in rows)
    tex.append("\\end{tabular}")
    stem.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def _save(fig: Any, stem: Path) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def _inputs(registry: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    payload = _read(registry)
    paths = {name: resolve_recorded_path(bundle["root"]) for name, bundle in payload["bundles"].items()}
    return payload, paths


def _source_rows(registry: Path) -> dict[str, list[dict[str, Any]]]:
    payload, roots = _inputs(registry)
    digest = hashlib.sha256(registry.read_bytes()).hexdigest()
    a, b, c, d = (_read(roots[key] / f"{key}.json") for key in ("gate_a", "gate_b", "gate_c", "gate_d"))
    a_summary = _read(roots["gate_a"] / "summary.json")
    result_root = registry.parent.parent
    ablation = _read(result_root / "05_mechanism_analysis" / "ablation_summary.json")
    importance = list(csv.DictReader((result_root / "05_mechanism_analysis" / "feature_importance" / "grouped_importance.csv").open(encoding="utf-8")))
    common = {"frozen_registry_sha256": digest}
    return {
        "table_01_official_reproduction": [{"scope": "Formal strict reproduction", "selected_model": "XGB + concat_delta", "macro_f1": f"{a_summary['candidate_ranking'][0]['macro_f1']:.3f}", "source": str(roots['gate_a'] / 'summary.json'), **common}],
        "table_02_strict_clean": [{"language": lang, "delta_f1_pp": _pp(value), "source": str(roots['gate_a'] / 'gate_a.json'), **common} for lang, value in a['strict']['language_deltas'].items()],
        "table_03_ablation": [{"contrast": key, "clean_delta_f1_pp": _pp(ablation['environments']['clean']['overall'][key]['mean_delta_f1']), "unseen_delta_f1_pp": _pp(ablation['environments']['unseen']['overall'][key]['mean_delta_f1']), "source": str(result_root / '05_mechanism_analysis' / 'ablation_summary.json'), **common} for key in ("C1", "C2", "C3", "C4", "C5")],
        "table_04_unseen_llm": [{"heldout_llm": name, "delta_f1_pp": _pp(value), "source": str(roots['gate_b'] / 'gate_b.json'), **common} for name, value in b['strict']['holdout_deltas'].items()],
        "table_05_style_attack": [{"condition": "combined", "mstf_f1": f"{c['strict']['candidate_attacked_f1']:.6f}", "lpcode_f1": f"{c['strict']['baseline_attacked_f1']:.6f}", "advantage_pp": _pp(c['strict']['attacked_f1_advantage']), "source": str(roots['gate_c'] / 'gate_c.json'), **common}],
        "table_06_cross_language": [{"heldout_language": name, "delta_f1_pp": _pp(value), "source": str(roots['gate_d'] / 'gate_d.json'), **common} for name, value in d['strict']['heldout_mean_delta_f1'].items()],
        "table_07_mechanism": [{"environment": row['environment'], "leading_group": row['group'], "permutation_f1_decrease": f"{float(row['permutation_mean']):.6f}", "source": str(result_root / '05_mechanism_analysis' / 'feature_importance' / 'grouped_importance.csv'), **common} for row in _leading_groups(importance)],
    }


def _leading_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if row['environment'] not in best or float(row['permutation_mean']) > float(best[row['environment']]['permutation_mean']): best[row['environment']] = row
    return [best[key] for key in sorted(best)]


def _figures(rows: dict[str, list[dict[str, Any]]], output: Path, registry: Path) -> None:
    for index, (figure_name, table_name) in enumerate(zip(FIGURES[1:], TABLES[1:]), 1):
        entries = rows[table_name]
        labels = [str(row.get('language', row.get('heldout_llm', row.get('heldout_language', row.get('contrast', row.get('condition', row.get('environment', ''))))))) for row in entries]
        numeric_key = next((key for key in entries[0] if key.endswith('_pp') or key in ('permutation_f1_decrease',)), None)
        values = [float(row[numeric_key]) for row in entries] if numeric_key else [float(row.get('mstf_f1', 0)) for row in entries]
        fig, ax = plt.subplots(figsize=(5.2, 2.8)); ax.bar(range(len(values)), values, color=COLORS[index % len(COLORS)]); ax.set_xticks(range(len(values)), labels, rotation=25, ha='right'); ax.set_ylabel(numeric_key.replace('_', ' ') if numeric_key else 'F1'); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); _save(fig, output / figure_name)
    _, roots = _inputs(registry)
    a = _read(roots["gate_a"] / "gate_a.json")
    ablation = _read(registry.parent.parent / "05_mechanism_analysis" / "ablation_summary.json")
    d = _read(roots["gate_d"] / "summary.json")
    def ci_bar(stem: str, labels: list[str], means: list[float], lows: list[float], highs: list[float], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 2.8)); lower = [mean - low for mean, low in zip(means, lows)]; upper = [high - mean for mean, high in zip(means, highs)]; ax.bar(range(len(means)), means, yerr=[lower, upper], capsize=3, color="#0072B2", ecolor="#222222"); ax.axhline(0, color="#444444", linewidth=.7); ax.set_xticks(range(len(means)), labels, rotation=25, ha='right'); ax.set_ylabel(ylabel); ax.set_title("Error bars: 95% seed-cluster bootstrap CI", fontsize=8); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); _save(fig, output / stem)
    languages = list(a["strict"]["language_deltas"])
    ci_bar("fig_02_strict_clean", languages, [100*a["strict"]["language_deltas"][x] for x in languages], [100*a["ci_summary"][x]["low"] for x in languages], [100*a["ci_summary"][x]["high"] for x in languages], "Δ F1 (pp)")
    contrasts = ["C1", "C2", "C3", "C4", "C5"]
    overall = ablation["environments"]["clean"]["overall"]
    ci_bar("fig_03_ablation", contrasts, [100*overall[x]["mean_delta_f1"] for x in contrasts], [100*overall[x]["ci_95"]["low"] for x in contrasts], [100*overall[x]["ci_95"]["high"] for x in contrasts], "Clean Δ F1 (pp)")
    heldout = list(d["paired_mstf_minus_lpcode"]["by_heldout_language"])
    values = d["paired_mstf_minus_lpcode"]["by_heldout_language"]
    ci_bar("fig_06_cross_language", heldout, [100*values[x]["mean_delta_f1"] for x in heldout], [100*values[x]["ci_95"]["low"] for x in heldout], [100*values[x]["ci_95"]["high"] for x in heldout], "Δ F1 (pp)")
    fig, ax = plt.subplots(figsize=(5.2, 2.8)); ax.axis('off'); ax.text(.03,.68,'Human endpoint\n28 features', ha='center', bbox={'boxstyle':'round','fc':'#56B4E9'}); ax.text(.38,.68,'Candidate endpoint\n28 features', ha='center', bbox={'boxstyle':'round','fc':'#E69F00'}); ax.text(.72,.68,'Δ and relative Δ\n56 features', ha='center', bbox={'boxstyle':'round','fc':'#009E73'}); ax.text(.42,.20,'112-D MSTF → fixed XGBoost', ha='center', fontsize=10, fontweight='bold'); _save(fig, output / FIGURES[0])


def _box(ax: Any, x: float, y: float, width: float, height: float, label: str, color: str, *, fontsize: int = 9) -> None:
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=color + "18", edgecolor="none", linewidth=0))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize, color="#18324A")


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], color: str = GRAY) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "shrinkA": 2, "shrinkB": 7})


def _mstf_figures(rows: dict[str, list[dict[str, Any]]], output: Path, registry: Path) -> None:
    """Render the five main-text MSTF figures from frozen evidence and protocol text."""
    _, roots = _inputs(registry)
    gate_a, gate_b, gate_c, gate_d = (_read(roots[key] / f"{key}.json") for key in ("gate_a", "gate_b", "gate_c", "gate_d"))
    result_root = registry.parent.parent
    ablation = _read(result_root / "05_mechanism_analysis" / "ablation_summary.json")
    importance = list(csv.DictReader((result_root / "05_mechanism_analysis" / "feature_importance" / "grouped_importance.csv").open(encoding="utf-8")))
    attack_rows = list(csv.DictReader((result_root / "05_mechanism_analysis" / "attack_decomposition.csv").open(encoding="utf-8")))

    # Figure 1: paired architecture, conceptual only.
    fig, ax = plt.subplots(figsize=(7.2, 3.1), facecolor="white"); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.set_facecolor(PAPER)
    ax.text(.03, .94, "PAIRED INPUT", color=NAVY, fontsize=8, fontweight="bold")
    ax.text(.50, .94, "MULTI-VIEW TRANSITION", color=TEAL, fontsize=8, fontweight="bold")
    ax.text(.89, .94, "DECISION", color=ORANGE, fontsize=8, fontweight="bold", ha="center")
    _box(ax, .03, .67, .16, .18, "Human source code", NAVY)
    _box(ax, .03, .25, .16, .18, "LLM-paraphrased\ncode", NAVY)
    _box(ax, .27, .58, .17, .28, "Shared style\nfeature extractor", TEAL)
    _box(ax, .50, .74, .12, .12, "F_h\n(28-D)", NAVY)
    _box(ax, .50, .55, .12, .12, "F_c\n(28-D)", NAVY)
    _box(ax, .50, .27, .12, .17, "Δ = F_c − F_h\nrelative Δ", TEAL)
    _box(ax, .70, .48, .15, .23, "MSTF\n112-D concat", TEAL, fontsize=10)
    _box(ax, .90, .48, .08, .23, "XGBoost\nscore", ORANGE)
    ax.text(.03, .08, "Absolute endpoint views are retained alongside signed and scale-normalized transitions.", color=GRAY, fontsize=9)
    _save(fig, output / MSTF_FIGURES[0])

    # Figure 2: leakage-safe protocol, conceptual only.
    fig, ax = plt.subplots(figsize=(7.2, 3.0), facecolor="white"); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.set_facecolor(PAPER)
    ax.text(.03, .93, "LEAKAGE-SAFE EVALUATION", color=NAVY, fontsize=8, fontweight="bold")
    _box(ax, .03, .61, .16, .20, "Paired code\nrecords", NAVY)
    _box(ax, .27, .61, .19, .20, "Exact-code + dual-endpoint\nisolation", TEAL)
    _box(ax, .54, .61, .18, .20, "Same split + paired\nevaluation", TEAL)
    _box(ax, .80, .61, .16, .20, "Seed-cluster\nresampling", ORANGE)
    gates = [("A", "Strict clean"), ("B", "Held-out LLM"), ("C", "Style attack"), ("D", "Held-out language")]
    for index, (letter, label) in enumerate(gates):
        x = .04 + index * .24; _box(ax, x, .16, .19, .22, f"Gate {letter}\n{label}", NAVY if index == 0 else TEAL)
    ax.text(.03, .04, "The isolation constraints are locked before every gate and shared across paired comparisons.", color=GRAY, fontsize=9)
    _save(fig, output / MSTF_FIGURES[1])

    # Figure 3: four headline results with CIs only where supplied by the formal gate.
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.75), sharey=True, facecolor="white")
    panels = [
        ("Strict clean", 100 * gate_a["strict"]["mean_delta_f1"], "Gate A", NAVY),
        ("Held-out LLM", 100 * gate_b["strict"]["overall_macro_mean_delta_f1"], "Gate B", TEAL),
        ("Combined attack", 100 * gate_c["strict"]["attacked_f1_advantage"], "Gate C", ORANGE),
        ("Held-out language", 100 * gate_d["strict"]["overall_equal_language_mean_delta_f1"], "Gate D · descriptive", TEAL),
    ]
    for ax, (label, value, note, color) in zip(axes, panels):
        ax.set_facecolor(PAPER); ax.bar([0], [value], width=.62, color=color); ax.axhline(0, color="#48525E", lw=.8); ax.set_xticks([]); ax.set_title(label, fontsize=10, fontweight="bold", pad=8); ax.text(0, value + .45, f"{value:+.3f} pp", ha="center", color=color, fontweight="bold", fontsize=11); ax.text(0, -.95, note, ha="center", color=GRAY, fontsize=8); ax.spines[["top", "right", "bottom"]].set_visible(False); ax.set_ylim(-1.5, 13.5)
    axes[0].set_ylabel("MSTF − baseline ΔF1 (pp)")
    fig.suptitle("Main performance across four evaluation axes", y=1.02, fontsize=12, fontweight="bold"); fig.tight_layout(w_pad=1.2); _save(fig, output / MSTF_FIGURES[2])

    # Figure 4: clean/unseen ablation plus leading grouped permutation contributions.
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.08, 1]}, facecolor="white")
    contrasts = ["C1", "C2", "C3", "C4", "C5"]; x = list(range(len(contrasts))); width = .36
    clean = ablation["environments"]["clean"]["overall"]; unseen = ablation["environments"]["unseen"]["overall"]
    left.bar([v - width / 2 for v in x], [100 * clean[c]["mean_delta_f1"] for c in contrasts], width, label="Clean", color=NAVY)
    left.bar([v + width / 2 for v in x], [100 * unseen[c]["mean_delta_f1"] for c in contrasts], width, label="Held-out LLM", color=TEAL)
    left.set_facecolor(PAPER); left.axhline(0, color="#48525E", lw=.8); left.set_xticks(x, contrasts); left.set_ylabel("ΔF1 vs. reference (pp)"); left.set_title("Controlled representation contrasts", loc="left", fontweight="bold"); left.legend(frameon=False, fontsize=8, loc="upper left"); left.grid(axis="y", color="#D9E1E8", linewidth=.65); left.set_axisbelow(True)
    env_order = ["clean", "unseen_llm", "combined_attack", "cross_language"]; labels = ["Clean", "Unseen", "Attack", "Cross-lang."]
    maxima = [max((row for row in importance if row["environment"] == env), key=lambda row: float(row["permutation_mean"])) for env in env_order]
    values = [100 * float(row["permutation_mean"]) for row in maxima]; right.set_facecolor(PAPER); right.barh(labels, values, color=[NAVY, TEAL, ORANGE, TEAL], height=.68); right.invert_yaxis(); right.set_xlabel("Permutation ΔF1 (pp)"); right.set_title("Exploratory feature ranking", loc="left", fontweight="bold"); right.grid(axis="x", color="#D9E1E8", linewidth=.65); right.set_axisbelow(True); right.set_xlim(0, max(values) + .35)
    feature_labels = [row["group"].replace("relative_delta:structural_syntax", "rel. Δ · structural/syntax").replace("delta:original_style", "Δ · original style") for row in maxima]
    right.set_yticks(range(len(labels)), [f"{environment}\n{feature}" for environment, feature in zip(labels, feature_labels)], fontsize=7.2)
    fig.text(.5, .01, "High rank does not imply a large independent gain; attribution is descriptive, not causal.", ha="center", color=GRAY, fontsize=8.5); fig.tight_layout(rect=(0, .07, 1, 1), w_pad=2.4); _save(fig, output / MSTF_FIGURES[3])

    # Figure 5: attack degradation plus compact generalization boundary summaries.
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.28, .72]}, facecolor="white")
    conditions = ["comment_removal", "identifier_rename", "format_normalization", "comment_injection", "combined"]
    condition_labels = ["Comment\nremoval", "Identifier\nrename", "Format\nnormalization", "Comment\ninjection", "Combined"]
    methods = [("lpcode_original", "LPcode", GRAY), ("mstf", "MSTF", TEAL)]
    for method, label, color in methods:
        values = []
        for condition in conditions:
            selected = [row for row in attack_rows if row["condition"] == condition and row["method"] == method]
            values.append(100 * sum(float(row["absolute_drop"]) for row in selected) / len(selected))
        left.plot(range(len(conditions)), values, marker="o", linewidth=2.2, label=label, color=color)
    left.set_facecolor(PAPER); left.set_xticks(range(len(conditions)), condition_labels); left.set_ylabel("Mean F1 drop (pp)"); left.set_title("Style-transformation robustness", loc="left", fontweight="bold"); left.grid(axis="y", color="#D9E1E8", linewidth=.65); left.set_axisbelow(True); left.set_ylim(0, 7.0)
    left.legend(handles=[Patch(facecolor=GRAY, edgecolor="none", label="LPcode"), Patch(facecolor=TEAL, edgecolor="none", label="MSTF")], frameon=False, loc="upper left", ncol=2, handlelength=.9, columnspacing=1.0)
    combined_mstf = 100 * gate_c["strict"]["candidate_drop"]; combined_lpcode = 100 * gate_c["strict"]["baseline_drop"]
    names = ["Unseen LLM", "Cross-language"]; values = [100 * gate_b["strict"]["overall_macro_mean_delta_f1"], 100 * gate_d["strict"]["overall_equal_language_mean_delta_f1"]]
    right.set_facecolor(PAPER); right.bar(names, values, color=[TEAL, TEAL], width=.72); right.axhline(0, color="#48525E", lw=.8); right.set_ylabel("MSTF − baseline ΔF1 (pp)"); right.set_title("Generalization boundary", loc="left", fontweight="bold", pad=8); right.grid(axis="y", color="#D9E1E8", linewidth=.65); right.set_axisbelow(True)
    for index, value in enumerate(values): right.text(index, value / 2, f"{value:+.3f} pp", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
    fig.text(.5, .01, f"Combined drop: LPcode {combined_lpcode:.2f} pp; MSTF {combined_mstf:.2f} pp. Boundary summaries are limited to the evaluated generators, languages, transformations, and classifiers.", ha="center", color=GRAY, fontsize=8.2); fig.tight_layout(rect=(0, .08, 1, 1), w_pad=2.4); _save(fig, output / MSTF_FIGURES[4])


def build_paper_assets(registry: Path, output: Path | None = None, *, rows: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Build format-independent tables and registry-provenanced numeric metadata."""
    registry = registry.resolve()
    digest = hashlib.sha256(registry.read_bytes()).hexdigest(); output = output or registry.parent
    output.mkdir(parents=True, exist_ok=True); tables = output / "tables"; figures = output / "figures"; tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    data = rows or _source_rows(registry)
    numeric = []
    for name, table_rows in data.items():
        bound = [{**row, "frozen_registry_sha256": digest} for row in table_rows]; _write_table(tables / name, bound); numeric.append({"name": name, "source": str(bound[0].get('source', registry)), "frozen_registry_sha256": digest})
    if rows is None:
        _figures(data, figures, registry)
        _mstf_figures(data, figures, registry)
        numeric.extend({"name": name, "source": str(registry), "frozen_registry_sha256": digest} for name in (*FIGURES, *MSTF_FIGURES))
    (output / "asset_provenance.json").write_text(json.dumps({"frozen_registry_sha256": digest, "numeric": numeric}, indent=2) + "\n", encoding="utf-8")
    return {"numeric": numeric}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('build',)); parser.add_argument('--registry', type=Path, required=True); args = parser.parse_args(); build_paper_assets(args.registry)


if __name__ == '__main__': main()
