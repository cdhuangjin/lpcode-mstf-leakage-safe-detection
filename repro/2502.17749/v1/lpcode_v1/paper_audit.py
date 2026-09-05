"""Registry-grounded numerical and overclaim audit for the draft manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from .release_paths import resolve_recorded_path
from typing import Any


PROHIBITED_PATTERNS: dict[str, str] = {
    "legacy_gate_a": r"legacy\s+gate\s*a",
    "smoke": r"\bsmoke\b",
    "preflight": r"\bpreflight\b",
    "statistical_significance": r"statistically\s+significant",
    "folds_as_datasets": r"(?:folds?\s+(?:are|as)\s+independent\s+datasets|independent\s+datasets?\s+(?:are|across)\s+folds?)",
    "universal_claim": r"\b(?:universal|all[- ]llm|all[- ]language|attack[- ]proof)\b",
}


def audit_text(manuscript_text: str) -> dict[str, Any]:
    """Return named prohibited-language hits without silently normalizing prose."""
    violations = []
    for name, pattern in PROHIBITED_PATTERNS.items():
        matches = list(re.finditer(pattern, manuscript_text, re.IGNORECASE))
        if name == "universal_claim":
            # A nearby explicit negation denotes an acknowledged limitation,
            # e.g. "does not claim a universal detector".
            def _is_negated(match: re.Match[str]) -> bool:
                prefix = manuscript_text[:match.start()]
                sentence = re.split(r"[.!?\n]", prefix)[-1]
                return bool(re.search(r"\b(?:not|no|without)\b", sentence, re.IGNORECASE))
            matches = [match for match in matches if not _is_negated(match)]
        if matches:
            violations.append(name)
    return {"violations": violations}


def audit_numbers(manuscript_text: str, canonical: dict[str, float], decimals: int = 2) -> dict[str, Any]:
    """Check percentages in prose against registered display values.

    Only percentages are audited here: unadorned counts, dimensions, seeds and
    equation labels are intentionally not mistaken for performance claims.
    """
    if decimals < 0:
        raise ValueError("decimals must be nonnegative")
    allowed = [float(value) for value in canonical.values()]
    tolerance = 0.5 * (10 ** -decimals) + 1e-12
    mismatches = []
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%", manuscript_text):
        value = float(match.group(1))
        trailing = manuscript_text[match.end():match.end() + 40]
        # Confidence levels are protocol notation, not a reported effect.
        if re.match(r"\s*(?:seed[- ]cluster\s+)?CI\b", trailing, re.IGNORECASE):
            continue
        if not any(abs(value - expected) <= tolerance for expected in allowed):
            mismatches.append({"observed_percent": value, "offset": match.start(), "context": manuscript_text[max(0, match.start() - 35):match.end() + 35]})
    language = audit_text(manuscript_text)
    return {"status": "PASS" if not mismatches and not language["violations"] else "FAIL", "major_mismatches": mismatches, **language}


def canonical_percentages(registry_path: Path) -> dict[str, float]:
    """Extract the four registered headline percentage values from Gate JSON."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    bundles = registry.get("bundles", {})
    paths = {name: resolve_recorded_path(bundle["files"][f"gate_{name[-1]}.json"]["path"]) for name, bundle in bundles.items()}
    gate_a = json.loads(paths["gate_a"].read_text(encoding="utf-8"))
    gate_b = json.loads(paths["gate_b"].read_text(encoding="utf-8"))
    gate_c = json.loads(paths["gate_c"].read_text(encoding="utf-8"))
    gate_d = json.loads(paths["gate_d"].read_text(encoding="utf-8"))
    return {
        "strict_clean_pp": 100 * float(gate_a["strict"]["mean_delta_f1"]),
        "unseen_llm_pp": 100 * float(gate_b["strict"]["overall_macro_mean_delta_f1"]),
        "combined_attack_pp": 100 * float(gate_c["strict"]["attacked_f1_advantage"]),
        "relative_drop_reduction_percent": 100 * float(gate_c["strict"]["relative_drop_reduction"]),
        "cross_language_pp": 100 * float(gate_d["strict"]["overall_equal_language_mean_delta_f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    canonical = canonical_percentages(args.registry)
    report = audit_numbers(args.manuscript.read_text(encoding="utf-8"), canonical)
    report.update({"canonical_percentages": canonical, "frozen_registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(), "manuscript": str(args.manuscript.resolve())})
    output = args.output or args.registry.parent.parent / "08_submission_audit" / "paper_number_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = ["# Paper numerical and language audit", "", f"Status: **{report['status']}**.", "", f"Frozen registry SHA-256: `{report['frozen_registry_sha256']}`.", "", "## Canonical percentages", ""]
    markdown.extend(f"- `{name}`: {value:.3f}%" for name, value in canonical.items())
    markdown.extend(["", "## Findings", "", f"- Major numerical mismatches: {len(report['major_mismatches'])}", f"- Prohibited-language violations: {', '.join(report['violations']) or 'none'}"])
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
