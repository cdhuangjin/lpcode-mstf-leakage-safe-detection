"""Read-only integrity checks for the frozen Gate registry and follow-on ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from .release_paths import resolve_recorded_path
from typing import Any


class IntegrityError(RuntimeError):
    """Raised when a declared immutable artifact is missing or altered."""


EXPECTED_GATE_RECORDS = {"gate_a": 480, "gate_b": 960, "gate_c": 1440, "gate_d": 48}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON root must be an object: {path}")
    return value


def audit_registry_hashes(registry_path: Path) -> dict[str, Any]:
    """Verify every frozen file's path, byte count and SHA-256 against registry."""
    registry = _load_json(registry_path)
    bundles = registry.get("bundles")
    if not isinstance(bundles, dict) or not bundles:
        raise IntegrityError("registry has no Gate bundles")
    checked: list[dict[str, Any]] = []
    for bundle_name, bundle in sorted(bundles.items()):
        files = bundle.get("files") if isinstance(bundle, dict) else None
        if not isinstance(files, dict):
            raise IntegrityError(f"registry has no files map for {bundle_name}")
        for filename, declared in sorted(files.items()):
            if not isinstance(declared, dict):
                raise IntegrityError(f"registry has invalid declaration for {bundle_name}/{filename}")
            path = resolve_recorded_path(str(declared.get("path", "")))
            if not path.is_file():
                raise IntegrityError(f"registered artifact is missing: {path}")
            actual_hash, actual_bytes = _sha256(path), path.stat().st_size
            if actual_hash != declared.get("sha256"):
                raise IntegrityError(f"hash mismatch for {bundle_name}/{filename}")
            if actual_bytes != declared.get("bytes"):
                raise IntegrityError(f"byte-count mismatch for {bundle_name}/{filename}")
            checked.append({"bundle": bundle_name, "file": filename, "path": str(path), "sha256": actual_hash, "bytes": actual_bytes})
    return {"registry_sha256": _sha256(registry_path), "checked_files": checked, "registry": registry}


def _ledger_count(path: Path, required_field: str = "record_sha256") -> int:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid JSONL ledger: {path}") from exc
    if not records or any(not isinstance(record, dict) or not isinstance(record.get(required_field), str) for record in records):
        raise IntegrityError(f"ledger lacks valid record bindings: {path}")
    return len(records)


def audit_integrity(registry_path: Path) -> dict[str, Any]:
    """Check frozen hashes, strict Gate states, ledgers and A0--A5 completeness."""
    hash_report = audit_registry_hashes(registry_path)
    registry = hash_report.pop("registry")
    if set(registry["bundles"]) != set(EXPECTED_GATE_RECORDS):
        raise IntegrityError("registry has an incomplete Gate bundle set")
    gate_rows: list[dict[str, Any]] = []
    for name, expected in EXPECTED_GATE_RECORDS.items():
        bundle = registry["bundles"][name]
        root = resolve_recorded_path(bundle["root"])
        gate_path = root / f"{name}.json"
        gate = _load_json(gate_path)
        if gate.get("status") != "evaluable" or gate.get("strict", {}).get("passed") is not True:
            raise IntegrityError(f"{name} is not a strict PASS gate")
        observed = _ledger_count(root / "folds.jsonl")
        if observed != expected:
            raise IntegrityError(f"{name} record count is {observed}, expected {expected}")
        gate_rows.append({"gate": name, "records": observed, "expected_records": expected, "protocol": bundle.get("protocol_version"), "status": "PASS"})
    results_root = registry_path.resolve().parent.parent
    ablation_root = results_root / "05_mechanism_analysis"
    manifest = _load_json(ablation_root / "manifest.json")
    registry_digest = hash_report["registry_sha256"]
    if manifest.get("frozen_registry_sha256") != registry_digest:
        raise IntegrityError("ablation manifest is not bound to the frozen registry")
    for filename, declared in manifest.get("files", {}).items():
        path = ablation_root / filename
        if not path.is_file() or _sha256(path) != declared.get("sha256") or path.stat().st_size != declared.get("bytes"):
            raise IntegrityError(f"ablation manifest mismatch for {filename}")
    ablation_records = _ledger_count(ablation_root / "folds.jsonl", "frozen_registry_sha256")
    if ablation_records != 1800:
        raise IntegrityError(f"ablation record count is {ablation_records}, expected 1800")
    ablation_rows = [json.loads(line) for line in (ablation_root / "folds.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row["frozen_registry_sha256"] != registry_digest for row in ablation_rows):
        raise IntegrityError("ablation ledger contains a record bound to another registry")
    config = _load_json(ablation_root / "config.json")
    if config.get("expected_clean") != 360 or config.get("expected_unseen") != 1440:
        raise IntegrityError("ablation config does not declare the closed 360/1440 matrix")
    return {"status": "PASS", **hash_report, "gates": gate_rows, "ablation": {"records": ablation_records, "expected_records": 1800, "status": "PASS"}, "pytest": "not run by this audit invocation"}


def run_full_pytest(report: dict[str, Any]) -> dict[str, Any]:
    """Run the complete suite and preserve documented MLP warnings as warnings."""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = completed.stdout
    summary_matches = re.findall(r"\d+ passed(?:, \d+ warnings)? in [^\n]+", output)
    if completed.returncode != 0 or not summary_matches:
        raise IntegrityError("full pytest suite failed: " + output[-1000:])
    warning_kinds = []
    if "ConvergenceWarning" in output:
        warning_kinds.append("documented MLP max_iter=200 ConvergenceWarning")
    if "PyparsingDeprecationWarning" in output:
        warning_kinds.append("matplotlib/pyparsing deprecation warning")
    return {**report, "pytest": {"status": "PASS", "summary": summary_matches[-1], "warning_handling": "; ".join(warning_kinds) or "none"}}


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Full integrity audit", "", f"Status: **{report['status']}**.", "", f"Frozen registry SHA-256: `{report['registry_sha256']}`.", "", "## Formal ledgers", "", "| Gate | Records | Expected | Protocol | Status |", "| --- | ---: | ---: | --- | --- |"]
    lines.extend(f"| {row['gate']} | {row['records']} | {row['expected_records']} | {row['protocol']} | {row['status']} |" for row in report["gates"])
    pytest_summary = report["pytest"] if isinstance(report["pytest"], str) else f"{report['pytest']['status']}; {report['pytest']['summary']}; warnings: {report['pytest']['warning_handling']}"
    lines.extend(["", f"A0–A5 ablation: {report['ablation']['records']}/{report['ablation']['expected_records']} records, PASS.", f"Registered files checked: {len(report['checked_files'])}.", "", f"Pytest status: {pytest_summary}."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-pytest", action="store_true")
    args = parser.parse_args()
    report = audit_integrity(args.registry)
    if args.run_pytest:
        report = run_full_pytest(report)
    output = args.output or args.registry.parent.parent / "08_submission_audit" / "full_integrity_audit.md"
    write_report(report, output)
    print(output)


if __name__ == "__main__":
    main()
