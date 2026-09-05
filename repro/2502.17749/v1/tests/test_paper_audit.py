"""Regression contracts for manuscript numerical and language audits."""

from __future__ import annotations

from lpcode_v1.paper_audit import audit_numbers, audit_text


def test_audit_fails_mismatched_manuscript_percentage() -> None:
    report = audit_numbers("MSTF improves by 99.99%.", {"strict_clean_pp": 1.683})

    assert report["status"] == "FAIL"
    assert report["major_mismatches"]


def test_audit_flags_folds_as_independent_and_universal_claims() -> None:
    report = audit_text("Five folds are independent datasets; universal detector.")

    assert {"folds_as_datasets", "universal_claim"} <= set(report["violations"])
