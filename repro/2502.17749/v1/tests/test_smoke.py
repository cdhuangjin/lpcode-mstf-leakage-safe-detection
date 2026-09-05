import json

from lpcode_v1.smoke import run_smoke, write_smoke_report


def test_smoke_validates_all_languages() -> None:
    report = run_smoke(limit_per_language=20)
    assert report["status"] == "passed"
    assert set(report["languages"]) == {"c", "cpp", "java", "py"}
    assert all(item["feature_shape"][1] == 20 for item in report["languages"].values())
    assert all(item["leakage_count"] == 0 for item in report["languages"].values())


def test_smoke_report_writes_json(tmp_path) -> None:
    output = tmp_path / "smoke.json"
    written = write_smoke_report(output, limit_per_language=12)
    assert written == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
