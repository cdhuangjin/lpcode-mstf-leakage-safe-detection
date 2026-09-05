import json

from lpcode_v1.manifest import build_official_manifest, sha256_file


def test_sha256_file_is_stable(tmp_path) -> None:
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"lpcode")
    assert (
        sha256_file(sample)
        == "e6b4ce4b0032c9be2e8bb4ffb1b3c45df231e34d6d0cfea9ef1d182ef45dddff"
    )


def test_manifest_contains_all_official_pickles() -> None:
    manifest = build_official_manifest()
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 8
    assert {item["task"] for item in artifacts} == {"task1", "task2"}
    assert {item["language"] for item in artifacts} == {"c", "cpp", "java", "py"}
    json.dumps(manifest, sort_keys=True)
