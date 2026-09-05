"""Freeze hashes and provenance for the official LPcode baseline outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import REPRO_ROOT, RESULTS_ROOT, WORKSPACE_ROOT, resolve_output_path


EXPECTED_COMMIT = "b3660c8262ae57e14498528119607ee673d4257a"
LANGUAGES = ("c", "cpp", "java", "py")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPRO_ROOT / "code"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"official checkout moved: expected {EXPECTED_COMMIT}, found {commit}")
    return commit


def build_official_manifest() -> dict[str, Any]:
    """Describe the fixed official checkout and all eight baseline pickles."""
    artifacts: list[dict[str, Any]] = []
    for task in ("task1", "task2"):
        suffix = "results.pkl" if task == "task1" else "total_results.pkl"
        for language in LANGUAGES:
            path = REPRO_ROOT / "code" / "experiment" / task / f"{language}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"missing official baseline artifact: {path}")
            artifacts.append(
                {
                    "task": task,
                    "language": language,
                    "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_commit": _official_commit(),
        "artifacts": artifacts,
    }


def write_official_manifest(output: Path) -> Path:
    """Atomically write the baseline manifest outside the official tree."""
    destination = resolve_output_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(build_official_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "00_official_baseline" / "manifest.json",
    )
    args = parser.parse_args()
    output = write_official_manifest(args.output)
    print(output)


if __name__ == "__main__":
    main()
