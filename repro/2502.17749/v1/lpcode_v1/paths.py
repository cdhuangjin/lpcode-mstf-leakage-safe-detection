"""Canonical workspace paths and output-safety checks."""

from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPRO_ROOT.parents[1]
OFFICIAL_EXPERIMENT_DIR = (REPRO_ROOT / "code" / "experiment").resolve()
RESULTS_ROOT = (WORKSPACE_ROOT / "results").resolve()


def resolve_output_path(path: str | Path) -> Path:
    """Resolve an output path while protecting the official experiment tree."""
    resolved = Path(path).resolve()
    if resolved == OFFICIAL_EXPERIMENT_DIR or OFFICIAL_EXPERIMENT_DIR in resolved.parents:
        raise ValueError(f"refusing output inside official experiment tree: {resolved}")
    return resolved
