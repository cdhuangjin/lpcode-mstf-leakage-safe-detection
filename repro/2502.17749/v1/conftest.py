"""Pytest configuration with process-isolated workspace temp directories."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _workspace_temp_root() -> Path:
    return Path(__file__).resolve().parent / ".pytest-tmp"


def _has_explicit_basetemp(config) -> bool:
    args = getattr(getattr(config, "invocation_params", None), "args", ())
    return any(argument == "--basetemp" or argument.startswith("--basetemp=") for argument in args)


def pytest_configure(config) -> None:
    """Avoid cross-process cleanup races and an inaccessible system temp root."""
    # pyproject's addopts supplies a shared workspace default.  Only a
    # command-line --basetemp is an intentional override of PID isolation.
    if not _has_explicit_basetemp(config):
        pid = os.getpid()
        path = _workspace_temp_root() / f"pid-{pid}"
        path.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(path)
        config._lpcode_pid_basetemp = (path, pid)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Remove only this process's default temp tree; never touch siblings."""
    owned = getattr(session.config, "_lpcode_pid_basetemp", None)
    if owned is None:
        return
    path, pid = owned
    root = _workspace_temp_root().resolve()
    expected = (root / f"pid-{pid}").resolve()
    if pid != os.getpid() or Path(path).resolve() != expected or expected.parent != root:
        return
    if expected.is_dir():
        shutil.rmtree(expected)
