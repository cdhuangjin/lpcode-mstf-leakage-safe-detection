from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _config(basetemp: str | None, args: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(option=SimpleNamespace(basetemp=basetemp), invocation_params=SimpleNamespace(args=args))


def test_pytest_configure_uses_pid_scoped_workspace_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import conftest

    monkeypatch.setattr(conftest, "_workspace_temp_root", lambda: tmp_path / ".pytest-tmp")
    monkeypatch.setattr(conftest.os, "getpid", lambda: 101)
    config = _config(str(tmp_path / ".pytest-tmp"))

    conftest.pytest_configure(config)

    assert config.option.basetemp == str(tmp_path / ".pytest-tmp" / "pid-101")


def test_pytest_configure_retains_explicit_basetemp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import conftest

    monkeypatch.setattr(conftest, "_workspace_temp_root", lambda: tmp_path / ".pytest-tmp")
    config = _config(str(tmp_path / "chosen"), ("--basetemp", str(tmp_path / "chosen")))

    conftest.pytest_configure(config)

    assert config.option.basetemp == str(tmp_path / "chosen")


def test_pytest_configure_uses_distinct_default_paths_per_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import conftest

    monkeypatch.setattr(conftest, "_workspace_temp_root", lambda: tmp_path / ".pytest-tmp")
    first, second = _config(str(tmp_path / ".pytest-tmp")), _config(str(tmp_path / ".pytest-tmp"))
    monkeypatch.setattr(conftest.os, "getpid", lambda: 101)
    conftest.pytest_configure(first)
    monkeypatch.setattr(conftest.os, "getpid", lambda: 202)
    conftest.pytest_configure(second)

    assert first.option.basetemp != second.option.basetemp


def test_pytest_sessionfinish_removes_only_its_own_pid_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import conftest

    root = tmp_path / ".pytest-tmp"
    active, sibling = root / "pid-101", root / "pid-202"
    active.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (active / "owned.txt").write_text("owned", encoding="utf-8")
    (sibling / "sibling.txt").write_text("sibling", encoding="utf-8")
    monkeypatch.setattr(conftest, "_workspace_temp_root", lambda: root)
    monkeypatch.setattr(conftest.os, "getpid", lambda: 101)
    config = _config(str(root))
    conftest.pytest_configure(config)

    conftest.pytest_sessionfinish(SimpleNamespace(config=config), 0)

    assert not active.exists()
    assert (sibling / "sibling.txt").read_text(encoding="utf-8") == "sibling"
