"""Shared collision-safe cache publication and per-path advisory locking."""

from __future__ import annotations

import errno
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def unique_temporary_path(path: Path, suffix: str = "") -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp{suffix}"
    )


def acquire_windows_file_lock(
    handle: Any, retry_delay_seconds: float = 0.05
) -> None:
    if retry_delay_seconds <= 0:
        raise ValueError("retry delay must be positive")
    import msvcrt

    contention_errnos = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
    contention_winerrors = {33, 36}
    while True:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if (
                exc.errno not in contention_errnos
                and getattr(exc, "winerror", None) not in contention_winerrors
            ):
                raise
            time.sleep(retry_delay_seconds)


@contextmanager
def exclusive_cache_lock(lock_path: Path) -> Iterator[None]:
    key = str(lock_path.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        locked = False
        try:
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            if os.name == "nt":
                acquire_windows_file_lock(handle)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
            yield
        finally:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = unique_temporary_path(path, suffix=path.suffix)
    try:
        temporary.write_bytes(contents)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "acquire_windows_file_lock",
    "atomic_write_bytes",
    "exclusive_cache_lock",
    "unique_temporary_path",
]
