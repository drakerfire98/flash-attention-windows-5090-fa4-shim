"""Installable wrapper for the repo's Windows fcntl shim."""

from __future__ import annotations

from pathlib import Path


def _shim_file() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    shim_file = repo_root / "native_probe_shims" / "fcntl.py"
    if not shim_file.is_file():
        raise ImportError(f"Unable to locate native probe fcntl shim at {shim_file}")
    return shim_file


_SHIM_FILE = _shim_file()
__file__ = str(_SHIM_FILE)

exec(compile(_SHIM_FILE.read_text(encoding="utf-8"), __file__, "exec"), globals())
