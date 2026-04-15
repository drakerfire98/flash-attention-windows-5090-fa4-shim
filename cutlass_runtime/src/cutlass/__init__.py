"""Installable wrapper that exposes the repo's Windows probe cutlass package."""

from __future__ import annotations

from pathlib import Path


def _shim_init() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    shim_init = repo_root / "native_probe_shims" / "cutlass" / "__init__.py"
    if not shim_init.is_file():
        raise ImportError(f"Unable to locate native probe cutlass shim at {shim_init}")
    return shim_init


_SHIM_INIT = _shim_init()
__file__ = str(_SHIM_INIT)
__path__ = [str(_SHIM_INIT.parent)]  # type: ignore[assignment]

exec(compile(_SHIM_INIT.read_text(encoding="utf-8"), __file__, "exec"), globals())
