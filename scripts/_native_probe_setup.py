"""Shared setup helpers for the native FA4 probe scripts."""

from __future__ import annotations

import sys
from pathlib import Path

from patch_flash_attn_sm120_backward import ensure_patch_applied


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def install_native_probe_paths() -> tuple[Path, Path]:
    root = repo_root()
    runtime_src = root / "cutlass_runtime" / "src"
    shim_root = root / "native_probe_shims"

    for entry in (str(runtime_src), str(shim_root)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(shim_root))
    sys.path.insert(0, str(runtime_src))
    return runtime_src, shim_root


def ensure_native_fa4_patch(*, verbose: bool = False) -> Path:
    target, _ = ensure_patch_applied(verbose=verbose)
    return target


def loaded_cutlass_shim_modules() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name, module in sys.modules.items():
        path = getattr(module, "__file__", None)
        if not name.startswith("cutlass") or not path:
            continue
        normalized = str(path).replace("\\", "/")
        if "native_probe_shims/" not in normalized:
            continue
        rows.append((name, str(path)))
    rows.sort()
    return rows

