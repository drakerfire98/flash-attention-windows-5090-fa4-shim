"""Shared setup helpers for the native FA4 probe scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def install_native_probe_paths() -> tuple[Path, Path, Path]:
    root = repo_root()
    flash_attn_runtime_src = root / "flash_attn_runtime" / "src"
    runtime_src = root / "cutlass_runtime" / "src"
    shim_root = root / "native_probe_shims"

    for entry in (str(flash_attn_runtime_src), str(runtime_src), str(shim_root)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(shim_root))
    sys.path.insert(0, str(runtime_src))
    sys.path.insert(0, str(flash_attn_runtime_src))
    return flash_attn_runtime_src, runtime_src, shim_root


def native_flash_attn_interface_path() -> Path:
    root = repo_root()
    return root / "flash_attn_runtime" / "src" / "flash_attn" / "cute" / "interface.py"


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


def loaded_flash_attn_overlay_modules() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name, module in sys.modules.items():
        path = getattr(module, "__file__", None)
        if not name.startswith("flash_attn") or not path:
            continue
        normalized = str(path).replace("\\", "/")
        if "flash_attn_runtime/src/" not in normalized:
            continue
        rows.append((name, str(path)))
    rows.sort()
    return rows
