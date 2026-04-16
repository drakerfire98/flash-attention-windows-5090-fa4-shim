"""Repo-owned FlashAttention CuTe overlay for the Windows FA4 probe path."""

from __future__ import annotations

from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import cutlass.cute as cute
from cutlass.cute._compile_bridge import NativePatchedRuntimeCompiler

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[4]
_UPSTREAM_CUTE_DIR = _REPO_ROOT / "third_party" / "flash-attention-for-windows" / "flash_attn" / "cute"

__path__ = [str(_PACKAGE_DIR), str(_UPSTREAM_CUTE_DIR)]  # type: ignore[assignment]

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from flash_attn.cute.cute_dsl_utils import cute_compile_patched

cute.compile = NativePatchedRuntimeCompiler(cute_compile_patched)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
