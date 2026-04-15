"""Minimal Windows import surface for the ``nvidia_cutlass_dsl`` package.

On this Windows FA4 environment the published ``nvidia-cutlass-dsl``
distribution metadata can be present without an importable top-level
``nvidia_cutlass_dsl`` module. This shim restores that module name and lazily
forwards useful DSL symbols to ``cutlass.cutlass_dsl``.
"""

from __future__ import annotations

import importlib
from importlib import metadata


try:
    __version__ = metadata.version("nvidia-cutlass-dsl")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

FA4_WINDOWS_RUNTIME_COMPAT = True


def _load_cutlass_dsl():
    return importlib.import_module("cutlass.cutlass_dsl")


def _load_cute():
    return importlib.import_module("cutlass.cute")


def __getattr__(name: str):
    if name == "cutlass_dsl":
        return _load_cutlass_dsl()
    if name == "cute":
        return _load_cute()
    dsl = _load_cutlass_dsl()
    if hasattr(dsl, name):
        return getattr(dsl, name)
    raise AttributeError(f"module 'nvidia_cutlass_dsl' has no attribute {name!r}")


__all__ = [
    "__version__",
    "FA4_WINDOWS_RUNTIME_COMPAT",
    "cutlass_dsl",
    "cute",
]
