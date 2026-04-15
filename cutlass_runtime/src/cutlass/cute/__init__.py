"""Repo-local CuTe compatibility surface for native FA4 probing on Windows.

This package lives under ``cutlass_runtime/src`` so the runtime owns the
``cutlass.cute`` import directly without depending on the older
``native_probe_shims`` submodule fallback path.
"""

from __future__ import annotations

from pathlib import Path

import torch

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[4]
__path__ = [str(_PACKAGE_DIR)]  # type: ignore[assignment]

from _probe_helpers import ProbePlaceholder, module_getattr, passthrough_decorator
from ._compile_bridge import compile_dispatch
from ._pycute_compat import (
    ComposedLayout,
    Layout,
    append,
    coalesce,
    composition,
    flatten_to_tuple,
    make_layout,
    make_shape,
    product,
    rank,
    size,
)
from .tensor import FakeTensor, TensorSSA
from . import nvgpu, runtime


class _ArchShim:
    WARP_SIZE = 32

    def __getattr__(self, name: str):
        return ProbePlaceholder(f"cutlass.cute.arch.{name}")


class _MathCompat:
    def tanh(self, value, fastmath: bool = False):
        del fastmath
        if torch.is_tensor(value):
            return torch.tanh(value)
        return value

    def __getattr__(self, name: str):
        return ProbePlaceholder(f"cutlass.cute.math.{name}")


jit = passthrough_decorator
kernel = passthrough_decorator
compile = compile_dispatch
sym_int = runtime.sym_int
sym_int64 = runtime.sym_int64
math = _MathCompat()
arch = _ArchShim()
typing = ProbePlaceholder("cutlass.cute.typing")
struct = ProbePlaceholder("cutlass.cute.struct")
ReductionOp = ProbePlaceholder("cutlass.cute.ReductionOp")
FastDivmodDivisor = ProbePlaceholder("cutlass.cute.FastDivmodDivisor")
AddressSpace = ProbePlaceholder("cutlass.cute.AddressSpace")
Shape = ProbePlaceholder("cutlass.cute.Shape")

Pointer = type("Pointer", (), {})
Tensor = FakeTensor
TiledMma = type("TiledMma", (), {})
TiledCopy = type("TiledCopy", (), {})
CopyAtom = type("CopyAtom", (), {})
Numeric = type("Numeric", (), {})
Coord = type("Coord", (), {})

__all__ = [
    "AddressSpace",
    "ComposedLayout",
    "Coord",
    "CopyAtom",
    "FastDivmodDivisor",
    "Layout",
    "Numeric",
    "Pointer",
    "ReductionOp",
    "Shape",
    "Tensor",
    "TensorSSA",
    "TiledCopy",
    "TiledMma",
    "append",
    "arch",
    "coalesce",
    "compile",
    "composition",
    "flatten_to_tuple",
    "jit",
    "kernel",
    "make_layout",
    "make_shape",
    "math",
    "nvgpu",
    "product",
    "rank",
    "runtime",
    "size",
    "sym_int",
    "sym_int64",
]

__getattr__ = module_getattr("cutlass.cute")
