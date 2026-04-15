"""Import-time bridge for ``cutlass.cute`` during native FA4 probing.

This is intentionally not a real CuTe runtime. It combines the pure-Python
``pycute`` layout helpers with permissive placeholders for the compiled DSL
surface so that we can push the native import chain forward and discover the
next blocker honestly.
"""

from __future__ import annotations

from pycute import *  # noqa: F401,F403

from _probe_helpers import ProbePlaceholder, module_getattr, passthrough_decorator
from ._compile_bridge import compile_dispatch
from .tensor import FakeTensor, TensorSSA

from . import nvgpu, runtime


class _ArchShim:
    WARP_SIZE = 32

    def __getattr__(self, name: str):
        return ProbePlaceholder(f"cutlass.cute.arch.{name}")


jit = passthrough_decorator
kernel = passthrough_decorator
compile = compile_dispatch
sym_int = runtime.sym_int
sym_int64 = runtime.sym_int64
math = ProbePlaceholder("cutlass.cute.math")
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
ComposedLayout = Layout

__getattr__ = module_getattr("cutlass.cute")
