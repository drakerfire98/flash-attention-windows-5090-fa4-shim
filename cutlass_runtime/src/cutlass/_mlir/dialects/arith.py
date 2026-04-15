"""Runtime-owned arith dialect placeholders for native FA4 probing."""

from __future__ import annotations

from _probe_helpers import module_getattr


__getattr__ = module_getattr("cutlass._mlir.dialects.arith")

