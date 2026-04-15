"""Minimal llvm dialect placeholders for native FA4 import probing."""

from __future__ import annotations

from _probe_helpers import module_getattr


__getattr__ = module_getattr("cutlass._mlir.dialects.llvm")
