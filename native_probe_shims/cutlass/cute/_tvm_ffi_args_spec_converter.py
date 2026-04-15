"""Minimal TVM FFI args converter placeholder for native FA4 import probing."""

from __future__ import annotations

from _probe_helpers import module_getattr


__getattr__ = module_getattr("cutlass.cute._tvm_ffi_args_spec_converter")

