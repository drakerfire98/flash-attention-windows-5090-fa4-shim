"""Minimal runtime placeholders for native FA4 import probing."""

from __future__ import annotations

from _probe_helpers import ProbePlaceholder, module_getattr


def find_runtime_libraries(*, enable_tvm_ffi: bool = False):
    del enable_tvm_ffi
    return []


def from_dlpack(*args, **kwargs):
    del args, kwargs
    return ProbePlaceholder("cutlass.cute.runtime.from_dlpack()")


__getattr__ = module_getattr("cutlass.cute.runtime")
