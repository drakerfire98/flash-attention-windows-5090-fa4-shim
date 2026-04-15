"""Minimal runtime helpers for native FA4 import probing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any

import torch

from _probe_helpers import ProbePlaceholder, module_getattr
from .tensor import FakeTensor


_SYM_COUNTER = count(2)
_TORCH_WIDTH_BITS = {
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
    torch.int32: 32,
    torch.int64: 64,
    torch.uint8: 8,
    torch.int8: 8,
    torch.bool: 1,
}


class SymbolicInt(int):
    """Simple int subclass with a readable symbolic repr for probe logs."""

    def __new__(cls, value: int, label: str):
        obj = int.__new__(cls, value)
        obj._label = label
        return obj

    def __repr__(self) -> str:
        return self._label


@dataclass(frozen=True)
class TorchCuteDType:
    torch_dtype: torch.dtype
    width: int

    def __repr__(self) -> str:
        return f"TorchCuteDType({self.torch_dtype})"


@dataclass(frozen=True)
class FakeStream:
    use_tvm_ffi_env_stream: bool = False


def _next_symbol(*, divisibility: int = 1, bits: int = 32) -> SymbolicInt:
    scale = max(int(divisibility), 1)
    ordinal = next(_SYM_COUNTER)
    value = ordinal * scale
    label = f"sym_int{bits}<{value}>"
    return SymbolicInt(value, label)


def sym_int(*, divisibility: int = 1):
    return _next_symbol(divisibility=divisibility, bits=32)


def sym_int64(*, divisibility: int = 1):
    return _next_symbol(divisibility=divisibility, bits=64)


def find_runtime_libraries(*, enable_tvm_ffi: bool = False):
    del enable_tvm_ffi
    return []


def _normalize_tuple(values: Any) -> tuple[Any, ...]:
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def _default_stride(shape: tuple[Any, ...]) -> tuple[Any, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        next_extent = shape[i + 1]
        try:
            stride[i] = int(stride[i + 1]) * int(next_extent)
        except Exception:
            stride[i] = 1
    return tuple(stride)


def _torch_dtype_to_cute(dtype: torch.dtype) -> TorchCuteDType:
    width = _TORCH_WIDTH_BITS.get(dtype, 32)
    return TorchCuteDType(dtype, width)


def make_fake_tensor(dtype, shape, *, stride=None, assumed_align=None):
    shape_tuple = _normalize_tuple(shape)
    stride_tuple = _normalize_tuple(stride) if stride is not None else _default_stride(shape_tuple)
    return FakeTensor(dtype, shape_tuple, stride_tuple, assumed_align=assumed_align)


def from_dlpack(value, *args, **kwargs):
    del args, kwargs
    if isinstance(value, torch.Tensor):
        return FakeTensor(
            _torch_dtype_to_cute(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
            assumed_align=value.element_size(),
            iterator=ProbePlaceholder("cutlass.cute.runtime.from_dlpack.iterator"),
        )
    return ProbePlaceholder("cutlass.cute.runtime.from_dlpack()")


def make_fake_stream(*, use_tvm_ffi_env_stream: bool = False):
    return FakeStream(use_tvm_ffi_env_stream=use_tvm_ffi_env_stream)


__getattr__ = module_getattr("cutlass.cute.runtime")
