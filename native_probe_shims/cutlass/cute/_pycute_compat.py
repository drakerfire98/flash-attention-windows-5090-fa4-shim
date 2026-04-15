"""Minimal local PyCuTe compatibility surface for Windows FA4 probing.

This is not a full PyCuTe reimplementation. It only provides enough of the
pure-Python layout surface for the repo-local CuTe shim to stop importing the
external legacy ``pycute`` package directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Iterable


def _normalize_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _default_stride(shape: tuple[Any, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        try:
            running *= int(shape[index])
        except Exception:
            running = 1
    return tuple(stride)


@dataclass(frozen=True)
class Layout:
    shape: tuple[Any, ...]
    stride: tuple[Any, ...]
    order: tuple[Any, ...] | None = None

    def __init__(self, shape=(), stride=None, order=None):
        shape_tuple = _normalize_tuple(shape)
        stride_tuple = (
            _normalize_tuple(stride) if stride is not None else _default_stride(shape_tuple)
        )
        object.__setattr__(self, "shape", shape_tuple)
        object.__setattr__(self, "stride", stride_tuple)
        object.__setattr__(self, "order", None if order is None else _normalize_tuple(order))

    def __len__(self) -> int:
        return len(self.shape)

    def __getitem__(self, item):
        return self.shape[item]

    def __iter__(self):
        return iter(self.shape)

    def __call__(self, *coords):
        return coords


class ComposedLayout(Layout):
    pass


def make_shape(*dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        return _normalize_tuple(dims[0])
    return tuple(dims)


def make_layout(shape=(), stride=None, order=None) -> Layout:
    return Layout(shape=shape, stride=stride, order=order)


def size(value: Any, mode: Any = None):
    shape = getattr(value, "shape", value)
    shape_tuple = _normalize_tuple(shape)
    if mode is not None:
        if isinstance(mode, (list, tuple)) and mode:
            mode = mode[0]
        try:
            return shape_tuple[int(mode)]
        except Exception:
            return 1
    total = 1
    for dim in shape_tuple:
        try:
            total *= int(dim)
        except Exception:
            total *= 1
    return total


def rank(value: Any) -> int:
    return len(_normalize_tuple(getattr(value, "shape", value)))


def flatten_to_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        parts: list[Any] = []
        for item in value:
            parts.extend(flatten_to_tuple(item))
        return tuple(parts)
    if isinstance(value, list):
        parts: list[Any] = []
        for item in value:
            parts.extend(flatten_to_tuple(item))
        return tuple(parts)
    return (value,)


def coalesce(value: Any):
    return value


def composition(a: Any, b: Any = None):
    del b
    if isinstance(a, Layout):
        return ComposedLayout(a.shape, a.stride, a.order)
    return a


def append(a: Iterable[Any], b: Iterable[Any]):
    return tuple(a) + tuple(b)


def product(values: Iterable[Any]) -> int:
    coerced = []
    for value in values:
        try:
            coerced.append(int(value))
        except Exception:
            coerced.append(1)
    return prod(coerced)


__all__ = [
    "Layout",
    "ComposedLayout",
    "make_shape",
    "make_layout",
    "size",
    "rank",
    "flatten_to_tuple",
    "coalesce",
    "composition",
    "append",
    "product",
]
