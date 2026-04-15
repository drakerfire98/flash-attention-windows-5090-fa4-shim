"""Minimal tensor objects for the runtime-owned CuTe probe path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeTensor:
    """Tiny tensor metadata container used by compile-time probes."""

    element_type: Any
    shape: tuple[Any, ...]
    stride: tuple[Any, ...]
    assumed_align: int | None = None
    iterator: Any | None = None

    def __post_init__(self) -> None:
        self.shape = tuple(self.shape)
        self.stride = tuple(self.stride)
        self.ndim = len(self.shape)
        self.layout = None
        self.leading_dim = None
        self.dynamic_modes: tuple[Any, ...] = ()

    def __len__(self) -> int:
        return len(self.shape)

    def __getitem__(self, item):
        return self.shape[item]

    def mark_layout_dynamic(self, leading_dim: int | None = None):
        self.leading_dim = leading_dim
        return self

    def mark_compact_shape_dynamic(
        self,
        *,
        mode: int | None = None,
        stride_order: Any = None,
        divisibility: int | None = None,
    ):
        self.dynamic_modes = (*self.dynamic_modes, (mode, stride_order, divisibility))
        return self

    def __repr__(self) -> str:
        return (
            f"FakeTensor(shape={self.shape}, stride={self.stride}, "
            f"element_type={self.element_type!r})"
        )


class TensorSSA(FakeTensor):
    pass
