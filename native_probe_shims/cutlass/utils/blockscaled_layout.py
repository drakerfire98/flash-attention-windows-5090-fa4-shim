"""Minimal blockscaled_layout placeholders for native FA4 import probing."""

from __future__ import annotations

from _probe_helpers import module_getattr


__getattr__ = module_getattr("cutlass.utils.blockscaled_layout")

