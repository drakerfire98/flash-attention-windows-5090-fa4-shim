"""Minimal tcgen05 package placeholders for the runtime-owned CuTe probe path."""

from __future__ import annotations

from _probe_helpers import module_getattr


__getattr__ = module_getattr("cutlass.cute.nvgpu.tcgen05")
