"""Minimal tcgen05.mma placeholders for native FA4 import probing."""

from __future__ import annotations

from _probe_helpers import module_getattr, ProbePlaceholder


CtaGroup = ProbePlaceholder("cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")

__getattr__ = module_getattr("cutlass.cute.nvgpu.tcgen05.mma")

