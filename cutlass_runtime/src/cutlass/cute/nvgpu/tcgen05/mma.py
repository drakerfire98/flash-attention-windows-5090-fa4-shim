"""Minimal tcgen05.mma placeholders for the runtime-owned CuTe probe path."""

from __future__ import annotations

from _probe_helpers import ProbePlaceholder, module_getattr


CtaGroup = ProbePlaceholder("cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")

__getattr__ = module_getattr("cutlass.cute.nvgpu.tcgen05.mma")
