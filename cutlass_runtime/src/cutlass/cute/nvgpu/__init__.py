"""Minimal nvgpu placeholders for the runtime-owned CuTe probe path."""

from __future__ import annotations

from _probe_helpers import ProbePlaceholder, module_getattr


cpasync = ProbePlaceholder("cutlass.cute.nvgpu.cpasync")
warp = ProbePlaceholder("cutlass.cute.nvgpu.warp")
warpgroup = ProbePlaceholder("cutlass.cute.nvgpu.warpgroup")
tcgen05 = ProbePlaceholder("cutlass.cute.nvgpu.tcgen05")

__getattr__ = module_getattr("cutlass.cute.nvgpu")
