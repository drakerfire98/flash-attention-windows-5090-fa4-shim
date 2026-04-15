"""Runtime-owned blackwell_helpers placeholders for native FA4 probing."""

from __future__ import annotations

from _probe_helpers import module_getattr


__getattr__ = module_getattr("cutlass.utils.blackwell_helpers")

