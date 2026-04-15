"""Runtime-local minimal base_dsl package for the Windows CUTLASS probe path."""

from __future__ import annotations

from . import runtime
from .arch import Arch


__all__ = ["Arch", "BaseDSL", "runtime"]


class _ProbeDSL:
    def get_arch_enum(self):
        return Arch.sm_120


class BaseDSL:
    @staticmethod
    def _get_dsl():
        return _ProbeDSL()
