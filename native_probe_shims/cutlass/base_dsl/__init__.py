"""Minimal base_dsl package for native FA4 import probing."""

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
