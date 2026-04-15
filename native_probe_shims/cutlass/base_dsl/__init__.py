"""Minimal base_dsl package for native FA4 import probing."""

from __future__ import annotations

from types import SimpleNamespace


class Arch:
    sm_90 = 90


class _ProbeDSL:
    def get_arch_enum(self):
        return Arch.sm_90


class BaseDSL:
    @staticmethod
    def _get_dsl():
        return _ProbeDSL()


runtime = SimpleNamespace(
    cuda=SimpleNamespace(
        load_cubin_module_data=lambda *args, **kwargs: None,
    )
)

