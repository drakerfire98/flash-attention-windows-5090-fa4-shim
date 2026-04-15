"""Minimal CUTLASS DSL compatibility layer for native FA4 probing."""

from __future__ import annotations

from _probe_helpers import ProbePlaceholder, module_getattr, passthrough_decorator


class NumericMeta(type):
    pass


class JitCompiledFunction:
    def export_to_c(self, *args, **kwargs):
        del args, kwargs
        return None


class _ProbeDSL:
    def get_arch_enum(self):
        return 120


class BaseDSL:
    @staticmethod
    def _get_dsl():
        return _ProbeDSL()


class Arch:
    pass


T = ProbePlaceholder("cutlass.cutlass_dsl.T")
dsl_user_op = passthrough_decorator


def if_generate(predicate, fn, *args, **kwargs):
    if predicate:
        return fn()
    return None


def and_(*values):
    return all(bool(value) for value in values)


def or_(*values):
    return any(bool(value) for value in values)

__getattr__ = module_getattr("cutlass.cutlass_dsl")
