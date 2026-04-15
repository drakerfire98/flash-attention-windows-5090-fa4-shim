"""Runtime-local base_dsl.typing placeholders for the Windows CUTLASS probe path."""

from __future__ import annotations

from _probe_helpers import ProbePlaceholder


Integer = ProbePlaceholder("cutlass.base_dsl.typing.Integer")


def get_mlir_types(*args, **kwargs):
    del args, kwargs
    return ProbePlaceholder("cutlass.base_dsl.typing.get_mlir_types()")
