"""Runtime-owned CUTLASS DSL compatibility layer for native FA4 probing."""

from __future__ import annotations

from types import SimpleNamespace

from _probe_helpers import ProbePlaceholder, module_getattr, passthrough_decorator


class NumericMeta(type):
    pass


class JitCompiledFunction:
    def export_to_c(self, *args, **kwargs):
        del args, kwargs
        return None


class CudaDialectJitCompiledFunction(JitCompiledFunction):
    def __init__(self, cubin_data=None, function_name: str | None = None, num_devices: int = 1):
        self.cubin_data = cubin_data
        self.function_name = function_name
        self.num_devices = int(num_devices)
        self._loaded_module = None

    def _load_cuda_library(self):
        if self._loaded_module is None:
            if self.cubin_data is None:
                raise RuntimeError("No cubin data is attached to this CudaDialectJitCompiledFunction")
            from cutlass.base_dsl.runtime import cuda as runtime_cuda

            self._loaded_module = runtime_cuda.load_cubin_module_data(self.cubin_data)
        return [self._loaded_module]

    def load_function(self, function_name: str | None = None):
        name = function_name or self.function_name
        if not name:
            raise RuntimeError("No function name is attached to this CudaDialectJitCompiledFunction")
        from cutlass.base_dsl.runtime import cuda as runtime_cuda

        module = self._load_cuda_library()[0]
        return runtime_cuda.get_module_function(module, name)


class _ProbeDSL:
    def get_arch_enum(self):
        return 120


class BaseDSL:
    @staticmethod
    def _get_dsl():
        return _ProbeDSL()


class Arch:
    sm_80 = 80
    sm_90 = 90
    sm_90a = 90
    sm_100 = 100
    sm_103 = 103
    sm_103f = 103
    sm_110 = 110
    sm_110f = 110
    sm_120 = 120


T = ProbePlaceholder("cutlass.cutlass_dsl.T")
dsl_user_op = passthrough_decorator


def if_generate(predicate, fn, *args, **kwargs):
    del args, kwargs
    if predicate:
        return fn()
    return None


def and_(*values):
    return all(bool(value) for value in values)


def or_(*values):
    return any(bool(value) for value in values)


cuda_jit_executor = SimpleNamespace(CudaDialectJitCompiledFunction=CudaDialectJitCompiledFunction)

__getattr__ = module_getattr("cutlass.cutlass_dsl")

