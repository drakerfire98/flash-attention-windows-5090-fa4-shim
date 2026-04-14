"""Best-effort CUTLASS/CuTe import shim for Windows FA4 probing.

This is intentionally minimal. It does not implement real CUTLASS DSL or CuTe
behavior. Its job is to expose enough import surface that we can probe how far
the FlashAttention-4 Python stack gets on Windows before the next hard blocker.
"""

from __future__ import annotations

import contextlib
import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class _ShimBase:
    """Tiny base object for no-op runtime placeholders."""

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __getattr__(self, name: str):
        return _ShimValue(f"{type(self).__name__}.{name}")


class _ShimValue:
    """Chainable placeholder used for unknown attrs and no-op call sites."""

    def __init__(self, name: str = "cutlass.shim"):
        self._name = name

    def __repr__(self) -> str:
        return f"<_ShimValue {self._name}>"

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self

    def __getattr__(self, name: str):
        return _ShimValue(f"{self._name}.{name}")

    def __getitem__(self, item):
        return self

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return True

    def __int__(self) -> int:
        return 0

    def __float__(self) -> float:
        return 0.0

    def __index__(self) -> int:
        return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def __mro_entries__(self, bases):
        return (_ShimBase,)

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self

    def __floordiv__(self, other):
        return self

    def __rfloordiv__(self, other):
        return self

    def __mod__(self, other):
        return self

    def __rmod__(self, other):
        return self

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def clone(self):
        return self

    def ir_value(self):
        return self

    def mark_layout_dynamic(self, *args, **kwargs):
        return self

    def mark_compact_shape_dynamic(self, *args, **kwargs):
        return self


def _identity_decorator(fn: Callable | None = None, **_kwargs):
    def decorator(inner: Callable):
        inner.__wrapped__ = getattr(inner, "__wrapped__", inner)
        return inner

    return decorator(fn) if fn is not None else decorator


def _module_getattr(module_name: str):
    def _getattr(name: str):
        return _ShimValue(f"{module_name}.{name}")

    return _getattr


def _make_module(name: str, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = __file__
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    module.__getattr__ = _module_getattr(name)  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _link(parent: types.ModuleType, child_name: str, child: types.ModuleType) -> None:
    setattr(parent, child_name, child)


class Constexpr:
    def __init__(self, value: Any = None):
        self.value = value

    def __bool__(self) -> bool:
        return bool(self.value)

    @classmethod
    def __class_getitem__(cls, item):
        return cls


class NumericMeta(type):
    pass


class Int8(int):
    pass


class Int32(int):
    pass


class Int64(int):
    pass


class Uint32(int):
    pass


class Boolean(int):
    def __new__(cls, value: Any = False):
        return super().__new__(cls, 1 if value else 0)


class Float16(float):
    pass


class BFloat16(float):
    pass


class Float32(float):
    pass


def const_expr(value: Any):
    return value


def extract_mlir_values(obj: Any):
    return []


def new_from_mlir_values(obj: Any, values: list[Any]):
    return obj


class JitCompiledFunction(_ShimBase):
    pass


class _DslType(_ShimValue):
    pass


T = _DslType("cutlass.cutlass_dsl.T")
CUDA_VERSION = "13.0"
__version__ = "0.0.0-shim"


class _ShimTensor(_ShimValue):
    def __init__(self, source: Any = None):
        super().__init__("cutlass.cute.Tensor")
        self.source = source


class Tensor(_ShimBase):
    pass


class TensorSSA(_ShimBase):
    pass


class Pointer(_ShimBase):
    pass


class CopyAtom(_ShimBase):
    pass


class TiledMma(_ShimBase):
    pass


class TiledCopy(_ShimBase):
    pass


class Numeric(metaclass=NumericMeta):
    width = 16


class FastDivmodDivisor(_ShimBase):
    def __init__(self, value: Any = 1):
        self.value = value

    def divmod(self, dividend: int) -> tuple[int, int]:
        divisor = int(self.value) if int(self.value) else 1
        return divmod(dividend, divisor)


def _shim_compile(*args, **kwargs):
    return _ShimValue("cutlass.cute.compile")


def _shim_from_dlpack(x, **_kwargs):
    return _ShimTensor(x)


def _shim_passthrough(*args, **kwargs):
    return _ShimValue("cutlass.cute.passthrough")


def _shim_size(value: Any):
    try:
        return int(value)
    except Exception:
        return 0


def _shim_assume(value: Any, **_kwargs):
    return value


def _shim_tanh(x, **_kwargs):
    return x


def _shim_sync_warp(*args, **kwargs):
    return None


def _shim_barrier(*args, **kwargs):
    return None


def _shim_elect_one():
    return contextlib.nullcontext()


class PipelineUserType(Enum):
    Producer = "producer"
    Consumer = "consumer"


@dataclass(frozen=True)
class PipelineState(_ShimBase):
    stages: int = 0
    count: Int32 = Int32(0)
    index: Int32 = Int32(0)
    phase: Int32 = Int32(0)


@dataclass(frozen=True)
class NamedBarrier(_ShimBase):
    barrier_id: int = 0
    num_threads: int = 0


class _PipelineBase(_ShimBase):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def producer_acquire(self, *args, **kwargs):
        return None

    def producer_commit(self, *args, **kwargs):
        return None

    def producer_get_barrier(self, *args, **kwargs):
        return NamedBarrier()

    def consumer_wait(self, *args, **kwargs):
        return None

    def consumer_try_wait(self, *args, **kwargs):
        return Boolean(True)

    def consumer_release(self, *args, **kwargs):
        return None


class PipelineAsync(_PipelineBase):
    pass


class PipelineCpAsync(_PipelineBase):
    pass


class PipelineTmaAsync(_PipelineBase):
    pass


class PipelineTmaUmma(_PipelineBase):
    pass


class PipelineUmmaAsync(_PipelineBase):
    pass


class PipelineAsyncUmma(_PipelineBase):
    pass


class Agent:
    Thread = "thread"


class CooperativeGroup(_ShimBase):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def make_pipeline_state(kind: PipelineUserType, stages: int):
    if kind is PipelineUserType.Producer:
        return PipelineState(stages=stages, count=Int32(0), index=Int32(0), phase=Int32(1))
    return PipelineState(stages=stages, count=Int32(0), index=Int32(0), phase=Int32(0))


def pipeline_init_arrive(*args, **kwargs):
    return None


def pipeline_init_wait(*args, **kwargs):
    return None


class HardwareInfo(_ShimBase):
    def get_max_active_clusters(self, cluster_size: int = 1) -> int:
        return 1


class WorkTileInfo(_ShimBase):
    pass


class SmemAllocator(_ShimBase):
    def allocate_tensor(self, *args, **kwargs):
        return _ShimTensor()


class TmemAllocator(_ShimBase):
    def allocate_tensor(self, *args, **kwargs):
        return _ShimTensor()


class LayoutEnum(_ShimBase):
    @staticmethod
    def from_tensor(_tensor):
        return _ShimValue("cutlass.utils.LayoutEnum")


class Arch:
    SM80 = 80
    SM90 = 90
    SM100 = 100
    SM120 = 120


def _register_submodules() -> None:
    root = sys.modules[__name__]

    cute_mod = _make_module("cutlass.cute", package=True)
    cute_mod.jit = _identity_decorator
    cute_mod.compile = _shim_compile
    cute_mod.Tensor = Tensor
    cute_mod.TensorSSA = TensorSSA
    cute_mod.Pointer = Pointer
    cute_mod.CopyAtom = CopyAtom
    cute_mod.TiledMma = TiledMma
    cute_mod.TiledCopy = TiledCopy
    cute_mod.Numeric = Numeric
    cute_mod.FastDivmodDivisor = FastDivmodDivisor
    cute_mod.assume = _shim_assume
    cute_mod.size = _shim_size
    cute_mod.make_tiled_copy_A = _shim_passthrough
    cute_mod.make_tiled_copy_B = _shim_passthrough
    cute_mod.make_tensor = _shim_passthrough
    cute_mod.make_layout = _shim_passthrough

    cute_math_mod = _make_module("cutlass.cute.math")
    cute_math_mod.tanh = _shim_tanh

    cute_arch_mod = _make_module("cutlass.cute.arch")
    cute_arch_mod.WARP_SIZE = 32
    cute_arch_mod.sync_warp = _shim_sync_warp
    cute_arch_mod.elect_one = _shim_elect_one
    cute_arch_mod.barrier = _shim_barrier
    cute_arch_mod.barrier_arrive = _shim_barrier

    cute_runtime_mod = _make_module("cutlass.cute.runtime")
    cute_runtime_mod.from_dlpack = _shim_from_dlpack

    cute_core_mod = _make_module("cutlass.cute.core")
    cute_core_mod.ThrMma = _ShimBase

    cute_tvm_ffi_converter_mod = _make_module("cutlass.cute._tvm_ffi_args_spec_converter")
    cute_tvm_ffi_converter_mod._convert_single_arg = lambda arg, arg_name, arg_type, ctx: arg

    cute_nvgpu_mod = _make_module("cutlass.cute.nvgpu", package=True)
    cute_nvgpu_mod.cpasync = _ShimValue("cutlass.cute.nvgpu.cpasync")

    tcgen05_mod = _make_module("cutlass.cute.nvgpu.tcgen05", package=True)

    class Field(Enum):
        ACCUMULATE = 0

    class OperandSource(Enum):
        TMEM = 0
        SMEM = 1

    mma_mod = _make_module("cutlass.cute.nvgpu.tcgen05.mma")

    class OperandMajorMode(Enum):
        K = 0
        MN = 1

    class MmaOp(_ShimBase):
        a_src = OperandSource.SMEM
        a_major_mode = OperandMajorMode.MN
        b_major_mode = OperandMajorMode.MN

    mma_mod.OperandMajorMode = OperandMajorMode
    mma_mod.MmaOp = MmaOp

    tcgen05_mod.Field = Field
    tcgen05_mod.OperandSource = OperandSource
    tcgen05_mod.mma = mma_mod

    cute_mod.math = cute_math_mod
    cute_mod.arch = cute_arch_mod
    cute_mod.runtime = cute_runtime_mod
    cute_mod.core = cute_core_mod
    cute_mod.nvgpu = cute_nvgpu_mod
    cute_mod._tvm_ffi_args_spec_converter = cute_tvm_ffi_converter_mod
    cute_nvgpu_mod.tcgen05 = tcgen05_mod

    dsl_mod = _make_module("cutlass.cutlass_dsl", package=True)
    dsl_mod.NumericMeta = NumericMeta
    dsl_mod.JitCompiledFunction = JitCompiledFunction
    dsl_mod.T = T
    dsl_mod.dsl_user_op = _identity_decorator
    dsl_mod.if_generate = lambda cond, true_fn, false_fn=None: (
        true_fn() if cond else false_fn() if false_fn is not None else None
    )

    dsl_cuda_exec_mod = _make_module("cutlass.cutlass_dsl.cuda_jit_executor")

    class CudaDialectJitCompiledFunction(JitCompiledFunction):
        pass

    dsl_cuda_exec_mod.CudaDialectJitCompiledFunction = CudaDialectJitCompiledFunction
    dsl_mod.cuda_jit_executor = dsl_cuda_exec_mod

    pipeline_mod = _make_module("cutlass.pipeline")
    pipeline_mod.PipelineState = PipelineState
    pipeline_mod.PipelineUserType = PipelineUserType
    pipeline_mod.NamedBarrier = NamedBarrier
    pipeline_mod.PipelineAsync = PipelineAsync
    pipeline_mod.PipelineCpAsync = PipelineCpAsync
    pipeline_mod.PipelineTmaAsync = PipelineTmaAsync
    pipeline_mod.PipelineTmaUmma = PipelineTmaUmma
    pipeline_mod.PipelineUmmaAsync = PipelineUmmaAsync
    pipeline_mod.PipelineAsyncUmma = PipelineAsyncUmma
    pipeline_mod.Agent = Agent
    pipeline_mod.CooperativeGroup = CooperativeGroup
    pipeline_mod.make_pipeline_state = make_pipeline_state
    pipeline_mod.pipeline_init_arrive = pipeline_init_arrive
    pipeline_mod.pipeline_init_wait = pipeline_init_wait

    mlir_mod = _make_module("cutlass._mlir", package=True)
    mlir_ir_mod = _make_module("cutlass._mlir.ir")

    class Value(_ShimBase):
        pass

    mlir_ir_mod.Value = Value

    mlir_dialects_mod = _make_module("cutlass._mlir.dialects", package=True)
    mlir_dialects_llvm_mod = _make_module("cutlass._mlir.dialects.llvm")
    mlir_dialects_nvvm_mod = _make_module("cutlass._mlir.dialects.nvvm")
    mlir_dialects_mod.llvm = mlir_dialects_llvm_mod
    mlir_dialects_mod.nvvm = mlir_dialects_nvvm_mod
    mlir_mod.ir = mlir_ir_mod
    mlir_mod.dialects = mlir_dialects_mod

    utils_mod = _make_module("cutlass.utils", package=True)
    utils_mod.HardwareInfo = HardwareInfo
    utils_mod.WorkTileInfo = WorkTileInfo
    utils_mod.SmemAllocator = SmemAllocator
    utils_mod.TmemAllocator = TmemAllocator
    utils_mod.LayoutEnum = LayoutEnum

    utils_hopper_mod = _make_module("cutlass.utils.hopper_helpers")
    utils_blackwell_mod = _make_module("cutlass.utils.blackwell_helpers")
    utils_mod.hopper_helpers = utils_hopper_mod
    utils_mod.blackwell_helpers = utils_blackwell_mod

    base_dsl_mod = _make_module("cutlass.base_dsl", package=True)
    base_dsl_arch_mod = _make_module("cutlass.base_dsl.arch")
    base_dsl_arch_mod.Arch = Arch
    base_dsl_runtime_mod = _make_module("cutlass.base_dsl.runtime", package=True)
    base_dsl_runtime_cuda_mod = _make_module("cutlass.base_dsl.runtime.cuda")
    base_dsl_runtime_cuda_mod.load_cubin_module_data = lambda *args, **kwargs: _ShimValue(
        "cutlass.base_dsl.runtime.cuda.load_cubin_module_data"
    )
    base_dsl_typing_mod = _make_module("cutlass.base_dsl.typing")
    base_dsl_typing_mod.get_mlir_types = lambda _value: []
    base_dsl_tvm_ffi_builder_mod = _make_module("cutlass.base_dsl.tvm_ffi_builder", package=True)
    base_dsl_tvm_ffi_spec_mod = _make_module("cutlass.base_dsl.tvm_ffi_builder.spec")

    class ConstNone(_ShimBase):
        def __init__(self, name: str):
            self.name = name

    base_dsl_tvm_ffi_spec_mod.ConstNone = ConstNone
    base_dsl_tvm_ffi_builder_mod.spec = base_dsl_tvm_ffi_spec_mod
    base_dsl_runtime_mod.cuda = base_dsl_runtime_cuda_mod
    base_dsl_mod.arch = base_dsl_arch_mod
    base_dsl_mod.runtime = base_dsl_runtime_mod
    base_dsl_mod.typing = base_dsl_typing_mod
    base_dsl_mod.tvm_ffi_builder = base_dsl_tvm_ffi_builder_mod

    _link(root, "cute", cute_mod)
    _link(root, "pipeline", pipeline_mod)
    _link(root, "cutlass_dsl", dsl_mod)
    _link(root, "_mlir", mlir_mod)
    _link(root, "utils", utils_mod)
    _link(root, "base_dsl", base_dsl_mod)


_register_submodules()


__all__ = [
    "BFloat16",
    "Boolean",
    "Constexpr",
    "CUDA_VERSION",
    "Float16",
    "Float32",
    "Int8",
    "Int32",
    "Int64",
    "Uint32",
    "const_expr",
    "extract_mlir_values",
    "new_from_mlir_values",
]


def __getattr__(name: str):
    return _ShimValue(f"cutlass.{name}")
