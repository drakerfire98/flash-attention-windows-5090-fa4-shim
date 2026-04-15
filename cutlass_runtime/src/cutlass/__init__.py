"""Repo-local Windows CUTLASS runtime root for FA4 probing.

This package intentionally stops executing the legacy editable CUTLASS root.
Instead, it owns the top-level ``cutlass`` package locally and exposes the
repo's probe subpackages through ``__path__``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from _probe_helpers import ProbePlaceholder, try_dist_version


_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[2]
_NATIVE_PROBE_CUTLASS_DIR = _REPO_ROOT / "native_probe_shims" / "cutlass"
_CUDA_COMPAT_INIT = _REPO_ROOT / "runtime_compat" / "src" / "cuda" / "__init__.py"

__path__ = [str(_PACKAGE_DIR), str(_NATIVE_PROBE_CUTLASS_DIR)]  # type: ignore[assignment]


def _ensure_probe_cuda_shim_loaded() -> None:
    cuda_mod = sys.modules.get("cuda")
    if cuda_mod is not None and hasattr(cuda_mod, "__version__"):
        return

    try:
        cuda_mod = importlib.import_module("cuda")
    except Exception:
        cuda_mod = None
    if cuda_mod is not None and hasattr(cuda_mod, "__version__"):
        return

    spec = importlib.util.spec_from_file_location(
        "cuda",
        _CUDA_COMPAT_INIT,
        submodule_search_locations=[str(_CUDA_COMPAT_INIT.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runtime_compat cuda shim from {_CUDA_COMPAT_INIT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cuda"] = module
    spec.loader.exec_module(module)


_ensure_probe_cuda_shim_loaded()

NATIVE_PROBE_MODE = "runtime-local-core"
NATIVE_PROBE_CUTLASS_INIT = str(Path(__file__).resolve())
NATIVE_PROBE_RUNTIME_OWNED_MODULES = (
    "cutlass",
    "cutlass.base_dsl",
    "cutlass.base_dsl.runtime.cuda",
    "cutlass.cute",
    "cutlass.cute._compile_bridge",
    "cutlass.cutlass_dsl",
    "cutlass.pipeline",
    "cutlass.utils",
    "cutlass.utils.blackwell_helpers",
    "cutlass.utils.hopper_helpers",
    "cutlass.utils.blockscaled_layout",
    "cutlass._mlir",
    "cutlass._mlir.ir",
    "cutlass._mlir.dialects",
    "cutlass._mlir.dialects.arith",
    "cutlass._mlir.dialects.cute_nvgpu",
    "cutlass._mlir.dialects.llvm",
    "cutlass._mlir.dialects.nvvm",
    "cutlass._mlir.dialects.vector",
)
NATIVE_PROBE_FALLBACK_ROOTS = (str(_NATIVE_PROBE_CUTLASS_DIR),)
NATIVE_PROBE_DIST_VERSIONS = {
    "cutlass": try_dist_version("cutlass"),
    "nvidia-cutlass-dsl": try_dist_version("nvidia-cutlass-dsl"),
}
NATIVE_PROBE_REASON = (
    "Using the repo-local cutlass_runtime root package and runtime-owned "
    "CUTLASS compatibility subpackages instead of the legacy editable "
    "CUTLASS root. The native_probe_shims tree remains only as a fallback "
    "for unimplemented modules."
)


def _noop_load_cubin_module_data(*args, **kwargs):
    del args, kwargs
    return None


try:
    base_dsl = importlib.import_module("cutlass.base_dsl")
except Exception:
    base_dsl = SimpleNamespace(
        runtime=SimpleNamespace(
            cuda=SimpleNamespace(load_cubin_module_data=_noop_load_cubin_module_data)
        )
    )

try:
    pipeline = importlib.import_module("cutlass.pipeline")
except Exception:
    pipeline = ProbePlaceholder("cutlass.pipeline")

try:
    utils = importlib.import_module("cutlass.utils")
except Exception:
    utils = ProbePlaceholder("cutlass.utils")

try:
    cutlass_dsl = importlib.import_module("cutlass.cutlass_dsl")
except Exception:
    cutlass_dsl = ProbePlaceholder("cutlass.cutlass_dsl")

try:
    _mlir = importlib.import_module("cutlass._mlir")
except Exception:
    _mlir = ProbePlaceholder("cutlass._mlir")


def __getattr__(name: str):
    if name == "cute":
        return importlib.import_module("cutlass.cute")
    raise AttributeError(f"module 'cutlass' has no attribute {name!r}")


class Constexpr:
    @classmethod
    def __class_getitem__(cls, item):
        del item
        return cls


class Numeric:
    width = 0

    def __init__(self, value=None):
        self.value = value

    def ir_value(self, *, loc=None, ip=None):
        del loc, ip
        return self.value


class Boolean(Numeric):
    width = 1


class Int8(Numeric):
    width = 8


class Int16(Numeric):
    width = 16


class Int32(Numeric):
    width = 32


class Uint32(Numeric):
    width = 32


class Int64(Numeric):
    width = 64


class Float16(Numeric):
    width = 16


class BFloat16(Numeric):
    width = 16


class Float32(Numeric):
    width = 32


class Pointer:
    pass


def const_expr(value):
    return value


def range_constexpr(*args):
    return range(*args)


CUDA_VERSION = 12080


__all__ = [
    "Constexpr",
    "Numeric",
    "Boolean",
    "Int8",
    "Int16",
    "Int32",
    "Uint32",
    "Int64",
    "Float16",
    "BFloat16",
    "Float32",
    "Pointer",
    "const_expr",
    "range_constexpr",
    "CUDA_VERSION",
    "base_dsl",
    "cute",
    "cutlass_dsl",
    "pipeline",
    "utils",
    "_mlir",
    "NATIVE_PROBE_MODE",
    "NATIVE_PROBE_CUTLASS_INIT",
    "NATIVE_PROBE_RUNTIME_OWNED_MODULES",
    "NATIVE_PROBE_FALLBACK_ROOTS",
    "NATIVE_PROBE_DIST_VERSIONS",
    "NATIVE_PROBE_REASON",
]
