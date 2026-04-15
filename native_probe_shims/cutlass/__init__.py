"""Wrapper package that executes the best available CUTLASS package for probing.

The native FA4 probe wants two different things at once:

- it should automatically prefer a real modern CUTLASS DSL package if one ever
  becomes importable in this env
- it still needs a stable fallback path today on Windows, where the editable
  legacy CUTLASS tree is usually the only importable ``cutlass`` package

This wrapper selects the best available real package first, then layers probe
fallbacks only where the selected package does not provide the newer DSL
surface that FA4 expects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from _probe_helpers import (
    ProbePlaceholder,
    find_package_init_candidates,
    try_dist_version,
)


_SHIM_DIR = Path(__file__).resolve().parent
_IMPORT_ORIGIN = Path(getattr(globals().get("__spec__"), "origin", __file__)).resolve()
_REPO_ROOT = _SHIM_DIR.parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent
_CUTLASS_RUNTIME_WRAPPER_ROOT = _REPO_ROOT / "cutlass_runtime"
_LEGACY_PYTHON_ROOT = (
    _WORKSPACE_ROOT / "third_party" / "flash-attention-for-windows" / "csrc" / "cutlass" / "python"
)
_LEGACY_PACKAGE_DIR = _LEGACY_PYTHON_ROOT / "cutlass"
_LEGACY_INIT = _LEGACY_PACKAGE_DIR / "__init__.py"

_MODERN_CANDIDATES = find_package_init_candidates(
    "cutlass",
    exclude_roots=[
        _SHIM_DIR,
        _IMPORT_ORIGIN.parent,
        _CUTLASS_RUNTIME_WRAPPER_ROOT,
        _LEGACY_PYTHON_ROOT,
    ],
)
_MODERN_INIT = _MODERN_CANDIDATES[0] if _MODERN_CANDIDATES else None

if _MODERN_INIT is not None:
    _REAL_INIT = _MODERN_INIT
    _REAL_PACKAGE_DIR = _REAL_INIT.parent
    _REAL_PYTHON_ROOT = _REAL_PACKAGE_DIR.parent
    _PACKAGE_MODE = "modern-cutlass-package"
else:
    _REAL_INIT = _LEGACY_INIT
    _REAL_PACKAGE_DIR = _LEGACY_PACKAGE_DIR
    _REAL_PYTHON_ROOT = _LEGACY_PYTHON_ROOT
    _PACKAGE_MODE = "legacy-editable-cutlass"

if str(_REAL_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_REAL_PYTHON_ROOT))


def _ensure_probe_cuda_shim_loaded() -> None:
    cuda_mod = sys.modules.get("cuda")
    if cuda_mod is not None and hasattr(cuda_mod, "__version__"):
        return
    cuda_init = _REPO_ROOT / "native_probe_shims" / "cuda" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "cuda",
        cuda_init,
        submodule_search_locations=[str(cuda_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load native probe cuda shim from {cuda_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cuda"] = module
    spec.loader.exec_module(module)


if _PACKAGE_MODE == "legacy-editable-cutlass":
    _ensure_probe_cuda_shim_loaded()

__path__ = (
    [str(_REAL_PACKAGE_DIR), str(_SHIM_DIR)]
    if _PACKAGE_MODE == "modern-cutlass-package"
    else [str(_SHIM_DIR), str(_REAL_PACKAGE_DIR)]
)
__file__ = str(_REAL_INIT)

NATIVE_PROBE_MODE = _PACKAGE_MODE
NATIVE_PROBE_CUTLASS_INIT = str(_REAL_INIT)
NATIVE_PROBE_DIST_VERSIONS = {
    "cutlass": try_dist_version("cutlass"),
    "nvidia-cutlass-dsl": try_dist_version("nvidia-cutlass-dsl"),
}
if _PACKAGE_MODE == "modern-cutlass-package":
    NATIVE_PROBE_REASON = (
        "Using a real non-legacy CUTLASS package discovered on sys.path; shim "
        "modules stay available only as fallback."
    )
elif NATIVE_PROBE_DIST_VERSIONS["nvidia-cutlass-dsl"] is not None:
    NATIVE_PROBE_REASON = (
        "Found the nvidia-cutlass-dsl distribution, but no separate modern "
        "importable cutlass package files on this Windows env. Falling back to "
        "the legacy editable CUTLASS tree plus probe shims."
    )
else:
    NATIVE_PROBE_REASON = (
        "No modern CUTLASS DSL package is importable in this env. Falling back "
        "to the legacy editable CUTLASS tree plus probe shims."
    )

exec(compile(_REAL_INIT.read_text(encoding="utf-8"), __file__, "exec"), globals())


def _noop_load_cubin_module_data(*args, **kwargs):
    del args, kwargs
    return None


if "base_dsl" not in globals():
    base_dsl = SimpleNamespace(
        runtime=SimpleNamespace(
            cuda=SimpleNamespace(load_cubin_module_data=_noop_load_cubin_module_data)
        )
    )

if "pipeline" not in globals():
    pipeline = ProbePlaceholder("cutlass.pipeline")


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
