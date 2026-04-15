"""Windows compatibility wrapper for the older top-level CUDA Python API.

Legacy CUTLASS Python code expects:

    from cuda import __version__, cuda, cudart, nvrtc

Modern ``cuda-python`` packages on Windows primarily expose binaries under
``cuda.bindings.*`` instead. This package restores the older import shape while
still delegating submodule loading to the installed runtime bindings.
"""

from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_REAL_CUDA_DIRS: list[str] = []
for entry in sys.path:
    try:
        candidate = Path(entry) / "cuda"
    except OSError:
        continue
    if candidate.is_dir() and candidate.resolve() != _THIS_DIR:
        _REAL_CUDA_DIRS.append(str(candidate))

__path__ = [str(_THIS_DIR), *_REAL_CUDA_DIRS]

try:
    __real_version__ = metadata.version("cuda-python")
except metadata.PackageNotFoundError:
    __real_version__ = "unknown"

# CUTLASS 3.x uses a brittle version compare against NVCC and can reject newer
# runtime versions incorrectly. The conservative default keeps the legacy import
# path live while still exposing the real installed version separately.
__version__ = os.environ.get("NATIVE_PROBE_CUDA_VERSION", "12.8")

cuda = importlib.import_module("cuda.bindings.driver")
cudart = importlib.import_module("cuda.bindings.runtime")
nvrtc = importlib.import_module("cuda.bindings.nvrtc")
bindings = importlib.import_module("cuda.bindings")

FA4_WINDOWS_RUNTIME_COMPAT = True

__all__ = [
    "__version__",
    "__real_version__",
    "bindings",
    "cuda",
    "cudart",
    "nvrtc",
    "FA4_WINDOWS_RUNTIME_COMPAT",
]
