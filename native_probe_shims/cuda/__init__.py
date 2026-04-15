"""Compatibility wrapper for the older top-level CUDA Python API shape.

CUTLASS 3.8-era Python code expects:

    from cuda import __version__, cuda, cudart, nvrtc

The modern Windows ``cuda-python`` packages expose modules under
``cuda.bindings.*`` instead. This wrapper restores the expected names for probe
purposes while still delegating to the installed bindings modules.
"""

from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_REAL_CUDA_DIRS = []
for entry in sys.path:
    try:
        candidate = Path(entry) / "cuda"
    except OSError:
        continue
    if candidate.is_dir() and candidate.resolve() != _THIS_DIR:
        _REAL_CUDA_DIRS.append(str(candidate))

__path__ = [str(_THIS_DIR), *_REAL_CUDA_DIRS]

try:
    real_version = metadata.version("cuda-python")
except metadata.PackageNotFoundError:
    real_version = "unknown"

# CUTLASS 3.8's version compare keeps checking later components even when the
# major version already wins, so "13.2.0" incorrectly fails against NVCC 12.8.
# For probe imports we expose a conservative default that satisfies that buggy
# compare while still recording the real installed version separately.
__real_version__ = real_version
__version__ = os.environ.get("NATIVE_PROBE_CUDA_VERSION", "12.8")

cuda = importlib.import_module("cuda.bindings.driver")
cudart = importlib.import_module("cuda.bindings.runtime")
nvrtc = importlib.import_module("cuda.bindings.nvrtc")

__all__ = ["__version__", "__real_version__", "cuda", "cudart", "nvrtc"]

