"""Compatibility wrapper for the runtime-local CUDA loader helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runtime_module():
    repo_root = Path(__file__).resolve().parents[4]
    runtime_file = (
        repo_root
        / "cutlass_runtime"
        / "src"
        / "cutlass"
        / "base_dsl"
        / "runtime"
        / "cuda.py"
    )
    spec = importlib.util.spec_from_file_location(
        "native_probe_runtime_local_base_dsl_runtime_cuda",
        runtime_file,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runtime-local CUDA helpers from {runtime_file}")
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


_RUNTIME_MODULE = _load_runtime_module()

for _name, _value in vars(_RUNTIME_MODULE).items():
    if _name in {"__name__", "__loader__", "__package__", "__spec__", "__file__", "__cached__"}:
        continue
    globals()[_name] = _value

__all__ = getattr(
    _RUNTIME_MODULE,
    "__all__",
    sorted(name for name in globals() if not name.startswith("_")),
)
