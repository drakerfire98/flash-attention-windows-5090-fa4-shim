"""Runtime-local entrypoint for the FA4 compile bridge.

The heavy lifting still lives in the repo-local probe implementation for now,
but the import surface is owned by ``cutlass_runtime/src/cutlass/cute``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_native_module():
    repo_root = Path(__file__).resolve().parents[4]
    native_file = repo_root / "native_probe_shims" / "cutlass" / "cute" / "_compile_bridge.py"
    spec = importlib.util.spec_from_file_location(
        "cutlass_runtime_native_probe_cute_compile_bridge",
        native_file,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load native probe compile bridge from {native_file}")
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


_NATIVE_MODULE = _load_native_module()

for _name, _value in vars(_NATIVE_MODULE).items():
    if _name in {"__name__", "__loader__", "__package__", "__spec__", "__file__", "__cached__"}:
        continue
    globals()[_name] = _value

__all__ = getattr(
    _NATIVE_MODULE,
    "__all__",
    sorted(name for name in globals() if not name.startswith("_")),
)
