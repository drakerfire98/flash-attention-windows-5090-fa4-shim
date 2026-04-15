"""Probe the native FA4 import chain with isolated compatibility shims.

This script does not use the stable fallback under ``shims/``. It prepends the
repo-local runtime package plus the probe shim tree so we can measure how far
the real native import path gets once the missing Windows runtime pieces are
shimmed for import-time purposes.
"""

from __future__ import annotations

import sys
import traceback

from _native_probe_setup import (
    ensure_native_fa4_patch,
    install_native_probe_paths,
    loaded_cutlass_shim_modules,
)


def main() -> int:
    runtime_src, shim_root = install_native_probe_paths()
    patched_target = ensure_native_fa4_patch()

    print(f"native_probe_shims={shim_root}")
    print(f"cutlass_runtime_src={runtime_src}")
    print(f"patched_interface={patched_target}")
    try:
        import flash_attn.cute as fa4  # noqa: F401
        import cutlass
        import cutlass.cute as cute
    except Exception:
        traceback.print_exc()
        return 1

    print("native_import=ok")
    print(f"flash_attn.cute={fa4.__file__}")
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"cutlass_probe_init={getattr(cutlass, 'NATIVE_PROBE_CUTLASS_INIT', '<unknown>')}")
    print(f"cutlass_cute_file={getattr(cute, '__file__', '<unknown>')}")
    print(f"pycute_loaded={'pycute' in sys.modules}")
    shim_modules = loaded_cutlass_shim_modules()
    print(f"cutlass_shim_module_count={len(shim_modules)}")
    for name, path in shim_modules:
        print(f"cutlass_shim_module={name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
