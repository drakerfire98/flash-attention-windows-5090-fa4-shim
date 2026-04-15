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
    install_native_probe_paths,
    loaded_cutlass_shim_modules,
    loaded_flash_attn_overlay_modules,
    native_flash_attn_interface_path,
)


def main() -> int:
    flash_attn_runtime_src, runtime_src, shim_root = install_native_probe_paths()
    interface_target = native_flash_attn_interface_path()

    print(f"native_probe_shims={shim_root}")
    print(f"flash_attn_runtime_src={flash_attn_runtime_src}")
    print(f"cutlass_runtime_src={runtime_src}")
    print(f"native_interface={interface_target}")
    try:
        import flash_attn.cute as fa4  # noqa: F401
        import flash_attn.cute.interface as fa4_interface
        import cutlass
        import cutlass.cute as cute
    except Exception:
        traceback.print_exc()
        return 1

    print("native_import=ok")
    print(f"flash_attn.cute={fa4.__file__}")
    print(f"flash_attn.cute.interface={fa4_interface.__file__}")
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"cutlass_probe_init={getattr(cutlass, 'NATIVE_PROBE_CUTLASS_INIT', '<unknown>')}")
    print(f"cutlass_cute_file={getattr(cute, '__file__', '<unknown>')}")
    print(f"pycute_loaded={'pycute' in sys.modules}")
    shim_modules = loaded_cutlass_shim_modules()
    print(f"cutlass_shim_module_count={len(shim_modules)}")
    for name, path in shim_modules:
        print(f"cutlass_shim_module={name} -> {path}")
    overlay_modules = loaded_flash_attn_overlay_modules()
    print(f"flash_attn_overlay_module_count={len(overlay_modules)}")
    for name, path in overlay_modules:
        print(f"flash_attn_overlay_module={name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
