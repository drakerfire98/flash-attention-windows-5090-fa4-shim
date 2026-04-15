"""Probe the native FA4 import chain with isolated compatibility shims.

This script does not use the stable fallback under ``shims/``. It prepends the
repo-local runtime package plus the probe shim tree so we can measure how far
the real native import path gets once the missing Windows runtime pieces are
shimmed for import-time purposes.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "native_probe_shims"))
    sys.path.insert(0, str(repo_root / "cutlass_runtime" / "src"))

    print(f"native_probe_shims={repo_root / 'native_probe_shims'}")
    print(f"cutlass_runtime_src={repo_root / 'cutlass_runtime' / 'src'}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
