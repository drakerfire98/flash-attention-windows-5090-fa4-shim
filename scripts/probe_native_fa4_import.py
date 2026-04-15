"""Probe the native FA4 import chain with isolated compatibility shims.

This script does not use the stable fallback under ``shims/``. It prepends the
separate ``native_probe_shims/`` directory so we can measure how far the real
native import path gets once the old CUDA API shape and missing ``cutlass.cute``
package are shimmed for import-time purposes.
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

    print(f"native_probe_shims={repo_root / 'native_probe_shims'}")
    try:
        import flash_attn.cute as fa4  # noqa: F401
        import cutlass
    except Exception:
        traceback.print_exc()
        return 1

    print("native_import=ok")
    print(f"flash_attn.cute={fa4.__file__}")
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"cutlass_probe_init={getattr(cutlass, 'NATIVE_PROBE_CUTLASS_INIT', '<unknown>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
