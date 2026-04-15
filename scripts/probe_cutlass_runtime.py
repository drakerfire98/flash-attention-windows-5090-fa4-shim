"""Diagnose which CUTLASS runtime the native FA4 probe is actually using."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<missing>"


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "native_probe_shims"))

    import cutlass
    import cutlass.cute as cute

    print(f"python={sys.version.split()[0]}")
    print(f"cutlass_dist={_dist_version('cutlass')}")
    print(f"nvidia_cutlass_dsl_dist={_dist_version('nvidia-cutlass-dsl')}")
    print(f"flash_attn_4_dist={_dist_version('flash-attn-4')}")
    print(f"cutlass_file={getattr(cutlass, '__file__', '<missing>')}")
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_init={getattr(cutlass, 'NATIVE_PROBE_CUTLASS_INIT', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"cutlass_probe_versions={getattr(cutlass, 'NATIVE_PROBE_DIST_VERSIONS', {})}")
    print(f"cute_file={getattr(cute, '__file__', '<missing>')}")
    print(f"cute_compile_type={type(getattr(cute, 'compile', None)).__name__}")
    print(f"cute_compile_repr={repr(getattr(cute, 'compile', None))}")

    runtime_cuda = getattr(getattr(getattr(cutlass, "base_dsl", None), "runtime", None), "cuda", None)
    load_cubin = getattr(runtime_cuda, "load_cubin_module_data", None)
    print(f"load_cubin_type={type(load_cubin).__name__}")
    print(f"load_cubin_repr={repr(load_cubin)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
