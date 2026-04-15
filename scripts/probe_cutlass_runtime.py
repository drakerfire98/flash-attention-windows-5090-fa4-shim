"""Diagnose which CUTLASS runtime the native FA4 probe is actually using."""

from __future__ import annotations

import importlib
import sys
from importlib import util
from importlib import metadata
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<missing>"


def _dist_requires(name: str) -> list[str]:
    try:
        return metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return []


def _spec_origin(name: str) -> str:
    spec = util.find_spec(name)
    if spec is None:
        return "<missing>"
    return str(spec.origin)


def _import_status(name: str) -> str:
    stale_keys = [key for key in sys.modules if key == name or key.startswith(f"{name}.")]
    for key in stale_keys:
        sys.modules.pop(key, None)
    try:
        importlib.import_module(name)
        return "ok"
    except Exception as exc:  # pragma: no cover - this is a probe script
        return f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_keys = [key for key in sys.modules if key == name or key.startswith(f"{name}.")]
        for key in cleanup_keys:
            sys.modules.pop(key, None)


def main() -> int:
    repo_root = _repo_root()
    raw_cutlass_spec = _spec_origin("cutlass")
    raw_cutlass_import = _import_status("cutlass")
    raw_cuda_spec = _spec_origin("cuda")
    raw_cuda_import = _import_status("cuda")
    raw_cuda_bindings_driver_spec = _spec_origin("cuda.bindings.driver")
    raw_nvidia_cutlass_dsl_spec = _spec_origin("nvidia_cutlass_dsl")
    raw_nvidia_cutlass_dsl_import = _import_status("nvidia_cutlass_dsl")
    print(f"python={sys.version.split()[0]}")
    print(f"raw_cutlass_spec={raw_cutlass_spec}")
    print(f"raw_cutlass_import={raw_cutlass_import}")
    print(f"raw_cuda_spec={raw_cuda_spec}")
    print(f"raw_cuda_import={raw_cuda_import}")
    print(f"raw_cuda_bindings_driver_spec={raw_cuda_bindings_driver_spec}")
    print(f"raw_nvidia_cutlass_dsl_spec={raw_nvidia_cutlass_dsl_spec}")
    print(f"raw_nvidia_cutlass_dsl_import={raw_nvidia_cutlass_dsl_import}")
    print(f"raw_flash_attn_4_dist={_dist_version('flash-attn-4')}")
    print(f"raw_cutlass_dist={_dist_version('cutlass')}")
    print(f"raw_nvidia_cutlass_dsl_dist={_dist_version('nvidia-cutlass-dsl')}")
    print(f"nvidia_cutlass_dsl_requires={_dist_requires('nvidia-cutlass-dsl')}")
    sys.path.insert(0, str(repo_root / "cutlass_runtime" / "src"))

    import cutlass
    import cutlass.cute as cute
    import cutlass.cute._compile_bridge as cute_compile_bridge
    import cutlass.base_dsl.runtime.cuda as runtime_cuda_module

    print(f"cutlass_dist={_dist_version('cutlass')}")
    print(f"nvidia_cutlass_dsl_dist={_dist_version('nvidia-cutlass-dsl')}")
    print(f"flash_attn_4_dist={_dist_version('flash-attn-4')}")
    print(f"cutlass_file={getattr(cutlass, '__file__', '<missing>')}")
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_init={getattr(cutlass, 'NATIVE_PROBE_CUTLASS_INIT', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"cutlass_probe_versions={getattr(cutlass, 'NATIVE_PROBE_DIST_VERSIONS', {})}")
    print(f"cute_file={getattr(cute, '__file__', '<missing>')}")
    print(f"cute_compile_bridge_file={getattr(cute_compile_bridge, '__file__', '<missing>')}")
    print(f"cute_compile_type={type(getattr(cute, 'compile', None)).__name__}")
    print(f"cute_compile_repr={repr(getattr(cute, 'compile', None))}")
    print(f"pycute_loaded={'pycute' in sys.modules}")
    print(f"runtime_cuda_file={getattr(runtime_cuda_module, '__file__', '<missing>')}")

    runtime_cuda = getattr(getattr(getattr(cutlass, "base_dsl", None), "runtime", None), "cuda", None)
    load_cubin = getattr(runtime_cuda, "load_cubin_module_data", None)
    print(f"load_cubin_type={type(load_cubin).__name__}")
    print(f"load_cubin_repr={repr(load_cubin)}")

    blockers: list[str] = []
    if "third_party\\flash-attention-for-windows\\csrc\\cutlass\\python\\cutlass\\__init__.py" in raw_cutlass_spec.lower():
        blockers.append("raw cutlass resolves to the legacy editable CUTLASS tree")
    if raw_cutlass_import != "ok":
        blockers.append(f"raw cutlass import still fails before shims: {raw_cutlass_import}")
    if raw_nvidia_cutlass_dsl_spec == "<missing>":
        blockers.append("no importable nvidia_cutlass_dsl module is present on sys.path")
    if not str(raw_cuda_bindings_driver_spec).endswith(".pyd"):
        blockers.append("cuda.bindings.driver binary runtime is not importable")
    if raw_cuda_import != "ok":
        blockers.append("top-level cuda package does not expose the modern runtime surface directly")
    if getattr(cutlass, "NATIVE_PROBE_MODE", None) not in {"modern-cutlass-package", "runtime-local-core"}:
        blockers.append("native probe still requires legacy-editable CUTLASS plus shims")
    if "pycute" in sys.modules:
        blockers.append("cutlass.cute still imported external pycute")
    if "cutlass_runtime\\src\\cutlass\\cute\\__init__.py" not in str(getattr(cute, "__file__", "")).lower():
        blockers.append("cutlass.cute is not resolving from the repo-local runtime package")
    if "cutlass_runtime\\src\\cutlass\\cute\\_compile_bridge.py" not in str(getattr(cute_compile_bridge, "__file__", "")).lower():
        blockers.append("cutlass.cute._compile_bridge is not resolving from the repo-local runtime package")
    if "cutlass_runtime\\src\\cutlass\\base_dsl\\runtime\\cuda.py" not in str(getattr(runtime_cuda_module, "__file__", "")).lower():
        blockers.append("cutlass.base_dsl.runtime.cuda is not resolving from the repo-local runtime package")
    print(f"modern_runtime_ready={len(blockers) == 0}")
    print(f"modern_runtime_blockers={blockers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
