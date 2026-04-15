"""Diagnose which CUTLASS runtime the native FA4 probe is actually using."""

from __future__ import annotations

import importlib
import sys
from importlib import util
from importlib import metadata

from _native_probe_setup import (
    install_native_probe_paths,
    loaded_cutlass_shim_modules,
    loaded_flash_attn_overlay_modules,
    native_flash_attn_interface_path,
)


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
    flash_attn_runtime_src, runtime_src, shim_root = install_native_probe_paths()
    interface_target = native_flash_attn_interface_path()

    import cutlass
    import cutlass.cute as cute
    import cutlass.cute._compile_bridge as cute_compile_bridge
    from cutlass.cute._native_backend import native_combine_backend_status
    from cutlass.cute._native_dense_backend import native_dense_backend_status
    import cutlass.base_dsl.runtime.cuda as runtime_cuda_module
    import flash_attn.cute as fa4
    import flash_attn.cute.interface as fa4_interface

    print(f"cutlass_dist={_dist_version('cutlass')}")
    print(f"nvidia_cutlass_dsl_dist={_dist_version('nvidia-cutlass-dsl')}")
    print(f"flash_attn_4_dist={_dist_version('flash-attn-4')}")
    print(f"native_probe_shims={shim_root}")
    print(f"flash_attn_runtime_src={flash_attn_runtime_src}")
    print(f"cutlass_runtime_src={runtime_src}")
    print(f"native_interface={interface_target}")
    print(f"cutlass_file={getattr(cutlass, '__file__', '<missing>')}")
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_init={getattr(cutlass, 'NATIVE_PROBE_CUTLASS_INIT', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"cutlass_runtime_owned_modules={getattr(cutlass, 'NATIVE_PROBE_RUNTIME_OWNED_MODULES', ())}")
    print(f"cutlass_fallback_roots={getattr(cutlass, 'NATIVE_PROBE_FALLBACK_ROOTS', ())}")
    print(f"cutlass_probe_versions={getattr(cutlass, 'NATIVE_PROBE_DIST_VERSIONS', {})}")
    print(f"cute_file={getattr(cute, '__file__', '<missing>')}")
    print(f"cute_compile_bridge_file={getattr(cute_compile_bridge, '__file__', '<missing>')}")
    print(f"cute_compile_type={type(getattr(cute, 'compile', None)).__name__}")
    print(f"cute_compile_repr={repr(getattr(cute, 'compile', None))}")
    print(f"pycute_loaded={'pycute' in sys.modules}")
    print(f"runtime_cuda_file={getattr(runtime_cuda_module, '__file__', '<missing>')}")
    print(f"flash_attn_cute_file={getattr(fa4, '__file__', '<missing>')}")
    print(f"flash_attn_cute_interface_file={getattr(fa4_interface, '__file__', '<missing>')}")
    print(f"native_combine_backend={native_combine_backend_status()}")
    print(f"native_dense_backend={native_dense_backend_status()}")

    runtime_cuda = getattr(getattr(getattr(cutlass, "base_dsl", None), "runtime", None), "cuda", None)
    load_cubin = getattr(runtime_cuda, "load_cubin_module_data", None)
    print(f"load_cubin_type={type(load_cubin).__name__}")
    print(f"load_cubin_repr={repr(load_cubin)}")
    shim_modules = loaded_cutlass_shim_modules()
    print(f"cutlass_shim_module_count={len(shim_modules)}")
    for name, path in shim_modules:
        print(f"cutlass_shim_module={name} -> {path}")
    overlay_modules = loaded_flash_attn_overlay_modules()
    print(f"flash_attn_overlay_module_count={len(overlay_modules)}")
    for name, path in overlay_modules:
        print(f"flash_attn_overlay_module={name} -> {path}")

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
    if shim_modules:
        blockers.append("some cutlass modules still resolved from native_probe_shims")
    if "flash_attn_runtime\\src\\flash_attn\\cute\\__init__.py" not in str(getattr(fa4, "__file__", "")).lower():
        blockers.append("flash_attn.cute is not resolving from the repo-local runtime overlay")
    if "flash_attn_runtime\\src\\flash_attn\\cute\\interface.py" not in str(getattr(fa4_interface, "__file__", "")).lower():
        blockers.append("flash_attn.cute.interface is not resolving from the repo-local runtime overlay")
    print(f"modern_runtime_ready={len(blockers) == 0}")
    print(f"modern_runtime_blockers={blockers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
