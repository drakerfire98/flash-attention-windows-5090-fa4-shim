"""Compiled backward preprocess/postprocess helpers for the Windows FA4 path."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import torch

from ._native_backend import _ensure_msvc_env


_NATIVE_BWD_HELPERS_MODULE: ModuleType | None = None
_NATIVE_BWD_HELPERS_ERROR: str | None = None
_NATIVE_BWD_HELPERS_LOCK = threading.Lock()
_NATIVE_BWD_HELPERS_LAST_OP: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_bwd_helpers_source_path() -> Path:
    return Path(__file__).resolve().with_name("_native_bwd_helpers_backend.cpp")


def native_bwd_helpers_setup_path() -> Path:
    return Path(__file__).resolve().with_name("_native_bwd_helpers_setup.py")


def native_bwd_helpers_build_dir() -> Path:
    return _repo_root() / ".torch_extensions" / "fa4_win_bwdh"


def native_bwd_helpers_extension_name() -> str:
    return "fa4_win_bwdh_ext"


def _load_bwd_helpers_extension_module_from_file(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(native_bwd_helpers_extension_name(), str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[native_bwd_helpers_extension_name()] = module
    spec.loader.exec_module(module)
    return module


def load_native_bwd_helpers_backend(*, verbose: bool = False, force_rebuild: bool = False) -> ModuleType | None:
    global _NATIVE_BWD_HELPERS_ERROR, _NATIVE_BWD_HELPERS_MODULE
    if os.environ.get("FA4_WINDOWS_NATIVE_BWD_HELPERS_DISABLE", "").strip() == "1":
        _NATIVE_BWD_HELPERS_ERROR = "disabled via FA4_WINDOWS_NATIVE_BWD_HELPERS_DISABLE=1"
        return None
    if _NATIVE_BWD_HELPERS_MODULE is not None and not force_rebuild:
        return _NATIVE_BWD_HELPERS_MODULE

    with _NATIVE_BWD_HELPERS_LOCK:
        if _NATIVE_BWD_HELPERS_MODULE is not None and not force_rebuild:
            return _NATIVE_BWD_HELPERS_MODULE
        try:
            build_dir = native_bwd_helpers_build_dir()
            build_dir.mkdir(parents=True, exist_ok=True)
            scripts_dir = str(Path(sys.executable).resolve().parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if scripts_dir not in path_entries:
                os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
            _ensure_msvc_env()
            module_candidates = list(build_dir.glob(f"{native_bwd_helpers_extension_name()}*.pyd"))
            if force_rebuild:
                for candidate in module_candidates:
                    candidate.unlink(missing_ok=True)
                module_candidates = []
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime) if module_candidates else None
            source_mtime = max(
                native_bwd_helpers_source_path().stat().st_mtime,
                native_bwd_helpers_setup_path().stat().st_mtime,
            )
            needs_rebuild = newest_module is None or newest_module.stat().st_mtime < source_mtime
            if needs_rebuild:
                build_temp_dir = build_dir / "temp"
                build_temp_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(native_bwd_helpers_setup_path()),
                    "build_ext",
                    "--build-lib",
                    str(build_dir),
                    "--build-temp",
                    str(build_temp_dir),
                ]
                if force_rebuild:
                    cmd.append("--force")
                proc = subprocess.run(
                    cmd,
                    cwd=str(native_bwd_helpers_setup_path().parent),
                    check=False,
                    capture_output=not verbose,
                    text=True,
                    env={
                        **os.environ,
                        "DISTUTILS_USE_SDK": "1",
                        "MSSdk": "1",
                    },
                )
                if proc.returncode != 0:
                    stderr = proc.stderr or ""
                    stdout = proc.stdout or ""
                    raise RuntimeError(
                        "native backward helpers build failed\n"
                        f"stdout:\n{stdout[-4000:]}\n"
                        f"stderr:\n{stderr[-4000:]}"
                    )
                module_candidates = list(build_dir.glob(f"{native_bwd_helpers_extension_name()}*.pyd"))
            if not module_candidates:
                raise RuntimeError(
                    f"native backward helpers build produced no {native_bwd_helpers_extension_name()}*.pyd"
                )
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime)
            _NATIVE_BWD_HELPERS_MODULE = _load_bwd_helpers_extension_module_from_file(newest_module)
            _NATIVE_BWD_HELPERS_ERROR = None
        except Exception as exc:  # pragma: no cover - runtime build failures are environment-specific
            _NATIVE_BWD_HELPERS_MODULE = None
            _NATIVE_BWD_HELPERS_ERROR = f"{type(exc).__name__}: {exc}"
        return _NATIVE_BWD_HELPERS_MODULE


def native_bwd_helpers_backend_status() -> dict[str, object]:
    module = _NATIVE_BWD_HELPERS_MODULE
    module_candidates = sorted(
        str(path) for path in native_bwd_helpers_build_dir().glob(f"{native_bwd_helpers_extension_name()}*.pyd")
    )
    return {
        "name": native_bwd_helpers_extension_name(),
        "source": str(native_bwd_helpers_source_path()),
        "setup": str(native_bwd_helpers_setup_path()),
        "build_dir": str(native_bwd_helpers_build_dir()),
        "built": bool(module_candidates),
        "built_candidates": module_candidates,
        "loaded": module is not None,
        "module_file": getattr(module, "__file__", None) if module is not None else None,
        "error": _NATIVE_BWD_HELPERS_ERROR,
        "last_op": _NATIVE_BWD_HELPERS_LAST_OP,
    }


def flash_attn_backward_preprocess_zero_native(
    dpsum: torch.Tensor | None,
    lse_log2: torch.Tensor | None,
    dq_accum: torch.Tensor | None,
    *,
    verbose_build: bool = False,
) -> None:
    global _NATIVE_BWD_HELPERS_LAST_OP
    backend = load_native_bwd_helpers_backend(verbose=verbose_build)
    if backend is None:
        status = native_bwd_helpers_backend_status()
        raise RuntimeError(f"native backward helpers backend unavailable: {status['error']}")
    backend.flash_attn_backward_preprocess_zero(dpsum, lse_log2, dq_accum)
    _NATIVE_BWD_HELPERS_LAST_OP = "preprocess_zero"


def flash_attn_backward_postprocess_copy_native(
    accum: torch.Tensor,
    output: torch.Tensor,
    *,
    verbose_build: bool = False,
) -> None:
    global _NATIVE_BWD_HELPERS_LAST_OP
    backend = load_native_bwd_helpers_backend(verbose=verbose_build)
    if backend is None:
        status = native_bwd_helpers_backend_status()
        raise RuntimeError(f"native backward helpers backend unavailable: {status['error']}")
    backend.flash_attn_backward_postprocess_copy(accum, output)
    _NATIVE_BWD_HELPERS_LAST_OP = "postprocess_copy"
