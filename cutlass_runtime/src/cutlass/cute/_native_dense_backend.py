"""Compiled dense backend helpers for the Windows FA4 runtime path."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch

from ._native_backend import _ensure_msvc_env


_NATIVE_DENSE_MODULE: ModuleType | None = None
_NATIVE_DENSE_ERROR: str | None = None
_NATIVE_DENSE_LOCK = threading.Lock()
_NATIVE_DENSE_LAST_CALL: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_dense_source_path() -> Path:
    return Path(__file__).resolve().with_name("_native_dense_backend.cpp")


def native_dense_setup_path() -> Path:
    return Path(__file__).resolve().with_name("_native_dense_setup.py")


def native_dense_build_dir() -> Path:
    return _repo_root() / ".torch_extensions" / "fa4_windows_native_dense"


def native_dense_extension_name() -> str:
    return "fa4_windows_native_dense_ext"


def _load_dense_extension_module_from_file(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(native_dense_extension_name(), str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[native_dense_extension_name()] = module
    spec.loader.exec_module(module)
    return module


def load_native_dense_backend(*, verbose: bool = False, force_rebuild: bool = False) -> ModuleType | None:
    global _NATIVE_DENSE_ERROR, _NATIVE_DENSE_MODULE
    if os.environ.get("FA4_WINDOWS_NATIVE_DENSE_DISABLE", "").strip() == "1":
        _NATIVE_DENSE_ERROR = "disabled via FA4_WINDOWS_NATIVE_DENSE_DISABLE=1"
        return None
    if _NATIVE_DENSE_MODULE is not None and not force_rebuild:
        return _NATIVE_DENSE_MODULE

    with _NATIVE_DENSE_LOCK:
        if _NATIVE_DENSE_MODULE is not None and not force_rebuild:
            return _NATIVE_DENSE_MODULE
        try:
            build_dir = native_dense_build_dir()
            build_dir.mkdir(parents=True, exist_ok=True)
            scripts_dir = str(Path(sys.executable).resolve().parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if scripts_dir not in path_entries:
                os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
            _ensure_msvc_env()
            module_candidates = list(build_dir.glob(f"{native_dense_extension_name()}*.pyd"))
            if force_rebuild:
                for candidate in module_candidates:
                    candidate.unlink(missing_ok=True)
                module_candidates = []
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime) if module_candidates else None
            source_mtime = max(
                native_dense_source_path().stat().st_mtime,
                native_dense_setup_path().stat().st_mtime,
            )
            needs_rebuild = newest_module is None or newest_module.stat().st_mtime < source_mtime
            if needs_rebuild:
                short_tmp = os.environ.get("FA4_WINDOWS_SHORT_BUILD_TMP", "").strip()
                if short_tmp:
                    build_temp_dir = Path(short_tmp) / native_dense_extension_name()
                else:
                    build_temp_dir = build_dir / "temp"
                build_temp_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(native_dense_setup_path()),
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
                    cwd=str(native_dense_setup_path().parent),
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
                        "native dense build failed\n"
                        f"stdout:\n{stdout[-4000:]}\n"
                        f"stderr:\n{stderr[-4000:]}"
                    )
                module_candidates = list(build_dir.glob(f"{native_dense_extension_name()}*.pyd"))
            if not module_candidates:
                raise RuntimeError(f"native dense build produced no {native_dense_extension_name()}*.pyd")
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime)
            _NATIVE_DENSE_MODULE = _load_dense_extension_module_from_file(newest_module)
            _NATIVE_DENSE_ERROR = None
        except Exception as exc:  # pragma: no cover - runtime build failures are environment-specific
            _NATIVE_DENSE_MODULE = None
            _NATIVE_DENSE_ERROR = f"{type(exc).__name__}: {exc}"
        return _NATIVE_DENSE_MODULE


def native_dense_backend_status() -> dict[str, object]:
    module = _NATIVE_DENSE_MODULE
    module_candidates = sorted(
        str(path) for path in native_dense_build_dir().glob(f"{native_dense_extension_name()}*.pyd")
    )
    return {
        "name": native_dense_extension_name(),
        "source": str(native_dense_source_path()),
        "setup": str(native_dense_setup_path()),
        "build_dir": str(native_dense_build_dir()),
        "built": bool(module_candidates),
        "built_candidates": module_candidates,
        "loaded": module is not None,
        "module_file": getattr(module, "__file__", None) if module is not None else None,
        "error": _NATIVE_DENSE_ERROR,
        "last_call": _NATIVE_DENSE_LAST_CALL,
    }


def flash_attn_dense_forward_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    extra_keep_mask: Optional[torch.Tensor] = None,
    extra_score_bias: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    return_lse: bool = False,
    verbose_build: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    global _NATIVE_DENSE_LAST_CALL

    backend = load_native_dense_backend(verbose=verbose_build)
    if backend is None:
        status = native_dense_backend_status()
        raise RuntimeError(f"native dense backend unavailable: {status['error']}")

    scale = float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5)
    window_left = -1 if window_size[0] is None else int(window_size[0])
    window_right = -1 if window_size[1] is None else int(window_size[1])
    sink_arg: torch.Tensor | None = None
    if learnable_sink is not None:
        if learnable_sink.ndim != 1:
            raise ValueError("learnable_sink must be a 1D tensor")
        sink_arg = learnable_sink.contiguous()
    keep_mask_arg: torch.Tensor | None = None
    if extra_keep_mask is not None:
        keep_mask_arg = extra_keep_mask.to(device=q.device, dtype=torch.bool).contiguous()
    score_bias_arg: torch.Tensor | None = None
    if extra_score_bias is not None:
        score_bias_arg = extra_score_bias.to(device=q.device, dtype=torch.float32).contiguous()
    native_out, native_lse = backend.flash_attn_dense_forward(
        q,
        k,
        v,
        scale,
        bool(causal),
        window_left,
        window_right,
        float(softcap),
        sink_arg,
        keep_mask_arg,
        score_bias_arg,
    )
    call_parts: list[str] = []
    if softcap > 0.0:
        call_parts.append("softcap")
    if learnable_sink is not None:
        call_parts.append("learnable_sink")
    if window_size != (None, None):
        call_parts.append("window")
    if extra_keep_mask is not None:
        call_parts.append("extra_keep_mask")
    if extra_score_bias is not None:
        call_parts.append("extra_score_bias")
    if not call_parts:
        call_parts.append("plain")
    _NATIVE_DENSE_LAST_CALL = "+".join(call_parts)

    if out is None:
        out = native_out
    else:
        if out.shape != native_out.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} does not match dense output shape {tuple(native_out.shape)}"
            )
        out.copy_(native_out.to(dtype=out.dtype))

    if not return_lse:
        return out, None
    return out, native_lse
