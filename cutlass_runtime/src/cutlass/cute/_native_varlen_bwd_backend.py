"""Compiled varlen backward backend helpers for the Windows FA4 runtime path."""

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
from ._native_varlen_backend import _build_varlen_layout


_NATIVE_VARLEN_BWD_MODULE: ModuleType | None = None
_NATIVE_VARLEN_BWD_ERROR: str | None = None
_NATIVE_VARLEN_BWD_LOCK = threading.Lock()
_NATIVE_VARLEN_BWD_LAST_CALL: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_varlen_bwd_source_path() -> Path:
    return Path(__file__).resolve().with_name("_native_varlen_bwd_backend.cpp")


def native_varlen_bwd_setup_path() -> Path:
    return Path(__file__).resolve().with_name("_native_varlen_bwd_setup.py")


def native_varlen_bwd_build_dir() -> Path:
    return _repo_root() / ".torch_extensions" / "fa4_win_vbwd"


def native_varlen_bwd_extension_name() -> str:
    return "fa4_win_vbwd_ext"


def _load_varlen_bwd_extension_module_from_file(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(native_varlen_bwd_extension_name(), str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[native_varlen_bwd_extension_name()] = module
    spec.loader.exec_module(module)
    return module


def load_native_varlen_bwd_backend(*, verbose: bool = False, force_rebuild: bool = False) -> ModuleType | None:
    global _NATIVE_VARLEN_BWD_ERROR, _NATIVE_VARLEN_BWD_MODULE
    if os.environ.get("FA4_WINDOWS_NATIVE_VARLEN_BWD_DISABLE", "").strip() == "1":
        _NATIVE_VARLEN_BWD_ERROR = "disabled via FA4_WINDOWS_NATIVE_VARLEN_BWD_DISABLE=1"
        return None
    if _NATIVE_VARLEN_BWD_MODULE is not None and not force_rebuild:
        return _NATIVE_VARLEN_BWD_MODULE

    with _NATIVE_VARLEN_BWD_LOCK:
        if _NATIVE_VARLEN_BWD_MODULE is not None and not force_rebuild:
            return _NATIVE_VARLEN_BWD_MODULE
        try:
            build_dir = native_varlen_bwd_build_dir()
            build_dir.mkdir(parents=True, exist_ok=True)
            scripts_dir = str(Path(sys.executable).resolve().parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if scripts_dir not in path_entries:
                os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
            _ensure_msvc_env()
            module_candidates = list(build_dir.glob(f"{native_varlen_bwd_extension_name()}*.pyd"))
            if force_rebuild:
                for candidate in module_candidates:
                    candidate.unlink(missing_ok=True)
                module_candidates = []
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime) if module_candidates else None
            source_mtime = max(
                native_varlen_bwd_source_path().stat().st_mtime,
                native_varlen_bwd_setup_path().stat().st_mtime,
            )
            needs_rebuild = newest_module is None or newest_module.stat().st_mtime < source_mtime
            if needs_rebuild:
                build_temp_dir = build_dir / "t"
                build_temp_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(native_varlen_bwd_setup_path()),
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
                    cwd=str(native_varlen_bwd_setup_path().parent),
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
                        "native varlen backward build failed\n"
                        f"stdout:\n{stdout[-4000:]}\n"
                        f"stderr:\n{stderr[-4000:]}"
                    )
                module_candidates = list(build_dir.glob(f"{native_varlen_bwd_extension_name()}*.pyd"))
            if not module_candidates:
                raise RuntimeError(
                    f"native varlen backward build produced no {native_varlen_bwd_extension_name()}*.pyd"
                )
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime)
            _NATIVE_VARLEN_BWD_MODULE = _load_varlen_bwd_extension_module_from_file(newest_module)
            _NATIVE_VARLEN_BWD_ERROR = None
        except Exception as exc:  # pragma: no cover - runtime build failures are environment-specific
            _NATIVE_VARLEN_BWD_MODULE = None
            _NATIVE_VARLEN_BWD_ERROR = f"{type(exc).__name__}: {exc}"
        return _NATIVE_VARLEN_BWD_MODULE


def native_varlen_bwd_backend_status() -> dict[str, object]:
    module = _NATIVE_VARLEN_BWD_MODULE
    module_candidates = sorted(
        str(path) for path in native_varlen_bwd_build_dir().glob(f"{native_varlen_bwd_extension_name()}*.pyd")
    )
    return {
        "name": native_varlen_bwd_extension_name(),
        "source": str(native_varlen_bwd_source_path()),
        "setup": str(native_varlen_bwd_setup_path()),
        "build_dir": str(native_varlen_bwd_build_dir()),
        "built": bool(module_candidates),
        "built_candidates": module_candidates,
        "loaded": module is not None,
        "module_file": getattr(module, "__file__", None) if module is not None else None,
        "error": _NATIVE_VARLEN_BWD_ERROR,
        "last_call": _NATIVE_VARLEN_BWD_LAST_CALL,
    }


def _build_paged_kv_layout(
    k: torch.Tensor,
    *,
    seqused_k: Optional[torch.Tensor],
    page_table: torch.Tensor,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    if page_table.ndim != 2:
        raise ValueError("page_table must be shaped (batch, max_num_pages_per_seq)")
    if k.ndim != 4:
        raise ValueError("paged KV expects k shaped (num_pages, page_size, heads, dim)")
    batch = int(page_table.shape[0])
    padded_length = int(page_table.shape[1]) * int(k.shape[1])
    starts = torch.zeros((batch,), dtype=torch.int64)
    if seqused_k is not None:
        if seqused_k.ndim != 1 or int(seqused_k.numel()) != batch:
            raise ValueError("k seqused must be 1D with one length per batch item")
        used = seqused_k.to(dtype=torch.int64, device="cpu").contiguous().clamp(min=0, max=padded_length)
    else:
        used = torch.full((batch,), padded_length, dtype=torch.int64)
    return False, starts, used


def flash_attn_varlen_backward_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    dlse: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    extra_keep_mask: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    verbose_build: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _NATIVE_VARLEN_BWD_LAST_CALL

    backend = load_native_varlen_bwd_backend(verbose=verbose_build)
    if backend is None:
        status = native_varlen_bwd_backend_status()
        raise RuntimeError(f"native varlen backward backend unavailable: {status['error']}")

    q_packed, q_starts, q_used = _build_varlen_layout(
        q,
        cu_seqlens=cu_seqlens_q,
        seqused=seqused_q,
        name="q",
    )
    if page_table is None:
        k_packed, k_starts, k_used = _build_varlen_layout(
            k,
            cu_seqlens=cu_seqlens_k,
            seqused=seqused_k,
            name="k",
        )
    else:
        if cu_seqlens_k is not None:
            raise ValueError("page_table is not supported together with cu_seqlens_k")
        k_packed, k_starts, k_used = _build_paged_kv_layout(
            k,
            seqused_k=seqused_k,
            page_table=page_table,
        )
    if q_used.numel() != k_used.numel():
        raise ValueError("q and kv must describe the same batch size")

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
    dlse_arg: torch.Tensor | None = None if dlse is None else dlse.contiguous()
    page_table_arg: torch.Tensor | None = None
    if page_table is not None:
        page_table_arg = page_table.to(device=k.device, dtype=torch.int32).contiguous()

    dq, dk, dv = backend.flash_attn_varlen_backward(
        q,
        k,
        v,
        q_starts,
        q_used,
        bool(q_packed),
        k_starts,
        k_used,
        bool(k_packed),
        page_table_arg,
        dout.contiguous(),
        dlse_arg,
        scale,
        bool(causal),
        window_left,
        window_right,
        float(softcap),
        sink_arg,
        keep_mask_arg,
    )

    call_parts: list[str] = []
    call_parts.append("packed_q" if q_packed else "padded_q")
    call_parts.append("packed_k" if k_packed else "padded_k")
    if page_table is not None:
        call_parts.append("page_table")
    if softcap > 0.0:
        call_parts.append("softcap")
    if learnable_sink is not None:
        call_parts.append("learnable_sink")
    if window_size != (None, None):
        call_parts.append("window")
    if extra_keep_mask is not None:
        call_parts.append("extra_keep_mask")
    if dlse is not None:
        call_parts.append("lse_grad")
    _NATIVE_VARLEN_BWD_LAST_CALL = "+".join(call_parts)

    return dq, dk, dv
