"""Compiled varlen backend helpers for the Windows FA4 runtime path."""

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


_NATIVE_VARLEN_MODULE: ModuleType | None = None
_NATIVE_VARLEN_ERROR: str | None = None
_NATIVE_VARLEN_LOCK = threading.Lock()
_NATIVE_VARLEN_LAST_CALL: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_varlen_source_path() -> Path:
    return Path(__file__).resolve().with_name("_native_varlen_backend.cpp")


def native_varlen_setup_path() -> Path:
    return Path(__file__).resolve().with_name("_native_varlen_setup.py")


def native_varlen_build_dir() -> Path:
    return _repo_root() / ".torch_extensions" / "fa4_win_varlen"


def native_varlen_extension_name() -> str:
    return "fa4_win_varlen_ext"


def _load_varlen_extension_module_from_file(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(native_varlen_extension_name(), str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[native_varlen_extension_name()] = module
    spec.loader.exec_module(module)
    return module


def load_native_varlen_backend(*, verbose: bool = False, force_rebuild: bool = False) -> ModuleType | None:
    global _NATIVE_VARLEN_ERROR, _NATIVE_VARLEN_MODULE
    if os.environ.get("FA4_WINDOWS_NATIVE_VARLEN_DISABLE", "").strip() == "1":
        _NATIVE_VARLEN_ERROR = "disabled via FA4_WINDOWS_NATIVE_VARLEN_DISABLE=1"
        return None
    if _NATIVE_VARLEN_MODULE is not None and not force_rebuild:
        return _NATIVE_VARLEN_MODULE

    with _NATIVE_VARLEN_LOCK:
        if _NATIVE_VARLEN_MODULE is not None and not force_rebuild:
            return _NATIVE_VARLEN_MODULE
        try:
            build_dir = native_varlen_build_dir()
            build_dir.mkdir(parents=True, exist_ok=True)
            scripts_dir = str(Path(sys.executable).resolve().parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if scripts_dir not in path_entries:
                os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
            _ensure_msvc_env()
            module_candidates = list(build_dir.glob(f"{native_varlen_extension_name()}*.pyd"))
            if force_rebuild:
                for candidate in module_candidates:
                    candidate.unlink(missing_ok=True)
                module_candidates = []
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime) if module_candidates else None
            source_mtime = max(
                native_varlen_source_path().stat().st_mtime,
                native_varlen_setup_path().stat().st_mtime,
            )
            needs_rebuild = newest_module is None or newest_module.stat().st_mtime < source_mtime
            if needs_rebuild:
                short_tmp = os.environ.get("FA4_WINDOWS_SHORT_BUILD_TMP", "").strip()
                if short_tmp:
                    build_temp_dir = Path(short_tmp) / native_varlen_extension_name()
                else:
                    build_temp_dir = build_dir / "temp"
                build_temp_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(native_varlen_setup_path()),
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
                    cwd=str(native_varlen_setup_path().parent),
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
                        "native varlen build failed\n"
                        f"stdout:\n{stdout[-4000:]}\n"
                        f"stderr:\n{stderr[-4000:]}"
                    )
                module_candidates = list(build_dir.glob(f"{native_varlen_extension_name()}*.pyd"))
            if not module_candidates:
                raise RuntimeError(f"native varlen build produced no {native_varlen_extension_name()}*.pyd")
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime)
            _NATIVE_VARLEN_MODULE = _load_varlen_extension_module_from_file(newest_module)
            _NATIVE_VARLEN_ERROR = None
        except Exception as exc:  # pragma: no cover - runtime build failures are environment-specific
            _NATIVE_VARLEN_MODULE = None
            _NATIVE_VARLEN_ERROR = f"{type(exc).__name__}: {exc}"
        return _NATIVE_VARLEN_MODULE


def native_varlen_backend_status() -> dict[str, object]:
    module = _NATIVE_VARLEN_MODULE
    module_candidates = sorted(
        str(path) for path in native_varlen_build_dir().glob(f"{native_varlen_extension_name()}*.pyd")
    )
    return {
        "name": native_varlen_extension_name(),
        "source": str(native_varlen_source_path()),
        "setup": str(native_varlen_setup_path()),
        "build_dir": str(native_varlen_build_dir()),
        "built": bool(module_candidates),
        "built_candidates": module_candidates,
        "loaded": module is not None,
        "module_file": getattr(module, "__file__", None) if module is not None else None,
        "error": _NATIVE_VARLEN_ERROR,
        "last_call": _NATIVE_VARLEN_LAST_CALL,
    }


def _materialize_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if page_table.ndim != 2:
        raise ValueError("page_table must be shaped (batch, max_num_pages_per_seq)")
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must use dtype torch.int32")
    if page_table.device != k.device:
        page_table = page_table.to(device=k.device)
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("paged KV expects k and v shaped (num_pages, page_size, heads, dim)")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError("paged KV expects k and v to share page geometry and KV heads")

    num_pages, page_size, num_heads = k.shape[:3]
    table = page_table.to(dtype=torch.long)
    if table.numel() > 0:
        table_min = int(table.min().item())
        table_max = int(table.max().item())
        if table_min < 0 or table_max >= num_pages:
            raise ValueError("page_table contains page indices outside the available KV cache range")

    batch_size, max_num_pages = table.shape
    gathered_k = k.index_select(0, table.reshape(-1)).reshape(
        batch_size, max_num_pages * page_size, num_heads, k.shape[-1]
    )
    gathered_v = v.index_select(0, table.reshape(-1)).reshape(
        batch_size, max_num_pages * page_size, num_heads, v.shape[-1]
    )
    return gathered_k, gathered_v


def _prepare_varlen_kv_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if page_table is None:
        return k, v, cu_seqlens_k, seqused_k
    if cu_seqlens_k is not None:
        raise ValueError("page_table is not supported together with cu_seqlens_k")
    k_dense, v_dense = _materialize_paged_kv_cache(k, v, page_table)
    return k_dense, v_dense, None, seqused_k


def _build_varlen_layout(
    tensor: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    name: str,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1:
            raise ValueError(f"{name} cu_seqlens must be 1D")
        if tensor.ndim != 3:
            raise ValueError(
                f"packed varlen {name} must be shaped (total, heads, dim) when cu_seqlens are provided"
            )
        batch = int(cu_seqlens.numel()) - 1
        if batch < 0:
            raise ValueError(f"{name} cu_seqlens must have at least one element")
        starts = cu_seqlens[:-1].to(dtype=torch.int64, device="cpu").contiguous()
        ends = cu_seqlens[1:].to(dtype=torch.int64, device="cpu").contiguous()
        spans = ends - starts
        if bool((spans < 0).any().item()):
            raise ValueError(f"{name} cu_seqlens must be nondecreasing")
        if seqused is not None:
            if seqused.ndim != 1 or int(seqused.numel()) != batch:
                raise ValueError(f"{name} seqused must be 1D with one length per batch item")
            used = torch.minimum(
                spans,
                seqused.to(dtype=torch.int64, device="cpu").contiguous().clamp(min=0),
            )
        else:
            used = spans
        return True, starts, used

    if tensor.ndim != 4:
        raise ValueError(
            f"padded varlen {name} must be shaped (batch, seqlen, heads, dim) when cu_seqlens are absent"
        )
    batch = int(tensor.shape[0])
    padded_length = int(tensor.shape[1])
    if seqused is not None:
        if seqused.ndim != 1 or int(seqused.numel()) != batch:
            raise ValueError(f"{name} seqused must be 1D with one length per batch item")
        used = seqused.to(dtype=torch.int64, device="cpu").contiguous().clamp(min=0, max=padded_length)
    else:
        used = torch.full((batch,), padded_length, dtype=torch.int64)
    starts = torch.zeros((batch,), dtype=torch.int64)
    return False, starts, used


def flash_attn_varlen_forward_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
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
    global _NATIVE_VARLEN_LAST_CALL

    backend = load_native_varlen_backend(verbose=verbose_build)
    if backend is None:
        status = native_varlen_backend_status()
        raise RuntimeError(f"native varlen backend unavailable: {status['error']}")

    k_input, v_input, cu_seqlens_k_input, seqused_k_input = _prepare_varlen_kv_inputs(
        k,
        v,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        page_table=page_table,
    )
    q_packed, q_starts, q_used = _build_varlen_layout(
        q,
        cu_seqlens=cu_seqlens_q,
        seqused=seqused_q,
        name="q",
    )
    k_packed, k_starts, k_used = _build_varlen_layout(
        k_input,
        cu_seqlens=cu_seqlens_k_input,
        seqused=seqused_k_input,
        name="k",
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
    score_bias_arg: torch.Tensor | None = None
    if extra_score_bias is not None:
        score_bias_arg = extra_score_bias.to(device=q.device, dtype=torch.float32).contiguous()

    native_out, native_lse = backend.flash_attn_varlen_forward(
        q,
        k_input,
        v_input,
        q_starts,
        q_used,
        bool(q_packed),
        k_starts,
        k_used,
        bool(k_packed),
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
    if extra_score_bias is not None:
        call_parts.append("extra_score_bias")
    _NATIVE_VARLEN_LAST_CALL = "+".join(call_parts)

    if out is None:
        out = native_out
    else:
        if out.shape != native_out.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} does not match varlen output shape {tuple(native_out.shape)}"
            )
        out.copy_(native_out.to(dtype=out.dtype))

    if not return_lse:
        return out, None
    return out, native_lse
