"""Compiled native backend helpers for the Windows FA4 runtime path."""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Optional

import torch

_NATIVE_COMBINE_MODULE: ModuleType | None = None
_NATIVE_COMBINE_ERROR: str | None = None
_NATIVE_COMBINE_LOCK = threading.Lock()
_MSVC_ENV_READY = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_combine_source_path() -> Path:
    return Path(__file__).resolve().with_name("_native_combine_backend.cpp")


def native_combine_setup_path() -> Path:
    return Path(__file__).resolve().with_name("_native_combine_setup.py")


def native_combine_build_dir() -> Path:
    return _repo_root() / ".torch_extensions" / "fa4_windows_native_combine"


def native_combine_extension_name() -> str:
    return "fa4_windows_native_combine_ext"


def _load_extension_module_from_file(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(native_combine_extension_name(), str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[native_combine_extension_name()] = module
    spec.loader.exec_module(module)
    return module


def _find_vsdevcmd() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ensure_msvc_env() -> None:
    global _MSVC_ENV_READY
    if os.name != "nt" or _MSVC_ENV_READY:
        return
    current_path = os.environ.get("PATH", "")
    try:
        subprocess.run(
            ["where", "cl"],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        _MSVC_ENV_READY = True
        return
    except Exception:
        pass

    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        raise RuntimeError("Unable to locate VsDevCmd.bat for the MSVC build environment")

    with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="utf-8") as batch_file:
        batch_file.write("@echo off\n")
        batch_file.write(f'call "{vsdevcmd}" -arch=amd64 -host_arch=amd64\n')
        batch_file.write("set\n")
        batch_path = Path(batch_file.name)
    try:
        proc = subprocess.run(
            ["cmd.exe", "/d", "/c", str(batch_path)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "VSCMD_SKIP_SENDTELEMETRY": "1"},
        )
    finally:
        batch_path.unlink(missing_ok=True)
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            os.environ[key] = value
    if "PATH" not in os.environ:
        os.environ["PATH"] = current_path
    if "INCLUDE" not in os.environ or "LIB" not in os.environ:
        raise RuntimeError(
            "VsDevCmd.bat did not populate INCLUDE/LIB for the MSVC build environment"
        )
    _MSVC_ENV_READY = True


def load_native_combine_backend(*, verbose: bool = False, force_rebuild: bool = False) -> ModuleType | None:
    global _NATIVE_COMBINE_MODULE, _NATIVE_COMBINE_ERROR
    if os.environ.get("FA4_WINDOWS_NATIVE_COMBINE_DISABLE", "").strip() == "1":
        _NATIVE_COMBINE_ERROR = "disabled via FA4_WINDOWS_NATIVE_COMBINE_DISABLE=1"
        return None
    if _NATIVE_COMBINE_MODULE is not None and not force_rebuild:
        return _NATIVE_COMBINE_MODULE

    with _NATIVE_COMBINE_LOCK:
        if _NATIVE_COMBINE_MODULE is not None and not force_rebuild:
            return _NATIVE_COMBINE_MODULE
        try:
            build_dir = native_combine_build_dir()
            build_dir.mkdir(parents=True, exist_ok=True)
            scripts_dir = str(Path(sys.executable).resolve().parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if scripts_dir not in path_entries:
                os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
            _ensure_msvc_env()
            module_candidates = list(build_dir.glob(f"{native_combine_extension_name()}*.pyd"))
            if force_rebuild:
                for candidate in module_candidates:
                    candidate.unlink(missing_ok=True)
                module_candidates = []
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime) if module_candidates else None
            source_mtime = max(
                native_combine_source_path().stat().st_mtime,
                native_combine_setup_path().stat().st_mtime,
            )
            needs_rebuild = newest_module is None or newest_module.stat().st_mtime < source_mtime
            if needs_rebuild:
                short_tmp = os.environ.get("FA4_WINDOWS_SHORT_BUILD_TMP", "").strip()
                if short_tmp:
                    build_temp_dir = Path(short_tmp) / native_combine_extension_name()
                else:
                    build_temp_dir = build_dir / "temp"
                build_temp_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(native_combine_setup_path()),
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
                    cwd=str(native_combine_setup_path().parent),
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
                        "native combine build failed\n"
                        f"stdout:\n{stdout[-4000:]}\n"
                        f"stderr:\n{stderr[-4000:]}"
                    )
                module_candidates = list(build_dir.glob(f"{native_combine_extension_name()}*.pyd"))
            if not module_candidates:
                raise RuntimeError(f"native combine build produced no {native_combine_extension_name()}*.pyd")
            newest_module = max(module_candidates, key=lambda path: path.stat().st_mtime)
            _NATIVE_COMBINE_MODULE = _load_extension_module_from_file(newest_module)
            _NATIVE_COMBINE_ERROR = None
        except Exception as exc:  # pragma: no cover - runtime build failures are environment-specific
            _NATIVE_COMBINE_MODULE = None
            _NATIVE_COMBINE_ERROR = f"{type(exc).__name__}: {exc}"
        return _NATIVE_COMBINE_MODULE


def native_combine_backend_status() -> dict[str, object]:
    module = _NATIVE_COMBINE_MODULE
    module_candidates = sorted(
        str(path) for path in native_combine_build_dir().glob(f"{native_combine_extension_name()}*.pyd")
    )
    return {
        "name": native_combine_extension_name(),
        "source": str(native_combine_source_path()),
        "setup": str(native_combine_setup_path()),
        "build_dir": str(native_combine_build_dir()),
        "built": bool(module_candidates),
        "built_candidates": module_candidates,
        "loaded": module is not None,
        "module_file": getattr(module, "__file__", None) if module is not None else None,
        "error": _NATIVE_COMBINE_ERROR,
    }


def _resolve_real_batch_index(
    *,
    batch_size: int,
    reference: torch.Tensor,
    varlen_batch_idx: Optional[torch.Tensor],
    name: str,
) -> torch.Tensor:
    if varlen_batch_idx is None:
        return torch.arange(batch_size, device=reference.device, dtype=torch.long)
    if varlen_batch_idx.ndim != 1 or varlen_batch_idx.numel() != batch_size:
        raise ValueError(f"{name} must be 1D with one entry per batch item")
    return varlen_batch_idx.to(device=reference.device, dtype=torch.long)


def build_combine_split_valid_mask(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    num_splits_dynamic_ptr: Optional[torch.Tensor],
    varlen_batch_idx: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    needs_mask = any(t is not None for t in (cu_seqlens, seqused, num_splits_dynamic_ptr, varlen_batch_idx))
    if not needs_mask:
        return None

    num_splits = int(out_partial.shape[0])
    device = out_partial.device
    split_ids = torch.arange(num_splits, device=device, dtype=torch.long)

    if out_partial.ndim == 5:
        _, batch_size, seqlen_q, num_heads = lse_partial.shape
        real_batch_idx = _resolve_real_batch_index(
            batch_size=batch_size,
            reference=out_partial,
            varlen_batch_idx=varlen_batch_idx,
            name="varlen_batch_idx",
        )
        seq_valid = torch.ones((batch_size, seqlen_q), device=device, dtype=torch.bool)
        if seqused is not None:
            used = seqused.to(device=device, dtype=torch.long)[real_batch_idx]
            q_ids = torch.arange(seqlen_q, device=device, dtype=torch.long).view(1, seqlen_q)
            seq_valid = q_ids < used.view(batch_size, 1)
        split_valid = torch.ones((num_splits, batch_size), device=device, dtype=torch.bool)
        if num_splits_dynamic_ptr is not None:
            split_counts = num_splits_dynamic_ptr.to(device=device, dtype=torch.long)[real_batch_idx].clamp(0, num_splits)
            split_valid = split_ids.view(num_splits, 1) < split_counts.view(1, batch_size)
        mask = split_valid.view(num_splits, batch_size, 1, 1) & seq_valid.view(1, batch_size, seqlen_q, 1)
        return mask.expand(num_splits, batch_size, seqlen_q, num_heads)

    if out_partial.ndim != 4:
        raise ValueError("out_partial must have 4 or 5 dimensions")

    _, total_q, num_heads = lse_partial.shape
    token_batch_idx = torch.zeros((total_q,), device=device, dtype=torch.long)
    token_valid = torch.ones((total_q,), device=device, dtype=torch.bool)
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
            raise ValueError("cu_seqlens must be 1D with at least one element")
        batch_size = int(cu_seqlens.numel() - 1)
        real_batch_idx = _resolve_real_batch_index(
            batch_size=batch_size,
            reference=out_partial,
            varlen_batch_idx=varlen_batch_idx,
            name="varlen_batch_idx",
        )
        token_batch_idx = torch.zeros((total_q,), device=device, dtype=torch.long)
        token_valid = torch.zeros((total_q,), device=device, dtype=torch.bool)
        for batch_idx in range(batch_size):
            start = int(cu_seqlens[batch_idx].item())
            end = int(cu_seqlens[batch_idx + 1].item())
            used = end - start
            real_idx = int(real_batch_idx[batch_idx].item())
            if seqused is not None:
                used = min(used, max(0, int(seqused[real_idx].item())))
            valid_end = min(start + used, end, total_q)
            if valid_end > start:
                token_batch_idx[start:valid_end] = real_idx
                token_valid[start:valid_end] = True
    elif seqused is not None and seqused.numel() == 1:
        used = min(total_q, max(0, int(seqused[0].item())))
        token_valid = torch.arange(total_q, device=device, dtype=torch.long) < used

    split_valid = token_valid.view(1, total_q, 1).expand(num_splits, total_q, num_heads)
    if num_splits_dynamic_ptr is not None:
        split_counts = torch.zeros((total_q,), device=device, dtype=torch.long)
        if token_valid.any():
            split_counts[token_valid] = num_splits_dynamic_ptr.to(device=device, dtype=torch.long)[
                token_batch_idx[token_valid]
            ].clamp(0, num_splits)
        split_valid = split_valid & (split_ids.view(num_splits, 1, 1) < split_counts.view(1, total_q, 1))
    return split_valid


def _combine_split_attention_local(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    *,
    split_valid_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out_partial.ndim not in (4, 5):
        raise ValueError("out_partial must have 4 or 5 dimensions")
    if lse_partial.ndim != out_partial.ndim - 1:
        raise ValueError("lse_partial rank must be exactly one less than out_partial rank")
    if out_partial.shape[:-1] != lse_partial.shape:
        raise ValueError("out_partial and lse_partial shapes are incompatible")
    if out_partial.shape[0] == 0:
        raise ValueError("out_partial must include at least one split")

    lse_float = lse_partial.float()
    valid = ~torch.isneginf(lse_float)
    if split_valid_mask is not None:
        if split_valid_mask.shape != lse_partial.shape:
            raise ValueError("split_valid_mask must match lse_partial shape")
        valid = valid & split_valid_mask
    any_valid = valid.any(dim=0)
    lse_max = torch.amax(lse_float, dim=0)
    safe_lse_max = torch.where(any_valid, lse_max, torch.zeros_like(lse_max))

    weights = torch.where(
        valid,
        torch.exp(lse_float - safe_lse_max.unsqueeze(0)),
        torch.zeros_like(lse_float),
    )
    denom = weights.sum(dim=0)
    numerator = (out_partial.float() * weights.unsqueeze(-1)).sum(dim=0)
    out = torch.where(
        denom.unsqueeze(-1) > 0,
        numerator / denom.unsqueeze(-1),
        torch.zeros_like(numerator),
    )
    lse = torch.where(
        denom > 0,
        torch.log(denom) + safe_lse_max,
        torch.full_like(safe_lse_max, float("-inf")),
    )
    return out, lse


def flash_attn_combine_native(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    verbose_build: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    backend = load_native_combine_backend(verbose=verbose_build)
    split_valid_mask = build_combine_split_valid_mask(
        out_partial,
        lse_partial,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        num_splits_dynamic_ptr=num_splits_dynamic_ptr,
        varlen_batch_idx=varlen_batch_idx,
    )
    if backend is None:
        combined_out, combined_lse = _combine_split_attention_local(
            out_partial,
            lse_partial,
            split_valid_mask=split_valid_mask,
        )
    else:
        combined_out, combined_lse = backend.flash_attn_combine_forward(
            out_partial,
            lse_partial,
            split_valid_mask,
        )
    if out is None:
        target_dtype = out_dtype if out_dtype is not None else out_partial.dtype
        out = combined_out.to(dtype=target_dtype)
    else:
        if out.shape != combined_out.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} does not match combined output shape {tuple(combined_out.shape)}"
            )
        out.copy_(combined_out.to(dtype=out.dtype))
    if not return_lse:
        return out, None
    return out, combined_lse
