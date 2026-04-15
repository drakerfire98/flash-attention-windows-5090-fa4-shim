"""Bridge selected FA4 CuTe compile calls onto the validated Windows shim path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch

from cutlass.cutlass_dsl import JitCompiledFunction
from _probe_helpers import ProbePlaceholder


_SHIM_MODULE: ModuleType | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_windows_shim_module() -> ModuleType:
    global _SHIM_MODULE
    if _SHIM_MODULE is not None:
        return _SHIM_MODULE

    shim_init = _repo_root() / "shims" / "flash_attn" / "cute" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "native_probe_windows_flash_attn_cute_shim",
        shim_init,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Windows shim module from {shim_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SHIM_MODULE = module
    return module


def _copy_tensor_like(dst: torch.Tensor | None, src: torch.Tensor | None) -> None:
    if dst is None or src is None:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    if dst.ndim == 3 and src.ndim == 3:
        dst.copy_(src.permute(0, 2, 1).contiguous())
        return
    if dst.ndim == 2 and src.ndim == 2:
        dst.copy_(src.transpose(-1, -2).contiguous())
        return
    raise ValueError(f"Cannot copy shim tensor with shape {src.shape} into target shape {dst.shape}")


def _window_tuple(window_left: int | None, window_right: int | None) -> tuple[int | None, int | None]:
    return (window_left, window_right)


class NativeProbeForwardBridge(JitCompiledFunction):
    def __init__(self, kernel: Any):
        self.kernel = kernel
        self.kernel_name = type(kernel).__name__

    def __repr__(self) -> str:
        return f"<NativeProbeForwardBridge {self.kernel_name}>"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor | None,
        softmax_scale: float | None,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
        seqused_q: torch.Tensor | None,
        seqused_k: torch.Tensor | None,
        page_table: torch.Tensor | None,
        window_size_left: int | None,
        window_size_right: int | None,
        learnable_sink: torch.Tensor | None,
        block_sparse_tensors: Any,
        aux_tensors: list[torch.Tensor] | None,
    ) -> None:
        if block_sparse_tensors is not None:
            raise NotImplementedError("Block-sparse native probe bridge is not implemented yet")

        shim = _load_windows_shim_module()
        common = dict(
            softmax_scale=softmax_scale,
            causal=bool(getattr(self.kernel, "is_causal", False)),
            window_size=_window_tuple(window_size_left, window_size_right),
            learnable_sink=learnable_sink,
            return_lse=lse is not None,
        )

        is_varlen = any(t is not None for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table))
        score_mod = getattr(self.kernel, "score_mod", None)
        mask_mod = getattr(self.kernel, "mask_mod", None)

        if is_varlen:
            shim_out, shim_lse = shim.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                page_table=page_table,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
                **common,
            )
        elif score_mod is not None:
            shim_out, shim_lse = shim._attention_forward_dense(
                q,
                k,
                v,
                score_mod=score_mod,
                mask_mod=mask_mod,
                aux_tensors=aux_tensors,
                batch_start_index=0,
                offset_q=0,
                offset_k=0,
                softcap=0.0,
                **common,
            )
        else:
            shim_out, shim_lse = shim.flash_attn_func(
                q,
                k,
                v,
                mask_mod=mask_mod,
                **common,
            )

        _copy_tensor_like(out, shim_out)
        _copy_tensor_like(lse, shim_lse)


class NativeProbeUnsupportedBridge(JitCompiledFunction):
    def __init__(self, op: Any):
        self.op = op
        self.op_name = type(op).__name__

    def __repr__(self) -> str:
        return f"<NativeProbeUnsupportedBridge {self.op_name}>"

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            f"Native probe compile bridge does not yet implement {self.op_name}"
        )


def compile_dispatch(op: Any, *args, **kwargs):
    del args, kwargs
    op_name = type(op).__name__
    if op_name.startswith("FlashAttentionForwardSm"):
        return NativeProbeForwardBridge(op)
    return NativeProbeUnsupportedBridge(op)


def compile_repr() -> ProbePlaceholder:
    return ProbePlaceholder("cutlass.cute.compile")
